from model.BrownianBridge.base.modules.diffusionmodules.util import linear, timestep_embedding
import torch
import torch.nn as nn
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np
from model.BrownianBridge.Restormer import ResUnet, Restormer
from model.utils import extract, default
from model.BrownianBridge.openaimodel import UNetModel
from utils.loss import AutomaticWeightedLoss, MIPloss


class BrownianBridgeModel(nn.Module):
    def __init__(self, options):
        super().__init__()
        # model hyperparameters
        model_params = options.BB
        self.num_timesteps = model_params.timesteps
        self.mt_type = model_params.mt_type
        
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.register_schedule()

        # objective and loss
        self.objective = model_params.objective
        self.mip_loss = MIPloss(options)
        self.weight_loss = AutomaticWeightedLoss()

        # UNet
        self.denoise_fn_1 = ResUnet(**vars(options.Restormer))
        self.denoise_fn_2 = ResUnet(**vars(options.Restormer))
        self.denoise_fn_3 = ResUnet(**vars(options.Restormer))

        self.cross1 = Restormer(15, 5, 5)
        self.cross2 = Restormer(15, 5, 5)
        self.cross3 = Restormer(15, 5, 5)

        self.x0_fusion = Restormer(15, 5, 5)

        self.fusion = UNetModel(**vars(options.FusionUnet))

        self.time_embed = nn.Sequential(
            linear(32, 32*4),
            nn.SiLU(),
            linear(32*4, 32*4),
        )
        

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2)
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    
    def forward_muti(self, batch):
        x0 = batch['3D-TOF-MRA']
        noise = torch.randn_like(x0)
        b, _, _, _, device = *x0.shape, x0.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        t_emb = self.time_embed(timestep_embedding(t, 32, repeat_only=False))

        x_t_1, _ = self.q_sample(x0, batch['3D-T1W'], t, noise)
        x_t_2, _ = self.q_sample(x0, batch['3D-T2W'], t, noise)
        x_t_3, _ = self.q_sample(x0, batch['3D-FLAIR'], t, noise)

        objective_recon_1 = self.denoise_fn_1(x_t_1, time_emb=t_emb, context=batch['3D-T1W'])
        objective_recon_2 = self.denoise_fn_2(x_t_2, time_emb=t_emb, context=batch['3D-T2W'])
        objective_recon_3 = self.denoise_fn_3(x_t_3, time_emb=t_emb, context=batch['3D-FLAIR'])
        
        x0_1 = self.predict_x0_from_objective(x_t_1, batch['3D-T1W'], t, objective_recon=objective_recon_1)
        x0_2 = self.predict_x0_from_objective(x_t_2, batch['3D-T2W'], t, objective_recon=objective_recon_2)
        x0_3 = self.predict_x0_from_objective(x_t_3, batch['3D-FLAIR'], t, objective_recon=objective_recon_3)
        
        feat_1 = self.cross1(torch.cat([x0_1, batch['3D-T2W'], batch['3D-FLAIR']], 1))
        feat_2 = self.cross2(torch.cat([x0_2, batch['3D-T1W'], batch['3D-FLAIR']], 1))
        feat_3 = self.cross3(torch.cat([x0_3, batch['3D-T1W'], batch['3D-T2W']], 1))
        
        input = self.x0_fusion(torch.cat([feat_1, feat_2, feat_3], 1))
        # input = torch.cat([feat_1, feat_2, feat_3], 1)
        x0_recon = self.fusion(input, t_emb)

        loss1 = self.mip_loss(x0_1, batch)
        loss2 = self.mip_loss(x0_2, batch)
        loss3 = self.mip_loss(x0_3, batch)
        loss0 = self.mip_loss(x0_recon, batch)
        
        sigma_t = self.variance_t[t[0]]
        loss = self.weight_loss([loss0, loss1, loss2, loss3], sigma_t)
        # (objective - objective_recon).abs().mean()
        return loss, loss0.detach(), loss1.detach(), loss2.detach(), loss3.detach()        

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def sample(self, batch, context=None, clip_denoised=True, sample_mid_step=False):
        x_t_1 = batch['3D-T1W']
        x_t_2 = batch['3D-T2W']
        x_t_3 = batch['3D-FLAIR']
        for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
            t = torch.full((x_t_1.shape[0],), self.steps[i], device=x_t_1.device, dtype=torch.long)
            t_emb = self.time_embed(timestep_embedding(t, 32, repeat_only=False))

            noise = torch.rand_like(x_t_1)

            x0_1 = self.p_sample_new(x_t=x_t_1, y=batch['3D-T1W'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_1)
            x0_2 = self.p_sample_new(x_t=x_t_2, y=batch['3D-T2W'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_2)
            x0_3 = self.p_sample_new(x_t=x_t_3, y=batch['3D-FLAIR'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_3)
            
            feat_1 = self.cross1(torch.cat([x0_1, batch['3D-T2W'], batch['3D-FLAIR']], 1))
            feat_2 = self.cross2(torch.cat([x0_2, batch['3D-T1W'], batch['3D-FLAIR']], 1))
            feat_3 = self.cross3(torch.cat([x0_3, batch['3D-T1W'], batch['3D-T2W']], 1))
        
            input = self.x0_fusion(torch.cat([feat_1, feat_2, feat_3], 1))

            x0_recon = self.fusion(input, t_emb)
            x_t_1 = self.get_xtminus(x_t_1, noise, t, batch['3D-T1W'], x0_recon, i)
            x_t_2 = self.get_xtminus(x_t_2, noise, t, batch['3D-T2W'], x0_recon, i)
            x_t_3 = self.get_xtminus(x_t_3, noise, t, batch['3D-FLAIR'], x0_recon, i)

        x0_1 = self.p_sample_new(x_t=x_t_1, y=batch['3D-T1W'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_1)
        x0_2 = self.p_sample_new(x_t=x_t_2, y=batch['3D-T2W'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_2)
        x0_3 = self.p_sample_new(x_t=x_t_3, y=batch['3D-FLAIR'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_3)
            
        feat_1 = self.cross1(torch.cat([x0_1, batch['3D-T2W'], batch['3D-FLAIR']], 1))
        feat_2 = self.cross2(torch.cat([x0_2, batch['3D-T1W'], batch['3D-FLAIR']], 1))
        feat_3 = self.cross3(torch.cat([x0_3, batch['3D-T1W'], batch['3D-T2W']], 1))
        
        input = self.x0_fusion(torch.cat([feat_1, feat_2, feat_3], 1))

        x0_recon = self.fusion(input, t_emb)

        return x0_recon.clamp_(-1., 1.), x0_1.clamp_(-1., 1.), x0_2.clamp_(-1., 1.), x0_3.clamp_(-1., 1.)

    @torch.no_grad()
    def get_xtminus(self, x_t, noise, t, y, x0_recon, i):
        shape, device = x0_recon.shape, x0_recon.device
        b = shape[0]
        if i != self.sample_step-1:
            n_t = torch.full((b,), self.steps[i+1], device=device, dtype=torch.long)
            x0_recon.clamp_(-1., 1.)
            m_t = extract(self.m_t, t, shape)
            m_nt = extract(self.m_t, n_t, shape)
            var_t = extract(self.variance_t, t, shape)
            var_nt = extract(self.variance_t, n_t, shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t)

            # noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)
            return x_tminus_mean + sigma_t * noise
        else:
            return x_t

    @torch.no_grad()
    def p_sample_new(self, x_t, y, t_emb, t, clip_denoised=True, fn=None):
        objective_recon = fn(x_t, time_emb=t_emb, context=y)
        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
        if clip_denoised:
            x0_recon.clamp_(-1., 1.)
        return x0_recon
    

    @torch.no_grad()
    def sample_faster(self, batch, context=None, clip_denoised=True, sample_mid_step=False):
        x_t_1 = batch['3D-T1W']
        x_t_2 = batch['3D-T2W']
        x_t_3 = batch['3D-FLAIR']
        for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
            t = torch.full((x_t_1.shape[0],), self.steps[i], device=x_t_1.device, dtype=torch.long)
            t_emb = self.time_embed(timestep_embedding(t, 32, repeat_only=False))

            noise = torch.rand_like(x_t_1)

            x0_1 = self.p_sample_new(x_t=x_t_1, y=batch['3D-T1W'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_1)
            x0_2 = self.p_sample_new(x_t=x_t_2, y=batch['3D-T2W'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_2)
            x0_3 = self.p_sample_new(x_t=x_t_3, y=batch['3D-FLAIR'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_3)
            
            x_t_1 = self.get_xtminus(x_t_1, noise, t, batch['3D-T1W'], x0_1, i)
            x_t_2 = self.get_xtminus(x_t_2, noise, t, batch['3D-T2W'], x0_2, i)
            x_t_3 = self.get_xtminus(x_t_3, noise, t, batch['3D-FLAIR'], x0_3, i)

        x0_1 = self.p_sample_new(x_t=x_t_1, y=batch['3D-T1W'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_1)
        x0_2 = self.p_sample_new(x_t=x_t_2, y=batch['3D-T2W'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_2)
        x0_3 = self.p_sample_new(x_t=x_t_3, y=batch['3D-FLAIR'], t_emb=t_emb,
                                      t=t, clip_denoised=clip_denoised, fn=self.denoise_fn_3)
            
        feat_1 = self.cross1(torch.cat([x0_1, batch['3D-T2W'], batch['3D-FLAIR']], 1))
        feat_2 = self.cross2(torch.cat([x0_2, batch['3D-T1W'], batch['3D-FLAIR']], 1))
        feat_3 = self.cross3(torch.cat([x0_3, batch['3D-T1W'], batch['3D-T2W']], 1))
        
        input = self.x0_fusion(torch.cat([feat_1, feat_2, feat_3], 1))

        x0_recon = self.fusion(input, t_emb)

        return x0_recon.clamp_(-1., 1.), x0_1.clamp_(-1., 1.), x0_2.clamp_(-1., 1.), x0_3.clamp_(-1., 1.)


        