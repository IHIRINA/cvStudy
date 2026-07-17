import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pytorch_msssim import ms_ssim as ssim


def gumbel_softmax_sample(logits, temperature, gumbel, dim):
    '''mip loss'''
    y = logits + gumbel
    return F.softmax(y / temperature, dim)


def cal_snr(noise_img, clean_img):
    noise_img, clean_img = noise_img.detach().cpu().numpy(), clean_img.detach().cpu().numpy()
    noise_signal = noise_img - clean_img
    clean_signal = clean_img
    noise_signal_2 = noise_signal**2
    clean_signal_2 = clean_signal**2
    sum1 = np.sum(clean_signal_2)
    sum2 = np.sum(noise_signal_2)
    snrr = 20*math.log10(math.sqrt(sum1)/math.sqrt(sum2))
    return snrr


class MIPloss(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.temp = options.mip_loss.temp
        self.num_slice = options.data.use_slice
        # self.L1 = torch.nn.L1Loss()
        #self.L2 = torch.nn.MSELoss()   不需要这个了，手动实现mask的mse

    def reset_gumbel(self, img_fake):
        U = torch.rand_like(img_fake)
        # U = torch.rand(img_fake.size()).cuda()
        self.gumbel = -torch.log(-torch.log(U + 1e-20) + 1e-20)  # sample_gumbel

    def forward(self, img_fake, batch):
        self.reset_gumbel(img_fake)
        target = batch['3D-TOF-MRA']

        mask = batch['mask']

        pred_mips_c1 = torch.zeros_like(img_fake)
        target_mips_c1 = torch.zeros_like(target)
        for idx in range(img_fake.shape[1]):
            pred_mip = gumbel_softmax_sample(img_fake[:, :idx+1], self.temp, self.gumbel[:, :idx+1], dim=1)
            target_mips_c1[:, idx] = torch.max(target[:, :idx+1], dim=1)[0]
            pred_mips_c1[:, idx] = torch.sum(pred_mip*img_fake[:, :idx+1], dim=1)

        pred_mips_c2 = torch.zeros_like(img_fake)
        target_mips_c2 = torch.zeros_like(target)
        for idx in range(img_fake.shape[1]):
            pred_mip = gumbel_softmax_sample(img_fake[:, self.num_slice-idx-1:], self.temp, self.gumbel[:, self.num_slice-idx-1:], dim=1)
            target_mips_c2[:, idx] = torch.max(target[:, self.num_slice-idx-1:], dim=1)[0]
            pred_mips_c2[:, idx] = torch.sum(pred_mip*img_fake[:, self.num_slice-idx-1:], dim=1)
    
        '''
        loss_ = self.L2(img_fake, target)
        loss_mip_c1 = self.L2(pred_mips_c1, target_mips_c1)
        loss_mip_c2 = self.L2(pred_mips_c2, target_mips_c2)
        loss = loss_ + loss_mip_c1 + loss_mip_c2
        '''

        loss_ = ((img_fake - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
        # MIP c1 loss（MIP 投影后的前景 mask）
        mask_mip_c1 = (target_mips_c1 > -0.999).float()
        loss_mip_c1 = ((pred_mips_c1 - target_mips_c1) ** 2 * mask_mip_c1).sum() / (mask_mip_c1.sum() + 1e-8)

        # MIP c2 loss
        mask_mip_c2 = (target_mips_c2 > -0.999).float()
        loss_mip_c2 = ((pred_mips_c2 - target_mips_c2) ** 2 * mask_mip_c2).sum() / (mask_mip_c2.sum() + 1e-8)

        loss = loss_ + loss_mip_c1 + loss_mip_c2
    
        return loss
        

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=4):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, losses, sigma_t):
        loss_sum = 0
       
        for i, loss in enumerate(losses):
            if i != 0:
                adjust_para = self.params[i] ** 2 + sigma_t
            else:
                adjust_para = self.params[i] ** 2
            loss_sum += 0.5 / adjust_para * loss + torch.log(1 + adjust_para)

        return loss_sum


