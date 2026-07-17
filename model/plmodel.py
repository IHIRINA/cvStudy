import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pytorch_lightning as pl
from torchvision.utils import make_grid
from model.BrownianBridge.BrownianBridge import BrownianBridgeModel
from pytorch_lightning.utilities import rank_zero_only


class MyPlModel(pl.LightningModule):
    def __init__(self, options, l_train, l_val):
        super().__init__()
        self.options = options
        self.use_modality = options.data.use_modality
        self.BBDM = BrownianBridgeModel(options)
        
        self.l_train = l_train
        self.l_val = l_val
        self.cuda_num = len(options.train.cuda_num)


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.BBDM.parameters(), lr=self.options.train.lr, betas=(0.5, 0.999))
        return {'optimizer': opt}
    

    def training_step(self, batch, batch_idx):
        loss, loss0, loss1, loss2, loss3 = self.BBDM.forward_muti(batch)
        self.log_dict({'loss': loss.detach(), 
                       'loss0': loss0, 'loss1': loss1, 'loss2': loss2, 'loss3': loss3},
                       on_step=False, on_epoch=True, sync_dist=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        # simple validation
        if batch_idx == 0:
            with torch.no_grad():
                context, x_t_1, x_t_2, x_t_3 = self.BBDM.sample(batch) # B C W H
                self.save_imgs(context, batch['3D-TOF-MRA'], self.current_epoch)
                self.save_imgs(x_t_1, batch['3D-TOF-MRA'], f'{self.current_epoch}_1')
                self.save_imgs(x_t_2, batch['3D-TOF-MRA'], f'{self.current_epoch}_2')
                self.save_imgs(x_t_3, batch['3D-TOF-MRA'], f'{self.current_epoch}_3')
    

    def save_imgs(self, img_fake, img_target, num):
        '''
        real = img_target[:, 0:1].data 
        fake = img_fake[:, 0:1].data 
        # real, fake = torch.log(real), torch.log(fake)
        real = (real - real.min()) / (real.max() - real.min())
        fake = (fake - fake.min()) / (fake.max() - fake.min())
        img = torch.cat((real, fake), -2)
        grid = make_grid((img*255).clip(0, 255))
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy().astype(np.uint8)
        im = Image.fromarray(ndarr)
        im.convert('L').save(f"/root/autodl-tmp/logs/images/{num}.png")
        '''
        B, C, H, W = img_target.shape

        # xy面的截图，取第零层
        real_axial = img_target[:, 0:1, :, :].data          # (B, 1, H, W)
        fake_axial = img_fake[:, 0:1, :, :].data
        real_axial = (real_axial - real_axial.min()) / (real_axial.max() - real_axial.min() + 1e-8)
        fake_axial = (fake_axial - fake_axial.min()) / (fake_axial.max() - fake_axial.min() + 1e-8)
        axial_comp = torch.cat((real_axial, fake_axial), dim=-2)   # (B, 1, 2*H, W)  real在上 fake在下
        # xz面的截图， 取中间层
        mid_y = H // 2
        real_cor = img_target[:, :, mid_y:mid_y+1, :].squeeze(2).unsqueeze(1).data   # (B, 1, C, W)
        fake_cor = img_fake[:, :, mid_y:mid_y+1, :].squeeze(2).unsqueeze(1).data
        real_cor = (real_cor - real_cor.min()) / (real_cor.max() - real_cor.min() + 1e-8)
        fake_cor = (fake_cor - fake_cor.min()) / (fake_cor.max() - fake_cor.min() + 1e-8)
        cor_comp = torch.cat((real_cor, fake_cor), dim=-2)         # (B, 1, 2*C, W)   Z轴作为高度
        # yz面的截图， 取中间层
        mid_x = W // 2
        real_sag = img_target[:, :, :, mid_x:mid_x+1].squeeze(3).unsqueeze(1).data   # (B, 1, C, H)
        fake_sag = img_fake[:, :, :, mid_x:mid_x+1].squeeze(3).unsqueeze(1).data
        real_sag = (real_sag - real_sag.min()) / (real_sag.max() - real_sag.min() + 1e-8)
        fake_sag = (fake_sag - fake_sag.min()) / (fake_sag.max() - fake_sag.min() + 1e-8)
        sag_comp = torch.cat((real_sag, fake_sag), dim=-2)         # (B, 1, 2*C, H)

        comps = [axial_comp, cor_comp, sag_comp]
        max_h = max(c.shape[2] for c in comps)
        max_w = max(c.shape[3] for c in comps)
        padded = []
        for comp in comps:
            # 用 0（黑色）把每组补齐到相同尺寸
            pad_h = max_h - comp.shape[2]
            pad_w = max_w - comp.shape[3]
            p = torch.zeros((B, 1, max_h, max_w), device=comp.device, dtype=comp.dtype)
            p[:, :, :comp.shape[2], :comp.shape[3]] = comp
            padded.append(p)

        # 横向拼接 → 一张大图（三组并排）
        big_img = torch.cat(padded, dim=-1)                    # (B, 1, max_h, 3*max_w)

        # 转成 0-255 并保存
        big_img = (big_img * 255).clip(0, 255).to(torch.uint8)
        grid = make_grid(big_img)
        ndarr = grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        im = Image.fromarray(ndarr)
        im.convert('L').save(f"/root/autodl-tmp/logs/images/{num}.png")



class SaveCheck(pl.Callback):
    def __init__(self, options):
        super().__init__()
        self.save_freq = options.train.save_freq
        self.start = options.train.start
        if not os.path.exists(f'/root/autodl-tmp/logs/checkpoints'):
            os.makedirs(f'/root/autodl-tmp/logs/checkpoints')
        if not os.path.exists(f'/root/autodl-tmp/logs/images/'):
            os.makedirs(f'/root/autodl-tmp/logs/images/')

    def on_train_epoch_start(self, trainer, pl_module):
        print(f'Epoch: {pl_module.current_epoch + 1}')
        pl_module.BBDM.train()
        self.pbar = tqdm(total=pl_module.l_train//pl_module.cuda_num, ncols=100)
    
    def on_train_batch_end(self, *args):
        self.pbar.update(1)
    
    @rank_zero_only
    def on_validation_epoch_start(self, trainer, pl_module):
        self.pbar.close()
        # pl_module.save_acc.show('train')
        print('val:')
        pl_module.BBDM.eval()

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        '''
        if self.start != 0 and self.start == epoch+1:
            pl_module.save_acc.del_more_val()
        pl_module.save_acc.show('val')
        '''
        if (epoch+1) % self.save_freq == 0 and epoch != 0:
            trainer.save_checkpoint(f"/root/autodl-tmp/logs/checkpoints/epoch_{epoch+1}.ckpt")
            print(f'Save {epoch+1} last Trainer!')

        print('\n')
        # pl_module.save_acc.save_results()