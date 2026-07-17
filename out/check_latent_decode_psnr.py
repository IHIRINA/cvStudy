import os
import sys
import json
import torch
import numpy as np
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.inferers import SlidingWindowInferer

sys.path.insert(0, '/root/mxr/code')
from model import Config, JiT3D_CBCT2CT


def load_nifti_float32(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data


def compute_psnr(gt, pred, data_range=None):
    if data_range is None:
        data_range = float(np.nanmax(gt) - np.nanmin(gt))
        if data_range <= 0:
            data_range = 1.0
    return psnr(gt, pred, data_range=data_range)


def normalize_minmax(x):
    x = x.astype(np.float32)
    lo = np.min(x)
    hi = np.max(x)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    json_path = '/root/autodl-tmp/data/latent_t1ce_tof_pairs.json'
    data_root = '/root/autodl-tmp/data'
    checkpoint_path = '/root/autodl-tmp/results/checkpoints_56*56*40_4ch/best_model_weights.pth'
    ae_ckpt_path = '/root/mxr/code/encoder_decoder/autoencoder_epoch273.pt'
    tof_gt_dir = '/root/autodl-tmp/data/TOF'

    with open(json_path, 'r') as f:
        training_data = json.load(f)['training']

    train_samples = [x for x in training_data if x['fold'] != 1]
    if not train_samples:
        raise RuntimeError('No training samples found in json.')

    samples = train_samples[:3]
    print(f'Loaded {len(samples)} samples for latent/decode PSNR evaluation.')

    config = Config()
    config.save_dir = '/root/autodl-tmp/results/checkpoints_56*56*40_4ch'
    model = JiT3D_CBCT2CT(config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    autoencoder = AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        num_channels=[64, 128, 256],
        num_res_blocks=[2, 2, 2],
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=True,
        num_splits=8,
        dim_split=1,
    ).to(device)
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device, weights_only=True)
    autoencoder.load_state_dict(ae_ckpt)
    autoencoder.eval()

    class DecodeWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, z):
            return self.model.decode(z)

    decode_wrapper = DecodeWrapper(autoencoder)
    decode_inferer = SlidingWindowInferer(
        roi_size=[56, 56, 40],
        sw_batch_size=1,
        overlap=0.5,
        progress=False,
        sw_device=device,
        mode='gaussian',
        device=torch.device('cpu'),
    )

    for idx, sample in enumerate(samples, start=1):
        print(f'\n=== Sample {idx}: {os.path.basename(sample["image"])} ===')
        cbct_path = os.path.join(data_root, sample['image'])
        gt_latent_path = os.path.join(data_root, sample['label'])

        cbct = load_nifti_float32(cbct_path)
        gt_latent = load_nifti_float32(gt_latent_path)

        if cbct.ndim == 4 and cbct.shape[-1] == 4:
            cbct = np.moveaxis(cbct, -1, 0)
        elif cbct.ndim == 3:
            cbct = cbct[np.newaxis, ...]
        else:
            raise ValueError(f'Unexpected cbct shape: {cbct.shape}')

        if gt_latent.ndim == 4 and gt_latent.shape[-1] == 4:
            gt_latent = np.moveaxis(gt_latent, -1, 0)
        elif gt_latent.ndim == 3:
            gt_latent = gt_latent[np.newaxis, ...]
        else:
            raise ValueError(f'Unexpected gt_latent shape: {gt_latent.shape}')

        cbct_tensor = torch.from_numpy(cbct).unsqueeze(0).to(device)

        t_seq = torch.linspace(0.95, 0.05, 50, device=device)
        dt = 1.0 / (50 - 1)
        ct_current = torch.randn_like(cbct_tensor)

        with torch.no_grad():
            for step in range(50):
                t = t_seq[step].repeat(1).view(1, 1, 1, 1, 1)
                ct_pred = model(cbct_tensor, ct_current, t)
                if step < 49:
                    v1 = (ct_pred - ct_current) / (1 - t + 1e-5)
                    ct_noisy_euler = ct_current + dt * v1
                    t_next = t_seq[step + 1].repeat(1).view(1, 1, 1, 1, 1)
                    ct_pred_next = model(cbct_tensor, ct_noisy_euler, t_next)
                    v2 = (ct_pred_next - ct_noisy_euler) / (1 - t_next + 1e-5)
                    ct_current = ct_current + dt * (v1 + v2) / 2.0
                else:
                    ct_current = ct_pred

        pred_latent = ct_current[0].cpu().numpy()
        gt_latent = gt_latent.astype(np.float32)

        if pred_latent.shape != gt_latent.shape:
            print(f'  跳过 latent PSNR: 形状不匹配 {pred_latent.shape} vs {gt_latent.shape}')
        else:
            latent_psnr = compute_psnr(gt_latent, pred_latent)
            print(f'  Latent PSNR: {latent_psnr:.2f} dB')

        sample_name = os.path.basename(sample['label'])
        raw_gt_path = os.path.join(tof_gt_dir, sample_name)
        if not os.path.exists(raw_gt_path):
            print(f'  未找到原始 GT 图像: {raw_gt_path}, 跳过 decode PSNR')
            continue

        with torch.no_grad(), torch.amp.autocast('cuda'):
            recon = decode_inferer(network=decode_wrapper, inputs=ct_current)
        recon_img = recon[0, 0].cpu().numpy().astype(np.float32)

        gt_img = load_nifti_float32(raw_gt_path)
        if recon_img.shape != gt_img.shape:
            print(f'  decode 形状不匹配: {recon_img.shape} vs {gt_img.shape}, 跳过 decode PSNR')
            continue

        recon_norm = normalize_minmax(recon_img)
        gt_norm = normalize_minmax(gt_img)
        decode_psnr = compute_psnr(gt_norm, recon_norm, data_range=1.0)

        mid = gt_norm.shape[2] // 2
        ssim_val = ssim(gt_norm[:, :, mid], recon_norm[:, :, mid], data_range=1.0)

        print(f'  Decode PSNR (normalized): {decode_psnr:.2f} dB')
        print(f'  Decode SSIM (middle slice): {ssim_val:.4f}')


if __name__ == '__main__':
    main()
