"""
============================================================
阶段 1：检查 Encoder + Decoder 本身是否正常（已根据 decoder 代码修正）
============================================================
测试链路：TOF 原图 [0,1] → autoencoder.forward() → 重建 [0,1]
对比：原图 vs 重建（脑区 PSNR）

如果你的 decoder 脚本（maisi_decoder.py）处理的是 DiT 预测的 latent，
那么这些 latent 是 [0,1] 空间，decoder 输出也是 [0,1]。
但原始 TOF 是 HU 值（~0-1000），直接比 PSNR 当然只有 11。

→ 本脚本在 [0,1] 空间内计算 PSNR，排除值域不匹配的影响。
============================================================
"""
import os
import torch
import numpy as np
import nibabel as nib
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

device = torch.device("cuda")

AE_CKPT = "/root/mxr/code/encoder_decoder/autoencoder_epoch273.pt"
TOF_DIR = "/root/autodl-tmp/data_reg/TOF"
NUM_SAMPLES = 3

# ---------- 加载 AE ----------
autoencoder = AutoencoderKlMaisi(
    spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4,
    num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2],
    norm_num_groups=32, norm_eps=1e-06,
    attention_levels=[False, False, False],
    with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
    use_checkpointing=False, use_convtranspose=False,
    norm_float16=True, num_splits=8, dim_split=1
).to(device)
ckpt = torch.load(AE_CKPT, map_location=device, weights_only=True)
autoencoder.load_state_dict(ckpt)
autoencoder.eval()
print("✅ AE 加载完成\n")

tof_files = sorted([f for f in os.listdir(TOF_DIR) if f.endswith(('.nii.gz', '.nii'))])[:NUM_SAMPLES]
psnr_list = []

for idx, tof_file in enumerate(tof_files):
    tof_path = os.path.join(TOF_DIR, tof_file)
    gt_np = nib.load(tof_path).get_fdata().astype(np.float32)
    D, H, W = gt_np.shape
    print(f"[{idx+1}] {tof_file}  原始 ({D},{H},{W})  值域 [{gt_np.min():.1f}, {gt_np.max():.1f}]")

    # 归一化到 [0,1]
    gt_min, gt_max = gt_np.min(), gt_np.max()
    gt_norm = (gt_np - gt_min) / (gt_max - gt_min + 1e-8)
    gt_tensor = torch.from_numpy(gt_norm).unsqueeze(0).unsqueeze(0).to(device).float()

    # AE forward
    with torch.no_grad(), torch.amp.autocast('cuda'):
        out = autoencoder(gt_tensor)
        recon_tensor = out[0] if isinstance(out, (tuple, list)) else out

    recon_np = recon_tensor[0, 0].cpu().numpy().astype(np.float32)

    # 脑区 PSNR（[0,1] 空间）
    mask = gt_norm > 1e-6
    if mask.sum() < 100:
        print(f"  ⚠️ 脑区像素过少，跳过\n")
        continue
    mse = ((gt_norm[mask] - recon_np[mask]) ** 2).mean()
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    psnr_list.append(psnr)
    print(f"  Recon 值域 [{recon_np.min():.4f}, {recon_np.max():.4f}]  |  PSNR: {psnr:.2f} dB\n")

print("=" * 60)
if psnr_list:
    avg = np.mean(psnr_list)
    print(f"📊 平均 AE PSNR: {avg:.2f} dB")
    print(f"{'✅ AE 正常 (≥30 dB)' if avg >= 30 else '❌ AE 有问题 (<30 dB)' if avg < 25 else '⚠️ AE 一般'}")
else:
    print("❌ 无有效结果")
