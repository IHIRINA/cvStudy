"""
快速验证：maisi_encoder_full.py 里 encoder 实际吃到的输入
"""
import os, sys
import torch
import numpy as np
import nibabel as nib

# 直接在 TOF 原图上做简单 min-max → encoder → 得到 latent
# 然后在同一个 TOF 图上走 VAE_Transform → encoder → 得到 latent
# 对比两版 latent

sys.path.insert(0, '/root/mxr/code/encoder_decoder')
from transforms import VAE_Transform
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.inferers import SlidingWindowInferer

device = torch.device("cuda")
AE_CKPT = "/root/mxr/code/encoder_decoder/autoencoder_epoch273.pt"

# 加载 AE
ae = AutoencoderKlMaisi(
    spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4,
    num_channels=[64,128,256], num_res_blocks=[2,2,2],
    norm_num_groups=32, norm_eps=1e-06,
    attention_levels=[False,False,False],
    with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
    use_checkpointing=False, use_convtranspose=False,
    norm_float16=True, num_splits=8, dim_split=1
).to(device)
ae.load_state_dict(torch.load(AE_CKPT, map_location=device, weights_only=True))
ae.eval()

# inferer（和 maisi_encoder_full.py 一致）
encode_inferer = SlidingWindowInferer(
    roi_size=[128,128,64], sw_batch_size=1, overlap=0.25,
    mode="gaussian", progress=False, sw_device=device, device=device
)

# VAE_Transform（和 maisi_encoder_full.py 一致）
transform = VAE_Transform(
    is_train=False, random_aug=False, k=4, val_patch_size=None,
    output_dtype=torch.float64, image_keys=["image"],
    label_keys=[], additional_keys=[], select_channel=0
)

# 取第一张 TOF
TOF_DIR = "/root/autodl-tmp/data_reg/TOF"
fname = sorted(os.listdir(TOF_DIR))[0]
tof_path = os.path.join(TOF_DIR, fname)
tof_raw = nib.load(tof_path).get_fdata().astype(np.float32)
print(f"📂 {fname}")
print(f"   文件值域: [{tof_raw.min():.6f}, {tof_raw.max():.6f}]")

# --- 方法1: 简单 min-max → [0,1] ---
norm1 = (tof_raw - tof_raw.min()) / (tof_raw.max() - tof_raw.min() + 1e-8)
t1 = torch.from_numpy(norm1).unsqueeze(0).unsqueeze(0).to(device)
with torch.no_grad(), torch.amp.autocast('cuda'):
    z1 = encode_inferer(network=ae.encode_stage_2_inputs, inputs=t1)
lat1 = z1[0].cpu().numpy()
print(f"\n方法1 (min-max): 输入值域 [{t1.min():.6f}, {t1.max():.6f}]")
print(f"   输出 latent 值域: [{lat1.min():.4f}, {lat1.max():.4f}]")

# --- 方法2: VAE_Transform ---
# VAE_Transform 需要 dict 格式输入
from monai.data import CacheDataset, DataLoader
data_dict = [{"image": tof_path, "class": "mri"}]
ds = CacheDataset(data=data_dict, transform=transform, cache_rate=1, num_workers=0)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
batch = next(iter(dl))[0]
img2 = batch["image"].to(device).contiguous()
model_dtype = next(ae.parameters()).dtype
img2 = img2.to(dtype=model_dtype)

print(f"\n方法2 (VAE_Transform): 输入值域 [{img2.min():.6f}, {img2.max():.6f}]")
print(f"   输入 shape: {tuple(img2.shape)}, dtype: {img2.dtype}")

# 打印 VAE_Transform 的百分位信息
if hasattr(batch["image"], 'percentile_lower'):
    print(f"   VAE percentile_lower: {batch['image'].percentile_lower}")
    print(f"   VAE percentile_upper: {batch['image'].percentile_upper}")

with torch.no_grad(), torch.amp.autocast('cuda'):
    z2 = encode_inferer(network=ae.encode_stage_2_inputs, inputs=img2)
lat2 = z2[0].cpu().numpy()
print(f"   输出 latent 值域: [{lat2.min():.4f}, {lat2.max():.4f}]")

# --- 对比 ---
print(f"\n{'='*60}")
print("两版输入是否一致:", torch.allclose(t1.squeeze(), img2.squeeze().cpu(), rtol=1e-4))
print(f"输入 max_diff: {(t1 - img2).abs().max().item():.6f}")

for ch in range(4):
    mse = ((lat1[ch] - lat2[ch]) ** 2).mean()
    dr = lat2[ch].max() - lat2[ch].min()
    psnr = 10 * np.log10((dr**2) / mse) if mse > 0 else float('inf')
    print(f"Ch{ch}: 两版 latent max_diff={np.abs(lat1[ch]-lat2[ch]).max():.6f}  PSNR={psnr:.2f} dB")

avg = 10 * np.log10((max(dr, 1))**2 / ((lat1-lat2)**2).mean()) if ((lat1-lat2)**2).mean() > 0 else float('inf')
print(f"\n整体 latent PSNR: {avg:.2f} dB")
print(f"如果 max_diff 接近 0 → 问题不在 VAE_Transform")
print(f"如果 PSNR 低 → encoder 输入被 VAE_Transform 改动了")
