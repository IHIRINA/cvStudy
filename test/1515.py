"""
1.5_deterministic.py - 单独检查 Encoder（确定性版本）
对比实时编码与预存 latent，应完全一致
"""
import os
import time
import torch
import numpy as np
import nibabel as nib
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.utils import set_determinism
from sys import path
path.insert(0, "/root/mxr/code/encoder_decoder")
from transforms import VAE_Transform

# ========== 强制确定性 ==========
set_determinism(seed=0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda")
AE_CKPT = "/root/mxr/code/encoder_decoder/autoencoder_epoch273.pt"
TOF_DIR = "/root/autodl-tmp/data_before/TOF"
TOF_LATENT_DIR = "/root/autodl-tmp/data/TOF_latent"
NUM_SAMPLES = 3  # 可调整

# ---------- 加载模型（与生成脚本完全一致） ----------
ae = AutoencoderKlMaisi(
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
    norm_float16=False,      # 与生成一致
    num_splits=1,
    dim_split=1
).to(device)
ae.load_state_dict(torch.load(AE_CKPT, map_location=device, weights_only=True))
ae.eval()
print("✅ AE 加载完成 (float32, num_splits=1)\n")

# ---------- 预处理（与生成脚本完全一致） ----------
vae_transform = VAE_Transform(
    is_train=False,
    random_aug=False,
    k=4,
    patch_size=[128, 128, 128],
    val_patch_size=None,
    output_dtype=torch.float32,
    spacing_type="original",
    image_keys=["image"],
    select_channel=0,
)

# ---------- 取 TOF 文件 ----------
tof_files = sorted([f for f in os.listdir(TOF_DIR) if f.endswith(('.nii.gz', '.nii'))])[:NUM_SAMPLES]

latent_psnr_per_ch = {0: [], 1: [], 2: [], 3: []}
avg_psnr_list = []

for idx, fname in enumerate(tof_files):
    tof_path = os.path.join(TOF_DIR, fname)
    latent_path = os.path.join(TOF_LATENT_DIR, fname)

    if not os.path.exists(latent_path):
        print(f"[{idx+1}] ❌ 预存 latent 不存在: {latent_path}")
        continue

    mtime = time.ctime(os.path.getmtime(latent_path))
    print(f"[{idx+1}] {fname}  (修改时间: {mtime})")

    # ---- 实时编码 ----
    data = {"image": tof_path, "class": "mri"}
    transformed, _ = vae_transform(data)
    image = transformed["image"].unsqueeze(0).to(device)  # float32
    print(f"  VAE_Transform 后: {image.shape} 值域 [{image.min():.4f}, {image.max():.4f}] (float32)")

    with torch.no_grad():
        z = ae.encode_stage_2_inputs(image)
    latent_realtime = z[0].cpu().numpy().astype(np.float32)
    print(f"  实时编码 latent: {latent_realtime.shape}  值域 [{latent_realtime.min():.4f}, {latent_realtime.max():.4f}]")

    # ---- 读取预存 latent ----
    latent_saved_img = nib.load(latent_path)
    latent_saved = latent_saved_img.get_fdata().astype(np.float32)
    latent_saved = np.transpose(latent_saved, (3, 0, 1, 2))
    print(f"  预存 latent: {latent_saved.shape}  值域 [{latent_saved.min():.4f}, {latent_saved.max():.4f}]")

    # ---- 逐通道 PSNR 与像素级差异 ----
    diff = np.abs(latent_realtime - latent_saved)
    max_diff = diff.max()
    mean_diff = diff.mean()
    print(f"  🔍 像素级对比: max_diff={max_diff:.10f}, mean_diff={mean_diff:.10f}")

    ch_psnrs = []
    for ch in range(4):
        r = latent_realtime[ch].ravel()
        s = latent_saved[ch].ravel()
        mse = ((r - s) ** 2).mean()
        dr = s.max() - s.min()
        psnr_ch = 10 * np.log10((dr**2) / mse) if mse > 0 else float('inf')
        latent_psnr_per_ch[ch].append(psnr_ch)
        ch_psnrs.append(psnr_ch)
        diff_max_ch = np.abs(r - s).max()
        print(f"  Ch{ch}: PSNR {psnr_ch:.2f} dB  |  实时[{r.min():.4f},{r.max():.4f}]  预存[{s.min():.4f},{s.max():.4f}]  max_diff={diff_max_ch:.6f}")

    avg = np.mean(ch_psnrs)
    avg_psnr_list.append(avg)
    print(f"  → 4通道平均 PSNR: {avg:.2f} dB\n")

# ---------- 汇总 ----------
print("=" * 60)
if avg_psnr_list:
    overall = np.mean(avg_psnr_list)
    print(f"📊 Encoder 一致性 PSNR: {overall:.2f} dB ({len(avg_psnr_list)} 样本)")
    for ch in range(4):
        print(f"   Ch{ch}: {np.mean(latent_psnr_per_ch[ch]):.2f} dB")
    print()
    if overall >= 50:
        print("✅ 判定: 实时编码与预存 latent 完全一致")
    else:
        print("❌ 判定: 实时编码与预存 latent 不一致，请检查生成脚本是否使用了相同的设置")
else:
    print("❌ 无有效结果")