"""
============================================================
阶段 2：检查 Decoder 单独解码预存 TOF latent
============================================================
测试链路：预存的 TOF_latent.nii.gz → Decoder → 重建 TOF
对比：原 TOF 图像 vs 解码结果（脑区 PSNR）

前提：阶段 1 的 AE PSNR ≥ 30 dB（Encoder+Decoder 本身正常）

如果这一步 PSNR 高 → Decoder 没问题，预存 latent 文件也正常
如果这一步 PSNR 低 → 要么 Decoder 有问题，要么预存的 latent 文件坏了
============================================================
"""
import os
import json
import torch
import numpy as np
import nibabel as nib
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

device = torch.device("cuda")

# ========== 配置 ==========
AE_CKPT = "/root/mxr/code/encoder_decoder/autoencoder_epoch273.pt"
JSON_PATH = "/root/autodl-tmp/data_reg/latent_t1ce_tof_pairs.json"
DATA_ROOT = "/root/autodl-tmp/data_reg"
TOF_DIR = "/root/autodl-tmp/data_reg/TOF"
NUM_SAMPLES = 3

# ========== 加载 AE（只用到 decoder） ==========
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

# ========== 加载 JSON（找训练集样本） ==========
with open(JSON_PATH) as f:
    all_data = json.load(f)["training"]
train_data = [x for x in all_data if x["fold"] != 1]
print(f"训练集样本数: {len(train_data)}")

psnr_list = []

for idx in range(min(NUM_SAMPLES, len(train_data))):
    sample = train_data[idx]

    # ---- 1. 加载预存的 TOF latent ----
    latent_rel_path = sample["label"]  # 例如 "TOF_latent/xxx.nii.gz"
    latent_path = os.path.join(DATA_ROOT, latent_rel_path)
    tof_name = os.path.basename(latent_rel_path)
    print(f"\n[{idx+1}] {tof_name}")

    if not os.path.exists(latent_path):
        print(f"  ❌ latent 文件不存在: {latent_path}")
        continue

    latent_img = nib.load(latent_path)
    latent_data = latent_img.get_fdata().astype(np.float32)  # (56,56,40,4)
    print(f"  Latent 文件 shape: {latent_data.shape}  值域: [{latent_data.min():.4f}, {latent_data.max():.4f}]")

    # channel-last → channel-first: (56,56,40,4) → (1,4,56,56,40)
    latent_tensor = torch.from_numpy(latent_data).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    # ---- 2. 加载原始 TOF 图像作为 Ground Truth ----
    tof_path = os.path.join(TOF_DIR, tof_name)
    if not os.path.exists(tof_path):
        print(f"  ❌ TOF 原图不存在: {tof_path}")
        continue
    gt_np = nib.load(tof_path).get_fdata().astype(np.float32)
    D, H, W = gt_np.shape
    print(f"  TOF 原图: ({D},{H},{W})")

    # ---- 3. Decoder 解码 ----
    with torch.no_grad(), torch.amp.autocast('cuda'):
        decoded = autoencoder.decode(latent_tensor)  # (1,1,D',H',W')

    decoded_np = decoded[0, 0].cpu().numpy().astype(np.float32)
    print(f"  Decoder 输出: {decoded_np.shape}  值域: [{decoded_np.min():.4f}, {decoded_np.max():.4f}]")

    # ---- 4. 裁剪到原始尺寸（解码可能因为 padding 略大） ----
    dD = min(D, decoded_np.shape[0])
    dH = min(H, decoded_np.shape[1])
    dW = min(W, decoded_np.shape[2])
    decoded_roi = decoded_np[:dD, :dH, :dW]
    gt_roi = gt_np[:dD, :dH, :dW]

    # ---- 5. 归一化后对比（消除 AE 输出偏移） ----
    gt_norm = (gt_roi - gt_roi.min()) / (gt_roi.max() - gt_roi.min() + 1e-8)
    dec_norm = (decoded_roi - decoded_roi.min()) / (decoded_roi.max() - decoded_roi.min() + 1e-8)

    mask = gt_norm > 1e-6
    if mask.sum() < 100:
        print(f"  ⚠️ 脑区像素太少，跳过")
        continue

    mse = ((gt_norm[mask] - dec_norm[mask]) ** 2).mean()
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    psnr_list.append(psnr)
    print(f"  MSE: {mse:.6f}  |  PSNR (norm): {psnr:.2f} dB")

    # ---- 6. 也在原始值域算一下（做交叉验证） ----
    gt_mask = gt_roi > 1e-6
    if gt_mask.sum() > 100:
        mse_raw = ((gt_roi[gt_mask] - decoded_roi[gt_mask]) ** 2).mean()
        dr = gt_roi[gt_mask].max() - gt_roi[gt_mask].min()
        psnr_raw = 10 * np.log10((dr**2) / mse_raw) if mse_raw > 0 else float('inf')
        print(f"  PSNR (raw HU): {psnr_raw:.2f} dB  (数据范围: {dr:.1f})")

# ========== 汇总 ==========
print("\n" + "=" * 60)
if psnr_list:
    avg = np.mean(psnr_list)
    print(f"📊 平均 PSNR: {avg:.2f} dB  ({len(psnr_list)} 样本)")
    print()
    if avg >= 30:
        print("✅ 判定: Decoder 单独工作正常，预存的 TOF latent 也正常")
        print("   → 问题根源大概率在 DiT 模型（继续阶段 3）")
    elif avg >= 25:
        print("⚠️  判定: Decoder 解码质量一般，可能对最终 PSNR 有一定影响")
    else:
        print("❌ 判定: Decoder 解码有问题 / 预存 latent 文件异常")
else:
    print("❌ 无有效结果")
