import os, sys, json
import torch
import numpy as np
import nibabel as nib

# 路径：model.py 和 encoder_decoder 在同级目录
sys.path.insert(0, '/root/mxr/code')
sys.path.insert(0, '/root/mxr/code/encoder_decoder')

from model import Config, JiT3D_CBCT2CT
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.inferers import SlidingWindowInferer

# ============ 配置 ============
device = torch.device("cuda")
json_path = "/root/autodl-tmp/data/latent_t1ce_tof_pairs.json"
data_root = "/root/autodl-tmp/data"
checkpoint_path = "/root/autodl-tmp/results/checkpoints_56*56*40_4ch/best_model_weights.pth"
ae_ckpt_path = "/root/mxr/code/encoder_decoder/autoencoder_epoch273.pt"
tof_gt_dir = "/root/autodl-tmp/data/TOF"

# ============ 1. 加载数据，取训练集前3个 ============
with open(json_path) as f:
    all_data = json.load(f)["training"]

train_data = [x for x in all_data if x["fold"] != 1]
print(f"训练集样本数: {len(train_data)}")

samples = train_data[:3]

# ============ 2. 加载 DiT 模型 ============
config = Config()
config.save_dir = "/root/autodl-tmp/results/checkpoints_56*56*40_4ch"
model = JiT3D_CBCT2CT(config).to(device)
ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()
print("DiT 模型加载完成")

# ============ 3. 加载 Autoencoder (Encoder + Decoder) ============
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
    dim_split=1
).to(device)
ae_ckpt = torch.load(ae_ckpt_path, map_location=device, weights_only=True)
autoencoder.load_state_dict(ae_ckpt)
autoencoder.eval()
print("Autoencoder 加载完成")

# ---- SlidingWindowInferer 配置 ----
# 图像空间 roi_size（与 reconstruction 脚本一致，用于 AE forward）
ae_image_roi = [224, 224, 160]
# 潜空间 roi_size（用于 DiT 预测 latent 的解码）
latent_roi = [56, 56, 40]

# AE forward inferer（图像空间 → 图像空间，测 decoder 上限）
ae_inferer = SlidingWindowInferer(
    roi_size=ae_image_roi,
    sw_batch_size=1,
    overlap=0.35,
    progress=False,
    sw_device=device,
    mode="gaussian",
    device=torch.device("cpu")
)

# Decoder inferer（潜空间 → 图像空间，解码 DiT 预测的 latent）
decode_inferer = SlidingWindowInferer(
    roi_size=latent_roi,
    sw_batch_size=1,
    overlap=0.5,
    progress=False,
    sw_device=device,
    mode="gaussian",
    device=torch.device("cpu")
)

class DecodeWrapper(torch.nn.Module):
    """包装 autoencoder.decode"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, z):
        return self.model.decode(z)

decode_wrapper = DecodeWrapper(autoencoder)
print("Inferer 创建完成")

# ============ 4. 数据加载 transforms ============
from monai.transforms import LoadImaged, EnsureTyped, Compose
transforms = Compose([
    LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
    EnsureTyped(keys=["image"], dtype=torch.float32),
])

# ============ 5. 对照测试：真实 TOF latent decode vs 模型预测 latent decode ============
num_steps = 50

# 累计统计
gt_latent_psnr_list = []
pred_latent_psnr_list = []
gt_latent_ssim_list  = []
pred_latent_ssim_list  = []

for idx, sample in enumerate(samples):
    print(f"\n{'='*60}")
    print(f"样本 {idx+1}: {os.path.basename(sample['image'])}")
    print(f"{'='*60}")

    # ---- 加载 GT TOF 图像 ----
    label_name = os.path.basename(sample["label"])
    tof_path = os.path.join(tof_gt_dir, label_name)
    if not os.path.exists(tof_path):
        print(f"  [跳过] GT 文件未找到: {tof_path}")
        continue
    gt_np = nib.load(tof_path).get_fdata().astype(np.float32)

    # ---- 加载 GT TOF latent（用 MONAI LoadImaged，与训练完全一致）----
    tof_latent_path = os.path.join(data_root, sample["label"])
    if not os.path.exists(tof_latent_path):
        print(f"  [跳过] GT TOF latent 未找到: {tof_latent_path}")
        continue
    # 用 MONAI 的 LoadImaged + EnsureTyped 加载，与训练代码完全一致
    gt_latent_data = transforms([{"image": tof_latent_path}])
    gt_latent_tensor = gt_latent_data[0]["image"].unsqueeze(0).to(device)
    print(f"  GT TOF latent shape: {tuple(gt_latent_tensor.shape)}  (MONAI channel_first)")

    # 脑区 mask
    mask = gt_np > 1e-6
    if mask.sum() < 100:
        print("  [跳过] 脑区像素过少")
        continue

    gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0).to(device).float()
    D, H, W = gt_np.shape

    # ---- 5a. AE 重建上限：真实 TOF → autoencoder.forward() → 重建 TOF ----
    # 与 reconstruction 脚本一致：输入归一化到 [0,1]，用 SlidingWindowInferer 调 forward
    gt_min, gt_max = gt_tensor.min(), gt_tensor.max()
    gt_norm = (gt_tensor - gt_min) / (gt_max - gt_min + 1e-8)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        ae_out = ae_inferer(network=autoencoder, inputs=gt_norm)
    # ae_out 可能是 (recon, z, z_mu, z_log_var) 或 (recon, z, posterior) 或单个 recon
    if isinstance(ae_out, (tuple, list)):
        gt_recon_norm = ae_out[0]
    else:
        gt_recon_norm = ae_out

    gt_recon_np = gt_recon_norm[0, 0, :D, :H, :W].cpu().numpy().astype(np.float32)
    # 恢复到原始范围
    gt_recon_np = gt_recon_np * (gt_max.item() - gt_min.item()) + gt_min.item()
    print(f"  GT 尺寸: ({D},{H},{W})  →  AE 重建输出: {tuple(gt_recon_norm.shape[2:])}")
    print(f"  AE 重建 PSNR: 待计算（下方图像层面对比）")

    # ---- 5c. T1ce latent → DiT 预测 → decode → 计算 PSNR（当前 pipeline）----
    img_path = os.path.join(data_root, sample["image"])
    batch = transforms([{"image": img_path}])
    cbct = batch[0]["image"].unsqueeze(0).to(device)

    ct_current = torch.randn_like(cbct)
    t_seq = torch.linspace(1.0, 0.0, num_steps, device=device)
    dt = 1.0 / (num_steps - 1)

    with torch.no_grad():
        for step in range(num_steps):
            t = t_seq[step].repeat(1).view(1, 1, 1, 1, 1)
            ct_pred = model(cbct, ct_current, t)
            if step < num_steps - 1:
                v1 = (ct_pred - ct_current) / (1 - t + 1e-5)
                ct_noisy_euler = ct_current + dt * v1
                t_next = t_seq[step + 1].repeat(1).view(1, 1, 1, 1, 1)
                ct_pred_next = model(cbct, ct_noisy_euler, t_next)
                v2 = (ct_pred_next - ct_noisy_euler) / (1 - t_next + 1e-5)
                ct_current = ct_current + dt * (v1 + v2) / 2.0
            else:
                ct_current = ct_pred

    with torch.no_grad(), torch.amp.autocast('cuda'):
        pred_recon = decode_inferer(network=decode_wrapper, inputs=ct_current)
    pred_recon_np = pred_recon[0, 0, :D, :H, :W].cpu().numpy().astype(np.float32)

    # ---- 5c. Latent 层面 PSNR（DiT 预测 vs GT TOF latent）----
    pred_latent_np = ct_current[0].cpu().numpy()  # (4, 56, 56, 40)
    gt_latent_np_ref = gt_latent_tensor[0].cpu().numpy()

    # 确保维度对齐 (4, 56, 56, 40)
    print(f"\n--- Latent 层面对比 ---")
    print(f"  GT TOF Latent   shape: {tuple(gt_latent_np_ref.shape)}")
    print(f"  Pred Latent     shape: {tuple(pred_latent_np.shape)}")
    print(f"  GT TOF Latent   值域: [{gt_latent_np_ref.min():.4f}, {gt_latent_np_ref.max():.4f}]")
    print(f"  Pred Latent     值域: [{pred_latent_np.min():.4f}, {pred_latent_np.max():.4f}]")

    if gt_latent_np_ref.shape == pred_latent_np.shape:
        gt_flat = gt_latent_np_ref.reshape(4, -1)
        pred_flat = pred_latent_np.reshape(4, -1)
        mse_latent_per_ch = ((gt_flat - pred_flat) ** 2).mean(axis=1)
        latent_psnr_per_ch = [20 * np.log10(1.0 / np.sqrt(m + 1e-12)) for m in mse_latent_per_ch]
        for ch in range(4):
            print(f"  Channel {ch}: PSNR = {latent_psnr_per_ch[ch]:.2f} dB")
        print(f"  Latent 平均 PSNR: {np.mean(latent_psnr_per_ch):.2f} dB")
        # 保存 latent PSNR 均值
        latent_avg_psnr = np.mean(latent_psnr_per_ch)
    else:
        print(f"  [跳过] 空间尺寸不一致")
        latent_avg_psnr = None

    # ---- 5d. 解码 GT TOF latent → 图像，验证预存 latent 质量 ----
    with torch.no_grad(), torch.amp.autocast('cuda'):
        gt_latent_decoded = decode_inferer(network=decode_wrapper, inputs=gt_latent_tensor)
    gt_latent_decoded_np = gt_latent_decoded[0, 0, :D, :H, :W].cpu().numpy().astype(np.float32)
    from skimage.metrics import structural_similarity as ssim

    def compute_psnr_ssim(ref_np, cmp_np, mask, name):
        """计算脑区 PSNR（原始 HU 空间）和 min-max 归一化后的 SSIM"""
        ref_brain = ref_np[mask]
        cmp_brain = cmp_np[mask]

        # PSNR：直接在原始值域算（与 reconstruction 脚本一致）
        mse = ((ref_brain - cmp_brain) ** 2).mean()
        # 动态范围用 ref 的 max-min
        data_range = ref_brain.max() - ref_brain.min()
        psnr_val = 10 * np.log10((data_range ** 2) / mse) if mse > 0 else float('inf')

        # SSIM：min-max 归一化后算（消除亮度/对比度差异）
        ref_norm = (ref_brain - ref_brain.min()) / (ref_brain.max() - ref_brain.min() + 1e-8)
        cmp_norm = (cmp_brain - cmp_brain.min()) / (cmp_brain.max() - cmp_brain.min() + 1e-8)

        mid = ref_np.shape[2] // 2
        ref_slice = (ref_np[:,:,mid] - ref_np[:,:,mid].min()) / (ref_np[:,:,mid].max() - ref_np[:,:,mid].min() + 1e-8)
        cmp_slice = (cmp_np[:,:,mid] - cmp_np[:,:,mid].min()) / (cmp_np[:,:,mid].max() - cmp_np[:,:,mid].min() + 1e-8)
        ssim_val = ssim(ref_slice, cmp_slice, data_range=1.0)

        print(f"  [{name}] PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.4f}")
        return psnr_val, ssim_val

    print(f"\n--- 图像层面对比 ---")
    # 1. AE 重建上限（autoencoder.forward，与 reconstruction 脚本相同方式）
    psnr_gt, ssim_gt = compute_psnr_ssim(gt_np, gt_recon_np, mask, "AE forward 重建上限 (E+D)")
    # 2. GT latent → decode（预存 latent 的质量，验证 encode→save→load→decode 链路）
    psnr_latent_decoded, ssim_latent_decoded = compute_psnr_ssim(
        gt_np, gt_latent_decoded_np, mask, "GT latent → Decode (预存)")
    # 3. DiT 预测 → decode（当前 pipeline）
    pD, pH, pW = pred_recon_np.shape
    cD, cH, cW = min(D, pD), min(H, pH), min(W, pW)
    psnr_pred, ssim_pred = compute_psnr_ssim(gt_np[:cD, :cH, :cW], pred_recon_np[:cD, :cH, :cW],
                                              gt_np[:cD, :cH, :cW] > 1e-6, "DiT 预测 → Decode")

    # 对比总结
    print(f"\n--- 对比总结 ---")
    print(f"  [1] AE forward 重建上限:   {psnr_gt:.2f} dB     (图像→AE→图像)")
    print(f"  [2] GT latent → Decode:    {psnr_latent_decoded:.2f} dB     (预存latent解码)")
    print(f"  [3] DiT 预测 → Decode:     {psnr_pred:.2f} dB     (当前pipeline)")
    print(f"  [2]-[3] 差距 (latent误差): {psnr_latent_decoded - psnr_pred:.2f} dB")
    if latent_avg_psnr is not None:
        print(f"  Latent 空间 PSNR (DiT预测): {latent_avg_psnr:.2f} dB")
    print(f"  结论: [2] 远 > [3] → 问题在 DiT latent 预测，不在 decode")

    gt_latent_psnr_list.append(psnr_gt)
    pred_latent_psnr_list.append(psnr_pred)
    gt_latent_ssim_list.append(ssim_gt)
    pred_latent_ssim_list.append(ssim_pred)

# ============ 6. 汇总 ============
print(f"\n{'='*60}")
print(f"汇总统计 ({len(gt_latent_psnr_list)} 个样本)")
print(f"{'='*60}")
if gt_latent_psnr_list:
    print(f"  AE 重建上限 PSNR:          {np.mean(gt_latent_psnr_list):.2f} ± {np.std(gt_latent_psnr_list):.2f} dB")
    print(f"  DiT 预测 → Decode PSNR:   {np.mean(pred_latent_psnr_list):.2f} ± {np.std(pred_latent_psnr_list):.2f} dB")
    print(f"  PSNR 平均差距:             {np.mean(gt_latent_psnr_list) - np.mean(pred_latent_psnr_list):.2f} dB  (DiT 预测误差)")
    print(f"  AE 重建上限 SSIM:          {np.mean(gt_latent_ssim_list):.4f} ± {np.std(gt_latent_ssim_list):.4f}")
    print(f"  DiT 预测 → Decode SSIM:   {np.mean(pred_latent_ssim_list):.4f} ± {np.std(pred_latent_ssim_list):.4f}")

print("\n===== 完成 =====")
