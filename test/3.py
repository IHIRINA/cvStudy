"""
============================================================
阶段 3：检查 DiT 模型在 latent 空间的预测质量
（已匹配 Inferrer.infer_and_save() 的实际推理参数）
============================================================
"""
import os, sys, json
import torch
import numpy as np
import nibabel as nib

sys.path.insert(0, '/root/mxr/code')
from model import Config, JiT3D_CBCT2CT

device = torch.device("cuda")

JSON_PATH = "/root/autodl-tmp/data/latent_t1ce_tof_pairs.json"
DATA_ROOT = "/root/autodl-tmp/data"
DIT_CKPT = "/root/autodl-tmp/results/checkpoints_56*56*40_4ch/best_model_weights.pth"
NUM_STEPS = 50
NUM_SAMPLES = 3

# ---------- 加载 JSON ----------
with open(JSON_PATH) as f:
    all_data = json.load(f)["training"]
train_data = [x for x in all_data if x["fold"] != 1]
print(f"训练集样本: {len(train_data)}")

# ---------- 加载 DiT ----------
config = Config()
config.save_dir = "/root/autodl-tmp/results/checkpoints_56*56*40_4ch"
model = JiT3D_CBCT2CT(config).to(device)
ckpt = torch.load(DIT_CKPT, map_location=device, weights_only=True)
if "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()
print(f"✅ DiT 加载完成  ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)\n")


def load_latent(fp):
    data = nib.load(fp).get_fdata().astype(np.float32)
    return torch.from_numpy(data).permute(3, 0, 1, 2).unsqueeze(0).to(device)

latent_psnr_per_ch = {0:[], 1:[], 2:[], 3:[]}
avg_psnr_list = []

for idx in range(min(NUM_SAMPLES, len(train_data))):
    s = train_data[idx]

    t1ce_path = os.path.join(DATA_ROOT, s["image"])
    tof_lat_path = os.path.join(DATA_ROOT, s["label"])
    if not os.path.exists(t1ce_path) or not os.path.exists(tof_lat_path):
        print(f"[{idx+1}] ❌ 文件不存在，跳过")
        continue

    t1ce_latent = load_latent(t1ce_path)
    gt_latent = load_latent(tof_lat_path)

    name = os.path.basename(s["image"])
    print(f"[{idx+1}/{NUM_SAMPLES}] {name}")
    print(f"  T1ce latent: {tuple(t1ce_latent.shape)}  值域 [{t1ce_latent.min():.4f}, {t1ce_latent.max():.4f}]")
    print(f"  GT latent:   {tuple(gt_latent.shape)}  值域 [{gt_latent.min():.4f}, {gt_latent.max():.4f}]")

    # ---- DiT 推理（与 Inferrer.infer_and_save 完全一致） ----
    noisy = torch.randn_like(t1ce_latent)
    t_seq = torch.linspace(0.95, 0.05, NUM_STEPS, device=device)  # ← 匹配实际推理
    dt = 1.0 / (NUM_STEPS - 1)

    with torch.no_grad():
        for step in range(NUM_STEPS):
            t = t_seq[step].repeat(1).view(1, 1, 1, 1, 1)
            ct_pred = model(t1ce_latent, noisy, t)
            if step < NUM_STEPS - 1:
                v1 = (ct_pred - noisy) / (1 - t + 1e-5)
                noisy_euler = noisy + dt * v1
                t_next = t_seq[step + 1].repeat(1).view(1, 1, 1, 1, 1)
                ct_pred_next = model(t1ce_latent, noisy_euler, t_next)
                v2 = (ct_pred_next - noisy_euler) / (1 - t_next + 1e-5)
                noisy = noisy + dt * (v1 + v2) / 2.0
            else:
                noisy = ct_pred

    pred_latent = noisy
    print(f"  Pred latent: {tuple(pred_latent.shape)}  值域 [{pred_latent.min():.4f}, {pred_latent.max():.4f}]")

    pred_np = pred_latent[0].cpu().numpy()
    gt_np = gt_latent[0].cpu().numpy()
    ch_psnrs = []
    for ch in range(4):
        mse = ((pred_np[ch] - gt_np[ch]) ** 2).mean()
        dr = gt_np[ch].max() - gt_np[ch].min()
        psnr_ch = 10 * np.log10((dr**2) / mse) if mse > 0 else float('inf')
        latent_psnr_per_ch[ch].append(psnr_ch)
        ch_psnrs.append(psnr_ch)
        print(f"  Ch{ch}: PSNR {psnr_ch:.2f} dB  (GT 值域 [{gt_np[ch].min():.3f},{gt_np[ch].max():.3f}])")

    avg = np.mean(ch_psnrs)
    avg_psnr_list.append(avg)
    print(f"  → 4通道平均: {avg:.2f} dB\n")

print("=" * 60)
if avg_psnr_list:
    overall = np.mean(avg_psnr_list)
    print(f"📊 Latent 空间平均 PSNR: {overall:.2f} dB ({len(avg_psnr_list)} 样本)")
    for ch in range(4):
        print(f"   Ch{ch}: {np.mean(latent_psnr_per_ch[ch]):.2f} dB")
    print()
    if overall >= 30:    print("✅ DiT latent 预测质量高")
    elif overall >= 20:  print("⚠️ DiT latent 中等，可能是瓶颈")
    else:                print("❌ DiT latent 预测差 → 模型训练有问题")
else:
    print("❌ 无有效结果")
