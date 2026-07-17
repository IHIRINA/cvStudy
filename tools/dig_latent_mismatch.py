"""诊断 latent 尺寸和训练数据的不匹配问题"""
import json, os, numpy as np, nibabel as nib, torch

json_path = "/root/autodl-tmp/data/latent_t1ce_tof_pairs.json"
tof_gt_dir = "/root/autodl-tmp/data/TOF"
data_root = "/root/autodl-tmp/data"

with open(json_path) as f:
    all_data = json.load(f)

train = [x for x in all_data["training"] if x["fold"] != 1][:3]

for i, s in enumerate(train):
    print(f"\n===== 样本 {i+1} =====")
    
    # 1. T1ce latent 文件
    t1ce_path = os.path.join(data_root, s["image"])
    print(f"T1ce 文件: {os.path.basename(t1ce_path)}")
    if t1ce_path.endswith(".npy"):
        t1ce = np.load(t1ce_path)
        print(f"T1ce latent shape: {t1ce.shape}")
    elif t1ce_path.endswith(".pt"):
        t1ce = torch.load(t1ce_path, map_location="cpu", weights_only=True)
        print(f"T1ce latent shape: {tuple(t1ce.shape)}")
    elif t1ce_path.endswith(".nii.gz") or t1ce_path.endswith(".nii"):
        t1ce = nib.load(t1ce_path).get_fdata()
        print(f"T1ce latent shape: {t1ce.shape}")
    else:
        print(f"未知格式: {t1ce_path}")
        continue

    # 2. TOF GT 图像
    label_name = os.path.basename(s["label"])
    tof_path = os.path.join(tof_gt_dir, label_name)
    tof = nib.load(tof_path).get_fdata()
    print(f"TOF GT 图像 shape: {tof.shape}")

    # 3. 如果 TOF 224→encode→28，T1ce latent 56 → 推断 T1ce 图像尺寸
    if len(t1ce.shape) >= 3:
        # latent 最后三维是空间
        lat_d = t1ce.shape[-3] if t1ce.shape[-3] > 4 else t1ce.shape[-3]
        lat_h = t1ce.shape[-2]
        lat_w = t1ce.shape[-1]
        if len(t1ce.shape) == 4:
            lat_d, lat_h, lat_w = t1ce.shape[1:]
        print(f"T1ce latent 空间尺寸: ({lat_d}, {lat_h}, {lat_w})")
        print(f"→ 推算 T1ce 原始图像: {lat_d*8}×{lat_h*8}×{lat_w*8}")
    
    print(f"TOF 图像: {tof.shape[0]}×{tof.shape[1]}×{tof.shape[2]}")
    print(f"→ TOF latent 应为: {tof.shape[0]//8}×{tof.shape[1]//8}×{tof.shape[2]//8}")
    
    # 4. 关键判断
    d, h, w = tof.shape
    if d//8 != lat_d or h//8 != lat_h or w//8 != lat_w:
        print(f"\n⚠️  【问题定位】T1ce latent 空间 ({lat_d},{lat_h},{lat_w}) ≠ TOF latent 空间 ({d//8},{h//8},{w//8})")
        print(f"   这意味着 DiT 输入/输出的 latent 空间和 TOF encoder 输出的 latent 空间不匹配！")
    else:
        print(f"\n✅  latent 空间一致")

print("\n===== 诊断完成 =====")
