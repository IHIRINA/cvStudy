"""
compare_latents.py - 直接对比两个 latent 文件，找出差异点
"""
import numpy as np
import nibabel as nib

real_path = "/root/autodl-tmp/data/current_latent_0.nii.gz"   # 1.5.py 中实时编码保存的
saved_path = "/root/autodl-tmp/data/TOF_latent/1-1001398096-TOF_pre.nii.gz"  # 预存

real = nib.load(real_path).get_fdata().astype(np.float32)
saved = nib.load(saved_path).get_fdata().astype(np.float32)

# 转置为 (C,D,H,W) 以便逐通道对比
real = np.transpose(real, (3,0,1,2))
saved = np.transpose(saved, (3,0,1,2))

print(f"real shape: {real.shape}, saved shape: {saved.shape}")
print(f"real stats: min={real.min()}, max={real.max()}, mean={real.mean()}")
print(f"saved stats: min={saved.min()}, max={saved.max()}, mean={saved.mean()}")

diff = np.abs(real - saved)
print(f"max diff: {diff.max()}, mean diff: {diff.mean()}")

# 找出最大差异的位置
idx = np.unravel_index(np.argmax(diff), diff.shape)
print(f"最大差异位置: 通道{idx[0]}, 坐标({idx[1]},{idx[2]},{idx[3]})")
print(f"real 值: {real[idx]}, saved 值: {saved[idx]}")

# 检查整体是否一致
if np.allclose(real, saved, atol=1e-5):
    print("✅ 两个文件数值一致")
else:
    print("❌ 两个文件数值不一致")