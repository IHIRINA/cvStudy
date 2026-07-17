'''
import nibabel as nib
import numpy as np

gt = nib.load('/root/autodl-tmp/data/TOF/8-1100208131-TOF_pre.nii.gz').get_fdata()
pred = nib.load('/root/autodl-tmp/results/checkpoints_56*56*40_4ch/inference_results_recon/8-1100208131-T1ce.nii.gz').get_fdata()

mask = gt > 1e-6
gt_brain = gt[mask]
pred_brain = pred[mask]

# 值域
print(f"GT 脑区范围: [{gt_brain.min():.4f}, {gt_brain.max():.4f}]")
print(f"Pred 脑区范围: [{pred_brain.min():.4f}, {pred_brain.max():.4f}]")

# 差值统计
diff = gt_brain - pred_brain
print(f"差值 mean: {diff.mean():.4f}, std: {diff.std():.4f}")
print(f"差值范围: [{diff.min():.4f}, {diff.max():.4f}]")
print(f"MAE: {np.abs(diff).mean():.4f}")
print(f"MSE: {(diff**2).mean():.4f}")

# 相关性
corr = np.corrcoef(gt_brain, pred_brain)[0,1]
print(f"Pearson 相关系数: {corr:.4f}")
'''

'''
import nibabel as nib
import numpy as np
from scipy.ndimage import shift

gt = nib.load('/root/autodl-tmp/data_reg/TOF/8-1100208131-TOF_pre.nii.gz').get_fdata()
pred = nib.load('/root/autodl-tmp/results/checkpoints_56*56*40_4ch/inference_results_recon/8-1100208131-T1ce.nii.gz').get_fdata()

mask = gt > 1e-6

# 先各自 minmax 归一化到 [0,1]
gt_norm = gt.copy()
pred_norm = pred.copy()
gt_norm[mask] = (gt[mask] - gt[mask].min()) / (gt[mask].max() - gt[mask].min())
pred_norm[mask] = (pred[mask] - pred[mask].min()) / (pred[mask].max() - pred[mask].min())

# 归一化后的 PSNR
mse = ((gt_norm[mask] - pred_norm[mask])**2).mean()
psnr = 20 * np.log10(1.0 / np.sqrt(mse))
print(f"各自 minmax 归一化后 PSNR: {psnr:.2f} dB")
print(f"MSE: {mse:.6f}")

# 用 3D SSIM 库算一下（考虑结构的）
from skimage.metrics import structural_similarity as ssim
# 取中间切片
mid = gt.shape[2] // 2
for i in range(mid-2, mid+3):
    s = ssim(gt_norm[:,:,i], pred_norm[:,:,i], data_range=1.0)
    print(f"Slice {i} SSIM: {s:.4f}")
'''

import nibabel as nib
import numpy as np

gt = nib.load('/root/autodl-tmp/data/TOF/1-1001398096-TOF_pre.nii.gz').get_fdata()
pred = nib.load('/root/autodl-tmp/data/TOF_recon/1-1001398096-TOF_pre.nii.gz').get_fdata()

# 只对比脑区
mask = gt > 1e-6

# 各自 minmax 归一化到 [0,1]
gt_norm = np.zeros_like(gt)
pred_norm = np.zeros_like(pred)
gt_norm[mask] = (gt[mask] - gt[mask].min()) / (gt[mask].max() - gt[mask].min())
pred_norm[mask] = (pred[mask] - pred[mask].min()) / (pred[mask].max() - pred[mask].min())

# 计算 PSNR
mse = ((gt_norm[mask] - pred_norm[mask])**2).mean()
psnr = 20 * np.log10(1.0 / np.sqrt(mse))
print(f"PSNR: {psnr:.2f} dB")

# 计算 SSIM
from skimage.metrics import structural_similarity as ssim
# 取中间切片
mid = gt.shape[2] // 2
s = ssim(gt_norm[:,:,mid], pred_norm[:,:,mid], data_range=1.0)
print(f"SSIM (slice {mid}): {s:.4f}")
