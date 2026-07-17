import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# ========== 用户配置区 ==========
tof_dir = "/root/autodl-tmp/data/TOF"               # TOF 原图文件夹
recon_dir = "/root/autodl-tmp/data/TOF_recon"       # TOF 重建图文件夹
output_dir = "/root/autodl-tmp/data/intensity_plots" # 对比图保存文件夹
# ================================

# 创建输出文件夹（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 获取 TOF 原图的所有 .nii.gz 文件
tof_files = glob.glob(os.path.join(tof_dir, "*.nii.gz"))
print(f"找到 {len(tof_files)} 个 TOF 原图文件")

for tof_path in tof_files:
    # 提取文件名（如 "10-1100216610-TOF_pre.nii.gz"）
    basename = os.path.basename(tof_path)
    # 对应的重建图路径
    recon_path = os.path.join(recon_dir, basename)

    # 检查重建图是否存在
    if not os.path.exists(recon_path):
        print(f"警告：未找到对应的重建图 {recon_path}，跳过 {basename}")
        continue

    try:
        # 读取 NIfTI 图像（返回 numpy 数组）
        tof_img = nib.load(tof_path).get_fdata()
        recon_img = nib.load(recon_path).get_fdata()

        # 确保两者形状一致
        if tof_img.shape != recon_img.shape:
            print(f"警告：{basename} 形状不一致 (GT:{tof_img.shape} vs Recon:{recon_img.shape})，跳过")
            continue

        # 假设图像维度顺序为 (X, Y, Z) 或 (x, y, z)
        # 计算三个正交方向上的平均强度曲线
        # 沿 X 轴：固定 x，平均所有 y,z → 长度 = X
        x_axis_gt = np.mean(tof_img, axis=(1, 2))
        x_axis_pred = np.mean(recon_img, axis=(1, 2))

        # 沿 Y 轴：固定 y，平均所有 x,z → 长度 = Y
        y_axis_gt = np.mean(tof_img, axis=(0, 2))
        y_axis_pred = np.mean(recon_img, axis=(0, 2))

        # 沿 Z 轴：固定 z，平均所有 x,y → 长度 = Z
        z_axis_gt = np.mean(tof_img, axis=(0, 1))
        z_axis_pred = np.mean(recon_img, axis=(0, 1))

        # 可选：强度归一化（使原图和重建图在相同尺度下对比）
        # 取消注释下面四行即可启用（各自除以其最大值）
        # max_gt = max(x_axis_gt.max(), y_axis_gt.max(), z_axis_gt.max())
        # max_pred = max(x_axis_pred.max(), y_axis_pred.max(), z_axis_pred.max())
        # x_axis_gt, y_axis_gt, z_axis_gt = x_axis_gt/max_gt, y_axis_gt/max_gt, z_axis_gt/max_gt
        # x_axis_pred, y_axis_pred, z_axis_pred = x_axis_pred/max_pred, y_axis_pred/max_pred, z_axis_pred/max_pred

        # 绘图
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # X 轴曲线
        axs[0].plot(x_axis_gt, label='Ground Truth', color='blue')
        axs[0].plot(x_axis_pred, label='Prediction', color='red')
        axs[0].set_title(f"{basename} - X-axis Intensity Profile")
        axs[0].set_xlabel("Position along X (voxel index)")
        axs[0].set_ylabel("Mean Intensity")
        axs[0].legend()
        axs[0].grid(True)

        # Y 轴曲线
        axs[1].plot(y_axis_gt, label='Ground Truth', color='blue')
        axs[1].plot(y_axis_pred, label='Prediction', color='red')
        axs[1].set_title(f"{basename} - Y-axis Intensity Profile")
        axs[1].set_xlabel("Position along Y (voxel index)")
        axs[1].set_ylabel("Mean Intensity")
        axs[1].legend()
        axs[1].grid(True)

        # Z 轴曲线
        axs[2].plot(z_axis_gt, label='Ground Truth', color='blue')
        axs[2].plot(z_axis_pred, label='Prediction', color='red')
        axs[2].set_title(f"{basename} - Z-axis Intensity Profile")
        axs[2].set_xlabel("Position along Z (slice index)")
        axs[2].set_ylabel("Mean Intensity")
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()

        # 保存图片（去掉 .nii.gz 后缀，加上 .png）
        save_name = basename.replace('.nii.gz', '.png')
        save_path = os.path.join(output_dir, save_name)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)   # 关闭图形，释放内存

        print(f"已保存：{save_path}")

    except Exception as e:
        print(f"处理 {basename} 时出错：{e}")

print("全部处理完成！")