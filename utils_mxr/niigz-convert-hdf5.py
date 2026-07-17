import os
import h5py
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


def nii_to_hdf5(nii_root: str, hdf5_root: str, target_inplane: int = 256):
    """
    将你「preprocessed_data」文件夹中的所有 .nii.gz 文件批量转换为 HDF5 格式
    
    已完全适配你最新的预处理流程（NiftyReg 配准 + 1x1x1 重采样 + [0,1] 非零归一化）：
    - 输入文件名格式：patient_id_T1_reg2TOF.nii.gz、patient_id_TOF_1x1x1.nii.gz 等
    - 统一所有病例的 in-plane 尺寸为 target_inplane x target_inplane（默认 256x256）
    - Z 轴（切片数）保持原始不变（max_len 仍按每个病人实际切片数保存）
    - 彻底解决 RuntimeError: stack expects each tensor to be equal size
    - 保留你训练/测试代码需要的 'data'、'max'、'max_len' 三个字段
    """
    if not os.path.exists(hdf5_root):
        os.makedirs(hdf5_root)

    # 1. 遍历所有病例文件夹
    patient_ids = [
        d for d in os.listdir(nii_root)
        if os.path.isdir(os.path.join(nii_root, d))
    ]

    # 2. 新文件名映射（完全适配你预处理后的输出文件名）
    modal_map = {
        'T1_reg2TOF': '3D-T1W',          # T1 配准后
        'T1ce_reg2TOF': '3D-T2W',        # T1ce 配准后
        'FLAIR_reg2TOF': '3D-FLAIR',     # FLAIR 配准后（注意是大写 FLAIR）
        'TOF_1x1x1': '3D-TOF-MRA',       # TOF 重采样后
    }

    print(f"共发现 {len(patient_ids)} 个病例，开始转换...")
    print(f"统一 in-plane 尺寸 → {target_inplane}×{target_inplane}（已解决不同病人 XY 尺寸不一致问题）\n")

    for patient_id in tqdm(patient_ids, desc="正在转换病例"):
        patient_nii_dir = os.path.join(nii_root, patient_id)
        patient_hdf5_dir = os.path.join(hdf5_root, patient_id)
        os.makedirs(patient_hdf5_dir, exist_ok=True)

        # 3. 找到该病例文件夹下所有 .nii.gz 文件
        nii_files = [
            f for f in os.listdir(patient_nii_dir)
            if f.endswith('.nii.gz') and f.startswith(f"{patient_id}_")
        ]

        for nii_filename in nii_files:
            # 解析 modal_key，例如 "1000454786_T1_reg2TOF.nii.gz" → "T1_reg2TOF"
            modal_key = nii_filename.replace(f"{patient_id}_", "").replace(".nii.gz", "")

            if modal_key not in modal_map:
                print(f"  ⚠️  跳过未知模态: {nii_filename}")
                continue

            hdf5_filename = modal_map[modal_key] + ".hdf5"
            nii_path = os.path.join(patient_nii_dir, nii_filename)
            hdf5_path = os.path.join(patient_hdf5_dir, hdf5_filename)

            try:
                # ================== 读取 + 统一 XY 尺寸 ==================
                img = sitk.ReadImage(nii_path)

                orig_size = img.GetSize()      # (X, Y, Z)
                orig_spacing = img.GetSpacing()

                # 新尺寸：只改 XY，Z 切片数保持不变
                new_size = (target_inplane, target_inplane, orig_size[2])
                new_spacing = (
                    orig_spacing[0] * orig_size[0] / target_inplane,
                    orig_spacing[1] * orig_size[1] / target_inplane,
                    orig_spacing[2]
                )

                resampler = sitk.ResampleImageFilter()
                resampler.SetSize(new_size)
                resampler.SetOutputSpacing(new_spacing)
                resampler.SetOutputOrigin(img.GetOrigin())
                resampler.SetOutputDirection(img.GetDirection())
                resampler.SetInterpolator(sitk.sitkLinear)
                resampler.SetDefaultPixelValue(0.0)
                resampled_img = resampler.Execute(img)

                arr = sitk.GetArrayFromImage(resampled_img).astype(np.float32)

                if arr.ndim != 3:
                    print(f"  ⚠️  {nii_filename} 不是 3D 图像，跳过 (shape={arr.shape})")
                    continue

                # 4. 计算训练代码需要的字段（即使你已经做了 [0,1] 归一化，这里依然保留 max）
                max_val = float(np.max(arr))
                max_len = int(arr.shape[0])

                # 5. 写入 HDF5
                with h5py.File(hdf5_path, 'w') as f:
                    f.create_dataset('data', data=arr, compression="gzip", compression_opts=4)
                    f.create_dataset('max', data=max_val)
                    f.create_dataset('max_len', data=max_len)

                print(f"  ✅  已转换 {patient_id} → {hdf5_filename}  "
                      f"(shape={arr.shape}, max={max_val:.4f})")

            except Exception as e:
                print(f"  ❌  转换失败 {nii_filename} : {e}")

    print("\n🎉 全部转换完成！")
    print(f"   HDF5 输出路径: {hdf5_root}")
    print(f"   统一尺寸: {target_inplane}×{target_inplane}（已解决 tensor stack 错误）")
    print("   现在可以直接运行你修改后的测试代码（选择 A 模式）了！")


if __name__ == "__main__":
    # ====================== 请在这里修改路径 ======================
    nii_root = r"E:\DCtools\Medical_Registration\preprocessed_data"   # ←←← 改成你预处理后的文件夹
    hdf5_root = r"E:\DCtools\Medical_Registration\hdf5_data"         # ←←← 改成你训练代码里使用的 hdf5_root
    # ============================================================

    nii_to_hdf5(nii_root, hdf5_root, target_inplane=256)
