import SimpleITK as sitk
import numpy as np
import os
import subprocess

def resample_to_isotropic(image, target_spacing=(1.0, 1.0, 1.0)):
    """
    使用 SimpleITK 将图像重采样为指定的各向同性间距 (例如 1x1x1)
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # 计算新的尺寸
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear) 
    resampler.SetDefaultPixelValue(0)
    
    return resampler.Execute(image)

def normalize_to_01_nonzero(image):
    """
    归一化到 [0, 1]，仅针对非零区域 + 强制背景为 0
    （解决灰背景 + 查看器全白问题）
    """
    img_array = sitk.GetArrayFromImage(image)
    mask = img_array > 0 
    
    if mask.any():  # 确保有非零像素
        min_val = img_array[mask].min()
        max_val = img_array[mask].max()
        
        # 避免除以 0 的情况
        if max_val - min_val > 1e-6:
            img_array[mask] = (img_array[mask] - min_val) / (max_val - min_val)
            
    # 强制所有 <=0 的 voxel 为 0（纯黑背景）
    img_array[img_array <= 0] = 0
    
    # 将 NumPy 数组转回 SimpleITK Image，并恢复元数据
    normalized_image = sitk.GetImageFromArray(img_array)
    normalized_image.CopyInformation(image)
    return normalized_image

def process_patient_data(patient_id, base_input_dir, base_output_dir):
    """
    处理单个患者的数据流
    """
    print(f"\n[{patient_id}] 开始处理...")
    
    reg_aladin_exe = r"E:\DCtools\NiftyReg-Windows-CUDA-v2.0.0\NiftyReg-Windows-CUDA-v2.0.0\reg_aladin.exe" 
    
    # 获取该患者专用的输入和输出文件夹路径
    patient_in_dir = os.path.join(base_input_dir, patient_id)
    patient_out_dir = os.path.join(base_output_dir, patient_id)
    os.makedirs(patient_out_dir, exist_ok=True)
    
    tof_path = os.path.join(patient_in_dir, f"{patient_id}_Tof.nii.gz")
    moving_modalities = {
        "T1": os.path.join(patient_in_dir, f"{patient_id}_T1.nii.gz"),
        "T1ce": os.path.join(patient_in_dir, f"{patient_id}_T1ce.nii.gz"),
        "FLAIR": os.path.join(patient_in_dir, f"{patient_id}_Flair.nii.gz")
    }
    
    if not os.path.exists(tof_path):
        print(f"[{patient_id}]  跳过：找不到参考图像 TOF ({tof_path})")
        return

    # 处理固定图像 (TOF): 读取 -> 重采样到 1x1x1
    print(f"[{patient_id}] 正在重采样 TOF...")
    tof_img = sitk.ReadImage(tof_path, sitk.sitkFloat32)
    tof_1x1x1 = resample_to_isotropic(tof_img, (1.0, 1.0, 1.0))
    
    # 保存重采样后的 TOF 作为后续配准的参考模板 (Reference)
    ref_tof_path = os.path.join(patient_out_dir, f"{patient_id}_TOF_1x1x1.nii.gz")
    sitk.WriteImage(tof_1x1x1, ref_tof_path)
    
    # 遍历其他模态进行配准和归一化
    for mod_name, moving_path in moving_modalities.items():
        if not os.path.exists(moving_path):
            print(f"[{patient_id}]  警告：找不到 {mod_name} 文件，跳过此模态。")
            continue
            
        print(f"[{patient_id}] 正在配准: {mod_name} -> TOF (调用 NiftyReg)...")
        output_reg_path = os.path.join(patient_out_dir, f"{patient_id}_{mod_name}_reg2TOF.nii.gz")
        
        try:
            # 核心配准命令: 使用 subprocess 调用外部 exe
            cmd = [
                reg_aladin_exe,
                "-ref", ref_tof_path,
                "-flo", moving_path,
                "-res", output_reg_path,
                "-affDirect",      # 直接执行仿射变换
                "-pad", "0"        # 新增：背景严格填充 0
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # 读取配准后的结果，进行 [0,1] 归一化
            print(f"[{patient_id}] 正在对 {mod_name} 进行 [0,1] 归一化...")
            reg_img = sitk.ReadImage(output_reg_path, sitk.sitkFloat32)
            norm_img = normalize_to_01_nonzero(reg_img)
            
            # 覆盖保存为归一化后的最终结果
            sitk.WriteImage(norm_img, output_reg_path)
            print(f"[{patient_id}]  {mod_name} 处理完成.")
            
        except subprocess.CalledProcessError as e:
            print(f"[{patient_id}]  {mod_name} NiftyReg 配准失败!\n错误信息: {e.stderr}")
        except Exception as e:
            print(f"[{patient_id}]  {mod_name} 后续处理出错: {e}")

    # 对重采样后的 TOF 本身归一化
    print(f"[{patient_id}] 正在对 TOF 进行 [0,1] 归一化...")
    norm_tof = normalize_to_01_nonzero(tof_1x1x1)
    sitk.WriteImage(norm_tof, ref_tof_path)
    

if __name__ == "__main__":
    input_folder = r"E:\DCtools\Medical_Registration\data_raw"
    output_folder = r"E:\DCtools\Medical_Registration\preprocessed_data"
    
    if not os.path.exists(input_folder):
        print(f"错误: 找不到输入文件夹 '{input_folder}'，请检查路径。")
    else:
        # 自动扫描 input_folder 下的所有子文件夹作为 patient_id
        patient_list = [
            folder_name for folder_name in os.listdir(input_folder) 
            if os.path.isdir(os.path.join(input_folder, folder_name))
        ]
        
        patient_list.sort() # 排序，让输出日志更整洁
        
        print(f"成功扫描数据目录，共发现 {len(patient_list)} 个患者样本。")
        print("开始批量预处理...\n" + "="*50)
        
        for pid in patient_list:
            process_patient_data(pid, input_folder, output_folder)
            
        print("="*50 + "\n恭喜！全部数据预处理完成！")
