import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_nifti(filepath):
    """加载NIfTI文件并返回numpy数组"""
    img = nib.load(filepath)
    data = img.get_fdata(dtype=np.float32)
    return data

def compute_psnr_ssim(img_true, img_test, data_range=None):
    """
    计算3D图像的PSNR和SSIM
    img_true, img_test: 3D numpy数组
    data_range: 用于PSNR和SSIM的动态范围，若为None则使用img_true.max() - img_true.min()
    """
    if img_true.shape != img_test.shape:
        raise ValueError(f"图像尺寸不匹配: {img_true.shape} vs {img_test.shape}")
    
    if data_range is None:
        data_range = img_true.max() - img_true.min()
        if data_range == 0:
            data_range = 1.0  # 避免除零
    
    psnr_val = psnr(img_true, img_test, data_range=data_range)
    ssim_val = ssim(img_true, img_test, data_range=data_range, multichannel=False)
    
    return psnr_val, ssim_val

def match_files_by_sort(dir1, dir2, pattern="*.nii.gz"):
    """
    默认匹配策略：按文件名排序后一一配对。
    要求两个文件夹中文件数量相等。
    """
    files1 = sorted(glob.glob(os.path.join(dir1, pattern)))
    files2 = sorted(glob.glob(os.path.join(dir2, pattern)))
    
    if len(files1) != len(files2):
        raise ValueError(f"文件夹文件数量不匹配: {dir1}有{len(files1)}个, {dir2}有{len(files2)}个")
    
    return list(zip(files1, files2))

def match_files_by_id(dir1, dir2, pattern="*.nii.gz", id_regex=r'(\d{10})'):
    """
    自定义匹配策略：从文件名中提取数字ID进行匹配（例如10位数字）。
    返回匹配上的文件对列表。
    """
    import re
    def extract_id(fpath):
        basename = os.path.basename(fpath)
        match = re.search(id_regex, basename)
        return match.group(1) if match else None
    
    files1 = {extract_id(f): f for f in glob.glob(os.path.join(dir1, pattern)) if extract_id(f)}
    files2 = {extract_id(f): f for f in glob.glob(os.path.join(dir2, pattern)) if extract_id(f)}
    
    common_ids = set(files1.keys()) & set(files2.keys())
    pairs = [(files1[pid], files2[pid]) for pid in common_ids]
    
    if not pairs:
        raise ValueError("未通过ID匹配到任何文件对，请检查正则表达式或改用排序匹配")
    return pairs

def main(tof_dir, recon_dir, output_excel="psnr_ssim_results.xlsx", match_func=None):
    """
    主函数：计算并输出配对的PSNR和SSIM，保存到Excel文件。
    
    参数:
        tof_dir: TOF图像文件夹路径
        recon_dir: 配准后T1图像文件夹路径
        output_excel: 输出Excel文件路径
        match_func: 文件配对函数，签名为 (dir1, dir2) -> list of (file1, file2)
                   若为None，则使用默认的排序配对
    """
    if match_func is None:
        match_func = match_files_by_sort
    
    pairs = match_func(tof_dir, recon_dir)
    print(f"找到 {len(pairs)} 对文件，开始计算...")
    
    results = []
    for tof_path, t1_path in pairs:
        tof_name = os.path.basename(tof_path)
        t1_name = os.path.basename(t1_path)
        print(f"处理: {tof_name} <-> {t1_name}")
        try:
            img_tof = load_nifti(tof_path)
            img_t1 = load_nifti(t1_path)
            psnr_val, ssim_val = compute_psnr_ssim(img_tof, img_t1)
            results.append({
                "TOF文件": tof_name,
                "tof_recon文件": t1_name,
                "PSNR (dB)": psnr_val,
                "SSIM": ssim_val
            })
            print(f"  PSNR: {psnr_val:.4f} dB, SSIM: {ssim_val:.4f}")
        except Exception as e:
            print(f"  计算失败: {e}")
            results.append({
                "TOF文件": tof_name,
                "tof_recon文件": t1_name,
                "PSNR (dB)": None,
                "SSIM": None,
                "错误信息": str(e)
            })
    
    # 转换为DataFrame并保存Excel
    if results:
        df = pd.DataFrame(results)
        # 计算平均值（跳过错误项）
        valid_df = df.dropna(subset=["PSNR (dB)", "SSIM"])
        if not valid_df.empty:
            avg_psnr = valid_df["PSNR (dB)"].mean()
            avg_ssim = valid_df["SSIM"].mean()
            # 添加汇总行
            summary = pd.DataFrame({
                "TOF文件": ["平均值"],
                "tof_recon文件": [""],
                "PSNR (dB)": [avg_psnr],
                "SSIM": [avg_ssim]
            })
            df = pd.concat([df, summary], ignore_index=True)
        
        df.to_excel(output_excel, index=False, engine="openpyxl")
        print(f"\n结果已保存至: {os.path.abspath(output_excel)}")
        
        if not valid_df.empty:
            print(f"平均 PSNR: {avg_psnr:.4f} dB")
            print(f"平均 SSIM: {avg_ssim:.4f}")
    else:
        print("没有成功计算的配对。")

if __name__ == "__main__":
    # ====== 请修改为实际的文件夹路径 ======
    tof_folder = "/root/autodl-tmp/data/TOF"               # TOF图像文件夹
    recon_folder = "/root/autodl-tmp/results/checkpoints_56*56*40_4ch/inference_results_recon"   # 配准后T1图像文件夹
    output_excel_path = "/root/autodl-tmp/results/checkpoints_56*56*40_4ch/psnr_ssim_results.xlsx"  # 输出Excel文件路径
    # ==================================
    
    # 选择匹配方式（二选一，注释掉不需要的）：
    # 1. 默认：按文件名排序一一对应（要求两个文件夹文件数量相同）
    #main(tof_folder, recon_folder, output_excel_path, match_func=match_files_by_sort)
    
    # 2. 按文件名中的10位数字ID进行匹配（示例，若ID不匹配会报错）
    from functools import partial
    main(tof_folder, recon_folder, output_excel_path, 
        match_func=lambda d1,d2: match_files_by_id(d1,d2, id_regex=r'(\d{10})'))