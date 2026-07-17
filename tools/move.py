import os
import shutil
from pathlib import Path

def reorganize_data(source_root: str, t1ce_output: str, tof_output: str):
    """
    将 source_root 下每个子目录中的 T1 和 TOF 文件分别复制到 t1ce_output 和 tof_output。
    
    参数:
        source_root: 原始数据根目录（例如 "./output_data"）
        t1ce_output:  存放 T1 文件的目录（自动创建）
        tof_output:   存放 TOF 文件的目录（自动创建）
    """
    source_path = Path(source_root).resolve()
    t1ce_path = Path(t1ce_output).resolve()
    tof_path = Path(tof_output).resolve()

    # 创建目标目录
    t1ce_path.mkdir(parents=True, exist_ok=True)
    tof_path.mkdir(parents=True, exist_ok=True)

    # 遍历 source_root 下的一级子目录（每个 ID 对应一个文件夹）
    for item in source_path.iterdir():
        if not item.is_dir():
            continue
        # 忽略 _tmp 等特殊目录（可按需调整）
        if item.name.startswith('_'):
            continue

        # 查找该 ID 目录下的 .nii.gz 文件
        nii_files = list(item.glob("*.nii.gz"))
        if not nii_files:
            print(f"⚠️ 跳过 {item.name}：未找到 .nii.gz 文件")
            continue

        for nii_file in nii_files:
            filename = nii_file.name
            # 根据文件名判断类型
            if "T1_pre_reg2TOF_aladin" in filename:
                dest = t1ce_path / filename
                shutil.copy2(nii_file, dest)  # copy2 保留元数据
                print(f"✅ 复制 T1: {nii_file} -> {dest}")
            elif "TOF_pre" in filename:
                dest = tof_path / filename
                shutil.copy2(nii_file, dest)
                print(f"✅ 复制 TOF: {nii_file} -> {dest}")
            else:
                print(f"⚠️ 未知类型文件: {filename}，跳过")

    print("\n🎉 重组完成！")
    print(f"T1 文件保存在: {t1ce_path}")
    print(f"TOF 文件保存在: {tof_path}")


if __name__ == "__main__":
    # 请根据实际路径修改以下三个变量
    SOURCE_DIR = "/root/autodl-tmp/output_data"      # 原始数据根目录（包含各个 ID 子文件夹）
    T1CE_DIR   = "/root/autodl-tmp/data/t1ce"             # 目标：存放所有 T1 文件的文件夹
    TOF_DIR    = "/root/autodl-tmp/data/tof"              # 目标：存放所有 TOF 文件的文件夹

    reorganize_data(SOURCE_DIR, T1CE_DIR, TOF_DIR)