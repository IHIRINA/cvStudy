import os
import SimpleITK as sitk


def get_subfolder_names(parent_folder):
    """
    获取指定文件夹内所有子文件夹的名称

    Args:
        parent_folder (str): 父文件夹的路径
    
    Returns:
        list: 包含所有子文件夹名称的列表
    """
    # 获取所有文件夹
    subfolders = [f.name for f in os.scandir(parent_folder) if f.is_dir()]
    
    return subfolders

def convert_dicom_to_nifti(dicom_folder, output_nii_path, target_modality="cs_FLAIR_3DView"): # 对于T1ce名称是cs T1w_3D +C, 对于TOF名称是TOF MRA 3 Min, T1对应的名称是T1w_3D, cs_T2W_3DView ，cs_FLAIR_3DView
    """
    从 DICOM 文件夹中找到模态为指定值的影像,并保存为 .nii.gz 文件.

    Args:
        dicom_folder (str): 包含 DICOM 文件的文件夹路径.
        output_nii_path (str): 输出的 NIfTI 文件路径 (.nii.gz).
        target_modality (str): 目标模态的名称 (默认 "TOF MRA 3 Min").
    
    Returns:
        bool: 如果成功找到并保存,返回 True;否则返回 False.
    """
    # 使用 SimpleITK 的 DICOM 文件读取器
    reader = sitk.ImageSeriesReader()

    # 获取文件夹内的所有 DICOM 文件系列
    series_ids = reader.GetGDCMSeriesIDs(dicom_folder)
    if not series_ids:
        print("未找到 DICOM 文件系列,请检查文件夹路径是否正确.")
        return False

    # print(f"在文件夹中找到 {len(series_ids)} 个 DICOM 系列.")

    # 遍历每个系列,找到模态为目标模态的系列
    for series_id in series_ids:
        series_file_names = reader.GetGDCMSeriesFileNames(dicom_folder, series_id)
        reader.SetFileNames(series_file_names)

        # 尝试从第一个文件中提取 Series Description(模态信息)
        try:
            # 加载第一个文件的元数据
            first_file = series_file_names[0]
            image = sitk.ReadImage(first_file)
            series_description = image.GetMetaData("0008|103e")  # DICOM 标签:Series Description

            # print(f"发现 DICOM 系列:{series_description}")

            if series_description.strip().lower() == target_modality.strip().lower():
                # print(f"找到目标模态:{series_description}")

                # 读取整个系列为 SimpleITK 图像
                image = reader.Execute()

                # 保存为 .nii.gz 文件
                sitk.WriteImage(image, output_nii_path)
                # print(f"成功将目标模态保存为 {output_nii_path}")
                return True

        except Exception as e:
            print(f"读取模态信息失败:{e}")

    print(f"未找到模态为 {target_modality} 的 DICOM 系列.")
    return False

if __name__ == '__main__':
    base_path = '/home/administrator/code/MMDM-Syn-main/data/niigz/Flair' # 数据结果的folder

    # 使用示例
    folder = "/home/administrator/code/MMDM-Syn-main/data/DICOM"  # 替换为你的 DICOM 文件夹路径 
    subfolders = get_subfolder_names(folder)
    for subfolder in subfolders:
        dicom_folder = os.path.join(folder,subfolder)
        output_path = os.path.join(base_path,f'{subfolder}.nii.gz')
        # dcm to nii
        convert_dicom_to_nifti(dicom_folder, output_path)
        print(f'{subfolder}处理完成')
