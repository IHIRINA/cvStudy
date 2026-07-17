import argparse
import gc
import os
import math  # 提前导入，避免动态导入导致的问题
from datetime import datetime
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import LoadImage, SaveImage
from monai.utils import set_determinism
from monai.bundle import ConfigParser
from monai.inferers import SlidingWindowInferer, SimpleInferer
from transforms import VAE_Transform
from monai.data import CacheDataset, DataLoader
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
import nibabel as nib

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 确保CUDA内存分配策略合理
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False

# 配置字典
CONFIG = {
    "trained_autoencoder_path": "/root/mxr/code/encoder_decoder/autoencoder_epoch273.pt",
    "spatial_dims": 3,
    "image_channels": 1,
    "latent_channels": 4,
    "autoencoder_def": {
        "_target_": "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi",
        "spatial_dims": "@spatial_dims",
        "in_channels": "@image_channels",
        "out_channels": "@image_channels",
        "latent_channels": "@latent_channels",
        "num_channels": [64, 128, 256],
        "num_res_blocks": [2, 2, 2],
        "norm_num_groups": 32,
        "norm_eps": 1e-06,
        "attention_levels": [False, False, False],
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
        "use_checkpointing": False,
        "use_convtranspose": False,
        "norm_float16": True,
        "num_splits": 8,
        "dim_split": 1
    }
}


def define_instance(config_dict: dict, instance_def_key: str):
    args = argparse.Namespace()
    for k, v in config_dict.items():
        setattr(args, k, v)
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)


def dynamic_infer(inferer, model, images):
    """Fallback to sliding window if image is too large."""
    if torch.numel(images[0:1, 0:1, ...]) < math.prod(inferer.roi_size):
        return model(images)
    else:
        return inferer(network=model, inputs=images)


# 定义解码包装器，解决直接decode显存溢出问题
class DecodeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z):
        return self.model.decode(z)


# ======================== 核心路径配置 ========================
# 输入文件夹路径（需要重建的CT文件）
input_folder = "/root/autodl-tmp/data/TOF"
# 输出文件夹路径（保存重建后的CT文件）
output_folder = "/root/autodl-tmp/data/TOF_recon"
# ==================================================================

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 初始化 data_dict 并记录文件路径和名称的映射
data_dict = []
file_path_mapping = {}  # 用于存储索引到文件名的映射
print(f"正在扫描输入文件夹: {input_folder}")

# 遍历输入文件夹中的所有NIfTI文件
for idx, filename in enumerate(sorted(os.listdir(input_folder))):
    if filename.endswith(".nii.gz") or filename.endswith(".nii"):
        file_path = os.path.join(input_folder, filename)
        data_dict.append({'image': file_path, 'class': 'mri'})
        file_path_mapping[idx] = filename  # 记录索引和文件名的对应关系
        print(f"✅ 找到文件: {filename}")

# 检查是否找到文件
if not data_dict:
    print(f"❌ 在 {input_folder} 中未找到任何.nii或.nii.gz文件！")
    exit(1)
print(f"\n总计找到 {len(data_dict)} 个待处理文件\n")

# 数据变换
val_transform = VAE_Transform(
    is_train=False,
    random_aug=False,
    k=4,  # patches should be divisible by k
    val_patch_size=None,  # None. if is none, will validate on whole image volume
    output_dtype=torch.float64,  # final data type
    image_keys=["image"],
    label_keys=[],
    additional_keys=[],
    select_channel=0,
)

# 构建数据集和数据加载器
datset_val = CacheDataset(data=data_dict, transform=val_transform, cache_rate=1, num_workers=0)
dataloader_val = DataLoader(datset_val, batch_size=1, shuffle=False, num_workers=4)

# 加载模型
autoencoder = define_instance(CONFIG, "autoencoder_def").to(device)
checkpoint = torch.load(CONFIG["trained_autoencoder_path"], weights_only=True)
autoencoder.load_state_dict(checkpoint)
autoencoder.eval()
print("✅ Autoencoder模型加载完成\n")

# 设置确定性
set_determinism(seed=0)

# 定义推理器
val_inferer = SlidingWindowInferer(
    roi_size=[224, 224, 160],
    sw_batch_size=1,
    overlap=0.35,
    progress=False,
    sw_device=device,
    device=torch.device("cpu")  # 输出放 CPU 节省显存
)

encode_inferer = SlidingWindowInferer(
    roi_size=[128, 128, 64],
    sw_batch_size=1,
    overlap=0.25,
    mode="gaussian",
    progress=False,
    sw_device=device,
    device=device,  # 编码结果先放GPU，避免频繁数据传输
)
'''
decode_inferer = SlidingWindowInferer(
    roi_size=[64, 64, 16],
    sw_batch_size=1,
    overlap=0.5,
    progress=True,
    sw_device=device,
    mode="gaussian",
    device=torch.device("cpu")
)
'''
# 开始推理和重建
total_files = len(dataloader_val)
for idx, batch in enumerate(dataloader_val, 0):  # 从0开始索引，匹配file_path_mapping
    try:
        # ======================== 修复文件名提取逻辑 ========================
        # 直接通过索引获取原始文件名（最可靠的方式）
        if idx in file_path_mapping:
            filename = file_path_mapping[idx]
        else:
            filename = f"unknown_file_{idx + 1}.nii.gz"
            print(f"⚠️  无法通过索引获取文件名，使用默认名称: {filename}")
        # ==================================================================

        # 构建输出路径（保持原文件名）
        output_path = os.path.join(output_folder, filename)

        print(f"[{idx + 1}/{total_files}] 正在处理: {filename}")

        # 处理图像数据
        batch_data = batch[0]
        image = batch[0]["image"].to(device).contiguous()
        original_affine = image.meta["affine"].squeeze().cpu().numpy()
        #=========================================
        raw_img = nib.load(os.path.join(input_folder, filename))
        raw_affine = raw_img.affine
        #==========================================
        model_dtype = next(autoencoder.parameters()).dtype
        image = image.to(dtype=model_dtype)  # 转换输入类型与模型一致


        # 在文件保存前添加逆变换（约第178行附近）
        with torch.no_grad(), torch.amp.autocast('cuda'):
            reconstruction, _, _ = dynamic_infer(val_inferer, autoencoder, image)
            '''
                # 将重建结果从[0,1]逆变换回原始HU范围[-1000, 1000]
                # 逆变换公式：HU = (normalized - b_min) / (b_max - b_min) * (a_max - a_min) + a_min
                # 即：HU = normalized * 2000 - 1000
                reconstruction_hu = reconstruction * 2000.0 - 1000.0

                # 可选：截断到合理范围（与输入clip=True保持一致）
                reconstruction_hu = torch.clamp(reconstruction_hu, min=-1000.0, max=1000.0)
            '''
            recon_np = reconstruction.squeeze().cpu().numpy().astype(np.float32)
            recon_np = recon_np[::-1, ::-1, :]

            # 获取保存的分位值
            orig_data = batch[0]["image"]          # 原始输入（经过transform前或后均可）
            
            if hasattr(orig_data, 'percentile_lower') and hasattr(orig_data, 'percentile_upper'):
                lower = orig_data.percentile_lower
                upper = orig_data.percentile_upper
            elif hasattr(orig_data, 'meta') and 'percentile_range' in orig_data.meta:
                lower, upper = orig_data.meta['percentile_range']

            lower = float(lower)
            upper = float(upper)

            
            # 精确恢复原始灰度范围
            if upper > lower:
                recon_np = recon_np * (upper - lower) + lower
            else:
                print(f"⚠️  分位值异常: lower={lower}, upper={upper}")

            # 可选：clip 到合理范围
            # recon_np = np.clip(recon_np, lower - 100, upper + 100)

            # 保存重建图像
            nib.save(nib.Nifti1Image(recon_np, raw_affine), output_path)
            #print(f"✅ 重建完成，HU值范围已恢复至[-1000, 1000]，保存至: {output_path}")
            print(f"✅ 重建完成，已使用分位值恢复原始范围: [{lower:.2f}, {upper:.2f}]")

        # 强制释放内存
        del image, reconstruction, recon_np, original_affine
        torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"❌ 处理文件 {filename} 时出错: {str(e)}")
        # 出错时也清理内存
        torch.cuda.empty_cache()
        gc.collect()
        continue

print(f"\n🎉 所有图像重建完成！")
print(f"📁 重建结果保存在: {output_folder}")