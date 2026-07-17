import argparse
import os
import numpy as np
import torch
import nibabel as nib
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.bundle import ConfigParser
from monai.inferers import SlidingWindowInferer

def define_instance(config_dict: dict, instance_def_key: str):
    args = argparse.Namespace()
    for k, v in config_dict.items():
        setattr(args, k, v)
    parser = ConfigParser(vars(args))
    parser.parse(True)
    return parser.get_parsed_content(instance_def_key, instantiate=True)

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型配置（保持不变）
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

# 关键修改：定义输入文件夹路径（批量读取该目录下所有nii.gz文件）
INPUT_DIR = "/root/autodl-tmp/results/checkpoints_56*56*40_4ch/inference_results"
# 输出目录（保持不变）
TARGET_OUTPUT_DIR = "/root/autodl-tmp/results/checkpoints_56*56*40_4ch/inference_results_recon"
# 确保输出目录存在
os.makedirs(TARGET_OUTPUT_DIR, exist_ok=True)

# 初始化AutoencoderKL模型并加载预训练权重（只需初始化一次）
autoencoder = define_instance(CONFIG, "autoencoder_def").to(device)
checkpoint = torch.load(CONFIG["trained_autoencoder_path"], weights_only=True)
autoencoder.load_state_dict(checkpoint)
autoencoder.eval()

# 定义SlidingWindowInferer用于解码（只需定义一次）
decode_inferer = SlidingWindowInferer(
    roi_size=[56, 56, 40],
    sw_batch_size=1,
    overlap=0.5,
    progress=True,
    sw_device=device,
    mode="gaussian",
    device=torch.device("cpu")
)

# 解码封装类（只需定义一次）
class DecodeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z):
        return self.model.decode(z)

# 批量处理逻辑
def process_single_file(latent_path):
    """处理单个nii.gz文件的解码逻辑"""
    # 加载潜在空间数据
    latent_img = nib.load(latent_path)
    latent_data = latent_img.get_fdata().astype(np.float32)
    source_affine = latent_img.affine
    latent_affine = source_affine.copy()
    original_affine = latent_affine.copy()
    original_affine[:3, :3] /= 4.0      # 逆操作：除以4
    original_affine[:3, 3] /= 4.0

    # 调整维度顺序以匹配模型期望输入
    latent_tensor = torch.from_numpy(latent_data).permute(3, 0, 1, 2).unsqueeze(0).to(device)

    # 解码过程
    with torch.no_grad(), torch.amp.autocast('cuda'):
        decode_wrapper = DecodeWrapper(autoencoder)
        reconstruction = decode_inferer(network=decode_wrapper, inputs=latent_tensor)

    # 转换为numpy数组
    reconstruction_np = reconstruction[0, 0].cpu().numpy().astype(np.float32)

    # 构建输出文件名和路径
    source_filename = os.path.basename(latent_path)
    output_filename = source_filename
    output_path = os.path.join(TARGET_OUTPUT_DIR, output_filename)

    # 保存重建后的图像
    nib.save(nib.Nifti1Image(reconstruction_np, original_affine), output_path)
    print(f"✅ 已处理：{latent_path} -> 保存至：{output_path}")

# 遍历输入文件夹下所有.nii.gz文件
if __name__ == "__main__":
    # 获取输入目录下所有nii.gz文件
    nii_gz_files = [
        os.path.join(INPUT_DIR, filename)
        for filename in os.listdir(INPUT_DIR)
        if filename.endswith(".nii.gz")
    ]

    if not nii_gz_files:
        print(f"⚠️ 未在目录 {INPUT_DIR} 中找到任何.nii.gz文件")
    else:
        print(f"📁 共找到 {len(nii_gz_files)} 个.nii.gz文件，开始批量解码...")
        # 逐个处理文件
        for file_path in nii_gz_files:
            try:
                process_single_file(file_path)
            except Exception as e:
                print(f"❌ 处理文件 {file_path} 时出错：{str(e)}")
        print("🎉 所有文件处理完成！")