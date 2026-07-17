import argparse
import gc
import os
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # print("走的1")
        return model(images)
    else:
        # print("走的2")
        return inferer(network=model, inputs=images) # 用的这个



import math  # needed for dynamic_infer


# 文件夹路径,输入文件夹路径
folder_path = "/root/autodl-tmp/data/TOF"

# 初始化 data_dict
data_dict = []

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith(".nii.gz"):
        # 构建完整的文件路径
        file_path = os.path.join(folder_path, filename)
        # 添加到 data_dict 中，假设所有文件的类别都是 'ct'
        data_dict.append({'image': file_path, 'class': 'mri'})

val_transform = VAE_Transform(
    is_train=False,
    random_aug=False,
    k=4,  # patches should be divisible by k
    val_patch_size=None,  # None. if is none, will validate on whole image volume
    output_dtype=torch.float16,  # final data type
    # spacing_type="fixed",
    # spacing=[1.5, 1.5, 2.0],
    image_keys=["image"],
    label_keys=[],
    additional_keys=[],
    select_channel=0,
)
datset_val = CacheDataset(data=data_dict, transform=val_transform, cache_rate=1, num_workers=0)
dataloader_val = DataLoader(datset_val, batch_size=1, shuffle=False, num_workers=4)

autoencoder = define_instance(CONFIG, "autoencoder_def").to(device)
checkpoint = torch.load(CONFIG["trained_autoencoder_path"], weights_only=True)
autoencoder.load_state_dict(checkpoint)
autoencoder.eval()
print("✅ Autoencoder loaded.")

class DecodeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z):
        return self.model.decode(z)



for batch in dataloader_val:  # 在这一步才进行图形读取、变换等操作。在哪里调用的__getitem__操作
    if isinstance(batch[1], tuple) and len(batch[1]) > 0:
        file_path = batch[1][0]  # 取出路径字符串
        filename = file_path.split('/')[-1]  # 提取文件名
        desired_name = filename.replace('zero_padded_nii_', '')  # 去掉前缀
    else:
        print("Error: batch[1] is not a tuple or is empty!")

     # 输出文件夹路径
    exist_dir = '/root/autodl-tmp/data/TOF_latent'

    os.makedirs(exist_dir, exist_ok=True)
    output_path = os.path.join(exist_dir, desired_name)
    # print("output_path", output_path)


    image = batch[0]["image"].to(device).contiguous()
    print("Transformed image shape:", batch[0]["image"].shape)
    set_determinism(seed=0)

    val_inferer = (
        SlidingWindowInferer(
            roi_size=[128, 128, 64],
            sw_batch_size=1,
            overlap=0.35,  # 25% overlap 减少边界效应
            # mode="gaussian",
            progress=False,
            sw_device=device,
            device=torch.device("cpu")  # 输出放 CPU 节省显存
        )
        if [96, 96, 64]
        else SimpleInferer()
    )
    # 用于 encode（输入图像 → latent）
    encode_inferer = SlidingWindowInferer(
        roi_size=[128, 128, 64],
        sw_batch_size=1,
        overlap=0.25,
        mode="gaussian",
        progress=False,
        sw_device=device,
        device=torch.device("cpu"),
    )
    decode_inferer = SlidingWindowInferer(
        roi_size=[64, 64, 16],  # 根据你的 latent 尺寸调整
        sw_batch_size=1,
        overlap=0.5,  # 适当增加重叠度以减少边界效应
        progress=True,
        sw_device=device,
        mode="gaussian",
        device=torch.device("cpu")  # 输出放 CPU 节省显存
    )
    # output_path1 = os.path.join('/public/home/xurui/xushijin/data/inference_data', 'reconstruction1.nii.gz')
    # output_path2 = os.path.join('/public/home/xurui/xushijin/data/inference_data', 'reconstruction2.nii.gz')
    with torch.no_grad(), torch.amp.autocast('cuda'):
        # # 提供的固定仿射矩阵
        # affine_matrix = np.array([
        #     [1.03515625, 0., 0., 265.],
        #     [0., 1.03515625, 0., 172.],
        #     [0., 0., 5., -230.5],
        #     [0., 0., 0., 1.]
        # ])
        original_affine = image.meta["affine"].squeeze().cpu().numpy()  # (4,4)

        # 下面使用AutoencoderKL中的forward函数进行图像重建
        # reconstruction, _, _ = dynamic_infer(val_inferer, autoencoder, image) # reconstruction是重建后的ct图像
        # recon1_np = reconstruction.squeeze().cpu().numpy()
        # nib.save(nib.Nifti1Image(recon1_np, original_affine), output_path)
        # print("aotuencoderkl_maisi中forward返回的重建图片已经保存！")

        # z = encode_inferer(network=autoencoder.encode_stage_2_inputs, inputs=image)
        # print("压缩后shape:", z.shape)  # torch.Size([1, 4, 128, 128, 16])
        # latent = z.squeeze(0).permute(1, 2, 3, 0).cpu().numpy().astype(np.float32)
        # print(latent.shape)


        # encode过程，内存爆炸
        z = autoencoder.encode_stage_2_inputs(image)
        print("压缩后shape:", z.shape)  # torch.Size([1, 4, 128, 128, 16])
        latent = z.squeeze(0).permute(1, 2, 3, 0).cpu().numpy().astype(np.float32)
        print(latent.shape)

        # # decode过程
        # # reconstruction2 = autoencoder.decode(z_mu) # 直接decode会cuda out of memory
        # decode_wrapper = DecodeWrapper(autoencoder)
        # reconstruction2 = decode_inferer(network=decode_wrapper, inputs=z)
        # # reconstruction2 = autoencoder.decode_stage_2_outputs(z) # 同样会out of memory
        # print("aotuencoderkl_maisi中，encode后再decode得到的图像")
        # reconstruction2_np = reconstruction2[0, 0].cpu().numpy().astype(np.float32)
        # print("此图像的shape:", reconstruction2_np.shape)  # torch.Size([512, 512, 64])
        # # reconstruction2 = decode_inferer(z_mu, autoencoder.decode)



        #original_affine = image.meta["affine"].cpu().numpy()  # 尽量使用这个？
        nib.save(nib.Nifti1Image(latent, original_affine), output_path)
        print(f"✅ Reconstructed image saved to: {output_path}/")
        del z, latent, image
        torch.cuda.empty_cache()
        gc.collect()
