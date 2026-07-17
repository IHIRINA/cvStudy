'''
import os
import torch
import numpy as np
import nibabel as nib
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from sys import path
path.insert(0, "/root/mxr/code/encoder_decoder")
from transforms import VAE_Transform

IMAGE_PATH = "/root/autodl-tmp/data_before/TOF/1-1001398096-TOF_pre.nii.gz"
MODEL_PATH = "/root/mxr/code/encoder_decoder/autoencoder_epoch273.pt"
SAVE_PATH = "/tmp/test_latent_self.nii.gz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ae = AutoencoderKlMaisi(
    spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4,
    num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2],
    norm_num_groups=32, norm_eps=1e-06,
    attention_levels=[False, False, False],
    with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
    use_checkpointing=False, use_convtranspose=False,
    norm_float16=True, num_splits=8, dim_split=1
).to(DEVICE)
ae.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
ae.eval()
print("✅ 模型加载完成")

vae_transform = VAE_Transform(
    is_train=False,
    random_aug=False,
    k=4,
    patch_size=[128, 128, 128],
    val_patch_size=None,
    output_dtype=torch.float32,
    spacing_type="original",
    image_keys=["image"],
    label_keys=[],
    additional_keys=[],
    select_channel=0,
)

data = {"image": IMAGE_PATH, "class": "mri"}
transformed, _ = vae_transform(data)
image = transformed["image"].unsqueeze(0).to(DEVICE)

print(f"预处理后图像 shape: {image.shape}, dtype: {image.dtype}")

# === 关键修复：对齐数据类型 ===
model_dtype = next(ae.parameters()).dtype
image = image.to(dtype=model_dtype)
image = image.half()
# 模型已经是 float16 吗？检查 ae 的参数 dtype，如果是 float32，则需转换：
ae = ae.half()
print(f"转换后 image dtype: {image.dtype}, 模型参数 dtype: {model_dtype}")

with torch.no_grad():
    z = ae.encode_stage_2_inputs(image)
    latent_np = z[0].cpu().numpy()

raw_img = nib.load(IMAGE_PATH)
affine = raw_img.affine
latent_save = np.transpose(latent_np, (1, 2, 3, 0))
nib.save(nib.Nifti1Image(latent_save.astype(np.float32), affine), SAVE_PATH)
print(f"✅ latent 保存至: {SAVE_PATH}")

latent_loaded_img = nib.load(SAVE_PATH)
latent_loaded = latent_loaded_img.get_fdata().astype(np.float32)
latent_loaded = np.transpose(latent_loaded, (3, 0, 1, 2))

max_diff = np.max(np.abs(latent_np - latent_loaded))
mean_diff = np.mean(np.abs(latent_np - latent_loaded))
print(f"最大绝对差异: {max_diff:.10f}")
print(f"平均绝对差异: {mean_diff:.10f}")

if max_diff < 1e-6:
    print("✅ 自洽性验证通过")
else:
    print("❌ 自洽性验证失败")
'''

"""
终极自洽验证：编码 -> 保存 -> 加载 -> 对比（同一进程）
"""
import os
import torch
import numpy as np
import nibabel as nib
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from sys import path
path.insert(0, "/root/mxr/code/encoder_decoder")
from transforms import VAE_Transform

device = torch.device("cuda")

# 路径
AE_CKPT = "/root/mxr/code/encoder_decoder/autoencoder_epoch273.pt"
TEST_IMAGE = "/root/autodl-tmp/data_before/TOF/1-1001398096-TOF_pre.nii.gz"
TEMP_LATENT = "/tmp/temp_latent_self.nii.gz"

# 加载模型
ae = AutoencoderKlMaisi(
    spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4,
    num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2],
    norm_num_groups=32, norm_eps=1e-06,
    attention_levels=[False, False, False],
    with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
    use_checkpointing=False, use_convtranspose=False,
    norm_float16=True, num_splits=8, dim_split=1
).to(device)
ae.load_state_dict(torch.load(AE_CKPT, map_location=device, weights_only=True))
ae.eval()
ae = ae.half()  # 统一 half
print("✅ 模型加载完成 (half)")

# 预处理
vae_transform = VAE_Transform(
    is_train=False,
    random_aug=False,
    k=4,
    patch_size=[128, 128, 128],
    val_patch_size=None,
    output_dtype=torch.float32,
    spacing_type="original",
    image_keys=["image"],
    select_channel=0,
)

# 1. 加载并预处理图像
data = {"image": TEST_IMAGE, "class": "mri"}
transformed, _ = vae_transform(data)
image = transformed["image"].unsqueeze(0).to(device).half()

# 2. 实时编码
with torch.no_grad():
    z = ae.encode_stage_2_inputs(image)
latent_realtime = z[0].cpu().numpy().astype(np.float32)  # (4,D,H,W)

# 3. 保存为 NIfTI (D,H,W,C)
latent_save = np.transpose(latent_realtime, (1,2,3,0))   # (D,H,W,4)
raw_img = nib.load(TEST_IMAGE)
affine = raw_img.affine
nib.save(nib.Nifti1Image(latent_save, affine), TEMP_LATENT)

# 4. 重新加载
latent_loaded_img = nib.load(TEMP_LATENT)
latent_loaded = latent_loaded_img.get_fdata().astype(np.float32)
latent_loaded = np.transpose(latent_loaded, (3,0,1,2))   # (4,D,H,W)

# 5. 对比
max_diff = np.max(np.abs(latent_realtime - latent_loaded))
mean_diff = np.mean(np.abs(latent_realtime - latent_loaded))
print(f"最大绝对差异: {max_diff:.10f}")
print(f"平均绝对差异: {mean_diff:.10f}")

if max_diff < 1e-6:
    print("✅ 自洽性验证通过 -> 流程完全正常")
else:
    print("❌ 自洽性验证失败 -> 保存/加载有问题")

# 6. 可选：计算该图像的 PSNR（与自身对比应为无穷大）
mse = np.mean((latent_realtime - latent_loaded) ** 2)
if mse == 0:
    print("PSNR: 无穷大 (完全一致)")
else:
    psnr = 10 * np.log10((latent_loaded.max() - latent_loaded.min())**2 / mse)
    print(f"PSNR: {psnr:.2f} dB")