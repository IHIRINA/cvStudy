"""
encoder_final.py - 与 1.5.py 完全一致的数据加载方式（直接调用 VAE_Transform）
"""
import os
import gc
import torch
import numpy as np
import nibabel as nib
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from sys import path
path.insert(0, "/root/mxr/code/encoder_decoder")
from transforms import VAE_Transform

device = torch.device("cuda")
AE_CKPT = "/root/mxr/code/encoder_decoder/autoencoder_epoch273.pt"
INPUT_FOLDER = "/root/autodl-tmp/data_before/TOF"
OUTPUT_FOLDER = "/root/autodl-tmp/data/TOF_latent"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- 加载模型（与 1.5.py 完全一致） ----------
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
ae = ae.half()
print("✅ 模型加载完成 (half)")

# ---------- 预处理（与 1.5.py 完全一致） ----------
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

# ---------- 遍历文件（直接调用 transform，与 1.5.py 一样） ----------
files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.endswith(('.nii.gz', '.nii'))])
for idx, fname in enumerate(files):
    try:
        in_path = os.path.join(INPUT_FOLDER, fname)
        out_path = os.path.join(OUTPUT_FOLDER, fname)
        print(f"[{idx+1}/{len(files)}] {fname}")

        # 与 1.5.py 完全相同的数据加载方式
        data = {"image": in_path, "class": "mri"}
        transformed, _ = vae_transform(data)
        image = transformed["image"].unsqueeze(0).to(device).half()

        with torch.no_grad():
            z = ae.encode_stage_2_inputs(image)
        latent = z[0].cpu().numpy().astype(np.float32)  # (4,D,H,W)
        latent_save = np.transpose(latent, (1,2,3,0))   # (D,H,W,4)
        affine = nib.load(in_path).affine
        nib.save(nib.Nifti1Image(latent_save, affine), out_path)
        print(f"  ✅ 已保存 {out_path}")

        del image, z, latent, latent_save
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"  ❌ 错误: {e}")

print("🎉 全部完成")