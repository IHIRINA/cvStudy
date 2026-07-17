import warnings
from typing import List, Optional

import torch

from monai.transforms import IntensityStatsd   # 新增这一行

from monai.transforms import (
    Compose,
    DivisiblePadd,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    LoadImaged,
    Orientationd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandFlipd,
    RandGibbsNoised,
    RandHistogramShiftd,
    RandRotate90d,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    RandZoomd,
    ResizeWithPadOrCropd,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    SelectItemsd,
    Spacingd,
    SpatialPadd,
)

SUPPORT_MODALITIES = ["ct", "mri"]

'''
def define_fixed_intensity_transform(modality: str, image_keys: List[str] = ["image"]) -> List:
    """
    Define fixed intensity transform based on the modality.

    Args:
        modality (str): The imaging modality, either 'ct' or 'mri'.
        image_keys (List[str], optional): List of image keys. Defaults to ["image"].

    Returns:
        List: A list of intensity transforms.
    """
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. Will not do any intensity transform and will use original intensities."
        )

    modality = modality.lower()  # Normalize modality to lowercase

    intensity_transforms = {
        "mri": [
            ScaleIntensityRangePercentilesd(keys=image_keys, lower=0.0, upper=99.5, b_min=0.0, b_max=1, clip=False, meta_key_postfix="intensity_range")
        ],
        "ct": [ScaleIntensityRanged(keys=image_keys, a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True, meta_key_postfix="intensity_range")],
    }

    if modality not in intensity_transforms:
        return []

    #return intensity_transforms[modality]
    return intensity_transforms.get(modality, [])
'''

import numpy as np

def _save_percentiles(x):
    """手动计算并保存 0% 和 99.5% 分位值"""
    if isinstance(x, torch.Tensor):
        arr = x.cpu().numpy()
    else:
        arr = np.asarray(x)
    
    lower = float(np.percentile(arr, 0))
    upper = float(np.percentile(arr, 99.5))
    
    # 保存到 meta（如果支持 MetaTensor）
    if hasattr(x, 'meta'):
        x.meta['percentile_range'] = (lower, upper)
    else:
        # 直接挂载属性（最保险）
        x.percentile_lower = lower
        x.percentile_upper = upper
    
    return x


def define_fixed_intensity_transform(modality: str, image_keys: List[str] = ["image"]) -> List:
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}."
        )
        return []

    modality = modality.lower()

    if modality == "mri":
        return [
            # 新增：手动计算并保存分位值（推荐放在 dict 里）
            Lambdad(
                keys=image_keys,
                func=_save_percentiles
            ),
            ScaleIntensityRangePercentilesd(
                keys=image_keys, 
                lower=0.0, 
                upper=99.5, 
                b_min=0.0, 
                b_max=1.0, 
                clip=False
            )
        ]
    elif modality == "ct":
        return [
            ScaleIntensityRanged(
                keys=image_keys, 
                a_min=-1000, 
                a_max=1000, 
                b_min=0.0, 
                b_max=1.0, 
                clip=True
            )
        ]
    return []


def define_random_intensity_transform(modality: str, image_keys: List[str] = ["image"]) -> List:
    """
    Define random intensity transform based on the modality.

    Args:
        modality (str): The imaging modality, either 'ct' or 'mri'.
        image_keys (List[str], optional): List of image keys. Defaults to ["image"].

    Returns:
        List: A list of random intensity transforms.
    """
    modality = modality.lower()  # Normalize modality to lowercase
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. Will not do any intensity transform and will use original intensities."
        )

    if modality == "ct":
        return []  # CT HU intensity is stable across different datasets
    elif modality == "mri":
        return [
            RandBiasFieldd(keys=image_keys, prob=0.3, coeff_range=(0.0, 0.3)),
            RandGibbsNoised(keys=image_keys, prob=0.3, alpha=(0.5, 1.0)),
            RandAdjustContrastd(keys=image_keys, prob=0.3, gamma=(0.5, 2.0)),
            RandHistogramShiftd(keys=image_keys, prob=0.05, num_control_points=10),
        ]
    else:
        return []


def define_vae_transform(
    is_train: bool,
    modality: str,
    random_aug: bool,
    k: int = 4,
    patch_size: List[int] = [128, 128, 128],
    val_patch_size: Optional[List[int]] = None,
    output_dtype: torch.dtype = torch.float32,
    spacing_type: str = "original",
    spacing: Optional[List[float]] = None,
    image_keys: List[str] = ["image"],
    label_keys: List[str] = [],
    additional_keys: List[str] = [],
    select_channel: int = 0,
) -> tuple:
    """
    Define the MAISI VAE transform pipeline for training or validation.

    Args:
        is_train (bool): Whether it's for training or not. If True, the output transform will consider random_aug, the cropping will use "patch_size" for random cropping. If False, the output transform will alwasy treat "random_aug" as False, will use "val_patch_size" for central cropping.
        modality (str): The imaging modality, either 'ct' or 'mri'.
        random_aug (bool): Whether to apply random data augmentation.
        k (int, optional): Patches should be divisible by k. Defaults to 4.
        patch_size (List[int], optional): Size of the patches. Defaults to [128, 128, 128]. Will random crop patch for training.
        val_patch_size (Optional[List[int]], optional): Size of validation patches. Defaults to None. If None, will use the whole volume for validation. If given, will central crop a patch for validation.
        output_dtype (torch.dtype, optional): Output data type. Defaults to torch.float32.
        spacing_type (str, optional): Type of spacing. Defaults to "original". Choose from ["original", "fixed", "rand_zoom"].
        spacing (Optional[List[float]], optional): Spacing values. Defaults to None.
        image_keys (List[str], optional): List of image keys. Defaults to ["image"].
        label_keys (List[str], optional): List of label keys. Defaults to [].
        additional_keys (List[str], optional): List of additional keys. Defaults to [].
        select_channel (int, optional): Channel to select for multi-channel MRI. Defaults to 0.

    Returns:
        tuple: A tuple containing Composed Transform train_transforms or val_transforms depending on 'is_train'.
    """
    modality = modality.lower()  # Normalize modality to lowercase
    if modality not in SUPPORT_MODALITIES:
        warnings.warn(
            f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. Will not do any intensity transform and will use original intensities."
        )

    if spacing_type not in ["original", "fixed", "rand_zoom"]:
        raise ValueError(f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}.")

    keys = image_keys + label_keys + additional_keys
    interp_mode = ["bilinear"] * len(image_keys) + ["nearest"] * len(label_keys)

    common_transform = [  # !!!!
        SelectItemsd(keys=keys, allow_missing_keys=True),
        LoadImaged(keys=keys, allow_missing_keys=True),  # here is the LoadImaged
        EnsureChannelFirstd(keys=keys, allow_missing_keys=True),
        Orientationd(keys=keys, axcodes="RAS", allow_missing_keys=True),
    ]

    if modality == "mri":
        common_transform.append(Lambdad(keys=image_keys, func=lambda x: x[select_channel : select_channel + 1, ...]))

    common_transform.extend(define_fixed_intensity_transform(modality, image_keys=image_keys))

    if spacing_type == "fixed":
        common_transform.append(
            Spacingd(keys=image_keys + label_keys, allow_missing_keys=True, pixdim=spacing, mode=interp_mode)
        )

    random_transform = []
    if is_train and random_aug:
        random_transform.extend(define_random_intensity_transform(modality, image_keys=image_keys))
        random_transform.extend(
            [RandFlipd(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axis=axis) for axis in range(3)]
            + [
                RandRotate90d(keys=keys, allow_missing_keys=True, prob=0.5, spatial_axes=axes)
                for axes in [(0, 1), (1, 2), (0, 2)]
            ]
            + [
                RandScaleIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, factors=(0.9, 1.1)),
                RandShiftIntensityd(keys=image_keys, allow_missing_keys=True, prob=0.3, offsets=0.05),
            ]
        )

        if spacing_type == "rand_zoom":
            random_transform.extend(
                [
                    RandZoomd(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        min_zoom=0.5,
                        max_zoom=1.5,
                        keep_size=False,
                        mode=interp_mode,
                    ),
                    RandRotated(
                        keys=image_keys + label_keys,
                        allow_missing_keys=True,
                        prob=0.3,
                        range_x=0.1,
                        range_y=0.1,
                        range_z=0.1,
                        keep_size=True,
                        mode=interp_mode,
                    ),
                ]
            )

    if is_train:
        train_crop = [
            SpatialPadd(keys=keys, spatial_size=patch_size, allow_missing_keys=True),
            RandSpatialCropd(
                keys=keys, roi_size=patch_size, allow_missing_keys=True, random_size=False, random_center=True
            ),
        ]
    else:
        val_crop = (  # here change the size of output ct shape? no...
            [DivisiblePadd(keys=keys, allow_missing_keys=True, k=k)]
            if val_patch_size is None
            else [ResizeWithPadOrCropd(keys=keys, allow_missing_keys=True, spatial_size=val_patch_size)]
        )

    final_transform = [EnsureTyped(keys=keys, dtype=output_dtype, allow_missing_keys=True)]

    if is_train:
        train_transforms = Compose(
            common_transform + random_transform + train_crop + final_transform
            if random_aug
            else common_transform + train_crop + final_transform
        )
        return train_transforms
    else:
        val_transforms = Compose(common_transform + val_crop + final_transform)
        return val_transforms


class VAE_Transform:
    """
    A class to handle MAISI VAE transformations for different modalities.
    """

    def __init__(
        self,
        is_train: bool,
        random_aug: bool,
        k: int = 4,
        patch_size: List[int] = [128, 128, 128],
        val_patch_size: Optional[List[int]] = None,
        output_dtype: torch.dtype = torch.float32,
        spacing_type: str = "original",
        spacing: Optional[List[float]] = None,
        image_keys: List[str] = ["image"],
        label_keys: List[str] = [],
        additional_keys: List[str] = [],
        select_channel: int = 0,
    ):
        """
        Initialize the VAE_Transform.

        Args:
            is_train (bool): Whether it's for training or not. If True, the output transform will consider random_aug, the cropping will use "patch_size" for random cropping. If False, the output transform will alwasy treat "random_aug" as False, will use "val_patch_size" for central cropping.
            random_aug (bool): Whether to apply random data augmentation for training.
            k (int, optional): Patches should be divisible by k. Defaults to 4.
            patch_size (List[int], optional): Size of the patches. Defaults to [128, 128, 128]. Will random crop patch for training.
            val_patch_size (Optional[List[int]], optional): Size of validation patches. Defaults to None. If None, will use the whole volume for validation. If given, will central crop a patch for validation.
            output_dtype (torch.dtype, optional): Output data type. Defaults to torch.float32.
            spacing_type (str, optional): Type of spacing. Defaults to "original". Choose from ["original", "fixed", "rand_zoom"].
            spacing (Optional[List[float]], optional): Spacing values. Defaults to None.
            image_keys (List[str], optional): List of image keys. Defaults to ["image"].
            label_keys (List[str], optional): List of label keys. Defaults to [].
            additional_keys (List[str], optional): List of additional keys. Defaults to [].
            select_channel (int, optional): Channel to select for multi-channel MRI. Defaults to 0.
        """
        if spacing_type not in ["original", "fixed", "rand_zoom"]:
            raise ValueError(
                f"spacing_type has to be chosen from ['original', 'fixed', 'rand_zoom']. Got {spacing_type}."
            )

        self.is_train = is_train
        self.transform_dict = {}

        for modality in ["ct", "mri"]:
            self.transform_dict[modality] = define_vae_transform(
                is_train=is_train,
                modality=modality,
                random_aug=random_aug,
                k=k,
                patch_size=patch_size,
                val_patch_size=val_patch_size,
                output_dtype=output_dtype,
                spacing_type=spacing_type,
                spacing=spacing,
                image_keys=image_keys,
                label_keys=label_keys,
                additional_keys=additional_keys,
                select_channel=select_channel,
            )

    def __call__(self, img: dict, fixed_modality: Optional[str] = None) -> dict:
        """
        Apply the appropriate transform to the input image.

        Args:
            img (dict): Input image dictionary.
            fixed_modality (Optional[str], optional): Fixed modality to use. Defaults to None.

        Returns:
            Composed Transform

        Raises:
            ValueError: If the modality is not 'ct' or 'mri'.
        """
        modality = fixed_modality or img["class"]
        modality = modality.lower()  # Normalize modality to lowercase
        if modality not in ["ct", "mri"]:
            warnings.warn(
                f"Intensity transform only support {SUPPORT_MODALITIES}. Got {modality}. Will not do any intensity transform and will use original intensities."
            )

        transform = self.transform_dict[modality]
        # print("here(scripts.transforms.py) we use the transforms")
        return transform(img), img['image']
