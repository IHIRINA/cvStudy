import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import monai
from monai.data import CacheDataset, DataLoader
from monai.losses import SSIMLoss
from monai.transforms import (
    Compose, LoadImaged, Lambdad, EnsureTyped,
    RandRotated, RandFlipd, RandZoomd
)
from monai.utils import set_determinism
from tqdm import tqdm
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import shutil
import matplotlib.pyplot as plt


# --------------------------- 配置类 ---------------------------
class Config:
    """模型配置参数统一管理类"""

    def __init__(self):
        # 数据相关配置
        self.json_path = "/root/autodl-tmp/data/latent_t1ce_tof_pairs.json"
        self.data_root = "/root/autodl-tmp/data"
        self.batch_size = 1
        self.fold = 1
        self.cache_rate = 0.8
        self.num_workers = 4

        # 模型相关配置
        self.img_size = (40, 56, 56)  # 模型输入 3D 图像的空间维度，格式为 (深度D, 高度H, 宽度W)。必须能被patch_size整除
        self.accumulation_steps = 1  # 梯度累积步数设置为1，每个批次更新一次
        self.patch_size = (2, 4, 4)  # 3D Patch（分块）的尺寸，将 3D 图像划分为多个不重叠的小立方体块
        self.embed_dim = 512  # 每个 3D Patch 的嵌入维度（特征维度），即每个 Patch 被编码为 embed_dim 维的向量。
        self.depth = 12  # Transformer Block 的堆叠层数（模型深度）
        self.num_heads = 16  # 多头自注意力（Multi-Head Attention）的头数，将 embed_dim 维特征拆分为 num_heads 个头。
                            # 核心意义：多头注意力让模型同时关注不同位置 / 维度的特征，num_heads 个头表示并行学习 num_heads 组不同的注意力模式。
                            # 约束：embed_dim必须能被num_heads整除。
        self.mlp_ratio = 4.0  # Transformer Block 中 MLP 层的隐藏层维度放大系数，隐藏层维度 = embed_dim × mlp_ratio。
        self.drop_rate = 0.08  # 0.0
        self.in_channels = 4
        self.bottleneck_dim = 512  # Patch 嵌入过程中的瓶颈层维度，是从原始 Patch 维度到 embed_dim 的过渡维度。
                                   # 嵌入流程原始 Patch 维度（4×4×8×8=1024）→ 瓶颈层（bottleneck_dim）→ 最终嵌入维度（embed_dim），通过瓶颈层减少参数、加速训练。
        self.num_cbct_tokens = 16  # 16

        # 训练相关配置
        self.epochs = 1000
        self.lr = 2e-4  # 2e-4
        self.weight_decay = 0.05
        self.save_dir = "/root/autodl-tmp/results/checkpoints_56*56*40_4ch"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42  # 新增随机种子配置

        # 推理相关配置
        self.num_steps_infer = 50
        self.affine_matrix = [
            [-2.03515625, 0.0, 0.0, 265.0],
            [0.0, -2.03515625, 0.0, 172.0],
            [0.0, 0.0, -2.0, -230.5],
            [0.0, 0.0, 0.0, 1.0]
        ]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典用于日志记录"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}

    def save(self, save_path: str) -> None:
        """保存配置到JSON文件"""
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)


# --------------------------- 工具函数 ---------------------------
def setup_logging(save_dir: str) -> None:
    """设置日志配置"""
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + '.log'
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


# --------------------------- 1. 数据加载器 ---------------------------
class CBCT2CTDataLoader:
    """CBCT到CT数据加载器"""

    def __init__(self, config: Config):
        self.config = config
        self.json_path = config.json_path
        self.data_root = config.data_root

        # 验证数据路径
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON文件不存在: {self.json_path}")
        if not os.path.exists(self.data_root):
            raise NotADirectoryError(f"数据根目录不存在: {self.data_root}")

        with open(self.json_path) as f:
            self.full_data = json.load(f)["training"]
        self._add_absolute_paths()
        self._validate_data_paths()

    def _add_absolute_paths(self) -> None:
        """为数据项添加绝对路径"""
        for item in self.full_data:
            item["image"] = os.path.join(self.data_root, item["image"])
            item["label"] = os.path.join(self.data_root, item["label"])

    def _validate_data_paths(self) -> None:
        """验证数据文件是否存在"""
        missing_files = []
        for item in self.full_data:
            for key in ["image", "label"]:
                if not os.path.exists(item[key]):
                    missing_files.append(item[key])

        if missing_files:
            raise FileNotFoundError(f"以下数据文件不存在: {', '.join(missing_files[:5])}...")

    def get_loaders(self, augment: bool = False) -> Tuple[DataLoader, DataLoader]:
        """获取训练和验证数据加载器"""
        # 划分训练集和验证集
        train_data = [x for x in self.full_data if x["fold"] != self.config.fold]
        val_data = [x for x in self.full_data if x["fold"] == self.config.fold]

        logging.info(f"数据集划分: 训练集 {len(train_data)} 样本, 验证集 {len(val_data)} 样本")

        # 基础变换
        # 文件：model.py
        # 定位到 CBCT2CTDataLoader.get_loaders 里的 base_transforms 列表
        base_transforms = [
            LoadImaged(keys=["image", "label"],
                       image_only=False,  # 关键：保留 meta_dict
                       ensure_channel_first=True),
            Lambdad(keys=["dim", "spacing"],
                    func=lambda x: torch.tensor(x, dtype=torch.float32)),
            EnsureTyped(keys=["image", "label"], dtype=torch.float32)  # 删掉 data_type="dict"
        ]

        # 训练变换（含数据增强）
        if augment:
            train_transforms = Compose(base_transforms + [
                RandRotated(keys=["image", "label"], prob=0.5, range_x=0.1, range_y=0.1, range_z=0.1),
                RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
                RandZoomd(keys=["image", "label"], prob=0.5, min_zoom=0.9, max_zoom=1.1),
            ])
        else:
            train_transforms = Compose(base_transforms)

        val_transforms = Compose(base_transforms)

        # 创建数据集
        train_ds = CacheDataset(
            data=train_data,
            transform=train_transforms,
            cache_rate=self.config.cache_rate,
            num_workers=self.config.num_workers
        )
        val_ds = CacheDataset(
            data=val_data,
            transform=val_transforms,
            cache_rate=self.config.cache_rate,
            num_workers=self.config.num_workers
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config.num_workers > 0
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=torch.cuda.is_available()
        )

        return train_loader, val_loader


# --------------------------- 2. 模型核心模块 ---------------------------
class adaLNZero3D(nn.Module):
    """adaLN-Zero条件注入模块（时间步+CBCT条件）- 修复CBCT时间步融合"""

    def __init__(self, embed_dim: int, cbct_embed_dim: int = 768, num_cls_tokens: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_cls_tokens = num_cls_tokens
        # 时间步嵌入网络：将 1D 时间步映射到特征维度
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        # CBCT 特征嵌入网络：将 CBCT Patch特征映射到特征维度（不再是全局特征）
        self.cbct_embed = nn.Sequential(
            nn.Linear(cbct_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor, t: torch.Tensor, cbct_patch_emb: torch.Tensor) -> torch.Tensor:
        # 时间步嵌入
        t_1d = t.mean(dim=[2, 3, 4]).unsqueeze(1)  # (B,1) -> (B,1,768)
        t_emb = self.time_embed(t_1d)

        # 方案1修改：取消全局平均，直接编码所有CBCT Patch特征
        cbct_cls_emb = self.cbct_embed(cbct_patch_emb)  # (B, N_patch, 768)
        # 关键修复1：CBCT融合时间步信息（与CT保持一致）
        cbct_cls_emb = cbct_cls_emb + t_emb

        # 自适应归一化+条件注入（CT原有逻辑）
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.var(dim=-1, keepdim=True) + self.eps).sqrt()
        x = x * self.gamma + self.beta + t_emb

        # 拼接：CBCT Patch特征 + CT Patch特征（维度：B, N_patch+N_patch, 768）
        x = torch.cat([cbct_cls_emb, x], dim=1)
        return x


class PatchEmbedding3D(nn.Module):
    """3D Patch嵌入"""

    def __init__(self, patch_size: Tuple[int, int, int], embed_dim: int, in_channels: int, bottleneck_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]

        self.bottleneck = nn.Sequential(
            nn.Linear(patch_dim, bottleneck_dim),
            nn.SiLU(),
            nn.Linear(bottleneck_dim, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        Dp, Hp, Wp = self.patch_size

        # 验证输入尺寸是否与patch尺寸匹配
        assert D % Dp == 0 and H % Hp == 0 and W % Wp == 0, \
            f"输入尺寸 {(D, H, W)} 必须能被patch尺寸 {(Dp, Hp, Wp)} 整除"

        x = rearrange(
            x,
            "b c (d dp) (h hp) (w wp) -> b (d h w) (c dp hp wp)",
            dp=Dp, hp=Hp, wp=Wp
        )
        x = self.bottleneck(x)
        x = self.norm(x)
        return x


class MultiHeadSelfAttention3D(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "嵌入维度必须是头数的整数倍"
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock3D(nn.Module):
    """修复：新增交叉注意力层，显式对齐CBCT和CT同位置Patch"""

    def __init__(self, embed_dim: int, num_heads: int = 12,
                 mlp_ratio: float = 4., drop_rate: float = 0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 自注意力层（保留CT内部特征依赖）
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention3D(embed_dim, num_heads)

        # 新增：交叉注意力层（CT Query ← CBCT Key/Value）
        self.norm_cross = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=drop_rate)

        # MLP层
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.SiLU(),
            nn.Dropout(drop_rate),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop_rate)
        )
        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 拆分拼接后的特征：前N个是CBCT，后N个是CT
        N = x.shape[1] // 2
        cbct_feat = x[:, :N, :]
        ct_feat = x[:, N:, :]

        # 1. CT自注意力（保留CT内部特征依赖）
        ct_feat = ct_feat + self.residual_weight * self.self_attn(self.norm1(ct_feat))

        # 2. 交叉注意力：CT每个Patch显式关注同位置CBCT Patch
        # 生成单位矩阵掩码，强制仅同位置匹配（可选，进一步约束对齐）
        attn_mask = torch.eye(N, device=x.device, dtype=torch.bool)
        ct_feat_cross, _ = self.cross_attn(
            query=self.norm_cross(ct_feat),  # CT（查询）
            key=self.norm_cross(cbct_feat),  # CBCT（键）
            value=self.norm_cross(cbct_feat),  # CBCT（值）
            attn_mask=~attn_mask,  # 掩码：仅允许同位置注意力
            is_causal=False
        )
        ct_feat = ct_feat + self.residual_weight * ct_feat_cross

        # 3. MLP层
        ct_feat = ct_feat + self.residual_weight * self.mlp(self.norm2(ct_feat))

        # 拼接回原格式供下一层使用
        x = torch.cat([cbct_feat, ct_feat], dim=1)
        return x


class JiT3D_CBCT2CT(nn.Module):
    """CBCT-to-CT 3D JiT模型 - 修复CBCT位置编码缺失"""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.embed_dim = config.embed_dim
        self.num_cbct_tokens = config.num_cbct_tokens

        # 1. 3D分块嵌入层：将3D图像转换为patch序列
        self.patch_embed = PatchEmbedding3D(
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
            in_channels=config.in_channels,
            bottleneck_dim=config.bottleneck_dim
        )

        # 2. 位置嵌入：为每个patch添加空间位置信息（CBCT复用CT的位置编码）
        num_patches = (config.img_size[0] // config.patch_size[0]) * \
                      (config.img_size[1] // config.patch_size[1]) * \
                      (config.img_size[2] // config.patch_size[2])
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, config.embed_dim))

        # 3. 条件注入模块：融合时间步t和CBCT特征
        self.ada_ln_zero = adaLNZero3D(
            embed_dim=config.embed_dim,
            cbct_embed_dim=config.embed_dim,
            num_cls_tokens=config.num_cbct_tokens
        )

        # 4. Transformer块序列：提取全局特征依赖（使用修复后的交叉注意力版本）
        self.blocks = nn.ModuleList([
            TransformerBlock3D(config.embed_dim, config.num_heads,
                               config.mlp_ratio, config.drop_rate)
            for _ in range(config.depth)
        ])

        # 5. 最终归一化和输出头：将特征映射回3D图像
        self.norm_final = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(
            config.embed_dim,
            config.in_channels * config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """初始化权重"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, cbct: torch.Tensor, ct_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = ct_noisy.shape
        Dp, Hp, Wp = self.patch_size
        num_patches = (D // Dp) * (H // Hp) * (W // Wp)  # 计算CT Patch数量（1024）

        # 修复2：CBCT添加与CT完全一致的位置编码
        cbct_emb = self.patch_embed(cbct) + self.pos_embed  # (B, 1024, 768)
        ct_noisy_emb = self.patch_embed(ct_noisy)  # (B, 1024, 768)
        ct_emb = ct_noisy_emb + self.pos_embed  # (B, 1024, 768)

        # 条件注入（CBCT已融合时间步）
        ct_emb = self.ada_ln_zero(ct_emb, t, cbct_emb)  # 拼接后：(B, 2048, 768)

        # 经过带交叉注意力的Transformer块
        for blk in self.blocks:
            ct_emb = blk(ct_emb)
        ct_emb = self.norm_final(ct_emb)

        # 切掉前num_patches个CBCT特征，保留后num_patches个CT特征
        ct_emb = ct_emb[:, num_patches:, :]  # (B, 1024, 768)

        # 映射回3D图像
        ct_patch_pred = self.head(ct_emb)  # (B, 1024, 1024)
        ct_pred = rearrange(
            ct_patch_pred,
            "b (d h w) (c dp hp wp) -> b c (d dp) (h hp) (w wp)",
            d=D // Dp, h=H // Hp, w=W // Wp,
            dp=Dp, hp=Hp, wp=Wp,
            c=C
        )
        return ct_pred


# --------------------------- 3. 时间步采样器 ---------------------------
class LogitNormalTSampler:
    def __init__(self, mu: float = -0.8, sigma: float = 0.8, device: str = "cuda"):
        self.mu = mu
        self.sigma = sigma
        self.device = device

    def sample(self, batch_size: int) -> torch.Tensor:
        """采样时间步"""
        s = torch.normal(self.mu, self.sigma, size=(batch_size,), device=self.device)
        t = torch.sigmoid(s)
        t = torch.clamp(t, 0.05, 0.95)
        return t.unsqueeze(1)


class MixedLoss(nn.Module):
    """混合损失：结合L1/L2与SSIM损失"""

    def __init__(self, pixel_loss_type="L1", ssim_weight=0.5, spatial_dims=3, win_size=7):
        super().__init__()
        # 像素级损失（L1或L2）
        self.pixel_loss = nn.L1Loss() if pixel_loss_type == "L1" else nn.MSELoss()
        # SSIM损失（修正参数名win_size）
        self.ssim_loss = SSIMLoss(
            spatial_dims=spatial_dims,
            win_size=win_size
        )
        # 权重设置
        self.ssim_weight = ssim_weight
        self.pixel_weight = 1.0 - ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pixel_loss_val = self.pixel_loss(pred, target)
        ssim_loss_val = self.ssim_loss(pred, target)
        return self.pixel_weight * pixel_loss_val + self.ssim_weight * ssim_loss_val


# --------------------------- 4. 训练/验证/推理逻辑 ---------------------------
class Trainer:
    """模型训练器"""

    def __init__(self, config: Config):
        self.config = config
        self.device = config.device

        # 设置随机种子确保可复现性
        set_determinism(seed=config.seed)
        torch.manual_seed(config.seed)

        # 初始化日志
        setup_logging(config.save_dir)
        self.logger = logging.getLogger(__name__)

        # 保存配置文件
        config.save(os.path.join(config.save_dir, 'config.json'))
        self.logger.info(f"配置已保存至 {config.save_dir}/config.json")

        # 初始化数据加载器
        self.data_loader = CBCT2CTDataLoader(config)
        self.train_loader, self.val_loader = self.data_loader.get_loaders(augment=False)

        # 初始化模型、优化器、损失函数
        self.model = JiT3D_CBCT2CT(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        # 在Trainer类__init__中修改scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=500, T_mult=2, eta_min=1e-6
        )
        self.criterion = MixedLoss(
            pixel_loss_type="L1",
            ssim_weight=0.7,
            spatial_dims=3,
            win_size=9
        )
        self.t_sampler = LogitNormalTSampler(mu=-0.8, sigma=0.8, device=self.device)

        # 训练状态
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.loss_log = {
            "train": {"loss": []},
            "val": {"loss": []}
        }

        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        self.log_file = os.path.join(config.save_dir, "training.log")

        self.logger.info(f"训练初始化完成，设备: {self.device}")
        self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def train_one_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training", leave=False)):
            self.optimizer.zero_grad()

            cbct = batch["image"].to(self.device, non_blocking=True)
            ct_clean = batch["label"].to(self.device, non_blocking=True)
            B = cbct.shape[0]

            # 采样时间步并调整维度
            t = self.t_sampler.sample(B)
            t = t.view(B, 1, 1, 1, 1)

            # 生成带噪声的CT
            eps = torch.randn_like(ct_clean, device=self.device)
            ct_noisy = t * ct_clean + (1 - t) * eps

            # 模型预测与计算损失
            ct_pred = self.model(cbct, ct_noisy, t)
            loss = self.criterion(ct_pred, ct_clean)

            # 反向传播与优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * B

        return total_loss / len(self.train_loader.dataset)

    def validate(self) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                cbct = batch["image"].to(self.device, non_blocking=True)
                ct_clean = batch["label"].to(self.device, non_blocking=True)
                B = cbct.shape[0]

                # 随机采样时间步（与训练一致）
                t = self.t_sampler.sample(B)
                t = t.view(B, 1, 1, 1, 1)

                # 生成带噪声的CT
                eps = torch.randn_like(ct_clean, device=self.device)
                ct_noisy = t * ct_clean + (1 - t) * eps

                # 模型预测与计算损失
                ct_pred = self.model(cbct, ct_noisy, t)
                loss = self.criterion(ct_pred, ct_clean)
                total_loss += loss.item() * B

        return total_loss / len(self.val_loader.dataset)

    def run(self) -> None:
        """运行训练过程"""
        self.logger.info(f"开始训练，共 {self.config.epochs} 个epoch")

        for epoch in range(self.config.epochs):
            # 训练与验证
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            self.scheduler.step()

            # 记录损失
            self.loss_log["train"]["loss"].append(float(train_loss))
            self.loss_log["val"]["loss"].append(float(val_loss))

            # 打印并记录当前epoch信息
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # 保存日志
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.loss_log, f, indent=2)

            if (epoch + 1) % 5 == 0:
                self.visualize_samples(epoch + 1)

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': val_loss,
                    },
                    os.path.join(self.config.save_dir, "best_model_weights.pth")
                )
                self.logger.info(f"更新并保存最佳模型权重（验证损失: {self.best_val_loss:.4f}）")

        self.logger.info(f"训练完成！最佳验证损失在第 {self.best_epoch} 个epoch，损失值为: {self.best_val_loss:.4f}")

    def visualize_samples(self, epoch: int, num_samples: int = 2, save_dir: str = None):
        """
        对验证集中前 num_samples 个样本进行推理，保存轴向中间切片的对比图
        Args:
            epoch: 当前 epoch 编号
            num_samples: 可视化的样本数量（取验证集前几个）
            save_dir: 保存图像的目录，默认 config.save_dir/vis
        """
        self.model.eval()
        save_dir = save_dir or os.path.join(self.config.save_dir, "vis")
        os.makedirs(save_dir, exist_ok=True)

        # 从验证集中取前 num_samples 个样本
        val_data = list(self.val_loader.dataset)
        if len(val_data) > num_samples:
            val_data = val_data[:num_samples]

        # 推理参数（步数可较少，加速可视化）
        num_steps = min(self.config.num_steps_infer, 20)  # 最多20步

        with torch.no_grad():
            for idx, sample in enumerate(val_data):
                # 获取 CBCT 和真实 CT
                cbct = sample["image"].unsqueeze(0).to(self.device)  # (1, C, D, H, W)
                ct_real = sample["label"].unsqueeze(0).to(self.device)

                # 去噪推理
                ct_current = torch.randn_like(cbct, device=self.device)
                # Use the same valid t range as training (0.05 to 0.95).
                t_seq = torch.linspace(0.95, 0.05, num_steps, device=self.device)
                dt = 1.0 / (num_steps - 1)

                for step in range(num_steps):
                    t = t_seq[step].repeat(1).view(1, 1, 1, 1, 1)
                    ct_pred = self.model(cbct, ct_current, t)
                    if step < num_steps - 1:
                        v1 = (ct_pred - ct_current) / (1 - t + 1e-5)
                        ct_noisy_euler = ct_current + dt * v1
                        t_next = t_seq[step + 1].repeat(1).view(1, 1, 1, 1, 1)
                        ct_pred_next = self.model(cbct, ct_noisy_euler, t_next)
                        v2 = (ct_pred_next - ct_noisy_euler) / (1 - t_next + 1e-5)
                        ct_current = ct_current + dt * (v1 + v2) / 2.0
                    else:
                        ct_current = ct_pred

                ct_pred = ct_current[0]  # (C, D, H, W)
                ct_real = ct_real[0]

                # 获取轴向中间切片索引（深度维）
                D = ct_pred.shape[1]
                slice_idx = D // 2

                # 转换为 numpy，值域范围（实际可能需根据数据调整，这里取绝对值或 min-max 归一化）
                pred_slice = ct_pred[0, slice_idx, :, :].cpu().numpy()  # 单通道
                real_slice = ct_real[0, slice_idx, :, :].cpu().numpy()

                # 归一化显示（0-1）
                vmin, vmax = min(pred_slice.min(), real_slice.min()), max(pred_slice.max(), real_slice.max())
                pred_norm = (pred_slice - vmin) / (vmax - vmin + 1e-6)
                real_norm = (real_slice - vmin) / (vmax - vmin + 1e-6)

                # 保存图像
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(pred_norm, cmap='gray')
                axes[0].set_title(f'Generated CT (Epoch {epoch})')
                axes[0].axis('off')
                axes[1].imshow(real_norm, cmap='gray')
                axes[1].set_title('Ground Truth CT')
                axes[1].axis('off')

                # 从 meta_dict 获取原始文件名作为标识
                fname = sample.get("image_meta_dict", {}).get("filename_or_obj", f"sample_{idx}")
                base_name = os.path.basename(fname).replace('.nii.gz', '')
                out_path = os.path.join(save_dir, f"epoch_{epoch:04d}_{base_name}.png")
                plt.tight_layout()
                plt.savefig(out_path, dpi=150)
                plt.close(fig)

        self.logger.info(f"可视化图像已保存至 {save_dir} (epoch {epoch})")

class Inferrer:
    """推理器类，用于加载模型并对验证集进行推理保存"""

    def __init__(self, config: Config, checkpoint_path: Optional[str] = None):
        self.config = config
        self.device = config.device
        self.model = JiT3D_CBCT2CT(config).to(self.device)

        # 自动查找最佳模型权重
        self.checkpoint_path = checkpoint_path or os.path.join(config.save_dir, "best_model_weights.pth")
        self._load_checkpoint()

        # 初始化数据加载器
        self.data_loader = CBCT2CTDataLoader(config)
        _, self.val_loader = self.data_loader.get_loaders(augment=False)

    def _load_checkpoint(self):
        """加载模型权重"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"未找到模型权重文件: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        # 适配不同的权重保存格式
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        logging.info(f"已加载模型权重: {self.checkpoint_path}")

    def infer_and_save(self, output_dir: Optional[str] = None, num_steps_infer: Optional[int] = None):
        """
        使用验证数据集进行推理并将结果保存为.nii.gz格式
        :param output_dir: 推理结果保存目录
        :param num_steps_infer: 推理步数（去噪步骤数）
        """
        output_dir = output_dir or os.path.join(self.config.save_dir, "inference_results")
        num_steps_infer = num_steps_infer or self.config.num_steps_infer
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Inference")):
                cbct = batch["image"].to(self.device, non_blocking=True)
                B = cbct.shape[0]

                # 直接从 meta_dict 里拿文件名和 affine
                meta_dict = batch["image_meta_dict"]
                fname_list = meta_dict["filename_or_obj"]  # list[str]
                #affines = meta_dict["affine"]  # list[np.ndarray]

                #================================================
                raw_affines = []
                for fpath in fname_list:
                    raw_img = nib.load(fpath)
                    raw_affines.append(raw_img.affine.copy())
                #================================================



                # 初始化噪声
                ct_current = torch.randn_like(cbct, device=self.device)

                # Heun solver逐步去噪，使用训练范围内的 t 值避免 t=1.0 的极端数值问题。
                t_seq = torch.linspace(0.95, 0.05, num_steps_infer, device=self.device)
                dt = 1.0 / (num_steps_infer - 1)

                for step in range(num_steps_infer):
                    t = t_seq[step].repeat(B, 1).view(B, 1, 1, 1, 1)
                    ct_pred = self.model(cbct, ct_current, t)

                    # Heun更新规则
                    if step < num_steps_infer - 1:
                        v1 = (ct_pred - ct_current) / (1 - t + 1e-5)
                        ct_noisy_euler = ct_current + dt * v1

                        t_next = t_seq[step + 1].repeat(B, 1).view(B, 1, 1, 1, 1)
                        ct_pred_next = self.model(cbct, ct_noisy_euler, t_next)
                        v2 = (ct_pred_next - ct_noisy_euler) / (1 - t_next + 1e-5)

                        ct_current = ct_current + dt * (v1 + v2) / 2.0
                    else:
                        ct_current = ct_pred

                # 处理并保存每个样本
                '''
                for idx in range(B):
                    fake_ct = ct_current[idx].permute(1, 2, 3, 0).cpu().numpy()
                    affine = affines[idx] if isinstance(affines[idx], np.ndarray) else np.array(affines[idx])

                    in_path = fname_list[idx]
                    out_name = os.path.basename(in_path)  # 不再加任何后缀
                    out_path = os.path.join(output_dir, out_name)

                    nii_img = nib.Nifti1Image(fake_ct, affine)
                    nib.save(nii_img, out_path)
                    logging.info(f"Saved -> {out_path}")
                '''
                for idx in range(B):
                    latent = ct_current[idx].permute(1, 2, 3, 0).cpu().numpy()   # [D, H, W, C] 通常 C=4

                    # 获取原始 affine 并转换为 latent affine
                    raw_affine = raw_affines[idx]
                    latent_affine = raw_affine.copy()
                    latent_affine[:3, :3] *= 4
                    latent_affine[:3, 3] *= 4


                    in_path = fname_list[idx]
                    out_name = os.path.basename(in_path)
                    out_path = os.path.join(output_dir, out_name)

                    nii_img = nib.Nifti1Image(latent, latent_affine)
                    nib.save(nii_img, out_path)
                    logging.info(f"Saved latent -> {out_path}")

        logging.info(f"所有推理结果已保存至: {output_dir}")


if __name__ == "__main__":
    import argparse

    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description='CBCT to CT Conversion Model')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'],
                        help='运行模式: train (训练) 或 infer (推理)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='推理时使用的模型权重路径，默认使用最佳模型')
    args = parser.parse_args()

    # 初始化配置
    config = Config()

    if args.mode == 'train':
        trainer = Trainer(config)
        trainer.run()
    else:
        setup_logging(config.save_dir)
        inferrer = Inferrer(config, args.checkpoint)
        inferrer.infer_and_save()