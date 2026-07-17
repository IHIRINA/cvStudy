import torch.nn as nn
import torch
import torch.nn.functional as F


class TembRes(nn.Module):
    def __init__(self, in_channels, emb_channels, out_channels):
        super().__init__()
        self.fea_layers = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(16, out_channels),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(16, out_channels),
        )

    def forward(self, x, emb):
        h = self.fea_layers(x)
        emb = self.emb_layers(emb)
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]
        h = h + emb
        out = self.out_layers(h)
        return out
    
class Xt_Fusion(nn.Module):
    def __init__(self, in_channels, emb_channels, out_channels):
        super().__init__()
        self.res_1 = TembRes(in_channels, emb_channels, out_channels)
        self.res_2 = TembRes(out_channels, emb_channels, out_channels)

    def forward(self, x, emb):
        out = self.res_1(x, emb)
        out = self.res_2(out, emb)
        return out
    

class Gate2(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels*2, 2, kernel_size=1)
    
    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=1)  # [B, 2*C, H, W]
        weights = torch.softmax(self.conv(x), dim=1)  # [B,2,H,W]
        return weights[:, 0:1]*f1 + weights[:, 1:2]*f2
    

class Gate3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels*3, 3, kernel_size=1)

    def forward(self, f1, f2, f3):
        x = torch.cat([f1, f2, f3], dim=1)  # [B, 3*C, H, W]
        weights = torch.softmax(self.conv(x), dim=1)  # [B, 3, H, W]
        return weights[:, 0:1]*f1 + weights[:, 1:2]*f2 + weights[:, 2:3]*f3
    

class CrossAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = Gate2(channels)
        # 保持Q/K的通道数与输入一致
        self.Q = nn.Conv2d(channels, channels, 1)
        self.K = nn.Conv2d(channels, channels, 1)
        self.V = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 初始化为0更稳定

    def forward(self, x_0, modal_1, modal_2):
        B, C, H, W = x_0.shape
        modal = self.gate(modal_1, modal_2)  # [B, C, H, W]

        # 1. 投影Q/K/V
        proj_query = self.Q(x_0).view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        proj_key = self.K(modal).view(B, C, -1)                   # [B, C, H*W]
        proj_value = self.V(modal).view(B, C, -1)                 # [B, C, H*W]

        # 2. 计算注意力权重
        energy = torch.bmm(proj_query, proj_key)                  # [B, H*W, H*W]
        attention = F.softmax(energy, dim=-1)

        # 3. 聚合Value
        out = torch.bmm(attention, proj_value.permute(0, 2, 1))   # [B, H*W, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)              

        # 4. 残差连接
        return x_0 + self.gamma * out
    

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = Gate3(channels)
        # 保持Q/K的通道数与输入一致
        self.Q = nn.Conv2d(channels, channels, 1)
        self.K = nn.Conv2d(channels, channels, 1)
        self.V = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))  # 初始化为0更稳定

    def forward(self, x_0_1, x_0_2, x_0_3):
        B, C, H, W = x_0_1.shape
        x_0 = self.gate(x_0_1, x_0_2, x_0_3)  # [B, C, H, W]

        # 1. 投影Q/K/V
        proj_query = self.Q(x_0).view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        proj_key = self.K(x_0).view(B, C, -1)                   # [B, C, H*W]
        proj_value = self.V(x_0).view(B, C, -1)                 # [B, C, H*W]

        # 2. 计算注意力权重
        energy = torch.bmm(proj_query, proj_key)                  # [B, H*W, H*W]
        attention = F.softmax(energy, dim=-1)

        # 3. 聚合Value
        out = torch.bmm(attention, proj_value.permute(0, 2, 1))   # [B, H*W, C]
        out = out.permute(0, 2, 1).view(B, C, H, W)              

        # 4. 残差连接
        return x_0 + self.gamma * out
    

class ModalGate3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 输入为三模态拼接 [B, 3C, H, W] → 输出3通道权重 [B, 3, H, W]
        self.weight_net = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 3, padding=1),  # 减少参数量
            nn.SiLU(),
            nn.Conv2d(channels, 3, 1),  # 输出3个通道，代表各模态权重
            nn.Softmax(dim=1)     # 在dim=1（模态维度）做Softmax
        )

    def forward(self, t1, t2, flair):
        x = torch.cat([t1, t2, flair], dim=1)  # [B, 3C, H, W]
        weights = self.weight_net(x)           # [B, 3, H, W]
        # 扩展权重以匹配原始通道数 [B, 3, H, W] → [B, 3, C, H, W]
        weights = weights.unsqueeze(2).repeat(1, 1, t1.size(1), 1, 1)
        # 加权求和
        out = (weights[:,0] * t1 + weights[:,1] * t2 + weights[:,2] * flair)
        return out




