"""
model/SegFormerB0.py — SegFormer-B0 风格语义分割模型

架构说明
--------
本文件实现 SegFormer-B0 风格的语义分割模型，采用：
  - 编码器：PVTv2-B0（Pyramid Vision Transformer v2 B0），
            与 MiT-B0 架构高度相似，同为层次化 Transformer，
            参数量约 3.4M，在 timm 1.0+ 中原生支持。
  - 解码器：SegFormer-style All-MLP Decoder，embed_dim=256。

编码器输出特征图尺寸（输入 256×512）：
  Stage 0: C=32,  H/4×W/4   = 64×128
  Stage 1: C=64,  H/8×W/8   = 32×64
  Stage 2: C=160, H/16×W/16 = 16×32
  Stage 3: C=256, H/32×W/32 = 8×16

解码器流程：
  1. 每个 Stage 特征通过 1×1 卷积统一映射到 embed_dim=256 通道
  2. 双线性上采样到 H/4×W/4（最大特征图尺寸）
  3. 拼接后通过融合卷积得到 [B, 256, H/4, W/4]
  4. 上采样 4× 恢复原始分辨率，输出 num_classes 通道

LayerNorm 特征提取（用于 FedGMHC 聚类）
-----------------------------------------
PVTv2-B0 使用 LayerNorm（共 30 层），不含 BatchNorm。
FedGMHC 的聚类特征从所有 LayerNorm 层的 weight（γ）和 bias（β）
参数中提取，拼接为 7168 维特征向量，经 PCA 降维后用于 GMM 聚类。
这些参数对数据分布敏感，能有效区分不同数据异质性的客户端。

总参数量：约 3.7M（编码器 3.4M + 解码器 0.3M）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm


class SegFormerDecoder(nn.Module):
    """
    SegFormer 轻量级 All-MLP 解码器。

    将编码器 4 个 Stage 的多尺度特征图统一映射到 embed_dim 通道，
    上采样到最大特征图尺寸后拼接，再经过融合卷积输出分割结果。
    """

    def __init__(self, in_channels, embed_dim=256, num_classes=19):
        """
        Parameters
        ----------
        in_channels : list[int]
            编码器各 Stage 输出通道数，例如 [32, 64, 160, 256]
        embed_dim : int
            解码器统一特征通道数，默认 256
        num_classes : int
            分割类别数，默认 19（Cityscapes）
        """
        super().__init__()
        self.embed_dim  = embed_dim
        self.num_stages = len(in_channels)

        # 每个 Stage 的线性投影层（1×1 卷积 = channel-wise MLP）
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])

        # 融合卷积：拼接后 num_stages × embed_dim → embed_dim
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(embed_dim * self.num_stages, embed_dim,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

        # 分类头
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, features):
        """
        Parameters
        ----------
        features : list[Tensor]
            编码器各 Stage 输出，shapes:
            [B, C0, H/4, W/4], [B, C1, H/8, W/8],
            [B, C2, H/16, W/16], [B, C3, H/32, W/32]

        Returns
        -------
        Tensor : [B, num_classes, H, W]  （与输入图像同分辨率）
        """
        # 目标尺寸：最大特征图（Stage 0 输出，H/4 × W/4）
        target_h, target_w = features[0].shape[2], features[0].shape[3]

        projected = []
        for i, feat in enumerate(features):
            x = self.proj_layers[i](feat)
            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = F.interpolate(x, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)
            projected.append(x)

        # 拼接并融合
        x = torch.cat(projected, dim=1)   # [B, embed_dim * 4, H/4, W/4]
        x = self.fuse_conv(x)             # [B, embed_dim, H/4, W/4]
        x = self.classifier(x)            # [B, num_classes, H/4, W/4]

        # 上采样 4× 恢复原始分辨率
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


class SegFormerB0(nn.Module):
    """
    SegFormer-B0 风格语义分割模型。

    编码器：timm 提供的 pvt_v2_b0（PVTv2-B0），
            与 MiT-B0 架构相似，均为层次化 Transformer，
            可选加载 ImageNet-1k 预训练权重。
    解码器：SegFormerDecoder（All-MLP Decoder），embed_dim=256。

    Parameters
    ----------
    num_classes : int
        分割类别数，默认 19（Cityscapes）
    pretrained : bool
        是否加载 ImageNet 预训练编码器权重，默认 True
    embed_dim : int
        解码器统一特征通道数，默认 256
    """

    # PVTv2-B0 各 Stage 输出通道数（与 MiT-B0 完全相同）
    ENCODER_CHANNELS = [32, 64, 160, 256]

    def __init__(self, num_classes=19, pretrained=True, embed_dim=256):
        super().__init__()

        # ---- 编码器：PVTv2-B0（层次化 Transformer，含 30 个 LayerNorm 层）----
        self.encoder = timm.create_model(
            'pvt_v2_b0',
            pretrained=pretrained,
            features_only=True,      # 返回各 Stage 中间特征图列表
            out_indices=(0, 1, 2, 3),
        )

        # ---- 解码器：All-MLP Decoder ----
        self.decoder = SegFormerDecoder(
            in_channels=self.ENCODER_CHANNELS,
            embed_dim=embed_dim,
            num_classes=num_classes,
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor  [B, 3, H, W]

        Returns
        -------
        Tensor  [B, num_classes, H, W]
        """
        features = self.encoder(x)   # list of 4 tensors
        return self.decoder(features)

    def get_layernorm_feature(self):
        """
        提取编码器中所有 LayerNorm 层的 weight（γ）和 bias（β）参数，
        拼接为一维 numpy 特征向量，用于 FedGMHC GMM 聚类。

        PVTv2-B0 共有 30 个 LayerNorm 层，特征向量维度约 7168。
        LayerNorm 参数对输入数据分布敏感：
          - γ（weight）反映特征的缩放程度
          - β（bias）反映特征的偏移程度
        两者合并可有效区分不同数据分布的客户端。

        Returns
        -------
        np.ndarray  形状 (D,)，D 约为 7168
        """
        parts = []
        for name, module in self.encoder.named_modules():
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    parts.append(module.weight.detach().cpu().float().numpy().ravel())
                if module.bias is not None:
                    parts.append(module.bias.detach().cpu().float().numpy().ravel())
        return np.concatenate(parts) if parts else np.array([])


def build_segformer_b0(num_classes=19, pretrained=True):
    """
    工厂函数：创建 SegFormer-B0 风格模型实例。

    Parameters
    ----------
    num_classes : int
        分割类别数
    pretrained : bool
        是否加载 ImageNet 预训练编码器权重

    Returns
    -------
    SegFormerB0
    """
    return SegFormerB0(num_classes=num_classes, pretrained=pretrained)


# ==================== 辅助函数：从 state_dict 提取 LN 特征 ====================

def extract_ln_feature(state_dict):
    """
    从模型 state_dict 中提取所有 LayerNorm 层的 weight 和 bias，
    拼接为一维 numpy 特征向量（用于 FedGMHC 聚类）。

    与 BN 特征提取（running_mean/running_var）不同，
    LN 特征使用可学习参数 γ/β，这些参数在训练中会随数据分布调整，
    能有效反映客户端的数据异质性。

    Parameters
    ----------
    state_dict : dict
        模型 state_dict（来自 model.state_dict()）

    Returns
    -------
    np.ndarray  形状 (D,)
    """
    parts = []
    for key, val in state_dict.items():
        # 匹配 encoder 中的 LayerNorm weight 和 bias
        # PVTv2 中 LN 层命名规律：*.norm.weight / *.norm.bias
        if ('encoder' in key) and (
            key.endswith('.norm.weight') or key.endswith('.norm.bias') or
            key.endswith('.norm1.weight') or key.endswith('.norm1.bias') or
            key.endswith('.norm2.weight') or key.endswith('.norm2.bias')
        ):
            parts.append(val.cpu().float().numpy().ravel())
    return np.concatenate(parts) if parts else np.array([])


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')

    model = build_segformer_b0(num_classes=19, pretrained=False).to(device)

    total_params   = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    print(f'总参数量:   {total_params:,}  ({total_params * 4 / 1024**2:.1f} MB in FP32)')
    print(f'编码器参数: {encoder_params:,}')
    print(f'解码器参数: {decoder_params:,}')

    # 前向传播测试
    x = torch.randn(2, 3, 256, 512).to(device)
    with torch.no_grad():
        out = model(x)
    print(f'输入尺寸: {tuple(x.shape)}')
    print(f'输出尺寸: {tuple(out.shape)}  (期望: [2, 19, 256, 512])')

    # LayerNorm 特征提取测试（方法1：直接从模型）
    feat1 = model.get_layernorm_feature()
    print(f'LayerNorm 特征向量维度 (get_layernorm_feature): {feat1.shape[0]}')

    # LayerNorm 特征提取测试（方法2：从 state_dict）
    feat2 = extract_ln_feature(model.state_dict())
    print(f'LayerNorm 特征向量维度 (extract_ln_feature):    {feat2.shape[0]}')
