"""
MobileNetV2-UNet 语义分割模型

基于 MobileNetV2 作为 Encoder，结合 UNet 风格的 Decoder，
适用于语义分割任务（如 CamVid 12 类分割）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 上采样：将特征图尺寸扩大两倍
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 拼接后的通道数为 in_channels + skip_channels
        self.conv = nn.Sequential(
            conv_bn_relu(in_channels + skip_channels, out_channels),
            conv_bn_relu(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if skip is not None:
            # ===== 修复：对齐空间尺寸 =====
            # 上采样后 x 的 H/W 可能与 skip 不一致（奇数尺寸导致），
            # 需要填充使两者空间维度完全匹配后再拼接。
            diff_h = skip.size(2) - x.size(2)
            diff_w = skip.size(3) - x.size(3)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2])
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetV2UNet(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()

        # 1. 加载预训练的 MobileNetV2 作为 Encoder
        backbone = models.mobilenet_v2(pretrained=True).features

        # 2. 提取不同尺度的特征图用于跳跃连接 (Skip Connections)
        self.enc0 = backbone[0:2]    # 1/2 尺度 (16 channels)
        self.enc1 = backbone[2:4]    # 1/4 尺度 (24 channels)
        self.enc2 = backbone[4:7]    # 1/8 尺度 (32 channels)
        self.enc3 = backbone[7:14]   # 1/16 尺度 (96 channels)
        self.enc4 = backbone[14:19]  # 1/32 尺度 (1280 channels)

        # 3. Decoder 部分
        self.dec3 = DecoderBlock(1280, 96, 256)   # 1/32 -> 1/16
        self.dec2 = DecoderBlock(256, 32, 128)     # 1/16 -> 1/8
        self.dec1 = DecoderBlock(128, 24, 64)      # 1/8  -> 1/4
        self.dec0 = DecoderBlock(64, 16, 32)       # 1/4  -> 1/2

        # 4. 最后的上采样和分类层
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[2:]

        # Encoder 路径
        e0 = self.enc0(x)   # 1/2
        e1 = self.enc1(e0)  # 1/4
        e2 = self.enc2(e1)  # 1/8
        e3 = self.enc3(e2)  # 1/16
        e4 = self.enc4(e3)  # 1/32

        # Decoder 路径 (带跳跃连接)
        d3 = self.dec3(e4, e3)  # 还原到 1/16
        d2 = self.dec2(d3, e2)  # 还原到 1/8
        d1 = self.dec1(d2, e1)  # 还原到 1/4
        d0 = self.dec0(d1, e0)  # 还原到 1/2

        out = self.final_upsample(d0)
        out = self.final_conv(out)

        # 确保输出与输入空间尺寸完全一致
        if out.shape[2:] != input_size:
            out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)

        return out
