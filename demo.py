import torch
import torch.nn as nn
from torchvision import models
from torchview import draw_graph

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
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetV2UNet(nn.Module):
    def __init__(self, num_classes=11):  # 以 CamVid 的 11 类为例
        super().__init__()

        # 1. 加载预训练的 MobileNetV2 作为 Encoder
        backbone = models.mobilenet_v2(pretrained=True).features

        # 2. 提取不同尺度的特征图用于跳跃连接 (Skip Connections)
        # 原始尺寸: 3xH×W
        self.enc0 = backbone[0:2]  # 1/2 尺度 (16 channels)
        self.enc1 = backbone[2:4]  # 1/4 尺度 (24 channels)
        self.enc2 = backbone[4:7]  # 1/8 尺度 (32 channels)
        self.enc3 = backbone[7:14]  # 1/16 尺度 (96 channels)
        self.enc4 = backbone[14:19]  # 1/32 尺度 (1280 channels)

        # 3. Decoder 部分
        # 注意：这里的输入通道数要对应 MobileNetV2 相应层的输出
        self.dec3 = DecoderBlock(1280, 96, 256)  # 1/32 -> 1/16
        self.dec2 = DecoderBlock(256, 32, 128)  # 1/16 -> 1/8
        self.dec1 = DecoderBlock(128, 24, 64)  # 1/8 -> 1/4
        self.dec0 = DecoderBlock(64, 16, 32)  # 1/4 -> 1/2

        # 4. 最后的上采样和分类层
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder 路径
        e0 = self.enc0(x)  # 1/2
        e1 = self.enc1(e0)  # 1/4
        e2 = self.enc2(e1)  # 1/8
        e3 = self.enc3(e2)  # 1/16
        e4 = self.enc4(e3)  # 1/32 (最深层特征)

        # Decoder 路径 (带跳跃连接)
        d3 = self.dec3(e4, e3)  # 还原到 1/16
        d2 = self.dec2(d3, e2)  # 还原到 1/8
        d1 = self.dec1(d2, e1)  # 还原到 1/4
        d0 = self.dec0(d1, e0)  # 还原到 1/2

        out = self.final_upsample(d0)  # 还原到原图尺寸
        out = self.final_conv(out)

        return out


# 测试代码
if __name__ == "__main__":
    # model = MobileNetV2UNet(num_classes=11)
    # dummy_input = torch.randn(1, 3, 224, 224)  # 模拟一张图像
    # output = model(dummy_input)
    # print(f"输入尺寸: {dummy_input.shape}")
    # print(f"输出尺寸: {output.shape}")  # 应为 [1, 11, 224, 224]
    # # 在你的代码末尾添加
    # print(model)

    # 或者只查看各层的名字和参数量
    # for name, param in model.named_parameters():
    #     print(f"Layer: {name} | Size: {param.size()}")

    model = MobileNetV2UNet(num_classes=11)
    batch_size = 1
    # graph_size 控制展开深度，depth=2 可以看到 encoder/decoder 大块
    model_graph = draw_graph(model, input_size=(batch_size, 3, 224, 224), expand_nested=True, depth=2)
    model_graph.visual_graph.render("my_unet_model", format="png")