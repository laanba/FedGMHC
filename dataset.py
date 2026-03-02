"""
数据集配置模块

包含：
- CamVid 颜色映射表和类别名称
- RGB 掩码转类别索引函数
- CamVidDataset 数据集类
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ===== CamVid 12 类 RGB 颜色映射表 =====
CAMVID_COLORS = np.array([
    [0,     0,   0],    # 0:  Void / Unlabelled
    [0,     0, 192],    # 1:  Pavement
    [0,   128, 192],    # 2:  Bicyclist
    [64,    0, 128],    # 3:  Car
    [64,   64,   0],    # 4:  Pedestrian
    [64,   64, 128],    # 5:  Fence
    [128,   0,   0],    # 6:  Building
    [128,  64, 128],    # 7:  Road
    [128, 128,   0],    # 8:  Tree
    [128, 128, 128],    # 9:  Sky
    [192, 128, 128],    # 10: SignSymbol / Pole
    [192, 192, 128],    # 11: Column_Pole
], dtype=np.uint8)

CLASS_NAMES = [
    'Void', 'Pavement', 'Bicyclist', 'Car', 'Pedestrian', 'Fence',
    'Building', 'Road', 'Tree', 'Sky', 'SignSymbol', 'Column_Pole'
]

NUM_CLASSES = len(CAMVID_COLORS)  # 12


def rgb_mask_to_class_index(mask_rgb, color_map=CAMVID_COLORS):
    """将 RGB 掩码图转换为类别索引图"""
    mask = np.array(mask_rgb)
    h, w = mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int64)
    for cls_idx, color in enumerate(color_map):
        match = np.all(mask == color, axis=-1)
        class_mask[match] = cls_idx
    return class_mask


class CamVidDataset(Dataset):
    """
    CamVid 语义分割数据集

    参数：
        root_dir:    数据集根目录，包含 train/, train_labels/, val/, val_labels/ 等子目录
        split:       数据集划分，'train' 或 'val' 或 'test'
        transform:   图像变换（如 transforms.ToTensor()）
        target_size: 统一缩放的目标尺寸 (H, W)，设为 None 则不缩放
    """

    def __init__(self, root_dir, split='train', transform=None, target_size=(256, 256)):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, split)
        self.mask_dir = os.path.join(root_dir, f'{split}_labels')
        self.images = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.target_size is not None:
            image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
            mask = mask.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(rgb_mask_to_class_index(mask)).long()

        return image, mask
