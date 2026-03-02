"""
cityscapes_dataset.py — Cityscapes 数据集模块

Cityscapes 数据集目录结构
-------------------------
<root>/
├── leftImg8bit/          ← 原始图像
│   ├── train/
│   │   ├── aachen/
│   │   │   ├── aachen_000000_000019_leftImg8bit.png
│   │   │   └── ...
│   │   └── <city>/...
│   └── val/
│       └── <city>/...
└── gtFine/               ← 精细标注
    ├── train/
    │   ├── aachen/
    │   │   ├── aachen_000000_000019_gtFine_labelIds.png   ← 原始 labelId（0~33）
    │   │   ├── aachen_000000_000019_gtFine_color.png      ← 彩色可视化
    │   │   └── ...
    │   └── <city>/...
    └── val/
        └── <city>/...

标签说明
--------
- 使用 `*_gtFine_labelIds.png`（单通道灰度图，像素值为 labelId 0~33）
- 通过 LABEL_ID_TO_TRAIN_ID 映射表将 34 个 labelId 转换为 19 个 trainId
- 不参与训练/评估的类别（void 类）映射到 IGNORE_INDEX = 255
- 训练时 CrossEntropyLoss 的 ignore_index 应设为 255

19 个训练类别
--------------
 0: road          1: sidewalk      2: building      3: wall
 4: fence         5: pole          6: traffic light 7: traffic sign
 8: vegetation    9: terrain      10: sky          11: person
12: rider        13: car          14: truck        15: bus
16: train        17: motorcycle   18: bicycle

接口兼容性
----------
本模块与 CamVidDataset 保持相同接口，可直接替换用于：
  - Fedavg.py / fedavgTest.py / FedGMHC.py 的训练流程
  - partition.py 的 Dirichlet Non-IID 数据划分
"""

import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


# ==================== 类别定义 ====================

# 19 个训练类别名称（trainId 0~18）
CLASS_NAMES = [
    'road',          # 0
    'sidewalk',      # 1
    'building',      # 2
    'wall',          # 3
    'fence',         # 4
    'pole',          # 5
    'traffic light', # 6
    'traffic sign',  # 7
    'vegetation',    # 8
    'terrain',       # 9
    'sky',           # 10
    'person',        # 11
    'rider',         # 12
    'car',           # 13
    'truck',         # 14
    'bus',           # 15
    'train',         # 16
    'motorcycle',    # 17
    'bicycle',       # 18
]

NUM_CLASSES = 19   # 参与训练和评估的类别数
IGNORE_INDEX = 255 # void 类像素在 loss 计算中被忽略

# labelId（gtFine_labelIds.png 中的像素值）→ trainId 映射
# 来源：https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
# trainId=255 表示该类别不参与训练（void）
LABEL_ID_TO_TRAIN_ID = {
    0:  255,  # unlabeled
    1:  255,  # ego vehicle
    2:  255,  # rectification border
    3:  255,  # out of roi
    4:  255,  # static
    5:  255,  # dynamic
    6:  255,  # ground
    7:    0,  # road
    8:    1,  # sidewalk
    9:  255,  # parking
    10: 255,  # rail track
    11:   2,  # building
    12:   3,  # wall
    13:   4,  # fence
    14: 255,  # guard rail
    15: 255,  # bridge
    16: 255,  # tunnel
    17:   5,  # pole
    18: 255,  # polegroup
    19:   6,  # traffic light
    20:   7,  # traffic sign
    21:   8,  # vegetation
    22:   9,  # terrain
    23:  10,  # sky
    24:  11,  # person
    25:  12,  # rider
    26:  13,  # car
    27:  14,  # truck
    28:  15,  # bus
    29: 255,  # caravan
    30: 255,  # trailer
    31:  16,  # train
    32:  17,  # motorcycle
    33:  18,  # bicycle
    -1: 255,  # license plate
}

# 预构建 numpy 查找表（labelId → trainId），加速逐像素转换
# 支持 labelId 范围 0~255，超出范围的值映射到 IGNORE_INDEX
_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _lid, _tid in LABEL_ID_TO_TRAIN_ID.items():
    if 0 <= _lid <= 255:
        _LUT[_lid] = _tid

# 19 个训练类别的 RGB 可视化颜色（用于结果可视化）
CLASS_COLORS = np.array([
    (128,  64, 128),  # 0  road
    (244,  35, 232),  # 1  sidewalk
    ( 70,  70,  70),  # 2  building
    (102, 102, 156),  # 3  wall
    (190, 153, 153),  # 4  fence
    (153, 153, 153),  # 5  pole
    (250, 170,  30),  # 6  traffic light
    (220, 220,   0),  # 7  traffic sign
    (107, 142,  35),  # 8  vegetation
    (152, 251, 152),  # 9  terrain
    ( 70, 130, 180),  # 10 sky
    (220,  20,  60),  # 11 person
    (255,   0,   0),  # 12 rider
    (  0,   0, 142),  # 13 car
    (  0,   0,  70),  # 14 truck
    (  0,  60, 100),  # 15 bus
    (  0,  80, 100),  # 16 train
    (  0,   0, 230),  # 17 motorcycle
    (119,  11,  32),  # 18 bicycle
], dtype=np.uint8)


# ==================== 标签转换函数 ====================

def labelid_to_trainid(label_img):
    """
    将 labelId 灰度图（PIL Image 或 ndarray）转换为 trainId ndarray。

    参数
    ----
    label_img : PIL.Image（模式 'L' 或 'I'）或 ndarray (H, W)

    返回
    ----
    train_mask : ndarray (H, W)，dtype=int64
                 有效类别值为 0~18，void 类为 255
    """
    if isinstance(label_img, Image.Image):
        arr = np.array(label_img, dtype=np.int32)
    else:
        arr = np.asarray(label_img, dtype=np.int32)

    # 将超出 [0, 255] 范围的值（如 -1）先 clip 到 0
    arr_clipped = np.clip(arr, 0, 255).astype(np.uint8)
    train_mask = _LUT[arr_clipped].astype(np.int64)
    return train_mask


# ==================== 数据集类 ====================

class CityscapesDataset(Dataset):
    """
    Cityscapes 语义分割数据集

    参数
    ----
    root_dir    : 数据集根目录，包含 leftImg8bit/ 和 gtFine/ 两个子目录
                  示例：'E:/Autonomous Driving Dataset/Cityscapes dataset(10g)'
    split       : 数据集划分，'train' 或 'val'
    transform   : 图像变换（如 transforms.ToTensor()）
    target_size : 统一缩放的目标尺寸 (H, W)；设为 None 则不缩放
                  推荐：(512, 1024) 保持原始宽高比；(256, 512) 节省显存

    使用示例
    --------
    from cityscapes_dataset import CityscapesDataset, NUM_CLASSES, CLASS_NAMES
    from torchvision import transforms

    train_ds = CityscapesDataset(
        root_dir='E:/Autonomous Driving Dataset/Cityscapes dataset(10g)',
        split='train',
        transform=transforms.ToTensor(),
        target_size=(512, 1024),
    )
    img, mask = train_ds[0]
    # img:  FloatTensor (3, H, W)，值域 [0, 1]
    # mask: LongTensor  (H, W)，值域 0~18 和 255（void）
    """

    def __init__(self, root_dir, split='train', transform=None,
                 target_size=(512, 1024)):
        self.root_dir    = root_dir
        self.split       = split
        self.transform   = transform
        self.target_size = target_size  # (H, W)

        img_root  = os.path.join(root_dir, 'leftImg8bit', split)
        mask_root = os.path.join(root_dir, 'gtFine', split)

        if not os.path.isdir(img_root):
            raise FileNotFoundError(
                f"找不到图像目录: {img_root}\n"
                f"请确认数据集根目录正确，且包含 leftImg8bit/{split}/ 子目录。"
            )
        if not os.path.isdir(mask_root):
            raise FileNotFoundError(
                f"找不到标签目录: {mask_root}\n"
                f"请确认数据集根目录正确，且包含 gtFine/{split}/ 子目录。"
            )

        # 递归扫描所有城市子目录，收集图像和对应标签路径
        self.img_paths  = []
        self.mask_paths = []

        img_files = sorted(glob.glob(
            os.path.join(img_root, '**', '*_leftImg8bit.png'), recursive=True
        ))

        for img_path in img_files:
            # 从图像文件名推导对应的 labelIds 文件路径
            # 图像：<city>/<city>_<frame>_<seq>_leftImg8bit.png
            # 标签：<city>/<city>_<frame>_<seq>_gtFine_labelIds.png
            fname = os.path.basename(img_path)
            stem  = fname.replace('_leftImg8bit.png', '')
            city  = stem.split('_')[0]
            mask_fname = f'{stem}_gtFine_labelIds.png'
            mask_path  = os.path.join(mask_root, city, mask_fname)

            if os.path.isfile(mask_path):
                self.img_paths.append(img_path)
                self.mask_paths.append(mask_path)
            else:
                # 尝试扁平目录结构（无城市子目录）
                mask_path_flat = os.path.join(mask_root, mask_fname)
                if os.path.isfile(mask_path_flat):
                    self.img_paths.append(img_path)
                    self.mask_paths.append(mask_path_flat)
                # 找不到对应标签则跳过，并给出警告
                else:
                    import warnings
                    warnings.warn(f"[CityscapesDataset] 找不到对应标签，跳过: {img_path}")

        if len(self.img_paths) == 0:
            raise RuntimeError(
                f"在 {img_root} 下未找到任何有效的图像-标签对。\n"
                f"请检查目录结构是否符合 Cityscapes 标准格式。"
            )

        print(f"[CityscapesDataset] {split} 集加载完成: {len(self.img_paths)} 张图像")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path  = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # 加载图像（RGB）
        image = Image.open(img_path).convert('RGB')

        # 加载标签（单通道灰度，像素值为 labelId）
        # gtFine_labelIds.png 是 8 位或 16 位灰度图
        label = Image.open(mask_path)

        # 缩放
        if self.target_size is not None:
            H, W = self.target_size
            image = image.resize((W, H), Image.BILINEAR)
            label = label.resize((W, H), Image.NEAREST)

        # labelId → trainId 转换
        train_mask = labelid_to_trainid(label)  # ndarray (H, W), int64

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        mask_tensor = torch.from_numpy(train_mask).long()

        return image, mask_tensor

    def get_img_path(self, idx):
        """返回第 idx 张图像的绝对路径（调试用）"""
        return self.img_paths[idx]

    def get_mask_path(self, idx):
        """返回第 idx 张标签的绝对路径（调试用）"""
        return self.mask_paths[idx]


# ==================== partition.py 兼容接口 ====================

def get_dominant_class_cityscapes(mask_path, num_classes=NUM_CLASSES,
                                  target_size=(512, 1024)):
    """
    读取单张 gtFine_labelIds.png，返回占像素数最多的 trainId 类别索引。
    供 partition.py 的 build_label_index_cityscapes() 调用。

    忽略 void 类（trainId=255）的像素，仅统计有效类别。
    """
    label = Image.open(mask_path)
    if target_size is not None:
        H, W = target_size
        label = label.resize((W, H), Image.NEAREST)
    train_mask = labelid_to_trainid(label)

    # 仅统计有效类别（排除 void=255）
    valid_mask = train_mask[train_mask != IGNORE_INDEX]
    if len(valid_mask) == 0:
        return 0  # 全为 void，默认归入类别 0
    counts = np.bincount(valid_mask, minlength=num_classes)
    return int(np.argmax(counts))


def build_label_index_cityscapes(dataset_root, split='train',
                                 num_classes=NUM_CLASSES,
                                 target_size=(512, 1024),
                                 cache=True):
    """
    为 Cityscapes 训练集中的每张图像计算主类别标签（trainId），
    返回 ndarray (N,)，供 partition.py 的 dirichlet_partition() 使用。

    结果缓存到 {dataset_root}/cityscapes_partition_cache_{split}.npy。

    参数
    ----
    dataset_root : 数据集根目录
    split        : 'train' 或 'val'
    num_classes  : 训练类别数（默认 19）
    target_size  : 掩码缩放尺寸 (H, W)，与训练时保持一致
    cache        : 是否启用磁盘缓存
    """
    cache_path = os.path.join(
        dataset_root, f'cityscapes_partition_cache_{split}.npy'
    )
    if cache and os.path.exists(cache_path):
        labels = np.load(cache_path)
        print(f"  [Partition] 加载 Cityscapes 主类别缓存: {cache_path}  ({len(labels)} 张)")
        return labels

    mask_root = os.path.join(dataset_root, 'gtFine', split)
    mask_files = sorted(glob.glob(
        os.path.join(mask_root, '**', '*_gtFine_labelIds.png'), recursive=True
    ))

    if len(mask_files) == 0:
        raise RuntimeError(
            f"在 {mask_root} 下未找到任何 *_gtFine_labelIds.png 文件。"
        )

    print(f"  [Partition] 正在提取 {len(mask_files)} 张 Cityscapes 掩码的主类别"
          f"（首次运行，约需 1~3 分钟）...")

    labels = []
    for i, fpath in enumerate(mask_files):
        dom = get_dominant_class_cityscapes(fpath, num_classes, target_size)
        labels.append(dom)
        if (i + 1) % 200 == 0:
            print(f"    已处理 {i + 1}/{len(mask_files)} 张...")

    labels = np.array(labels, dtype=np.int64)

    if cache:
        np.save(cache_path, labels)
        print(f"  [Partition] 主类别缓存已保存: {cache_path}")

    return labels


# ==================== 快速验证入口 ====================

if __name__ == '__main__':
    """
    运行此脚本可快速验证数据集是否正确加载。
    用法：python cityscapes_dataset.py <数据集根目录>
    """
    import sys
    from torchvision import transforms

    root = sys.argv[1] if len(sys.argv) > 1 else \
        r'E:\Autonomous Driving Dataset\Cityscapes dataset(10g)'

    print(f"\n{'='*60}")
    print(f"Cityscapes 数据集验证")
    print(f"根目录: {root}")
    print(f"{'='*60}")

    for split in ['train', 'val']:
        try:
            ds = CityscapesDataset(
                root_dir=root,
                split=split,
                transform=transforms.ToTensor(),
                target_size=(512, 1024),
            )
            img, mask = ds[0]
            unique_ids = torch.unique(mask).tolist()
            valid_ids  = [x for x in unique_ids if x != 255]
            print(f"\n  [{split}] 样本数: {len(ds)}")
            print(f"  [{split}] 图像尺寸: {tuple(img.shape)}  (C, H, W)")
            print(f"  [{split}] 掩码尺寸: {tuple(mask.shape)}  (H, W)")
            print(f"  [{split}] 掩码中出现的 trainId: {valid_ids}")
            print(f"  [{split}] void 像素占比: "
                  f"{(mask == 255).float().mean().item() * 100:.1f}%")
        except Exception as e:
            print(f"\n  [{split}] 加载失败: {e}")

    print(f"\n{'='*60}")
    print(f"NUM_CLASSES = {NUM_CLASSES}")
    print(f"CLASS_NAMES = {CLASS_NAMES}")
    print(f"{'='*60}\n")
