"""
FedGMHC_Cityscapes.py — 基于高斯混合模型（GMM）分簇的联邦学习方法
                         专用于 Cityscapes 数据集

数据集说明
----------
- 图像目录: <DATASET_ROOT>/leftImg8bit/train|val/<city>/
- 标签目录: <DATASET_ROOT>/gtFine/train|val/<city>/*_gtFine_labelIds.png
- 训练类别: 19 类（road, sidewalk, building, ..., bicycle）
- 忽略类别: void（trainId=255），Focal Loss 中 ignore_index=255

极速验证配置（8GB 显存 + MobileUnet）v2
-----------------------------------------
本版本在 v1 基础上进行了以下关键增强：

  [新增] Focal Loss（γ=2.0）
    替代 CrossEntropyLoss，动态降低易分类像素（如 road）的梯度权重，
    强迫网络将注意力集中在稀有类（car、vegetation、bicycle 等）上。
    解决 Cityscapes 中极端的类别不平衡问题。

  [新增] 数据增强管线
    每个客户端仅 150~200 张图片，极易过拟合。加入：
    - RandomHorizontalFlip（p=0.5）
    - ColorJitter（亮度/对比度/饱和度随机扰动，模拟不同光照）
    - RandomScaleCrop（随机缩放 0.5~2.0 倍后裁切回 256×512）
    让 200 张图片发挥出 2000 张的效果。

  [新增] 两阶段学习率衰减
    热身期使用 LR=1e-3；分簇后立刻降为 LR×0.1=1e-4，
    将分簇阶段当作"微调（Fine-tuning）"，避免预热好的权重被刷爆。

  [修改] 热身/分簇比例 2:3
    50 轮 = 前 20 轮全局 FedAvg 预热 + 后 30 轮分簇个性化训练。
    20 轮预热足够让 BN 统计量充分反映各客户端数据分布。

  [修改] 重聚类融合比例
    迁移客户端的模型 = 70% 新簇模型 + 30% Global Anchor Model，
    而非之前的 30% 全局 + 70% 原簇。新簇模型权重更大，让迁移客户端
    快速适应新簇的数据分布。

  [新增] 个性化 mIoU 评估（论文核心指标）
    测试集也按 Non-IID 划分到各客户端。分簇后，用各客户端所属簇的
    聚合模型评估该客户端的本地测试集，算出本地 mIoU，再对 10 个
    客户端求平均。这才能真正体现"因地制宜"的个性化增益。

冷启动解决策略（组合方案）
--------------------------
方案一  延迟聚类（热身期）
        前 WARMUP_ROUNDS 轮执行标准 FedAvg，等待模型收敛、BN 统计量
        充分反映各客户端数据分布后，再进行首次 GMM 聚类。

方案二  动态重聚类
        首次聚类后，每隔 RECLUSTER_INTERVAL 轮重新提取 BN 特征并
        重新拟合 GMM，让分簇结果随训练进程自我修正并逐渐稳定。
        当客户端从簇 A 迁移到簇 B 时，该客户端的模型 =
        70% × 簇 B 最新模型 + 30% × Global Anchor Model，
        保证性能曲线平滑过渡，避免跳跃式下降。

算法流程
--------
热身期（Round 1 ~ 20）  LR = 1e-3
    所有客户端从全局模型出发，完成本地训练，执行标准 FedAvg 聚合。
    每轮结束后，所有簇模型与全局模型保持同步。

首次聚类（Round 20 结束后）
    提取各客户端 Bottleneck 层 BN 的 running_mean / running_var，
    拼接为特征向量，拟合 GMM（K=3，对角协方差），
    按后验概率最大值分配各客户端到对应簇。
    同时冻结当前全局模型为 Global Anchor Model。

分簇训练期（Round 21 ~ 50）  LR = 1e-4（微调）
    每轮：
      a. 每个客户端从所属簇模型出发，完成本地训练。
      b. 同簇客户端按数据量加权 FedAvg，更新各簇模型。
      c. 各簇模型按各簇总数据量加权 FedAvg，更新全局模型。
      d. 若当前轮满足重聚类条件（每 5 轮一次），重新提取 BN 特征
         并更新分簇结果。迁移客户端的模型做插值融合。
      e. 个性化评估：用各客户端所属簇模型评估其本地测试集。

结果保存
--------
每次运行结果统一保存在 result_save/FedGMHC_MMDDHHmm/ 子目录下：
  result_save/
  └── FedGMHC_MMDDHHmm/
      ├── gmm_cluster_log.json           ← 每次聚类的详细记录
      ├── cluster_val_results.csv        ← 每轮每簇验证数据（全局验证集）
      ├── global_val_results.csv         ← 每轮全局模型验证数据
      ├── personalized_val_results.csv   ← 每轮个性化 mIoU（本地测试集）
      ├── pixel_accuracy.png             ← Pixel Accuracy 折线图
      ├── miou.png                       ← mIoU 折线图（含个性化 mIoU）
      └── intra_cluster_distance.png     ← 簇内平均距离折线图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import json
import random
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

import os
import sys

# ===== 路径修复：将项目根目录加入 sys.path，适配 FedGMHC/ 子目录运行 =====
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model import MobileNetV2UNet
from dataset.cityscapes_dataset import (
    CityscapesDataset,
    NUM_CLASSES,
    CLASS_NAMES,
    IGNORE_INDEX,
    build_label_index_cityscapes,
)
from partition import dirichlet_partition, print_partition_stats
import time
import csv
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


# ==================== 超参数 ====================
NUM_CLUSTERS        = 3     # GMM 部件数 / 簇数（模拟城市核心区/市郊/高速公路）
WARMUP_ROUNDS       = 20    # 热身轮数：前 20 轮执行标准 FedAvg（50 轮的 2/5）
RECLUSTER_INTERVAL  = 5     # 动态重聚类间隔 F=5：每隔 5 轮重新聚类一次
# 重聚类后迁移客户端的模型融合比例：
#   迁移客户端模型 = RECLUSTER_CLUSTER_WEIGHT × 新簇模型
#                  + (1 - RECLUSTER_CLUSTER_WEIGHT) × Global Anchor Model
# 70% 新簇模型 + 30% Global Anchor，让迁移客户端快速适应新簇
RECLUSTER_CLUSTER_WEIGHT = 0.7

# ==================== BN 特征提取策略 ====================
BOTTLENECK_BN_PREFIX = 'enc4.18'

# GMM 后验概率温度软化参数
GMM_TEMPERATURE     = 5.0

# ==================== Focal Loss 参数 ====================
FOCAL_GAMMA         = 2.0   # 聚焦参数：γ=2.0 让网络对难分样本极度敏感
FOCAL_ALPHA         = None  # 类别权重：None 表示不使用（也可设为各类频率倒数）

# ==================== 学习率衰减 ====================
LR_WARMUP           = 1e-3  # 热身期学习率
LR_DECAY_FACTOR     = 0.1   # 分簇后学习率衰减因子：LR_WARMUP × 0.1 = 1e-4
# LR_CLUSTERED = LR_WARMUP * LR_DECAY_FACTOR = 1e-4


# ==================== Focal Loss 实现 ====================

class FocalLoss(nn.Module):
    """
    Focal Loss for Dense Object Detection (Lin et al., 2017)
    专为解决语义分割中极端类别不平衡而设计。

    核心公式：
        FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    当网络对某像素的预测已经很准（p_t 接近 1）时，
    (1 - p_t)^γ 趋近于 0，该像素对 Loss 的贡献被大幅降低。
    网络被迫将梯度集中在那些预测不准的稀有像素上。

    参数
    ----
    gamma        : float, 聚焦参数 γ，默认 2.0
                   γ=0 退化为标准 CrossEntropy；γ 越大越关注难分样本
    alpha        : float 或 Tensor 或 None
                   - None: 不使用类别权重
                   - float: 所有类别统一权重
                   - Tensor(C,): 每个类别单独权重（如频率倒数）
    ignore_index : int, 忽略的标签值（Cityscapes void 类 = 255）
    """

    def __init__(self, gamma=2.0, alpha=None, ignore_index=255):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index

        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([alpha])
            else:
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        参数
        ----
        logits  : (B, C, H, W) 模型原始输出（未经 softmax）
        targets : (B, H, W) 真实标签，值域 0~18 和 255（void）

        返回
        ----
        loss : scalar，Focal Loss 均值
        """
        B, C, H, W = logits.shape

        # Step 1: 展平为 (B*H*W, C) 和 (B*H*W,)
        logits_flat  = logits.permute(0, 2, 3, 1).reshape(-1, C)   # (N, C)
        targets_flat = targets.reshape(-1)                          # (N,)

        # Step 2: 过滤 ignore_index 像素
        valid_mask   = (targets_flat != self.ignore_index)
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0  # 全为 void，返回 0 loss

        logits_valid  = logits_flat[valid_mask]    # (M, C)
        targets_valid = targets_flat[valid_mask]    # (M,)

        # Step 3: 计算 log_softmax 和 softmax
        log_probs = F.log_softmax(logits_valid, dim=1)              # (M, C)
        probs     = log_probs.exp()                                 # (M, C)

        # Step 4: 提取真实类别的概率 p_t
        # gather 沿 dim=1 取出每个像素对应真实类别的概率
        targets_onehot = targets_valid.unsqueeze(1)                 # (M, 1)
        log_pt = log_probs.gather(1, targets_onehot).squeeze(1)     # (M,)
        pt     = probs.gather(1, targets_onehot).squeeze(1)         # (M,)

        # Step 5: 计算 Focal 权重 (1 - p_t)^γ
        focal_weight = (1.0 - pt).pow(self.gamma)                   # (M,)

        # Step 6: 可选的类别权重 α_t
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)
            if alpha_t.numel() == 1:
                at = alpha_t.expand_as(log_pt)
            else:
                at = alpha_t[targets_valid]                         # (M,)
            focal_weight = focal_weight * at

        # Step 7: Focal Loss = -α_t × (1-p_t)^γ × log(p_t)
        loss = -focal_weight * log_pt                               # (M,)
        return loss.mean()


# ==================== 数据增强管线 ====================

class CityscapesAugTransform:
    """
    Cityscapes 语义分割专用数据增强管线。

    对图像和标签同步执行以下增强操作：
      1. RandomHorizontalFlip（p=0.5）— 水平翻转
      2. ColorJitter — 随机改变亮度/对比度/饱和度（仅作用于图像）
      3. RandomScaleCrop — 随机缩放 [0.5, 2.0] 倍后裁切回 target_size

    设计要点：
      - 标签（mask）必须使用最近邻插值（NEAREST），避免引入无效类别
      - ColorJitter 仅作用于图像，不影响标签
      - 每个客户端仅 150~200 张图片，数据增强是防过拟合的最后一道防线

    参数
    ----
    target_size : (H, W)，最终输出尺寸，默认 (256, 512)
    scale_range : (min, max)，随机缩放比例范围
    flip_prob   : 水平翻转概率
    """

    def __init__(self, target_size=(256, 512), scale_range=(0.5, 2.0),
                 flip_prob=0.5):
        from torchvision import transforms as T
        from PIL import Image

        self.target_h, self.target_w = target_size
        self.scale_range = scale_range
        self.flip_prob   = flip_prob

        # ColorJitter 仅作用于图像（亮度 ±0.3, 对比度 ±0.3, 饱和度 ±0.3）
        self.color_jitter = T.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.0
        )
        self.to_tensor = T.ToTensor()

    def __call__(self, image, mask):
        """
        参数
        ----
        image : PIL.Image (RGB)
        mask  : numpy.ndarray (H, W), dtype=int64, 值域 0~18 和 255

        返回
        ----
        image_tensor : FloatTensor (3, H, W)
        mask_tensor  : LongTensor  (H, W)
        """
        from PIL import Image as PILImage

        # 将 mask 转为 PIL Image 以便同步变换
        mask_pil = PILImage.fromarray(mask.astype(np.uint8), mode='L')

        # --- 1. RandomHorizontalFlip ---
        if random.random() < self.flip_prob:
            image    = image.transpose(PILImage.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.transpose(PILImage.FLIP_LEFT_RIGHT)

        # --- 2. ColorJitter（仅图像）---
        image = self.color_jitter(image)

        # --- 3. RandomScaleCrop ---
        # 随机缩放
        scale = random.uniform(*self.scale_range)
        w, h  = image.size  # PIL: (W, H)
        new_h = int(h * scale)
        new_w = int(w * scale)

        image    = image.resize((new_w, new_h), PILImage.BILINEAR)
        mask_pil = mask_pil.resize((new_w, new_h), PILImage.NEAREST)

        # 随机裁切（或 pad + 裁切）到 target_size
        if new_h >= self.target_h and new_w >= self.target_w:
            # 随机裁切
            top  = random.randint(0, new_h - self.target_h)
            left = random.randint(0, new_w - self.target_w)
            image    = image.crop((left, top, left + self.target_w, top + self.target_h))
            mask_pil = mask_pil.crop((left, top, left + self.target_w, top + self.target_h))
        else:
            # 缩放后比目标小，先 pad 再裁切
            # pad 用 0（图像）和 255（mask，即 void）
            pad_h = max(self.target_h - new_h, 0)
            pad_w = max(self.target_w - new_w, 0)

            # 图像 pad（黑色）
            padded_img = PILImage.new('RGB', (new_w + pad_w, new_h + pad_h), (0, 0, 0))
            padded_img.paste(image, (0, 0))
            # mask pad（255 = void）
            padded_mask = PILImage.new('L', (new_w + pad_w, new_h + pad_h), 255)
            padded_mask.paste(mask_pil, (0, 0))

            # 随机裁切
            crop_h = padded_img.size[1]  # height
            crop_w = padded_img.size[0]  # width
            top  = random.randint(0, max(crop_h - self.target_h, 0))
            left = random.randint(0, max(crop_w - self.target_w, 0))
            image    = padded_img.crop((left, top, left + self.target_w, top + self.target_h))
            mask_pil = padded_mask.crop((left, top, left + self.target_w, top + self.target_h))

        # --- 转为 Tensor ---
        image_tensor = self.to_tensor(image)                          # (3, H, W)
        mask_arr     = np.array(mask_pil, dtype=np.int64)
        # 恢复 255（void）：uint8 转换可能丢失，确保 > 18 的值都是 255
        mask_arr[mask_arr > 18] = 255
        mask_tensor  = torch.from_numpy(mask_arr).long()              # (H, W)

        return image_tensor, mask_tensor


class CityscapesAugDataset(torch.utils.data.Dataset):
    """
    带数据增强的 Cityscapes 数据集包装器。

    在原始 CityscapesDataset 基础上，替换 __getitem__ 的变换逻辑：
    - 训练时：使用 CityscapesAugTransform（含翻转、色彩抖动、随机裁切）
    - 验证时：仅 Resize + ToTensor（无增强）

    设计理由：
    每个客户端仅 150~200 张图片，不加数据增强会导致 MobileUnet 在前几轮
    直接过拟合（Training Loss 极低，Validation mIoU 纹丝不动）。

    参数
    ----
    base_dataset : CityscapesDataset 实例
    aug_transform : CityscapesAugTransform 实例（训练用）
    """

    def __init__(self, base_dataset, aug_transform):
        self.base = base_dataset
        self.aug  = aug_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        from PIL import Image as PILImage

        img_path  = self.base.img_paths[idx]
        mask_path = self.base.mask_paths[idx]

        # 加载原始图像和标签
        image = PILImage.open(img_path).convert('RGB')
        label = PILImage.open(mask_path)

        # 先缩放到 target_size（与 base_dataset 一致）
        if self.base.target_size is not None:
            H, W = self.base.target_size
            image = image.resize((W, H), PILImage.BILINEAR)
            label = label.resize((W, H), PILImage.NEAREST)

        # labelId → trainId
        from dataset.cityscapes_dataset import labelid_to_trainid
        train_mask = labelid_to_trainid(label)  # ndarray (H, W), int64

        # 应用数据增强（图像+标签同步变换）
        image_tensor, mask_tensor = self.aug(image, train_mask)

        return image_tensor, mask_tensor


# ==================== 显存监控工具 ====================

def get_gpu_memory_info(device):
    if not torch.cuda.is_available():
        return None
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
    cached    = torch.cuda.memory_reserved(device)  / 1024 ** 2
    total     = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    return {
        'allocated_mb':    allocated,
        'cached_mb':       cached,
        'total_mb':        total,
        'free_mb':         total - cached,
        'utilization_pct': (cached / total) * 100,
    }


def print_gpu_status(device, label=""):
    info = get_gpu_memory_info(device)
    if info is None:
        print(f"  [{label}] 未检测到 GPU，使用 CPU 训练")
        return
    print(f"  [{label}] 显存: 已分配 {info['allocated_mb']:.0f}MB | "
          f"已缓存 {info['cached_mb']:.0f}MB | "
          f"空闲 {info['free_mb']:.0f}MB | "
          f"总计 {info['total_mb']:.0f}MB | "
          f"利用率 {info['utilization_pct']:.1f}%")


# ==================== 评估指标 ====================

def compute_pixel_accuracy(pred, target, ignore_index=IGNORE_INDEX):
    """计算像素准确率，忽略 void 类（ignore_index=255）"""
    valid_mask = (target != ignore_index)
    if valid_mask.sum() == 0:
        return 0.0
    correct = ((pred == target) & valid_mask).sum().item()
    total   = valid_mask.sum().item()
    return correct / total


def compute_iou_per_class(pred, target, num_classes, ignore_index=IGNORE_INDEX):
    """计算各类别 IoU，忽略 void 类"""
    ious = []
    valid_mask = (target != ignore_index)
    for cls in range(num_classes):
        pred_c   = (pred  == cls) & valid_mask
        target_c = (target == cls) & valid_mask
        inter = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        ious.append(inter / union if union > 0 else float('nan'))
    return ious


def compute_miou(pred, target, num_classes, ignore_index=IGNORE_INDEX):
    ious  = compute_iou_per_class(pred, target, num_classes, ignore_index)
    valid = [v for v in ious if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def evaluate_model(model, val_loader, device, use_amp=True):
    """返回 (avg_pixel_acc, avg_miou)，评估时忽略 void 类（trainId=255）"""
    model.eval()
    total_pa, total_miou, n = 0.0, 0.0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with autocast(enabled=use_amp and torch.cuda.is_available()):
                output = model(images)
            preds = output.argmax(dim=1)
            for i in range(preds.size(0)):
                total_pa   += compute_pixel_accuracy(preds[i], labels[i])
                total_miou += compute_miou(preds[i], labels[i], NUM_CLASSES)
                n += 1
    model.train()
    return (total_pa / n if n else 0.0), (total_miou / n if n else 0.0)


def evaluate_personalized(cluster_models, client_cluster, client_posteriors,
                          val_dataset, client_val_indices, device,
                          num_clients, num_clusters, batch_size=8,
                          num_workers=0, pin_memory=True, use_amp=True):
    """
    个性化 mIoU 评估（论文核心指标）。

    正确做法：测试集也按 Non-IID 划分到各客户端。分簇后，用各客户端
    所属簇的聚合模型评估该客户端的本地测试集，算出本地 mIoU。
    最后对 10 个客户端的本地 mIoU 求平均，作为这一轮的总性能。

    错误做法：拿分簇后的模型去跑一个包含所有场景的全局验证集。
    这就失去了"个性化"的意义。

    参数
    ----
    cluster_models     : list[nn.Module]，各簇聚合模型
    client_cluster     : list[int]，各客户端的主簇分配
    client_posteriors  : ndarray (N, K)，软分簇权重（此处仅用 argmax）
    val_dataset        : CityscapesDataset，验证集（无增强）
    client_val_indices : list[list[int]]，各客户端的本地验证集索引
    device, batch_size, num_workers, pin_memory, use_amp : 训练参数

    返回
    ----
    avg_personalized_miou : float，所有客户端本地 mIoU 的平均值
    avg_personalized_pa   : float，所有客户端本地 Pixel Acc 的平均值
    per_client_results    : list[dict]，每个客户端的详细结果
    """
    per_client_results = []

    for i in range(num_clients):
        cluster_id = client_cluster[i]
        model      = cluster_models[cluster_id]
        indices    = client_val_indices[i]

        if len(indices) == 0:
            per_client_results.append({
                'client': i, 'cluster': cluster_id,
                'num_val_samples': 0, 'pixel_acc': 0.0, 'miou': 0.0,
            })
            continue

        local_loader = DataLoader(
            Subset(val_dataset, indices),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        pa, miou = evaluate_model(model, local_loader, device, use_amp=use_amp)
        per_client_results.append({
            'client': i, 'cluster': cluster_id,
            'num_val_samples': len(indices), 'pixel_acc': pa, 'miou': miou,
        })

    valid_clients = [r for r in per_client_results if r['num_val_samples'] > 0]
    avg_pa   = float(np.mean([r['pixel_acc'] for r in valid_clients])) if valid_clients else 0.0
    avg_miou = float(np.mean([r['miou']      for r in valid_clients])) if valid_clients else 0.0

    return avg_miou, avg_pa, per_client_results


# ==================== BN 特征提取（仅 Bottleneck 层）====================

def extract_bn_feature(state_dict):
    """
    从模型 state_dict 中仅提取 Bottleneck 层（enc4.18）的
    BatchNorm running_mean 和 running_var，拼接为一维特征向量（numpy）。
    """
    parts = []
    for key, val in state_dict.items():
        if key.startswith(BOTTLENECK_BN_PREFIX) and \
           ('running_mean' in key or 'running_var' in key):
            parts.append(val.cpu().float().numpy().ravel())
    return np.concatenate(parts) if parts else np.array([])


# ==================== GMM 聚类 ====================

def run_gmm_clustering(local_weights, num_clients, n_clusters, round_idx, run_dir,
                       cluster_log, prev_assignments=None):
    """
    提取 Bottleneck 层 BN 特征 → 标准化 → 拟合 GMM（对角协方差）
    → 温度软化后验概率 → 返回软分配权重。
    """
    print(f"\n  [GMM] 提取 Bottleneck 层 BN 统计特征（Round {round_idx + 1}）...")
    features = [extract_bn_feature(w) for w in local_weights]
    feat_dim = features[0].shape[0]
    print(f"  [GMM] Bottleneck BN 特征向量维度: {feat_dim}")

    X = np.stack(features, axis=0)   # (N, D)
    N = X.shape[0]
    effective_k = min(n_clusters, N)

    # ---- Step 1: 标准化 ----
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Step 2: 拟合 GMM（对角协方差，数值稳定）----
    gmm = GaussianMixture(
        n_components=effective_k,
        covariance_type='diag',
        max_iter=500,
        n_init=10,
        random_state=42,
        reg_covar=1e-4,
    )
    try:
        gmm.fit(X_scaled)
        print(f"  [GMM] 拟合成功 (covariance_type='diag', reg_covar=1e-4)")
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"  [GMM] 首次拟合失败 ({e})，尝试增大正则化...")
        gmm = GaussianMixture(
            n_components=effective_k,
            covariance_type='diag',
            max_iter=500,
            n_init=10,
            random_state=42,
            reg_covar=1e-2,
        )
        gmm.fit(X_scaled)
        print(f"  [GMM] 拟合成功 (covariance_type='diag', reg_covar=1e-2)")

    # ---- Step 3: 基于欧式距离的温度 softmax 软化 ----
    means = gmm.means_                                                 # (K, D)
    dists = np.linalg.norm(
        X_scaled[:, None, :] - means[None, :, :], axis=2
    )                                                                  # (N, K)
    neg_dist_T = -dists / GMM_TEMPERATURE
    neg_dist_T -= neg_dist_T.max(axis=1, keepdims=True)
    posteriors  = np.exp(neg_dist_T)
    posteriors /= posteriors.sum(axis=1, keepdims=True)

    max_probs       = posteriors.max(axis=1)
    new_assignments = posteriors.argmax(axis=1).tolist()
    changed         = (prev_assignments is None) or (new_assignments != prev_assignments)

    print(f"  [GMM] 欧式距离温度软化（T={GMM_TEMPERATURE}）:")
    print(f"        最大概率范围: {max_probs.min():.3f} ~ {max_probs.max():.3f}")
    print(f"        各簇平均权重: {posteriors.mean(axis=0).round(3).tolist()}")

    def _save_scatter(X_data, labels, post, run_dir, round_idx):
        """生成 2D 聚类散点图（对高维特征用 PCA 降到 2D 仅用于可视化）。"""
        from sklearn.decomposition import PCA as PCA_vis
        if X_data.shape[1] > 2:
            pca_vis = PCA_vis(n_components=2, random_state=42)
            X_2d = pca_vis.fit_transform(X_data)
        else:
            X_2d = X_data

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        base_colors = np.array([
            [0.91, 0.30, 0.24],
            [0.18, 0.80, 0.44],
            [0.20, 0.60, 0.86],
            [0.95, 0.61, 0.07],
            [0.61, 0.35, 0.71],
        ])
        fig, ax = plt.subplots(figsize=(7, 6))
        for i in range(len(labels)):
            color = np.zeros(3)
            for k in range(min(effective_k, len(base_colors))):
                color += post[i, k] * base_colors[k % len(base_colors)]
            color = np.clip(color, 0, 1)
            ax.scatter(X_2d[i, 0], X_2d[i, 1], c=[color], s=140,
                       zorder=3, edgecolors='white', linewidths=0.8)
            ax.annotate(f'C{i}', (X_2d[i, 0], X_2d[i, 1]),
                        textcoords='offset points', xytext=(6, 4), fontsize=9)
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=base_colors[k % len(base_colors)],
                                 label=f'Cluster {k}') for k in range(effective_k)]
        ax.legend(handles=legend_elements, fontsize=10)
        ax.set_xlabel('PC 1', fontsize=11)
        ax.set_ylabel('PC 2', fontsize=11)
        ax.set_title(f'Client Soft Clustering (Round {round_idx + 1}, T={GMM_TEMPERATURE})', fontsize=12)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        scatter_path = os.path.join(run_dir, f'cluster_scatter_round{round_idx + 1}.png')
        fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [GMM] 聚类散点图已保存: {scatter_path}")

    label_str = '首次' if prev_assignments is None else '重聚类'
    print(f"  [GMM] 客户端软分簇权重（{label_str}）:")
    for i in range(N):
        prob_str   = ', '.join([f'C{j}:{posteriors[i, j]:.3f}'
                                for j in range(posteriors.shape[1])])
        change_tag = ''
        if prev_assignments is not None and new_assignments[i] != prev_assignments[i]:
            change_tag = f'  ← 主簇从 Cluster {prev_assignments[i]} 迁移到 Cluster {new_assignments[i]}'
        print(f"    Client {i} 主簇 Cluster {new_assignments[i]}  ({prob_str}){change_tag}")

    for k in range(effective_k):
        members = [i for i, c in enumerate(new_assignments) if c == k]
        print(f"  Cluster {k}: {members} ({len(members)} 个客户端)")

    if not changed:
        print(f"  [GMM] 主簇分配结果与上次相同。")

    _save_scatter(X_scaled, new_assignments, posteriors, run_dir, round_idx)

    cluster_log.append({
        'round':                 round_idx + 1,
        'trigger':               'warmup_end' if prev_assignments is None else 'recluster',
        'method':                'gmm_soft',
        'temperature':           GMM_TEMPERATURE,
        'feature_dim':           int(feat_dim),
        'feature_source':        f'Bottleneck BN ({BOTTLENECK_BN_PREFIX})',
        'gmm_covariance_type':   'diag',
        'gmm_reg_covar':         1e-4,
        'assignments':           new_assignments,
        'changed':               changed,
        'posteriors':            posteriors.tolist(),
        'cluster_members':       {
            str(k): [i for i, c in enumerate(new_assignments) if c == k]
            for k in range(effective_k)
        },
    })

    log_path = os.path.join(run_dir, 'gmm_cluster_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_log, f, ensure_ascii=False, indent=2)
    print(f"  [GMM] 聚类日志已更新: {log_path}")

    return new_assignments, changed, posteriors


# ==================== 模型插值融合 ====================

def interpolate_models(target_model, cluster_model, anchor_model, cluster_weight):
    """
    迁移客户端的模型融合：
        新模型参数 = cluster_weight × 新簇模型 + (1 - cluster_weight) × Global Anchor

    参数
    ----
    target_model   : 待更新的模型（就地修改）
    cluster_model  : 新簇的最新聚合模型
    anchor_model   : Global Anchor Model（热身期结束时冻结的全局模型）
    cluster_weight : 新簇模型的权重（0.7 = 70% 新簇 + 30% Anchor）
    """
    cluster_sd = cluster_model.state_dict()
    anchor_sd  = anchor_model.state_dict()
    blended    = {}
    for key in cluster_sd:
        if cluster_sd[key].is_floating_point():
            blended[key] = cluster_weight * cluster_sd[key] + \
                           (1.0 - cluster_weight) * anchor_sd[key]
        else:
            blended[key] = cluster_sd[key]
    target_model.load_state_dict(blended)
    return target_model


# ==================== 联邦聚合 ====================

def fedavg(base_model, weights_list, lens_list):
    """按数据量加权平均聚合，结果写入 base_model 并返回。"""
    total       = sum(lens_list)
    global_dict = copy.deepcopy(weights_list[0])
    for key in global_dict:
        global_dict[key] = global_dict[key] * (lens_list[0] / total)
    for i in range(1, len(weights_list)):
        frac = lens_list[i] / total
        for key in global_dict:
            global_dict[key] += weights_list[i][key] * frac
    base_model.load_state_dict(global_dict)
    return base_model


# ==================== 客户端 ====================

class Client:
    def __init__(self, client_id, dataset, indices, device, use_amp=True):
        self.client_id = client_id
        self.device    = device
        self.use_amp   = use_amp and torch.cuda.is_available()
        self.dataset   = dataset
        self.indices   = indices

    def local_train(self, model, batch_size=8, epochs=1, lr=0.01,
                    num_workers=0, pin_memory=False):
        loader = DataLoader(
            Subset(self.dataset, self.indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )
        model.train()
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # 使用 Focal Loss 替代 CrossEntropyLoss
        # γ=2.0：对预测准确的像素（如 road）大幅降低权重
        # 强迫网络关注稀有类（car、vegetation、bicycle 等）
        criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA,
                              ignore_index=IGNORE_INDEX)
        scaler    = GradScaler(enabled=self.use_amp)

        epoch_losses = []
        for _ in range(epochs):
            running_loss, num_batches = 0.0, 0
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    output = model(images)
                    loss   = criterion(output, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                num_batches  += 1
            epoch_losses.append(running_loss / max(num_batches, 1))

        return model.state_dict(), float(np.mean(epoch_losses))


# ==================== 结果保存 ====================

def save_cluster_csv(cluster_history, run_dir):
    path = os.path.join(run_dir, 'cluster_val_results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Phase', 'Cluster', 'Num_Clients', 'Num_Samples',
                         'Pixel_Accuracy', 'mIoU', 'Avg_Loss'])
        for r in cluster_history:
            writer.writerow([
                r['round'], r['phase'], r['cluster'],
                r['num_clients'], r['num_samples'],
                f"{r['pixel_acc']:.6f}", f"{r['miou']:.6f}", f"{r['avg_loss']:.6f}",
            ])
    return path


def save_global_csv(global_history, run_dir):
    path = os.path.join(run_dir, 'global_val_results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Phase', 'Pixel_Accuracy', 'mIoU',
                         'Personalized_mIoU', 'Personalized_PA',
                         'Avg_Loss', 'LR', 'Time_s'])
        for r in global_history:
            writer.writerow([
                r['round'], r['phase'],
                f"{r['pixel_acc']:.6f}", f"{r['miou']:.6f}",
                f"{r.get('pers_miou', 0.0):.6f}", f"{r.get('pers_pa', 0.0):.6f}",
                f"{r['avg_loss']:.6f}", f"{r.get('lr', 0.0):.6f}",
                f"{r['time']:.1f}",
            ])
    return path


def save_personalized_csv(pers_history, run_dir):
    """保存每轮个性化评估的详细结果（每个客户端一行）"""
    path = os.path.join(run_dir, 'personalized_val_results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Client', 'Cluster', 'Num_Val_Samples',
                         'Pixel_Accuracy', 'mIoU'])
        for r in pers_history:
            writer.writerow([
                r['round'], r['client'], r['cluster'],
                r['num_val_samples'],
                f"{r['pixel_acc']:.6f}", f"{r['miou']:.6f}",
            ])
    return path


def save_curves(cluster_history, global_history, num_clusters, warmup_rounds, run_dir):
    """
    生成折线图：
      1. pixel_accuracy.png — 各簇 + 全局 + 个性化 Pixel Accuracy
      2. miou.png           — 各簇 + 全局 + 个性化 mIoU
    """
    cluster_data = {k: {'rounds': [], 'pa': [], 'miou': []} for k in range(num_clusters)}
    for r in cluster_history:
        k = r['cluster']
        cluster_data[k]['rounds'].append(r['round'])
        cluster_data[k]['pa'].append(r['pixel_acc'])
        cluster_data[k]['miou'].append(r['miou'])

    global_rounds = [r['round']     for r in global_history]
    global_pa     = [r['pixel_acc'] for r in global_history]
    global_miou   = [r['miou']      for r in global_history]

    # 个性化 mIoU（分簇期才有）
    pers_rounds = [r['round']          for r in global_history if r.get('pers_miou', 0) > 0]
    pers_pa     = [r.get('pers_pa', 0) for r in global_history if r.get('pers_miou', 0) > 0]
    pers_miou   = [r.get('pers_miou', 0) for r in global_history if r.get('pers_miou', 0) > 0]

    colors = plt.cm.tab10.colors

    for ylabel, title, suffix, c_key, g_data, p_data, p_rounds in [
        ('Pixel Accuracy',
         'Cityscapes — Pixel Accuracy (Global vs. Personalized vs. Cluster)',
         'pixel_accuracy.png', 'pa', global_pa, pers_pa, pers_rounds),
        ('mIoU',
         'Cityscapes — mIoU (Global vs. Personalized vs. Cluster)',
         'miou.png', 'miou', global_miou, pers_miou, pers_rounds),
    ]:
        plt.figure(figsize=(14, 6))

        if warmup_rounds > 0 and global_rounds and warmup_rounds < max(global_rounds):
            plt.axvline(x=warmup_rounds + 0.5, color='gray', linestyle=':', linewidth=1.5,
                        label=f'Warmup End (R{warmup_rounds})')

        for k in range(num_clusters):
            d = cluster_data[k]
            if d['rounds']:
                plt.plot(d['rounds'], d[c_key],
                         linestyle='--', marker='o', linewidth=1.5, markersize=3,
                         color=colors[k % len(colors)],
                         label=f'Cluster {k}')

        if global_rounds:
            plt.plot(global_rounds, g_data,
                     linestyle='-', marker='s', linewidth=2.5, markersize=4,
                     color='black', label='Global')

        # 个性化曲线（红色粗线，论文核心指标）
        if p_rounds:
            plt.plot(p_rounds, p_data,
                     linestyle='-', marker='D', linewidth=2.5, markersize=5,
                     color='red', label='Personalized (Avg Local)')

        plt.xlabel('Communication Round', fontsize=13)
        plt.ylabel(ylabel, fontsize=13)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(run_dir, suffix)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  已保存: {save_path}")


def save_intra_dist_curve(intra_dist_history, num_clusters, warmup_rounds, run_dir):
    """生成簇内平均距离折线图"""
    rounds        = [r['round']         for r in intra_dist_history]
    overall_intra = [r['overall_intra'] for r in intra_dist_history]

    per_cluster = {k: [] for k in range(num_clusters)}
    per_rounds  = {k: [] for k in range(num_clusters)}
    for r in intra_dist_history:
        for k in range(num_clusters):
            if k in r['per_cluster']:
                per_cluster[k].append(r['per_cluster'][k])
                per_rounds[k].append(r['round'])

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(13, 6))

    if warmup_rounds > 0 and rounds and warmup_rounds < max(rounds):
        ax.axvline(x=warmup_rounds + 0.5, color='gray', linestyle=':', linewidth=1.5,
                   label=f'Warmup End (R{warmup_rounds})')

    for k in range(num_clusters):
        if per_rounds[k]:
            ax.plot(per_rounds[k], per_cluster[k],
                    linestyle='--', marker='o', linewidth=1.5, markersize=3,
                    color=colors[k % len(colors)], alpha=0.8,
                    label=f'Cluster {k}')

    ax.plot(rounds, overall_intra,
            linestyle='-', marker='s', linewidth=2.5, markersize=4,
            color='black', label='Overall Mean')

    ax.set_xlabel('Communication Round', fontsize=13)
    ax.set_ylabel('Intra-cluster Distance (Bottleneck BN Feature Space)', fontsize=13)
    ax.set_title('FedGMHC — Intra-cluster Distance vs. Round', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()

    save_path = os.path.join(run_dir, 'intra_cluster_distance.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存: {save_path}")

    csv_path = os.path.join(run_dir, 'intra_cluster_distance.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Round', 'Phase', 'Overall_Intra'] + \
                     [f'Cluster_{k}_Intra' for k in range(num_clusters)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in intra_dist_history:
            row = {
                'Round':         r['round'],
                'Phase':         r['phase'],
                'Overall_Intra': round(r['overall_intra'], 6) if r['overall_intra'] is not None else '',
            }
            for k in range(num_clusters):
                row[f'Cluster_{k}_Intra'] = round(r['per_cluster'].get(k, 0), 6) \
                    if k in r['per_cluster'] else ''
            writer.writerow(row)
    print(f"  已保存: {csv_path}")


# ==================== 主函数 ====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if torch.cuda.is_available():
        print(f"GPU 型号: {torch.cuda.get_device_name(device)}")
        print(f"GPU 总显存: {torch.cuda.get_device_properties(device).total_memory / 1024**2:.0f} MB")
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark: 已启用")
    else:
        print("警告：未检测到 GPU，将使用 CPU 训练（速度会非常慢）")

    from torchvision import transforms

    # ==================== 配置区 ====================
    DATASET_ROOT = r'E:\Autonomous Driving Dataset\Cityscapes dataset(10g)'

    USE_AMP      = True
    TARGET_SIZE  = (256, 512)   # 保持 1:2 宽高比

    # ==================== 极速验证配置 v2 ====================
    NUM_ROUNDS      = 50    # 总轮次 = 20 轮热身 + 30 轮分簇（2:3 比例）
    NUM_CLIENTS     = 10
    LOCAL_EPOCHS    = 1
    BATCH_SIZE      = 8
    DIRICHLET_ALPHA = 0.5
    MIN_SAMPLES     = 150
    MAX_SAMPLES     = 200
    EVAL_EVERY      = 1

    NUM_WORKERS  = 0 if sys.platform == 'win32' else 4
    PIN_MEMORY   = True

    # 两阶段学习率
    current_lr = LR_WARMUP  # 热身期 LR = 1e-3，分簇后 LR = 1e-4

    print(f"\n[极速验证配置 v2]")
    print(f"  分辨率: {TARGET_SIZE[0]}×{TARGET_SIZE[1]} | 客户端: {NUM_CLIENTS} | 簇数: {NUM_CLUSTERS}")
    print(f"  每端数据: {MIN_SAMPLES}~{MAX_SAMPLES} 张 | Dirichlet α={DIRICHLET_ALPHA}")
    print(f"  热身轮数: {WARMUP_ROUNDS} | 分簇轮数: {NUM_ROUNDS - WARMUP_ROUNDS} | 总轮次: {NUM_ROUNDS}")
    print(f"  学习率: 热身 {LR_WARMUP} → 分簇 {LR_WARMUP * LR_DECAY_FACTOR}")
    print(f"  损失函数: Focal Loss (γ={FOCAL_GAMMA})")
    print(f"  数据增强: RandomHFlip + ColorJitter + RandomScaleCrop")
    print(f"  重聚类间隔: F={RECLUSTER_INTERVAL} | 融合比例: {RECLUSTER_CLUSTER_WEIGHT:.0%} 新簇 + {1-RECLUSTER_CLUSTER_WEIGHT:.0%} Anchor")
    print(f"  评估方式: 全局 mIoU + 个性化 mIoU（Non-IID 本地测试集）")
    print(f"  BN 特征: 仅 Bottleneck 层 ({BOTTLENECK_BN_PREFIX})")
    print(f"  GMM: K={NUM_CLUSTERS}, covariance_type='diag', reg_covar=1e-4")

    # ===== 时间戳运行目录 =====
    run_timestamp = datetime.now().strftime('%m%d%H%M')
    run_dir = os.path.join('../result_save', f'FedGMHC_{run_timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n本次运行结果将保存至: {run_dir}/")

    # ===== 加载 Cityscapes 数据集 =====
    print(f"\n正在加载 Cityscapes 数据集（路径: {DATASET_ROOT}）...")

    # 训练集（带数据增强）
    base_train_dataset = CityscapesDataset(
        root_dir=DATASET_ROOT,
        split='train',
        transform=transforms.ToTensor(),
        target_size=TARGET_SIZE,
    )
    aug_transform = CityscapesAugTransform(
        target_size=TARGET_SIZE,
        scale_range=(0.5, 2.0),
        flip_prob=0.5,
    )
    train_dataset = CityscapesAugDataset(base_train_dataset, aug_transform)
    print(f"  [数据增强] 已启用: RandomHFlip(p=0.5) + ColorJitter + RandomScaleCrop(0.5~2.0)")

    # 验证集（无增强，仅 Resize + ToTensor）
    val_dataset = CityscapesDataset(
        root_dir=DATASET_ROOT,
        split='val',
        transform=transforms.ToTensor(),
        target_size=TARGET_SIZE,
    )

    num_images = len(train_dataset)

    # ===== Dirichlet Non-IID 数据划分（训练集）=====
    print(f"\n  [Partition] 使用 Dirichlet(α={DIRICHLET_ALPHA}) Non-IID 划分训练集...")
    train_labels = build_label_index_cityscapes(
        DATASET_ROOT, split='train',
        num_classes=NUM_CLASSES,
        target_size=TARGET_SIZE,
    )
    user_groups = dirichlet_partition(
        num_clients=NUM_CLIENTS, labels=train_labels, num_classes=NUM_CLASSES,
        alpha=DIRICHLET_ALPHA, min_samples=MIN_SAMPLES, seed=42,
    )
    if MAX_SAMPLES is not None:
        random.seed(42)
        clipped = 0
        for i in range(len(user_groups)):
            if len(user_groups[i]) > MAX_SAMPLES:
                user_groups[i] = random.sample(list(user_groups[i]), MAX_SAMPLES)
                clipped += 1
        if clipped > 0:
            print(f"  [Partition] MAX_SAMPLES={MAX_SAMPLES}: {clipped} 个客户端数据被截断")

    print_partition_stats(user_groups, train_labels, NUM_CLASSES, CLASS_NAMES)

    # ===== Dirichlet Non-IID 数据划分（验证集）=====
    # 论文核心：测试集也按 Non-IID 划分，用簇模型评估客户端本地测试集
    print(f"\n  [Partition] 使用 Dirichlet(α={DIRICHLET_ALPHA}) Non-IID 划分验证集...")
    val_labels = build_label_index_cityscapes(
        DATASET_ROOT, split='val',
        num_classes=NUM_CLASSES,
        target_size=TARGET_SIZE,
    )
    val_user_groups = dirichlet_partition(
        num_clients=NUM_CLIENTS, labels=val_labels, num_classes=NUM_CLASSES,
        alpha=DIRICHLET_ALPHA, min_samples=10, seed=42,
    )
    for i in range(NUM_CLIENTS):
        print(f"    Client {i} 验证集: {len(val_user_groups[i])} 张")

    min_data = min(len(g) for g in user_groups)

    print(f"\n{'='*70}")
    print(f"训练配置（Cityscapes 极速验证 v2）:")
    print(f"  数据集根目录: {DATASET_ROOT}")
    print(f"  图像尺寸: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}  |  训练类别: {NUM_CLASSES}  |  IGNORE_INDEX: {IGNORE_INDEX}")
    print(f"  训练集: {num_images} 张 | 验证集: {len(val_dataset)} 张")
    print(f"  客户端: {NUM_CLIENTS} 个 | 簇数: {NUM_CLUSTERS} | Dirichlet α={DIRICHLET_ALPHA}")
    print(f"  热身: {WARMUP_ROUNDS} 轮 (LR={LR_WARMUP}) | 分簇: {NUM_ROUNDS - WARMUP_ROUNDS} 轮 (LR={LR_WARMUP * LR_DECAY_FACTOR})")
    print(f"  重聚类间隔: F={RECLUSTER_INTERVAL} | 融合: {RECLUSTER_CLUSTER_WEIGHT:.0%} 新簇 + {1-RECLUSTER_CLUSTER_WEIGHT:.0%} Anchor")
    print(f"  损失函数: Focal Loss (γ={FOCAL_GAMMA})")
    print(f"  数据增强: RandomHFlip + ColorJitter + RandomScaleCrop")
    print(f"  Batch Size: {BATCH_SIZE} | Local Epochs: {LOCAL_EPOCHS}")
    print(f"  BN 特征: Bottleneck ({BOTTLENECK_BN_PREFIX}) | GMM: diag, reg=1e-4")
    print(f"{'='*70}")

    # ===== 初始化全局模型 =====
    global_model = MobileNetV2UNet(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"\n模型总参数量: {total_params:,} ({total_params * 4 / 1024**2:.1f} MB in FP32)")

    # 全局验证集 DataLoader（用于全局 mIoU 评估）
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    save_dir = '../checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    print_gpu_status(device, "训练前基线")

    # ===== 状态变量 =====
    cluster_models     = [copy.deepcopy(global_model) for _ in range(NUM_CLUSTERS)]
    global_anchor      = None   # Global Anchor Model：热身结束时冻结的全局模型
    client_cluster     = None
    client_posteriors  = None
    last_cluster_round = -1
    cluster_log        = []

    cluster_history    = []
    global_history     = []
    pers_history       = []     # 个性化评估详细记录
    intra_dist_history = []
    best_miou          = 0.0
    best_pers_miou     = 0.0
    total_time         = 0.0

    print(f"\n{'='*80}")
    print(f"开始 FedGMHC 训练（Cityscapes 极速验证 v2）")
    print(f"热身 {WARMUP_ROUNDS} 轮 (LR={LR_WARMUP}) → "
          f"分簇 {NUM_ROUNDS - WARMUP_ROUNDS} 轮 (LR={LR_WARMUP * LR_DECAY_FACTOR})")
    print(f"损失函数: Focal Loss (γ={FOCAL_GAMMA}) | 数据增强: 已启用")
    print(f"{'='*80}")

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        is_warmup   = (round_idx < WARMUP_ROUNDS)
        phase_label = f'Warmup({round_idx + 1}/{WARMUP_ROUNDS})' if is_warmup \
                      else f'Clustered({round_idx + 1 - WARMUP_ROUNDS}/{NUM_ROUNDS - WARMUP_ROUNDS})'

        # 两阶段学习率
        if is_warmup:
            current_lr = LR_WARMUP
        else:
            current_lr = LR_WARMUP * LR_DECAY_FACTOR

        print(f"\n--- Round {round_idx + 1}/{NUM_ROUNDS}  [{phase_label}]  LR={current_lr:.1e} ---")

        # ================================================================
        # 阶段 A：每个客户端本地训练
        # ================================================================
        local_weights = []
        local_losses  = []
        local_lens    = []

        for i in range(NUM_CLIENTS):
            client = Client(i, train_dataset, user_groups[i], device, use_amp=USE_AMP)

            if is_warmup or client_cluster is None:
                start_model = copy.deepcopy(global_model)
            else:
                start_model = copy.deepcopy(cluster_models[client_cluster[i]])

            weights, loss = client.local_train(
                start_model,
                batch_size=BATCH_SIZE,
                epochs=LOCAL_EPOCHS,
                lr=current_lr,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )
            local_weights.append(weights)
            local_losses.append(loss)
            local_lens.append(len(user_groups[i]))

            del start_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_loss = float(np.mean(local_losses))
        print(f"  所有客户端本地训练完成 | 平均 Focal Loss: {avg_loss:.4f}")

        # ================================================================
        # 阶段 B：热身期 → 标准 FedAvg 聚合
        # ================================================================
        if is_warmup:
            global_model = fedavg(global_model, local_weights, local_lens)
            for k in range(NUM_CLUSTERS):
                cluster_models[k].load_state_dict(copy.deepcopy(global_model.state_dict()))

        # ================================================================
        # 阶段 C：热身期结束 → 首次 GMM 聚类 + 冻结 Global Anchor
        # ================================================================
        if round_idx == WARMUP_ROUNDS - 1:
            print(f"\n  [GMM] 热身期结束（{WARMUP_ROUNDS} 轮），执行首次聚类...")
            print(f"  [LR] 学习率将从 {LR_WARMUP} 衰减至 {LR_WARMUP * LR_DECAY_FACTOR}（微调模式）")

            # 冻结 Global Anchor Model（用于后续重聚类时的模型融合）
            global_anchor = copy.deepcopy(global_model)
            print(f"  [Anchor] Global Anchor Model 已冻结（Round {round_idx + 1}）")

            client_cluster, _, client_posteriors = run_gmm_clustering(
                local_weights, NUM_CLIENTS, NUM_CLUSTERS,
                round_idx, run_dir, cluster_log,
                prev_assignments=None,
            )
            last_cluster_round = round_idx

        # ================================================================
        # 阶段 D：分簇期 → 簇内聚合 + 全局聚合 + 动态重聚类
        # ================================================================
        if not is_warmup and client_cluster is not None:

            # ---- D1: 检查是否触发动态重聚类 ----
            rounds_since_last = round_idx - last_cluster_round
            should_recluster  = (
                RECLUSTER_INTERVAL > 0
                and rounds_since_last >= RECLUSTER_INTERVAL
                and round_idx > WARMUP_ROUNDS - 1
            )

            if should_recluster:
                print(f"\n  [GMM] 触发动态重聚类（距上次聚类已过 {rounds_since_last} 轮）...")
                new_assignments, changed, new_posteriors = run_gmm_clustering(
                    local_weights, NUM_CLIENTS, NUM_CLUSTERS,
                    round_idx, run_dir, cluster_log,
                    prev_assignments=client_cluster,
                )
                if changed:
                    # 找出所有发生迁移的客户端
                    migrated_clients = []
                    for i in range(NUM_CLIENTS):
                        if new_assignments[i] != client_cluster[i]:
                            migrated_clients.append(i)
                            old_k = client_cluster[i]
                            new_k = new_assignments[i]
                            # 迁移客户端模型 = 70% 新簇模型 + 30% Global Anchor
                            print(f"  [迁移] Client {i}: Cluster {old_k} → Cluster {new_k}")
                            print(f"         模型融合: {RECLUSTER_CLUSTER_WEIGHT:.0%} Cluster {new_k} 模型 "
                                  f"+ {1-RECLUSTER_CLUSTER_WEIGHT:.0%} Global Anchor")
                    # 对有新成员迁入的簇，用新簇模型和 Anchor 做融合
                    changed_clusters = set(new_assignments[i] for i in migrated_clients)
                    for k in changed_clusters:
                        interpolate_models(
                            cluster_models[k],
                            cluster_models[k],
                            global_anchor,
                            RECLUSTER_CLUSTER_WEIGHT,
                        )
                        print(f"  [融合] Cluster {k} 模型已更新: "
                              f"{RECLUSTER_CLUSTER_WEIGHT:.0%} 簇模型 + "
                              f"{1-RECLUSTER_CLUSTER_WEIGHT:.0%} Global Anchor")

                client_cluster     = new_assignments
                client_posteriors  = new_posteriors
                last_cluster_round = round_idx

            # ---- D2: 簇内软加权聚合 ----
            cluster_total_lens = []
            for k in range(NUM_CLUSTERS):
                soft_lens = [client_posteriors[i, k] * local_lens[i]
                             for i in range(NUM_CLIENTS)]
                total_soft = sum(soft_lens)
                if total_soft < 1e-8:
                    cluster_total_lens.append(0)
                    continue

                new_sd = copy.deepcopy(local_weights[0])
                for key in new_sd:
                    new_sd[key] = sum(
                        local_weights[i][key] * (soft_lens[i] / total_soft)
                        for i in range(NUM_CLIENTS)
                    )
                cluster_models[k].load_state_dict(new_sd)
                cluster_total_lens.append(total_soft)

            # ---- D3: 各簇模型 → 全局模型 ----
            active_weights = [cluster_models[k].state_dict()
                              for k in range(NUM_CLUSTERS) if cluster_total_lens[k] > 1e-8]
            active_lens    = [cluster_total_lens[k]
                              for k in range(NUM_CLUSTERS) if cluster_total_lens[k] > 1e-8]
            if active_weights:
                global_model = fedavg(global_model, active_weights, active_lens)

        # ================================================================
        # 阶段 D+: 簇内平均距离
        # ================================================================
        _feats = np.stack([extract_bn_feature(w) for w in local_weights], axis=0)
        _scaler = StandardScaler()
        _feats_scaled = _scaler.fit_transform(_feats)

        if client_cluster is not None and client_posteriors is not None:
            cluster_intra = {}
            for k in range(NUM_CLUSTERS):
                weights_k = client_posteriors[:, k]
                if weights_k.sum() < 1e-8:
                    continue
                center_k = np.average(_feats_scaled, axis=0, weights=weights_k)
                dists_k = np.linalg.norm(_feats_scaled - center_k, axis=1)
                intra_k = float(np.average(dists_k, weights=weights_k))
                cluster_intra[k] = intra_k
            overall_intra = float(np.mean(list(cluster_intra.values()))) if cluster_intra else None
        else:
            center_all = _feats_scaled.mean(axis=0)
            dists_all  = np.linalg.norm(_feats_scaled - center_all, axis=1)
            cluster_intra  = {0: float(dists_all.mean())}
            overall_intra  = float(dists_all.mean())

        intra_dist_history.append({
            'round':         round_idx + 1,
            'phase':         'warmup' if is_warmup else 'clustered',
            'overall_intra': overall_intra,
            'per_cluster':   cluster_intra,
        })

        del local_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ================================================================
        # 阶段 E：验证（全局 + 个性化）
        # ================================================================
        round_time  = time.time() - round_start
        total_time += round_time

        do_eval = ((round_idx + 1) % EVAL_EVERY == 0) or (round_idx + 1 == NUM_ROUNDS)

        pers_miou_this_round = 0.0
        pers_pa_this_round   = 0.0

        if do_eval:
            # ---- E1: 各簇模型在全局验证集上的表现 ----
            for k in range(NUM_CLUSTERS):
                if client_cluster is not None and client_posteriors is not None:
                    members   = [i for i, c in enumerate(client_cluster) if c == k]
                    n_samples = float(sum(client_posteriors[i, k] * local_lens[i]
                                         for i in range(NUM_CLIENTS)))
                    k_loss    = float(np.average([local_losses[i] for i in range(NUM_CLIENTS)],
                                                weights=[client_posteriors[i, k] for i in range(NUM_CLIENTS)]))
                else:
                    members   = list(range(NUM_CLIENTS))
                    n_samples = sum(local_lens)
                    k_loss    = avg_loss

                pa, miou = evaluate_model(cluster_models[k], val_loader, device, use_amp=USE_AMP)
                cluster_history.append({
                    'round':       round_idx + 1,
                    'phase':       'warmup' if is_warmup else 'clustered',
                    'cluster':     k,
                    'num_clients': len(members),
                    'num_samples': n_samples,
                    'pixel_acc':   pa,
                    'miou':        miou,
                    'avg_loss':    k_loss,
                })
                member_str = str(members) if client_cluster is not None else 'all(warmup)'
                print(f"  [Cluster {k}] Pixel Acc: {pa:.4f} | mIoU: {miou:.4f} | 主簇成员: {member_str}")

            # ---- E2: 全局模型在全局验证集上的表现 ----
            g_pa, g_miou = evaluate_model(global_model, val_loader, device, use_amp=USE_AMP)
            print(f"  [Global]    Pixel Acc: {g_pa:.4f} | mIoU: {g_miou:.4f}", end="")

            # ---- E3: 个性化评估（分簇期才执行）----
            if client_cluster is not None:
                pers_miou_this_round, pers_pa_this_round, per_client = \
                    evaluate_personalized(
                        cluster_models, client_cluster, client_posteriors,
                        val_dataset, val_user_groups, device,
                        num_clients=NUM_CLIENTS, num_clusters=NUM_CLUSTERS,
                        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                        pin_memory=PIN_MEMORY, use_amp=USE_AMP,
                    )
                print(f" | Pers mIoU: {pers_miou_this_round:.4f}", end="")

                # 记录每个客户端的详细结果
                for r in per_client:
                    pers_history.append({
                        'round':           round_idx + 1,
                        'client':          r['client'],
                        'cluster':         r['cluster'],
                        'num_val_samples': r['num_val_samples'],
                        'pixel_acc':       r['pixel_acc'],
                        'miou':            r['miou'],
                    })

                # 打印每个客户端的个性化结果
                print()
                for r in per_client:
                    print(f"    Client {r['client']} (Cluster {r['cluster']}, "
                          f"{r['num_val_samples']} val samples): "
                          f"PA={r['pixel_acc']:.4f} | mIoU={r['miou']:.4f}")

            global_history.append({
                'round':     round_idx + 1,
                'phase':     'warmup' if is_warmup else 'clustered',
                'pixel_acc': g_pa,
                'miou':      g_miou,
                'pers_miou': pers_miou_this_round,
                'pers_pa':   pers_pa_this_round,
                'avg_loss':  avg_loss,
                'lr':        current_lr,
                'time':      round_time,
            })

            print(f"  耗时: {round_time:.1f}s", end="")
        else:
            g_pa, g_miou = 0.0, 0.0
            print(f"  [跳过验证] 平均 Loss: {avg_loss:.4f} | 耗时: {round_time:.1f}s", end="")

        # ---- 保存最优模型 ----
        if g_miou > best_miou:
            best_miou = g_miou
            torch.save({
                'round':            round_idx + 1,
                'model_state_dict': global_model.state_dict(),
                'num_classes':      NUM_CLASSES,
                'pixel_acc':        g_pa,
                'miou':             g_miou,
                'dataset':          'cityscapes',
                'target_size':      TARGET_SIZE,
            }, os.path.join(save_dir, 'best_model_cityscapes.pth'))
            print(f"  ★ Best Global mIoU: {g_miou:.4f}")
        else:
            print()

        if pers_miou_this_round > best_pers_miou:
            best_pers_miou = pers_miou_this_round
            print(f"  ★ Best Personalized mIoU: {pers_miou_this_round:.4f}")

        if (round_idx + 1) % 10 == 0:
            ckpt = os.path.join(save_dir, f'global_model_cityscapes_round_{round_idx + 1}.pth')
            torch.save({
                'round':            round_idx + 1,
                'model_state_dict': global_model.state_dict(),
                'num_classes':      NUM_CLASSES,
                'pixel_acc':        g_pa,
                'miou':             g_miou,
                'pers_miou':        pers_miou_this_round,
                'dataset':          'cityscapes',
                'target_size':      TARGET_SIZE,
            }, ckpt)
            print(f"  >> 检查点已保存: {ckpt}")

        # 每轮实时更新 CSV
        save_cluster_csv(cluster_history, run_dir)
        save_global_csv(global_history, run_dir)
        if pers_history:
            save_personalized_csv(pers_history, run_dir)

    # ===== 保存最终全局模型 =====
    final_path = os.path.join(save_dir, 'global_model_cityscapes_final.pth')
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'num_classes':      NUM_CLASSES,
        'num_rounds':       NUM_ROUNDS,
        'final_pixel_acc':  global_history[-1]['pixel_acc'],
        'final_miou':       global_history[-1]['miou'],
        'final_pers_miou':  global_history[-1].get('pers_miou', 0.0),
        'dataset':          'cityscapes',
        'target_size':      TARGET_SIZE,
    }, final_path)

    # ===== 生成折线图 =====
    print(f"\n{'='*60}")
    print(f"正在生成折线图...")
    save_curves(cluster_history, global_history, NUM_CLUSTERS, WARMUP_ROUNDS, run_dir)
    save_intra_dist_curve(intra_dist_history, NUM_CLUSTERS, WARMUP_ROUNDS, run_dir)

    # ===== 打印训练总结 =====
    print(f"\n{'='*80}")
    print(f"训练完成！总结如下（Cityscapes 极速验证 v2）：")
    print(f"{'='*80}")

    if torch.cuda.is_available():
        peak_mem  = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        print(f"\n[GPU 显存总结]")
        print(f"  GPU 型号:     {torch.cuda.get_device_name(device)}")
        print(f"  总显存:       {total_mem:.0f} MB")
        print(f"  训练峰值显存: {peak_mem:.0f} MB ({peak_mem/total_mem*100:.1f}%)")

    print(f"\n[训练性能总结]")
    print(f"  总训练时间:       {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  平均每轮耗时:     {total_time/NUM_ROUNDS:.1f}s")
    print(f"  最优全局 mIoU:    {best_miou:.4f}")
    print(f"  最优个性化 mIoU:  {best_pers_miou:.4f}")
    print(f"  GMM 聚类次数:     {len(cluster_log)} 次")

    print(f"\n[v2 配置回顾]")
    print(f"  分辨率: {TARGET_SIZE[0]}×{TARGET_SIZE[1]} | 客户端: {NUM_CLIENTS} | 簇数: {NUM_CLUSTERS}")
    print(f"  热身: {WARMUP_ROUNDS} 轮 (LR={LR_WARMUP}) | 分簇: {NUM_ROUNDS - WARMUP_ROUNDS} 轮 (LR={LR_WARMUP * LR_DECAY_FACTOR})")
    print(f"  损失函数: Focal Loss (γ={FOCAL_GAMMA})")
    print(f"  数据增强: RandomHFlip + ColorJitter + RandomScaleCrop")
    print(f"  重聚类: F={RECLUSTER_INTERVAL} | 融合: {RECLUSTER_CLUSTER_WEIGHT:.0%} 新簇 + {1-RECLUSTER_CLUSTER_WEIGHT:.0%} Anchor")
    print(f"  评估: 全局 mIoU + 个性化 mIoU（Non-IID 本地测试集）")

    print(f"\n{'Round':<8} {'Phase':<12} {'LR':<10} {'Loss':<10} {'G-PA':<10} {'G-mIoU':<10} {'P-mIoU':<10} {'Time':<8}")
    print(f"{'-'*78}")
    for h in global_history:
        print(f"{h['round']:<8} {h['phase']:<12} {h.get('lr', 0):<10.1e} {h['avg_loss']:<10.4f} "
              f"{h['pixel_acc']:<10.4f} {h['miou']:<10.4f} "
              f"{h.get('pers_miou', 0):<10.4f} {h['time']:<8.1f}")
    print(f"{'-'*78}")

    print(f"\n本次运行所有结果已保存至: {run_dir}/")
    print(f"  ├── gmm_cluster_log.json           ← 每次聚类的详细记录")
    print(f"  ├── cluster_val_results.csv        ← 每轮每簇验证数据（全局验证集）")
    print(f"  ├── global_val_results.csv         ← 每轮全局模型验证数据")
    print(f"  ├── personalized_val_results.csv   ← 每轮个性化 mIoU（本地测试集）")
    print(f"  ├── pixel_accuracy.png             ← Pixel Accuracy 折线图")
    print(f"  ├── miou.png                       ← mIoU 折线图（含个性化曲线）")
    print(f"  └── intra_cluster_distance.png     ← 簇内平均距离折线图")
    print(f"\n最优模型: {os.path.join(save_dir, 'best_model_cityscapes.pth')}")
    print(f"最终模型: {final_path}")


if __name__ == "__main__":
    main()
