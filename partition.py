"""
partition.py — 联邦学习数据划分工具

提供 Dirichlet Non-IID 划分函数，供 Fedavg.py / fedavgTest.py / FedGMHC.py 共用。

Dirichlet 划分原理
------------------
1. 为每张训练图像提取"主类别"（占像素数最多的语义类别）。
2. 按主类别将图像分组。
3. 对每个客户端，从 Dirichlet(α) 分布采样各类别的数据比例，
   再按比例从各类别中抽取图像索引，组成该客户端的本地数据集。
4. 若某客户端分配到的图像数量低于 min_samples，则从其他客户端
   随机借调图像，保证每个客户端都有足够的训练数据。

参数说明
--------
alpha        : Dirichlet 浓度参数，越小异质性越强
               推荐：0.5（强异质）、1.0（中等）、2.0（弱异质）
min_samples  : 每个客户端最少图像数量，防止极端稀少
"""

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os


def get_dominant_class(mask_path, num_classes, target_size=(256, 256)):
    """
    读取单张分割掩码，返回占像素数最多的类别索引。
    使用 NEAREST 插值缩放，避免引入不存在的颜色值。
    """
    from dataset import rgb_mask_to_class_index
    mask = Image.open(mask_path).convert("RGB")
    if target_size is not None:
        mask = mask.resize((target_size[1], target_size[0]), Image.NEAREST)
    class_mask = rgb_mask_to_class_index(mask)
    counts = np.bincount(class_mask.ravel(), minlength=num_classes)
    return int(np.argmax(counts))


def build_label_index(dataset_root, split, num_classes,
                      target_size=(256, 256), cache=True):
    """
    为训练集中的每张图像计算主类别标签，返回 ndarray (N,)。
    结果缓存到 {dataset_root}/partition_cache_{split}.npy，避免重复计算。

    参数
    ----
    dataset_root : 数据集根目录
    split        : 'train' / 'val'
    num_classes  : 类别总数
    target_size  : 掩码缩放尺寸，与训练时保持一致
    cache        : 是否启用缓存
    """
    cache_path = os.path.join(dataset_root, f'partition_cache_{split}.npy')
    if cache and os.path.exists(cache_path):
        labels = np.load(cache_path)
        print(f"  [Partition] 加载主类别缓存: {cache_path}  ({len(labels)} 张)")
        return labels

    mask_dir = os.path.join(dataset_root, f'{split}_labels')
    mask_files = sorted(os.listdir(mask_dir))
    print(f"  [Partition] 正在提取 {len(mask_files)} 张掩码的主类别（首次运行，稍候）...")

    labels = []
    for fname in mask_files:
        path = os.path.join(mask_dir, fname)
        dom = get_dominant_class(path, num_classes, target_size)
        labels.append(dom)
    labels = np.array(labels, dtype=np.int64)

    if cache:
        np.save(cache_path, labels)
        print(f"  [Partition] 主类别缓存已保存: {cache_path}")

    return labels


def dirichlet_partition(num_clients, labels, num_classes,
                        alpha=1.0, min_samples=20,
                        seed=42, max_retry=200):
    """
    基于 Dirichlet 分布的 Non-IID 数据划分。

    参数
    ----
    num_clients  : 客户端数量
    labels       : 每张图像的主类别标签，ndarray (N,)
    num_classes  : 类别总数
    alpha        : Dirichlet 浓度参数（越小异质性越强）
    min_samples  : 每个客户端最少图像数量
    seed         : 随机种子，保证实验可复现
    max_retry    : 若某次划分不满足 min_samples，最多重试次数

    返回
    ----
    user_groups  : list of ndarray，每个元素是一个客户端的图像索引数组
    """
    rng = np.random.default_rng(seed)
    N   = len(labels)

    # 按类别建立图像索引列表
    class_indices = [np.where(labels == c)[0].tolist() for c in range(num_classes)]
    # 打乱各类别内部顺序
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    for attempt in range(max_retry):
        user_groups = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            cls_idx = class_indices[c]
            if len(cls_idx) == 0:
                continue

            # 从 Dirichlet 分布采样各客户端对该类别的比例
            proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))

            # 按比例分配该类别图像
            proportions = np.array(proportions)
            proportions = proportions / proportions.sum()  # 归一化
            splits = (proportions * len(cls_idx)).astype(int)

            # 修正舍入误差，确保总数等于该类别图像数
            diff = len(cls_idx) - splits.sum()
            if diff > 0:
                # 将余量分给比例最大的客户端
                top_idx = np.argsort(proportions)[::-1][:diff]
                splits[top_idx] += 1
            elif diff < 0:
                # 从分配最多的客户端减去
                top_idx = np.argsort(splits)[::-1][:-diff]
                splits[top_idx] -= 1
                splits = np.clip(splits, 0, None)

            # 将图像索引分配给各客户端
            ptr = 0
            for i in range(num_clients):
                user_groups[i].extend(cls_idx[ptr: ptr + splits[i]])
                ptr += splits[i]

        # 检查是否满足最小数据量要求
        sizes = [len(g) for g in user_groups]
        if min(sizes) >= min_samples:
            break

        # 不满足：从数据量最多的客户端借调图像给数据量最少的客户端
        for i in range(num_clients):
            while len(user_groups[i]) < min_samples:
                donor = int(np.argmax([len(g) for g in user_groups]))
                if donor == i or len(user_groups[donor]) <= min_samples:
                    break
                # 随机借调一张
                borrow_pos = rng.integers(0, len(user_groups[donor]))
                idx = user_groups[donor].pop(borrow_pos)
                user_groups[i].append(idx)

        sizes = [len(g) for g in user_groups]
        if min(sizes) >= min_samples:
            break

    # 转为 ndarray
    user_groups = [np.array(g, dtype=np.int64) for g in user_groups]

    return user_groups


def print_partition_stats(user_groups, labels, num_classes, class_names=None):
    """
    打印数据划分统计信息：每个客户端的数据量和类别分布。
    """
    num_clients = len(user_groups)
    print(f"\n  [Partition] Non-IID 数据划分统计（共 {num_clients} 个客户端）:")
    print(f"  {'Client':<10} {'Samples':<10} {'主类别分布（前3）'}")
    print(f"  {'-'*55}")

    for i, group in enumerate(user_groups):
        if len(group) == 0:
            print(f"  Client {i:<5} {'0':<10} (空)")
            continue
        client_labels = labels[group]
        counts = np.bincount(client_labels, minlength=num_classes)
        # 取前3个主要类别
        top3 = np.argsort(counts)[::-1][:3]
        top3_str = ', '.join(
            f"{class_names[c] if class_names else c}:{counts[c]}"
            for c in top3 if counts[c] > 0
        )
        print(f"  Client {i:<5} {len(group):<10} {top3_str}")

    sizes = [len(g) for g in user_groups]
    print(f"  {'-'*55}")
    print(f"  总计: {sum(sizes)} 张 | 最多: {max(sizes)} | 最少: {min(sizes)} | "
          f"均值: {np.mean(sizes):.1f} | 标准差: {np.std(sizes):.1f}")
