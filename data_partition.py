"""
联邦学习 Non-IID 数据划分模块

提供 3 种异质性数据划分策略：
1. Dirichlet 分布划分（推荐，可连续控制异质程度）
2. 按类别数量限制划分（每个客户端只拥有指定数量的类别）
3. IID 均匀划分（基线对照）

用法：
    from data_partition import partition_data

    # Dirichlet 划分，alpha 越小异质性越强
    user_groups = partition_data(dataset, num_clients=5, method='dirichlet', alpha=0.5)

    # 每个客户端只有 3 个类别
    user_groups = partition_data(dataset, num_clients=5, method='class_limit', max_classes=3)

    # IID 均匀划分
    user_groups = partition_data(dataset, num_clients=5, method='iid')
"""

import numpy as np
from collections import defaultdict


def get_image_dominant_class(dataset, idx, num_classes):
    """
    获取一张语义分割图片的主要类别分布。
    返回每个类别的像素占比向量。
    """
    _, mask = dataset[idx]
    mask_np = mask.numpy() if hasattr(mask, 'numpy') else np.array(mask)
    total_pixels = mask_np.size
    class_ratios = np.zeros(num_classes)
    for c in range(num_classes):
        class_ratios[c] = (mask_np == c).sum() / total_pixels
    return class_ratios


def get_image_primary_label(dataset, idx, num_classes, ignore_class=0):
    """
    获取一张图片的"主要类别"（像素占比最大的非背景类别）。
    ignore_class: 忽略的背景类别索引（默认为 0）。
    """
    class_ratios = get_image_dominant_class(dataset, idx, num_classes)
    # 将背景类别的占比设为 0，避免所有图片都被归为背景
    class_ratios[ignore_class] = 0
    if class_ratios.sum() == 0:
        # 如果图片全是背景，返回背景类
        return ignore_class
    return np.argmax(class_ratios)


def build_class_index(dataset, num_classes, ignore_class=0):
    """
    为数据集建立 类别 -> 图片索引 的映射表。
    每张图片按其主要类别归类。
    """
    class_to_indices = defaultdict(list)
    print("正在分析数据集中每张图片的类别分布...")
    for idx in range(len(dataset)):
        primary_label = get_image_primary_label(dataset, idx, num_classes, ignore_class)
        class_to_indices[primary_label].append(idx)

    print(f"类别分布统计:")
    for c in sorted(class_to_indices.keys()):
        print(f"  类别 {c}: {len(class_to_indices[c])} 张图片")

    return class_to_indices


# ==================== 划分策略 ====================

def partition_dirichlet(dataset, num_clients, num_classes, alpha=0.5, ignore_class=0, seed=42):
    """
    Dirichlet 分布划分（最常用的 Non-IID 划分方法）

    参数：
        dataset:      数据集对象
        num_clients:  客户端数量
        num_classes:  类别总数
        alpha:        Dirichlet 浓度参数，控制异质程度
                      alpha=0.1 → 极端 Non-IID（每个客户端几乎只有 1-2 个类别）
                      alpha=0.5 → 中度 Non-IID
                      alpha=1.0 → 轻度 Non-IID
                      alpha=100 → 接近 IID
        ignore_class: 忽略的背景类别索引
        seed:         随机种子

    返回：
        list[np.array]: 每个客户端的数据索引列表
    """
    np.random.seed(seed)
    class_to_indices = build_class_index(dataset, num_classes, ignore_class)

    # 为每个客户端分配数据
    client_indices = [[] for _ in range(num_clients)]

    for c in class_to_indices:
        indices = np.array(class_to_indices[c])
        np.random.shuffle(indices)

        # 从 Dirichlet 分布采样每个客户端分到该类别数据的比例
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # 按比例分配
        proportions = (proportions * len(indices)).astype(int)
        # 将余数分给第一个客户端
        proportions[0] += len(indices) - proportions.sum()

        start = 0
        for i in range(num_clients):
            client_indices[i].extend(indices[start:start + proportions[i]].tolist())
            start += proportions[i]

    # 转换为 numpy 数组并打乱每个客户端内部顺序
    user_groups = []
    for i in range(num_clients):
        arr = np.array(client_indices[i])
        np.random.shuffle(arr)
        user_groups.append(arr)

    return user_groups


def partition_class_limit(dataset, num_clients, num_classes, max_classes=2, ignore_class=0, seed=42):
    """
    按类别数量限制划分（每个客户端只拥有指定数量的类别）

    参数：
        dataset:      数据集对象
        num_clients:  客户端数量
        num_classes:  类别总数
        max_classes:  每个客户端最多拥有的类别数量
                      max_classes=1 → 极端 Non-IID
                      max_classes=2 → 强 Non-IID
                      max_classes=3 → 中度 Non-IID
        ignore_class: 忽略的背景类别索引
        seed:         随机种子

    返回：
        list[np.array]: 每个客户端的数据索引列表
    """
    np.random.seed(seed)
    class_to_indices = build_class_index(dataset, num_classes, ignore_class)

    # 获取有数据的类别列表（排除背景类）
    available_classes = [c for c in class_to_indices if c != ignore_class and len(class_to_indices[c]) > 0]

    # 为每个客户端随机分配 max_classes 个类别
    client_indices = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        chosen_classes = np.random.choice(available_classes, size=min(max_classes, len(available_classes)), replace=False)

        for c in chosen_classes:
            indices = class_to_indices[c]
            # 将该类别的数据均分给选择了该类别的客户端
            # 简化处理：每个客户端获取该类别的一部分数据
            per_client = len(indices) // num_clients
            start = i * per_client
            end = start + per_client if i < num_clients - 1 else len(indices)
            client_indices[i].extend(indices[start:end])

    # 转换为 numpy 数组
    user_groups = []
    for i in range(num_clients):
        arr = np.array(client_indices[i])
        np.random.shuffle(arr)
        user_groups.append(arr)

    return user_groups


def partition_iid(dataset, num_clients, seed=42):
    """
    IID 均匀划分（基线对照）

    参数：
        dataset:      数据集对象
        num_clients:  客户端数量
        seed:         随机种子

    返回：
        list[np.array]: 每个客户端的数据索引列表
    """
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    return [arr for arr in np.array_split(indices, num_clients)]


# ==================== 统一接口 ====================

def partition_data(dataset, num_clients, num_classes=12, method='dirichlet',
                   alpha=0.5, max_classes=2, ignore_class=0, seed=42):
    """
    统一的数据划分接口

    参数：
        dataset:      数据集对象
        num_clients:  客户端数量
        num_classes:  类别总数（默认 12，CamVid）
        method:       划分方法
                      'iid'         → IID 均匀划分
                      'dirichlet'   → Dirichlet 分布划分
                      'class_limit' → 按类别数量限制划分
        alpha:        Dirichlet 参数（仅 method='dirichlet' 时有效）
        max_classes:  每客户端最大类别数（仅 method='class_limit' 时有效）
        ignore_class: 忽略的背景类别索引
        seed:         随机种子

    返回：
        list[np.array]: 每个客户端的数据索引列表
    """
    if method == 'iid':
        user_groups = partition_iid(dataset, num_clients, seed)
    elif method == 'dirichlet':
        user_groups = partition_dirichlet(dataset, num_clients, num_classes, alpha, ignore_class, seed)
    elif method == 'class_limit':
        user_groups = partition_class_limit(dataset, num_clients, num_classes, max_classes, ignore_class, seed)
    else:
        raise ValueError(f"不支持的划分方法: {method}，可选: 'iid', 'dirichlet', 'class_limit'")

    # 打印划分结果
    print(f"\n数据划分结果 (method={method}):")
    print(f"{'客户端':<10} {'数据量':<10} {'类别分布'}")
    print(f"{'-'*60}")
    for i, indices in enumerate(user_groups):
        if len(indices) > 0:
            # 统计该客户端拥有的类别
            labels = []
            for idx in indices:
                primary = get_image_primary_label(dataset, idx, num_classes, ignore_class)
                labels.append(primary)
            unique, counts = np.unique(labels, return_counts=True)
            dist_str = ", ".join([f"C{c}:{n}" for c, n in zip(unique, counts)])
        else:
            dist_str = "无数据"
        print(f"Client {i:<4} {len(indices):<10} {dist_str}")
    print(f"{'-'*60}")

    return user_groups
