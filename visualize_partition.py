"""
visualize_partition.py
======================
可视化 Dirichlet 划分后各客户端的数据集类别分布。

生成以下图表：
  1. partition_heatmap.png       — 热力图（客户端 × 类别，颜色=样本比例）
  2. partition_stacked_bar.png   — 堆叠条形图（每客户端类别构成）
  3. partition_sample_count.png  — 各客户端样本总量柱状图

用法：
  python visualize_partition.py
  （修改下方 DATASET_ROOT、NUM_CLIENTS、DIRICHLET_ALPHA 即可）
"""

import os
import sys

# ===== 路径修复（无论从哪个目录运行都能找到 dataset/、partition.py）=====
_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from dataset.cityscapes_dataset import (
    NUM_CLASSES, CLASS_NAMES, IGNORE_INDEX,
    build_label_index_cityscapes,
)
from partition import dirichlet_partition

# ==================== 配置区（按需修改）====================
DATASET_ROOT    = r'E:\Autonomous Driving Dataset\Cityscapes dataset(10g)'
NUM_CLIENTS     = 20       # 客户端数量
DIRICHLET_ALPHA = 0.5      # Dirichlet 浓度参数（越小异质性越强）
MIN_SAMPLES     = 30       # 每客户端最少样本数
MAX_SAMPLES     = None     # None = 不截断；整数 = 每客户端最多样本数
SPLIT           = 'train'  # 使用训练集
RANDOM_SEED     = 42
OUTPUT_DIR      = './partition_vis'   # 图表保存目录
# ===========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_class_distribution(user_groups, labels, num_clients, num_classes):
    """
    计算每个客户端的类别样本数矩阵。

    Returns
    -------
    count_matrix : ndarray, shape (num_clients, num_classes)
        count_matrix[i, c] = 客户端 i 中主类别为 c 的样本数
    ratio_matrix : ndarray, shape (num_clients, num_classes)
        count_matrix 按行归一化（每客户端内的类别比例）
    """
    count_matrix = np.zeros((num_clients, num_classes), dtype=np.int64)
    for i, indices in enumerate(user_groups):
        for idx in indices:
            c = int(labels[idx])
            if 0 <= c < num_classes:
                count_matrix[i, c] += 1
    # 行归一化（避免除零）
    row_sum = count_matrix.sum(axis=1, keepdims=True).astype(float)
    row_sum[row_sum == 0] = 1.0
    ratio_matrix = count_matrix / row_sum
    return count_matrix, ratio_matrix


def plot_heatmap(ratio_matrix, class_names, num_clients, alpha, output_dir):
    """热力图：横轴类别，纵轴客户端，颜色=该类别在该客户端中的样本比例。"""
    n_clients, n_classes = ratio_matrix.shape

    # 自定义颜色映射：白 → 深蓝
    cmap = LinearSegmentedColormap.from_list(
        'white_blue', ['#ffffff', '#1a5fa8'], N=256
    )

    fig_w = max(14, n_classes * 0.7)
    fig_h = max(6,  n_clients * 0.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(ratio_matrix, aspect='auto', cmap=cmap, vmin=0, vmax=ratio_matrix.max())

    # 坐标轴
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_clients))
    ax.set_yticklabels([f'Client {i}' for i in range(n_clients)], fontsize=9)

    ax.set_xlabel('Semantic Class', fontsize=12)
    ax.set_ylabel('Client', fontsize=12)
    ax.set_title(
        f'Data Heterogeneity Distribution (Dirichlet α={alpha}, {n_clients} Clients)\n'
        f'Color intensity = proportion of samples with that dominant class',
        fontsize=13
    )

    # 色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Sample Proportion', fontsize=10)

    # 在格子内标注比例值（仅标注 > 0.05 的格子，避免拥挤）
    for i in range(n_clients):
        for j in range(n_classes):
            val = ratio_matrix[i, j]
            if val > 0.05:
                text_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=6.5, color=text_color)

    fig.tight_layout()
    save_path = os.path.join(output_dir, 'partition_heatmap.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [保存] {save_path}')


def plot_stacked_bar(ratio_matrix, class_names, num_clients, alpha, output_dir):
    """堆叠条形图：每个客户端一个条，按类别比例分色堆叠。"""
    n_clients, n_classes = ratio_matrix.shape
    colors = plt.cm.tab20.colors  # 20 种颜色

    fig_w = max(12, n_clients * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, 6))

    bottoms = np.zeros(n_clients)
    for c in range(n_classes):
        vals = ratio_matrix[:, c]
        ax.bar(range(n_clients), vals, bottom=bottoms,
               color=colors[c % len(colors)], label=class_names[c],
               width=0.8, edgecolor='white', linewidth=0.3)
        bottoms += vals

    ax.set_xticks(range(n_clients))
    ax.set_xticklabels([f'C{i}' for i in range(n_clients)], fontsize=9)
    ax.set_xlabel('Client', fontsize=12)
    ax.set_ylabel('Class Proportion', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title(
        f'Per-client Class Distribution (Dirichlet α={alpha}, {n_clients} Clients)',
        fontsize=13
    )
    # 图例放在图外右侧
    ax.legend(handles=ax.patches[:n_classes],
              labels=class_names,
              loc='upper left', bbox_to_anchor=(1.01, 1),
              fontsize=8, ncol=1)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()

    save_path = os.path.join(output_dir, 'partition_stacked_bar.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [保存] {save_path}')


def plot_sample_count(count_matrix, num_clients, alpha, output_dir):
    """各客户端样本总量柱状图。"""
    total_counts = count_matrix.sum(axis=1)
    mean_count   = total_counts.mean()

    fig, ax = plt.subplots(figsize=(max(10, num_clients * 0.55), 5))
    bars = ax.bar(range(num_clients), total_counts,
                  color='#4c8fbd', edgecolor='white', linewidth=0.5)
    ax.axhline(mean_count, color='red', linestyle='--', linewidth=1.5,
               label=f'Mean = {mean_count:.0f}')

    # 在柱顶标注数量
    for bar, cnt in zip(bars, total_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(int(cnt)), ha='center', va='bottom', fontsize=8)

    ax.set_xticks(range(num_clients))
    ax.set_xticklabels([f'C{i}' for i in range(num_clients)], fontsize=9)
    ax.set_xlabel('Client', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title(
        f'Per-client Sample Count (Dirichlet α={alpha}, {n_clients} Clients)',
        fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()

    save_path = os.path.join(output_dir, 'partition_sample_count.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [保存] {save_path}')


# ==================== 主流程 ====================

if __name__ == '__main__':
    print('=' * 60)
    print('Cityscapes Dirichlet Partition Visualization')
    print('=' * 60)
    print(f'  数据集路径:    {DATASET_ROOT}')
    print(f'  客户端数量:    {NUM_CLIENTS}')
    print(f'  Dirichlet α:  {DIRICHLET_ALPHA}')
    print(f'  输出目录:      {OUTPUT_DIR}')
    print()

    # Step 1: 构建每张图像的主类别标签
    print('[1/3] 构建主类别标签索引（首次运行较慢，结果会缓存）...')
    labels = build_label_index_cityscapes(
        dataset_root=DATASET_ROOT,
        split=SPLIT,
        num_classes=NUM_CLASSES,
    )
    print(f'  共 {len(labels)} 张训练图像')

    # Step 2: Dirichlet 划分
    print(f'\n[2/3] 执行 Dirichlet 划分（α={DIRICHLET_ALPHA}）...')
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    user_groups = dirichlet_partition(
        num_clients=NUM_CLIENTS,
        labels=labels,
        num_classes=NUM_CLASSES,
        alpha=DIRICHLET_ALPHA,
        min_samples=MIN_SAMPLES,
        random_seed=RANDOM_SEED,
    )
    # 可选截断
    if MAX_SAMPLES is not None:
        import random as _random
        for i in range(len(user_groups)):
            if len(user_groups[i]) > MAX_SAMPLES:
                user_groups[i] = _random.sample(list(user_groups[i]), MAX_SAMPLES)

    total_samples = sum(len(g) for g in user_groups)
    print(f'  划分完成，共 {total_samples} 个样本分配给 {NUM_CLIENTS} 个客户端')
    for i, g in enumerate(user_groups):
        print(f'    Client {i:2d}: {len(g):4d} 张')

    # Step 3: 计算类别分布矩阵
    print(f'\n[3/3] 生成可视化图表...')
    n_clients = NUM_CLIENTS
    count_matrix, ratio_matrix = build_class_distribution(
        user_groups, labels, NUM_CLIENTS, NUM_CLASSES
    )

    plot_heatmap(ratio_matrix, CLASS_NAMES, NUM_CLIENTS, DIRICHLET_ALPHA, OUTPUT_DIR)
    plot_stacked_bar(ratio_matrix, CLASS_NAMES, NUM_CLIENTS, DIRICHLET_ALPHA, OUTPUT_DIR)
    plot_sample_count(count_matrix, NUM_CLIENTS, DIRICHLET_ALPHA, OUTPUT_DIR)

    print(f'\n完成！所有图表已保存至: {OUTPUT_DIR}/')
    print(f'  ├── partition_heatmap.png       ← 热力图（论文首选）')
    print(f'  ├── partition_stacked_bar.png   ← 堆叠条形图')
    print(f'  └── partition_sample_count.png  ← 各客户端样本量')
