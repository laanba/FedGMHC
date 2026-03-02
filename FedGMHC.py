"""
FedGMHC.py — 基于高斯混合模型（GMM）分簇的联邦学习方法

冷启动解决策略（组合方案）
--------------------------
方案一  延迟聚类（热身期）
        前 WARMUP_ROUNDS 轮执行标准 FedAvg，等待模型收敛、BN 统计量
        充分反映各客户端数据分布后，再进行首次 GMM 聚类。

方案二  动态重聚类
        首次聚类后，每隔 RECLUSTER_INTERVAL 轮重新提取 BN 特征并
        重新拟合 GMM，让分簇结果随训练进程自我修正并逐渐稳定。
        每次重聚类时，若分簇结果发生变化，簇模型会从全局模型重新初始化，
        保证客户端切换簇后有一个干净的起点。

算法流程
--------
热身期（Round 1 ~ WARMUP_ROUNDS）
    所有客户端从全局模型出发，完成本地训练，执行标准 FedAvg 聚合。
    每轮结束后，所有簇模型与全局模型保持同步。

首次聚类（Round WARMUP_ROUNDS 结束后）
    提取各客户端 BN 层 running_mean / running_var，
    拼接为特征向量，拟合 GMM（K = NUM_CLUSTERS，对角协方差），
    按后验概率最大值分配各客户端到对应簇。

分簇训练期（Round WARMUP_ROUNDS+1 起）
    每轮：
      a. 每个客户端从所属簇模型出发，完成本地训练。
      b. 同簇客户端按数据量加权 FedAvg，更新各簇模型。
      c. 各簇模型按各簇总数据量加权 FedAvg，更新全局模型。
      d. 若当前轮满足重聚类条件（距上次聚类已过 RECLUSTER_INTERVAL 轮），
         重新提取 BN 特征并更新分簇结果。
      e. 在验证集上分别评估每个簇模型和全局模型，记录结果。

结果保存
--------
每次运行结果统一保存在 result_save/MMDDHHmm/ 子目录下：
  result_save/
  └── MMDDHHmm/
      ├── gmm_cluster_log.json       ← 每次聚类的详细记录（含轮次、分配、后验概率）
      ├── cluster_val_results.csv    ← 每轮每簇验证数据汇总表（实时更新）
      ├── global_val_results.csv     ← 每轮全局模型验证数据汇总表（实时更新）
      ├── pixel_accuracy.png         ← 各簇 + 全局 Pixel Accuracy 折线图
      └── miou.png                   ← 各簇 + 全局 mIoU 折线图
"""

import torch
import torch.nn.functional as F
import copy
import json
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from model import MobileNetV2UNet
from dataset import CamVidDataset, NUM_CLASSES, CLASS_NAMES

import os
import sys
import time
import csv
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ==================== 超参数 ====================
NUM_CLUSTERS        = 3    # GMM 部件数 / 簇数
WARMUP_ROUNDS       = 5    # 热身轮数：前 N 轮执行标准 FedAvg，之后再首次聚类
RECLUSTER_INTERVAL  = 10   # 动态重聚类间隔：每隔 M 轮重新聚类一次（0 = 禁用重聚类）
# PCA 目标维度：降维后的特征维度，需满足 PCA_N_COMPONENTS < min(NUM_CLIENTS, feat_dim)
# 推荐范围：[NUM_CLUSTERS, NUM_CLIENTS - 1]，默认取 NUM_CLIENTS // 2
PCA_N_COMPONENTS    = None  # None = 自动设置为 min(NUM_CLIENTS-1, 8)


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


def auto_batch_size(device, num_data_per_client, base_batch_size=32):
    data_limit = max(8, num_data_per_client // 2)
    if torch.cuda.is_available():
        total     = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        gpu_limit = int((total - 500 - 200) / 25)
        gpu_limit = max(8, gpu_limit)
    else:
        gpu_limit = base_batch_size
    recommended = min(data_limit, gpu_limit)
    power = 1
    while power * 2 <= recommended:
        power *= 2
    return max(4, power)


# ==================== 评估指标 ====================

def compute_pixel_accuracy(pred, target):
    return (pred == target).sum().item() / target.numel()


def compute_iou_per_class(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        ious.append(inter / union if union > 0 else float('nan'))
    return ious


def compute_miou(pred, target, num_classes):
    ious = compute_iou_per_class(pred, target, num_classes)
    valid = [v for v in ious if not np.isnan(v)]
    return float(np.mean(valid)) if valid else 0.0


def evaluate_model(model, val_loader, device, use_amp=True):
    """返回 (avg_pixel_acc, avg_miou)"""
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


# ==================== BN 特征提取 ====================

def extract_bn_feature(state_dict):
    """
    从模型 state_dict 中提取所有 BatchNorm 层的
    running_mean 和 running_var，拼接为一维特征向量（numpy）。
    """
    parts = []
    for key, val in state_dict.items():
        if 'running_mean' in key or 'running_var' in key:
            parts.append(val.cpu().float().numpy().ravel())
    return np.concatenate(parts) if parts else np.array([])


# ==================== GMM 聚类 ====================

def run_gmm_clustering(local_weights, num_clients, n_clusters, round_idx, run_dir,
                       cluster_log, prev_assignments=None):
    """
    提取 BN 特征 → 标准化 → PCA 降维 → 拟合 GMM → 分配客户端到簇。

    高维 BN 特征（维度通常远大于客户端数）直接送入 GMM 会导致后验概率
    退化为 0/1（硬分配），失去概率软分配的意义。
    解决方案：先用 StandardScaler 标准化，再用 PCA 将特征压缩到
    PCA_N_COMPONENTS 维（默认 min(N-1, 8)），使特征维度远小于客户端数，
    从而让 GMM 的高斯分布保持合理的宽度，后验概率恢复为有意义的软分配。

    参数
    ----
    local_weights    : 本轮各客户端训练后的 state_dict 列表
    num_clients      : 客户端总数
    n_clusters       : GMM 部件数
    round_idx        : 当前轮次索引（0-based），用于日志
    run_dir          : 结果保存目录
    cluster_log      : 聚类日志列表（原地追加）
    prev_assignments : 上一次的分配结果，用于检测分配变化

    返回
    ----
    new_assignments  : list[int]，新的客户端分簇结果
    changed          : bool，分配结果是否发生变化
    posteriors       : ndarray (N, K)，后验概率矩阵（基于降维后特征）
    """
    print(f"\n  [GMM] 提取 BN 层统计特征（Round {round_idx + 1}）...")
    features = [extract_bn_feature(w) for w in local_weights]
    feat_dim = features[0].shape[0]
    print(f"  [GMM] 原始特征向量维度: {feat_dim}")

    X = np.stack(features, axis=0)   # (N, D)
    N = X.shape[0]
    effective_k = min(n_clusters, N)

    # ---- Step 1: 标准化（消除不同 BN 层数值尺度差异）----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)   # (N, D)

    # ---- Step 2: PCA 降维 ----
    # 目标维度：需满足 n_components < min(N, D)
    # 推荐设为 [n_clusters, N-1] 之间，默认取 min(N-1, 8)
    n_components = PCA_N_COMPONENTS if PCA_N_COMPONENTS is not None \
        else min(N - 1, 8)
    n_components = max(effective_k, min(n_components, N - 1, feat_dim))

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)   # (N, n_components)

    explained_var = pca.explained_variance_ratio_.sum() * 100
    print(f"  [GMM] PCA 降维: {feat_dim} → {n_components} 维 "
          f"（累计解释方差: {explained_var:.1f}%）")

    # ---- Step 3: 拟合 GMM ----
    gmm = GaussianMixture(
        n_components=effective_k,
        covariance_type='full',   # 降维后维度低，可用 full 协方差
        max_iter=300,
        n_init=10,                # 多次随机初始化，选对数似然最优
        random_state=None,        # 不固定种子，每次重聚类可探索不同解
        reg_covar=1e-4,
    )
    gmm.fit(X_pca)

    posteriors       = gmm.predict_proba(X_pca)       # (N, K)
    new_assignments  = posteriors.argmax(axis=1).tolist()

    # 检测分配变化
    changed = (prev_assignments is None) or (new_assignments != prev_assignments)

    print(f"  [GMM] 客户端分簇结果（{'首次' if prev_assignments is None else '重聚类'}）:")
    for i, k in enumerate(new_assignments):
        prob_str = ', '.join([f'C{j}:{posteriors[i, j]:.3f}'
                              for j in range(posteriors.shape[1])])
        change_tag = ''
        if prev_assignments is not None and new_assignments[i] != prev_assignments[i]:
            change_tag = f'  ← 从 Cluster {prev_assignments[i]} 迁移'
        print(f"    Client {i} → Cluster {k}  ({prob_str}){change_tag}")

    for k in range(effective_k):
        members = [i for i, c in enumerate(new_assignments) if c == k]
        print(f"  Cluster {k}: {members} ({len(members)} 个客户端)")

    if not changed:
        print(f"  [GMM] 分配结果与上次相同，无需重置簇模型。")

    # 追加到聚类日志
    cluster_log.append({
        'round':            round_idx + 1,
        'trigger':          'warmup_end' if prev_assignments is None else 'recluster',
        'feature_dim_raw':  int(feat_dim),
        'feature_dim_pca':  int(n_components),
        'pca_explained_var_pct': round(float(explained_var), 2),
        'assignments':      new_assignments,
        'changed':          changed,
        'posteriors':       posteriors.tolist(),
        'cluster_members':  {
            str(k): [i for i, c in enumerate(new_assignments) if c == k]
            for k in range(effective_k)
        },
    })

    # 实时写入聚类日志 JSON
    log_path = os.path.join(run_dir, 'gmm_cluster_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(cluster_log, f, ensure_ascii=False, indent=2)
    print(f"  [GMM] 聚类日志已更新: {log_path}")

    return new_assignments, changed, posteriors


# ==================== 联邦聚合 ====================

def fedavg(base_model, weights_list, lens_list):
    """按数据量加权平均聚合，结果写入 base_model 并返回。"""
    total = sum(lens_list)
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

    def local_train(self, model, batch_size=16, epochs=1, lr=0.01,
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
        criterion = torch.nn.CrossEntropyLoss()
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
        writer.writerow(['Round', 'Phase', 'Pixel_Accuracy', 'mIoU', 'Avg_Loss', 'Time_s'])
        for r in global_history:
            writer.writerow([
                r['round'], r['phase'],
                f"{r['pixel_acc']:.6f}", f"{r['miou']:.6f}",
                f"{r['avg_loss']:.6f}", f"{r['time']:.1f}",
            ])
    return path


def save_curves(cluster_history, global_history, num_clusters, warmup_rounds, run_dir):
    """
    生成两张折线图：
      1. pixel_accuracy.png — 各簇（虚线）+ 全局（黑色实线）Pixel Accuracy
      2. miou.png           — 各簇（虚线）+ 全局（黑色实线）mIoU
    热身期与分簇期之间用竖虚线分隔。
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

    colors = plt.cm.tab10.colors

    for ylabel, title, suffix, c_key, g_data in [
        ('Pixel Accuracy',
         'Pixel Accuracy per Cluster & Global vs. Round',
         'pixel_accuracy.png', 'pa', global_pa),
        ('mIoU',
         'mIoU per Cluster & Global vs. Round',
         'miou.png', 'miou', global_miou),
    ]:
        plt.figure(figsize=(13, 6))

        # 热身期与分簇期分隔线
        if warmup_rounds > 0 and global_rounds and warmup_rounds < max(global_rounds):
            plt.axvline(x=warmup_rounds + 0.5, color='gray', linestyle=':', linewidth=1.5,
                        label=f'Warmup End (R{warmup_rounds})')

        # 各簇曲线（虚线，仅分簇期有数据）
        for k in range(num_clusters):
            d = cluster_data[k]
            if d['rounds']:
                plt.plot(d['rounds'], d[c_key],
                         linestyle='--', marker='o', linewidth=1.5, markersize=3,
                         color=colors[k % len(colors)],
                         label=f'Cluster {k}')

        # 全局曲线（实线，加粗，贯穿全程）
        if global_rounds:
            plt.plot(global_rounds, g_data,
                     linestyle='-', marker='s', linewidth=2.5, markersize=4,
                     color='black', label='Global')

        plt.xlabel('Communication Round', fontsize=13)
        plt.ylabel(ylabel, fontsize=13)
        plt.title(title, fontsize=15)
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        save_path = os.path.join(run_dir, suffix)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  已保存: {save_path}")


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
    USE_AMP      = True
    TARGET_SIZE  = (256, 256)
    NUM_ROUNDS   = 50
    NUM_CLIENTS  = 10
    LOCAL_EPOCHS = 5
    LR           = 0.01
    NUM_WORKERS  = 0 if sys.platform == 'win32' else 4
    PIN_MEMORY   = True
    BATCH_SIZE   = 0        # 0 = 自动推荐
    # ================================================

    # ===== 时间戳运行目录 =====
    run_timestamp = datetime.now().strftime('%m%d%H%M')
    run_dir = os.path.join('./result_save', f'FedGMHC_{run_timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n本次运行结果将保存至: {run_dir}/")

    # ===== 加载数据集 =====
    train_dataset = CamVidDataset('./data', split='train',
                                  transform=transforms.ToTensor(),
                                  target_size=TARGET_SIZE)
    val_dataset   = CamVidDataset('./data', split='val',
                                  transform=transforms.ToTensor(),
                                  target_size=TARGET_SIZE)

    num_images = len(train_dataset)
    indices    = np.arange(num_images)
    np.random.shuffle(indices)
    user_groups = np.array_split(indices, NUM_CLIENTS)

    min_data = min(len(g) for g in user_groups)
    if BATCH_SIZE == 0:
        BATCH_SIZE = auto_batch_size(device, min_data)

    print(f"\n{'='*65}")
    print(f"训练配置:")
    print(f"  训练集: {num_images} 张 | 验证集: {len(val_dataset)} 张")
    print(f"  客户端: {NUM_CLIENTS} 个 | 簇数: {NUM_CLUSTERS}")
    print(f"  热身轮数: {WARMUP_ROUNDS} | 重聚类间隔: "
          f"{'禁用' if RECLUSTER_INTERVAL == 0 else f'每 {RECLUSTER_INTERVAL} 轮'}")
    print(f"  Batch Size: {BATCH_SIZE} | Local Epochs: {LOCAL_EPOCHS} | 联邦轮数: {NUM_ROUNDS}")
    print(f"  学习率: {LR} | AMP: {'已启用' if USE_AMP and torch.cuda.is_available() else '未启用'}")
    print(f"{'='*65}")

    # ===== 初始化全局模型 =====
    global_model = MobileNetV2UNet(num_classes=NUM_CLASSES).to(device)
    total_params = sum(p.numel() for p in global_model.parameters())
    print(f"\n模型总参数量: {total_params:,} ({total_params * 4 / 1024**2:.1f} MB in FP32)")

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    print_gpu_status(device, "训练前基线")

    # ===== 状态变量 =====
    cluster_models  = [copy.deepcopy(global_model) for _ in range(NUM_CLUSTERS)]
    client_cluster  = None          # 当前分簇结果，None 表示热身期
    last_cluster_round = -1         # 上次执行聚类的轮次索引
    cluster_log     = []            # 每次聚类的详细记录

    cluster_history = []
    global_history  = []
    best_miou       = 0.0
    total_time      = 0.0

    print(f"\n{'='*80}")
    print(f"开始 FedGMHC 训练（热身 {WARMUP_ROUNDS} 轮 + 动态重聚类间隔 "
          f"{'禁用' if RECLUSTER_INTERVAL == 0 else RECLUSTER_INTERVAL} 轮）...")
    print(f"{'='*80}")

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        is_warmup   = (round_idx < WARMUP_ROUNDS)
        phase_label = f'Warmup({round_idx + 1}/{WARMUP_ROUNDS})' if is_warmup else 'Clustered'
        print(f"\n--- Round {round_idx + 1}/{NUM_ROUNDS}  [{phase_label}] ---")

        # ================================================================
        # 阶段 A：每个客户端本地训练
        #   热身期：从全局模型出发
        #   分簇期：从所属簇模型出发
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
                lr=LR,
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
        print(f"  所有客户端本地训练完成 | 平均 Loss: {avg_loss:.4f}")

        # ================================================================
        # 阶段 B：热身期 → 标准 FedAvg 聚合，同步所有簇模型
        # ================================================================
        if is_warmup:
            global_model = fedavg(global_model, local_weights, local_lens)
            # 热身期簇模型始终与全局模型保持同步
            for k in range(NUM_CLUSTERS):
                cluster_models[k].load_state_dict(copy.deepcopy(global_model.state_dict()))

        # ================================================================
        # 阶段 C：热身期结束后 → 首次 GMM 聚类
        # ================================================================
        if round_idx == WARMUP_ROUNDS - 1:
            print(f"\n  [GMM] 热身期结束，执行首次聚类...")
            client_cluster, _, _ = run_gmm_clustering(
                local_weights, NUM_CLIENTS, NUM_CLUSTERS,
                round_idx, run_dir, cluster_log,
                prev_assignments=None,
            )
            last_cluster_round = round_idx

        # ================================================================
        # 阶段 D：分簇期 → 簇内聚合 + 全局聚合
        # ================================================================
        if not is_warmup and client_cluster is not None:

            # ---- D1: 检查是否触发动态重聚类 ----
            rounds_since_last = round_idx - last_cluster_round
            should_recluster  = (
                RECLUSTER_INTERVAL > 0
                and rounds_since_last >= RECLUSTER_INTERVAL
                and round_idx > WARMUP_ROUNDS - 1   # 不在热身期末尾重复聚类
            )

            if should_recluster:
                print(f"\n  [GMM] 触发动态重聚类（距上次聚类已过 {rounds_since_last} 轮）...")
                new_assignments, changed, _ = run_gmm_clustering(
                    local_weights, NUM_CLIENTS, NUM_CLUSTERS,
                    round_idx, run_dir, cluster_log,
                    prev_assignments=client_cluster,
                )
                if changed:
                    # 分配发生变化：将发生迁移的客户端所在新簇模型重置为全局模型，
                    # 避免旧簇知识污染新分组
                    changed_clusters = set()
                    for i in range(NUM_CLIENTS):
                        if new_assignments[i] != client_cluster[i]:
                            changed_clusters.add(new_assignments[i])
                    for k in changed_clusters:
                        cluster_models[k].load_state_dict(
                            copy.deepcopy(global_model.state_dict())
                        )
                        print(f"  [GMM] Cluster {k} 模型已重置为全局模型（因有新成员加入）")
                client_cluster     = new_assignments
                last_cluster_round = round_idx

            # ---- D2: 簇内 FedAvg ----
            cluster_weights_map = {k: [] for k in range(NUM_CLUSTERS)}
            cluster_lens_map    = {k: [] for k in range(NUM_CLUSTERS)}
            for i in range(NUM_CLIENTS):
                k = client_cluster[i]
                cluster_weights_map[k].append(local_weights[i])
                cluster_lens_map[k].append(local_lens[i])

            cluster_total_lens = []
            for k in range(NUM_CLUSTERS):
                if cluster_weights_map[k]:
                    cluster_models[k] = fedavg(
                        cluster_models[k],
                        cluster_weights_map[k],
                        cluster_lens_map[k],
                    )
                    cluster_total_lens.append(sum(cluster_lens_map[k]))
                else:
                    cluster_total_lens.append(0)

            # ---- D3: 各簇模型 → 全局模型 ----
            active_weights = [cluster_models[k].state_dict()
                              for k in range(NUM_CLUSTERS) if cluster_total_lens[k] > 0]
            active_lens    = [cluster_total_lens[k]
                              for k in range(NUM_CLUSTERS) if cluster_total_lens[k] > 0]
            if active_weights:
                global_model = fedavg(global_model, active_weights, active_lens)

        del local_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ================================================================
        # 阶段 E：验证
        # ================================================================
        # 各簇模型验证（分簇期才有意义；热身期也记录，便于对比）
        for k in range(NUM_CLUSTERS):
            if client_cluster is not None:
                members   = [i for i, c in enumerate(client_cluster) if c == k]
                n_samples = sum(local_lens[i] for i in members)
                k_loss    = float(np.mean([local_losses[i] for i in members])) if members else avg_loss
            else:
                # 热身期：簇模型与全局模型相同，均匀分配
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
            print(f"  [Cluster {k}] Pixel Acc: {pa:.4f} | mIoU: {miou:.4f} | 成员: {member_str}")

        # 全局模型验证
        round_time = time.time() - round_start
        total_time += round_time

        g_pa, g_miou = evaluate_model(global_model, val_loader, device, use_amp=USE_AMP)
        global_history.append({
            'round':     round_idx + 1,
            'phase':     'warmup' if is_warmup else 'clustered',
            'pixel_acc': g_pa,
            'miou':      g_miou,
            'avg_loss':  avg_loss,
            'time':      round_time,
        })
        print(f"  [Global]    Pixel Acc: {g_pa:.4f} | mIoU: {g_miou:.4f} | 耗时: {round_time:.1f}s",
              end="")

        # 保存最优全局模型
        if g_miou > best_miou:
            best_miou = g_miou
            torch.save({
                'round':            round_idx + 1,
                'model_state_dict': global_model.state_dict(),
                'num_classes':      NUM_CLASSES,
                'pixel_acc':        g_pa,
                'miou':             g_miou,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  ★ Best (mIoU: {g_miou:.4f})")
        else:
            print()

        # 每 10 轮保存检查点
        if (round_idx + 1) % 10 == 0:
            ckpt = os.path.join(save_dir, f'global_model_round_{round_idx + 1}.pth')
            torch.save({
                'round':            round_idx + 1,
                'model_state_dict': global_model.state_dict(),
                'num_classes':      NUM_CLASSES,
                'pixel_acc':        g_pa,
                'miou':             g_miou,
            }, ckpt)
            print(f"  >> 检查点已保存: {ckpt}")

        # 每轮实时更新 CSV
        save_cluster_csv(cluster_history, run_dir)
        save_global_csv(global_history, run_dir)

    # ===== 保存最终全局模型 =====
    final_path = os.path.join(save_dir, 'global_model_final.pth')
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'num_classes':      NUM_CLASSES,
        'num_rounds':       NUM_ROUNDS,
        'final_pixel_acc':  global_history[-1]['pixel_acc'],
        'final_miou':       global_history[-1]['miou'],
    }, final_path)

    # ===== 生成折线图 =====
    print(f"\n{'='*60}")
    print(f"正在生成折线图...")
    save_curves(cluster_history, global_history, NUM_CLUSTERS, WARMUP_ROUNDS, run_dir)

    # ===== 打印训练总结 =====
    print(f"\n{'='*80}")
    print(f"训练完成！总结如下：")
    print(f"{'='*80}")

    if torch.cuda.is_available():
        peak_mem  = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        print(f"\n[GPU 显存总结]")
        print(f"  GPU 型号:     {torch.cuda.get_device_name(device)}")
        print(f"  总显存:       {total_mem:.0f} MB")
        print(f"  训练峰值显存: {peak_mem:.0f} MB ({peak_mem/total_mem*100:.1f}%)")

    print(f"\n[训练性能总结]")
    print(f"  总训练时间:   {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  平均每轮耗时: {total_time/NUM_ROUNDS:.1f}s")
    print(f"  最优全局 mIoU: {best_miou:.4f}")
    print(f"  GMM 聚类次数: {len(cluster_log)} 次")

    print(f"\n{'Round':<8} {'Phase':<12} {'Loss':<12} {'Pixel Acc':<14} {'mIoU':<14} {'耗时(s)':<10}")
    print(f"{'-'*70}")
    for h in global_history:
        print(f"{h['round']:<8} {h['phase']:<12} {h['avg_loss']:<12.4f} "
              f"{h['pixel_acc']:<14.4f} {h['miou']:<14.4f} {h['time']:<10.1f}")
    print(f"{'-'*70}")

    print(f"\n本次运行所有结果已保存至: {run_dir}/")
    print(f"  ├── gmm_cluster_log.json      ← 每次聚类的详细记录")
    print(f"  ├── cluster_val_results.csv   ← 每轮每簇验证数据")
    print(f"  ├── global_val_results.csv    ← 每轮全局模型验证数据")
    print(f"  ├── pixel_accuracy.png        ← Pixel Accuracy 折线图")
    print(f"  └── miou.png                  ← mIoU 折线图")
    print(f"\n最优模型: {os.path.join(save_dir, 'best_model.pth')}")
    print(f"最终模型: {final_path}")


if __name__ == "__main__":
    main()
