"""
FedGMHC_Cityscapes_SegFormer.py — 基于高斯混合模型（GMM）分簇的联邦学习方法
                                    专用于 Cityscapes 数据集 + SegFormer-B0 模型

数据集说明
----------
- 图像目录: <DATASET_ROOT>/leftImg8bit/train|val/<city>/
- 标签目录: <DATASET_ROOT>/gtFine/train|val/<city>/*_gtFine_labelIds.png
- 训练类别: 19 类（road, sidewalk, building, ..., bicycle）
- 忽略类别: void（trainId=255），CrossEntropyLoss 中 ignore_index=255

模型说明
--------
SegFormer-B0 风格模型（PVTv2-B0 编码器 + All-MLP 解码器）：
  - 编码器：pvt_v2_b0（层次化 Transformer，3.4M 参数，含 30 个 LayerNorm 层）
  - 解码器：SegFormer All-MLP Decoder（embed_dim=256，0.4M 参数）
  - 总参数量：约 3.8M（比 MobileNetV2-UNet 6.6M 更轻量）

LayerNorm 特征提取（替代 BN 特征）
-----------------------------------
SegFormer 使用 LayerNorm 而非 BatchNorm，因此 FedGMHC 的聚类特征
从所有 LayerNorm 层的 weight（γ）和 bias（β）参数中提取：
  - 这些参数在训练中随数据分布调整，对客户端数据异质性敏感
  - PVTv2-B0 共 30 个 LayerNorm 层，原始特征向量维度约 7168
  - 经 PCA 降维到 2 维后用于 GMM 聚类，保证数值稳定性

冷启动解决策略（组合方案）
--------------------------
方案一  延迟聚类（热身期）
        前 WARMUP_ROUNDS 轮执行标准 FedAvg，等待模型收敛、LN 参数
        充分反映各客户端数据分布后，再进行首次 GMM 聚类。

方案二  动态重聚类
        首次聚类后，每隔 RECLUSTER_INTERVAL 轮重新提取 LN 特征并
        重新拟合 GMM，让分簇结果随训练进程自我修正并逐渐稳定。
        每次重聚类时，若分簇结果发生变化，对有新成员迁入的簇模型执行
        插值融合（α × 全局模型 + (1-α) × 原簇模型），而非硬重置，
        保证性能曲线平滑过渡，避免跳跃式下降。

算法流程
--------
热身期（Round 1 ~ WARMUP_ROUNDS）
    所有客户端从全局模型出发，完成本地训练，执行标准 FedAvg 聚合。
    每轮结束后，所有簇模型与全局模型保持同步。

首次聚类（Round WARMUP_ROUNDS 结束后）
    提取各客户端 LN 层 weight / bias，
    拼接为特征向量，拟合 GMM（K = NUM_CLUSTERS，full 协方差），
    按后验概率最大值分配各客户端到对应簇。

分簇训练期（Round WARMUP_ROUNDS+1 起）
    每轮：
      a. 每个客户端从所属簇模型出发，完成本地训练。
      b. 同簇客户端按数据量加权 FedAvg，更新各簇模型。
      c. 各簇模型按各簇总数据量加权 FedAvg，更新全局模型。
      d. 若当前轮满足重聚类条件（距上次聚类已过 RECLUSTER_INTERVAL 轮），
         重新提取 LN 特征并更新分簇结果。
      e. 在验证集上分别评估每个簇模型和全局模型，记录结果。

结果保存
--------
每次运行结果统一保存在 result_save/1FedGMHC_SF_MMDDHHmm/ 子目录下：
  result_save/
  └── 1FedGMHC_SF_MMDDHHmm/
      ├── gmm_cluster_log.json       ← 每次聚类的详细记录（含轮次、分配、后验概率）
      ├── cluster_val_results.csv    ← 每轮每簇验证数据汇总表（实时更新）
      ├── global_val_results.csv     ← 每轮全局模型验证数据汇总表（实时更新）
      ├── pixel_accuracy.png         ← 各簇 + 全局 Pixel Accuracy 折线图
      ├── miou.png                   ← 各簇 + 全局 mIoU 折线图
      └── cluster_scatter_roundN.png ← 每次聚类的 2D 散点图
"""

import torch
import copy
import json
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from model import build_segformer_b0
from model.SegFormerB0 import extract_ln_feature
from dataset.cityscapes_dataset import (
    CityscapesDataset,
    NUM_CLASSES,
    CLASS_NAMES,
    IGNORE_INDEX,
    build_label_index_cityscapes,
)
from partition import dirichlet_partition, print_partition_stats

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
# 重聚类后簇模型融合比例：新簇模型 = α × 全局模型 + (1-α) × 当前簇模型
# α=0 完全保留原簇模型；α=1 等同于硬重置；推荐 0.2~0.4
RECLUSTER_ALPHA     = 0.3
# PCA 目标维度：固定为 2 维
# 2 维的优势：
#   1. 参数数量极少（GMM 每部件只需 2+2+1=5 个参数），10个样本下不会奇异
#   2. 可直接生成 2D 散点图，直观展示客户端聚类分布，适合放入论文
PCA_N_COMPONENTS    = 2     # 固定 2 维，保证 GMM 数值稳定


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


def auto_batch_size(device, num_data_per_client, base_batch_size=4):
    """
    Cityscapes 图像尺寸为 (256, 512)，SegFormer 每张约 12MB 显存（含梯度）。
    """
    data_limit = max(4, num_data_per_client // 4)
    if torch.cuda.is_available():
        total     = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        # SegFormer 256×512 每张约 12MB（含梯度），保留 1GB 余量
        gpu_limit = int((total - 1000) / 12)
        gpu_limit = max(2, gpu_limit)
    else:
        gpu_limit = base_batch_size
    recommended = min(data_limit, gpu_limit)
    power = 1
    while power * 2 <= recommended:
        power *= 2
    return max(2, power)


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


# ==================== LayerNorm 特征提取 ====================

def extract_ln_feature_from_state_dict(state_dict):
    """
    从模型 state_dict 中提取所有编码器 LayerNorm 层的
    weight（γ）和 bias（β），拼接为一维特征向量（numpy）。

    与 BN 特征提取（running_mean/running_var）不同：
      - LN 使用可学习参数 γ/β，在训练中随数据分布调整
      - 这些参数能有效反映客户端的数据异质性
      - PVTv2-B0 共 30 个 LN 层，特征向量维度约 7168

    Parameters
    ----------
    state_dict : dict
        模型 state_dict（来自 model.state_dict()）

    Returns
    -------
    np.ndarray  形状 (D,)，D 约为 7168
    """
    return extract_ln_feature(state_dict)


# ==================== GMM 聚类 ====================

def run_gmm_clustering(local_weights, num_clients, n_clusters, round_idx, run_dir,
                       cluster_log, prev_assignments=None):
    """
    提取 LN 特征 → 标准化 → PCA 降维（2维）→ 拟合 GMM → 分配客户端到簇。

    PCA 固定降到 2 维的原因：
      - 10 个客户端样本下，高维 GMM 协方差矩阵奇异，后验概率退化为 0/1
      - 2 维时 GMM 每部件只需 5 个参数，10 个样本足够支撑稳定拟合
      - 2 维可直接生成散点图，直观展示客户端聚类分布
    K-Means++ 兜底：若 GMM 全部失败，自动切换为 K-Means++ 硬分配。
    """
    print(f"\n  [GMM] 提取 LayerNorm 参数特征（Round {round_idx + 1}）...")
    features = [extract_ln_feature_from_state_dict(w) for w in local_weights]
    feat_dim = features[0].shape[0]
    print(f"  [GMM] 原始特征向量维度: {feat_dim}（LN weight + bias，共 {feat_dim} 维）")

    X = np.stack(features, axis=0)   # (N, D)
    N = X.shape[0]
    effective_k = min(n_clusters, N)

    # ---- Step 1: 标准化 ----
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---- Step 2: PCA 降维（固定 2 维）----
    # 2 维保证 GMM 数值稳定，且可生成 2D 散点图
    n_components = max(2, min(PCA_N_COMPONENTS, N - 1, feat_dim))

    pca   = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled).astype(np.float64)

    explained_var = pca.explained_variance_ratio_.sum() * 100
    print(f"  [GMM] PCA 降维: {feat_dim} → {n_components} 维 "
          f"（累计解释方差: {explained_var:.1f}%）")

    # ---- Step 3: 拟合 GMM（full 协方差，2维下完全可行）----
    def _fit_gmm(X_data, k, reg, cov_type='full'):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=cov_type,
            max_iter=500,
            n_init=10,          # 10次随机初始化选最优，避免局部最优
            random_state=None,
            reg_covar=reg,
        )
        gmm.fit(X_data)
        return gmm

    # 2维下优先用 full 协方差，更准确；失败时退化到 diag，最后兜底 KMeans++
    reg_list = [1e-3, 1e-2, 1e-1, 0.5, 1.0]
    gmm = None
    for reg in reg_list:
        for cov_type in ['full', 'diag']:
            try:
                gmm = _fit_gmm(X_pca, effective_k, reg, cov_type)
                print(f"  [GMM] 拟合成功 (covariance_type={cov_type}, reg_covar={reg})")
                break
            except (ValueError, np.linalg.LinAlgError):
                pass
        if gmm is not None:
            break

    def _save_scatter(X_2d, labels, posteriors_2d, method_name, run_dir, round_idx):
        """生成 2D 聚类散点图，每次聚类时保存。"""
        if X_2d.shape[1] != 2:
            return
        colors = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
        fig, ax = plt.subplots(figsize=(7, 6))
        for k in range(effective_k):
            idx = [i for i, c in enumerate(labels) if c == k]
            if idx:
                ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                           c=colors[k % len(colors)], s=120,
                           label=f'Cluster {k}', zorder=3, edgecolors='white', linewidths=0.8)
        # 标注客户端编号
        for i in range(len(labels)):
            ax.annotate(f'C{i}', (X_2d[i, 0], X_2d[i, 1]),
                        textcoords='offset points', xytext=(6, 4), fontsize=9)
        ax.set_xlabel('PC 1', fontsize=11)
        ax.set_ylabel('PC 2', fontsize=11)
        ax.set_title(f'Client Clustering (Round {round_idx + 1}, {method_name})', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        scatter_path = os.path.join(run_dir, f'cluster_scatter_round{round_idx + 1}.png')
        fig.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [GMM] 聚类散点图已保存: {scatter_path}")

    if gmm is None:
        # K-Means++ 兜底
        print("  [GMM] 警告：GMM 全部失败，降级为 KMeans++ 硬分配")
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=effective_k, init='k-means++', n_init=20, random_state=None)
        km.fit(X_pca)
        hard_labels  = km.labels_
        posteriors   = np.zeros((N, effective_k), dtype=np.float64)
        posteriors[np.arange(N), hard_labels] = 1.0
        new_assignments = hard_labels.tolist()
        changed = (prev_assignments is None) or (new_assignments != prev_assignments)
        print(f"  [KMeans++] 分簇结果: {new_assignments}")
        _save_scatter(X_pca, new_assignments, posteriors, 'KMeans++', run_dir, round_idx)
        cluster_log.append({
            'round':                 round_idx + 1,
            'trigger':               'warmup_end' if prev_assignments is None else 'recluster',
            'method':                'kmeans_fallback',
            'feature_type':          'LayerNorm_weight_bias',
            'feature_dim_raw':       int(feat_dim),
            'feature_dim_pca':       int(n_components),
            'pca_explained_var_pct': round(float(explained_var), 2),
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
        return new_assignments, changed, posteriors

    posteriors      = gmm.predict_proba(X_pca)
    new_assignments = posteriors.argmax(axis=1).tolist()
    changed = (prev_assignments is None) or (new_assignments != prev_assignments)

    # 检查后验概率是否仍然退化（最大概率均 > 0.99 视为退化）
    max_probs = posteriors.max(axis=1)
    if (max_probs > 0.99).all():
        print("  [GMM] 警告：后验概率仍退化为 0/1，自动切换为 KMeans++ 硬分配")
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=effective_k, init='k-means++', n_init=20, random_state=None)
        km.fit(X_pca)
        hard_labels  = km.labels_
        posteriors   = np.zeros((N, effective_k), dtype=np.float64)
        posteriors[np.arange(N), hard_labels] = 1.0
        new_assignments = hard_labels.tolist()
        changed = (prev_assignments is None) or (new_assignments != prev_assignments)
        method_used = 'kmeans_fallback'
        print(f"  [KMeans++] 分簇结果: {new_assignments}")
    else:
        method_used = 'gmm'
        print(f"  [GMM] 后验概率正常（最大概率范围: {max_probs.min():.3f} ~ {max_probs.max():.3f}）")

    label_str = '首次' if prev_assignments is None else '重聚类'
    print(f"  [GMM] 客户端分簇结果（{label_str}，方法: {method_used}）:")
    for i, k in enumerate(new_assignments):
        prob_str   = ', '.join([f'C{j}:{posteriors[i, j]:.3f}'
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

    # 生成 2D 聚类散点图
    _save_scatter(X_pca, new_assignments, posteriors, method_used.upper(), run_dir, round_idx)

    cluster_log.append({
        'round':                 round_idx + 1,
        'trigger':               'warmup_end' if prev_assignments is None else 'recluster',
        'method':                method_used,
        'feature_type':          'LayerNorm_weight_bias',
        'feature_dim_raw':       int(feat_dim),
        'feature_dim_pca':       int(n_components),
        'pca_explained_var_pct': round(float(explained_var), 2),
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

def interpolate_models(cluster_model, global_model, alpha):
    """
    将簇模型与全局模型做加权插值：
        新簇模型参数 = α × 全局模型参数 + (1 - α) × 当前簇模型参数
    """
    cluster_sd = cluster_model.state_dict()
    global_sd  = global_model.state_dict()
    blended    = {}
    for key in cluster_sd:
        if cluster_sd[key].is_floating_point():
            blended[key] = (1.0 - alpha) * cluster_sd[key] + alpha * global_sd[key]
        else:
            blended[key] = global_sd[key]
    cluster_model.load_state_dict(blended)
    return cluster_model


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

    def local_train(self, model, batch_size=4, epochs=1, lr=1e-4,
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
        # SegFormer 使用 AdamW 优化器（Transformer 标准配置）
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        # Cityscapes 有 void 类（trainId=255），必须设置 ignore_index
        criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
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
         'Cityscapes FedGMHC (SegFormer-B0) — Pixel Accuracy per Cluster & Global vs. Round',
         'pixel_accuracy.png', 'pa', global_pa),
        ('mIoU',
         'Cityscapes FedGMHC (SegFormer-B0) — mIoU per Cluster & Global vs. Round',
         'miou.png', 'miou', global_miou),
    ]:
        plt.figure(figsize=(13, 6))

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
    # Cityscapes 数据集根目录（包含 leftImg8bit/ 和 gtFine/ 两个子目录）
    DATASET_ROOT = r'E:\Autonomous Driving Dataset\Cityscapes dataset(10g)'

    USE_AMP      = True
    # Cityscapes 原始分辨率为 1024×2048；推荐使用 (512, 1024) 保持宽高比 1:2
    # 4060 Ti 8GB 建议使用 (256, 512)，可用 BATCH_SIZE=4，速度与显存均衡
    TARGET_SIZE  = (256, 512)
    NUM_ROUNDS   = 100
    NUM_CLIENTS  = 20
    LOCAL_EPOCHS = 1        # 联邦学习标准设置；通过增加通信轮数补偿
    # SegFormer 使用 AdamW，学习率比 SGD 小一个数量级
    LR           = 1e-4
    NUM_WORKERS  = 0 if sys.platform == 'win32' else 4
    PIN_MEMORY   = True
    BATCH_SIZE   = 4        # 4060 Ti 8GB + 256×512 图像，AMP 下 SegFormer 每张约 12MB
    DIRICHLET_ALPHA = 1.0   # Dirichlet 浓度参数（越小异质性越强；推荐 0.5/1.0/2.0）
    MIN_SAMPLES     = 100   # 每个客户端最少图像数量（Cityscapes 共 2974 张，每人约 149 张）
    MAX_SAMPLES     = 200   # 每个客户端最多图像数量（限制数据量过多的客户端，加快每轮训练）
                            # 设为 None 则不限制
    PRETRAINED      = True  # 是否使用 ImageNet 预训练编码器权重
    # ================================================

    # ===== 时间戳运行目录（前缀 1FedGMHC_SF 区分 SegFormer 版本）=====
    run_timestamp = datetime.now().strftime('%m%d%H%M')
    run_dir = os.path.join('../result_save', f'1FedGMHC_SF_{run_timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"\n本次运行结果将保存至: {run_dir}/")

    # ===== 加载 Cityscapes 数据集 =====
    print(f"\n正在加载 Cityscapes 数据集（路径: {DATASET_ROOT}）...")
    train_dataset = CityscapesDataset(
        root_dir=DATASET_ROOT,
        split='train',
        transform=transforms.ToTensor(),
        target_size=TARGET_SIZE,
    )
    val_dataset = CityscapesDataset(
        root_dir=DATASET_ROOT,
        split='val',
        transform=transforms.ToTensor(),
        target_size=TARGET_SIZE,
    )

    num_images = len(train_dataset)

    # ===== Dirichlet Non-IID 数据划分 =====
    print(f"\n  [Partition] 使用 Dirichlet(α={DIRICHLET_ALPHA}) Non-IID 划分...")
    labels = build_label_index_cityscapes(
        DATASET_ROOT, split='train',
        num_classes=NUM_CLASSES,
        target_size=TARGET_SIZE,
    )
    user_groups = dirichlet_partition(
        num_clients=NUM_CLIENTS, labels=labels, num_classes=NUM_CLASSES,
        alpha=DIRICHLET_ALPHA, min_samples=MIN_SAMPLES, seed=42,
    )
    # ===== 截断超出 MAX_SAMPLES 限制的客户端数据 =====
    if MAX_SAMPLES is not None:
        import random as _random
        _random.seed(42)
        clipped = 0
        for i in range(len(user_groups)):
            if len(user_groups[i]) > MAX_SAMPLES:
                user_groups[i] = _random.sample(list(user_groups[i]), MAX_SAMPLES)
                clipped += 1
        if clipped > 0:
            print(f"  [Partition] MAX_SAMPLES={MAX_SAMPLES}: {clipped} 个客户端数据被截断")

    print_partition_stats(user_groups, labels, NUM_CLASSES, CLASS_NAMES)

    min_data = min(len(g) for g in user_groups)
    if BATCH_SIZE == 0:
        BATCH_SIZE = auto_batch_size(device, min_data)

    print(f"\n{'='*65}")
    print(f"训练配置（FedGMHC — Cityscapes + SegFormer-B0）:")
    print(f"  数据集根目录: {DATASET_ROOT}")
    print(f"  图像尺寸: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}  |  训练类别: {NUM_CLASSES}  |  IGNORE_INDEX: {IGNORE_INDEX}")
    print(f"  训练集: {num_images} 张 | 验证集: {len(val_dataset)} 张")
    max_data = max(len(g) for g in user_groups)
    _max_str = f' | 最多 {MAX_SAMPLES} 张/客户端' if MAX_SAMPLES is not None else ''
    print(f"  客户端: {NUM_CLIENTS} 个 | 簇数: {NUM_CLUSTERS} | Dirichlet α={DIRICHLET_ALPHA} | 最少 {min_data} 张{_max_str}")
    _recluster_str = '禁用' if RECLUSTER_INTERVAL == 0 else f'每 {RECLUSTER_INTERVAL} 轮'
    print(f"  热身轮数: {WARMUP_ROUNDS} | 重聚类间隔: {_recluster_str}")
    print(f"  Batch Size: {BATCH_SIZE} | Local Epochs: {LOCAL_EPOCHS} | 联邦轮数: {NUM_ROUNDS}")
    print(f"  学习率: {LR} (AdamW) | AMP: {'已启用' if USE_AMP and torch.cuda.is_available() else '未启用'}")
    print(f"  预训练编码器: {'是' if PRETRAINED else '否'}")
    print(f"  聚类特征: LayerNorm weight + bias（替代 BN running_mean/var）")
    print(f"{'='*65}")

    # ===== 初始化全局模型（SegFormer-B0）=====
    print(f"\n正在初始化 SegFormer-B0 模型（pretrained={PRETRAINED}）...")
    global_model = build_segformer_b0(num_classes=NUM_CLASSES, pretrained=PRETRAINED).to(device)
    total_params   = sum(p.numel() for p in global_model.parameters())
    encoder_params = sum(p.numel() for p in global_model.encoder.parameters())
    decoder_params = sum(p.numel() for p in global_model.decoder.parameters())
    print(f"模型总参数量: {total_params:,} ({total_params * 4 / 1024**2:.1f} MB in FP32)")
    print(f"  编码器 (PVTv2-B0): {encoder_params:,} 参数，{sum(1 for m in global_model.encoder.modules() if isinstance(m, torch.nn.LayerNorm))} 个 LayerNorm 层")
    print(f"  解码器 (All-MLP):  {decoder_params:,} 参数")

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
    client_cluster     = None   # 当前分簇结果，None 表示热身期
    last_cluster_round = -1     # 上次执行聚类的轮次索引
    cluster_log        = []     # 每次聚类的详细记录

    cluster_history = []
    global_history  = []
    best_miou       = 0.0
    total_time      = 0.0

    print(f"\n{'='*80}")
    print(f"开始 FedGMHC 训练（Cityscapes + SegFormer-B0）")
    print(f"热身 {WARMUP_ROUNDS} 轮 + 动态重聚类间隔 "
          f"{'禁用' if RECLUSTER_INTERVAL == 0 else RECLUSTER_INTERVAL} 轮")
    print(f"聚类特征: LayerNorm weight + bias（PVTv2-B0 共 30 个 LN 层，7168 维 → PCA 2 维）")
    print(f"{'='*80}")

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        is_warmup   = (round_idx < WARMUP_ROUNDS)
        phase_label = f'Warmup({round_idx + 1}/{WARMUP_ROUNDS})' if is_warmup else 'Clustered'
        print(f"\n--- Round {round_idx + 1}/{NUM_ROUNDS}  [{phase_label}] ---")

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
            for k in range(NUM_CLUSTERS):
                cluster_models[k].load_state_dict(copy.deepcopy(global_model.state_dict()))

        # ================================================================
        # 阶段 C：热身期结束后 → 首次 GMM 聚类
        # ================================================================
        if not is_warmup and client_cluster is None:
            print(f"\n  [FedGMHC] 热身期结束，执行首次 GMM 聚类...")
            client_cluster, _, _ = run_gmm_clustering(
                local_weights, NUM_CLIENTS, NUM_CLUSTERS,
                round_idx, run_dir, cluster_log,
                prev_assignments=None,
            )
            last_cluster_round = round_idx

        # ================================================================
        # 阶段 D：分簇训练期 → 簇内 FedAvg + 全局聚合
        # ================================================================
        if not is_warmup and client_cluster is not None:
            # D1. 簇内 FedAvg
            for k in range(NUM_CLUSTERS):
                members = [i for i, c in enumerate(client_cluster) if c == k]
                if not members:
                    continue
                k_weights = [local_weights[i] for i in members]
                k_lens    = [local_lens[i]    for i in members]
                cluster_models[k] = fedavg(cluster_models[k], k_weights, k_lens)

            # D2. 全局聚合（各簇按总数据量加权）
            cluster_lens_total = []
            cluster_weights    = []
            for k in range(NUM_CLUSTERS):
                members = [i for i, c in enumerate(client_cluster) if c == k]
                if members:
                    cluster_lens_total.append(sum(local_lens[i] for i in members))
                    cluster_weights.append(cluster_models[k].state_dict())
            if cluster_weights:
                global_model = fedavg(global_model, cluster_weights, cluster_lens_total)

            # D3. 动态重聚类
            if (RECLUSTER_INTERVAL > 0 and
                    (round_idx - last_cluster_round) >= RECLUSTER_INTERVAL):
                print(f"\n  [FedGMHC] 触发动态重聚类（距上次聚类 {round_idx - last_cluster_round} 轮）...")
                new_assignments, changed, _ = run_gmm_clustering(
                    local_weights, NUM_CLIENTS, NUM_CLUSTERS,
                    round_idx, run_dir, cluster_log,
                    prev_assignments=client_cluster,
                )
                if changed:
                    print(f"  [FedGMHC] 分簇结果已变化，对迁入新成员的簇执行插值融合（α={RECLUSTER_ALPHA}）...")
                    for k in range(NUM_CLUSTERS):
                        old_members = set(i for i, c in enumerate(client_cluster) if c == k)
                        new_members = set(i for i, c in enumerate(new_assignments) if c == k)
                        if new_members - old_members:  # 有新成员迁入
                            cluster_models[k] = interpolate_models(
                                cluster_models[k], global_model, RECLUSTER_ALPHA)
                client_cluster     = new_assignments
                last_cluster_round = round_idx

        del local_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ================================================================
        # 阶段 E：验证
        # ================================================================
        round_time  = time.time() - round_start
        total_time += round_time

        # 验证全局模型
        g_pa, g_miou = evaluate_model(global_model, val_loader, device, use_amp=USE_AMP)
        global_history.append({
            'round':     round_idx + 1,
            'phase':     'warmup' if is_warmup else 'clustered',
            'pixel_acc': g_pa,
            'miou':      g_miou,
            'avg_loss':  avg_loss,
            'time':      round_time,
        })
        print(f"  [Global] Pixel Acc: {g_pa:.4f} | mIoU: {g_miou:.4f} | 耗时: {round_time:.1f}s",
              end="")

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
                'method':           'FedGMHC_SegFormer',
                'model':            'SegFormerB0',
            }, os.path.join(save_dir, 'best_model_fedgmhc_segformer_cityscapes.pth'))
            print(f"  ★ Best (mIoU: {g_miou:.4f})")
        else:
            print()

        # 验证各簇模型（仅在分簇期）
        if not is_warmup and client_cluster is not None:
            for k in range(NUM_CLUSTERS):
                members = [i for i, c in enumerate(client_cluster) if c == k]
                if not members:
                    continue
                c_pa, c_miou = evaluate_model(cluster_models[k], val_loader, device, use_amp=USE_AMP)
                num_samples  = sum(local_lens[i] for i in members) if 'local_lens' in dir() else 0
                cluster_history.append({
                    'round':       round_idx + 1,
                    'phase':       'clustered',
                    'cluster':     k,
                    'num_clients': len(members),
                    'num_samples': sum(len(user_groups[i]) for i in members),
                    'pixel_acc':   c_pa,
                    'miou':        c_miou,
                    'avg_loss':    avg_loss,
                })
                print(f"  [Cluster {k}] Pixel Acc: {c_pa:.4f} | mIoU: {c_miou:.4f} "
                      f"| 成员: {members}")

        if (round_idx + 1) % 10 == 0:
            ckpt = os.path.join(save_dir, f'fedgmhc_segformer_cityscapes_round_{round_idx + 1}.pth')
            torch.save({
                'round':            round_idx + 1,
                'model_state_dict': global_model.state_dict(),
                'num_classes':      NUM_CLASSES,
                'pixel_acc':        g_pa,
                'miou':             g_miou,
                'dataset':          'cityscapes',
                'target_size':      TARGET_SIZE,
                'method':           'FedGMHC_SegFormer',
                'model':            'SegFormerB0',
            }, ckpt)
            print(f"  >> 检查点已保存: {ckpt}")

        # 每轮实时更新 CSV
        save_cluster_csv(cluster_history, run_dir)
        save_global_csv(global_history, run_dir)

    # ===== 保存最终全局模型 =====
    final_path = os.path.join(save_dir, 'fedgmhc_segformer_cityscapes_final.pth')
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'num_classes':      NUM_CLASSES,
        'num_rounds':       NUM_ROUNDS,
        'final_pixel_acc':  global_history[-1]['pixel_acc'],
        'final_miou':       global_history[-1]['miou'],
        'dataset':          'cityscapes',
        'target_size':      TARGET_SIZE,
        'method':           'FedGMHC_SegFormer',
        'model':            'SegFormerB0',
    }, final_path)

    # ===== 生成折线图 =====
    print(f"\n{'='*60}")
    print(f"正在生成折线图...")
    save_curves(cluster_history, global_history, NUM_CLUSTERS, WARMUP_ROUNDS, run_dir)

    # ===== 打印训练总结 =====
    print(f"\n{'='*80}")
    print(f"训练完成！总结如下（FedGMHC — Cityscapes + SegFormer-B0）：")
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

    print(f"\n{'Round':<8} {'Phase':<12} {'Loss':<12} {'Pixel Acc':<14} {'mIoU':<14} {'耗时(s)':<10}")
    print(f"{'-'*72}")
    for h in global_history:
        print(f"{h['round']:<8} {h['phase']:<12} {h['avg_loss']:<12.4f} "
              f"{h['pixel_acc']:<14.4f} {h['miou']:<14.4f} {h['time']:<10.1f}")
    print(f"{'-'*72}")

    print(f"\n本次运行所有结果已保存至: {run_dir}/")
    print(f"  ├── gmm_cluster_log.json       ← 每次聚类的详细记录")
    print(f"  ├── cluster_val_results.csv    ← 每轮每簇验证数据")
    print(f"  ├── global_val_results.csv     ← 每轮全局模型验证数据")
    print(f"  ├── pixel_accuracy.png         ← Pixel Accuracy 折线图")
    print(f"  ├── miou.png                   ← mIoU 折线图")
    print(f"  └── cluster_scatter_roundN.png ← 聚类散点图")
    print(f"\n最优模型: {os.path.join(save_dir, 'best_model_fedgmhc_segformer_cityscapes.pth')}")
    print(f"最终模型: {final_path}")


if __name__ == "__main__":
    main()
