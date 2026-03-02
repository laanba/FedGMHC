"""
FedGMHC.py — 基于高斯混合模型（GMM）分簇的联邦学习方法

算法流程
--------
阶段 0  初始化训练（第 1 轮）
        所有客户端从同一全局模型出发，完成本地训练。

阶段 1  BN 特征提取与 GMM 聚类（仅在第 1 轮结束后执行一次）
        1. 从每个客户端训练后的本地模型中提取所有 BatchNorm 层的
           running_mean 和 running_var，拼接为客户端特征向量。
        2. 对所有客户端特征向量拟合高斯混合模型（GMM，K=NUM_CLUSTERS 个部件）。
        3. 计算每个客户端对各高斯部件的后验概率（responsibility），
           取后验概率最大的部件编号作为该客户端所属簇。

阶段 2  分簇联邦训练（第 2 轮起）
        每轮：
          a. 每个客户端从 **所属簇的簇模型** 出发，完成本地训练。
          b. 同簇客户端按数据量加权聚合，更新各簇模型。
          c. 所有簇模型按各簇总数据量加权聚合，更新全局模型。
          d. 在验证集上分别评估每个簇模型和全局模型，记录结果。

结果保存
--------
每次运行结果统一保存在 result_save/MMDDHHmm/ 子目录下：
  result_save/
  └── MMDDHHmm/
      ├── cluster_val_results.csv    ← 每轮每簇验证数据汇总表
      ├── global_val_results.csv     ← 每轮全局模型验证数据汇总表
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


# ==================== 超参数 ====================
NUM_CLUSTERS = 3   # GMM 部件数 / 簇数


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

def fit_gmm_and_assign(client_features, n_clusters=NUM_CLUSTERS, random_state=42):
    """
    用 GMM 对客户端特征向量聚类，返回：
      - gmm        : 拟合好的 GaussianMixture 对象
      - assignments: list[int]，每个客户端所属簇编号（0-based）
      - posteriors : ndarray (N, K)，每个客户端对每个部件的后验概率
    """
    X = np.stack(client_features, axis=0)   # (N, D)

    # 若客户端数 < 部件数，退化为 k-means 式硬分配
    effective_k = min(n_clusters, len(X))

    gmm = GaussianMixture(
        n_components=effective_k,
        covariance_type='diag',   # 对角协方差，适合高维特征
        max_iter=200,
        random_state=random_state,
        reg_covar=1e-4,           # 正则化，防止数值奇异
    )
    gmm.fit(X)

    posteriors  = gmm.predict_proba(X)          # (N, K)
    assignments = posteriors.argmax(axis=1).tolist()

    return gmm, assignments, posteriors


# ==================== 联邦聚合 ====================

def fedavg(base_model, weights_list, lens_list):
    """
    加权平均聚合：按数据量比例对 state_dict 进行加权平均，
    结果写入 base_model 并返回。
    """
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
    """保存每轮每簇验证数据到 CSV"""
    path = os.path.join(run_dir, 'cluster_val_results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Cluster', 'Num_Clients', 'Num_Samples',
                         'Pixel_Accuracy', 'mIoU', 'Avg_Loss'])
        for r in cluster_history:
            writer.writerow([
                r['round'], r['cluster'], r['num_clients'], r['num_samples'],
                f"{r['pixel_acc']:.6f}", f"{r['miou']:.6f}", f"{r['avg_loss']:.6f}",
            ])
    return path


def save_global_csv(global_history, run_dir):
    """保存每轮全局模型验证数据到 CSV"""
    path = os.path.join(run_dir, 'global_val_results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Pixel_Accuracy', 'mIoU', 'Avg_Loss', 'Time_s'])
        for r in global_history:
            writer.writerow([
                r['round'], f"{r['pixel_acc']:.6f}", f"{r['miou']:.6f}",
                f"{r['avg_loss']:.6f}", f"{r['time']:.1f}",
            ])
    return path


def save_curves(cluster_history, global_history, num_clusters, run_dir):
    """
    生成两张折线图：
      1. pixel_accuracy.png — 各簇 + 全局 Pixel Accuracy 随轮次变化
      2. miou.png           — 各簇 + 全局 mIoU 随轮次变化
    """
    # 整理数据
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

    for metric, ylabel, title, suffix, c_key, g_data in [
        ('pixel_acc', 'Pixel Accuracy',
         'Pixel Accuracy per Cluster & Global vs. Round',
         'pixel_accuracy.png', 'pa', global_pa),
        ('miou', 'mIoU',
         'mIoU per Cluster & Global vs. Round',
         'miou.png', 'miou', global_miou),
    ]:
        plt.figure(figsize=(12, 6))

        # 各簇曲线（虚线）
        for k in range(num_clusters):
            d = cluster_data[k]
            if d['rounds']:
                plt.plot(d['rounds'], d[c_key],
                         linestyle='--', marker='o', linewidth=1.5, markersize=3,
                         color=colors[k % len(colors)],
                         label=f'Cluster {k}')

        # 全局曲线（实线，加粗）
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
    USE_AMP       = True
    TARGET_SIZE   = (256, 256)
    NUM_ROUNDS    = 50
    NUM_CLIENTS   = 5
    LOCAL_EPOCHS  = 5
    LR            = 0.01
    NUM_WORKERS   = 0 if sys.platform == 'win32' else 4
    PIN_MEMORY    = True
    BATCH_SIZE    = 0       # 0 = 自动推荐
    # ================================================

    # ===== 时间戳运行目录 =====
    run_timestamp = datetime.now().strftime('%m%d%H%M')
    run_dir = os.path.join('./result_save', run_timestamp)
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

    print(f"\n{'='*60}")
    print(f"训练配置:")
    print(f"  训练集: {num_images} 张 | 验证集: {len(val_dataset)} 张")
    print(f"  客户端: {NUM_CLIENTS} 个 | 簇数: {NUM_CLUSTERS}")
    print(f"  Batch Size: {BATCH_SIZE} | Local Epochs: {LOCAL_EPOCHS} | 联邦轮数: {NUM_ROUNDS}")
    print(f"  学习率: {LR} | AMP: {'已启用' if USE_AMP and torch.cuda.is_available() else '未启用'}")
    print(f"{'='*60}")

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
    # cluster_models[k]: 第 k 簇的当前簇模型（初始为全局模型副本）
    cluster_models     = [copy.deepcopy(global_model) for _ in range(NUM_CLUSTERS)]
    # client_cluster[i]: 客户端 i 所属簇编号（GMM 分配后确定）
    client_cluster     = None
    gmm_fitted         = False

    cluster_history = []   # 每轮每簇验证记录
    global_history  = []   # 每轮全局模型验证记录
    best_miou       = 0.0
    total_time      = 0.0

    print(f"\n{'='*80}")
    print(f"开始 FedGMHC 训练...")
    print(f"{'='*80}")

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n--- Round {round_idx + 1}/{NUM_ROUNDS} ---")

        # ================================================================
        # 阶段 A：每个客户端本地训练
        #   - 第 1 轮：从全局模型出发
        #   - 第 2 轮起：从所属簇模型出发
        # ================================================================
        local_weights = []   # list of state_dict
        local_losses  = []
        local_lens    = []

        for i in range(NUM_CLIENTS):
            client = Client(i, train_dataset, user_groups[i], device, use_amp=USE_AMP)

            if round_idx == 0 or client_cluster is None:
                # 第 1 轮：所有客户端使用全局模型
                start_model = copy.deepcopy(global_model)
            else:
                # 第 2 轮起：使用所属簇模型
                k = client_cluster[i]
                start_model = copy.deepcopy(cluster_models[k])

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
        # 阶段 B：第 1 轮结束后 → 提取 BN 特征 → GMM 聚类 → 分配客户端到簇
        # ================================================================
        if round_idx == 0 and not gmm_fitted:
            print(f"\n  [GMM] 提取 BN 层统计特征...")
            features = [extract_bn_feature(w) for w in local_weights]
            feat_dim = features[0].shape[0]
            print(f"  [GMM] 客户端特征向量维度: {feat_dim}")

            print(f"  [GMM] 拟合高斯混合模型（K={NUM_CLUSTERS}）...")
            gmm, assignments, posteriors = fit_gmm_and_assign(features, n_clusters=NUM_CLUSTERS)
            client_cluster = assignments
            gmm_fitted = True

            print(f"  [GMM] 客户端分簇结果:")
            for i, k in enumerate(client_cluster):
                prob_str = ', '.join([f'C{j}:{posteriors[i,j]:.3f}' for j in range(posteriors.shape[1])])
                print(f"    Client {i} → Cluster {k}  (后验概率: {prob_str})")

            # 簇分配摘要
            for k in range(NUM_CLUSTERS):
                members = [i for i, c in enumerate(client_cluster) if c == k]
                print(f"  Cluster {k}: {members} ({len(members)} 个客户端)")

            # 保存分簇信息到 JSON
            cluster_info = {
                'num_clusters': NUM_CLUSTERS,
                'feature_dim': int(feat_dim),
                'assignments': client_cluster,
                'posteriors': posteriors.tolist(),
                'cluster_members': {
                    str(k): [i for i, c in enumerate(client_cluster) if c == k]
                    for k in range(NUM_CLUSTERS)
                },
            }
            with open(os.path.join(run_dir, 'gmm_cluster_info.json'), 'w', encoding='utf-8') as f:
                json.dump(cluster_info, f, ensure_ascii=False, indent=2)
            print(f"  [GMM] 分簇信息已保存: {run_dir}/gmm_cluster_info.json")

        # ================================================================
        # 阶段 C：簇内聚合 → 更新各簇模型
        # ================================================================
        if client_cluster is None:
            # 第 1 轮 GMM 尚未完成时，退化为全局 FedAvg（仅此一轮）
            global_model = fedavg(global_model, local_weights, local_lens)
            # 同步到所有簇模型（作为下一轮的起点）
            for k in range(NUM_CLUSTERS):
                cluster_models[k].load_state_dict(copy.deepcopy(global_model.state_dict()))
        else:
            # 按簇分组聚合
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
                    # 该簇本轮无客户端，保持原模型不变
                    cluster_total_lens.append(0)

            # ================================================================
            # 阶段 D：各簇模型聚合 → 更新全局模型
            # ================================================================
            active_cluster_weights = []
            active_cluster_lens    = []
            for k in range(NUM_CLUSTERS):
                if cluster_total_lens[k] > 0:
                    active_cluster_weights.append(cluster_models[k].state_dict())
                    active_cluster_lens.append(cluster_total_lens[k])

            if active_cluster_weights:
                global_model = fedavg(global_model, active_cluster_weights, active_cluster_lens)

        del local_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ================================================================
        # 阶段 E：验证 — 每个簇模型 + 全局模型
        # ================================================================
        # 各簇模型验证
        for k in range(NUM_CLUSTERS):
            members = ([i for i, c in enumerate(client_cluster) if c == k]
                       if client_cluster else list(range(NUM_CLIENTS)))
            n_samples = sum(local_lens[i] for i in members) if members else 0
            k_loss    = (float(np.mean([local_losses[i] for i in members]))
                         if members else avg_loss)

            pa, miou = evaluate_model(cluster_models[k], val_loader, device, use_amp=USE_AMP)
            cluster_history.append({
                'round':       round_idx + 1,
                'cluster':     k,
                'num_clients': len(members),
                'num_samples': n_samples,
                'pixel_acc':   pa,
                'miou':        miou,
                'avg_loss':    k_loss,
            })
            print(f"  [Cluster {k}] Pixel Acc: {pa:.4f} | mIoU: {miou:.4f} "
                  f"| 成员: {members}")

        # 全局模型验证
        round_time = time.time() - round_start
        total_time += round_time

        g_pa, g_miou = evaluate_model(global_model, val_loader, device, use_amp=USE_AMP)
        global_history.append({
            'round':     round_idx + 1,
            'pixel_acc': g_pa,
            'miou':      g_miou,
            'avg_loss':  avg_loss,
            'time':      round_time,
        })
        print(f"  [Global] Pixel Acc: {g_pa:.4f} | mIoU: {g_miou:.4f} | 耗时: {round_time:.1f}s",
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

        # 每轮实时更新 CSV（防止中途崩溃丢失数据）
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
    save_curves(cluster_history, global_history, NUM_CLUSTERS, run_dir)

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

    print(f"\n{'Round':<8} {'Loss':<12} {'Pixel Acc':<14} {'mIoU':<14} {'耗时(s)':<10}")
    print(f"{'-'*58}")
    for h in global_history:
        print(f"{h['round']:<8} {h['avg_loss']:<12.4f} {h['pixel_acc']:<14.4f} "
              f"{h['miou']:<14.4f} {h['time']:<10.1f}")
    print(f"{'-'*58}")

    print(f"\n本次运行所有结果已保存至: {run_dir}/")
    print(f"  ├── gmm_cluster_info.json      ← GMM 分簇详情")
    print(f"  ├── cluster_val_results.csv    ← 每轮每簇验证数据")
    print(f"  ├── global_val_results.csv     ← 每轮全局模型验证数据")
    print(f"  ├── pixel_accuracy.png         ← Pixel Accuracy 折线图")
    print(f"  └── miou.png                   ← mIoU 折线图")
    print(f"\n最优模型: {os.path.join(save_dir, 'best_model.pth')}")
    print(f"最终模型: {final_path}")


if __name__ == "__main__":
    main()
