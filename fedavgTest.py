"""
fedavgTest.py — 联邦平均算法（含客户端聚合前验证）

与 Fedavg.py 的核心区别：
  1. 在每一轮服务端聚合 **之前**，对每一个客户端完成本地训练后的模型
     单独在验证集上进行评估，记录 Pixel Accuracy、mIoU 及各类别 IoU。
  2. 每次运行时，所有输出（客户端验证数据 + 全局模型图表）统一保存在
     以 **月日时分** 编号的子文件夹中：

       result_save/
       └── MMDDHHmm/                       ← 本次运行根目录（如 03021435）
           ├── datasave/                   ← 客户端聚合前验证数据
           │   ├── client_val_results.csv  ← 所有轮次所有客户端汇总表（每轮实时更新）
           │   ├── round_1/
           │   │   ├── client_0.json
           │   │   └── ...
           │   ├── round_2/ ...
           │   ├── pixel_accuracy_clients.png
           │   └── miou_clients.png
           ├── pixel_accuracy.png          ← 全局模型 Pixel Accuracy 曲线
           ├── miou.png                    ← 全局模型 mIoU 曲线
           └── results.csv                 ← 全局模型验证汇总表
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


# ==================== 显存监控工具 ====================

def get_gpu_memory_info(device):
    """获取 GPU 显存信息（单位：MB）"""
    if not torch.cuda.is_available():
        return None
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
    cached = torch.cuda.memory_reserved(device) / 1024 ** 2
    total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
    free = total - cached
    return {
        'allocated_mb': allocated,
        'cached_mb': cached,
        'total_mb': total,
        'free_mb': free,
        'utilization_pct': (cached / total) * 100
    }


def print_gpu_status(device, label=""):
    """打印当前 GPU 显存状态"""
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
    """根据 GPU 显存和每个客户端的数据量，智能推荐 batch_size。"""
    data_limit = max(8, num_data_per_client // 2)
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        available = total - 500
        per_sample_mb = 25
        fixed_overhead_mb = 200
        gpu_limit = int((available - fixed_overhead_mb) / per_sample_mb)
        gpu_limit = max(8, gpu_limit)
    else:
        gpu_limit = base_batch_size
    recommended = min(data_limit, gpu_limit)
    power = 1
    while power * 2 <= recommended:
        power *= 2
    recommended = max(4, power)
    return recommended


# ==================== 评估指标 ====================

def compute_pixel_accuracy(pred, target):
    """计算像素准确率"""
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def compute_iou_per_class(pred, target, num_classes):
    """计算每个类别的 IoU"""
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious


def compute_miou(pred, target, num_classes):
    """计算 mIoU"""
    ious = compute_iou_per_class(pred, target, num_classes)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0


# ==================== 验证函数 ====================

def evaluate_model(model, val_loader, device, use_amp=True):
    """在验证集上评估模型，返回 (pixel_acc, miou, per_class_iou)"""
    model.eval()
    total_pixel_acc = 0.0
    total_miou = 0.0
    num_samples = 0
    class_iou_sum = [0.0] * NUM_CLASSES
    class_iou_cnt = [0] * NUM_CLASSES

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            with autocast(enabled=use_amp and torch.cuda.is_available()):
                output = model(images)
            preds = output.argmax(dim=1)
            for i in range(preds.size(0)):
                pixel_acc = compute_pixel_accuracy(preds[i], labels[i])
                ious = compute_iou_per_class(preds[i], labels[i], NUM_CLASSES)
                miou = np.nanmean([v for v in ious if not np.isnan(v)]) if any(
                    not np.isnan(v) for v in ious) else 0.0
                total_pixel_acc += pixel_acc
                total_miou += miou
                num_samples += 1
                for cls in range(NUM_CLASSES):
                    if not np.isnan(ious[cls]):
                        class_iou_sum[cls] += ious[cls]
                        class_iou_cnt[cls] += 1

    avg_pixel_acc = total_pixel_acc / num_samples if num_samples > 0 else 0.0
    avg_miou = total_miou / num_samples if num_samples > 0 else 0.0
    per_class_iou = {
        CLASS_NAMES[cls]: (class_iou_sum[cls] / class_iou_cnt[cls]
                           if class_iou_cnt[cls] > 0 else None)
        for cls in range(NUM_CLASSES)
    }

    model.train()
    return avg_pixel_acc, avg_miou, per_class_iou


# ==================== 客户端验证数据保存 ====================

def save_client_val_data(round_idx, client_id, pixel_acc, miou, per_class_iou,
                         loss, num_samples, datasave_dir):
    """
    将单个客户端的验证数据保存为 JSON 文件。

    文件路径：<datasave_dir>/round_{round}/client_{client_id}.json
    """
    round_dir = os.path.join(datasave_dir, f'round_{round_idx + 1}')
    os.makedirs(round_dir, exist_ok=True)

    data = {
        'round': round_idx + 1,
        'client_id': client_id,
        'num_train_samples': num_samples,
        'pixel_accuracy': round(pixel_acc, 6),
        'miou': round(miou, 6),
        'per_class_iou': {
            cls: (round(v, 6) if v is not None else None)
            for cls, v in per_class_iou.items()
        },
        'train_loss': round(loss, 6),
    }

    json_path = os.path.join(round_dir, f'client_{client_id}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return json_path


def save_all_client_csv(client_history, datasave_dir):
    """
    将所有轮次、所有客户端的验证记录汇总保存为 CSV 文件（每轮实时更新）。

    列：Round, Client, Num_Samples, Pixel_Accuracy, mIoU, Train_Loss
    """
    csv_path = os.path.join(datasave_dir, 'client_val_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Client', 'Num_Samples',
                         'Pixel_Accuracy', 'mIoU', 'Train_Loss'])
        for record in client_history:
            writer.writerow([
                record['round'],
                record['client_id'],
                record['num_train_samples'],
                f"{record['pixel_accuracy']:.6f}",
                f"{record['miou']:.6f}",
                f"{record['train_loss']:.6f}",
            ])
    return csv_path


def save_client_curves(client_history, num_clients, datasave_dir):
    """
    绘制并保存各客户端 Pixel Accuracy 和 mIoU 随轮次变化的曲线图。
    """
    client_data = {i: {'rounds': [], 'pixel_acc': [], 'miou': []}
                   for i in range(num_clients)}
    for record in client_history:
        cid = record['client_id']
        client_data[cid]['rounds'].append(record['round'])
        client_data[cid]['pixel_acc'].append(record['pixel_accuracy'])
        client_data[cid]['miou'].append(record['miou'])

    colors = plt.cm.tab10.colors

    # ===== Pixel Accuracy 曲线 =====
    plt.figure(figsize=(12, 6))
    for i in range(num_clients):
        d = client_data[i]
        if d['rounds']:
            plt.plot(d['rounds'], d['pixel_acc'],
                     marker='o', linewidth=1.5, markersize=3,
                     color=colors[i % len(colors)],
                     label=f'Client {i}')
    plt.xlabel('Communication Round', fontsize=13)
    plt.ylabel('Pixel Accuracy', fontsize=13)
    plt.title('Per-Client Pixel Accuracy Before Aggregation', fontsize=15)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    pa_path = os.path.join(datasave_dir, 'pixel_accuracy_clients.png')
    plt.savefig(pa_path, dpi=150)
    plt.close()
    print(f"  已保存: {pa_path}")

    # ===== mIoU 曲线 =====
    plt.figure(figsize=(12, 6))
    for i in range(num_clients):
        d = client_data[i]
        if d['rounds']:
            plt.plot(d['rounds'], d['miou'],
                     marker='s', linewidth=1.5, markersize=3,
                     color=colors[i % len(colors)],
                     label=f'Client {i}')
    plt.xlabel('Communication Round', fontsize=13)
    plt.ylabel('mIoU', fontsize=13)
    plt.title('Per-Client mIoU Before Aggregation', fontsize=15)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    miou_path = os.path.join(datasave_dir, 'miou_clients.png')
    plt.savefig(miou_path, dpi=150)
    plt.close()
    print(f"  已保存: {miou_path}")


# ==================== 全局验证结果保存 ====================

def save_global_results(history, run_dir):
    """
    保存全局模型（聚合后）的实验结果到本次运行根目录：
      - pixel_accuracy.png
      - miou.png
      - results.csv
    """
    rounds = [h['round'] for h in history]
    pixel_accs = [h['pixel_acc'] for h in history]
    mious = [h['miou'] for h in history]

    # ===== 图 1: Pixel Accuracy =====
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, pixel_accs, 'b-o', linewidth=2, markersize=4, label='Pixel Accuracy')
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Pixel Accuracy', fontsize=14)
    plt.title('Global Model Pixel Accuracy vs. Communication Round', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    pa_path = os.path.join(run_dir, 'pixel_accuracy.png')
    plt.savefig(pa_path, dpi=150)
    plt.close()
    print(f"  已保存: {pa_path}")

    # ===== 图 2: mIoU =====
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, mious, 'r-s', linewidth=2, markersize=4, label='mIoU')
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('mIoU', fontsize=14)
    plt.title('Global Model mIoU vs. Communication Round', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    miou_path = os.path.join(run_dir, 'miou.png')
    plt.savefig(miou_path, dpi=150)
    plt.close()
    print(f"  已保存: {miou_path}")

    # ===== 表: results.csv =====
    csv_path = os.path.join(run_dir, 'results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Pixel Accuracy', 'mIoU'])
        for h in history:
            writer.writerow([h['round'], f"{h['pixel_acc']:.6f}", f"{h['miou']:.6f}"])
    print(f"  已保存: {csv_path}")


# ==================== 客户端 ====================

class Client:
    def __init__(self, client_id, dataset, indices, device, use_amp=True):
        self.client_id = client_id
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.train_loader = None
        self.dataset = dataset
        self.indices = indices

    def local_train(self, model, batch_size=32, epochs=1, lr=0.01,
                    num_workers=0, pin_memory=False):
        """本地训练"""
        self.train_loader = DataLoader(
            Subset(self.dataset, self.indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            drop_last=False
        )

        model.train()
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()
        scaler = GradScaler(enabled=self.use_amp)

        epoch_losses = []
        for epoch in range(epochs):
            running_loss = 0.0
            num_batches = 0
            for images, labels in self.train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.use_amp):
                    output = model(images)
                    loss = criterion(output, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                num_batches += 1
            epoch_losses.append(running_loss / max(num_batches, 1))

        avg_loss = np.mean(epoch_losses)
        return model.state_dict(), avg_loss


# ==================== 联邦聚合 ====================

def federated_aggregate(global_model, client_weights, client_lens):
    total_data = sum(client_lens)
    global_dict = copy.deepcopy(client_weights[0])

    for key in global_dict.keys():
        global_dict[key] = global_dict[key] * (client_lens[0] / total_data)

    for i in range(1, len(client_weights)):
        fraction = client_lens[i] / total_data
        for key in global_dict.keys():
            global_dict[key] += client_weights[i][key] * fraction

    global_model.load_state_dict(global_dict)
    return global_model


# ==================== 主函数 ====================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        print(f"GPU 型号: {gpu_name}")
        print(f"GPU 总显存: {total_mem:.0f} MB")
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark: 已启用")
    else:
        print("警告：未检测到 GPU，将使用 CPU 训练（速度会非常慢）")

    from torchvision import transforms

    # ==================== 配置区 ====================
    USE_AMP = True            # 是否启用混合精度训练 (推荐开启)
    TARGET_SIZE = (256, 256)  # 输入图像尺寸
    NUM_ROUNDS = 50           # 联邦训练轮数
    NUM_CLIENTS = 5           # 客户端数量
    LOCAL_EPOCHS = 5          # 每个客户端的本地训练轮数
    LR = 0.01                 # 学习率

    # ★ Windows 多进程修复
    if sys.platform == 'win32':
        NUM_WORKERS = 0
        PIN_MEMORY = True
    else:
        NUM_WORKERS = 4
        PIN_MEMORY = True

    BATCH_SIZE = 0  # 0 = 自动推荐
    # ================================================

    # ===== 生成本次运行的时间戳目录（月日时分，格式 MMDDHHmm） =====
    run_timestamp = datetime.now().strftime('%m%d%H%M')
    run_dir = os.path.join('./result_save', run_timestamp)
    datasave_dir = os.path.join(run_dir, 'datasave')
    os.makedirs(datasave_dir, exist_ok=True)
    print(f"\n本次运行结果将保存至: {run_dir}/")

    # ===== 1. 加载数据集 =====
    train_dataset = CamVidDataset('./data', split='train',
                                  transform=transforms.ToTensor(),
                                  target_size=TARGET_SIZE)
    val_dataset = CamVidDataset('./data', split='val',
                                transform=transforms.ToTensor(),
                                target_size=TARGET_SIZE)

    num_images = len(train_dataset)
    indices = np.arange(num_images)
    np.random.shuffle(indices)
    user_groups = np.array_split(indices, NUM_CLIENTS)

    avg_data_per_client = num_images // NUM_CLIENTS
    min_data_per_client = min(len(g) for g in user_groups)

    # ===== 自动推荐 batch_size =====
    if BATCH_SIZE == 0:
        BATCH_SIZE = auto_batch_size(device, min_data_per_client)

    # 打印训练配置
    print(f"\n{'='*60}")
    print(f"训练配置:")
    print(f"  训练集: {num_images} 张 | 验证集: {len(val_dataset)} 张")
    print(f"  客户端: {NUM_CLIENTS} 个 | 每客户端约 {avg_data_per_client} 张 (最少 {min_data_per_client} 张)")
    print(f"  Batch Size: {BATCH_SIZE} (每客户端每 epoch 约 {min_data_per_client // BATCH_SIZE + 1} 步)")
    print(f"  Local Epochs: {LOCAL_EPOCHS} | 联邦轮数: {NUM_ROUNDS}")
    print(f"  每客户端每轮总梯度更新步数: ~{LOCAL_EPOCHS * (min_data_per_client // BATCH_SIZE + 1)}")
    print(f"  学习率: {LR}")
    print(f"  混合精度 (AMP): {'已启用' if USE_AMP and torch.cuda.is_available() else '未启用'}")
    print(f"  数据加载进程数: {NUM_WORKERS}")
    print(f"{'='*60}")

    # ===== 2. 初始化全局模型 =====
    global_model = MobileNetV2UNet(num_classes=NUM_CLASSES).to(device)

    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params:,} ({total_params * 4 / 1024 ** 2:.1f} MB in FP32)")
    print(f"可训练参数量: {trainable_params:,}")

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # ===== 3. 创建模型检查点目录 =====
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # ===== 4. 显存基线 =====
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    print_gpu_status(device, "训练前基线")

    # ===== 5. 训练循环 =====
    history = []          # 全局模型（聚合后）验证历史
    client_history = []   # 所有客户端聚合前验证历史（用于汇总 CSV）
    best_miou = 0.0

    print(f"\n{'='*80}")
    print(f"开始联邦训练（含客户端聚合前验证）...")
    print(f"{'='*80}")

    total_train_time = 0.0

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n--- Round {round_idx + 1}/{NUM_ROUNDS} ---")

        local_weights = []
        local_lens = []
        round_losses = []

        # ===== 每个客户端：本地训练 → 聚合前验证 → 收集权重 =====
        for i in range(NUM_CLIENTS):
            client = Client(i, train_dataset, user_groups[i], device, use_amp=USE_AMP)
            local_model = copy.deepcopy(global_model)

            if i == 0 and round_idx == 0 and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            # --- 本地训练 ---
            weights, avg_loss = client.local_train(
                local_model,
                batch_size=BATCH_SIZE,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )

            if i == 0 and round_idx == 0 and torch.cuda.is_available():
                train_peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2
                total_gpu = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
                print(f"  [训练时峰值显存] {train_peak:.0f}MB / {total_gpu:.0f}MB "
                      f"({train_peak/total_gpu*100:.1f}%)")

            # --- 聚合前：在验证集上评估该客户端本地模型 ---
            local_model.load_state_dict(weights)
            pixel_acc, miou, per_class_iou = evaluate_model(
                local_model, val_loader, device, use_amp=USE_AMP
            )
            print(f"  [Client {i}] Loss: {avg_loss:.4f} | "
                  f"Val Pixel Acc: {pixel_acc:.4f} | Val mIoU: {miou:.4f}")

            # 保存单个客户端 JSON（存入 run_dir/datasave/）
            save_client_val_data(
                round_idx=round_idx,
                client_id=i,
                pixel_acc=pixel_acc,
                miou=miou,
                per_class_iou=per_class_iou,
                loss=avg_loss,
                num_samples=len(user_groups[i]),
                datasave_dir=datasave_dir
            )

            # 追加到汇总列表
            client_history.append({
                'round': round_idx + 1,
                'client_id': i,
                'num_train_samples': len(user_groups[i]),
                'pixel_accuracy': pixel_acc,
                'miou': miou,
                'train_loss': avg_loss,
            })

            local_weights.append(weights)
            local_lens.append(len(user_groups[i]))
            round_losses.append(avg_loss)

            del local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        avg_round_loss = np.mean(round_losses)

        # ----- 服务端聚合 -----
        global_model = federated_aggregate(global_model, local_weights, local_lens)

        del local_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        round_time = time.time() - round_start
        total_train_time += round_time

        # ----- 每轮全局模型验证（聚合后） -----
        pixel_acc, miou, _ = evaluate_model(global_model, val_loader, device, use_amp=USE_AMP)
        history.append({
            'round': round_idx + 1,
            'pixel_acc': pixel_acc,
            'miou': miou,
            'loss': avg_round_loss,
            'time': round_time
        })

        print(f"  [Global] Loss: {avg_round_loss:.4f} | "
              f"Pixel Acc: {pixel_acc:.4f} | mIoU: {miou:.4f} | 耗时: {round_time:.1f}s",
              end="")

        # ----- 保存最优模型 -----
        if miou > best_miou:
            best_miou = miou
            best_path = os.path.join(save_dir, 'best_model.pth')
            torch.save({
                'round': round_idx + 1,
                'model_state_dict': global_model.state_dict(),
                'num_classes': NUM_CLASSES,
                'pixel_acc': pixel_acc,
                'miou': miou,
            }, best_path)
            print(f"  ★ Best (mIoU: {miou:.4f})")
        else:
            print()

        # ----- 每 10 轮保存检查点 -----
        if (round_idx + 1) % 10 == 0:
            ckpt_path = os.path.join(save_dir, f'global_model_round_{round_idx + 1}.pth')
            torch.save({
                'round': round_idx + 1,
                'model_state_dict': global_model.state_dict(),
                'num_classes': NUM_CLASSES,
                'pixel_acc': pixel_acc,
                'miou': miou,
            }, ckpt_path)
            print(f"  >> 检查点已保存: {ckpt_path}")

        # ----- 每轮更新汇总 CSV（实时写入，防止中途崩溃丢失数据） -----
        save_all_client_csv(client_history, datasave_dir)

    # ===== 6. 保存最终模型 =====
    final_path = os.path.join(save_dir, 'global_model_final.pth')
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'num_classes': NUM_CLASSES,
        'num_rounds': NUM_ROUNDS,
        'final_pixel_acc': history[-1]['pixel_acc'],
        'final_miou': history[-1]['miou'],
    }, final_path)

    # ===== 7. 保存客户端验证曲线图（存入 run_dir/datasave/） =====
    print(f"\n{'='*60}")
    print(f"正在生成客户端验证曲线图...")
    save_client_curves(client_history, NUM_CLIENTS, datasave_dir)

    # ===== 8. 生成全局模型实验结果图表（存入 run_dir/） =====
    print(f"\n正在生成全局模型实验结果图表...")
    save_global_results(history, run_dir)
    print(f"{'='*60}")

    # ===== 9. 打印训练总结 =====
    print(f"\n{'='*80}")
    print(f"训练完成！总结如下：")
    print(f"{'='*80}")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        total_mem = torch.cuda.get_device_properties(device).total_memory / 1024 ** 2
        print(f"\n[GPU 显存总结]")
        print(f"  GPU 型号:     {torch.cuda.get_device_name(device)}")
        print(f"  总显存:       {total_mem:.0f} MB")
        print(f"  训练峰值显存: {peak_mem:.0f} MB")
        print(f"  显存利用率:   {(peak_mem / total_mem) * 100:.1f}%")
        print(f"  AMP 混合精度: {'已启用' if USE_AMP else '未启用'}")
        print(f"  Batch Size:   {BATCH_SIZE}")

    print(f"\n[训练性能总结]")
    print(f"  总训练时间:   {total_train_time:.1f}s ({total_train_time / 60:.1f} min)")
    print(f"  平均每轮耗时: {total_train_time / NUM_ROUNDS:.1f}s")

    print(f"\n{'Round':<8} {'Loss':<12} {'Pixel Acc':<14} {'mIoU':<14} {'耗时(s)':<10}")
    print(f"{'-'*58}")
    for h in history:
        print(f"{h['round']:<8} {h['loss']:<12.4f} {h['pixel_acc']:<14.4f} "
              f"{h['miou']:<14.4f} {h['time']:<10.1f}")
    print(f"{'-'*58}")
    print(f"\n最优 mIoU: {best_miou:.4f}")
    print(f"最优模型: {os.path.join(save_dir, 'best_model.pth')}")
    print(f"最终模型: {final_path}")
    print(f"\n本次运行所有结果已保存至: {run_dir}/")
    print(f"  ├── datasave/                  ← 客户端聚合前验证数据")
    print(f"  │   ├── client_val_results.csv")
    print(f"  │   ├── round_*/client_*.json")
    print(f"  │   ├── pixel_accuracy_clients.png")
    print(f"  │   └── miou_clients.png")
    print(f"  ├── pixel_accuracy.png         ← 全局模型曲线")
    print(f"  ├── miou.png")
    print(f"  └── results.csv")


if __name__ == "__main__":
    main()
