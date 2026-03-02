import torch
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

from model import MobileNetV2UNet
from dataset import CamVidDataset, NUM_CLASSES, CLASS_NAMES
from partition import build_label_index, dirichlet_partition, print_partition_stats

import os
import sys
import time
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境下也能生成图片
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
    """在验证集上评估模型"""
    model.eval()
    total_pixel_acc = 0.0
    total_miou = 0.0
    num_samples = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast(enabled=use_amp and torch.cuda.is_available()):
                output = model(images)

            preds = output.argmax(dim=1)

            for i in range(preds.size(0)):
                pixel_acc = compute_pixel_accuracy(preds[i], labels[i])
                miou = compute_miou(preds[i], labels[i], NUM_CLASSES)
                total_pixel_acc += pixel_acc
                total_miou += miou
                num_samples += 1

    avg_pixel_acc = total_pixel_acc / num_samples if num_samples > 0 else 0.0
    avg_miou = total_miou / num_samples if num_samples > 0 else 0.0

    model.train()
    return avg_pixel_acc, avg_miou


# ==================== 结果保存 ====================

def save_results(history, save_dir):
    """
    保存实验结果：两张图 + 一张表

    1. pixel_accuracy.png  - 像素准确率随训练轮次变化曲线
    2. miou.png            - mIoU 随训练轮次变化曲线
    3. results.csv         - 训练轮次、像素准确率、mIoU 的完整表格
    """
    os.makedirs(save_dir, exist_ok=True)

    rounds = [h['round'] for h in history]
    pixel_accs = [h['pixel_acc'] for h in history]
    mious = [h['miou'] for h in history]

    # ===== 图 1: Pixel Accuracy =====
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, pixel_accs, 'b-o', linewidth=2, markersize=4, label='Pixel Accuracy')
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('Pixel Accuracy', fontsize=14)
    plt.title('Pixel Accuracy vs. Communication Round', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    pa_path = os.path.join(save_dir, 'pixel_accuracy.png')
    plt.savefig(pa_path, dpi=150)
    plt.close()
    print(f"  已保存: {pa_path}")

    # ===== 图 2: mIoU =====
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, mious, 'r-s', linewidth=2, markersize=4, label='mIoU')
    plt.xlabel('Communication Round', fontsize=14)
    plt.ylabel('mIoU', fontsize=14)
    plt.title('mIoU vs. Communication Round', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    miou_path = os.path.join(save_dir, 'miou.png')
    plt.savefig(miou_path, dpi=150)
    plt.close()
    print(f"  已保存: {miou_path}")

    # ===== 表: results.csv =====
    csv_path = os.path.join(save_dir, 'results.csv')
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

    def local_train(self, model, batch_size=32, epochs=1, lr=0.01, num_workers=0, pin_memory=False):
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

    # ===== GPU 信息 =====
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
    NUM_CLIENTS = 10          # 客户端数量
    LOCAL_EPOCHS = 5          # 每个客户端的本地训练轮数
    LR = 0.01                 # 学习率
    DIRICHLET_ALPHA = 1.0     # Dirichlet 浓度参数（越小异质性越强；推荐 0.5/1.0/2.0）
    MIN_SAMPLES     = 20      # 每个客户端最少图像数量

    # ★ Windows 多进程修复
    if sys.platform == 'win32':
        NUM_WORKERS = 0
        PIN_MEMORY = True
    else:
        NUM_WORKERS = 4
        PIN_MEMORY = True

    BATCH_SIZE = 0  # 0 = 自动推荐
    # ================================================

    # ===== 1. 加载数据集（从 dataset.py 导入） =====
    train_dataset = CamVidDataset('./data', split='train', transform=transforms.ToTensor(), target_size=TARGET_SIZE)
    val_dataset = CamVidDataset('./data', split='val', transform=transforms.ToTensor(), target_size=TARGET_SIZE)

    num_images = len(train_dataset)

    # ===== Dirichlet Non-IID 数据划分 =====
    print(f"\n  [Partition] 使用 Dirichlet(α={DIRICHLET_ALPHA}) Non-IID 划分...")
    labels = build_label_index('./data', split='train', num_classes=NUM_CLASSES,
                               target_size=TARGET_SIZE)
    user_groups = dirichlet_partition(
        num_clients=NUM_CLIENTS, labels=labels, num_classes=NUM_CLASSES,
        alpha=DIRICHLET_ALPHA, min_samples=MIN_SAMPLES, seed=42)
    print_partition_stats(user_groups, labels, NUM_CLASSES, CLASS_NAMES)

    min_data_per_client = min(len(g) for g in user_groups)

    # ===== 自动推荐 batch_size =====
    if BATCH_SIZE == 0:
        BATCH_SIZE = auto_batch_size(device, min_data_per_client)

    # 打印训练配置
    print(f"\n{'='*60}")
    print(f"训练配置:")
    print(f"  训练集: {num_images} 张 | 验证集: {len(val_dataset)} 张")
    print(f"  客户端: {NUM_CLIENTS} 个 | Dirichlet α={DIRICHLET_ALPHA} | 最少 {min_data_per_client} 张/客户端")
    print(f"  Batch Size: {BATCH_SIZE} | Local Epochs: {LOCAL_EPOCHS} | 联邦轮数: {NUM_ROUNDS}")
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

    # ===== 3. 创建保存目录 =====
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    result_dir = './result_save'
    os.makedirs(result_dir, exist_ok=True)

    # ===== 4. 显存基线 =====
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    print_gpu_status(device, "训练前基线")

    # ===== 5. 训练循环 =====
    history = []
    best_miou = 0.0

    print(f"\n{'='*80}")
    print(f"开始联邦训练...")
    print(f"{'='*80}")

    total_train_time = 0.0

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n--- Round {round_idx + 1}/{NUM_ROUNDS} ---")

        local_weights = []
        local_lens = []
        round_losses = []

        for i in range(NUM_CLIENTS):
            client = Client(i, train_dataset, user_groups[i], device, use_amp=USE_AMP)
            local_model = copy.deepcopy(global_model)

            if i == 0 and round_idx == 0 and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

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
                print(f"  [训练时峰值显存] {train_peak:.0f}MB / {total_gpu:.0f}MB ({train_peak/total_gpu*100:.1f}%)")

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

        # ----- 每轮验证 -----
        pixel_acc, miou = evaluate_model(global_model, val_loader, device, use_amp=USE_AMP)
        history.append({
            'round': round_idx + 1,
            'pixel_acc': pixel_acc,
            'miou': miou,
            'loss': avg_round_loss,
            'time': round_time
        })

        print(f"  Loss: {avg_round_loss:.4f} | Pixel Acc: {pixel_acc:.4f} | mIoU: {miou:.4f} | 耗时: {round_time:.1f}s", end="")

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

    # ===== 6. 保存最终模型 =====
    final_path = os.path.join(save_dir, 'global_model_final.pth')
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'num_classes': NUM_CLASSES,
        'num_rounds': NUM_ROUNDS,
        'final_pixel_acc': history[-1]['pixel_acc'],
        'final_miou': history[-1]['miou'],
    }, final_path)

    # ===== 7. 生成实验结果图表 =====
    print(f"\n{'='*60}")
    print(f"正在生成实验结果图表...")
    save_results(history, result_dir)
    print(f"所有结果已保存到: {result_dir}/")
    print(f"{'='*60}")

    # ===== 8. 打印训练总结 =====
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
        print(f"{h['round']:<8} {h['loss']:<12.4f} {h['pixel_acc']:<14.4f} {h['miou']:<14.4f} {h['time']:<10.1f}")
    print(f"{'-'*58}")
    print(f"\n最优 mIoU: {best_miou:.4f}")
    print(f"最优模型: {os.path.join(save_dir, 'best_model.pth')}")
    print(f"最终模型: {final_path}")
    print(f"实验结果: {result_dir}/")


if __name__ == "__main__":
    main()
