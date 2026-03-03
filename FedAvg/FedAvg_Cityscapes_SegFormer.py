"""
FedAvg_Cityscapes_SegFormer.py — 标准联邦平均（FedAvg）基线方法
                                  专用于 Cityscapes 数据集 + SegFormer-B0 模型

用途
----
作为 FedGMHC-SegFormer 方法的对比基线，使用相同的模型（SegFormer-B0）、
相同的数据集（Cityscapes）和相同的 Dirichlet Non-IID 数据划分，
仅执行标准 FedAvg 聚合，不做任何聚类或分组操作。

模型说明
--------
SegFormer-B0 风格模型（PVTv2-B0 编码器 + All-MLP 解码器）：
  - 编码器：pvt_v2_b0（层次化 Transformer，3.4M 参数，含 30 个 LayerNorm 层）
  - 解码器：SegFormer All-MLP Decoder（embed_dim=256，0.4M 参数）
  - 总参数量：约 3.8M（比 MobileNetV2-UNet 6.6M 更轻量）
  - 使用 ImageNet 预训练编码器权重（pretrained=True）

算法流程
--------
每轮：
  a. 每个客户端从全局模型出发，完成本地训练。
  b. 所有客户端按数据量加权 FedAvg，更新全局模型。
  c. 在验证集上评估全局模型，记录结果。

结果保存
--------
每次运行结果统一保存在 result_save/FedAvg_SF_MMDDHHmm/ 子目录下：
  result_save/
  └── FedAvg_SF_MMDDHHmm/
      ├── global_val_results.csv    ← 每轮全局模型验证数据汇总表（实时更新）
      ├── pixel_accuracy.png        ← 全局模型 Pixel Accuracy 折线图
      └── miou.png                  ← 全局模型 mIoU 折线图
"""

import torch
import copy
import csv
import os
import sys
import time
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import build_segformer_b0
from dataset.cityscapes_dataset import (
    CityscapesDataset,
    NUM_CLASSES,
    CLASS_NAMES,
    IGNORE_INDEX,
    build_label_index_cityscapes,
)
from partition import dirichlet_partition, print_partition_stats


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

def save_global_csv(global_history, run_dir):
    path = os.path.join(run_dir, 'global_val_results.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Round', 'Pixel_Accuracy', 'mIoU', 'Avg_Loss', 'Time_s'])
        for r in global_history:
            writer.writerow([
                r['round'],
                f"{r['pixel_acc']:.6f}",
                f"{r['miou']:.6f}",
                f"{r['avg_loss']:.6f}",
                f"{r['time']:.1f}",
            ])
    return path


def save_curves(global_history, run_dir):
    """生成 Pixel Accuracy 和 mIoU 随轮次变化的折线图。"""
    rounds    = [r['round']     for r in global_history]
    pa_vals   = [r['pixel_acc'] for r in global_history]
    miou_vals = [r['miou']      for r in global_history]

    for ylabel, title, suffix, ydata in [
        ('Pixel Accuracy',
         'Cityscapes FedAvg (SegFormer-B0) — Pixel Accuracy vs. Round',
         'pixel_accuracy.png', pa_vals),
        ('mIoU',
         'Cityscapes FedAvg (SegFormer-B0) — mIoU vs. Round',
         'miou.png', miou_vals),
    ]:
        plt.figure(figsize=(12, 5))
        plt.plot(rounds, ydata,
                 linestyle='-', marker='o', linewidth=2, markersize=4,
                 color='steelblue', label='FedAvg Global (SegFormer-B0)')
        plt.xlabel('Communication Round', fontsize=13)
        plt.ylabel(ylabel, fontsize=13)
        plt.title(title, fontsize=15)
        plt.legend(fontsize=11, loc='lower right')
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
    NUM_ROUNDS   = 50           # 训练轮数
    NUM_CLIENTS  = 10           # 客户端数量
    LOCAL_EPOCHS = 1            # 每轮本地训练轮数
    # SegFormer 使用 AdamW，学习率比 SGD 小一个数量级
    LR           = 1e-4
    NUM_WORKERS  = 0 if sys.platform == 'win32' else 4
    PIN_MEMORY   = True
    BATCH_SIZE   = 4            # 4060 Ti 8GB + 256×512 + AMP，SegFormer 约 12MB/张
    DIRICHLET_ALPHA = 0.5       # 异质性参数（0.5 = 强异质性）
    MIN_SAMPLES     = 100       # 每个客户端最少图像数量
    MAX_SAMPLES     = 200       # 每个客户端最多图像数量（None = 不限制）
    PRETRAINED      = True      # 是否使用 ImageNet 预训练编码器权重
    # ================================================

    # ===== 时间戳运行目录 =====
    run_timestamp = datetime.now().strftime('%m%d%H%M')
    run_dir = os.path.join('../result_save', f'FedAvg_SF_{run_timestamp}')
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
    _max_str = f' | 最多 {MAX_SAMPLES} 张/客户端' if MAX_SAMPLES is not None else ''

    print(f"\n{'='*65}")
    print(f"训练配置（FedAvg Baseline — Cityscapes + SegFormer-B0）:")
    print(f"  数据集根目录: {DATASET_ROOT}")
    print(f"  图像尺寸: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}  |  训练类别: {NUM_CLASSES}  |  IGNORE_INDEX: {IGNORE_INDEX}")
    print(f"  训练集: {num_images} 张 | 验证集: {len(val_dataset)} 张")
    print(f"  客户端: {NUM_CLIENTS} 个 | Dirichlet α={DIRICHLET_ALPHA} | 最少 {min_data} 张{_max_str}")
    print(f"  Batch Size: {BATCH_SIZE} | Local Epochs: {LOCAL_EPOCHS} | 联邦轮数: {NUM_ROUNDS}")
    print(f"  学习率: {LR} (AdamW) | AMP: {'已启用' if USE_AMP and torch.cuda.is_available() else '未启用'}")
    print(f"  预训练编码器: {'是' if PRETRAINED else '否'}")
    print(f"{'='*65}")

    # ===== 初始化全局模型（SegFormer-B0）=====
    print(f"\n正在初始化 SegFormer-B0 模型（pretrained={PRETRAINED}）...")
    global_model = build_segformer_b0(num_classes=NUM_CLASSES, pretrained=PRETRAINED).to(device)
    total_params   = sum(p.numel() for p in global_model.parameters())
    encoder_params = sum(p.numel() for p in global_model.encoder.parameters())
    decoder_params = sum(p.numel() for p in global_model.decoder.parameters())
    print(f"模型总参数量: {total_params:,} ({total_params * 4 / 1024**2:.1f} MB in FP32)")
    print(f"  编码器 (PVTv2-B0): {encoder_params:,} 参数")
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

    global_history = []
    best_miou      = 0.0
    total_time     = 0.0

    print(f"\n{'='*80}")
    print(f"开始 FedAvg 训练（Cityscapes + SegFormer-B0）— 标准基线")
    print(f"{'='*80}")

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n--- Round {round_idx + 1}/{NUM_ROUNDS} ---")

        # ================================================================
        # 阶段 A：每个客户端从全局模型出发，完成本地训练
        # ================================================================
        local_weights = []
        local_losses  = []
        local_lens    = []

        for i in range(NUM_CLIENTS):
            client      = Client(i, train_dataset, user_groups[i], device, use_amp=USE_AMP)
            start_model = copy.deepcopy(global_model)

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
        # 阶段 B：标准 FedAvg 全局聚合
        # ================================================================
        global_model = fedavg(global_model, local_weights, local_lens)

        del local_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ================================================================
        # 阶段 C：验证全局模型
        # ================================================================
        round_time  = time.time() - round_start
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
                'method':           'FedAvg_SegFormer',
                'model':            'SegFormerB0',
            }, os.path.join(save_dir, 'best_model_fedavg_segformer_cityscapes.pth'))
            print(f"  ★ Best (mIoU: {g_miou:.4f})")
        else:
            print()

        if (round_idx + 1) % 10 == 0:
            ckpt = os.path.join(save_dir, f'fedavg_segformer_cityscapes_round_{round_idx + 1}.pth')
            torch.save({
                'round':            round_idx + 1,
                'model_state_dict': global_model.state_dict(),
                'num_classes':      NUM_CLASSES,
                'pixel_acc':        g_pa,
                'miou':             g_miou,
                'dataset':          'cityscapes',
                'target_size':      TARGET_SIZE,
                'method':           'FedAvg_SegFormer',
                'model':            'SegFormerB0',
            }, ckpt)
            print(f"  >> 检查点已保存: {ckpt}")

        # 每轮实时更新 CSV（防止意外中断丢失数据）
        save_global_csv(global_history, run_dir)

    # ===== 保存最终全局模型 =====
    final_path = os.path.join(save_dir, 'fedavg_segformer_cityscapes_final.pth')
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'num_classes':      NUM_CLASSES,
        'num_rounds':       NUM_ROUNDS,
        'final_pixel_acc':  global_history[-1]['pixel_acc'],
        'final_miou':       global_history[-1]['miou'],
        'dataset':          'cityscapes',
        'target_size':      TARGET_SIZE,
        'method':           'FedAvg_SegFormer',
        'model':            'SegFormerB0',
    }, final_path)

    # ===== 生成折线图 =====
    print(f"\n{'='*60}")
    print(f"正在生成折线图...")
    save_curves(global_history, run_dir)

    # ===== 打印训练总结 =====
    print(f"\n{'='*80}")
    print(f"训练完成！总结如下（FedAvg Baseline — Cityscapes + SegFormer-B0）：")
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
    print(f"{'-'*60}")
    for h in global_history:
        print(f"{h['round']:<8} {h['avg_loss']:<12.4f} "
              f"{h['pixel_acc']:<14.4f} {h['miou']:<14.4f} {h['time']:<10.1f}")
    print(f"{'-'*60}")

    print(f"\n本次运行所有结果已保存至: {run_dir}/")
    print(f"  ├── global_val_results.csv    ← 每轮全局模型验证数据")
    print(f"  ├── pixel_accuracy.png        ← Pixel Accuracy 折线图")
    print(f"  └── miou.png                  ← mIoU 折线图")
    print(f"\n最优模型: {os.path.join(save_dir, 'best_model_fedavg_segformer_cityscapes.pth')}")
    print(f"最终模型: {final_path}")


if __name__ == "__main__":
    main()
