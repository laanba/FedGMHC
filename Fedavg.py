import torch
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, Subset, Dataset
from torch.cuda.amp import autocast, GradScaler

from demo import MobileNetV2UNet
import os
import sys
import time
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# ===== CamVid 12 类 RGB 颜色映射表 =====
CAMVID_COLORS = np.array([
    [0,     0,   0],    # 0:  Void / Unlabelled
    [0,     0, 192],    # 1:  Pavement
    [0,   128, 192],    # 2:  Bicyclist
    [64,    0, 128],    # 3:  Car
    [64,   64,   0],    # 4:  Pedestrian
    [64,   64, 128],    # 5:  Fence
    [128,   0,   0],    # 6:  Building
    [128,  64, 128],    # 7:  Road
    [128, 128,   0],    # 8:  Tree
    [128, 128, 128],    # 9:  Sky
    [192, 128, 128],    # 10: SignSymbol / Pole
    [192, 192, 128],    # 11: Column_Pole
], dtype=np.uint8)

CLASS_NAMES = [
    'Void', 'Pavement', 'Bicyclist', 'Car', 'Pedestrian', 'Fence',
    'Building', 'Road', 'Tree', 'Sky', 'SignSymbol', 'Column_Pole'
]

NUM_CLASSES = len(CAMVID_COLORS)  # 12


def rgb_mask_to_class_index(mask_rgb, color_map=CAMVID_COLORS):
    """将 RGB 掩码图转换为类别索引图"""
    mask = np.array(mask_rgb)
    h, w = mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int64)
    for cls_idx, color in enumerate(color_map):
        match = np.all(mask == color, axis=-1)
        class_mask[match] = cls_idx
    return class_mask


# ==================== 显存监控工具 ====================

def get_gpu_memory_info(device):
    """
    获取 GPU 显存信息（单位：MB）
    返回: (已分配, 已缓存, 总显存, 空闲显存)
    """
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated(device) / 1024 ** 2
    cached = torch.cuda.memory_reserved(device) / 1024 ** 2
    total = torch.cuda.get_device_properties(device).total_mem / 1024 ** 2
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


def auto_batch_size(device, base_batch_size=32):
    """
    根据 GPU 空闲显存自动推荐 batch_size
    MobileNetV2-UNet 在 256x256 输入下，每增加 batch_size 1 约需 ~40MB 显存
    """
    if not torch.cuda.is_available():
        return base_batch_size

    total = torch.cuda.get_device_properties(device).total_mem / 1024 ** 2
    # 预留 500MB 给系统和其他进程
    available = total - 500

    # 估算：模型固定开销 ~200MB，每个样本 ~25MB (AMP 模式)
    per_sample_mb = 25
    fixed_overhead_mb = 200

    recommended = int((available - fixed_overhead_mb) / per_sample_mb)
    # 取最近的 2 的幂次，且不小于 8
    recommended = max(8, min(recommended, 512))
    # 向下取到最近的 2 的幂次
    power = 1
    while power * 2 <= recommended:
        power *= 2
    recommended = power

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


# ==================== 客户端 & 数据集 ====================

class Client:
    def __init__(self, client_id, dataset, indices, device, use_amp=True):
        self.client_id = client_id
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.train_loader = None
        self.dataset = dataset
        self.indices = indices

    def local_train(self, model, batch_size=32, epochs=1, lr=0.01, num_workers=0, pin_memory=False):
        """
        加速优化点：
        1. AMP 混合精度训练 (FP16)：显存减半，速度提升 1.5-3x
        2. 更大的 batch_size：充分利用空闲显存
        3. num_workers > 0：多进程数据加载，减少 CPU-GPU 等待
        4. pin_memory=True：加速 CPU→GPU 数据传输
        5. cudnn.benchmark=True：自动选择最优卷积算法
        """
        self.train_loader = DataLoader(
            Subset(self.dataset, self.indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),  # 避免每次重建子进程的开销
            drop_last=False
        )

        model.train()
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = torch.nn.CrossEntropyLoss()

        scaler = GradScaler(enabled=self.use_amp)

        for epoch in range(epochs):
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

        return model.state_dict()


class CamVidDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, target_size=(256, 256)):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, split)
        self.mask_dir = os.path.join(root_dir, f'{split}_labels')
        self.images = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.target_size is not None:
            image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
            mask = mask.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)

        if self.transform:
            image = self.transform(image)
            mask = torch.from_numpy(rgb_mask_to_class_index(mask)).long()

        return image, mask


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
        total_mem = torch.cuda.get_device_properties(device).total_mem / 1024 ** 2
        print(f"GPU 型号: {gpu_name}")
        print(f"GPU 总显存: {total_mem:.0f} MB")

        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark: 已启用")
    else:
        print("警告：未检测到 GPU，将使用 CPU 训练（速度会非常慢）")

    from torchvision import transforms

    # ==================== 配置区 ====================
    # 您可以根据自己的 GPU 显存大小调整以下参数

    USE_AMP = True            # 是否启用混合精度训练 (推荐开启)
    TARGET_SIZE = (256, 256)  # 输入图像尺寸
    NUM_ROUNDS = 20           # 联邦训练轮数
    NUM_CLIENTS = 5           # 客户端数量
    LOCAL_EPOCHS = 1          # 每个客户端的本地训练轮数
    LR = 0.01                 # 学习率

    # ★ Windows 多进程修复：
    # Windows 上 num_workers > 0 需要在 if __name__ == "__main__" 保护下运行，
    # 否则子进程会重复执行主模块导致卡死。
    # 如果您在 Windows 上仍然遇到卡死问题，请将 NUM_WORKERS 设为 0。
    if sys.platform == 'win32':
        NUM_WORKERS = 0       # Windows 默认使用 0，避免多进程卡死
        PIN_MEMORY = True     # 锁页内存仍然可以启用
    else:
        NUM_WORKERS = 4       # Linux/Mac 可以使用多进程加载
        PIN_MEMORY = True

    # BATCH_SIZE: 设为 0 表示自动推荐，或手动指定一个固定值
    BATCH_SIZE = 0  # 0 = 自动推荐

    # ================================================

    # ===== 自动推荐 batch_size =====
    if BATCH_SIZE == 0:
        BATCH_SIZE = auto_batch_size(device)
    print(f"\n实际 Batch Size: {BATCH_SIZE}")
    print(f"混合精度训练 (AMP): {'已启用' if USE_AMP and torch.cuda.is_available() else '未启用'}")
    print(f"数据加载进程数: {NUM_WORKERS}")
    print(f"锁页内存 (Pin Memory): {'已启用' if PIN_MEMORY else '未启用'}")

    # ===== 1. 初始化全局模型 =====
    global_model = MobileNetV2UNet(num_classes=NUM_CLASSES).to(device)

    # 打印模型参数量
    total_params = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params:,} ({total_params * 4 / 1024 ** 2:.1f} MB in FP32)")
    print(f"可训练参数量: {trainable_params:,}")

    # ===== 2. 加载数据集 =====
    train_dataset = CamVidDataset('./data', split='train', transform=transforms.ToTensor(), target_size=TARGET_SIZE)
    val_dataset = CamVidDataset('./data', split='val', transform=transforms.ToTensor(), target_size=TARGET_SIZE)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    num_images = len(train_dataset)
    indices = np.arange(num_images)
    np.random.shuffle(indices)
    user_groups = np.array_split(indices, NUM_CLIENTS)

    # ===== 3. 创建保存目录 =====
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # ===== 4. 显存基线 =====
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    print_gpu_status(device, "训练前基线")

    # ===== 5. 训练循环 =====
    history = []
    best_miou = 0.0

    print(f"\n训练集: {len(train_dataset)} 张 | 验证集: {len(val_dataset)} 张 | "
          f"客户端: {NUM_CLIENTS} | 总轮数: {NUM_ROUNDS}")
    print(f"{'='*80}")

    total_train_time = 0.0

    for round_idx in range(NUM_ROUNDS):
        round_start = time.time()
        print(f"\n--- Round {round_idx + 1}/{NUM_ROUNDS} ---")

        local_weights = []
        local_lens = []

        for i in range(NUM_CLIENTS):
            client = Client(i, train_dataset, user_groups[i], device, use_amp=USE_AMP)
            local_model = copy.deepcopy(global_model)
            weights = client.local_train(
                local_model,
                batch_size=BATCH_SIZE,
                epochs=LOCAL_EPOCHS,
                lr=LR,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )
            local_weights.append(weights)
            local_lens.append(len(user_groups[i]))

            # 及时释放不再需要的本地模型，回收显存
            del local_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ----- 服务端聚合 -----
        global_model = federated_aggregate(global_model, local_weights, local_lens)

        # 释放本地权重
        del local_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        round_time = time.time() - round_start
        total_train_time += round_time

        # ----- 显存监控 -----
        print_gpu_status(device, "聚合后")
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2
            print(f"  [峰值显存] {peak_mem:.0f} MB")

        # ----- 每轮验证 -----
        pixel_acc, miou = evaluate_model(global_model, val_loader, device, use_amp=USE_AMP)
        history.append({
            'round': round_idx + 1,
            'pixel_acc': pixel_acc,
            'miou': miou,
            'time': round_time
        })

        print(f"  [验证] Pixel Acc: {pixel_acc:.4f} | mIoU: {miou:.4f} | 耗时: {round_time:.1f}s", end="")

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
            print(f"  ★ 新最优模型 (mIoU: {miou:.4f})")
        else:
            print()

        # ----- 每 5 轮保存检查点 -----
        if (round_idx + 1) % 5 == 0:
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

    # ===== 7. 打印训练总结 =====
    print(f"\n{'='*80}")
    print(f"训练完成！总结如下：")
    print(f"{'='*80}")

    # GPU 显存总结
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        total_mem = torch.cuda.get_device_properties(device).total_mem / 1024 ** 2
        print(f"\n[GPU 显存总结]")
        print(f"  GPU 型号:     {torch.cuda.get_device_name(device)}")
        print(f"  总显存:       {total_mem:.0f} MB")
        print(f"  训练峰值显存: {peak_mem:.0f} MB")
        print(f"  显存利用率:   {(peak_mem / total_mem) * 100:.1f}%")
        print(f"  AMP 混合精度: {'已启用' if USE_AMP else '未启用'}")
        print(f"  Batch Size:   {BATCH_SIZE}")

    # 性能总结
    print(f"\n[训练性能总结]")
    print(f"  总训练时间:   {total_train_time:.1f}s ({total_train_time / 60:.1f} min)")
    print(f"  平均每轮耗时: {total_train_time / NUM_ROUNDS:.1f}s")

    print(f"\n{'Round':<8} {'Pixel Acc':<14} {'mIoU':<14} {'耗时(s)':<10}")
    print(f"{'-'*46}")
    for h in history:
        print(f"{h['round']:<8} {h['pixel_acc']:<14.4f} {h['miou']:<14.4f} {h['time']:<10.1f}")
    print(f"{'-'*46}")
    print(f"\n最优 mIoU: {best_miou:.4f}")
    print(f"最优模型: {os.path.join(save_dir, 'best_model.pth')}")
    print(f"最终模型: {final_path}")


if __name__ == "__main__":
    main()
