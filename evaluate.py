"""
模型验证脚本 - 用于评估联邦学习训练后的语义分割模型

功能：
  1. 加载保存的模型
  2. 在验证集/测试集上计算指标（Pixel Accuracy, mIoU）
  3. 可视化分割结果并保存对比图

使用方法：
  python evaluate.py --checkpoint ./checkpoints/global_model_final.pth --data ./data --split val
"""

import torch
import numpy as np
import os
import argparse
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from demo import MobileNetV2UNet

# ===== CamVid 12 类 RGB 颜色映射表（与 Fedavg.py 保持一致） =====
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

NUM_CLASSES = len(CAMVID_COLORS)


def rgb_mask_to_class_index(mask_rgb, color_map=CAMVID_COLORS):
    """将 RGB 掩码图转换为类别索引图"""
    mask = np.array(mask_rgb)
    h, w = mask.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int64)
    for cls_idx, color in enumerate(color_map):
        match = np.all(mask == color, axis=-1)
        class_mask[match] = cls_idx
    return class_mask


def class_index_to_rgb(class_mask, color_map=CAMVID_COLORS):
    """将类别索引图转换回 RGB 彩色图（用于可视化）"""
    h, w = class_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(color_map):
        rgb[class_mask == cls_idx] = color
    return rgb


# ==================== 评估指标 ====================

def compute_pixel_accuracy(pred, target):
    """计算像素准确率 (Pixel Accuracy)"""
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
            ious.append(float('nan'))  # 该类别不存在，跳过
        else:
            ious.append(intersection / union)
    return ious


def compute_miou(pred, target, num_classes):
    """计算 mIoU (Mean Intersection over Union)"""
    ious = compute_iou_per_class(pred, target, num_classes)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid_ious) if valid_ious else 0.0


# ==================== 模型加载 ====================

def load_model(checkpoint_path, device):
    """
    加载模型的两种方式：
    - 如果保存的是 state_dict（推荐方式），需要先创建模型再加载参数
    - 如果保存的是完整模型，直接 torch.load 即可
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 判断保存格式
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 方式一：从 state_dict 加载（推荐）
        num_classes = checkpoint.get('num_classes', NUM_CLASSES)
        model = MobileNetV2UNet(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型参数 (Round {checkpoint.get('round', 'N/A')}, num_classes={num_classes})")
    else:
        # 方式二：加载完整模型
        model = checkpoint
        print("已加载完整模型")

    model.to(device)
    model.eval()
    return model


# ==================== 验证流程 ====================

def evaluate(model, data_dir, split, device, target_size=(256, 256)):
    """在验证集/测试集上评估模型"""
    img_dir = os.path.join(data_dir, split)
    mask_dir = os.path.join(data_dir, f'{split}_labels')

    if not os.path.exists(img_dir):
        print(f"错误：找不到目录 {img_dir}")
        print(f"提示：请确认 data 目录下有 '{split}' 和 '{split}_labels' 文件夹")
        return

    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))

    print(f"\n{'='*50}")
    print(f"开始评估 | 数据集: {split} | 图片数: {len(images)}")
    print(f"{'='*50}")

    transform = transforms.ToTensor()

    total_pixel_acc = 0.0
    total_miou = 0.0
    all_ious = [[] for _ in range(NUM_CLASSES)]

    with torch.no_grad():
        for i, (img_name, mask_name) in enumerate(zip(images, masks)):
            # 加载图像
            image = Image.open(os.path.join(img_dir, img_name)).convert("RGB")
            mask = Image.open(os.path.join(mask_dir, mask_name)).convert("RGB")

            # resize
            image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
            mask = mask.resize((target_size[1], target_size[0]), Image.NEAREST)

            # 转换
            input_tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, H, W)
            target = torch.from_numpy(rgb_mask_to_class_index(mask)).long()  # (H, W)

            # 前向推理
            output = model(input_tensor)  # (1, num_classes, H, W)
            pred = output.argmax(dim=1).squeeze(0).cpu()  # (H, W)

            # 计算指标
            pixel_acc = compute_pixel_accuracy(pred, target)
            miou = compute_miou(pred, target, NUM_CLASSES)
            ious = compute_iou_per_class(pred, target, NUM_CLASSES)

            total_pixel_acc += pixel_acc
            total_miou += miou
            for cls in range(NUM_CLASSES):
                if not np.isnan(ious[cls]):
                    all_ious[cls].append(ious[cls])

            if (i + 1) % 10 == 0 or (i + 1) == len(images):
                print(f"  [{i+1}/{len(images)}] Pixel Acc: {pixel_acc:.4f} | mIoU: {miou:.4f}")

    # 汇总结果
    avg_pixel_acc = total_pixel_acc / len(images)
    avg_miou = total_miou / len(images)

    print(f"\n{'='*50}")
    print(f"总体结果:")
    print(f"  Pixel Accuracy: {avg_pixel_acc:.4f}")
    print(f"  mIoU:           {avg_miou:.4f}")
    print(f"\n各类别 IoU:")
    print(f"  {'类别':<20} {'IoU':<10} {'样本数'}")
    print(f"  {'-'*40}")
    for cls in range(NUM_CLASSES):
        if all_ious[cls]:
            cls_iou = np.mean(all_ious[cls])
            print(f"  {CLASS_NAMES[cls]:<20} {cls_iou:.4f}     {len(all_ious[cls])}")
        else:
            print(f"  {CLASS_NAMES[cls]:<20} {'N/A':<10} 0")
    print(f"{'='*50}")

    return avg_pixel_acc, avg_miou


# ==================== 可视化 ====================

def visualize_predictions(model, data_dir, split, device, num_samples=5, target_size=(256, 256), save_dir='./results'):
    """可视化分割结果：原图 | 真实标签 | 预测结果"""
    img_dir = os.path.join(data_dir, split)
    mask_dir = os.path.join(data_dir, f'{split}_labels')

    if not os.path.exists(img_dir):
        print(f"错误：找不到目录 {img_dir}")
        return

    os.makedirs(save_dir, exist_ok=True)

    images = sorted(os.listdir(img_dir))
    masks = sorted(os.listdir(mask_dir))

    transform = transforms.ToTensor()
    num_samples = min(num_samples, len(images))

    # 随机选取样本
    sample_indices = np.random.choice(len(images), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            # 加载
            image = Image.open(os.path.join(img_dir, images[idx])).convert("RGB")
            mask = Image.open(os.path.join(mask_dir, masks[idx])).convert("RGB")

            image = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
            mask = mask.resize((target_size[1], target_size[0]), Image.NEAREST)

            input_tensor = transform(image).unsqueeze(0).to(device)
            target = rgb_mask_to_class_index(mask)

            # 推理
            output = model(input_tensor)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

            # 转回 RGB 用于可视化
            target_rgb = class_index_to_rgb(target)
            pred_rgb = class_index_to_rgb(pred)

            # 绘图
            axes[row, 0].imshow(image)
            axes[row, 0].set_title(f'Original: {images[idx]}', fontsize=10)
            axes[row, 0].axis('off')

            axes[row, 1].imshow(target_rgb)
            axes[row, 1].set_title('Ground Truth', fontsize=10)
            axes[row, 1].axis('off')

            axes[row, 2].imshow(pred_rgb)
            axes[row, 2].set_title('Prediction', fontsize=10)
            axes[row, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'segmentation_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n可视化结果已保存: {save_path}")


# ==================== 主函数 ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='语义分割模型验证脚本')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/global_model_final.pth',
                        help='模型检查点路径')
    parser.add_argument('--data', type=str, default='./data',
                        help='数据集根目录')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val', 'test'],
                        help='评估的数据集划分 (train/val/test)')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化对比图')
    parser.add_argument('--num_vis', type=int, default=5,
                        help='可视化样本数量')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='可视化结果保存目录')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载模型
    model = load_model(args.checkpoint, device)

    # 2. 评估指标
    evaluate(model, args.data, args.split, device)

    # 3. 可视化（可选）
    if args.visualize:
        visualize_predictions(model, args.data, args.split, device,
                              num_samples=args.num_vis, save_dir=args.save_dir)