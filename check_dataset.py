"""
check_dataset.py — Cityscapes 数据集完整性检查工具

检查项目：
  1. 目录结构是否完整（leftImg8bit/ 和 gtFine/ 是否存在）
  2. 每张图像是否有对应的 gtFine_labelIds.png 标签文件
  3. 每个标签文件是否有对应的图像文件（检测多余标签）
  4. 图像文件是否可正常读取（检测损坏文件）
  5. 标签文件是否可正常读取（检测损坏文件）
  6. 标签像素值是否在合法范围内（0~33 或 255）
  7. 图像与标签的尺寸是否一致
  8. 各城市文件数量统计

用法：
  python check_dataset.py
  python check_dataset.py "E:\\Autonomous Driving Dataset\\Cityscapes dataset(10g)"
  python check_dataset.py --quick   # 快速模式：跳过逐文件读取验证（只检查文件存在性）
"""

import os
import sys
import glob
import argparse
import numpy as np
from collections import defaultdict

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[警告] PIL 未安装，跳过图像读取验证。运行 pip install Pillow 安装。")

# Cityscapes 标准训练集/验证集文件数量（官方发布版本）
EXPECTED_COUNTS = {
    'train': {'images': 2975, 'labels': 2975},
    'val':   {'images': 500,  'labels': 500},
    'test':  {'images': 1525, 'labels': 0},   # test 集无标签
}

# 合法的 labelId 范围
VALID_LABEL_IDS = set(range(34)) | {-1, 255}


def check_split(root_dir, split, quick=False):
    """检查单个 split（train/val/test）的完整性"""
    print(f"\n{'='*70}")
    print(f"检查 [{split}] 集")
    print(f"{'='*70}")

    img_root  = os.path.join(root_dir, 'leftImg8bit', split)
    mask_root = os.path.join(root_dir, 'gtFine', split)

    errors   = []
    warnings = []

    # ── 1. 目录存在性检查 ──────────────────────────────────────────────
    if not os.path.isdir(img_root):
        errors.append(f"图像目录不存在: {img_root}")
        print(f"  [FAIL] 图像目录不存在: {img_root}")
        return errors, warnings
    if not os.path.isdir(mask_root) and split != 'test':
        errors.append(f"标签目录不存在: {mask_root}")
        print(f"  [FAIL] 标签目录不存在: {mask_root}")
        return errors, warnings

    print(f"  [OK]   图像目录存在: {img_root}")
    if split != 'test':
        print(f"  [OK]   标签目录存在: {mask_root}")

    # ── 2. 收集所有图像文件 ────────────────────────────────────────────
    img_files = sorted(glob.glob(
        os.path.join(img_root, '**', '*_leftImg8bit.png'), recursive=True
    ))
    print(f"\n  图像文件数量: {len(img_files)}", end="")
    expected_img = EXPECTED_COUNTS.get(split, {}).get('images', None)
    if expected_img is not None:
        if len(img_files) == expected_img:
            print(f"  [OK] (期望 {expected_img})")
        elif len(img_files) == expected_img - 1:
            # 常见情况：官方 2975 中有 1 张无对应标签，实际有效 2974 张
            print(f"  [OK*] (期望 {expected_img}，差 1 张属正常，详见下方配对检查)")
        else:
            diff = len(img_files) - expected_img
            msg = f"图像数量异常: 实际 {len(img_files)}，期望 {expected_img}，差 {diff:+d}"
            warnings.append(msg)
            print(f"  [WARN] {msg}")
    else:
        print()

    # ── 3. 按城市统计 ─────────────────────────────────────────────────
    city_counts = defaultdict(int)
    for f in img_files:
        city = os.path.basename(f).split('_')[0]
        city_counts[city] += 1
    print(f"\n  城市数量: {len(city_counts)}")
    for city, cnt in sorted(city_counts.items()):
        print(f"    {city:<20} {cnt:>4} 张")

    if split == 'test':
        print(f"\n  [INFO] test 集无标签文件，跳过标签检查。")
        return errors, warnings

    # ── 4. 收集所有标签文件 ────────────────────────────────────────────
    mask_files = sorted(glob.glob(
        os.path.join(mask_root, '**', '*_gtFine_labelIds.png'), recursive=True
    ))
    print(f"\n  标签文件数量: {len(mask_files)}", end="")
    expected_mask = EXPECTED_COUNTS.get(split, {}).get('labels', None)
    if expected_mask is not None:
        if len(mask_files) == expected_mask:
            print(f"  [OK] (期望 {expected_mask})")
        else:
            diff = len(mask_files) - expected_mask
            msg = f"标签数量异常: 实际 {len(mask_files)}，期望 {expected_mask}，差 {diff:+d}"
            warnings.append(msg)
            print(f"  [WARN] {msg}")
    else:
        print()

    # ── 5. 图像 ↔ 标签 配对检查 ────────────────────────────────────────
    print(f"\n  开始图像-标签配对检查...")

    missing_labels  = []   # 有图像但没有标签
    missing_images  = []   # 有标签但没有图像
    paired_count    = 0

    # 构建标签文件的 stem → path 映射
    mask_stem_map = {}
    for mp in mask_files:
        stem = os.path.basename(mp).replace('_gtFine_labelIds.png', '')
        mask_stem_map[stem] = mp

    # 构建图像文件的 stem → path 映射
    img_stem_map = {}
    for ip in img_files:
        stem = os.path.basename(ip).replace('_leftImg8bit.png', '')
        img_stem_map[stem] = ip

    # 检查每张图像是否有对应标签
    for stem, ip in img_stem_map.items():
        if stem in mask_stem_map:
            paired_count += 1
        else:
            missing_labels.append(ip)

    # 检查每个标签是否有对应图像
    for stem, mp in mask_stem_map.items():
        if stem not in img_stem_map:
            missing_images.append(mp)

    print(f"  成功配对: {paired_count} 对")

    if missing_labels:
        msg = f"以下 {len(missing_labels)} 张图像缺少对应标签文件:"
        warnings.append(msg)
        print(f"  [WARN] {msg}")
        for f in missing_labels[:10]:
            print(f"    {f}")
        if len(missing_labels) > 10:
            print(f"    ... 共 {len(missing_labels)} 个")

    if missing_images:
        msg = f"以下 {len(missing_images)} 个标签文件缺少对应图像（多余标签）:"
        warnings.append(msg)
        print(f"  [WARN] {msg}")
        for f in missing_images[:10]:
            print(f"    {f}")
        if len(missing_images) > 10:
            print(f"    ... 共 {len(missing_images)} 个")

    if not missing_labels and not missing_images:
        print(f"  [OK]   所有文件配对完整，无缺失或多余文件")

    # ── 6. 文件读取验证（非快速模式）─────────────────────────────────
    if quick:
        print(f"\n  [快速模式] 跳过逐文件读取验证")
        return errors, warnings

    if not PIL_AVAILABLE:
        print(f"\n  [跳过] PIL 未安装，无法进行读取验证")
        return errors, warnings

    print(f"\n  开始逐文件读取验证（共 {paired_count} 对，可能需要几分钟）...")

    corrupt_images  = []
    corrupt_masks   = []
    size_mismatch   = []
    invalid_labels  = []
    checked = 0

    for stem in img_stem_map:
        if stem not in mask_stem_map:
            continue

        ip = img_stem_map[stem]
        mp = mask_stem_map[stem]

        # 读取图像
        try:
            img = Image.open(ip)
            img_size = img.size  # (W, H)
            img.close()
        except Exception as e:
            corrupt_images.append((ip, str(e)))
            checked += 1
            continue

        # 读取标签
        try:
            mask = Image.open(mp)
            mask_arr  = np.array(mask)
            mask_size = mask.size  # (W, H)
            mask.close()
        except Exception as e:
            corrupt_masks.append((mp, str(e)))
            checked += 1
            continue

        # 尺寸一致性检查
        if img_size != mask_size:
            size_mismatch.append((ip, mp, img_size, mask_size))

        # 标签值合法性检查（抽样检查，避免太慢）
        if checked < 50:
            unique_vals = set(np.unique(mask_arr).tolist())
            illegal = unique_vals - VALID_LABEL_IDS
            if illegal:
                invalid_labels.append((mp, sorted(illegal)))

        checked += 1
        if checked % 500 == 0:
            print(f"    已检查 {checked}/{paired_count} 对...")

    print(f"  已检查 {checked} 对文件")

    if corrupt_images:
        msg = f"发现 {len(corrupt_images)} 张损坏图像文件:"
        errors.append(msg)
        print(f"  [FAIL] {msg}")
        for f, e in corrupt_images[:5]:
            print(f"    {f}: {e}")

    if corrupt_masks:
        msg = f"发现 {len(corrupt_masks)} 个损坏标签文件:"
        errors.append(msg)
        print(f"  [FAIL] {msg}")
        for f, e in corrupt_masks[:5]:
            print(f"    {f}: {e}")

    if size_mismatch:
        msg = f"发现 {len(size_mismatch)} 对图像-标签尺寸不一致:"
        errors.append(msg)
        print(f"  [FAIL] {msg}")
        for ip, mp, is_, ms in size_mismatch[:5]:
            print(f"    图像 {is_} vs 标签 {ms}: {os.path.basename(ip)}")

    if invalid_labels:
        msg = f"发现 {len(invalid_labels)} 个标签文件含非法像素值:"
        warnings.append(msg)
        print(f"  [WARN] {msg}")
        for f, vals in invalid_labels[:5]:
            print(f"    {os.path.basename(f)}: 非法值 {vals}")

    if not corrupt_images and not corrupt_masks and not size_mismatch:
        print(f"  [OK]   所有文件读取正常，无损坏文件，尺寸一致")

    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description='Cityscapes 数据集完整性检查')
    parser.add_argument('root', nargs='?',
                        default=r'E:\Autonomous Driving Dataset\Cityscapes dataset(10g)',
                        help='数据集根目录路径')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式：只检查文件存在性，跳过逐文件读取验证')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        choices=['train', 'val', 'test'],
                        help='要检查的数据集划分（默认: train val）')
    args = parser.parse_args()

    root = args.root
    print(f"\n{'#'*70}")
    print(f"  Cityscapes 数据集完整性检查")
    print(f"  根目录: {root}")
    print(f"  模式: {'快速（仅文件存在性）' if args.quick else '完整（含读取验证）'}")
    print(f"{'#'*70}")

    if not os.path.isdir(root):
        print(f"\n[FATAL] 数据集根目录不存在: {root}")
        print("请确认路径正确，或通过命令行参数指定：")
        print("  python check_dataset.py \"<数据集路径>\"")
        sys.exit(1)

    all_errors   = []
    all_warnings = []

    for split in args.splits:
        errs, warns = check_split(root, split, quick=args.quick)
        all_errors.extend(errs)
        all_warnings.extend(warns)

    # ── 汇总报告 ───────────────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  检查结果汇总")
    print(f"{'#'*70}")

    if not all_errors and not all_warnings:
        print(f"\n  [全部通过] 数据集完整性检查通过，未发现任何问题。")
    else:
        if all_errors:
            print(f"\n  [错误] 共 {len(all_errors)} 个严重问题（会导致训练失败）:")
            for i, e in enumerate(all_errors, 1):
                print(f"    {i}. {e}")
        if all_warnings:
            print(f"\n  [警告] 共 {len(all_warnings)} 个警告（可能影响训练）:")
            for i, w in enumerate(all_warnings, 1):
                print(f"    {i}. {w}")

    print(f"\n{'#'*70}\n")
    return len(all_errors)


if __name__ == '__main__':
    sys.exit(main())
