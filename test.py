from PIL import Image
import numpy as np
import os

mask_dir = './data/train_labels'
masks = sorted(os.listdir(mask_dir))
sample_mask = Image.open(os.path.join(mask_dir, masks[0]))
print(f"掩码模式: {sample_mask.mode}")
print(f"掩码尺寸: {sample_mask.size}")
arr = np.array(sample_mask)
print(f"掩码形状: {arr.shape}")
print(f"像素值范围: {arr.min()} ~ {arr.max()}")
print(f"唯一值: {np.unique(arr)}")

from PIL import Image
import numpy as np
import os

mask_dir = './data/train_labels'
masks = sorted(os.listdir(mask_dir))

# 收集所有掩码中出现的唯一 RGB 颜色
all_colors = set()
for m in masks:
    mask = Image.open(os.path.join(mask_dir, m)).convert("RGB")
    arr = np.array(mask).reshape(-1, 3)
    colors = set(map(tuple, arr))
    all_colors.update(colors)

print(f"掩码中共出现 {len(all_colors)} 种唯一颜色:")
for i, c in enumerate(sorted(all_colors)):
    print(f"  类别 {i}: RGB{c}")