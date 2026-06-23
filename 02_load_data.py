"""
第三步：数据加载与采样
使用 RasterDataset 加载 Sentinel-2 影像和水体掩膜
"""

import os
import torch
import numpy as np
import rasterio
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler

# ============================================================
# 1. 数据路径配置
# ============================================================
print("=" * 60)
print("1. 数据路径配置")
print("=" * 60)

# 数据根目录
DATA_ROOT = "./data/dset-s2"

# 训练集路径
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "tra_scene")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "tra_truth")

# 验证集路径
VAL_IMG_DIR = os.path.join(DATA_ROOT, "val_scene")
VAL_MASK_DIR = os.path.join(DATA_ROOT, "val_truth")

print(f"训练影像: {TRAIN_IMG_DIR}")
print(f"训练掩膜: {TRAIN_MASK_DIR}")
print(f"验证影像: {VAL_IMG_DIR}")
print(f"验证掩膜: {VAL_MASK_DIR}")

# ============================================================
# 2. 创建 RasterDataset
# ============================================================
print("\n" + "=" * 60)
print("2. 创建 RasterDataset")
print("=" * 60)

# RasterDataset 关键参数说明：
# - crs: 坐标参考系，所有数据会自动转换到此坐标系
# - res: 分辨率（米），所有数据会重采样到此分辨率
# - transforms: 数据变换函数

# 训练影像数据集
train_imgs = RasterDataset(
    paths=TRAIN_IMG_DIR,
    crs="EPSG:32642",  # 使用原始数据的 UTM 投影
    res=10,             # 10米分辨率
)

# 训练掩膜数据集
train_masks = RasterDataset(
    paths=TRAIN_MASK_DIR,
    crs="EPSG:32642",
    res=10,
)
train_masks.is_image = False  # 标记为掩膜数据（非影像）

# 验证影像数据集
val_imgs = RasterDataset(
    paths=VAL_IMG_DIR,
    crs="EPSG:32642",
    res=10,
)

# 验证掩膜数据集
val_masks = RasterDataset(
    paths=VAL_MASK_DIR,
    crs="EPSG:32642",
    res=10,
)
val_masks.is_image = False  # 标记为掩膜数据（非影像）

print(f"训练影像数据集: {len(train_imgs)} 个文件")
print(f"训练掩膜数据集: {len(train_masks)} 个文件")
print(f"验证影像数据集: {len(val_imgs)} 个文件")
print(f"验证掩膜数据集: {len(val_masks)} 个文件")

# ============================================================
# 3. 合并影像和掩膜（使用 & 操作符）
# ============================================================
print("\n" + "=" * 60)
print("3. 合并影像和掩膜")
print("=" * 60)

# 使用 & 操作符合并影像和掩膜
# 这样采样时会同时返回影像 patch 和对应的掩膜 patch
train_dataset = train_imgs & train_masks
val_dataset = val_imgs & val_masks

print("合并完成！")
print(f"训练集: 影像 + 掩膜")
print(f"验证集: 影像 + 掩膜")

# ============================================================
# 4. 创建采样器
# ============================================================
print("\n" + "=" * 60)
print("4. 创建采样器")
print("=" * 60)

# RandomGeoSampler: 训练时随机采样 patch
# - size: patch 大小（像素数），这里用 256x256
# - length: 每个 epoch 采样多少个 patch
train_sampler = RandomGeoSampler(
    dataset=train_dataset,
    size=256,       # 256x256 像素的 patch
    length=1000,    # 每个 epoch 采样 1000 个 patch
)

# GridGeoSampler: 推理时按网格覆盖整景影像
# - size: patch 大小
# - stride: 步长（小于 size 可以有重叠）
val_sampler = GridGeoSampler(
    dataset=val_dataset,
    size=256,       # 256x256 像素的 patch
    stride=128,     # 步长 128，有 50% 重叠
)

print(f"训练采样器 (RandomGeoSampler):")
print(f"  patch size: 256x256")
print(f"  每 epoch 采样数: 1000")
print()
print(f"验证采样器 (GridGeoSampler):")
print(f"  patch size: 256x256")
print(f"  stride: 128 (50% 重叠)")

# ============================================================
# 5. 创建 DataLoader
# ============================================================
print("\n" + "=" * 60)
print("5. 创建 DataLoader")
print("=" * 60)

# 自定义 collate_fn 来处理数据
def collate_fn(batch):
    """
    自定义批处理函数
    将采样器返回的字典列表合并为批次张量
    """
    # 堆叠所有样本
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])

    # 确保掩膜是长整型（CrossEntropyLoss 需要）
    masks = masks.long()

    return {'image': images, 'mask': masks}

# 训练数据加载器
# Windows 下 num_workers > 0 需要在 if __name__ == '__main__' 中运行
# 这里先用 num_workers=0 测试，正式训练时再调整
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=train_sampler,
    batch_size=8,           # 每批 8 个 patch
    num_workers=0,          # Windows 下先用 0，正式训练时可改为 4
    collate_fn=collate_fn,
    pin_memory=True,        # 加速 GPU 传输
)

# 验证数据加载器
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    sampler=val_sampler,
    batch_size=8,
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=True,
)

print(f"训练 DataLoader: batch_size=8, num_workers=4")
print(f"验证 DataLoader: batch_size=8, num_workers=4")

# ============================================================
# 6. 测试数据加载
# ============================================================
print("\n" + "=" * 60)
print("6. 测试数据加载")
print("=" * 60)

# 取一个批次测试
print("尝试加载一个训练批次...")
try:
    batch = next(iter(train_loader))
    images = batch['image']
    masks = batch['mask']

    print(f"成功！")
    print(f"  影像形状: {images.shape}")  # [B, C, H, W]
    print(f"  掩膜形状: {masks.shape}")   # [B, 1, H, W]
    print(f"  影像数据类型: {images.dtype}")
    print(f"  掩膜数据类型: {masks.dtype}")
    print(f"  影像值范围: [{images.min():.2f}, {images.max():.2f}]")
    print(f"  掩膜唯一值: {torch.unique(masks).tolist()}")

    # 检查是否在 GPU 可用
    if torch.cuda.is_available():
        print(f"\n  GPU 可用: {torch.cuda.get_device_name(0)}")
        print(f"  可以使用 .to('cuda') 将数据移到 GPU")

except Exception as e:
    print(f"加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("数据加载完成！")
print("=" * 60)
print("\n关键概念总结:")
print("""
1. RasterDataset: 加载地理栅格数据，自动处理 CRS 和分辨率
2. & 操作符: 合并影像和掩膜，采样时返回配对的 patch
3. RandomGeoSampler: 训练时随机采样
4. GridGeoSampler: 推理时按网格覆盖
5. collate_fn: 将字典列表合并为批次张量
""")
