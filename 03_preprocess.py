"""
第四步：数据预处理与增强
- 数据标准化（归一化）
- 计算 NDWI 光谱指数
- 使用 Kornia 进行数据增强
"""

import os
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import RandomGeoSampler
import kornia.augmentation as K

# ============================================================
# 1. 数据标准化（归一化）
# ============================================================
print("=" * 60)
print("1. 数据标准化")
print("=" * 60)

# 方法一：Min-Max 归一化（简单直接）
def normalize_minmax(image):
    """
    Min-Max 归一化到 [0, 1]
    Sentinel-2 原始值范围 0-10000
    """
    return image / 10000.0

# 方法二：Z-score 标准化（更稳定）
def normalize_zscore(image, mean, std):
    """
    Z-score 标准化
    image: [C, H, W]
    mean: [C] 每个波段的均值
    std: [C] 每个波段的标准差
    """
    # 确保形状匹配
    mean = mean.view(-1, 1, 1)  # [C, 1, 1]
    std = std.view(-1, 1, 1)    # [C, 1, 1]
    return (image - mean) / std

# 训练集统计信息（需要预先计算）
# 这里先用示例值，实际应该从训练集计算
TRAIN_MEAN = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
TRAIN_STD = torch.tensor([500.0, 500.0, 500.0, 500.0, 500.0, 500.0])

print("标准化方法:")
print("  方法一: Min-Max 归一化 (image / 10000)")
print("  方法二: Z-score 标准化 ((image - mean) / std)")
print()
print("注意: 遥感数据的均值/标准差需要从训练集计算，不能用 ImageNet 的值")

# ============================================================
# 2. 计算训练集统计信息
# ============================================================
print("\n" + "=" * 60)
print("2. 计算训练集统计信息")
print("=" * 60)

def calculate_dataset_statistics(dataset, num_samples=100):
    """
    计算数据集的均值和标准差
    dataset: RasterDataset
    num_samples: 采样数量
    """
    print(f"正在计算数据集统计信息（采样 {num_samples} 个 patch）...")

    sampler = RandomGeoSampler(dataset=dataset, size=256, length=num_samples)

    all_images = []
    for bbox in sampler:
        sample = dataset[bbox]
        image = sample['image']  # [C, H, W]
        all_images.append(image)

    # 堆叠所有图像
    all_images = torch.stack(all_images)  # [N, C, H, W]

    # 计算每个波段的均值和标准差
    mean = all_images.mean(dim=[0, 2, 3])  # [C]
    std = all_images.std(dim=[0, 2, 3])    # [C]

    print(f"均值 (每波段): {mean.tolist()}")
    print(f"标准差 (每波段): {std.tolist()}")

    return mean, std

# ============================================================
# 3. 计算 NDWI 光谱指数
# ============================================================
print("\n" + "=" * 60)
print("3. 计算 NDWI 光谱指数")
print("=" * 60)

def calculate_ndwi(image):
    """
    计算 NDWI (归一化差异水体指数)
    公式: NDWI = (Green - NIR) / (Green + NIR)

    image: [C, H, W]
    假设波段顺序: [B2, B3, B4, B5, B6, B7] 或类似
    需要根据实际波段顺序调整索引
    """
    # 假设波段顺序: B2(蓝), B3(绿), B4(红), B5(红边1), B6(红边2), B7(红边3)
    # 如果有 B8(近红外)，需要单独处理
    # 这里假设 B3 是绿光 (index=1), B8 是近红外 (如果有)

    # 根据实际数据调整索引
    # 假设 6 波段数据: [B2, B3, B4, B5, B6, B7]
    green = image[1]  # B3 (绿光)
    # 注意: 如果没有 B8，可以用其他波段近似
    # 这里用 B5 (红边) 近似近红外
    nir = image[4]    # B5 或 B8

    # 计算 NDWI
    ndwi = (green - nir) / (green + nir + 1e-8)  # 避免除以 0

    return ndwi

def add_ndwi_channel(image):
    """
    将 NDWI 作为额外通道添加到图像
    image: [C, H, W] -> [C+1, H, W]
    """
    ndwi = calculate_ndwi(image)  # [H, W]
    ndwi = ndwi.unsqueeze(0)      # [1, H, W]
    return torch.cat([image, ndwi], dim=0)

print("NDWI 计算说明:")
print("  公式: NDWI = (Green - NIR) / (Green + NIR)")
print("  作用: 突出水体特征，抑制植被和土壤")
print("  结果: NDWI > 0 通常是水体")

# ============================================================
# 4. 数据增强（Kornia）
# ============================================================
print("\n" + "=" * 60)
print("4. 数据增强 (Kornia)")
print("=" * 60)

# 定义增强策略
# 注意: Kornia 的增强需要同时处理图像和掩膜
augmentation = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),      # 水平翻转
    K.RandomVerticalFlip(p=0.5),        # 垂直翻转
    K.RandomRotation(degrees=90, p=0.5), # 随机旋转 90°
    data_keys=["input", "mask"],         # 同时处理图像和掩膜
)

def apply_augmentation(image, mask):
    """
    应用数据增强
    image: [C, H, W]
    mask: [H, W]
    """
    # Kornia 需要 batch 维度
    image = image.unsqueeze(0)  # [1, C, H, W]
    mask = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

    # 应用增强
    image_aug, mask_aug = augmentation(image, mask)

    # 移除 batch 维度
    image_aug = image_aug.squeeze(0)  # [C, H, W]
    mask_aug = mask_aug.squeeze(0).squeeze(0).long()  # [H, W]

    return image_aug, mask_aug

print("数据增强策略:")
print("  1. 随机水平翻转 (p=0.5)")
print("  2. 随机垂直翻转 (p=0.5)")
print("  3. 随机旋转 90° (p=0.5)")
print()
print("为什么用 Kornia?")
print("  - 可微分，支持 GPU 加速")
print("  - 与 PyTorch 深度集成")
print("  - 支持批量处理")
print("  - 同时处理图像和掩膜")

# ============================================================
# 5. 完整的预处理流程
# ============================================================
print("\n" + "=" * 60)
print("5. 完整的预处理流程")
print("=" * 60)

class PreprocessTransform:
    """
    完整的预处理变换
    包含: 归一化 + NDWI + 数据增强
    """
    def __init__(self, mean=None, std=None, apply_augmentation=True):
        """
        mean: 训练集均值 (用于标准化)
        std: 训练集标准差 (用于标准化)
        apply_augmentation: 是否应用数据增强
        """
        self.mean = mean
        self.std = std
        self.apply_augmentation = apply_augmentation

    def __call__(self, sample):
        """
        sample: dict with 'image' and 'mask'
        """
        image = sample['image']  # [C, H, W]
        mask = sample['mask']    # [H, W]

        # 1. 归一化
        if self.mean is not None and self.std is not None:
            image = normalize_zscore(image, self.mean, self.std)
        else:
            image = normalize_minmax(image)

        # 2. 添加 NDWI 通道（可选）
        # image = add_ndwi_channel(image)

        # 3. 数据增强（仅训练时）
        if self.apply_augmentation:
            image, mask = apply_augmentation(image, mask)

        return {'image': image, 'mask': mask}

print("预处理流程:")
print("  1. 数据标准化 (Min-Max 或 Z-score)")
print("  2. 可选: 添加 NDWI 通道")
print("  3. 可选: 数据增强 (仅训练时)")
print()
print("完整代码已封装在 PreprocessTransform 类中")

# ============================================================
# 6. 测试预处理流程
# ============================================================
print("\n" + "=" * 60)
print("6. 测试预处理流程")
print("=" * 60)

# 创建数据集
train_imgs = RasterDataset(paths="./data/dset-s2/tra_scene", crs="EPSG:32642", res=10)
train_masks = RasterDataset(paths="./data/dset-s2/tra_truth", crs="EPSG:32642", res=10)
train_masks.is_image = False
train_dataset = train_imgs & train_masks

# 创建采样器
sampler = RandomGeoSampler(dataset=train_dataset, size=256, length=10)

# 测试预处理
print("测试预处理流程...")
try:
    # 获取一个样本
    bbox = next(iter(sampler))
    sample = train_dataset[bbox]

    image = sample['image']  # [C, H, W]
    mask = sample['mask']    # [H, W]

    print(f"原始影像形状: {image.shape}")
    print(f"原始掩膜形状: {mask.shape}")
    print(f"原始影像值范围: [{image.min():.2f}, {image.max():.2f}]")

    # 应用预处理
    preprocess = PreprocessTransform(apply_augmentation=True)
    sample_processed = preprocess(sample)

    image_processed = sample_processed['image']
    mask_processed = sample_processed['mask']

    print(f"\n处理后影像形状: {image_processed.shape}")
    print(f"处理后掩膜形状: {mask_processed.shape}")
    print(f"处理后影像值范围: [{image_processed.min():.2f}, {image_processed.max():.2f}]")
    print(f"处理后掩膜值: {torch.unique(mask_processed).tolist()}")

    print("\n预处理测试成功！")

except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("第四步完成！")
print("=" * 60)
