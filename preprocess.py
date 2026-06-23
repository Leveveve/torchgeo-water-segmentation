"""
预处理模块
提供数据标准化、NDWI计算、数据增强等功能
"""

import torch
import torch.nn as nn
import kornia.augmentation as K


def normalize_minmax(image):
    """
    Min-Max 归一化到 [0, 1]
    Sentinel-2 原始值范围 0-10000

    Args:
        image: [C, H, W] 张量

    Returns:
        归一化后的图像
    """
    return image / 10000.0


def normalize_zscore(image, mean, std):
    """
    Z-score 标准化

    Args:
        image: [C, H, W] 张量
        mean: [C] 每个波段的均值
        std: [C] 每个波段的标准差

    Returns:
        标准化后的图像
    """
    mean = mean.view(-1, 1, 1)  # [C, 1, 1]
    std = std.view(-1, 1, 1)    # [C, 1, 1]
    return (image - mean) / std


def calculate_ndwi(image, green_idx=1, nir_idx=4):
    """
    计算 NDWI (归一化差异水体指数)
    公式: NDWI = (Green - NIR) / (Green + NIR)

    Args:
        image: [C, H, W] 张量
        green_idx: 绿光波段索引 (默认 B3=index 1)
        nir_idx: 近红外波段索引 (默认 B5=index 4)

    Returns:
        NDWI 图像 [H, W]
    """
    green = image[green_idx]  # 绿光波段
    nir = image[nir_idx]      # 近红外波段
    ndwi = (green - nir) / (green + nir + 1e-8)  # 避免除以 0
    return ndwi


def add_ndwi_channel(image, green_idx=1, nir_idx=4):
    """
    将 NDWI 作为额外通道添加到图像

    Args:
        image: [C, H, W] 张量
        green_idx: 绿光波段索引
        nir_idx: 近红外波段索引

    Returns:
        添加 NDWI 通道后的图像 [C+1, H, W]
    """
    ndwi = calculate_ndwi(image, green_idx, nir_idx)  # [H, W]
    ndwi = ndwi.unsqueeze(0)  # [1, H, W]
    return torch.cat([image, ndwi], dim=0)


class AugmentationPipeline:
    """
    数据增强管道
    使用 Kornia 进行 GPU 加速的数据增强
    """

    def __init__(self, p=0.5):
        """
        Args:
            p: 增强概率
        """
        self.augmentation = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=p),
            K.RandomVerticalFlip(p=p),
            K.RandomRotation(degrees=90, p=p),
            data_keys=["input", "mask"],
        )

    def __call__(self, image, mask):
        """
        应用数据增强

        Args:
            image: [C, H, W] 张量
            mask: [H, W] 张量

        Returns:
            增强后的图像和掩膜
        """
        # 添加 batch 维度
        image = image.unsqueeze(0)  # [1, C, H, W]
        mask = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

        # 应用增强
        image_aug, mask_aug = self.augmentation(image, mask)

        # 移除 batch 维度
        image_aug = image_aug.squeeze(0)  # [C, H, W]
        mask_aug = mask_aug.squeeze(0).squeeze(0).long()  # [H, W]

        return image_aug, mask_aug


class PreprocessTransform:
    """
    完整的预处理变换
    包含: 归一化 + NDWI + 数据增强
    """

    def __init__(self, mean=None, std=None, apply_augmentation=True,
                 add_ndwi=False, green_idx=1, nir_idx=4):
        """
        Args:
            mean: 训练集均值 (用于标准化)，形状 [C]
            std: 训练集标准差 (用于标准化)，形状 [C]
            apply_augmentation: 是否应用数据增强
            add_ndwi: 是否添加 NDWI 通道
            green_idx: 绿光波段索引
            nir_idx: 近红外波段索引
        """
        self.mean = mean
        self.std = std
        self.apply_augmentation = apply_augmentation
        self.add_ndwi = add_ndwi
        self.green_idx = green_idx
        self.nir_idx = nir_idx

        if apply_augmentation:
            self.augmentation = AugmentationPipeline(p=0.5)

    def __call__(self, sample):
        """
        应用预处理

        Args:
            sample: dict with 'image' [C, H, W] and 'mask' [H, W]

        Returns:
            处理后的 sample
        """
        image = sample['image']
        mask = sample['mask']

        # 1. 归一化
        if self.mean is not None and self.std is not None:
            image = normalize_zscore(image, self.mean, self.std)
        else:
            image = normalize_minmax(image)

        # 2. 添加 NDWI 通道（可选）
        if self.add_ndwi:
            image = add_ndwi_channel(image, self.green_idx, self.nir_idx)

        # 3. 数据增强（仅训练时）
        if self.apply_augmentation:
            image, mask = self.augmentation(image, mask)

        return {'image': image, 'mask': mask}


def calculate_dataset_statistics(dataset, num_samples=100, size=256):
    """
    计算数据集的均值和标准差

    Args:
        dataset: TorchGeo 数据集
        num_samples: 采样数量
        size: patch 大小

    Returns:
        mean: [C] 每个波段的均值
        std: [C] 每个波段的标准差
    """
    from torchgeo.samplers import RandomGeoSampler

    sampler = RandomGeoSampler(dataset=dataset, size=size, length=num_samples)

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

    return mean, std
