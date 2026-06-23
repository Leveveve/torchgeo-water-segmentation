"""
第七步：模型评估与可视化
- 加载最佳模型
- 在验证集上计算最终指标
- 可视化预测结果
- 分析模型表现
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import RandomGeoSampler
from model import build_deeplabv3, calculate_iou, calculate_accuracy
from preprocess import PreprocessTransform

# ============================================================
# 1. 加载最佳模型
# ============================================================
print("=" * 60)
print("1. 加载最佳模型")
print("=" * 60)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 构建模型（必须与训练时相同的配置）
model = build_deeplabv3(in_channels=6, num_classes=2, pretrained=True)
model = model.to(device)

# 加载最佳权重
checkpoint_path = './checkpoints/best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)

print(f"模型加载成功")
print(f"  最佳 Epoch: {checkpoint['epoch']}")
print(f"  验证 IoU: {checkpoint['val_iou']:.4f}")

# ============================================================
# 2. 准备验证数据
# ============================================================
print("\n" + "=" * 60)
print("2. 准备验证数据")
print("=" * 60)

# 创建验证数据集
val_imgs = RasterDataset(paths="./data/dset-s2/val_scene", crs="EPSG:32642", res=10)
val_masks = RasterDataset(paths="./data/dset-s2/val_truth", crs="EPSG:32642", res=10)
val_masks.is_image = False
val_dataset = val_imgs & val_masks

# 创建采样器（采样更多样本用于评估）
val_sampler = RandomGeoSampler(
    dataset=val_dataset,
    size=256,
    length=500,  # 采样 500 个 patch
)

# collate_fn
def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    masks = masks.long()
    return {'image': images, 'mask': masks}

# DataLoader
val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    sampler=val_sampler,
    batch_size=8,
    num_workers=0,
    collate_fn=collate_fn,
)

print(f"验证集: {len(val_dataset)} 个文件")
print(f"采样数: 500 个 patch")

# ============================================================
# 3. 在验证集上计算最终指标
# ============================================================
print("\n" + "=" * 60)
print("3. 计算最终指标")
print("=" * 60)

# 预处理（不带增强）
preprocess = PreprocessTransform(apply_augmentation=False)

# 评估
model.eval()
all_ious = []
all_accs = []
all_images = []
all_masks = []
all_preds = []

with torch.no_grad():
    for i, batch in enumerate(val_loader):
        images = batch['image']
        masks = batch['mask']

        # 预处理
        processed_images = []
        processed_masks = []
        for j in range(images.shape[0]):
            sample = {'image': images[j], 'mask': masks[j]}
            processed = preprocess(sample)
            processed_images.append(processed['image'])
            processed_masks.append(processed['mask'])

        images = torch.stack(processed_images).to(device)
        masks = torch.stack(processed_masks).to(device)

        # 前向传播
        outputs = model(images)
        preds = outputs['out']

        # 计算指标
        iou = calculate_iou(preds, masks)
        acc = calculate_accuracy(preds, masks)

        all_ious.append(iou.item())
        all_accs.append(acc.item())

        # 保存一些样本用于可视化
        if i < 5:  # 保存前 5 个批次
            all_images.append(images.cpu())
            all_masks.append(masks.cpu())
            all_preds.append(preds.cpu())

# 计算平均指标
mean_iou = np.mean(all_ious)
mean_acc = np.mean(all_accs)
std_iou = np.std(all_ious)
std_acc = np.std(all_accs)

print(f"最终评估结果 (500 个 patch):")
print(f"  平均 IoU: {mean_iou:.4f} +/- {std_iou:.4f}")
print(f"  平均准确率: {mean_acc:.4f} +/- {std_acc:.4f}")
print(f"  最佳 IoU: {np.max(all_ious):.4f}")
print(f"  最差 IoU: {np.min(all_ious):.4f}")

# ============================================================
# 4. 可视化预测结果
# ============================================================
print("\n" + "=" * 60)
print("4. 可视化预测结果")
print("=" * 60)

# 创建输出目录
os.makedirs('./outputs', exist_ok=True)

# 选择一些样本可视化
num_samples = 6
fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))

# 波段名称
band_names = ['B2 (Blue)', 'B3 (Green)', 'B4 (Red)', 'B5 (Red Edge 1)', 'B6 (Red Edge 2)', 'B7 (Red Edge 3)']

for i in range(num_samples):
    # 获取样本
    batch_idx = i // 8
    sample_idx = i % 8

    if batch_idx >= len(all_images):
        break

    image = all_images[batch_idx][sample_idx]  # [C, H, W]
    mask = all_masks[batch_idx][sample_idx]    # [H, W]
    pred = all_preds[batch_idx][sample_idx]    # [C, H, W]
    pred_class = pred.argmax(dim=0)            # [H, W]

    # 1. 原始影像 (使用 B4, B3, B2 合成 RGB)
    # 假设波段顺序: [B2, B3, B4, B5, B6, B7]
    rgb = torch.stack([image[2], image[1], image[0]], dim=-1)  # [H, W, 3]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    rgb = rgb.numpy()

    axes[i, 0].imshow(rgb)
    axes[i, 0].set_title(f'Sample {i+1}: RGB Image')
    axes[i, 0].axis('off')

    # 2. 近红外波段 (B5)
    nir = image[3].numpy()  # B5 (Red Edge 1)
    axes[i, 1].imshow(nir, cmap='gray')
    axes[i, 1].set_title(f'Sample {i+1}: NIR (B5)')
    axes[i, 1].axis('off')

    # 3. 真实掩膜
    axes[i, 2].imshow(mask.numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[i, 2].set_title(f'Sample {i+1}: Ground Truth')
    axes[i, 2].axis('off')

    # 4. 预测掩膜
    axes[i, 3].imshow(pred_class.numpy(), cmap='Blues', vmin=0, vmax=1)
    iou = calculate_iou(pred.unsqueeze(0), mask.unsqueeze(0))
    axes[i, 3].set_title(f'Sample {i+1}: Prediction (IoU={iou:.2f})')
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig('./outputs/prediction_samples.png', dpi=150, bbox_inches='tight')
print("预测可视化已保存: ./outputs/prediction_samples.png")

# ============================================================
# 5. 分析模型表现
# ============================================================
print("\n" + "=" * 60)
print("5. 分析模型表现")
print("=" * 60)

# 计算不同 IoU 范围的样本分布
iou_ranges = [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
iou_labels = ['<50%', '50-70%', '70-90%', '90-100%']

print("\nIoU 分布:")
for (low, high), label in zip(iou_ranges, iou_labels):
    count = sum(1 for iou in all_ious if low <= iou < high)
    percentage = count / len(all_ious) * 100
    print(f"  {label}: {count} 个样本 ({percentage:.1f}%)")

# 找出表现最好和最差的样本
best_idx = np.argmax(all_ious)
worst_idx = np.argmin(all_ious)

print(f"\n表现最好的样本:")
print(f"  IoU: {all_ious[best_idx]:.4f}")
print(f"  准确率: {all_accs[best_idx]:.4f}")

print(f"\n表现最差的样本:")
print(f"  IoU: {all_ious[worst_idx]:.4f}")
print(f"  准确率: {all_accs[worst_idx]:.4f}")

# ============================================================
# 6. 绘制 IoU 分布图
# ============================================================
print("\n" + "=" * 60)
print("6. 绘制 IoU 分布图")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# IoU 直方图
axes[0].hist(all_ious, bins=20, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('IoU')
axes[0].set_ylabel('Frequency')
axes[0].set_title('IoU Distribution')
axes[0].axvline(mean_iou, color='red', linestyle='--', label=f'Mean: {mean_iou:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 准确率直方图
axes[1].hist(all_accs, bins=20, edgecolor='black', alpha=0.7, color='orange')
axes[1].set_xlabel('Accuracy')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Accuracy Distribution')
axes[1].axvline(mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.4f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./outputs/iou_distribution.png', dpi=150, bbox_inches='tight')
print("IoU 分布图已保存: ./outputs/iou_distribution.png")

# ============================================================
# 7. 保存评估报告
# ============================================================
print("\n" + "=" * 60)
print("7. 保存评估报告")
print("=" * 60)

import json

report = {
    'model': 'DeepLabV3_ResNet50',
    'dataset': 'Earth Surface Water',
    'best_epoch': checkpoint['epoch'],
    'num_samples': len(all_ious),
    'metrics': {
        'mean_iou': float(mean_iou),
        'std_iou': float(std_iou),
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'max_iou': float(np.max(all_ious)),
        'min_iou': float(np.min(all_ious)),
    },
    'iou_distribution': {
        label: int(sum(1 for iou in all_ious if low <= iou < high))
        for (low, high), label in zip(iou_ranges, iou_labels)
    },
}

with open('./outputs/evaluation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("评估报告已保存: ./outputs/evaluation_report.json")

print("\n" + "=" * 60)
print("第七步完成！")
print("=" * 60)
