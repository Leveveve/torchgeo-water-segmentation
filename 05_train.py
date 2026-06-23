"""
第六步：模型训练
编写完整的训练循环，包括：
- 训练循环
- 验证评估
- 保存最佳模型
- 记录训练指标
"""

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from model import build_deeplabv3, CombinedLoss, calculate_iou, calculate_accuracy
from preprocess import PreprocessTransform

# ============================================================
# 1. 配置训练参数
# ============================================================
print("=" * 60)
print("1. 配置训练参数")
print("=" * 60)

# 训练参数
CONFIG = {
    # 数据参数
    'patch_size': 256,          # patch 大小
    'batch_size': 8,            # 批次大小
    'train_length': 1000,       # 每 epoch 训练采样数
    'val_length': 200,          # 每 epoch 验证采样数

    # 模型参数
    'in_channels': 6,           # 输入通道数
    'num_classes': 2,           # 输出类别数
    'pretrained': True,         # 使用预训练权重

    # 训练参数
    'epochs': 20,               # 训练轮数
    'lr': 1e-4,                 # 学习率
    'weight_decay': 1e-5,       # 权重衰减
    'ce_weight': 1.0,           # CrossEntropyLoss 权重
    'dice_weight': 1.0,         # Dice Loss 权重

    # 学习率调度
    'scheduler_patience': 5,    # 等待轮数
    'scheduler_factor': 0.5,    # 衰减因子

    # 保存参数
    'checkpoint_dir': './checkpoints',
    'save_best_only': True,     # 只保存最佳模型
}

# 创建保存目录
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

print("训练配置:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================
# 2. 准备数据
# ============================================================
print("\n" + "=" * 60)
print("2. 准备数据")
print("=" * 60)

# 创建数据集
train_imgs = RasterDataset(paths="./data/dset-s2/tra_scene", crs="EPSG:32642", res=10)
train_masks = RasterDataset(paths="./data/dset-s2/tra_truth", crs="EPSG:32642", res=10)
train_masks.is_image = False

val_imgs = RasterDataset(paths="./data/dset-s2/val_scene", crs="EPSG:32642", res=10)
val_masks = RasterDataset(paths="./data/dset-s2/val_truth", crs="EPSG:32642", res=10)
val_masks.is_image = False

# 合并影像和掩膜
train_dataset = train_imgs & train_masks
val_dataset = val_imgs & val_masks

# 创建采样器
train_sampler = RandomGeoSampler(
    dataset=train_dataset,
    size=CONFIG['patch_size'],
    length=CONFIG['train_length'],
)

val_sampler = RandomGeoSampler(
    dataset=val_dataset,
    size=CONFIG['patch_size'],
    length=CONFIG['val_length'],
)

# 自定义 collate_fn
def collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    masks = masks.long()
    return {'image': images, 'mask': masks}

# 创建 DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    sampler=train_sampler,
    batch_size=CONFIG['batch_size'],
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=True,
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    sampler=val_sampler,
    batch_size=CONFIG['batch_size'],
    num_workers=0,
    collate_fn=collate_fn,
    pin_memory=True,
)

print(f"训练集: {len(train_dataset)} 个文件")
print(f"验证集: {len(val_dataset)} 个文件")
print(f"训练 DataLoader: {len(train_loader)} 个批次/epoch")
print(f"验证 DataLoader: {len(val_loader)} 个批次/epoch")

# ============================================================
# 3. 创建预处理
# ============================================================
print("\n" + "=" * 60)
print("3. 创建预处理")
print("=" * 60)

# 训练预处理（带增强）
train_preprocess = PreprocessTransform(
    mean=None,  # 使用 Min-Max 归一化
    std=None,
    apply_augmentation=True,
)

# 验证预处理（不带增强）
val_preprocess = PreprocessTransform(
    mean=None,
    std=None,
    apply_augmentation=False,
)

print("训练预处理: Min-Max 归一化 + 数据增强")
print("验证预处理: Min-Max 归一化（无增强）")

# ============================================================
# 4. 创建模型、损失函数、优化器
# ============================================================
print("\n" + "=" * 60)
print("4. 创建模型、损失函数、优化器")
print("=" * 60)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 模型
model = build_deeplabv3(
    in_channels=CONFIG['in_channels'],
    num_classes=CONFIG['num_classes'],
    pretrained=CONFIG['pretrained'],
)
model = model.to(device)

# 损失函数
criterion = CombinedLoss(
    ce_weight=CONFIG['ce_weight'],
    dice_weight=CONFIG['dice_weight'],
)
criterion = criterion.to(device)

# 优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=CONFIG['lr'],
    weight_decay=CONFIG['weight_decay'],
)

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=CONFIG['scheduler_factor'],
    patience=CONFIG['scheduler_patience'],
    verbose=True,
)

print("模型、损失函数、优化器创建完成")

# ============================================================
# 5. 训练循环
# ============================================================
print("\n" + "=" * 60)
print("5. 开始训练")
print("=" * 60)

# 记录训练指标
train_losses = []
val_losses = []
val_ious = []
val_accs = []
best_iou = 0.0

def train_one_epoch(model, loader, criterion, optimizer, device, preprocess):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in loader:
        # 预处理
        images = batch['image']
        masks = batch['mask']

        # 应用预处理（逐样本）
        processed_images = []
        processed_masks = []
        for i in range(images.shape[0]):
            sample = {'image': images[i], 'mask': masks[i]}
            processed = preprocess(sample)
            processed_images.append(processed['image'])
            processed_masks.append(processed['mask'])

        images = torch.stack(processed_images).to(device)
        masks = torch.stack(processed_masks).to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs['out'], masks)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

def validate(model, loader, criterion, device, preprocess):
    """验证"""
    model.eval()
    total_loss = 0
    total_iou = 0
    total_acc = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            # 预处理
            images = batch['image']
            masks = batch['mask']

            processed_images = []
            processed_masks = []
            for i in range(images.shape[0]):
                sample = {'image': images[i], 'mask': masks[i]}
                processed = preprocess(sample)
                processed_images.append(processed['image'])
                processed_masks.append(processed['mask'])

            images = torch.stack(processed_images).to(device)
            masks = torch.stack(processed_masks).to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs['out'], masks)

            # 计算指标
            iou = calculate_iou(outputs['out'], masks)
            acc = calculate_accuracy(outputs['out'], masks)

            total_loss += loss.item()
            total_iou += iou.item()
            total_acc += acc.item()
            num_batches += 1

    return total_loss / num_batches, total_iou / num_batches, total_acc / num_batches

# 训练循环
for epoch in range(CONFIG['epochs']):
    print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}")
    print("-" * 40)

    # 训练
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, device, train_preprocess
    )

    # 验证
    val_loss, val_iou, val_acc = validate(
        model, val_loader, criterion, device, val_preprocess
    )

    # 更新学习率
    scheduler.step(val_iou)
    current_lr = optimizer.param_groups[0]['lr']

    # 记录指标
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_ious.append(val_iou)
    val_accs.append(val_acc)

    # 打印结果
    print(f"  训练损失: {train_loss:.4f}")
    print(f"  验证损失: {val_loss:.4f}")
    print(f"  验证 IoU: {val_iou:.4f}")
    print(f"  验证准确率: {val_acc:.4f}")
    print(f"  学习率: {current_lr:.6f}")

    # 保存最佳模型
    if val_iou > best_iou:
        best_iou = val_iou
        checkpoint_path = os.path.join(CONFIG['checkpoint_dir'], 'best_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_iou': val_iou,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"  >> 保存最佳模型 (IoU: {val_iou:.4f})")

print("\n" + "=" * 60)
print("训练完成！")
print("=" * 60)
print(f"最佳验证 IoU: {best_iou:.4f}")
print(f"模型保存路径: {CONFIG['checkpoint_dir']}/best_model.pth")

# ============================================================
# 6. 绘制训练曲线
# ============================================================
print("\n" + "=" * 60)
print("6. 绘制训练曲线")
print("=" * 60)

# 创建输出目录
os.makedirs('./outputs', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 损失曲线
axes[0].plot(train_losses, label='Train Loss', color='blue')
axes[0].plot(val_losses, label='Val Loss', color='red')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True)

# IoU 曲线
axes[1].plot(val_ious, label='Val IoU', color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('IoU')
axes[1].set_title('Validation IoU')
axes[1].legend()
axes[1].grid(True)

# 准确率曲线
axes[2].plot(val_accs, label='Val Accuracy', color='orange')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].set_title('Validation Accuracy')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('./outputs/training_curves.png', dpi=150, bbox_inches='tight')
print("训练曲线已保存: ./outputs/training_curves.png")

# 保存训练指标
import json
metrics = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_ious': val_ious,
    'val_accs': val_accs,
    'best_iou': best_iou,
}
with open('./outputs/training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("训练指标已保存: ./outputs/training_metrics.json")

print("\n第六步完成！")
