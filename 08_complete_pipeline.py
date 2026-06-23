"""
第九步：完整项目代码汇总
将所有步骤整合到一个完整的脚本中
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler
from model import build_deeplabv3, CombinedLoss, calculate_iou, calculate_accuracy
from preprocess import PreprocessTransform

# ============================================================
# 第一步：环境配置
# ============================================================
def check_environment():
    """检查环境配置"""
    print("=" * 60)
    print("第一步：环境配置")
    print("=" * 60)

    # 检查 PyTorch
    print(f"PyTorch 版本: {torch.__version__}")

    # 检查 CUDA
    if torch.cuda.is_available():
        print(f"CUDA 可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
    else:
        print("CUDA 不可用，使用 CPU")

    # 检查其他包
    import torchgeo
    print(f"TorchGeo 版本: {torchgeo.__version__}")

    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# 第三步：数据加载
# ============================================================
def load_data():
    """加载数据集"""
    print("\n" + "=" * 60)
    print("第三步：数据加载")
    print("=" * 60)

    # 数据路径
    DATA_ROOT = "./data/dset-s2"

    # 创建数据集
    train_imgs = RasterDataset(paths=f"{DATA_ROOT}/tra_scene", crs="EPSG:32642", res=10)
    train_masks = RasterDataset(paths=f"{DATA_ROOT}/tra_truth", crs="EPSG:32642", res=10)
    train_masks.is_image = False

    val_imgs = RasterDataset(paths=f"{DATA_ROOT}/val_scene", crs="EPSG:32642", res=10)
    val_masks = RasterDataset(paths=f"{DATA_ROOT}/val_truth", crs="EPSG:32642", res=10)
    val_masks.is_image = False

    # 合并影像和掩膜
    train_dataset = train_imgs & train_masks
    val_dataset = val_imgs & val_masks

    print(f"训练集: {len(train_dataset)} 个文件")
    print(f"验证集: {len(val_dataset)} 个文件")

    return train_dataset, val_dataset


# ============================================================
# 第四步：数据预处理
# ============================================================
def create_preprocessors():
    """创建预处理器"""
    print("\n" + "=" * 60)
    print("第四步：数据预处理")
    print("=" * 60)

    # 训练预处理（带增强）
    train_preprocess = PreprocessTransform(
        mean=None,
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

    return train_preprocess, val_preprocess


# ============================================================
# 第五步：构建模型
# ============================================================
def create_model(device):
    """创建模型"""
    print("\n" + "=" * 60)
    print("第五步：构建模型")
    print("=" * 60)

    # 模型
    model = build_deeplabv3(in_channels=6, num_classes=2, pretrained=True)
    model = model.to(device)

    # 损失函数
    criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0)
    criterion = criterion.to(device)

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
    )

    print("模型: DeepLabV3 + ResNet50")
    print("损失函数: CombinedLoss (CE + Dice)")
    print("优化器: Adam (lr=1e-4)")
    print("学习率调度: ReduceLROnPlateau")

    return model, criterion, optimizer, scheduler


# ============================================================
# 第六步：训练循环
# ============================================================
def train_model(model, train_dataset, val_dataset, criterion, optimizer, scheduler,
                device, train_preprocess, val_preprocess, epochs=20):
    """训练模型"""
    print("\n" + "=" * 60)
    print("第六步：模型训练")
    print("=" * 60)

    # 创建采样器和 DataLoader
    train_sampler = RandomGeoSampler(dataset=train_dataset, size=256, length=1000)
    val_sampler = RandomGeoSampler(dataset=val_dataset, size=256, length=200)

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        masks = masks.long()
        return {'image': images, 'mask': masks}

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        batch_size=8,
        num_workers=0,
        collate_fn=collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=8,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 训练指标
    train_losses = []
    val_losses = []
    val_ious = []
    best_iou = 0.0

    # 训练循环
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)

        # 训练
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            images = batch['image']
            masks = batch['mask']

            # 预处理
            processed_images = []
            processed_masks = []
            for i in range(images.shape[0]):
                sample = {'image': images[i], 'mask': masks[i]}
                processed = train_preprocess(sample)
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

        train_loss = total_loss / num_batches

        # 验证
        model.eval()
        total_loss = 0
        total_iou = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image']
                masks = batch['mask']

                processed_images = []
                processed_masks = []
                for i in range(images.shape[0]):
                    sample = {'image': images[i], 'mask': masks[i]}
                    processed = val_preprocess(sample)
                    processed_images.append(processed['image'])
                    processed_masks.append(processed['mask'])

                images = torch.stack(processed_images).to(device)
                masks = torch.stack(processed_masks).to(device)

                outputs = model(images)
                loss = criterion(outputs['out'], masks)
                iou = calculate_iou(outputs['out'], masks)

                total_loss += loss.item()
                total_iou += iou.item()
                num_batches += 1

        val_loss = total_loss / num_batches
        val_iou = total_iou / num_batches

        # 更新学习率
        scheduler.step(val_iou)

        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ious.append(val_iou)

        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        print(f"  验证 IoU: {val_iou:.4f}")

        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            os.makedirs('./checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_iou': val_iou,
            }, './checkpoints/best_model.pth')
            print(f"  >> 保存最佳模型 (IoU: {val_iou:.4f})")

    # 绘制训练曲线
    os.makedirs('./outputs', exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(val_ious, label='Val IoU', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('IoU Curve')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('./outputs/training_curves.png', dpi=150)

    return train_losses, val_losses, val_ious, best_iou


# ============================================================
# 第七步：模型评估
# ============================================================
def evaluate_model(model, val_dataset, device, val_preprocess):
    """评估模型"""
    print("\n" + "=" * 60)
    print("第七步：模型评估")
    print("=" * 60)

    # 加载最佳模型
    checkpoint = torch.load('./checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # 创建采样器
    val_sampler = RandomGeoSampler(dataset=val_dataset, size=256, length=500)

    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        masks = torch.stack([item['mask'] for item in batch])
        masks = masks.long()
        return {'image': images, 'mask': masks}

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        sampler=val_sampler,
        batch_size=8,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # 评估
    all_ious = []
    all_accs = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image']
            masks = batch['mask']

            processed_images = []
            processed_masks = []
            for i in range(images.shape[0]):
                sample = {'image': images[i], 'mask': masks[i]}
                processed = val_preprocess(sample)
                processed_images.append(processed['image'])
                processed_masks.append(processed['mask'])

            images = torch.stack(processed_images).to(device)
            masks = torch.stack(processed_masks).to(device)

            outputs = model(images)
            iou = calculate_iou(outputs['out'], masks)
            acc = calculate_accuracy(outputs['out'], masks)

            all_ious.append(iou.item())
            all_accs.append(acc.item())

    mean_iou = np.mean(all_ious)
    mean_acc = np.mean(all_accs)

    print(f"平均 IoU: {mean_iou:.4f}")
    print(f"平均准确率: {mean_acc:.4f}")

    return mean_iou, mean_acc


# ============================================================
# 第八步：应用到里约热内卢
# ============================================================
def predict_rio(model, device):
    """应用到里约热内卢"""
    print("\n" + "=" * 60)
    print("第八步：应用到里约热内卢")
    print("=" * 60)

    # 检查数据
    rio_dir = './data/rio'
    os.makedirs(rio_dir, exist_ok=True)

    tif_files = [f for f in os.listdir(rio_dir) if f.endswith('.tif')]

    if len(tif_files) == 0:
        print("未找到里约热内卢数据，使用验证集作为示例")
        import shutil
        sample_file = './data/dset-s2/val_scene/S2A_L2A_20190318_N0211_R061_6Bands_S1.tif'
        if os.path.exists(sample_file):
            shutil.copy(sample_file, os.path.join(rio_dir, 'rio_sentinel2.tif'))
            tif_files = ['rio_sentinel2.tif']

    if len(tif_files) == 0:
        print("无法获取数据，跳过此步骤")
        return

    # 加载影像
    import rasterio
    rio_tif = os.path.join(rio_dir, tif_files[0])

    with rasterio.open(rio_tif) as src:
        rio_image = src.read()
        rio_crs = src.crs
        rio_transform = src.transform
        rio_bounds = src.bounds
        rio_shape = rio_shape

    print(f"影像尺寸: {rio_shape[1]} x {rio_shape[2]}")
    print(f"CRS: {rio_crs}")

    # 创建数据集和采样器
    rio_dataset = RasterDataset(paths=rio_dir, crs=rio_crs, res=10)
    sampler = GridGeoSampler(dataset=rio_dataset, size=256, stride=128)

    # 预处理
    preprocess = PreprocessTransform(apply_augmentation=False)

    # 分块推理
    model.eval()
    all_preds = []
    all_bboxes = []

    with torch.no_grad():
        for bbox in sampler:
            sample = rio_dataset[bbox]
            image = sample['image']

            processed = preprocess({'image': image, 'mask': torch.zeros_like(image[0])})
            image = processed['image'].unsqueeze(0).to(device)

            output = model(image)
            pred = output['out'].argmax(dim=1).squeeze(0)

            all_preds.append(pred.cpu().numpy())
            all_bboxes.append(bbox)

    # 拼接预测结果
    H, W = rio_shape[1], rio_shape[2]
    prediction_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for pred, bbox in zip(all_preds, all_bboxes):
        x_slice, y_slice, _ = bbox
        minx, maxx = x_slice.start, x_slice.stop
        miny, maxy = y_slice.start, y_slice.stop

        col_start = int((minx - rio_bounds.left) / 10)
        row_start = int((rio_bounds.top - maxy) / 10)

        col_start = max(0, min(col_start, W - 1))
        row_start = max(0, min(row_start, H - 1))

        patch_h, patch_w = pred.shape
        col_end = min(col_start + patch_w, W)
        row_end = min(row_start + patch_h, H)

        actual_w = col_end - col_start
        actual_h = row_end - row_start

        if actual_w > 0 and actual_h > 0:
            prediction_map[row_start:row_end, col_start:col_end] += pred[:actual_h, :actual_w]
            count_map[row_start:row_end, col_start:col_end] += 1

    count_map[count_map == 0] = 1
    prediction_map = prediction_map / count_map
    prediction_binary = (prediction_map > 0.5).astype(np.uint8)

    # 保存 GeoTIFF
    output_path = './outputs/rio_water_prediction.tif'
    with rasterio.open(output_path, 'w', driver='GTiff',
                       height=H, width=W, count=1,
                       dtype='uint8', crs=rio_crs, transform=rio_transform) as dst:
        dst.write(prediction_binary, 1)

    print(f"水体占比: {np.sum(prediction_binary) / (H * W) * 100:.2f}%")
    print(f"GeoTIFF 已保存: {output_path}")

    return prediction_binary


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("TorchGeo 遥感水体分割项目 - 完整流程")
    print("=" * 60)

    # 第一步：环境配置
    device = check_environment()

    # 第三步：数据加载
    train_dataset, val_dataset = load_data()

    # 第四步：数据预处理
    train_preprocess, val_preprocess = create_preprocessors()

    # 第五步：构建模型
    model, criterion, optimizer, scheduler = create_model(device)

    # 第六步：模型训练
    train_losses, val_losses, val_ious, best_iou = train_model(
        model, train_dataset, val_dataset, criterion, optimizer, scheduler,
        device, train_preprocess, val_preprocess, epochs=20
    )

    # 第七步：模型评估
    mean_iou, mean_acc = evaluate_model(model, val_dataset, device, val_preprocess)

    # 第八步：应用到里约热内卢
    prediction = predict_rio(model, device)

    # 总结
    print("\n" + "=" * 60)
    print("项目完成！")
    print("=" * 60)
    print(f"最佳验证 IoU: {best_iou:.4f}")
    print(f"平均评估 IoU: {mean_iou:.4f}")
    print(f"平均准确率: {mean_acc:.4f}")
    print("\n输出文件:")
    print("  checkpoints/best_model.pth - 最佳模型权重")
    print("  outputs/training_curves.png - 训练曲线")
    print("  outputs/rio_water_prediction.tif - 里约热内卢水体预测")
