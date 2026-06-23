"""
第八步：应用到里约热内卢
- 加载训练好的模型
- 使用 GridGeoSampler 分块推理
- 拼接预测结果
- 保存为 GeoTIFF
"""

import os
import torch
import numpy as np
import rasterio
from torchgeo.datasets import RasterDataset
from torchgeo.samplers import GridGeoSampler
from model import build_deeplabv3
from preprocess import PreprocessTransform

# ============================================================
# 1. 配置
# ============================================================
print("=" * 60)
print("1. 配置")
print("=" * 60)

CONFIG = {
    'model_path': './checkpoints/best_model.pth',
    'rio_data_dir': './data/rio',
    'output_dir': './outputs',
    'patch_size': 256,
    'stride': 128,  # 50% 重叠
    'batch_size': 8,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['rio_data_dir'], exist_ok=True)

print("配置:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ============================================================
# 2. 检查/下载里约热内卢数据
# ============================================================
print("\n" + "=" * 60)
print("2. 检查里约热内卢数据")
print("=" * 60)

rio_dir = CONFIG['rio_data_dir']

# 检查是否已有数据
tif_files = [f for f in os.listdir(rio_dir) if f.endswith('.tif')]

if len(tif_files) == 0:
    print("未找到里约热内卢 Sentinel-2 数据")
    print("\n请手动下载数据:")
    print("-" * 40)
    print("""
下载方式一: Copernicus Open Access Hub
  1. 访问 https://scihub.copernicus.eu/
  2. 注册账号并登录
  3. 搜索里约热内卢区域 (经纬度: -22.9, -43.2)
  4. 选择 Sentinel-2 L2A 产品
  5. 下载并解压到 ./data/rio/ 目录

下载方式二: Microsoft Planetary Computer
  1. 访问 https://planetarycomputer.microsoft.com/
  2. 搜索 Sentinel-2 L2A
  3. 选择里约热内卢区域
  4. 下载 B2, B3, B4, B5, B6, B7 波段

下载方式三: 使用示例数据
  - 如果只是测试代码，可以复制训练数据到 ./data/rio/ 目录
    """)
    print("-" * 40)

    # 使用训练数据作为示例
    print("\n使用验证集数据作为示例进行测试...")
    import shutil
    sample_file = './data/dset-s2/val_scene/S2A_L2A_20190318_N0211_R061_6Bands_S1.tif'
    if os.path.exists(sample_file):
        dest_file = os.path.join(rio_dir, 'rio_sentinel2.tif')
        shutil.copy(sample_file, dest_file)
        print(f"已复制示例文件: {dest_file}")
        tif_files = ['rio_sentinel2.tif']
    else:
        print("错误: 未找到示例文件")
        exit(1)
else:
    print(f"找到 {len(tif_files)} 个 TIF 文件:")
    for f in tif_files:
        print(f"  {f}")

# ============================================================
# 3. 加载模型
# ============================================================
print("\n" + "=" * 60)
print("3. 加载模型")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

model = build_deeplabv3(in_channels=6, num_classes=2, pretrained=True)
model = model.to(device)

checkpoint = torch.load(CONFIG['model_path'], map_location=device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

print(f"模型加载成功 (Epoch {checkpoint['epoch']}, IoU {checkpoint['val_iou']:.4f})")

# ============================================================
# 4. 加载里约热内卢影像
# ============================================================
print("\n" + "=" * 60)
print("4. 加载里约热内卢影像")
print("=" * 60)

# 加载 TIF 文件
rio_tif = os.path.join(rio_dir, tif_files[0])

with rasterio.open(rio_tif) as src:
    rio_image = src.read()  # [C, H, W]
    rio_crs = src.crs
    rio_transform = src.transform
    rio_bounds = src.bounds
    rio_shape = rio_image.shape

print(f"影像文件: {rio_tif}")
print(f"波段数: {rio_shape[0]}")
print(f"尺寸: {rio_shape[1]} x {rio_shape[2]}")
print(f"CRS: {rio_crs}")
print(f"边界: {rio_bounds}")

# ============================================================
# 5. 创建数据集和采样器
# ============================================================
print("\n" + "=" * 60)
print("5. 创建数据集和采样器")
print("=" * 60)

# 创建 RasterDataset
rio_dataset = RasterDataset(
    paths=rio_dir,
    crs=rio_crs,
    res=10,
)

# 创建 GridGeoSampler
sampler = GridGeoSampler(
    dataset=rio_dataset,
    size=CONFIG['patch_size'],
    stride=CONFIG['stride'],
)

print(f"数据集: {len(rio_dataset)} 个文件")
print(f"采样器: GridGeoSampler (size={CONFIG['patch_size']}, stride={CONFIG['stride']})")

# ============================================================
# 6. 分块推理
# ============================================================
print("\n" + "=" * 60)
print("6. 分块推理")
print("=" * 60)

# 预处理
preprocess = PreprocessTransform(apply_augmentation=False)

# 收集所有预测结果
all_preds = []
all_bboxes = []

with torch.no_grad():
    for i, bbox in enumerate(sampler):
        # 获取 patch
        sample = rio_dataset[bbox]
        image = sample['image']  # [C, H, W]

        # 预处理
        processed = preprocess({'image': image, 'mask': torch.zeros_like(image[0])})
        image = processed['image'].unsqueeze(0).to(device)  # [1, C, H, W]

        # 预测
        output = model(image)
        pred = output['out'].argmax(dim=1).squeeze(0)  # [H, W]

        all_preds.append(pred.cpu().numpy())
        all_bboxes.append(bbox)

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1} 个 patch...")

print(f"总共处理 {len(all_preds)} 个 patch")

# ============================================================
# 7. 拼接预测结果
# ============================================================
print("\n" + "=" * 60)
print("7. 拼接预测结果")
print("=" * 60)

# 获取影像尺寸
H, W = rio_shape[1], rio_shape[2]

# 创建输出数组
prediction_map = np.zeros((H, W), dtype=np.float32)
count_map = np.zeros((H, W), dtype=np.float32)

# 拼接预测结果
for pred, bbox in zip(all_preds, all_bboxes):
    # bbox 格式: (x_slice, y_slice, time_slice)
    # x_slice: slice(minx, maxx, None)
    # y_slice: slice(miny, maxy, None)
    x_slice, y_slice, _ = bbox

    minx = x_slice.start
    maxx = x_slice.stop
    miny = y_slice.start
    maxy = y_slice.stop

    # 转换为像素坐标
    col_start = int((minx - rio_bounds.left) / 10)
    row_start = int((rio_bounds.top - maxy) / 10)

    # 确保不越界
    col_start = max(0, min(col_start, W - 1))
    row_start = max(0, min(row_start, H - 1))

    # patch 尺寸
    patch_h, patch_w = pred.shape

    # 计算结束位置
    col_end = min(col_start + patch_w, W)
    row_end = min(row_start + patch_h, H)

    # 实际可用的 patch 区域
    actual_w = col_end - col_start
    actual_h = row_end - row_start

    if actual_w > 0 and actual_h > 0:
        # 累加预测结果
        prediction_map[row_start:row_end, col_start:col_end] += pred[:actual_h, :actual_w]
        count_map[row_start:row_end, col_start:col_end] += 1

# 平均预测结果（处理重叠区域）
count_map[count_map == 0] = 1  # 避免除以 0
prediction_map = prediction_map / count_map

# 转换为二值掩膜
prediction_binary = (prediction_map > 0.5).astype(np.uint8)

print("预测结果拼接完成")
print(f"  水体像素数: {np.sum(prediction_binary)}")
print(f"  水体占比: {np.sum(prediction_binary) / (H * W) * 100:.2f}%")

# ============================================================
# 8. 保存为 GeoTIFF
# ============================================================
print("\n" + "=" * 60)
print("8. 保存为 GeoTIFF")
print("=" * 60)

output_path = os.path.join(CONFIG['output_dir'], 'rio_water_prediction.tif')

# 使用 rasterio 保存
with rasterio.open(
    output_path,
    'w',
    driver='GTiff',
    height=H,
    width=W,
    count=1,
    dtype='uint8',
    crs=rio_crs,
    transform=rio_transform,
) as dst:
    # 写入预测结果
    dst.write(prediction_binary, 1)
    # 设置 nodata 值
    dst.nodata = 255

print("GeoTIFF 保存完成")
print(f"  文件: {output_path}")
print(f"  CRS: {rio_crs}")
print(f"  尺寸: {H} x {W}")
print(f"  水体像素: {np.sum(prediction_binary)}")

# ============================================================
# 9. 可视化结果
# ============================================================
print("\n" + "=" * 60)
print("9. 可视化结果")
print("=" * 60)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 原始影像 (RGB)
if rio_shape[0] >= 3:
    rgb = np.stack([rio_image[2], rio_image[1], rio_image[0]], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    axes[0].imshow(rgb)
else:
    axes[0].imshow(rio_image[0], cmap='gray')
axes[0].set_title('Original Image (RGB)')
axes[0].axis('off')

# 预测结果
axes[1].imshow(prediction_binary, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Water Prediction')
axes[1].axis('off')

# 叠加显示
if rio_shape[0] >= 3:
    overlay = rgb.copy()
    # 蓝色标记水体
    water_mask = prediction_binary == 1
    overlay[water_mask] = [0, 0, 1]
    axes[2].imshow(overlay)
else:
    axes[2].imshow(rio_image[0], cmap='gray')
    axes[2].imshow(prediction_binary, cmap='Blues', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
vis_path = os.path.join(CONFIG['output_dir'], 'rio_prediction_visualization.png')
plt.savefig(vis_path, dpi=150, bbox_inches='tight')
print(f"可视化已保存: {vis_path}")

print("\n" + "=" * 60)
print("第八步完成！")
print("=" * 60)
