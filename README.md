# 基于 TorchGeo 的遥感水体分割

使用 Sentinel-2 多光谱卫星影像训练 DeepLabV3 语义分割模型，自动识别水体区域，并将模型应用于里约热内卢地区生成带地理坐标的水体预测 GeoTIFF。

## 项目简介

本项目是一个完整的遥感水体分割实战项目，涵盖从数据加载、预处理、模型训练到推理预测的全流程。核心使用 [TorchGeo](https://torchgeo.readthedocs.io/) 框架处理遥感数据，结合 PyTorch 的 DeepLabV3 模型实现像素级水体识别。

### 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 深度学习框架 | PyTorch | 2.6.0+cu124 |
| 遥感数据处理 | TorchGeo | 0.9.0 |
| 数据增强 | Kornia | 0.8.3 |
| 地理数据读写 | Rasterio | 1.5.0 |
| 可视化 | Matplotlib | 3.11.0 |

### 项目结构

```
├── 01_download_data.py        # 数据下载
├── 02_load_data.py            # 数据加载与采样
├── 03_preprocess.py           # 数据预处理与增强
├── 04_build_model.py          # 模型构建
├── 05_train.py                # 模型训练
├── 06_evaluate.py             # 模型评估与可视化
├── 07_predict_rio.py          # 里约热内卢水体预测
├── 08_complete_pipeline.py    # 完整流程脚本
├── model.py                   # 模型模块（DeepLabV3、损失函数、评估指标）
├── preprocess.py              # 预处理模块（归一化、NDWI、数据增强）
└── notes/                     # 学习笔记
```

## 快速开始

### 环境要求

- Python 3.12+
- NVIDIA GPU（推荐 8GB 以上显存）
- CUDA 12.x

### 安装依赖

```bash
# 创建虚拟环境
uv venv --python 3.12

# 安装 PyTorch（CUDA 12.4）
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 安装其他依赖
uv pip install torchgeo scikit-learn kornia rasterio matplotlib numpy
```

### 数据准备

从 [Hugging Face](https://huggingface.co/datasets/cordmaur/earth_surface_water) 下载 Earth Surface Water 数据集，解压到 `./data/` 目录：

```
data/dset-s2/
├── tra_scene/      # 训练影像（64 个 .tif）
├── tra_truth/      # 训练掩膜（64 个 .tif）
├── val_scene/      # 验证影像（31 个 .tif）
└── val_truth/      # 验证掩膜（31 个 .tif）
```

### 运行方式

**方式一：分步运行**

```bash
python 01_download_data.py    # 数据下载
python 02_load_data.py        # 测试数据加载
python 03_preprocess.py       # 测试预处理
python 04_build_model.py      # 构建模型
python 05_train.py            # 训练模型（约 15 分钟）
python 06_evaluate.py         # 评估模型
python 07_predict_rio.py      # 里约热内卢预测
```

**方式二：一键运行**

```bash
python 08_complete_pipeline.py
```

## 核心流程

```
数据下载 → 数据加载 → 预处理 → 模型构建 → 训练 → 评估 → 推理
                                                    ↓
                                          生成 GeoTIFF 预测文件
```

### 1. 数据加载

使用 TorchGeo 的 `RasterDataset` 加载 GeoTIFF 格式的遥感影像，自动处理坐标参考系（CRS）和分辨率。通过 `&` 操作符合并影像和掩膜数据集。

```python
from torchgeo.datasets import RasterDataset

train_imgs = RasterDataset(paths="./data/dset-s2/tra_scene", crs="EPSG:32642", res=10)
train_masks = RasterDataset(paths="./data/dset-s2/tra_truth", crs="EPSG:32642", res=10)
train_masks.is_image = False
train_dataset = train_imgs & train_masks
```

### 2. 数据采样

训练时使用 `RandomGeoSampler` 随机采样 patch，推理时使用 `GridGeoSampler` 按网格覆盖整景影像。

```python
from torchgeo.samplers import RandomGeoSampler, GridGeoSampler

# 训练
train_sampler = RandomGeoSampler(dataset=train_dataset, size=256, length=1000)

# 推理
sampler = GridGeoSampler(dataset=rio_dataset, size=256, stride=128)
```

### 3. 模型构建

使用 torchvision 的 DeepLabV3（ResNet50 backbone），修改输入通道数为 6（Sentinel-2 的 6 个波段），输出类别数为 2（水体/非水体）。

```python
from model import build_deeplabv3

model = build_deeplabv3(in_channels=6, num_classes=2, pretrained=True)
```

### 4. 损失函数

使用组合损失 `CombinedLoss = CrossEntropyLoss + Dice Loss`，兼顾像素级精度和整体重叠区域。

### 5. 推理输出

将预测结果保存为 GeoTIFF，保留完整的地理坐标信息，可在 QGIS 等 GIS 软件中直接加载。

## 数据格式说明

| 项目 | 影像 | 掩膜 |
|------|------|------|
| 波段数 | 6（B2-B7） | 1 |
| 尺寸 | 868×764 | 868×764 |
| 数据类型 | uint16 | uint8 |
| 值范围 | 0-10000 | 0（非水体）, 1（水体） |
| CRS | EPSG:32642（UTM） | EPSG:32642 |
| 分辨率 | 10m/pixel | 10m/pixel |

### Sentinel-2 波段

| 波段 | 名称 | 分辨率 | 用途 |
|------|------|--------|------|
| B2 | 蓝光 | 10m | 水体检测 |
| B3 | 绿光 | 10m | NDWI 计算 |
| B4 | 红光 | 10m | 植被/土壤 |
| B5 | 红边1 | 20m | 植被健康 |
| B6 | 红边2 | 20m | 植被健康 |
| B7 | 红边3 | 20m | 植被健康 |

## 模型性能

| 指标 | 值 |
|------|-----|
| 验证 IoU | 95.32% |
| 评估 IoU | 94.24% |
| 准确率 | 98.84% |
| 最佳 Epoch | 16 |
| 训练时间 | ~15 分钟（RTX 4060） |

### IoU 分布

| IoU 范围 | 样本数 | 占比 |
|----------|--------|------|
| 90-100% | 54 | 85.7% |
| 70-90% | 8 | 12.7% |
| 50-70% | 1 | 1.6% |
| <50% | 0 | 0.0% |

## 输出文件

运行完成后，在 `./outputs/` 目录下生成：

| 文件 | 说明 |
|------|------|
| `training_curves.png` | 训练曲线图（Loss、IoU） |
| `training_metrics.json` | 训练指标 |
| `prediction_samples.png` | 预测结果可视化 |
| `iou_distribution.png` | IoU 分布图 |
| `evaluation_report.json` | 评估报告 |
| `rio_water_prediction.tif` | 里约热内卢水体预测 GeoTIFF |
| `rio_prediction_visualization.png` | 里约热内卢预测可视化 |

## 关键概念

### 语义分割

给图像中每个像素分配一个类别标签。本项目中，每个像素被分类为"水体"或"非水体"。

### NDWI（归一化差异水体指数）

```
NDWI = (Green - NIR) / (Green + NIR)
```

水体在近红外波段吸收强、绿光波段反射强，NDWI > 0 通常是水体。

### GeoTIFF

带地理坐标信息的 TIFF 格式，包含 CRS（坐标参考系）和 Transform（仿射变换），可在 GIS 软件中定位和分析。

### GeoSampler

在地理坐标空间采样，而非像素坐标空间，保持遥感数据的空间连续性。

## 参考资源

- [TorchGeo 官方文档](https://torchgeo.readthedocs.io/)
- [Earth Surface Water 教程](https://torchgeo.readthedocs.io/en/stable/tutorials/earth_surface_water.html)
- [DeepLabV3 论文](https://arxiv.org/abs/1706.05587)
- [Sentinel-2 波段说明](https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial)

## 许可证

本项目仅用于学习和研究目的。
