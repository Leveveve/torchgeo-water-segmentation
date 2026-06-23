"""
第三步：数据下载与加载
Earth Surface Water 数据集下载、探索与加载
"""

import os
import glob
from huggingface_hub import snapshot_download

# ============================================================
# 1. 下载数据集
# ============================================================
print("=" * 60)
print("1. 下载 Earth Surface Water 数据集")
print("=" * 60)

# 数据保存路径
DATA_DIR = "./data"
DATASET_REPO = "torchgeo/earth_surface_water"

# 检查是否已经下载
extracted_dir = os.path.join(DATA_DIR, "water_v1.0")
if os.path.exists(extracted_dir) and len(os.listdir(extracted_dir)) > 0:
    print(f"数据已存在: {extracted_dir}")
else:
    print(f"开始从 Hugging Face 下载数据集...")
    print(f"仓库: {DATASET_REPO}")
    print(f"保存路径: {DATA_DIR}")
    print("（首次下载需要一些时间，请耐心等待）")

    # 使用 huggingface_hub 下载整个数据集
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=DATA_DIR,
    )
    print("下载完成！")

# ============================================================
# 2. 探索数据集结构
# ============================================================
print("\n" + "=" * 60)
print("2. 探索数据集结构")
print("=" * 60)

# 列出数据目录结构
print(f"\n数据目录: {DATA_DIR}")
for root, dirs, files in os.walk(DATA_DIR):
    level = root.replace(DATA_DIR, "").count(os.sep)
    if level > 4:  # 限制深度
        continue
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    if level < 3:
        subindent = " " * 2 * (level + 1)
        for file in sorted(files)[:8]:
            fsize = os.path.getsize(os.path.join(root, file)) / 1024 / 1024
            print(f"{subindent}{file} ({fsize:.1f} MB)")
        if len(files) > 8:
            print(f"{subindent}... 还有 {len(files)-8} 个文件")

# ============================================================
# 3. 统计数据集信息
# ============================================================
print("\n" + "=" * 60)
print("3. 数据集统计")
print("=" * 60)

# 查找所有 tif 文件
tif_files = glob.glob(os.path.join(DATA_DIR, "**", "*.tif"), recursive=True)
png_files = glob.glob(os.path.join(DATA_DIR, "**", "*.png"), recursive=True)
all_files = tif_files + png_files

print(f"\n总共找到:")
print(f"  TIF 文件: {len(tif_files)} 个")
print(f"  PNG 文件: {len(png_files)} 个")

# 分类统计
images = [f for f in all_files if "images" in f.lower() or "image" in f.lower()]
masks = [f for f in all_files if "masks" in f.lower() or "mask" in f.lower() or "label" in f.lower()]
print(f"\n  影像文件: {len(images)} 个")
print(f"  掩膜文件: {len(masks)} 个")

# 打印前几个文件路径示例
if images:
    print(f"\n影像示例:")
    for f in images[:3]:
        print(f"  {f}")
if masks:
    print(f"\n掩膜示例:")
    for f in masks[:3]:
        print(f"  {f}")

print("\n" + "=" * 60)
print("数据下载与探索完成！")
print("=" * 60)
