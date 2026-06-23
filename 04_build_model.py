"""
第五步：构建模型
使用 DeepLabV3 + ResNet50 构建语义分割模型
"""

import torch
import torch.nn as nn
import torchvision.models.segmentation as models

# ============================================================
# 1. DeepLabV3 架构说明
# ============================================================
print("=" * 60)
print("1. DeepLabV3 架构说明")
print("=" * 60)

print("""
DeepLabV3 架构:

输入图像 (6通道, 256×256)
    ↓
Backbone (ResNet50) - 提取特征
    ↓
ASPP 模块 - 多尺度特征融合
    ├─ 空洞卷积 rate=6
    ├─ 空洞卷积 rate=12
    ├─ 空洞卷积 rate=18
    └─ 全局平均池化
    ↓
分类头 - 像素级分类
    ↓
输出分割图 (2类, 256×256)

Atrous Convolution (空洞卷积):
- 普通卷积: 3×3 核, 感受野 3×3
- 空洞卷积: 3×3 核 + rate=2, 感受野 5×5
- 优点: 不增加参数的情况下扩大感受野

ASPP (Atrous Spatial Pyramid Pooling):
- 并行使用多个不同 rate 的空洞卷积
- 捕获多尺度特征 (大水体 vs 细小河流)
- 包含全局平均池化分支
""")

# ============================================================
# 2. 构建 DeepLabV3 模型
# ============================================================
print("\n" + "=" * 60)
print("2. 构建 DeepLabV3 模型")
print("=" * 60)

def build_deeplabv3(in_channels=6, num_classes=2, pretrained=True):
    """
    构建 DeepLabV3 模型

    Args:
        in_channels: 输入通道数 (默认 6，Sentinel-2 的 6 个波段)
        num_classes: 输出类别数 (默认 2：水体/非水体)
        pretrained: 是否使用预训练权重

    Returns:
        model: DeepLabV3 模型
    """
    print(f"构建 DeepLabV3 模型:")
    print(f"  输入通道: {in_channels}")
    print(f"  输出类别: {num_classes}")
    print(f"  预训练: {pretrained}")

    # 加载预训练的 DeepLabV3 (backbone: ResNet50)
    model = models.deeplabv3_resnet50(pretrained=pretrained)

    # 修改输入通道数
    # 原始 ResNet50 的第一层卷积是 3 通道 (RGB)
    # 我们需要改成 6 通道 (Sentinel-2 的 6 个波段)
    original_conv1 = model.backbone.conv1
    model.backbone.conv1 = nn.Conv2d(
        in_channels=in_channels,      # 新的输入通道数
        out_channels=original_conv1.out_channels,  # 保持输出通道数 64
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias,
    )

    # 如果使用预训练权重，需要适配新的输入通道
    if pretrained:
        # 方法：将原始 3 通道权重复制到 6 通道
        # 前 3 通道保持预训练权重，后 3 通道用均值初始化
        with torch.no_grad():
            # 原始权重形状: [64, 3, 7, 7]
            original_weight = original_conv1.weight

            # 新权重形状: [64, 6, 7, 7]
            new_weight = torch.zeros(
                original_conv1.out_channels,
                in_channels,
                *original_conv1.kernel_size
            )

            # 复制前 3 通道的权重
            new_weight[:, :3, :, :] = original_weight

            # 后 3 通道用前 3 通道的均值初始化
            new_weight[:, 3:, :, :] = original_weight.mean(dim=1, keepdim=True)

            model.backbone.conv1.weight = nn.Parameter(new_weight)

        print("  已适配预训练权重到 6 通道输入")

    # 修改输出类别数
    # 原始 DeepLabV3 输出 21 类 (PASCAL VOC)
    # 我们需要改成 2 类 (水体/非水体)

    # 分类器的最后一层
    model.classifier[4] = nn.Conv2d(
        in_channels=model.classifier[4].in_channels,
        out_channels=num_classes,
        kernel_size=1,
    )

    # 辅助分类器的最后一层 (如果存在)
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(
            in_channels=model.aux_classifier[4].in_channels,
            out_channels=num_classes,
            kernel_size=1,
        )

    print("  已修改输出类别数为 2")

    return model

# 构建模型
model = build_deeplabv3(in_channels=6, num_classes=2, pretrained=True)

# ============================================================
# 3. 查看模型结构
# ============================================================
print("\n" + "=" * 60)
print("3. 模型结构概览")
print("=" * 60)

# 统计模型参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n模型参数统计:")
print(f"  总参数数: {total_params:,}")
print(f"  可训练参数: {trainable_params:,}")
print(f"  模型大小: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")

# 查看主要模块
print(f"\n主要模块:")
print(f"  backbone.conv1: {model.backbone.conv1}")
print(f"  classifier: {model.classifier}")
if model.aux_classifier:
    print(f"  aux_classifier: {model.aux_classifier}")

# ============================================================
# 4. 配置损失函数
# ============================================================
print("\n" + "=" * 60)
print("4. 配置损失函数")
print("=" * 60)

class DiceLoss(nn.Module):
    """
    Dice Loss
    适用于类别不平衡的分割任务
    """
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] 模型输出 (logits)
            target: [B, H, W] 真实标签

        Returns:
            dice loss
        """
        # softmax 将 logits 转换为概率
        pred = torch.softmax(pred, dim=1)

        # 将 target 转换为 one-hot 编码
        # target: [B, H, W] -> [B, C, H, W]
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=2)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        # 计算交集和并集
        intersection = (pred * target_one_hot).sum()
        union = pred.sum() + target_one_hot.sum()

        # 计算 Dice 系数
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice

class CombinedLoss(nn.Module):
    """
    组合损失: CrossEntropyLoss + Dice Loss
    结合两种损失的优势
    """
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] 模型输出 (logits)
            target: [B, H, W] 真实标签

        Returns:
            combined loss
        """
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)

        return self.ce_weight * ce + self.dice_weight * dice

# 创建损失函数
criterion = CombinedLoss(ce_weight=1.0, dice_weight=1.0)

print("损失函数配置:")
print("  类型: CombinedLoss (CrossEntropyLoss + Dice Loss)")
print("  CrossEntropyLoss 权重: 1.0")
print("  Dice Loss 权重: 1.0")
print()
print("为什么用组合损失?")
print("  - CrossEntropyLoss: 对每个像素独立计算，适合类别平衡")
print("  - Dice Loss: 关注整体重叠区域，适合类别不平衡")
print("  - 组合损失: 结合两者优势，平衡像素级和区域级")

# ============================================================
# 5. 配置优化器
# ============================================================
print("\n" + "=" * 60)
print("5. 配置优化器")
print("=" * 60)

# Adam 优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,           # 学习率
    weight_decay=1e-5, # L2 正则化
)

print("优化器配置:")
print("  类型: Adam")
print("  学习率: 1e-4")
print("  权重衰减: 1e-5")
print()
print("Adam vs SGD:")
print("  - Adam: 自适应学习率，收敛快，适合大多数任务")
print("  - SGD: 需要手动调学习率，但泛化性可能更好")
print("  - 本项目使用 Adam，因为更容易调参")

# 学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',      # 监控指标越大越好 (IoU)
    factor=0.5,      # 学习率衰减因子
    patience=5,      # 等待 5 个 epoch 没有改善
    verbose=True,
)

print("\n学习率调度器:")
print("  类型: ReduceLROnPlateau")
print("  监控指标: 验证集 IoU (越大越好)")
print("  衰减因子: 0.5")
print("  耐心: 5 个 epoch")

# ============================================================
# 6. 配置评估指标
# ============================================================
print("\n" + "=" * 60)
print("6. 配置评估指标")
print("=" * 60)

def calculate_iou(pred, target, num_classes=2):
    """
    计算 IoU (Intersection over Union)

    Args:
        pred: [B, C, H, W] 或 [B, H, W]
        target: [B, H, W]
        num_classes: 类别数

    Returns:
        平均 IoU
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)  # [B, H, W]

    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            iou = 1.0  # 如果该类不存在，IoU 为 1
        else:
            iou = intersection / union

        iou_per_class.append(iou)

    return torch.mean(torch.tensor(iou_per_class))

def calculate_accuracy(pred, target):
    """
    计算准确率

    Args:
        pred: [B, C, H, W] 或 [B, H, W]
        target: [B, H, W]

    Returns:
        准确率
    """
    if pred.dim() == 4:
        pred = pred.argmax(dim=1)  # [B, H, W]

    correct = (pred == target).sum().float()
    total = target.numel()

    return correct / total

print("评估指标:")
print("  1. IoU (Intersection over Union)")
print("     - 公式: 交集 / 并集")
print("     - 优点: 对类别不平衡敏感")
print("     - 用途: 主要评估指标")
print()
print("  2. 准确率 (Accuracy)")
print("     - 公式: 正确像素 / 总像素")
print("     - 缺点: 类别不平衡时可能误导")
print("     - 用途: 辅助评估指标")

# ============================================================
# 7. 移动到 GPU
# ============================================================
print("\n" + "=" * 60)
print("7. 移动到 GPU")
print("=" * 60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

print(f"设备: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============================================================
# 8. 测试模型前向传播
# ============================================================
print("\n" + "=" * 60)
print("8. 测试模型前向传播")
print("=" * 60)

try:
    # 创建测试输入
    batch_size = 2
    test_input = torch.randn(batch_size, 6, 256, 256).to(device)
    test_target = torch.randint(0, 2, (batch_size, 256, 256)).to(device)

    print(f"测试输入形状: {test_input.shape}")
    print(f"测试目标形状: {test_target.shape}")

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(test_input)

    print(f"\n输出类型: {type(output)}")
    if isinstance(output, dict):
        print(f"输出键值: {output.keys()}")
        main_output = output['out']
        print(f"主输出形状: {main_output.shape}")

        # 计算损失
        loss = criterion(main_output, test_target)
        print(f"损失值: {loss.item():.4f}")

        # 计算 IoU
        iou = calculate_iou(main_output, test_target)
        print(f"IoU: {iou.item():.4f}")

    print("\n模型前向传播测试成功！")

except Exception as e:
    print(f"测试失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("第五步完成！")
print("=" * 60)
