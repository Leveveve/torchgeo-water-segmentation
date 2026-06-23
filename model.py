"""
模型模块
提供 DeepLabV3 模型构建、损失函数、评估指标等功能
"""

import torch
import torch.nn as nn
import torchvision.models.segmentation as models


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
    # 加载预训练的 DeepLabV3 (backbone: ResNet50)
    model = models.deeplabv3_resnet50(weights=models.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None)

    # 修改输入通道数
    original_conv1 = model.backbone.conv1
    model.backbone.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=original_conv1.bias,
    )

    # 适配预训练权重到新的输入通道
    if pretrained:
        with torch.no_grad():
            original_weight = original_conv1.weight
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

    # 修改输出类别数
    model.classifier[4] = nn.Conv2d(
        in_channels=model.classifier[4].in_channels,
        out_channels=num_classes,
        kernel_size=1,
    )

    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(
            in_channels=model.aux_classifier[4].in_channels,
            out_channels=num_classes,
            kernel_size=1,
        )

    return model


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
        pred = torch.softmax(pred, dim=1)
        target_one_hot = torch.nn.functional.one_hot(target, num_classes=2)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        intersection = (pred * target_one_hot).sum()
        union = pred.sum() + target_one_hot.sum()

        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    组合损失: CrossEntropyLoss + Dice Loss
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
        pred = pred.argmax(dim=1)

    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            iou = 1.0
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
        pred = pred.argmax(dim=1)

    correct = (pred == target).sum().float()
    total = target.numel()

    return correct / total
