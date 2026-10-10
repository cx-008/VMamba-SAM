"""
损失函数模块
包含各种用于分割任务的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class TverskyLoss(nn.Module):
    """Tversky Loss for handling class imbalance"""
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        true_pos = (pred_flat * target_flat).sum()
        false_neg = (target_flat * (1 - pred_flat)).sum()
        false_pos = ((1 - target_flat) * pred_flat).sum()
        
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        
        return 1 - tversky


class IoULoss(nn.Module):
    """IoU Loss for segmentation"""
    def __init__(self, smooth=1e-5):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        return 1 - iou


class CombinedLoss(nn.Module):
    """Combined loss function for better training"""
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        focal = self.focal_loss(pred, target)
        dice = self.dice_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        
        return self.alpha * focal + self.beta * dice + self.gamma * tversky


class BCEDiceLoss(nn.Module):
    """BCE + Dice Loss combination"""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)
        dice = self.dice(pred, target)
        
        return self.bce_weight * bce + self.dice_weight * dice


class FocalDiceLoss(nn.Module):
    """Focal + Dice Loss combination"""
    def __init__(self, focal_weight=0.5, dice_weight=0.5):
        super(FocalDiceLoss, self).__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal = FocalLoss(alpha=1, gamma=2)
        self.dice = DiceLoss()

    def forward(self, pred, target):
        focal = self.focal(pred, target)
        dice = self.dice(pred, target)
        
        return self.focal_weight * focal + self.dice_weight * dice


class WeightedIoULoss(nn.Module):
    """
    加权IoU损失函数 - 用于训练时引导模型学习关键区域
    通过边缘感知权重突出重要区域（如目标边缘、细节等）
    
    设计理念：
    - 训练阶段：使用加权IoU引导模型关注难分割区域
    - 评估阶段：使用标准MIoU进行客观性能评估
    - 两者互补：wiou是"导航工具"，MIoU是"验收标准"
    """
    def __init__(self, kernel_size=31, padding=15, weight_factor=5.0, 
                 edge_enhancement=True, detail_focus=True):
        super(WeightedIoULoss, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight_factor = weight_factor
        self.edge_enhancement = edge_enhancement
        self.detail_focus = detail_focus

    def forward(self, pred, target):
        """
        Args:
            pred: 模型预测输出 (B, C, H, W)
            target: 目标掩码 (B, C, H, W)
        Returns:
            weighted IoU loss - 引导模型学习关键区域
        """
        # 计算边缘感知权重
        # 通过平均池化检测边缘区域，边缘区域权重更高
        weit = 1 + self.weight_factor * torch.abs(
            F.avg_pool2d(target, kernel_size=self.kernel_size, stride=1, padding=self.padding) - target
        )
        
        # 边缘增强：进一步突出边缘区域
        if self.edge_enhancement:
            # 使用Sobel算子检测边缘
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=target.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=target.device)
            
            # 计算边缘强度
            edge_x = F.conv2d(target, sobel_x.view(1, 1, 3, 3), padding=1)
            edge_y = F.conv2d(target, sobel_y.view(1, 1, 3, 3), padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            
            # 将边缘信息融入权重
            weit = weit + 0.5 * edge_magnitude
        
        # 细节聚焦：突出小目标区域
        if self.detail_focus:
            # 检测小连通区域
            small_regions = F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
            detail_mask = (target > 0) & (small_regions < 0.3)
            weit = weit + 2.0 * detail_mask.float()
        
        # 计算加权BCE损失
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        
        # 计算加权IoU损失
        pred_sigmoid = torch.sigmoid(pred)
        inter = ((pred_sigmoid * target) * weit).sum(dim=(2, 3))
        union = ((pred_sigmoid + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        
        # 返回加权BCE和加权IoU的组合
        return (wbce + wiou).mean()


class WeightedStructureLoss(nn.Module):
    """
    加权结构损失函数 - 结合加权BCE和加权IoU
    专门用于训练阶段，引导模型关注边缘和细节区域
    
    训练策略说明：
    - 训练时：使用此加权损失引导模型学习关键区域
    - 评估时：使用标准MIoU进行客观性能评估
    - 目的：通过边缘感知权重帮助模型更好地学习难分割区域
    """
    def __init__(self, kernel_size=31, padding=15, weight_factor=5.0, 
                 bce_weight=0.5, iou_weight=0.5, edge_enhancement=True, detail_focus=True):
        super(WeightedStructureLoss, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.weight_factor = weight_factor
        self.bce_weight = bce_weight
        self.iou_weight = iou_weight
        self.edge_enhancement = edge_enhancement
        self.detail_focus = detail_focus

    def forward(self, pred, target):
        """
        Args:
            pred: 模型预测输出 (B, C, H, W)
            target: 目标掩码 (B, C, H, W)
        Returns:
            weighted structure loss - 训练时的导航工具
        """
        # 计算边缘感知权重
        weit = 1 + self.weight_factor * torch.abs(
            F.avg_pool2d(target, kernel_size=self.kernel_size, stride=1, padding=self.padding) - target
        )
        
        # 边缘增强：突出边缘区域
        if self.edge_enhancement:
            # 使用Sobel算子检测边缘
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=target.device)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=target.device)
            
            # 计算边缘强度
            edge_x = F.conv2d(target, sobel_x.view(1, 1, 3, 3), padding=1)
            edge_y = F.conv2d(target, sobel_y.view(1, 1, 3, 3), padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            
            # 将边缘信息融入权重
            weit = weit + 0.5 * edge_magnitude
        
        # 细节聚焦：突出小目标区域
        if self.detail_focus:
            # 检测小连通区域
            small_regions = F.avg_pool2d(target, kernel_size=5, stride=1, padding=2)
            detail_mask = (target > 0) & (small_regions < 0.3)
            weit = weit + 2.0 * detail_mask.float()
        
        # 加权BCE损失
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
        
        # 加权IoU损失
        pred_sigmoid = torch.sigmoid(pred)
        inter = ((pred_sigmoid * target) * weit).sum(dim=(2, 3))
        union = ((pred_sigmoid + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        
        # 组合损失
        total_loss = self.bce_weight * wbce + self.iou_weight * wiou
        return total_loss.mean()