"""
评估指标模块
包含各种用于分割任务的评估指标和模型复杂度计算
"""

import torch
import torch.nn.functional as F

# Import thop only when needed to avoid global hook registration
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    profile = None


def calculate_model_complexity(model, input_size=(1, 3, 352, 352)):
    """计算模型复杂度和参数量"""
    model.eval()
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算 GFLOPs - 确保输入张量在正确的设备上
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    if THOP_AVAILABLE and profile is not None:
        try:
            # 尝试使用thop计算FLOPs，如果失败则使用替代方法
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            gflops = flops / 1e9
        except AttributeError as e:
            if "'MaxPool2d' object has no attribute 'total_ops'" in str(e):
                print("Warning: thop library compatibility issue with MaxPool2d. Using alternative FLOPs calculation.")
                # 使用替代方法计算FLOPs
                gflops = calculate_flops_alternative(model, dummy_input)
            else:
                print(f"Warning: Could not calculate GFLOPs: {e}")
                gflops = 0.0
        except Exception as e:
            print(f"Warning: Could not calculate GFLOPs: {e}")
            gflops = 0.0
    else:
        print("Warning: thop not available. Using alternative FLOPs calculation.")
        gflops = calculate_flops_alternative(model, dummy_input)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'gflops': gflops,
        'params_mb': total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    }


def calculate_flops_alternative(model, dummy_input):
    """
    用 hook 统计 Conv2d/Linear 的 FLOPs（比参数量估算准确）
    """
    flops = [0]

    def conv_hook(module, inp, out):
        b, c_out, h, w = out.shape
        c_in = inp[0].shape[1]
        kh, kw = module.kernel_size if hasattr(module.kernel_size, '__len__') else (module.kernel_size, module.kernel_size)
        groups = module.groups
        # FLOPs = 2 * B * C_out * H * W * (C_in/groups) * kH * kW
        flops[0] += 2 * b * c_out * h * w * (c_in // groups) * kh * kw

    def linear_hook(module, inp, out):
        b = inp[0].shape[0] if inp[0].dim() > 1 else 1
        in_f = module.in_features
        out_f = module.out_features
        flops[0] += 2 * b * in_f * out_f

    hooks = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, torch.nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    try:
        with torch.no_grad():
            model(dummy_input)
    except Exception:
        pass
    finally:
        for h in hooks:
            h.remove()

    return flops[0] / 1e9


class ComprehensiveMetrics:
    """综合指标计算类"""
    
    @staticmethod
    def calculate_metrics(pred, target, threshold=0.5):
        """计算所有相关指标"""
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = target.float()
        
        # 确保尺寸一致
        if pred_binary.shape != target_binary.shape:
            pred_binary = F.interpolate(
                pred_binary, size=target_binary.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        # 计算混淆矩阵元素
        tp = (pred_binary * target_binary).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        tn = ((1 - pred_binary) * (1 - target_binary)).sum()
        fn = ((1 - pred_binary) * target_binary).sum()
        
        # 基本指标
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # IoU (Jaccard Index)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + 1e-8)
        
        # Dice coefficient
        dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Specificity
        specificity = tn / (tn + fp + 1e-8)
        
        # 计算 MIoU 和 mAcc (按照标准公式)
        # 前景类 (class 1)
        iou_foreground = tp / (tp + fp + fn + 1e-8)
        recall_foreground = tp / (tp + fn + 1e-8)
        
        # 背景类 (class 0)
        iou_background = tn / (tn + fp + fn + 1e-8)
        recall_background = tn / (tn + fp + 1e-8)
        
        # MIoU 和 mAcc
        miou = (iou_foreground + iou_background) / 2
        macc = (recall_foreground + recall_background) / 2
        
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'iou': iou.item(),
            'dice': dice.item(),
            'accuracy': accuracy.item(),
            'specificity': specificity.item(),
            'miou': miou.item(),
            'macc': macc.item(),
            'iou_foreground': iou_foreground.item(),
            'iou_background': iou_background.item(),
            'recall_foreground': recall_foreground.item(),
            'recall_background': recall_background.item()
        }
    
    @staticmethod
    def calculate_batch_metrics(pred, target, threshold=0.5):
        """计算批次指标（用于训练过程中的快速计算）"""
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = target.float()
        
        # 确保尺寸一致
        if pred_binary.shape != target_binary.shape:
            pred_binary = F.interpolate(
                pred_binary, size=target_binary.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        # 计算混淆矩阵元素
        tp = (pred_binary * target_binary).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        tn = ((1 - pred_binary) * (1 - target_binary)).sum()
        fn = ((1 - pred_binary) * target_binary).sum()
        
        # 基本指标
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # IoU
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + 1e-8)
        
        # Dice
        dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Specificity
        specificity = tn / (tn + fp + 1e-8)
        
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'iou': iou.item(),
            'dice': dice.item(),
            'accuracy': accuracy.item(),
            'specificity': specificity.item()
        }


def calculate_hausdorff_distance(pred, target, max_dist=100):
    """计算近似Hausdorff距离"""
    try:
        import cv2
        import numpy as np
        
        # 转换为numpy
        pred_np = pred.squeeze().cpu().numpy()
        target_np = target.squeeze().cpu().numpy()
        
        # 寻找轮廓
        pred_contours, _ = cv2.findContours(
            (pred_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        target_contours, _ = cv2.findContours(
            (target_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(pred_contours) == 0 or len(target_contours) == 0:
            return max_dist
        
        # 计算距离
        pred_points = np.vstack(pred_contours).squeeze()
        target_points = np.vstack(target_contours).squeeze()
        
        if len(pred_points.shape) == 1:
            pred_points = pred_points.reshape(1, -1)
        if len(target_points.shape) == 1:
            target_points = target_points.reshape(1, -1)
        
        # 计算最小距离
        dist1 = np.min(np.linalg.norm(pred_points[:, None] - target_points, axis=2), axis=1)
        dist2 = np.min(np.linalg.norm(target_points[:, None] - pred_points, axis=2), axis=1)
        
        hausdorff = max(np.max(dist1), np.max(dist2))
        return min(hausdorff, max_dist)
        
    except:
        return max_dist


def calculate_specificity(pred, target, threshold=0.5):
    """计算特异性"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()
    
    if pred_binary.shape != target_binary.shape:
        pred_binary = F.interpolate(
            pred_binary, size=target_binary.shape[2:], 
            mode='bilinear', align_corners=False
        )
    
    tn = ((1 - pred_binary) * (1 - target_binary)).sum()
    fp = (pred_binary * (1 - target_binary)).sum()
    
    specificity = tn / (tn + fp + 1e-8)
    return specificity.item()


# 独立的便捷函数
def calculate_metrics(pred, target, threshold=0.5):
    """计算所有相关指标的便捷函数"""
    return ComprehensiveMetrics.calculate_metrics(pred, target, threshold)


def calculate_iou(pred, target, threshold=0.5):
    """计算IoU的便捷函数"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()
    
    # 确保尺寸一致
    if pred_binary.shape != target_binary.shape:
        pred_binary = F.interpolate(
            pred_binary, size=target_binary.shape[2:], 
            mode='bilinear', align_corners=False
        )
    
    # 计算IoU
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    return iou.item()


def calculate_dice(pred, target, threshold=0.5):
    """计算Dice系数的便捷函数"""
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()
    
    # 确保尺寸一致
    if pred_binary.shape != target_binary.shape:
        pred_binary = F.interpolate(
            pred_binary, size=target_binary.shape[2:], 
            mode='bilinear', align_corners=False
        )
    
    # 计算Dice
    intersection = (pred_binary * target_binary).sum()
    dice = (2. * intersection) / (pred_binary.sum() + target_binary.sum() + 1e-8)
    
    return dice.item()