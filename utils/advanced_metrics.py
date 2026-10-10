"""
高级评估指标模块
基于论文2409.04038v1和其他相关研究的评估标准
包含多种用于分割任务的高级评估指标
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from skimage import measure, morphology
from sklearn.metrics import precision_recall_curve, auc
import warnings
warnings.filterwarnings("ignore")


class AdvancedSegmentationMetrics:
    """高级分割评估指标类"""
    
    @staticmethod
    def boundary_iou(pred, target, dilation_ratio=0.02):
        """
        边界IoU (Boundary IoU)
        专门评估边界分割的准确性
        """
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        target_binary = target.float()
        
        # 确保尺寸一致
        if pred_binary.shape != target_binary.shape:
            pred_binary = F.interpolate(
                pred_binary, size=target_binary.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        # 转换为numpy进行边界检测
        pred_np = pred_binary.squeeze().cpu().numpy()
        target_np = target_binary.squeeze().cpu().numpy()
        
        # 计算膨胀半径
        img_diag = np.sqrt(pred_np.shape[0]**2 + pred_np.shape[1]**2)
        dilation_radius = int(dilation_ratio * img_diag)
        
        if dilation_radius < 1:
            dilation_radius = 1
        
        # 创建结构元素
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (2*dilation_radius+1, 2*dilation_radius+1))
        
        # 获取边界
        pred_boundary = cv2.morphologyEx(pred_np.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        target_boundary = cv2.morphologyEx(target_np.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        
        # 计算边界IoU
        intersection = np.logical_and(pred_boundary, target_boundary).sum()
        union = np.logical_or(pred_boundary, target_boundary).sum()
        
        boundary_iou = intersection / (union + 1e-8)
        return boundary_iou
    
    @staticmethod
    def hausdorff_distance_95(pred, target):
        """
        95%分位数Hausdorff距离
        更鲁棒的形状相似性度量
        """
        try:
            pred_binary = (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy()
            target_binary = target.squeeze().cpu().numpy()
            
            # 获取边界点
            pred_contours, _ = cv2.findContours(
                pred_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            target_contours, _ = cv2.findContours(
                target_binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(pred_contours) == 0 or len(target_contours) == 0:
                return 100.0  # 返回一个大的距离值
            
            # 获取边界点
            pred_points = np.vstack(pred_contours).squeeze()
            target_points = np.vstack(target_contours).squeeze()
            
            if len(pred_points.shape) == 1:
                pred_points = pred_points.reshape(1, -1)
            if len(target_points.shape) == 1:
                target_points = target_points.reshape(1, -1)
            
            # 计算所有点之间的距离
            distances_1 = []
            distances_2 = []
            
            for p_point in pred_points:
                min_dist = np.min(np.linalg.norm(target_points - p_point, axis=1))
                distances_1.append(min_dist)
            
            for t_point in target_points:
                min_dist = np.min(np.linalg.norm(pred_points - t_point, axis=1))
                distances_2.append(min_dist)
            
            # 计算95%分位数
            hd95_1 = np.percentile(distances_1, 95)
            hd95_2 = np.percentile(distances_2, 95)
            
            return max(hd95_1, hd95_2)
            
        except Exception as e:
            return 100.0
    
    @staticmethod
    def average_surface_distance(pred, target):
        """
        平均表面距离 (Average Surface Distance, ASD)
        """
        try:
            pred_binary = (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy()
            target_binary = target.squeeze().cpu().numpy()
            
            # 获取表面点
            pred_surface = morphology.binary_erosion(pred_binary) ^ pred_binary
            target_surface = morphology.binary_erosion(target_binary) ^ target_binary
            
            pred_coords = np.argwhere(pred_surface)
            target_coords = np.argwhere(target_surface)
            
            if len(pred_coords) == 0 or len(target_coords) == 0:
                return 0.0
            
            # 计算距离
            distances = []
            for coord in pred_coords:
                min_dist = np.min(np.linalg.norm(target_coords - coord, axis=1))
                distances.append(min_dist)
            
            for coord in target_coords:
                min_dist = np.min(np.linalg.norm(pred_coords - coord, axis=1))
                distances.append(min_dist)
            
            return np.mean(distances)
            
        except Exception as e:
            return 0.0
    
    @staticmethod
    def volumetric_similarity(pred, target):
        """
        体积相似性 (Volumetric Similarity)
        VS = 1 - |V_pred - V_target| / (V_pred + V_target)
        """
        pred_binary = (torch.sigmoid(pred) > 0.5).float()
        target_binary = target.float()
        
        # 确保尺寸一致
        if pred_binary.shape != target_binary.shape:
            pred_binary = F.interpolate(
                pred_binary, size=target_binary.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        v_pred = pred_binary.sum().item()
        v_target = target_binary.sum().item()
        
        if v_pred + v_target == 0:
            return 1.0
        
        vs = 1 - abs(v_pred - v_target) / (v_pred + v_target)
        return vs
    
    @staticmethod
    def normalized_surface_distance(pred, target):
        """
        归一化表面距离 (Normalized Surface Distance, NSD)
        """
        try:
            pred_binary = (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy()
            target_binary = target.squeeze().cpu().numpy()
            
            # 计算图像对角线长度用于归一化
            img_diag = np.sqrt(pred_binary.shape[0]**2 + pred_binary.shape[1]**2)
            
            # 获取表面点
            pred_surface = morphology.binary_erosion(pred_binary) ^ pred_binary
            target_surface = morphology.binary_erosion(target_binary) ^ target_binary
            
            pred_coords = np.argwhere(pred_surface)
            target_coords = np.argwhere(target_surface)
            
            if len(pred_coords) == 0 or len(target_coords) == 0:
                return 1.0
            
            # 计算归一化距离
            tolerance = 0.1 * img_diag  # 10%的对角线长度作为容差
            
            correct_pred = 0
            for coord in pred_coords:
                min_dist = np.min(np.linalg.norm(target_coords - coord, axis=1))
                if min_dist <= tolerance:
                    correct_pred += 1
            
            correct_target = 0
            for coord in target_coords:
                min_dist = np.min(np.linalg.norm(pred_coords - coord, axis=1))
                if min_dist <= tolerance:
                    correct_target += 1
            
            nsd = (correct_pred + correct_target) / (len(pred_coords) + len(target_coords))
            return nsd
            
        except Exception as e:
            return 0.0
    
    @staticmethod
    def connectivity_error(pred, target):
        """
        连通性误差 (Connectivity Error)
        评估分割的拓扑正确性
        """
        try:
            pred_binary = (torch.sigmoid(pred) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
            target_binary = target.squeeze().cpu().numpy().astype(np.uint8)
            
            # 计算连通分量
            pred_labels = measure.label(pred_binary, connectivity=2)
            target_labels = measure.label(target_binary, connectivity=2)
            
            pred_components = pred_labels.max()
            target_components = target_labels.max()
            
            # 连通性误差 = |预测连通分量数 - 真实连通分量数|
            connectivity_error = abs(pred_components - target_components)
            
            return connectivity_error
            
        except Exception as e:
            return 0
    
    @staticmethod
    def panoptic_quality(pred, target, threshold=0.5):
        """
        全景质量 (Panoptic Quality, PQ)
        PQ = (TP * IoU_sum) / (TP + 0.5 * FP + 0.5 * FN)
        """
        pred_binary = (torch.sigmoid(pred) > threshold).squeeze().cpu().numpy().astype(np.uint8)
        target_binary = target.squeeze().cpu().numpy().astype(np.uint8)
        
        # 计算连通分量
        pred_labels = measure.label(pred_binary, connectivity=2)
        target_labels = measure.label(target_binary, connectivity=2)
        
        # 获取所有区域的属性
        pred_props = measure.regionprops(pred_labels)
        target_props = measure.regionprops(target_labels)
        
        if len(pred_props) == 0 and len(target_props) == 0:
            return 1.0
        
        if len(pred_props) == 0 or len(target_props) == 0:
            return 0.0
        
        # 计算IoU矩阵
        iou_matrix = np.zeros((len(pred_props), len(target_props)))
        
        for i, pred_prop in enumerate(pred_props):
            pred_mask = (pred_labels == pred_prop.label)
            for j, target_prop in enumerate(target_props):
                target_mask = (target_labels == target_prop.label)
                
                intersection = np.logical_and(pred_mask, target_mask).sum()
                union = np.logical_or(pred_mask, target_mask).sum()
                
                if union > 0:
                    iou_matrix[i, j] = intersection / union
        
        # 匹配预测和真实区域
        matched_pairs = []
        used_pred = set()
        used_target = set()
        
        # 贪婪匹配：选择IoU最大的配对
        while True:
            max_iou = 0
            max_i, max_j = -1, -1
            
            for i in range(len(pred_props)):
                if i in used_pred:
                    continue
                for j in range(len(target_props)):
                    if j in used_target:
                        continue
                    if iou_matrix[i, j] > max_iou and iou_matrix[i, j] > threshold:
                        max_iou = iou_matrix[i, j]
                        max_i, max_j = i, j
            
            if max_i == -1:
                break
            
            matched_pairs.append((max_i, max_j, max_iou))
            used_pred.add(max_i)
            used_target.add(max_j)
        
        # 计算PQ
        tp = len(matched_pairs)
        fp = len(pred_props) - tp
        fn = len(target_props) - tp
        
        if tp == 0:
            return 0.0
        
        iou_sum = sum([pair[2] for pair in matched_pairs])
        pq = (tp * iou_sum) / (tp + 0.5 * fp + 0.5 * fn) / tp
        
        return pq


class RobustMetrics:
    """鲁棒性评估指标"""
    
    @staticmethod
    def calculate_all_advanced_metrics(pred, target):
        """计算所有高级指标"""
        metrics = {}
        
        # 基础高级指标
        metrics['boundary_iou'] = AdvancedSegmentationMetrics.boundary_iou(pred, target)
        metrics['hausdorff_95'] = AdvancedSegmentationMetrics.hausdorff_distance_95(pred, target)
        metrics['avg_surface_distance'] = AdvancedSegmentationMetrics.average_surface_distance(pred, target)
        metrics['volumetric_similarity'] = AdvancedSegmentationMetrics.volumetric_similarity(pred, target)
        metrics['normalized_surface_distance'] = AdvancedSegmentationMetrics.normalized_surface_distance(pred, target)
        metrics['connectivity_error'] = AdvancedSegmentationMetrics.connectivity_error(pred, target)
        metrics['panoptic_quality'] = AdvancedSegmentationMetrics.panoptic_quality(pred, target)
        
        return metrics
    
    @staticmethod
    def precision_recall_auc(pred, target):
        """
        精确率-召回率曲线下面积 (PR-AUC)
        """
        pred_probs = torch.sigmoid(pred).flatten().cpu().numpy()
        target_binary = target.flatten().cpu().numpy()
        
        if len(np.unique(target_binary)) < 2:
            return 0.0
        
        precision, recall, _ = precision_recall_curve(target_binary, pred_probs)
        pr_auc = auc(recall, precision)
        
        return pr_auc
    
    @staticmethod
    def matthews_correlation_coefficient(pred, target, threshold=0.5):
        """
        马修斯相关系数 (Matthews Correlation Coefficient, MCC)
        """
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
        
        # 计算MCC
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            return 0.0
        
        mcc = numerator / denominator
        return mcc.item()
    
    @staticmethod
    def balanced_accuracy(pred, target, threshold=0.5):
        """
        平衡准确率 (Balanced Accuracy)
        """
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = target.float()
        
        # 确保尺寸一致
        if pred_binary.shape != target_binary.shape:
            pred_binary = F.interpolate(
                pred_binary, size=target_binary.shape[2:], 
                mode='bilinear', align_corners=False
            )
        
        # 计算敏感性和特异性
        tp = (pred_binary * target_binary).sum()
        fp = (pred_binary * (1 - target_binary)).sum()
        tn = ((1 - pred_binary) * (1 - target_binary)).sum()
        fn = ((1 - pred_binary) * target_binary).sum()
        
        sensitivity = tp / (tp + fn + 1e-8)  # 召回率
        specificity = tn / (tn + fp + 1e-8)
        
        balanced_acc = (sensitivity + specificity) / 2
        return balanced_acc.item()


class MultiScaleMetrics:
    """多尺度评估指标"""
    
    @staticmethod
    def multi_scale_iou(pred, target, scales=[0.5, 0.75, 1.0, 1.25, 1.5]):
        """
        多尺度IoU评估
        在不同尺度下评估分割性能
        """
        original_size = target.shape[2:]
        ious = []
        
        for scale in scales:
            # 计算新尺寸
            new_size = [int(original_size[0] * scale), int(original_size[1] * scale)]
            
            # 调整预测和目标尺寸
            pred_scaled = F.interpolate(pred, size=new_size, mode='bilinear', align_corners=False)
            target_scaled = F.interpolate(target, size=new_size, mode='bilinear', align_corners=False)
            
            # 计算IoU
            pred_binary = (torch.sigmoid(pred_scaled) > 0.5).float()
            target_binary = target_scaled.float()
            
            intersection = (pred_binary * target_binary).sum()
            union = pred_binary.sum() + target_binary.sum() - intersection
            iou = intersection / (union + 1e-8)
            
            ious.append(iou.item())
        
        return {
            'multi_scale_ious': ious,
            'mean_multi_scale_iou': np.mean(ious),
            'std_multi_scale_iou': np.std(ious)
        }


def calculate_comprehensive_metrics(pred, target, threshold=0.5):
    """
    计算综合评估指标
    包含基础指标、高级指标和鲁棒性指标
    """
    # 基础指标
    from .metrics import ComprehensiveMetrics
    basic_metrics = ComprehensiveMetrics.calculate_metrics(pred, target, threshold)
    
    # 高级指标
    advanced_metrics = RobustMetrics.calculate_all_advanced_metrics(pred, target)
    
    # 鲁棒性指标
    robust_metrics = {
        'pr_auc': RobustMetrics.precision_recall_auc(pred, target),
        'mcc': RobustMetrics.matthews_correlation_coefficient(pred, target, threshold),
        'balanced_accuracy': RobustMetrics.balanced_accuracy(pred, target, threshold)
    }
    
    # 多尺度指标
    multi_scale_metrics = MultiScaleMetrics.multi_scale_iou(pred, target)
    
    # 合并所有指标
    all_metrics = {
        **basic_metrics,
        **advanced_metrics,
        **robust_metrics,
        **multi_scale_metrics
    }
    
    return all_metrics


if __name__ == "__main__":
    # 测试代码
    print("Testing advanced metrics...")
    
    # 创建测试数据
    pred = torch.randn(1, 1, 256, 256)
    target = torch.randint(0, 2, (1, 1, 256, 256)).float()
    
    # 测试高级指标
    try:
        metrics = calculate_comprehensive_metrics(pred, target)
        print("Comprehensive metrics calculated successfully!")
        
        print("\nMetrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error calculating metrics: {e}")
    
    print("\nAdvanced metrics test completed!")
