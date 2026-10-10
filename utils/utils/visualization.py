"""
可视化模块
包含训练过程可视化和结果展示功能
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class TrainingVisualizer:
    """训练可视化工具类"""
    
    def __init__(self, save_path, use_tensorboard=True):
        self.save_path = save_path
        self.use_tensorboard = use_tensorboard
        
        # 创建可视化目录
        self.viz_dir = os.path.join(save_path, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # 初始化tensorboard writer
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=os.path.join(save_path, 'tensorboard'))
        
        # 初始化matplotlib样式
        plt.style.use('seaborn-v0_8')
    
    def log_metrics(self, epoch, train_metrics, val_metrics, lr):
        """记录指标到tensorboard"""
        if self.use_tensorboard:
            # 训练指标
            for key, value in train_metrics.items():
                self.tb_writer.add_scalar(f'Train/{key.upper()}', value, epoch)
            
            # 验证指标
            for key, value in val_metrics.items():
                self.tb_writer.add_scalar(f'Val/{key.upper()}', value, epoch)
            
            # 学习率
            self.tb_writer.add_scalar('Learning_Rate', lr, epoch)
    
    def log_images(self, epoch, images, targets, predictions, num_samples=4):
        """记录样本图像到tensorboard"""
        if self.use_tensorboard:
            # 选择随机样本
            indices = torch.randperm(images.size(0))[:num_samples]
            
            # 创建可视化网格
            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
            
            for i, idx in enumerate(indices):
                # 原始图像
                img = images[idx].cpu().permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())
                axes[i, 0].imshow(img)
                axes[i, 0].set_title('Original Image')
                axes[i, 0].axis('off')
                
                # 真实标签
                gt = targets[idx].cpu().squeeze().numpy()
                axes[i, 1].imshow(gt, cmap='gray')
                axes[i, 1].set_title('Ground Truth')
                axes[i, 1].axis('off')
                
                # 预测结果
                pred = torch.sigmoid(predictions[idx]).cpu().squeeze().numpy()
                axes[i, 2].imshow(pred, cmap='gray')
                axes[i, 2].set_title('Prediction')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            self.tb_writer.add_figure('Sample_Predictions', fig, epoch)
            plt.close(fig)
    
    def plot_training_curves(self, train_losses, val_losses, val_metrics, train_metrics=None):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(train_losses) + 1)
        
        # 损失曲线
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice分数
        val_dice_scores = [m['dice'] for m in val_metrics]
        axes[0, 1].plot(epochs, val_dice_scores, 'r-', label='Val Dice')
        if train_metrics:
            train_dice_scores = [m['dice'] for m in train_metrics]
            axes[0, 1].plot(epochs, train_dice_scores, 'b-', label='Train Dice')
        axes[0, 1].set_title('Training and Validation Dice Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # IoU分数
        val_iou_scores = [m['iou'] for m in val_metrics]
        axes[1, 0].plot(epochs, val_iou_scores, 'r-', label='Val IoU')
        if train_metrics:
            train_iou_scores = [m['iou'] for m in train_metrics]
            axes[1, 0].plot(epochs, train_iou_scores, 'b-', label='Train IoU')
        axes[1, 0].set_title('Training and Validation IoU Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # F1分数
        val_f1_scores = [m['f1'] for m in val_metrics]
        axes[1, 1].plot(epochs, val_f1_scores, 'r-', label='Val F1')
        if train_metrics:
            train_f1_scores = [m['f1'] for m in train_metrics]
            axes[1, 1].plot(epochs, train_f1_scores, 'b-', label='Train F1')
        axes[1, 1].set_title('Training and Validation F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_comprehensive_curves(self, train_history, val_history):
        """绘制综合训练曲线"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        epochs = range(1, len(train_history['loss']) + 1)
        
        metrics = ['loss', 'precision', 'recall', 'f1', 'iou', 'dice', 'accuracy', 'miou', 'macc']
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                axes[i].plot(epochs, train_history[metric], 'b-', label=f'Train {metric.upper()}')
                axes[i].plot(epochs, val_history[metric], 'r-', label=f'Val {metric.upper()}')
                axes[i].set_title(f'{metric.upper()} Curve')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.upper())
                axes[i].legend()
                axes[i].grid(True)
        
        # 移除多余的子图
        for i in range(len(metrics), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dir, 'comprehensive_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_metrics_distribution(self, df_metrics, save_dir='./evaluation_results'):
        """绘制指标分布图"""
        metrics_to_plot = ['precision', 'recall', 'f1', 'iou', 'dice', 'specificity', 'accuracy']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_to_plot):
            if i < len(axes):
                axes[i].hist(df_metrics[metric], bins=30, alpha=0.7, edgecolor='black')
                display_name = 'IOU' if metric == 'iou' else metric.upper()
                axes[i].set_title(f'{display_name} Distribution')
                axes[i].set_xlabel(display_name)
                axes[i].set_ylabel('Frequency')
                axes[i].axvline(df_metrics[metric].mean(), color='red', linestyle='--', 
                               label=f'Mean: {df_metrics[metric].mean():.3f}')
                axes[i].legend()
        
        # 移除多余子图
        if len(metrics_to_plot) < len(axes):
            axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def close(self):
        """关闭tensorboard writer"""
        if self.use_tensorboard:
            self.tb_writer.close()
