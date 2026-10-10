# Utils 模块说明

这个 `utils` 模块包含了 SAM2-UNet 项目中使用的所有工具函数，包括损失函数、评估指标、模型复杂度计算和可视化功能。

## 模块结构

```
utils/
├── __init__.py          # 模块初始化文件
├── losses.py            # 损失函数模块
├── metrics.py           # 评估指标模块
└── visualization.py     # 可视化模块
```

## 主要功能

### 1. 损失函数 (losses.py)

#### 基础损失函数
- **FocalLoss** - 处理类别不平衡问题
- **DiceLoss** - 优化分割重叠度
- **TverskyLoss** - 平衡精确率和召回率
- **IoULoss** - 基于IoU的损失函数

#### 组合损失函数
- **CombinedLoss** - 结合Focal、Dice和Tversky损失
- **BCEDiceLoss** - BCE + Dice损失组合
- **FocalDiceLoss** - Focal + Dice损失组合

#### 使用示例
```python
from utils import CombinedLoss, FocalLoss, DiceLoss

# 使用组合损失
criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)

# 使用单个损失
focal_loss = FocalLoss(alpha=1, gamma=2)
dice_loss = DiceLoss()
```

### 2. 评估指标 (metrics.py)

#### 模型复杂度计算
- **calculate_model_complexity()** - 计算参数量、GFLOPs等

#### 综合指标计算
- **ComprehensiveMetrics** - 计算所有相关指标
  - IoU, MIoU, Dice, Accuracy, mAcc
  - Precision, Recall, F1
  - 前景和背景类别的单独指标

#### 使用示例
```python
from utils import ComprehensiveMetrics, calculate_model_complexity

# 计算模型复杂度
complexity = calculate_model_complexity(model, (1, 3, 352, 352))
print(f"Parameters: {complexity['total_params']:,}")
print(f"GFLOPs: {complexity['gflops']:.2f}")

# 计算评估指标
metrics = ComprehensiveMetrics.calculate_metrics(pred, target)
print(f"IoU: {metrics['iou']:.4f}")
print(f"MIoU: {metrics['miou']:.4f}")
```

### 3. 可视化 (visualization.py)

#### 训练可视化
- **TrainingVisualizer** - 训练过程可视化工具
  - TensorBoard日志记录
  - 训练曲线绘制
  - 样本图像可视化
  - 指标分布图

#### 使用示例
```python
from utils import TrainingVisualizer

# 创建可视化器
visualizer = TrainingVisualizer(save_path="./results", use_tensorboard=True)

# 记录指标
visualizer.log_metrics(epoch, train_metrics, val_metrics, lr)

# 绘制训练曲线
visualizer.plot_training_curves(train_losses, val_losses, val_metrics)
```

## 在项目中的使用

### 训练脚本中使用
```python
from utils import CombinedLoss, ComprehensiveMetrics, calculate_model_complexity, TrainingVisualizer

# 初始化损失函数
criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)

# 计算指标
metrics = ComprehensiveMetrics.calculate_metrics(pred, target)

# 计算模型复杂度
complexity = calculate_model_complexity(model)
```

### 评估脚本中使用
```python
from utils import ComprehensiveMetrics

# 在评估过程中使用
evaluator = LeafSegmentationEvaluator(model_path, model_type)
metrics = evaluator.calculate_metrics(pred, target)
```

## 优势

1. **模块化设计** - 功能分离，易于维护
2. **统一接口** - 所有工具函数使用一致的接口
3. **可重用性** - 可以在不同脚本中重复使用
4. **易于扩展** - 可以轻松添加新的损失函数或指标
5. **文档完善** - 每个函数都有详细的文档说明

## 测试

运行测试脚本验证模块功能：
```bash
python test_utils.py
```

## 依赖

- PyTorch
- NumPy
- Matplotlib
- TensorBoard
- thop (用于计算GFLOPs)
- OpenCV (用于Hausdorff距离计算)

## 注意事项

1. 确保所有依赖包已正确安装
2. 在使用前检查输入数据的格式和尺寸
3. 某些函数需要GPU支持以获得最佳性能
4. 可视化功能需要适当的显示环境
