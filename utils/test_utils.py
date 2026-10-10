#!/usr/bin/env python3
"""
测试 utils 模块是否正常工作
"""

import torch
from utils import CombinedLoss, ComprehensiveMetrics, calculate_model_complexity
from lightweight_sam2unet import LightweightSAM2UNet


def test_losses():
    """测试损失函数"""
    print("Testing loss functions...")
    
    # 创建测试数据
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    # 测试各种损失函数
    losses = {
        'CombinedLoss': CombinedLoss(),
    }
    
    for name, loss_fn in losses.items():
        try:
            loss = loss_fn(pred, target)
            print(f"✓ {name}: {loss.item():.4f}")
        except Exception as e:
            print(f"✗ {name}: {e}")


def test_metrics():
    """测试评估指标"""
    print("\nTesting metrics...")
    
    # 创建测试数据
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    try:
        metrics = ComprehensiveMetrics.calculate_metrics(pred, target)
        print("✓ ComprehensiveMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    except Exception as e:
        print(f"✗ ComprehensiveMetrics: {e}")


def test_model_complexity():
    """测试模型复杂度计算"""
    print("\nTesting model complexity...")
    
    try:
        model = LightweightSAM2UNet(num_classes=1)
        complexity = calculate_model_complexity(model, (1, 3, 352, 352))
        print("✓ Model complexity:")
        for key, value in complexity.items():
            if isinstance(value, int):
                print(f"  {key}: {value:,}")
            else:
                print(f"  {key}: {value:.2f}")
    except Exception as e:
        print(f"✗ Model complexity: {e}")


def main():
    print("Testing utils module...")
    print("=" * 40)
    
    test_losses()
    test_metrics()
    test_model_complexity()
    
    print("\n" + "=" * 40)
    print("Utils module test completed!")


if __name__ == "__main__":
    main()
