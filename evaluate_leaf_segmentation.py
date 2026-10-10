import os
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
import argparse
import json
from datetime import datetime

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
mamba_ablation_dir = os.path.join(current_dir, 'mamba_ablation_experiments')

for path in [current_dir, mamba_ablation_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

try:
    import mamba_ablation_configs
    print("✅ 成功导入mamba_ablation_configs模块")
except ImportError as e:
    print(f"⚠️ 无法导入mamba_ablation_configs模块: {e}")
    print("🔄 创建备用配置类...")

    from dataclasses import dataclass
    from typing import Dict, List, Any

    @dataclass
    class MambaAblationConfig:

        name: str = "default"
        description: str = "Default configuration"
        use_mamba: bool = False
        use_skip_connections: bool = True
        use_auxiliary_loss: bool = True
        d_state: int = 8
        epochs: int = 50
        batch_size: int = 8
        lr: float = 1e-4
        weight_decay: float = 1e-4
        save_interval: int = 10

    class MambaAblationConfigsModule:
        MambaAblationConfig = MambaAblationConfig

    mamba_ablation_configs = MambaAblationConfigsModule()
    print("✅ 创建备用配置类成功")

print(f"Root directory: {root_dir}")

from utils import ComprehensiveMetrics, calculate_model_complexity
from dataset import TestDataset

from models import UNet, UNetPlusPlus, DeepLabV3Plus, PSPNet, FCN, SegNet, FPN

sam2_path = os.path.join(root_dir, 'sam2')
sam2_configs_path = os.path.join(root_dir, 'sam2_configs')

print(f"Sam2 path: {sam2_path}")
print(f"Sam2 configs path: {sam2_configs_path}")

if os.path.exists(sam2_path) and sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)
    print(f"✅ 添加sam2路径: {sam2_path}")

if os.path.exists(sam2_configs_path) and sam2_configs_path not in sys.path:
    sys.path.insert(0, sam2_configs_path)
    print(f"✅ 添加sam2_configs路径: {sam2_configs_path}")

try:
    import sam2
    print("✅ sam2模块导入成功")
except ImportError as e:
    print(f"❌ sam2模块导入失败: {e}")

try:
    import sam2_configs
    print("✅ sam2_configs模块导入成功")
except ImportError as e:
    print(f"❌ sam2_configs模块导入失败: {e}")

try:
    import ablation_configs
    print("✅ ablation_configs模块导入成功")
except ImportError:
    import types
    ablation_configs = types.ModuleType('ablation_configs')
    ablation_configs.AblationConfig = type('AblationConfig', (), {})
    sys.modules['ablation_configs'] = ablation_configs
    print("✅ 创建虚拟ablation_configs模块")

mamba_ablation_paths = [
    os.path.join(current_dir, 'mamba_ablation'),
    os.path.join(current_dir, 'mamba_ablation_experiments'),
    os.path.join(root_dir, 'SAM2-UNet', 'mamba_ablation'),
    os.path.join(root_dir, 'mamba_ablation'),
    os.path.join(root_dir, 'mamba_ablation_experiments'),
    os.path.join(root_dir, 'lightweight_ablation')
]

for mamba_path in mamba_ablation_paths:
    if os.path.exists(mamba_path) and mamba_path not in sys.path:
        sys.path.insert(0, mamba_path)
        print(f"✅ 添加mamba路径: {mamba_path}")
    elif os.path.exists(mamba_path):
        print(f"🔍 路径已存在: {mamba_path}")
    else:
        print(f"❌ 路径不存在: {mamba_path}")

try:
    from latest_mamba_sam2unet import LatestMambaLightweightSAM2UNet
    LATEST_MAMBA_AVAILABLE = True
    print("✅ LatestMambaLightweightSAM2UNet 导入成功")
except ImportError as e:
    LATEST_MAMBA_AVAILABLE = False
    print(f"❌ LatestMambaLightweightSAM2UNet 导入失败: {e}")
    for mamba_path in mamba_ablation_paths:
        try:
            sys.path.insert(0, mamba_path)
            from latest_mamba_sam2unet import LatestMambaLightweightSAM2UNet
            LATEST_MAMBA_AVAILABLE = True
            print(f"✅ 从 {mamba_path} 成功导入 LatestMambaLightweightSAM2UNet")
            break
        except ImportError:
            continue

try:
    from mamba_ablation.mamba_sam2unet_v3 import MambaSAM2UNetV3
    MAMBA_V3_AVAILABLE = True
    print("✅ MambaSAM2UNetV3 导入成功")
except ImportError as e:
    MAMBA_V3_AVAILABLE = False
    print(f"⚠️ MambaSAM2UNetV3 导入失败（需要 mamba_ssm）: {e}")
    for mamba_path in mamba_ablation_paths:
        try:
            sys.path.insert(0, mamba_path)
            from mamba_sam2unet_v3 import MambaSAM2UNetV3
            MAMBA_V3_AVAILABLE = True
            print(f"✅ 从 {mamba_path} 成功导入 MambaSAM2UNetV3")
            break
        except ImportError:
            continue

try:
    from mamba_ablation.comparison_models.vm_unet import VMUNet
    VM_UNET_AVAILABLE = True
    print("✅ VMUNet 导入成功")
except ImportError as e:
    try:
        from mamba_ablation_experiments.comparison_models.vm_unet import VMUNet
        VM_UNET_AVAILABLE = True
        print("✅ VMUNet 从 mamba_ablation_experiments 导入成功")
    except ImportError:
        try:
            from comparison_models.vm_unet import VMUNet
            VM_UNET_AVAILABLE = True
            print("✅ VMUNet 从 comparison_models 导入成功")
        except ImportError:
            VM_UNET_AVAILABLE = False
            print(f"⚠️ VMUNet 导入失败: {e}")

try:
    from mamba_ablation.comparison_models.garden_net import GARDEN
    GARDEN_NET_AVAILABLE = True
    print("✅ GARDEN 导入成功")
except ImportError as e:
    try:
        from mamba_ablation_experiments.comparison_models.garden_net import GARDEN
        GARDEN_NET_AVAILABLE = True
        print("✅ GARDEN 从 mamba_ablation_experiments 导入成功")
    except ImportError:
        try:
            from comparison_models.garden_net import GARDEN
            GARDEN_NET_AVAILABLE = True
            print("✅ GARDEN 从 comparison_models 导入成功")
        except ImportError:
            GARDEN_NET_AVAILABLE = False
            print(f"⚠️ GARDEN 导入失败: {e}")

try:
    from mamba_ablation.comparison_models.star_net import STARNet
    STAR_NET_AVAILABLE = True
    print("✅ STARNet 导入成功")
except ImportError as e:
    try:
        from mamba_ablation_experiments.comparison_models.star_net import STARNet
        STAR_NET_AVAILABLE = True
        print("✅ STARNet 从 mamba_ablation_experiments 导入成功")
    except ImportError:
        try:
            from comparison_models.star_net import STARNet
            STAR_NET_AVAILABLE = True
            print("✅ STARNet 从 comparison_models 导入成功")
        except ImportError:
            STAR_NET_AVAILABLE = False
            print(f"⚠️ STARNet 导入失败: {e}")

try:
    from mamba_ablation.comparison_models.double_unet import DoubleUNet
    DOUBLE_UNET_AVAILABLE = True
    print("✅ DoubleUNet 导入成功")
except ImportError as e:
    try:
        from mamba_ablation_experiments.comparison_models.double_unet import DoubleUNet
        DOUBLE_UNET_AVAILABLE = True
        print("✅ DoubleUNet 从 mamba_ablation_experiments 导入成功")
    except ImportError:
        try:
            from comparison_models.double_unet import DoubleUNet
            DOUBLE_UNET_AVAILABLE = True
            print("✅ DoubleUNet 从 comparison_models 导入成功")
        except ImportError:
            DOUBLE_UNET_AVAILABLE = False
            print(f"⚠️ DoubleUNet 导入失败: {e}")

try:
    from mamba_ablation.comparison_models.emcad import EMCAD
    EMCAD_AVAILABLE = True
    print("✅ EMCAD 导入成功")
except ImportError as e:
    try:
        from mamba_ablation_experiments.comparison_models.emcad import EMCAD
        EMCAD_AVAILABLE = True
        print("✅ EMCAD 从 mamba_ablation_experiments 导入成功")
    except ImportError:
        try:
            from comparison_models.emcad import EMCAD
            EMCAD_AVAILABLE = True
            print("✅ EMCAD 从 comparison_models 导入成功")
        except ImportError:
            EMCAD_AVAILABLE = False
            print(f"⚠️ EMCAD 导入失败: {e}")

try:
    from mamba_ablation.comparison_models.nnwnet import nnWNet
    NNWNET_AVAILABLE = True
    print("✅ nnWNet 导入成功")
except ImportError as e:
    try:
        from mamba_ablation_experiments.comparison_models.nnwnet import nnWNet
        NNWNET_AVAILABLE = True
        print("✅ nnWNet 从 mamba_ablation_experiments 导入成功")
    except ImportError:
        try:
            from comparison_models.nnwnet import nnWNet
            NNWNET_AVAILABLE = True
            print("✅ nnWNet 从 comparison_models 导入成功")
        except ImportError:
            NNWNET_AVAILABLE = False
            print(f"⚠️ nnWNet 导入失败: {e}")

try:
    from mamba_ablation.comparison_models.attention_unet import AttentionUNet
    ATTENTION_UNET_AVAILABLE = True
    print("✅ AttentionUNet 导入成功")
except ImportError as e:
    try:
        from mamba_ablation_experiments.comparison_models.attention_unet import AttentionUNet
        ATTENTION_UNET_AVAILABLE = True
        print("✅ AttentionUNet 从 mamba_ablation_experiments 导入成功")
    except ImportError:
        try:
            from comparison_models.attention_unet import AttentionUNet
            ATTENTION_UNET_AVAILABLE = True
            print("✅ AttentionUNet 从 comparison_models 导入成功")
        except ImportError:
            ATTENTION_UNET_AVAILABLE = False
            print(f"⚠️ AttentionUNet 导入失败: {e}")

try:
    from mamba_ablation.comparison_models.emsam_net import EMSAM
    EMSAM_AVAILABLE = True
    print("✅ EMSAM 导入成功")
except ImportError as e:
    try:
        from mamba_ablation_experiments.comparison_models.emsam_net import EMSAM
        EMSAM_AVAILABLE = True
        print("✅ EMSAM 从 mamba_ablation_experiments 导入成功")
    except ImportError:
        try:
            from comparison_models.emsam_net import EMSAM
            EMSAM_AVAILABLE = True
            print("✅ EMSAM 从 comparison_models 导入成功")
        except ImportError:
            EMSAM_AVAILABLE = False
            print(f"⚠️ EMSAM 导入失败: {e}")

try:
    from mamba_ablation.comparison_models.swin_unet_agri import SwinUNetAgri
    SWIN_UNET_AVAILABLE = True
    print("✅ SwinUNetAgri 导入成功")
except ImportError as e:
    try:
        from mamba_ablation_experiments.comparison_models.swin_unet_agri import SwinUNetAgri
        SWIN_UNET_AVAILABLE = True
        print("✅ SwinUNetAgri 从 mamba_ablation_experiments 导入成功")
    except ImportError:
        try:
            from comparison_models.swin_unet_agri import SwinUNetAgri
            SWIN_UNET_AVAILABLE = True
            print("✅ SwinUNetAgri 从 comparison_models 导入成功")
        except ImportError:
            SWIN_UNET_AVAILABLE = False
            print(f"⚠️ SwinUNetAgri 导入失败: {e}")

try:
    from mamba_ablation.comparison_models.transunet_agri import TransUNetAgri
    TRANSUNET_AVAILABLE = True
    print("✅ TransUNetAgri 导入成功")
except ImportError as e:
    try:
        from mamba_ablation_experiments.comparison_models.transunet_agri import TransUNetAgri
        TRANSUNET_AVAILABLE = True
        print("✅ TransUNetAgri 从 mamba_ablation_experiments 导入成功")
    except ImportError:
        try:
            from comparison_models.transunet_agri import TransUNetAgri
            TRANSUNET_AVAILABLE = True
            print("✅ TransUNetAgri 从 comparison_models 导入成功")
        except ImportError:
            TRANSUNET_AVAILABLE = False
            print(f"⚠️ TransUNetAgri 导入失败: {e}")

try:
    from mamba_ablation.comparison_models.lightm_unet import LightMUNet
    LIGHTM_UNET_AVAILABLE = True
    print("✅ LightM-UNet 导入成功")
except ImportError as e:
    try:
        from mamba_ablation_experiments.comparison_models.lightm_unet import LightMUNet
        LIGHTM_UNET_AVAILABLE = True
        print("✅ LightM-UNet 从 mamba_ablation_experiments 导入成功")
    except ImportError:
        try:
            from comparison_models.lightm_unet import LightMUNet
            LIGHTM_UNET_AVAILABLE = True
            print("✅ LightM-UNet 从 comparison_models 导入成功")
        except ImportError:
            LIGHTM_UNET_AVAILABLE = False
            print(f"⚠️ LightM-UNet 导入失败: {e}")

try:
    from mamba_ablation.baseline_sam2unet import BaselineSAM2UNet
    BASELINE_SAM2UNET_AVAILABLE = True
    print("✅ BaselineSAM2UNet 导入成功")
except ImportError as e:
    BASELINE_SAM2UNET_AVAILABLE = False
    print(f"❌ BaselineSAM2UNet 导入失败: {e}")

    for mamba_path in mamba_ablation_paths:
        try:
            sys.path.insert(0, mamba_path)
            from baseline_sam2unet import BaselineSAM2UNet
            BASELINE_SAM2UNET_AVAILABLE = True
            print(f"✅ 从 {mamba_path} 成功导入 BaselineSAM2UNet")
            break
        except ImportError:
            continue

try:
    from SAM2UNet import SAM2UNet
    SAM2UNET_AVAILABLE = True
    print("✅ SAM2UNet 导入成功")
except ImportError as e:
    SAM2UNET_AVAILABLE = False
    print(f"❌ SAM2UNet 导入失败: {e}")

class LeafSegmentationEvaluator:

    def __init__(self, model_path, model_type='unet', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        if model_type == 'auto':

            print("🔍 自动检测模型类型...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model' in checkpoint:
                    model_state_dict = checkpoint['model']
                else:
                    model_state_dict = checkpoint

                detected_model_type = self.detect_model_type(model_state_dict)
                print(f"🔍 检测到的模型类型: {detected_model_type}")

                if detected_model_type == 'mamba_v3':
                    if not MAMBA_V3_AVAILABLE:
                        raise ImportError("MambaSAM2UNetV3 不可用，请安装 mamba_ssm")
                    checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                    print(f"🔍 检测到checkpoint路径: {checkpoint_path}")
                    self.model = MambaSAM2UNetV3(
                        checkpoint_path=checkpoint_path,
                        num_classes=1
                    )
                    self.model_type = 'mamba_v3'
                elif detected_model_type == 'vm_unet':
                    if not VM_UNET_AVAILABLE:
                        raise ImportError("VMUNet 不可用，请安装 mamba_ssm")
                    self.model = VMUNet(
                        in_channels=3, num_classes=1,
                        depths=(2, 2, 9, 2), depths_decoder=(2, 9, 2, 2),
                        d_state=8, drop_path_rate=0.2,
                    )
                    self.model_type = 'vm_unet'
                elif detected_model_type == 'garden_net':
                    if not GARDEN_NET_AVAILABLE:
                        raise ImportError("GARDEN 不可用，请安装 mamba_ssm")
                    self.model = GARDEN(in_channels=3, num_classes=1)
                    self.model_type = 'garden_net'
                elif detected_model_type == 'star_net':
                    if not STAR_NET_AVAILABLE:
                        raise ImportError("STARNet 不可用")
                    self.model = STARNet(in_channels=3, num_classes=1)
                    self.model_type = 'star_net'
                elif detected_model_type == 'double_unet':
                    if not DOUBLE_UNET_AVAILABLE:
                        raise ImportError("DoubleUNet 不可用")
                    self.model = DoubleUNet(in_channels=3, num_classes=1, base_ch=32)
                    self.model_type = 'double_unet'
                elif detected_model_type == 'emcad':
                    if not EMCAD_AVAILABLE:
                        raise ImportError("EMCAD 不可用")
                    self.model = EMCAD(in_channels=3, num_classes=1, base_ch=32, use_attention_gates=True)
                    self.model_type = 'emcad'
                elif detected_model_type == 'nnwnet':
                    if not NNWNET_AVAILABLE:
                        raise ImportError("nnWNet 不可用")
                    self.model = nnWNet(in_channels=3, num_classes=1, base_ch=32)
                    self.model_type = 'nnwnet'
                elif detected_model_type == 'attention_unet':
                    if not ATTENTION_UNET_AVAILABLE:
                        raise ImportError("AttentionUNet 不可用")
                    self.model = AttentionUNet(in_channels=3, num_classes=1, base_ch=64)
                    self.model_type = 'attention_unet'
                elif detected_model_type == 'emsam':
                    if not EMSAM_AVAILABLE:
                        raise ImportError("EMSAM 不可用")
                    self.model = EMSAM(in_channels=3, num_classes=1, base_ch=64, img_size=352)
                    self.model_type = 'emsam'
                elif detected_model_type == 'swin_unet':
                    if not SWIN_UNET_AVAILABLE:
                        raise ImportError("SwinUNetAgri 不可用")
                    self.model = SwinUNetAgri(in_channels=3, num_classes=1, img_size=352)
                    self.model_type = 'swin_unet'
                elif detected_model_type == 'mamba_v2':
                    if not MAMBA_V2_AVAILABLE:
                        raise ImportError("MambaSAM2UNetV2 不可用，请安装 mamba_ssm")
                    checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                    use_mamba = any('mamba' in k for k in model_state_dict.keys() if not k.startswith('msca'))
                    use_skip  = any('gate' in k for k in model_state_dict.keys())
                    use_aux   = any('side1' in k for k in model_state_dict.keys())
                    self.model = MambaSAM2UNetV2(
                        checkpoint_path=checkpoint_path, num_classes=1,
                        use_mamba=use_mamba, use_skip_connections=use_skip,
                        use_auxiliary_loss=use_aux,
                    )
                    self.model_type = 'mamba_v2'
                elif detected_model_type == 'baseline_sam2unet':
                    if not BASELINE_SAM2UNET_AVAILABLE:
                        raise ImportError("BaselineSAM2UNet is not available. Please check if baseline_sam2unet module is installed.")

                    checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                    print(f"🔍 检测到checkpoint路径: {checkpoint_path}")

                    model_size = 'small'

                    if checkpoint_path:
                        if 'large' in checkpoint_path.lower():
                            model_size = 'large'
                        elif 'tiny' in checkpoint_path.lower():
                            model_size = 'tiny'
                        elif 'small' in checkpoint_path.lower():
                            model_size = 'small'

                    if 'rfb1.conv_res.conv.weight' in model_state_dict:
                        rfb1_in_channels = model_state_dict['rfb1.conv_res.conv.weight'].shape[1]
                        print(f"   - 检测到RFB1输入通道数: {rfb1_in_channels}")
                        if rfb1_in_channels == 144:
                            model_size = 'large'
                            print(f"   - 推断模型大小: Large (144/288/576/1152)")
                        elif rfb1_in_channels == 96:
                            model_size = 'small'
                            print(f"   - 推断模型大小: Small/Tiny (96/192/384/768)")

                    print(f"[BASELINE] 配置:")
                    print(f"   - checkpoint_path: {checkpoint_path if checkpoint_path else '无'}")
                    print(f"   - 使用 Large config (sam2_hiera_l.yaml) + Large channels (144/288/576/1152)")

                    self.model = BaselineSAM2UNet(
                        checkpoint_path=checkpoint_path
                    )
                    self.model_type = 'baseline_sam2unet'
                elif detected_model_type == 'latest_mamba':
                    if not LATEST_MAMBA_AVAILABLE:
                        raise ImportError("LatestMambaLightweightSAM2UNet is not available. Please check if sam2 module is installed.")

                    checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                    print(f"🔍 检测到checkpoint路径: {checkpoint_path}")

                    self.model = LatestMambaLightweightSAM2UNet(
                        checkpoint_path=checkpoint_path,
                        num_classes=1
                    )
                    self.model_type = 'latest_mamba'
                elif detected_model_type == 'unet':
                    self.model = UNet(n_channels=3, n_classes=1)
                    self.model_type = 'unet'
                elif detected_model_type == 'unet_plus_plus':
                    self.model = UNetPlusPlus(num_classes=1)
                    self.model_type = 'unet_plus_plus'
                elif detected_model_type == 'deeplabv3_plus':
                    self.model = DeepLabV3Plus(num_classes=1)
                    self.model_type = 'deeplabv3_plus'
                elif detected_model_type == 'pspnet':
                    self.model = PSPNet(num_classes=1)
                    self.model_type = 'pspnet'
                elif detected_model_type == 'fcn':
                    self.model = FCN(num_classes=1)
                    self.model_type = 'fcn'
                elif detected_model_type == 'segnet':
                    self.model = SegNet(num_classes=1)
                    self.model_type = 'segnet'
                elif detected_model_type == 'fpn':
                    self.model = FPN(num_classes=1)
                    self.model_type = 'fpn'
                elif detected_model_type == 'sam2unet' and 'SAM2UNET_AVAILABLE' in globals() and SAM2UNET_AVAILABLE:

                    checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                    print(f"🔍 检测到checkpoint路径: {checkpoint_path}")
                    self.model = SAM2UNet(checkpoint_path=checkpoint_path)
                    self.model_type = 'sam2unet'
                else:
                    raise ValueError(f"Unsupported detected model type: {detected_model_type}")

                try:
                    self.model.load_state_dict(model_state_dict, strict=True)
                    print("✅ 使用strict=True成功加载权重")
                except RuntimeError as e:
                    print(f"⚠️ strict=True失败，使用strict=False: {e}")
                    missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
                    if missing_keys:
                        print(f"⚠️ 缺少的键: {len(missing_keys)} 个")
                    if unexpected_keys:
                        print(f"⚠️ 意外的键: {len(unexpected_keys)} 个")
                    print("✅ 使用strict=False成功加载权重")

            except Exception as e:
                print(f"⚠️ 自动检测失败: {e}")
                raise ValueError(f"Failed to auto-detect model type: {e}")
        elif model_type == 'baseline_sam2unet':
            if not BASELINE_SAM2UNET_AVAILABLE:
                raise ImportError("BaselineSAM2UNet is not available. Please check if baseline_sam2unet module is installed.")

            print("🔍 加载 baseline_sam2unet 模型...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model' in checkpoint:
                    model_state_dict = checkpoint['model']
                else:
                    model_state_dict = checkpoint

                checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                print(f"🔍 检测到checkpoint路径: {checkpoint_path}")

                print(f"[BASELINE] 配置:")
                print(f"   - checkpoint_path: {checkpoint_path if checkpoint_path else '无'}")
                print(f"   - 使用 Large config (sam2_hiera_l.yaml) + Large channels (144/288/576/1152)")

                self.model = BaselineSAM2UNet(
                    checkpoint_path=checkpoint_path
                )

                try:
                    self.model.load_state_dict(model_state_dict, strict=True)
                    print("✅ 使用strict=True成功加载baseline_sam2unet权重")
                except RuntimeError as e:
                    print(f"⚠️ strict=True失败，使用strict=False: {e}")
                    missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
                    if missing_keys:
                        print(f"⚠️ 缺少的键: {len(missing_keys)} 个")
                    if unexpected_keys:
                        print(f"⚠️ 意外的键: {len(unexpected_keys)} 个")
                    print("✅ 使用strict=False成功加载baseline_sam2unet权重")
            except Exception as e:
                print(f"⚠️ 无法加载权重，使用默认配置: {e}")
                import traceback
                traceback.print_exc()
                self.model = BaselineSAM2UNet(checkpoint_path=None)
        elif model_type == 'mamba_v2':
            if not MAMBA_V2_AVAILABLE:
                raise ImportError("MambaSAM2UNetV2 不可用，请安装 mamba_ssm：pip install causal-conv1d mamba-ssm")
            print("🔍 加载 MambaSAM2UNetV2 模型...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                model_state_dict = checkpoint.get('model_state_dict', checkpoint)
                checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)

                use_mamba = any('mamba' in k for k in model_state_dict.keys()
                                if not k.startswith('msca'))
                use_skip  = any('gate' in k for k in model_state_dict.keys())
                use_aux   = any('side1' in k for k in model_state_dict.keys())
                print(f"  use_mamba={use_mamba}, skip={use_skip}, aux={use_aux}")
                self.model = MambaSAM2UNetV2(
                    checkpoint_path=checkpoint_path,
                    num_classes=1,
                    use_mamba=use_mamba,
                    use_skip_connections=use_skip,
                    use_auxiliary_loss=use_aux,
                )
            except Exception as e:
                print(f"⚠️ 无法推断配置，使用默认配置: {e}")
                self.model = MambaSAM2UNetV2(num_classes=1)
        elif model_type == 'mamba_v3':
            if not MAMBA_V3_AVAILABLE:
                raise ImportError("MambaSAM2UNetV3 不可用，请安装 mamba_ssm：pip install causal-conv1d mamba-ssm")
            print("🔍 加载 MambaSAM2UNetV3 模型（VMamba优化版）...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                model_state_dict = checkpoint.get('model_state_dict', checkpoint)
                checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                print(f"🔍 检测到checkpoint路径: {checkpoint_path}")
                self.model = MambaSAM2UNetV3(
                    checkpoint_path=checkpoint_path,
                    num_classes=1
                )
            except Exception as e:
                print(f"⚠️ 无法推断配置，使用默认配置: {e}")
                self.model = MambaSAM2UNetV3(num_classes=1)
        elif model_type == 'vm_unet':
            if not VM_UNET_AVAILABLE:
                raise ImportError("VMUNet 不可用，请安装 mamba_ssm：pip install causal-conv1d mamba-ssm")
            print("🔍 加载 VM-UNet 模型 (官方架构, depths=[2,2,9,2], d_state=8)...")
            self.model = VMUNet(
                in_channels=3,
                num_classes=1,
                depths=(2, 2, 9, 2),
                depths_decoder=(2, 9, 2, 2),
                d_state=8,
                drop_path_rate=0.2,
            )
        elif model_type == 'garden_net':
            if not GARDEN_NET_AVAILABLE:
                raise ImportError("GARDEN 不可用，请安装 mamba_ssm：pip install causal-conv1d mamba-ssm")
            print("🔍 加载 GARDEN-Net 模型...")
            self.model = GARDEN(in_channels=3, num_classes=1)
        elif model_type == 'star_net':
            if not STAR_NET_AVAILABLE:
                raise ImportError("STARNet 不可用")
            print("🔍 加载 STAR-Net 模型...")
            self.model = STARNet(in_channels=3, num_classes=1)
        elif model_type == 'double_unet':
            if not DOUBLE_UNET_AVAILABLE:
                raise ImportError("DoubleUNet 不可用")
            print("🔍 加载 DoubleU-Net 模型...")
            self.model = DoubleUNet(in_channels=3, num_classes=1, base_ch=32)
        elif model_type == 'emcad':
            if not EMCAD_AVAILABLE:
                raise ImportError("EMCAD 不可用")
            print("🔍 加载 EMCAD 模型...")
            self.model = EMCAD(in_channels=3, num_classes=1, base_ch=32, use_attention_gates=True)
        elif model_type == 'nnwnet':
            if not NNWNET_AVAILABLE:
                raise ImportError("nnWNet 不可用")
            print("🔍 加载 nnWNet 模型...")
            self.model = nnWNet(in_channels=3, num_classes=1, base_ch=32)
        elif model_type == 'attention_unet':
            if not ATTENTION_UNET_AVAILABLE:
                raise ImportError("AttentionUNet 不可用")
            print("🔍 加载 Attention U-Net 模型...")
            self.model = AttentionUNet(in_channels=3, num_classes=1, base_ch=64)
        elif model_type == 'emsam':
            if not EMSAM_AVAILABLE:
                raise ImportError("EMSAM 不可用")
            print("🔍 加载 EMSAM 模型...")
            self.model = EMSAM(in_channels=3, num_classes=1, base_ch=64, img_size=352)
        elif model_type == 'swin_unet':
            if not SWIN_UNET_AVAILABLE:
                raise ImportError("SwinUNetAgri 不可用")
            print("🔍 加载 Swin-UNet (Agri) 模型...")
            self.model = SwinUNetAgri(in_channels=3, num_classes=1, img_size=352)
        elif model_type == 'transunet':
            if not TRANSUNET_AVAILABLE:
                raise ImportError("TransUNetAgri 不可用")
            print("🔍 加载 TransUNet (Agri) 模型...")
            self.model = TransUNetAgri(in_channels=3, num_classes=1, base_ch=64,
                                       transformer_dim=256, transformer_depth=4, num_heads=8)
        elif model_type == 'lightm_unet':
            if not LIGHTM_UNET_AVAILABLE:
                raise ImportError("LightM-UNet 不可用，请安装 mamba_ssm：pip install causal-conv1d mamba-ssm")
            print("🔍 加载 LightM-UNet 模型...")
            self.model = LightMUNet(in_channels=3, num_classes=1, base_channels=32, d_state=16)

        elif model_type == 'latest_mamba':
            if not LATEST_MAMBA_AVAILABLE:
                raise ImportError("LatestMambaLightweightSAM2UNet is not available. Please check if sam2 module is installed.")

            print("🔍 检测latest_mamba模型的权重类型...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model' in checkpoint:
                    model_state_dict = checkpoint['model']
                else:
                    model_state_dict = checkpoint

                checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                print(f"🔍 检测到checkpoint路径: {checkpoint_path}")

                if 'baseline_no_mamba' in model_path:
                    print("🔍 检测到baseline_no_mamba模型，使用原始基线配置")

                    self.model = LatestMambaLightweightSAM2UNet(
                        checkpoint_path=checkpoint_path,
                        num_classes=1,
                        use_mamba=False,
                        use_skip_connections=False,
                        use_auxiliary_loss=False
                    )

                    try:
                        self.model.load_state_dict(model_state_dict, strict=True)
                        print("✅ 使用strict=True成功加载baseline_no_mamba权重")
                    except RuntimeError as e:
                        print(f"⚠️ strict=True失败，使用strict=False: {e}")
                        missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
                        if missing_keys:
                            print(f"⚠️ 缺少的键: {len(missing_keys)} 个")
                        if unexpected_keys:
                            print(f"⚠️ 意外的键: {len(unexpected_keys)} 个")
                        print("✅ 使用strict=False成功加载baseline_no_mamba权重")
                else:
                    self.model = LatestMambaLightweightSAM2UNet(
                        checkpoint_path=checkpoint_path,
                        num_classes=1
                    )
            except Exception as e:
                print(f"⚠️ 无法检测权重类型，使用默认Small权重: {e}")
                self.model = LatestMambaLightweightSAM2UNet(num_classes=1)
        elif model_type == 'sam2unet':
            if 'SAM2UNET_AVAILABLE' not in globals() or not SAM2UNET_AVAILABLE:
                raise ImportError("SAM2UNet is not available. Please ensure SAM2UNet.py is importable.")
            print("🔍 检测sam2unet模型的权重类型...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model' in checkpoint:
                    model_state_dict = checkpoint['model']
                else:
                    model_state_dict = checkpoint
                checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                print(f"🔍 检测到checkpoint路径: {checkpoint_path}")
                self.model = SAM2UNet(checkpoint_path=checkpoint_path)
            except Exception as e:
                print(f"⚠️ 无法检测权重类型，使用默认Small权重: {e}")
                self.model = SAM2UNet()
        elif model_type == 'unet':
            self.model = UNet(n_channels=3, n_classes=1)
        elif model_type == 'unet_plus_plus':
            self.model = UNetPlusPlus(num_classes=1)
        elif model_type == 'deeplabv3_plus':
            self.model = DeepLabV3Plus(num_classes=1)
        elif model_type == 'pspnet':
            self.model = PSPNet(num_classes=1)
        elif model_type == 'fcn':
            self.model = FCN(num_classes=1)
        elif model_type == 'segnet':
            self.model = SegNet(num_classes=1)
        elif model_type == 'fpn':
            self.model = FPN(num_classes=1)
        else:
            available_models = ['auto']
            if LATEST_MAMBA_AVAILABLE:
                available_models.append('latest_mamba')
            if MAMBA_V2_AVAILABLE:
                available_models.append('mamba_v2')
            if BASELINE_SAM2UNET_AVAILABLE:
                available_models.append('baseline_sam2unet')
            available_models.extend(['unet', 'unet_plus_plus', 'deeplabv3_plus', 'pspnet', 'fcn', 'segnet', 'fpn'])
            if 'SAM2UNET_AVAILABLE' in globals() and SAM2UNET_AVAILABLE:
                available_models.append('sam2unet')

            raise ValueError(f"Model type must be one of: {available_models}")

        try:

            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            print("✅ 使用weights_only=True成功加载checkpoint")
        except Exception as e:
            print(f"❌ weights_only=True加载失败: {e}")
            print("🔄 尝试添加安全全局变量...")
            try:

                import torch.serialization as torch_serialization

                torch_serialization.add_safe_globals([mamba_ablation_configs.MambaAblationConfig])
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                print("✅ 使用安全全局变量成功加载checkpoint")
            except Exception as e2:
                print(f"❌ 安全全局变量方案失败: {e2}")
                print("🔄 使用自定义unpickler...")
                try:

                    import pickle
                    import io

                    class CustomUnpickler(pickle.Unpickler):
                        def find_class(self, module, name):

                            if module == 'mamba_ablation_configs' and name == 'MambaAblationConfig':
                                return mamba_ablation_configs.MambaAblationConfig
                            return super().find_class(module, name)

                        def persistent_load(self, pid):

                            return None

                    with open(model_path, 'rb') as f:
                        checkpoint = CustomUnpickler(f).load()
                    print("✅ 使用自定义unpickler成功加载checkpoint")
                except Exception as e3:
                    print(f"❌ 自定义unpickler方案失败: {e3}")
                    print("🔄 使用weights_only=False作为最终方案...")
                    try:

                        class CustomUnpicklerFallback(pickle.Unpickler):
                            def find_class(self, module, name):

                                if module == 'mamba_ablation_configs' and name == 'MambaAblationConfig':
                                    return mamba_ablation_configs.MambaAblationConfig
                                return super().find_class(module, name)

                        with open(model_path, 'rb') as f:
                            checkpoint = CustomUnpicklerFallback(f).load()
                        print("✅ 使用自定义unpickler成功加载checkpoint")
                    except Exception as e4:
                        print(f"❌ 自定义unpickler最终方案失败: {e4}")
                        print("🔄 尝试直接使用torch.load...")
                        try:
                            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                            print("✅ 使用torch.load成功加载checkpoint")
                        except Exception as e5:
                            print(f"❌ 所有方案都失败: {e5}")
                            print("🔄 尝试使用PyTorch内置方法...")
                            try:

                                import torch.serialization as torch_ser

                                import sys
                                sys.modules['mamba_ablation_configs'] = mamba_ablation_configs

                                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                                print("✅ 使用PyTorch内置方法成功加载checkpoint")
                            except Exception as e6:
                                print(f"❌ PyTorch内置方法失败: {e6}")
                                print("🔄 尝试只加载模型权重...")

                                checkpoint = {'model_state_dict': {}}
                                print("⚠️ 无法加载完整checkpoint，将使用随机初始化的权重")

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        detected_model_type = self.detect_model_type(state_dict)
        print(f"🔍 模型类型检测结果:")
        print(f"   用户指定类型: {model_type}")
        print(f"   检测到的类型: {detected_model_type}")

        if detected_model_type != model_type:
            print(f"⚠️ 检测到模型类型不匹配！")
            print(f"   指定类型: {model_type}")
            print(f"   检测类型: {detected_model_type}")
            print(f"🔄 重新创建正确的模型...")

            if detected_model_type == 'mamba_v3':
                if not MAMBA_V3_AVAILABLE:
                    raise ImportError("MambaSAM2UNetV3 不可用，请安装 mamba_ssm")
                checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                print(f"🔍 检测到checkpoint路径: {checkpoint_path}")

                use_mamba = any('mamba' in k for k in model_state_dict.keys()
                                if not k.startswith('msca'))
                use_skip = any('gate' in k or 'skip_eca' in k for k in model_state_dict.keys())
                use_aux = any('side1' in k for k in model_state_dict.keys())
                use_msca = any('msca' in k for k in model_state_dict.keys())
                use_csfa = any('csfa' in k for k in model_state_dict.keys())
                use_mamba_adapter = any('mamba_branch' in k for k in model_state_dict.keys())

                d_state = 4
                for k, v in model_state_dict.items():
                    if 'mamba1' in k and 'A_log' in k:
                        d_state = v.shape[1] if len(v.shape) > 1 else 4
                        break

                print(f"[MAMBA_V3] 从权重推断的配置:")
                print(f"   - use_mamba: {use_mamba}")
                print(f"   - use_skip_connections: {use_skip}")
                print(f"   - use_auxiliary_loss: {use_aux}")
                print(f"   - use_msca: {use_msca}")
                print(f"   - use_csfa: {use_csfa}")
                print(f"   - use_mamba_adapter: {use_mamba_adapter}")
                print(f"   - d_state: {d_state}")

                self.model = MambaSAM2UNetV3(
                    checkpoint_path=checkpoint_path,
                    num_classes=1,
                    use_mamba=use_mamba,
                    use_skip_connections=use_skip,
                    use_auxiliary_loss=use_aux,
                    use_msca=use_msca,
                    use_csfa=use_csfa,
                    use_mamba_adapter=use_mamba_adapter,
                    d_state=d_state
                )
                self.model_type = 'mamba_v3'
            elif detected_model_type == 'vm_unet':
                if not VM_UNET_AVAILABLE:
                    raise ImportError("VMUNet 不可用，请安装 mamba_ssm")
                self.model = VMUNet(
                    in_channels=3, num_classes=1,
                    depths=(2, 2, 9, 2), depths_decoder=(2, 9, 2, 2),
                    d_state=8, drop_path_rate=0.2,
                )
                self.model_type = 'vm_unet'
            elif detected_model_type == 'garden_net':
                if not GARDEN_NET_AVAILABLE:
                    raise ImportError("GARDEN 不可用，请安装 mamba_ssm")
                self.model = GARDEN(in_channels=3, num_classes=1)
                self.model_type = 'garden_net'
            elif detected_model_type == 'star_net':
                if not STAR_NET_AVAILABLE:
                    raise ImportError("STARNet 不可用")
                self.model = STARNet(in_channels=3, num_classes=1)
                self.model_type = 'star_net'
            elif detected_model_type == 'double_unet':
                if not DOUBLE_UNET_AVAILABLE:
                    raise ImportError("DoubleUNet 不可用")
                self.model = DoubleUNet(in_channels=3, num_classes=1, base_ch=32)
                self.model_type = 'double_unet'
            elif detected_model_type == 'emcad':
                if not EMCAD_AVAILABLE:
                    raise ImportError("EMCAD 不可用")
                self.model = EMCAD(in_channels=3, num_classes=1, base_ch=32, use_attention_gates=True)
                self.model_type = 'emcad'
            elif detected_model_type == 'nnwnet':
                if not NNWNET_AVAILABLE:
                    raise ImportError("nnWNet 不可用")
                self.model = nnWNet(in_channels=3, num_classes=1, base_ch=32)
                self.model_type = 'nnwnet'
            elif detected_model_type == 'attention_unet':
                if not ATTENTION_UNET_AVAILABLE:
                    raise ImportError("AttentionUNet 不可用")
                self.model = AttentionUNet(in_channels=3, num_classes=1, base_ch=64)
                self.model_type = 'attention_unet'
            elif detected_model_type == 'emsam':
                if not EMSAM_AVAILABLE:
                    raise ImportError("EMSAM 不可用")
                self.model = EMSAM(in_channels=3, num_classes=1, base_ch=64, img_size=352)
                self.model_type = 'emsam'
            elif detected_model_type == 'swin_unet':
                if not SWIN_UNET_AVAILABLE:
                    raise ImportError("SwinUNetAgri 不可用")
                self.model = SwinUNetAgri(in_channels=3, num_classes=1, img_size=352)
                self.model_type = 'swin_unet'
            elif detected_model_type == 'transunet':
                if not TRANSUNET_AVAILABLE:
                    raise ImportError("TransUNetAgri 不可用")
                self.model = TransUNetAgri(in_channels=3, num_classes=1, base_ch=64,
                                          transformer_dim=256, transformer_depth=4, num_heads=8)
                self.model_type = 'transunet'
            elif detected_model_type == 'lightm_unet':
                if not LIGHTM_UNET_AVAILABLE:
                    raise ImportError("LightM-UNet 不可用")
                self.model = LightMUNet(in_channels=3, num_classes=1, base_channels=32, d_state=16)
                self.model_type = 'lightm_unet'
            elif detected_model_type == 'unet':
                self.model = UNet(n_channels=3, n_classes=1)
                self.model_type = 'unet'
            elif detected_model_type == 'unet_plus_plus':
                self.model = UNetPlusPlus(num_classes=1)
                self.model_type = 'unet_plus_plus'
            elif detected_model_type == 'baseline_sam2unet':
                if not BASELINE_SAM2UNET_AVAILABLE:
                    raise ImportError("BaselineSAM2UNet is not available. Please check if baseline_sam2unet module is installed.")

                checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                print(f"🔍 检测到checkpoint路径: {checkpoint_path}")

                model_size = 'small'

                if checkpoint_path:
                    if 'large' in checkpoint_path.lower():
                        model_size = 'large'
                    elif 'tiny' in checkpoint_path.lower():
                        model_size = 'tiny'
                    elif 'small' in checkpoint_path.lower():
                        model_size = 'small'

                if 'rfb1.conv_res.conv.weight' in model_state_dict:
                    rfb1_in_channels = model_state_dict['rfb1.conv_res.conv.weight'].shape[1]
                    if rfb1_in_channels == 144:
                        model_size = 'large'
                    elif rfb1_in_channels == 96:
                        model_size = 'small'

                print(f"[BASELINE] 配置: 使用 Large config (sam2_hiera_l.yaml) + Large channels")

                self.model = BaselineSAM2UNet(
                    checkpoint_path=checkpoint_path
                )
                self.model_type = 'baseline_sam2unet'
            elif detected_model_type == 'latest_mamba':
                if not LATEST_MAMBA_AVAILABLE:
                    raise ImportError("LatestMambaLightweightSAM2UNet is not available. Please check if sam2 module is installed.")

                checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                print(f"🔍 检测到checkpoint路径: {checkpoint_path}")

                self.model = LatestMambaLightweightSAM2UNet(
                    checkpoint_path=checkpoint_path,
                    num_classes=1
                )
                self.model_type = 'latest_mamba'
            elif detected_model_type == 'mamba_v2':
                if not MAMBA_V2_AVAILABLE:
                    raise ImportError("MambaSAM2UNetV2 不可用，请安装 mamba_ssm")

                checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                print(f"🔍 检测到checkpoint路径: {checkpoint_path}")

                use_mamba = any('mamba' in k for k in model_state_dict.keys()
                                if not k.startswith('msca'))
                use_skip  = any('gate' in k or 'skip_eca' in k for k in model_state_dict.keys())
                use_aux   = any('side1' in k for k in model_state_dict.keys())

                print(f"[MAMBA_V2] 从权重推断的配置:")
                print(f"   - use_mamba: {use_mamba}")
                print(f"   - use_skip_connections: {use_skip}")
                print(f"   - use_auxiliary_loss: {use_aux}")

                self.model = MambaSAM2UNetV2(
                    checkpoint_path=checkpoint_path,
                    num_classes=1,
                    use_mamba=use_mamba,
                    use_skip_connections=use_skip,
                    use_auxiliary_loss=use_aux,
                )
                self.model_type = 'mamba_v2'
            elif detected_model_type == 'deeplabv3_plus':
                self.model = DeepLabV3Plus(num_classes=1)
                self.model_type = 'deeplabv3_plus'
            elif detected_model_type == 'pspnet':
                self.model = PSPNet(num_classes=1)
                self.model_type = 'pspnet'
            elif detected_model_type == 'fcn':
                self.model = FCN(num_classes=1)
                self.model_type = 'fcn'
            elif detected_model_type == 'segnet':
                self.model = SegNet(num_classes=1)
                self.model_type = 'segnet'
            elif detected_model_type == 'fpn':
                self.model = FPN(num_classes=1)
                self.model_type = 'fpn'
            elif detected_model_type == 'sam2unet' and 'SAM2UNET_AVAILABLE' in globals() and SAM2UNET_AVAILABLE:
                checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                print(f"🔍 检测到checkpoint路径: {checkpoint_path}")
                self.model = SAM2UNet(checkpoint_path=checkpoint_path)
                self.model_type = 'sam2unet'
            else:
                print(f"⚠️ 无法识别的模型类型: {detected_model_type}")

        model_state_dict = {}
        for key, value in state_dict.items():
            new_key = key

            if key.startswith('m.'):
                new_key = key.replace('m.', '', 1)
                print(f"移除m.前缀: {key} -> {new_key}")

            if key.startswith('model.'):
                new_key = key.replace('model.', '')
                print(f"移除model.前缀: {key} -> {new_key}")

            if key.startswith('base_model.'):
                new_key = key.replace('base_model.', '')
                print(f"移除base_model.前缀: {key} -> {new_key}")

            model_state_dict[new_key] = value

        try:
            self.model.load_state_dict(model_state_dict)
            print("✅ 模型权重加载成功")
        except RuntimeError as e:
            print(f"❌ 模型权重加载失败: {e}")
            print("🔄 尝试使用strict=False加载...")
            try:
                missing_keys, unexpected_keys = self.model.load_state_dict(model_state_dict, strict=False)
                if missing_keys:
                    print(f"⚠️ 缺少的键: {len(missing_keys)} 个")
                if unexpected_keys:
                    print(f"⚠️ 意外的键: {len(unexpected_keys)} 个")
                print("✅ 使用strict=False成功加载模型权重")
            except RuntimeError as e2:
                print(f"❌ strict=False也失败: {e2}")
                print("🔄 尝试重新创建匹配的模型架构...")

                checkpoint_path = self._detect_checkpoint_path_from_state_dict(model_state_dict)
                print(f"🔍 重新检测checkpoint路径: {checkpoint_path}")

                if self.model_type == 'baseline_sam2unet' and BASELINE_SAM2UNET_AVAILABLE:
                    print("🔍 重新创建baseline_sam2unet模型")

                    model_size = 'small'

                    if checkpoint_path:
                        if 'large' in checkpoint_path.lower():
                            model_size = 'large'
                        elif 'tiny' in checkpoint_path.lower():
                            model_size = 'tiny'
                        elif 'small' in checkpoint_path.lower():
                            model_size = 'small'

                    if 'rfb1.conv_res.conv.weight' in model_state_dict:
                        rfb1_in_channels = model_state_dict['rfb1.conv_res.conv.weight'].shape[1]
                        if rfb1_in_channels == 144:
                            model_size = 'large'
                        elif rfb1_in_channels == 96:
                            model_size = 'small'

                    print(f"[BASELINE] 重新创建配置: Large config + Large channels")

                    self.model = BaselineSAM2UNet(
                        checkpoint_path=checkpoint_path
                    )
                    print("✅ 重新创建baseline_sam2unet模型成功")

                    try:
                        self.model.load_state_dict(model_state_dict, strict=False)
                        print("✅ 重新创建baseline_sam2unet模型后权重加载成功")
                    except RuntimeError as e3:
                        print(f"❌ 重新创建baseline_sam2unet模型后仍然失败: {e3}")
                        print("⚠️ 跳过权重加载，使用随机初始化的模型")
                elif self.model_type == 'latest_mamba' and LATEST_MAMBA_AVAILABLE:

                    if 'baseline_no_mamba' in model_path:
                        print("🔍 重新创建baseline_no_mamba模型，使用原始基线配置")
                        self.model = LatestMambaLightweightSAM2UNet(
                            checkpoint_path=checkpoint_path,
                            num_classes=1,
                            use_mamba=False,
                            use_skip_connections=False,
                            use_auxiliary_loss=False
                        )
                    else:
                        self.model = LatestMambaLightweightSAM2UNet(
                            checkpoint_path=checkpoint_path,
                            num_classes=1
                        )
                    print("✅ 重新创建latest_mamba模型成功")

                    try:
                        self.model.load_state_dict(model_state_dict, strict=False)
                        print("✅ 重新创建latest_mamba模型后权重加载成功")
                    except RuntimeError as e3:
                        print(f"❌ 重新创建latest_mamba模型后仍然失败: {e3}")
                        print("⚠️ 跳过权重加载，使用随机初始化的模型")
                else:
                    print(f"❌ {self.model_type}模型不可用")
                    raise e2

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded from {model_path}")
        print(f"Model type: {self.model_type}")
        print(f"Device: {self.device}")

    def _get_sam2_checkpoint_path(self, weight_type="small"):

        import os

        weight_files = {
            "tiny": "sam2_hiera_tiny.pt",
            "small": "sam2_hiera_small.pt",
            "large": "sam2_hiera_large.pt"
        }

        filename = weight_files.get(weight_type, "sam2_hiera_small.pt")

        possible_paths = [
            f"./pt/{filename}",
            f"../pt/{filename}",
            f"/root/SAM2UNet/pt/{filename}",
            f"/root/SAM2-UNet/pt/{filename}"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return f"./pt/{filename}"

    def _detect_checkpoint_path_from_state_dict(self, state_dict):

        keys = list(state_dict.keys())

        encoder_keys = [k for k in keys if k.startswith('encoder.blocks.')]

        if not encoder_keys:
            print("⚠️ 未找到encoder.blocks键，使用默认Small权重")
            return self._get_sam2_checkpoint_path("small")

        for block_num in [10, 11]:
            qkv_key = f"encoder.blocks.{block_num}.block.attn.qkv.weight"
            if qkv_key in state_dict:
                qkv_shape = state_dict[qkv_key].shape
                print(f"🔍 检测到 {qkv_key} 形状: {qkv_shape}")

                if qkv_shape[0] == 2304:

                    pos_embed_key = "encoder.pos_embed"
                    if pos_embed_key in state_dict:
                        pos_embed_shape = state_dict[pos_embed_key].shape
                        print(f"🔍 检测到 {pos_embed_key} 形状: {pos_embed_shape}")

                        if pos_embed_shape[1] == 96:

                            max_block = max([int(k.split('.')[2]) for k in encoder_keys if 'encoder.blocks.' in k])
                            if max_block >= 15:
                                print("✅ 检测到Small权重（pos_embed=96, 16层）")
                                return self._get_sam2_checkpoint_path("small")
                            else:
                                print("✅ 检测到Tiny权重（pos_embed=96, 12层）")
                                return self._get_sam2_checkpoint_path("tiny")
                        elif pos_embed_shape[1] == 144:
                            print("✅ 检测到Large权重（pos_embed=144）")
                            return self._get_sam2_checkpoint_path("large")
                        else:
                            print("✅ 检测到Large权重（基于qkv形状）")
                            return self._get_sam2_checkpoint_path("large")
                    else:
                        print("✅ 检测到Large权重（基于qkv形状）")
                        return self._get_sam2_checkpoint_path("large")
                elif qkv_shape[0] == 1728:

                    pos_embed_key = "encoder.pos_embed"
                    if pos_embed_key in state_dict:
                        pos_embed_shape = state_dict[pos_embed_key].shape
                        print(f"🔍 检测到 {pos_embed_key} 形状: {pos_embed_shape}")

                        if pos_embed_shape[1] == 96:

                            max_block = max([int(k.split('.')[2]) for k in encoder_keys if 'encoder.blocks.' in k])
                            if max_block >= 15:
                                print("✅ 检测到Small权重（pos_embed=96, 16层）")
                                return self._get_sam2_checkpoint_path("small")
                            else:
                                print("✅ 检测到Tiny权重（pos_embed=96, 12层）")
                                return self._get_sam2_checkpoint_path("tiny")
                        elif pos_embed_shape[1] == 144:
                            print("✅ 检测到Large权重（pos_embed=144）")
                            return self._get_sam2_checkpoint_path("large")
                        else:
                            print("✅ 检测到Large权重（基于qkv形状1728）")
                            return self._get_sam2_checkpoint_path("large")
                    else:
                        print("✅ 检测到Large权重（基于qkv形状1728）")
                        return self._get_sam2_checkpoint_path("large")
                elif qkv_shape[0] == 1152:

                    max_block = max([int(k.split('.')[2]) for k in encoder_keys if 'encoder.blocks.' in k])
                    if max_block >= 15:
                        print("✅ 检测到Small权重（16层）")
                        return self._get_sam2_checkpoint_path("small")
                    else:
                        print("✅ 检测到Tiny权重（12层）")
                        return self._get_sam2_checkpoint_path("tiny")

        print("⚠️ 无法确定权重类型，使用默认Small权重")
        return self._get_sam2_checkpoint_path("small")

    def detect_model_type(self, state_dict):

        keys = list(state_dict.keys())
        print(f"🔍 检测模型类型，键名示例: {keys[:10]}...")

        def _has(patterns):
            return any(any(p in k for p in patterns) for k in keys)

        clean_keys = []
        for key in keys:
            clean_key = key

            if clean_key.startswith('m.'):
                clean_key = clean_key[2:]

            elif clean_key.startswith('model.'):
                clean_key = clean_key[6:]

            elif clean_key.startswith('base_model.'):
                clean_key = clean_key[11:]
            clean_keys.append(clean_key)

        def _has_clean(patterns):
            return any(any(p in k for p in patterns) for k in clean_keys)

        if _has(['vmunet.layers.', 'vmunet.patch_embed.', 'vmunet.layers_up.']):
            print("✅ 检测到 VM-UNet 特征 (官方架构)")
            return 'vm_unet'

        if _has(['enc1.', 'enc2.', 'enc3.', 'enc4.']) and\
           (_has(['ss2d.mamba_h_fwd', 'ss2d.mamba_v_fwd']) or _has(['ss2d.mamba.'])):
            print("✅ 检测到 VM-UNet 特征 (旧架构)")
            return 'vm_unet'

        psp_indicators = []
        has_psp_module = False
        has_psp_pools = False

        for key in clean_keys:
            if 'psp.' in key or key.startswith('psp.'):
                psp_indicators.append(key)
                has_psp_module = True

                if 'pool' in key or 'conv' in key:
                    has_psp_pools = True

        if has_psp_module and has_psp_pools:
            print(f"✅ 检测到PSPNet特征 (psp模块={has_psp_module}, 金字塔池化={has_psp_pools}): {psp_indicators[:5]}...")
            return 'pspnet'

        fpn_indicators = []
        has_lateral = False
        has_fpn = False

        for key in clean_keys:
            if 'lateral' in key:
                fpn_indicators.append(key)
                has_lateral = True
            if 'fpn' in key:
                fpn_indicators.append(key)
                has_fpn = True

        if has_lateral or has_fpn:
            print(f"✅ 检测到FPN特征 (lateral={has_lateral}, fpn={has_fpn}): {fpn_indicators[:5]}...")
            return 'fpn'

        deeplab_indicators = []
        has_aspp = False
        has_decoder = False
        has_backbone = False

        for key in clean_keys:
            if 'aspp.' in key or key.startswith('aspp.'):
                deeplab_indicators.append(key)
                has_aspp = True
            if 'decoder.' in key or key.startswith('decoder.'):
                deeplab_indicators.append(key)
                has_decoder = True
            if 'backbone.' in key or key.startswith('backbone.'):
                deeplab_indicators.append(key)
                has_backbone = True

        if has_aspp and has_decoder:
            print(f"✅ 检测到DeepLabV3+特征 (aspp={has_aspp}, decoder={has_decoder}, backbone={has_backbone}): {deeplab_indicators[:5]}...")
            return 'deeplabv3_plus'

        unet_indicators = []
        has_inc = False
        has_down = False
        has_up = False
        has_outc = False

        for key in clean_keys:
            if key.startswith('inc.'):
                unet_indicators.append(key)
                has_inc = True
            if key.startswith('down'):
                unet_indicators.append(key)
                has_down = True
            if key.startswith('up') and not 'upscore' in key:
                unet_indicators.append(key)
                has_up = True
            if key.startswith('outc.'):
                unet_indicators.append(key)
                has_outc = True

        has_fcn_features = any('score_pool' in k or 'upscore' in k for k in clean_keys)
        if (has_inc or has_outc) and not has_fcn_features:
            print(f"✅ 检测到UNet特征 (inc={has_inc}, down={has_down}, up={has_up}, outc={has_outc}): {unet_indicators[:5]}...")
            return 'unet'

        fcn_indicators = []
        has_classifier = False
        has_score_pool = False
        has_upscore = False

        for key in clean_keys:
            if key.startswith('classifier.'):
                fcn_indicators.append(key)
                has_classifier = True
            if 'score_pool' in key:
                fcn_indicators.append(key)
                has_score_pool = True
            if 'upscore' in key:
                fcn_indicators.append(key)
                has_upscore = True

        if has_score_pool and has_upscore:
            print(f"✅ 检测到FCN特征 (classifier={has_classifier}, score_pool={has_score_pool}, upscore={has_upscore}): {fcn_indicators[:5]}...")
            return 'fcn'

        if _has(['encoder.pos_embed', 'msca1.', 'msca2.']) and (_has(['up4.']) or _has(['side3.'])):
            print("✅ 检测到 MambaSAM2UNetV3 特征（4层解码器）")
            return 'mamba_v3'

        if _has(['register_tokens', 'importance_net']) or _has(['mamba_deform']):
            print("✅ 检测到 MambaSAM2UNetV3 (VMamba优化版) 特征")
            return 'mamba_v3'

        if _has(['encoder.pos_embed', 'encoder.patch_embed', 'msca1.', 'msca2.', 'msca3.', 'msca4.']):
            print("✅ 检测到 MambaSAM2UNetV3 baseline/消融配置特征")
            return 'mamba_v3'

        if _has(['lfem.', 'gfem.', 'ffm.']):
            print("✅ 检测到 EMSAM 特征")
            return 'emsam'

        if _has(['enc_stage1.', 'enc_stage2.', 'dec_stage1.', 'dec_stage2.']) or\
           _has(['bottleneck.blocks.', 'merge1.', 'expand1.']):
            print("✅ 检测到 Swin-UNet 特征")
            return 'swin_unet'

        if _has(['cnn_encoder.stage1.', 'cnn_encoder.stage2.', 'transformer.proj.', 'transformer.blocks.']) or\
           (_has(['trans_align.', 'dec1.', 'dec2.', 'dec3.']) and _has(['transformer.'])):
            print("✅ 检测到 TransUNet 特征")
            return 'transunet'

        if (_has(['stem_rvm.', 'enc1.rvm.', 'enc2.rvm.', 'enc3.rvm.']) or
            _has(['bottleneck.1.mamba', 'dec1.rvm.', 'dec2.rvm.', 'dec3.rvm.'])):
            print("✅ 检测到 LightM-UNet 特征")
            return 'lightm_unet'

        if _has(['bass.mamba_h_fwd', 'bass.mamba_v_fwd']) or _has(['msca.branches', 'up1.bass', 'up2.bass']):
            print("✅ 检测到 GARDEN-Net 特征")
            return 'garden_net'

        if _has(['hbaa.b1.', 'hbaa.b2.', 'hbaa.b3.']) or\
           (_has(['encoder.stage1.', 'encoder.stage4.']) and _has(['shallow_align.', 'deep_up2a.'])):
            print("✅ 检测到 STAR-Net 特征")
            return 'star_net'

        if _has(['enc1.', 'enc2.']) and _has(['aspp1.', 'aspp2.']):
            print("✅ 检测到 DoubleU-Net 特征")
            return 'double_unet'

        if _has(['dec1.', 'dec2.', 'dec3.', 'dec4.']) and _has(['encoder.']):
            print("✅ 检测到 EMCAD 特征")
            return 'emcad'

        if _has(['encoder1.', 'encoder2.', 'encoder3.', 'encoder4.']) and _has(['bottleneck.']):
            print("✅ 检测到 nnWNet 特征")
            return 'nnwnet'

        if _has(['ag1.', 'ag2.', 'ag3.', 'ag4.']):
            print("✅ 检测到 Attention U-Net 特征")
            return 'attention_unet'

        if _has(['mamba1.', 'mamba2.', 'mamba3.', 'mamba4.']):
            print("✅ 检测到 LatestMamba (BidirectionalMambaBlock) 特征")
            return 'latest_mamba'

        if _has(['rfb1.', 'rfb2.', 'rfb3.', 'rfb4.']):
            print("✅ 检测到 BaselineSAM2UNet (rfb) 特征")
            return 'baseline_sam2unet'

        if _has(['side1', 'side2', 'up1.', 'up2.', 'up3.']):
            if _has(['encoder.']):
                print("✅ 检测到 SAM2UNet 特征")
                return 'sam2unet'

            print(f"✅ 检测到FCN特征: {fcn_indicators[:5]}...")
            return 'fcn'

        unet_pp_indicators = []
        for key in clean_keys:

            if (key.startswith('conv0_') or key.startswith('conv1_') or
                key.startswith('conv2_') or key.startswith('conv3_') or
                key.startswith('conv4_')):
                unet_pp_indicators.append(key)

            elif (key.startswith('side') or key.startswith('X_') or
                  'side1' in key or 'side2' in key or 'side3' in key or 'side4' in key):
                unet_pp_indicators.append(key)

        if unet_pp_indicators:
            print(f"✅ 检测到UNet++特征: {unet_pp_indicators[:5]}...")
            return 'unet_plus_plus'

        attention_unet_indicators = []
        has_attention_gate = False
        has_bottleneck = False

        for key in clean_keys:

            if '.ag.' in key:
                attention_unet_indicators.append(key)
                has_attention_gate = True

            elif key.startswith('bottleneck.'):
                attention_unet_indicators.append(key)
                has_bottleneck = True

        if has_attention_gate or has_bottleneck:
            print(f"✅ 检测到Attention U-Net特征 (AG={has_attention_gate}, Bottleneck={has_bottleneck}): {attention_unet_indicators[:5]}...")
            return 'attention_unet'

        segnet_indicators = []
        for key in clean_keys:

            if ('enc' in key or 'dec' in key) and '_' in key:

                parts = key.split('.')
                if parts and ('enc' in parts[0] or 'dec' in parts[0]) and '_' in parts[0]:
                    segnet_indicators.append(key)

        if segnet_indicators:
            print(f"✅ 检测到SegNet特征: {segnet_indicators[:5]}...")
            return 'segnet'

        print("⚠️ 无法自动检测模型类型，默认使用UNet")
        return 'unet'

    def measure_inference_speed(self, input_tensor, num_runs=100):

        self.model.eval()

        with torch.no_grad():
            for _ in range(10):
                _ = self.model(input_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(input_tensor)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        total_time = end_time - start_time
        fps = num_runs / total_time

        return fps, total_time / num_runs

    def calculate_metrics(self, pred, target, threshold=0.5):

        metrics = ComprehensiveMetrics.calculate_metrics(pred, target, threshold)

        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = target.float()

        if pred_binary.shape != target_binary.shape:
            pred_binary = F.interpolate(pred_binary, size=target_binary.shape[2:],
                                      mode='bilinear', align_corners=False)

        hausdorff = self.calculate_hausdorff_distance(pred_binary, target_binary)

        metrics['hausdorff'] = hausdorff

        return metrics

    def calculate_hausdorff_distance(self, pred, target, max_dist=100):

        try:

            pred_np = pred.squeeze().cpu().numpy()
            target_np = target.squeeze().cpu().numpy()

            pred_contours, _ = cv2.findContours(
                (pred_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            target_contours, _ = cv2.findContours(
                (target_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if len(pred_contours) == 0 or len(target_contours) == 0:
                return max_dist

            pred_points = np.vstack(pred_contours).squeeze()
            target_points = np.vstack(target_contours).squeeze()

            if len(pred_points.shape) == 1:
                pred_points = pred_points.reshape(1, -1)
            if len(target_points.shape) == 1:
                target_points = target_points.reshape(1, -1)

            dist1 = np.min(np.linalg.norm(pred_points[:, None] - target_points, axis=2), axis=1)
            dist2 = np.min(np.linalg.norm(target_points[:, None] - pred_points, axis=2), axis=1)

            hausdorff = max(np.max(dist1), np.max(dist2))
            return min(hausdorff, max_dist)

        except:
            return max_dist

    def evaluate_dataset(self, test_dataset, save_results=True, output_dir='./evaluation_results', batch_size=None):

        if save_results:
            os.makedirs(output_dir, exist_ok=True)

        all_metrics = []

        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        total_pixels = 0

        print(f"Evaluating on {test_dataset.size} test samples...")

        print("\nCalculating model complexity (Params + GFLOPs)...")
        try:
            complexity = calculate_model_complexity(self.model, input_size=(1, 3, 352, 352))
            total_params    = complexity['total_params']
            trainable_params = complexity['trainable_params']
            gflops          = complexity['gflops']
            params_mb       = complexity['params_mb']
            print(f"  Total params:     {total_params:,}")
            print(f"  Trainable params: {trainable_params:,}")
            print(f"  Model size:       {params_mb:.2f} MB")
            print(f"  GFLOPs:           {gflops:.4f}")
        except Exception as e:
            print(f"  Warning: Could not calculate complexity: {e}")
            total_params = trainable_params = 0
            gflops = params_mb = 0.0

        print("Measuring inference speed...")
        try:
            sample_image, _, _ = test_dataset.load_data()

            if sample_image.dim() == 3:
                sample_tensor = sample_image.unsqueeze(0).to(self.device)
            else:
                sample_tensor = sample_image.to(self.device)
            fps, avg_time = self.measure_inference_speed(sample_tensor)
            print(f"Inference speed: {fps:.2f} FPS, Average time: {avg_time*1000:.2f} ms")
        except Exception as e:
            print(f"Warning: Could not measure inference speed: {e}")
            fps, avg_time = 0.0, 0.0

        with torch.no_grad():
            for i in tqdm(range(test_dataset.size)):

                if test_dataset.index >= test_dataset.size:
                    break
                image, target, name = test_dataset.load_data()
                image = image.to(self.device)

                target = torch.from_numpy(target).float() / 255.0
                target = target.unsqueeze(0).unsqueeze(0).to(self.device)

                if self.model_type in ['latest_mamba', 'baseline_sam2unet', 'sam2unet',
                                       'mamba_v2', 'mamba_v3', 'vm_unet', 'garden_net', 'star_net',
                                       'double_unet', 'emcad', 'nnwnet', 'attention_unet', 'emsam', 'swin_unet', 'transunet']:
                    model_output = self.model(image)
                    if isinstance(model_output, tuple) and len(model_output) >= 2:

                        pred = model_output[0]
                        pred1, pred2 = model_output[1] if len(model_output) > 1 else None,\
                                       model_output[2] if len(model_output) > 2 else None
                    else:

                        pred = model_output
                        pred1, pred2 = None, None
                else:

                    pred = self.model(image)
                    pred1, pred2 = None, None

                metrics = self.calculate_metrics(pred, target)
                metrics['image_name'] = name
                metrics['fps'] = fps
                metrics['avg_inference_time'] = avg_time * 1000

                pred_binary = (torch.sigmoid(pred) > 0.5).float()
                target_binary = target.float()

                if pred_binary.shape != target_binary.shape:
                    pred_binary = F.interpolate(pred_binary, size=target_binary.shape[2:],
                                              mode='bilinear', align_corners=False)

                intersection = (pred_binary * target_binary).sum()
                union = pred_binary.sum() + target_binary.sum() - intersection
                iou = intersection / (union + 1e-8)

                metrics['iou'] = iou.item()
                metrics['miou'] = iou.item()

                all_metrics.append(metrics)

                pred_binary = (torch.sigmoid(pred) > 0.5).float()

                if pred_binary.shape != target.shape:
                    pred_binary = F.interpolate(pred_binary, size=target.shape[2:],
                                              mode='bilinear', align_corners=False)

                tp = (pred_binary * target).sum().item()
                fp = (pred_binary * (1 - target)).sum().item()
                tn = ((1 - pred_binary) * (1 - target)).sum().item()
                fn = ((1 - pred_binary) * target).sum().item()

                total_tp += tp
                total_fp += fp
                total_tn += tn
                total_fn += fn
                total_pixels += tp + fp + tn + fn

                if i % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

        overall_metrics = {
            'precision': total_tp / (total_tp + total_fp + 1e-8),
            'recall': total_tp / (total_tp + total_fn + 1e-8),
            'f1': 2 * total_tp / (2 * total_tp + total_fp + total_fn + 1e-8),
            'accuracy': (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-8),
            'fps': fps,
            'avg_inference_time': avg_time * 1000
        }

        df_metrics = pd.DataFrame(all_metrics)

        mean_metrics = df_metrics.drop('image_name', axis=1).mean()
        std_metrics = df_metrics.drop('image_name', axis=1).std()

        mean_iou = mean_metrics['iou']
        std_iou = std_metrics['iou']

        tp_foreground = total_tp
        fp_foreground = total_fp
        fn_foreground = total_fn

        tp_background = total_tn
        fp_background = total_fn
        fn_background = total_fp

        iou_foreground = tp_foreground / (tp_foreground + fp_foreground + fn_foreground + 1e-8)
        iou_background = tp_background / (tp_background + fp_background + fn_background + 1e-8)

        recall_foreground = tp_foreground / (tp_foreground + fn_foreground + 1e-8)
        recall_background = tp_background / (tp_background + fn_background + 1e-8)

        miou = (iou_foreground + iou_background) / 2
        macc = (recall_foreground + recall_background) / 2

        overall_metrics['mean_iou'] = mean_iou
        overall_metrics['miou'] = miou
        overall_metrics['macc'] = macc
        overall_metrics['iou_foreground'] = iou_foreground
        overall_metrics['iou_background'] = iou_background
        overall_metrics['recall_foreground'] = recall_foreground
        overall_metrics['recall_background'] = recall_background

        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Metrics:")
        for metric, value in overall_metrics.items():
            if metric == 'mean_iou':
                print(f"  MEAN IOU (Per-Image Average): {value:.4f} ± {std_iou:.4f}")
            elif metric == 'miou':
                print(f"  MIOU (Class-wise Average): {value:.4f}")
            elif metric == 'macc':
                print(f"  MACC (Class-wise Average): {value:.4f}")
            elif metric == 'fps':
                print(f"  FPS: {value:.2f}")
            elif metric == 'avg_inference_time':
                print(f"  AVG INFERENCE TIME: {value:.2f} ms")
            elif metric == 'iou_foreground':
                print(f"  IOU FOREGROUND: {value:.4f}")
            elif metric == 'iou_background':
                print(f"  IOU BACKGROUND: {value:.4f}")
            elif metric == 'recall_foreground':
                print(f"  RECALL FOREGROUND: {value:.4f}")
            elif metric == 'recall_background':
                print(f"  RECALL BACKGROUND: {value:.4f}")
            else:
                print(f"  {metric.upper()}: {value:.4f}")

        print(f"\nPer-Image Metrics (Mean ± Std):")
        for metric in ['precision', 'recall', 'f1', 'iou', 'dice', 'specificity', 'accuracy']:
            mean_val = mean_metrics[metric]
            std_val = std_metrics[metric]
            if metric == 'iou':
                print(f"  IOU: {mean_val:.4f} ± {std_val:.4f}")
            else:
                print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")

        print(f"\n" + "="*60)
        print(f"FINAL EVALUATION RESULTS")
        print(f"="*60)
        print(f"  [Global MIoU]  (TP+TN累积后类别平均，论文标准指标)")
        print(f"    MIoU:              {miou:.4f}")
        print(f"    mAcc:              {macc:.4f}")
        print(f"    IoU (Foreground):  {iou_foreground:.4f}")
        print(f"    IoU (Background):  {iou_background:.4f}")
        print(f"  [Per-Image Average] (每张图单独算后平均)")
        print(f"    Mean IoU:          {mean_iou:.4f} ± {std_iou:.4f}")
        print(f"  [Per-Image Metrics] (Mean ± Std)")
        for metric in ['precision', 'recall', 'f1', 'dice', 'specificity', 'accuracy']:
            print(f"    {metric.upper():12s}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")
        print(f"  [Model Complexity]")
        print(f"    Total params:      {total_params:,}")
        print(f"    Trainable params:  {trainable_params:,}")
        print(f"    Model size:        {params_mb:.2f} MB")
        print(f"    GFLOPs:            {gflops:.4f}")
        print(f"  [Speed]")
        print(f"    FPS:               {overall_metrics.get('fps', 0):.2f}")
        print(f"    Avg Inference:     {overall_metrics.get('avg_inference_time', 0):.2f} ms")
        print(f"="*60)

        if save_results:

            df_metrics.to_csv(os.path.join(output_dir, 'detailed_metrics.csv'), index=False)

            self.save_top20_results_comprehensive(df_metrics, output_dir)

            summary = {
                'overall_metrics': overall_metrics,
                'mean_metrics': mean_metrics.to_dict(),
                'std_metrics': std_metrics.to_dict(),
                'miou_calculation': {
                    'formula': 'MIoU = (1/N) * Σ[TP_i / (TP_i + FP_i + FN_i)]',
                    'macc_formula': 'mAcc = (1/N) * Σ[TP_i / (TP_i + FN_i)]',
                    'classes': {
                        'foreground': {
                            'iou': iou_foreground,
                            'recall': recall_foreground,
                            'tp': tp_foreground,
                            'fp': fp_foreground,
                            'fn': fn_foreground
                        },
                        'background': {
                            'iou': iou_background,
                            'recall': recall_background,
                            'tp': tp_background,
                            'fp': fp_background,
                            'fn': fn_background
                        }
                    },
                    'final_metrics': {
                        'miou': miou,
                        'macc': macc,
                        'mean_iou_per_image': mean_iou,
                        'std_iou_per_image': std_iou
                    }
                },
                'model_type': self.model_type,
                'model_complexity': {
                    'total_params':     total_params,
                    'trainable_params': trainable_params,
                    'model_size_mb':    round(params_mb, 2),
                    'gflops':           round(gflops, 4),
                },
                'evaluation_date': datetime.now().isoformat()
            }

            with open(os.path.join(output_dir, 'evaluation_summary.json'), 'w') as f:
                json.dump(summary, f, indent=2)

            print(f"\nResults saved to {output_dir}")

        return df_metrics, overall_metrics, mean_metrics, std_metrics

    def save_top20_results(self, df_metrics, output_dir, sort_by='iou'):

        print("\n" + "="*60)
        print(f"🏆 选择表现最好的前20个样本 (按 {sort_by.upper()} 排序)")
        print("="*60)

        top20_df = df_metrics.nlargest(20, sort_by)

        top20_file = os.path.join(output_dir, 'top20_best_results.csv')
        top20_df.to_csv(top20_file, index=False)

        print(f"📊 前20个最佳样本统计:")
        print(f"   IoU范围: {top20_df['iou'].min():.4f} - {top20_df['iou'].max():.4f}")
        print(f"   平均IoU: {top20_df['iou'].mean():.4f}")
        print(f"   F1范围: {top20_df['f1'].min():.4f} - {top20_df['f1'].max():.4f}")
        print(f"   平均F1: {top20_df['f1'].mean():.4f}")
        print(f"   平均精度: {top20_df['precision'].mean():.4f}")
        print(f"   平均召回率: {top20_df['recall'].mean():.4f}")

        print(f"\n🥇 前10个最佳样本详情:")
        print("-" * 80)
        print(f"{'排名':<4} {'图像名称':<25} {'IoU':<8} {'F1':<8} {'精度':<8} {'召回率':<8}")
        print("-" * 80)

        for i, (_, row) in enumerate(top20_df.head(10).iterrows(), 1):
            print(f"{i:<4} {row['image_name']:<25} {row['iou']:<8.4f} {row['f1']:<8.4f} {row['precision']:<8.4f} {row['recall']:<8.4f}")

        top20_summary = {
            'total_samples': len(df_metrics),
            'top20_count': len(top20_df),
            'top20_statistics': {
                'iou': {
                    'min': float(top20_df['iou'].min()),
                    'max': float(top20_df['iou'].max()),
                    'mean': float(top20_df['iou'].mean()),
                    'std': float(top20_df['iou'].std())
                },
                'f1': {
                    'min': float(top20_df['f1'].min()),
                    'max': float(top20_df['f1'].max()),
                    'mean': float(top20_df['f1'].mean()),
                    'std': float(top20_df['f1'].std())
                },
                'precision': {
                    'mean': float(top20_df['precision'].mean()),
                    'std': float(top20_df['precision'].std())
                },
                'recall': {
                    'mean': float(top20_df['recall'].mean()),
                    'std': float(top20_df['recall'].std())
                }
            },
            'top20_samples': top20_df[['image_name', 'iou', 'f1', 'precision', 'recall']].to_dict('records')
        }

        summary_file = os.path.join(output_dir, 'top20_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(top20_summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 前20个最佳结果已保存到:")
        print(f"   📄 详细结果: {top20_file}")
        print(f"   📊 摘要信息: {summary_file}")
        print("="*60)

        return top20_df

    def save_top20_results_comprehensive(self, df_metrics, output_dir):

        print("\n" + "="*80)
        print("🏆 综合评估：多种排序标准下的前20个最佳样本")
        print("="*80)

        criteria = {
            'iou': 'IoU (Intersection over Union)',
            'f1': 'F1 Score',
            'precision': 'Precision',
            'recall': 'Recall',
            'accuracy': 'Accuracy'
        }

        all_top20_results = {}

        for criterion, description in criteria.items():
            print(f"\n📊 按 {description} 排序的前20个样本:")
            print("-" * 60)

            top20_df = df_metrics.nlargest(20, criterion)
            all_top20_results[criterion] = top20_df

            print(f"   {criterion.upper()} 范围: {top20_df[criterion].min():.4f} - {top20_df[criterion].max():.4f}")
            print(f"   平均 {criterion.upper()}: {top20_df[criterion].mean():.4f}")

            print(f"   前5个最佳样本:")
            for i, (_, row) in enumerate(top20_df.head(5).iterrows(), 1):
                print(f"     {i}. {row['image_name']:<20} {criterion.upper()}: {row[criterion]:.4f}")

            criterion_file = os.path.join(output_dir, f'top20_by_{criterion}.csv')
            top20_df.to_csv(criterion_file, index=False)

        print(f"\n🎯 综合排名 (基于IoU, F1, Precision, Recall的加权平均):")
        print("-" * 60)

        df_metrics['composite_score'] = (
            0.4 * df_metrics['iou'] +
            0.3 * df_metrics['f1'] +
            0.15 * df_metrics['precision'] +
            0.15 * df_metrics['recall']
        )

        top20_composite = df_metrics.nlargest(20, 'composite_score')
        composite_file = os.path.join(output_dir, 'top20_composite_ranking.csv')
        top20_composite.to_csv(composite_file, index=False)

        print(f"   综合得分范围: {top20_composite['composite_score'].min():.4f} - {top20_composite['composite_score'].max():.4f}")
        print(f"   平均综合得分: {top20_composite['composite_score'].mean():.4f}")

        print(f"   综合排名前10个样本:")
        for i, (_, row) in enumerate(top20_composite.head(10).iterrows(), 1):
            print(f"     {i:2d}. {row['image_name']:<20} 综合得分: {row['composite_score']:.4f} "
                  f"(IoU: {row['iou']:.3f}, F1: {row['f1']:.3f})")

        comprehensive_summary = {
            'evaluation_criteria': criteria,
            'composite_ranking_weights': {
                'iou': 0.4,
                'f1': 0.3,
                'precision': 0.15,
                'recall': 0.15
            },
            'top20_by_criterion': {
                criterion: {
                    'count': len(df),
                    'min': float(df[criterion].min()),
                    'max': float(df[criterion].max()),
                    'mean': float(df[criterion].mean()),
                    'std': float(df[criterion].std())
                } for criterion, df in all_top20_results.items()
            },
            'composite_ranking': {
                'count': len(top20_composite),
                'min_score': float(top20_composite['composite_score'].min()),
                'max_score': float(top20_composite['composite_score'].max()),
                'mean_score': float(top20_composite['composite_score'].mean()),
                'std_score': float(top20_composite['composite_score'].std())
            },
            'files_generated': {
                'by_criterion': {criterion: f'top20_by_{criterion}.csv' for criterion in criteria.keys()},
                'composite_ranking': 'top20_composite_ranking.csv'
            }
        }

        summary_file = os.path.join(output_dir, 'comprehensive_top20_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_summary, f, indent=2, ensure_ascii=False)

        print(f"\n💾 所有前20个结果已保存到:")
        for criterion in criteria.keys():
            print(f"   📄 按 {criterion.upper()} 排序: top20_by_{criterion}.csv")
        print(f"   🎯 综合排名: top20_composite_ranking.csv")
        print(f"   📊 综合摘要: comprehensive_top20_summary.json")
        print("="*80)

        return all_top20_results, top20_composite

    def visualize_predictions(self, test_dataset, num_samples=10, save_dir='./visualizations'):

        os.makedirs(save_dir, exist_ok=True)

        test_dataset.reset()

        num_samples = min(num_samples, test_dataset.size)
        print(f"Generating visualizations for {num_samples} samples...")

        with torch.no_grad():
            for i in tqdm(range(num_samples)):
                try:

                    image, target, name = test_dataset.load_data()
                    image_tensor = image.to(self.device)
                    target = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).to(self.device)
                except IndexError:
                    print(f"Warning: Only {i} samples available, stopping visualization.")
                    break

                if self.model_type in ['latest_mamba', 'baseline_sam2unet', 'sam2unet',
                                       'mamba_v2', 'mamba_v3', 'vm_unet', 'garden_net', 'star_net',
                                       'double_unet', 'emcad', 'nnwnet', 'attention_unet', 'emsam', 'swin_unet', 'transunet']:
                    model_output = self.model(image_tensor)
                    if isinstance(model_output, tuple) and len(model_output) >= 2:

                        pred = model_output[0]
                        pred1, pred2 = model_output[1] if len(model_output) > 1 else None,\
                                       model_output[2] if len(model_output) > 2 else None
                    else:

                        pred = model_output
                        pred1, pred2 = None, None
                else:

                    pred = self.model(image_tensor)
                    pred1, pred2 = None, None

                if isinstance(pred, tuple):
                    pred = pred[0]
                pred_prob = torch.sigmoid(pred)
                pred_binary = (pred_prob > 0.5).float()

                metrics = self.calculate_metrics(pred, target)

                image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
                image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
                image_np = np.clip(image_np, 0, 1)

                target_np = target.squeeze().cpu().numpy()
                pred_np = pred_prob.squeeze().cpu().numpy()
                pred_binary_np = pred_binary.squeeze().cpu().numpy()

                target_height, target_width = target_np.shape
                pred_height, pred_width = pred_np.shape
                image_height, image_width = image_np.shape[:2]

                if target_height != image_height or target_width != image_width:
                    from PIL import Image
                    target_np = np.array(Image.fromarray((target_np * 255).astype(np.uint8)).resize((image_width, image_height), Image.NEAREST)) / 255.0

                if pred_height != image_height or pred_width != image_width:
                    from PIL import Image
                    pred_np = np.array(Image.fromarray((pred_np * 255).astype(np.uint8)).resize((image_width, image_height), Image.BILINEAR)) / 255.0
                    pred_binary_np = np.array(Image.fromarray((pred_binary_np * 255).astype(np.uint8)).resize((image_width, image_height), Image.NEAREST)) / 255.0

                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                fig.suptitle(f'{name}', fontsize=16, fontweight='bold')

                axes[0, 0].imshow(image_np)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(target_np, cmap='gray')
                axes[0, 1].set_title('Ground Truth')
                axes[0, 1].axis('off')

                im1 = axes[0, 2].imshow(pred_np, cmap='hot')
                axes[0, 2].set_title('Prediction Probability')
                axes[0, 2].axis('off')
                plt.colorbar(im1, ax=axes[0, 2])

                axes[1, 0].imshow(pred_binary_np, cmap='gray')
                axes[1, 0].set_title('Binary Prediction')
                axes[1, 0].axis('off')

                overlay = image_np.copy()
                try:

                    target_mask = target_np > 0.5
                    pred_mask = pred_binary_np > 0.5

                    if target_mask.shape != overlay.shape[:2]:
                        print(f"Warning: Shape mismatch - target: {target_mask.shape}, overlay: {overlay.shape[:2]}")
                        target_mask = np.zeros_like(overlay[:,:,0], dtype=bool)
                    if pred_mask.shape != overlay.shape[:2]:
                        print(f"Warning: Shape mismatch - pred: {pred_mask.shape}, overlay: {overlay.shape[:2]}")
                        pred_mask = np.zeros_like(overlay[:,:,0], dtype=bool)

                    overlay[target_mask] = [1, 0, 0]
                    overlay[pred_mask] = [0, 1, 0]
                except Exception as e:
                    print(f"Warning: Could not create overlay for {name}: {e}")

                    pass

                axes[1, 1].imshow(overlay)
                axes[1, 1].set_title(f'Overlay (Red: GT, Green: Pred)\nPrecision: {metrics["precision"]:.3f}, Recall: {metrics["recall"]:.3f}')
                axes[1, 1].axis('off')

                try:
                    if target_np.shape == pred_binary_np.shape:
                        diff = np.abs(target_np - pred_binary_np)
                    else:

                        min_h = min(target_np.shape[0], pred_binary_np.shape[0])
                        min_w = min(target_np.shape[1], pred_binary_np.shape[1])
                        diff = np.abs(target_np[:min_h, :min_w] - pred_binary_np[:min_h, :min_w])
                except Exception as e:
                    print(f"Warning: Could not create difference map for {name}: {e}")
                    diff = np.zeros_like(target_np)

                axes[1, 2].imshow(diff, cmap='Reds')
                axes[1, 2].set_title(f'Difference Map\nF1: {metrics["f1"]:.3f}, Dice: {metrics["dice"]:.3f}')
                axes[1, 2].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'{name}_visualization.png'), dpi=150, bbox_inches='tight')
                plt.close()

        print(f"Visualizations saved to {save_dir}")

    def plot_metrics_distribution(self, df_metrics, save_dir='./evaluation_results'):

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

        if len(metrics_to_plot) < len(axes):
            axes[-1].remove()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 8))
        correlation_matrix = df_metrics[metrics_to_plot].corr()

        correlation_matrix.index = ['PRECISION', 'RECALL', 'F1', 'IOU', 'DICE', 'SPECIFICITY', 'ACCURACY']
        correlation_matrix.columns = ['PRECISION', 'RECALL', 'F1', 'IOU', 'DICE', 'SPECIFICITY', 'ACCURACY']

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Metrics Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'metrics_correlation.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print("Metrics distribution plots saved")

def main():

    print("=" * 60)
    print("可用的模型类型:")
    print("  ✅ auto - 自动检测模型类型")
    if MAMBA_V2_AVAILABLE:
        print("  ✅ mamba_v2 - MambaSAM2UNetV2（CS3-Mamba，推荐）")
    else:
        print("  ❌ mamba_v2 - MambaSAM2UNetV2（需要 mamba_ssm）")
    if MAMBA_V3_AVAILABLE:
        print("  ✅ mamba_v3 - MambaSAM2UNetV3（VMamba优化版，最新）")
    else:
        print("  ❌ mamba_v3 - MambaSAM2UNetV3（需要 mamba_ssm）")
    if LATEST_MAMBA_AVAILABLE:
        print("  ✅ latest_mamba - LatestMambaLightweightSAM2UNet")
    else:
        print("  ❌ latest_mamba - LatestMambaLightweightSAM2UNet (需要sam2模块)")

    if BASELINE_SAM2UNET_AVAILABLE:
        print("  ✅ baseline_sam2unet - BaselineSAM2UNet")
    else:
        print("  ❌ baseline_sam2unet - BaselineSAM2UNet (需要baseline_sam2unet模块)")

    if VM_UNET_AVAILABLE:
        print("  ✅ vm_unet - VM-UNet (Vision Mamba UNet)")
    else:
        print("  ❌ vm_unet - VM-UNet (需要 mamba_ssm)")
    if GARDEN_NET_AVAILABLE:
        print("  ✅ garden_net - GARDEN-Net (Gradient-guided boundary-aware)")
    else:
        print("  ❌ garden_net - GARDEN-Net (需要 mamba_ssm)")
    if STAR_NET_AVAILABLE:
        print("  ✅ star_net - STAR-Net (Heterogeneous Branch Attention)")
    else:
        print("  ❌ star_net - STAR-Net")
    if DOUBLE_UNET_AVAILABLE:
        print("  ✅ double_unet - DoubleU-Net (Dual Network Cascade)")
    else:
        print("  ❌ double_unet - DoubleU-Net")
    if EMCAD_AVAILABLE:
        print("  ✅ emcad - EMCAD (Efficient Multi-scale Convolutional Attention)")
    else:
        print("  ❌ emcad - EMCAD")
    if NNWNET_AVAILABLE:
        print("  ✅ nnwnet - nnWNet (CVPR 2025)")
    else:
        print("  ❌ nnwnet - nnWNet")
    if ATTENTION_UNET_AVAILABLE:
        print("  ✅ attention_unet - Attention U-Net")
    else:
        print("  ❌ attention_unet - Attention U-Net")
    if EMSAM_AVAILABLE:
        print("  ✅ emsam - EMSAM (Enhanced Multi-Scale SAM)")
    else:
        print("  ❌ emsam - EMSAM")
    if SWIN_UNET_AVAILABLE:
        print("  ✅ swin_unet - Swin-UNet (Agricultural Version)")
    else:
        print("  ❌ swin_unet - Swin-UNet")
    if TRANSUNET_AVAILABLE:
        print("  ✅ transunet - TransUNet (Transformer + CNN)")
    else:
        print("  ❌ transunet - TransUNet")
    if LIGHTM_UNET_AVAILABLE:
        print("  ✅ lightm_unet - LightM-UNet (Lightweight Mamba-UNet)")
    else:
        print("  ❌ lightm_unet - LightM-UNet (需要 mamba_ssm)")

    print("  ✅ unet - UNet")
    print("  ✅ unet_plus_plus - UNet++")
    print("  ✅ deeplabv3_plus - DeepLabV3+")
    print("  ✅ pspnet - PSPNet")
    print("  ✅ fcn - FCN")
    print("  ✅ segnet - SegNet")
    print("  ✅ fpn - FPN")
    if 'SAM2UNET_AVAILABLE' in globals() and SAM2UNET_AVAILABLE:
        print("  ✅ sam2unet - SAM2UNet")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='Evaluate Leaf Disease Segmentation Model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')

    available_models = ['auto']
    if MAMBA_V2_AVAILABLE:
        available_models.append('mamba_v2')
    if MAMBA_V3_AVAILABLE:
        available_models.append('mamba_v3')
    if LATEST_MAMBA_AVAILABLE:
        available_models.append('latest_mamba')
    if BASELINE_SAM2UNET_AVAILABLE:
        available_models.append('baseline_sam2unet')
    if VM_UNET_AVAILABLE:
        available_models.append('vm_unet')
    if GARDEN_NET_AVAILABLE:
        available_models.append('garden_net')
    if STAR_NET_AVAILABLE:
        available_models.append('star_net')
    if DOUBLE_UNET_AVAILABLE:
        available_models.append('double_unet')
    if EMCAD_AVAILABLE:
        available_models.append('emcad')
    if NNWNET_AVAILABLE:
        available_models.append('nnwnet')
    if ATTENTION_UNET_AVAILABLE:
        available_models.append('attention_unet')
    if EMSAM_AVAILABLE:
        available_models.append('emsam')
    if SWIN_UNET_AVAILABLE:
        available_models.append('swin_unet')
    if TRANSUNET_AVAILABLE:
        available_models.append('transunet')
    if LIGHTM_UNET_AVAILABLE:
        available_models.append('lightm_unet')
    available_models.extend(['unet', 'unet_plus_plus', 'deeplabv3_plus', 'pspnet', 'fcn', 'segnet', 'fpn'])
    if 'SAM2UNET_AVAILABLE' in globals() and SAM2UNET_AVAILABLE:
        available_models.append('sam2unet')

    default_model = 'mamba_v3' if MAMBA_V3_AVAILABLE else ('mamba_v2' if MAMBA_V2_AVAILABLE else ('latest_mamba' if LATEST_MAMBA_AVAILABLE else 'unet'))

    parser.add_argument('--model_type', type=str,
                       choices=available_models,
                       default=default_model, help='Type of model to evaluate')
    parser.add_argument('--test_image_path', type=str,
                       default='../coco_dataset_matched/images/test',
                       help='Path to test images')
    parser.add_argument('--test_mask_path', type=str,
                       default='../coco_dataset_matched/masks/test',
                       help='Path to test masks')
    parser.add_argument('--input_size', type=int, default=352,
                       help='Input image size')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--num_visualizations', type=int, default=20,
                       help='Number of samples to visualize')
    parser.add_argument('--visualize_all', action='store_true',
                       help='Visualize all test images (overrides num_visualizations)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')

    args = parser.parse_args()

    evaluator = LeafSegmentationEvaluator(args.model_path, args.model_type, args.device)

    test_dataset = TestDataset(args.test_image_path, args.test_mask_path, args.input_size)

    df_metrics, overall_metrics, mean_metrics, std_metrics = evaluator.evaluate_dataset(
        test_dataset, save_results=True, output_dir=args.output_dir
    )

    test_dataset.reset()

    if args.visualize_all:
        num_samples = test_dataset.size
        print(f"Visualizing all {num_samples} test images...")
    else:
        num_samples = args.num_visualizations
        print(f"Visualizing {num_samples} test images...")

    evaluator.visualize_predictions(
        test_dataset, num_samples=num_samples,
        save_dir=os.path.join(args.output_dir, 'visualizations')
    )

    evaluator.plot_metrics_distribution(df_metrics, save_dir=args.output_dir)

    print(f"\nEvaluation completed! Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()
