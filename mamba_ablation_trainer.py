
import os
import sys
import time
import math
import json
import argparse
import importlib
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
from tqdm import tqdm

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in [_THIS_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from latest_mamba_sam2unet import LatestMambaLightweightSAM2UNet

try:
    from mamba_sam2unet_v2 import MambaSAM2UNetV2
    MAMBA_V2_AVAILABLE = True
except Exception as _e_v2:
    MambaSAM2UNetV2 = None
    MAMBA_V2_AVAILABLE = False
    print(f"[警告] mamba_sam2unet_v2 导入失败: {_e_v2}")

try:
    from VMamba_SAM import VMambaSAM
    MAMBA_V3_AVAILABLE = True
except Exception as _e_v3:
    VMambaSAM = None
    MAMBA_V3_AVAILABLE = False
    print(f"[Warning] VMamba-SAM import failed: {_e_v3}")

try:
    from baseline_sam2unet import BaselineSAM2UNet
except ImportError:
    BaselineSAM2UNet = None

try:
    from SAM2UNet import SAM2UNet
except ImportError:
    SAM2UNet = None
from utils import CombinedLoss, ComprehensiveMetrics, TrainingVisualizer
from utils.losses import WeightedStructureLoss
from utils.model_analysis import calculate_model_complexity_safe
from utils.font_utils import safe_plot_setup
from dataset import FullDataset
from mamba_ablation_configs import MambaAblationConfig, get_mamba_config_by_name

try:
    import models
    importlib.reload(models)
    from models import (
        UNet, UNetPlusPlus as NestedUNet, DeepLabV3Plus,
        PSPNet, FCN, SegNet, FPN
    )
except ImportError as e:
    print(f"经典模型导入失败（仅影响对比实验）: {e}")
    class _NotAvailable:
        def __init__(self, name):
            self._name = name
        def __call__(self, *a, **kw):
            raise NotImplementedError(f"{self._name} not available")
    UNet = _NotAvailable("UNet")
    NestedUNet = _NotAvailable("UNet++")
    DeepLabV3Plus = _NotAvailable("DeepLabV3+")
    PSPNet = _NotAvailable("PSPNet")
    FCN = _NotAvailable("FCN")
    SegNet = _NotAvailable("SegNet")
    FPN = _NotAvailable("FPN")

safe_plot_setup(use_chinese=False)

class MambaAblationTrainer:

    def set_seed(self, seed):

        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"🔒 随机种子已设置: {seed}")

    def __init__(self, config: MambaAblationConfig, args, seed=42):
        self.config = config
        self.args = args
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.set_seed(seed)

        if args.epochs is not None:
            self.config.epochs = args.epochs
            print(f"使用命令行指定的训练轮数: {args.epochs}")

        if hasattr(config, 'dataset_name') and config.dataset_name != "default":

            self.exp_dir = os.path.join(args.save_path, f"dataset_{config.dataset_name}", f"{config.name}_seed{seed}")
        else:

            self.exp_dir = os.path.join(args.save_path, f"{config.name}_seed{seed}")

        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, "plots"), exist_ok=True)

        self.model = self._create_model()
        self.model.to(self.device)

        self.model_complexity = calculate_model_complexity_safe(
            self.model, (1, 3, args.input_size, args.input_size)
        )

        if hasattr(config, 'experiment_group') and config.experiment_group == 'encoder_freezing':
            self._apply_encoder_freezing_strategy(config.name)

        self.gradient_clip_max_norm = getattr(config, 'gradient_clip_max_norm', 1.0)

        self.accumulation_steps = getattr(config, 'accumulation_steps', 1)

        print(f"\n{'='*60}")
        print(f"曼巴消融实验: {config.name} (seed={seed})")
        print(f"描述: {config.description}")
        print(f"{'='*60}")
        print(f"随机种子: {seed}")
        print(f"模型配置:")
        print(f"  - use_mamba: {config.use_mamba}")
        print(f"  - use_skip_connections: {config.use_skip_connections}")
        print(f"  - use_auxiliary_loss: {config.use_auxiliary_loss}")
        print(f"  - d_state: {config.d_state}")
        print(f"训练参数:")
        print(f"  - 梯度裁剪 max_norm: {self.gradient_clip_max_norm}")
        if self.accumulation_steps > 1:
            print(f"  - 梯度累积步数: {self.accumulation_steps}")
            print(f"  - 有效batch size: {config.batch_size * self.accumulation_steps}")
        print(f"模型复杂度:")
        print(f"  - 总参数: {self.model_complexity['total_params']:,}")
        print(f"  - 可训练参数: {self.model_complexity['trainable_params']:,}")
        print(f"  - 模型大小: {self.model_complexity['params_mb']:.2f} MB")
        print(f"  - GFLOPs: {self.model_complexity['gflops']:.2f}")
        print(f"{'='*60}")

        self.start_epoch = 0
        if hasattr(args, 'resume_from') and args.resume_from:
            self.load_checkpoint(args.resume_from)

        loss_type = getattr(config, 'loss_type', 'weighted')

        if loss_type == "bce_only":

            print("🔧 使用纯BCE损失（最稳定模式）")
            self.train_criterion = torch.nn.BCEWithLogitsLoss()
            self.eval_criterion = torch.nn.BCEWithLogitsLoss()
        else:

            weight_factor = getattr(config, 'weight_factor', 2.0)
            self.train_criterion = WeightedStructureLoss(
                kernel_size=31,
                padding=15,
                weight_factor=weight_factor,
                bce_weight=0.5,
                iou_weight=0.5,
                edge_enhancement=True,
                detail_focus=True
            )
            self.eval_criterion = CombinedLoss(alpha=0.5, beta=0.3, gamma=0.2)

        lora_params    = []
        adapter_params = []
        decoder_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'lora_A' in name or 'lora_B' in name:

                lora_params.append(param)
            elif 'encoder' in name:

                adapter_params.append(param)
            else:

                decoder_params.append(param)

        param_groups = []
        if lora_params:
            param_groups.append({'params': lora_params,    'lr': config.lr * 2.0,  'name': 'lora'})
        if adapter_params:
            param_groups.append({'params': adapter_params, 'lr': config.lr * 0.5,  'name': 'adapter'})
        if decoder_params:
            param_groups.append({'params': decoder_params, 'lr': config.lr,         'name': 'decoder'})

        if not param_groups:

            param_groups = [{'params': [p for p in self.model.parameters() if p.requires_grad],
                             'lr': config.lr}]

        self.optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)

        print(f"优化器参数组: lora={len(lora_params)}, adapter={len(adapter_params)}, "
              f"decoder={len(decoder_params)} 个参数")

        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
        )
        self._use_cosine_scheduler = True
        self._use_onecycle = False
        self._pending_onecycle = True

        dataset_available = (
            os.path.exists(args.train_image_path) and
            os.path.exists(args.train_mask_path) and
            os.path.exists(args.val_image_path) and
            os.path.exists(args.val_mask_path)
        )

        if dataset_available:

            print(f"创建训练数据集: {args.train_image_path} -> {args.train_mask_path}")
            print(f"检查训练图像路径: {os.path.exists(args.train_image_path)}")
            print(f"检查训练掩码路径: {os.path.exists(args.train_mask_path)}")
            if os.path.exists(args.train_image_path):
                train_images = os.listdir(args.train_image_path)
                print(f"训练图像文件数量: {len(train_images)}")
            if os.path.exists(args.train_mask_path):
                train_masks = os.listdir(args.train_mask_path)
                print(f"训练掩码文件数量: {len(train_masks)}")

            self.train_dataset = FullDataset(
                args.train_image_path,
                args.train_mask_path,
                args.input_size,
                mode='train'
            )
            print(f"训练数据集大小: {len(self.train_dataset)}")

            print(f"创建验证数据集: {args.val_image_path} -> {args.val_mask_path}")
            print(f"检查验证图像路径: {os.path.exists(args.val_image_path)}")
            print(f"检查验证掩码路径: {os.path.exists(args.val_mask_path)}")
            if os.path.exists(args.val_image_path):
                val_images = os.listdir(args.val_image_path)
                print(f"验证图像文件数量: {len(val_images)}")
            if os.path.exists(args.val_mask_path):
                val_masks = os.listdir(args.val_mask_path)
                print(f"验证掩码文件数量: {len(val_masks)}")

            self.val_dataset = FullDataset(
                args.val_image_path,
                args.val_mask_path,
                args.input_size,
                mode='val'
            )
            print(f"验证数据集大小: {len(self.val_dataset)}")

            if len(self.train_dataset) == 0:
                raise ValueError(f"训练数据集为空！请检查数据路径和文件是否存在。\n"
                               f"训练图像路径: {args.train_image_path}\n"
                               f"训练掩码路径: {args.train_mask_path}")
            if len(self.val_dataset) == 0:
                raise ValueError(f"验证数据集为空！请检查数据路径和文件是否存在。\n"
                               f"验证图像路径: {args.val_image_path}\n"
                               f"验证掩码路径: {args.val_mask_path}")

            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=True
            )
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            self.dataset_available = True
        else:
            print("警告: 数据集路径不存在，将跳过训练，仅进行模型测试")
            self.train_loader = None
            self.val_loader = None
            self.dataset_available = False

        self.use_tensorboard = args.use_tensorboard
        if self.use_tensorboard:
            log_dir = os.path.join(self.exp_dir, "logs", f"tensorboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self.tb_writer = SummaryWriter(log_dir)
            print(f"TensorBoard日志: {log_dir}")

        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.train_metrics = []
        self.best_val_loss = float('inf')
        self.best_val_dice = 0.0
        self.best_val_miou = 0.0

    def _create_model(self):

        checkpoint_path = self.config.checkpoint_path if self.config.checkpoint_path else self.args.checkpoint_path

        if self.config.model_type == "unet":
            print("使用经典 UNet...")
            return UNet(n_channels=3, n_classes=1)

        elif self.config.model_type == "unet_plus_plus":
            print("使用 UNet++...")
            base = NestedUNet(num_classes=1)
            class _Wrap(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x):
                    o = self.m(x)
                    return o[0] if isinstance(o, (list, tuple)) else o
            return _Wrap(base)

        elif self.config.model_type in ("deeplabv3_plus", "pspnet", "fcn", "segnet"):
            cls_map = {"deeplabv3_plus": DeepLabV3Plus, "pspnet": PSPNet,
                       "fcn": FCN, "segnet": SegNet}
            print(f"使用 {self.config.model_type}...")
            base = cls_map[self.config.model_type](num_classes=1)
            class _Wrap(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x):
                    o = self.m(x)
                    if o.shape[2:] != (352, 352):
                        o = F.interpolate(o, size=(352, 352), mode='bilinear', align_corners=True)
                    return o
            return _Wrap(base)

        elif self.config.model_type == "fpn":
            print("使用 FPN...")
            base = FPN(num_classes=1, backbone='resnet50', pretrained=False)
            class _Wrap(nn.Module):
                def __init__(self, m): super().__init__(); self.m = m
                def forward(self, x):
                    o = self.m(x)
                    if o.shape[2:] != (352, 352):
                        o = F.interpolate(o, size=(352, 352), mode='bilinear', align_corners=True)
                    return o
            return _Wrap(base)

        elif self.config.model_type == "baseline_sam2unet":
            print("使用 BaselineSAM2UNet...")

            model_size = 'small'
            if checkpoint_path:
                if 'large' in checkpoint_path.lower():
                    model_size = 'large'
                elif 'tiny' in checkpoint_path.lower():
                    model_size = 'tiny'
                elif 'small' in checkpoint_path.lower():
                    model_size = 'small'

            print(f"   - 模型大小: {model_size}")
            print(f"   - 预训练权重: {checkpoint_path if checkpoint_path else '无'}")

            return BaselineSAM2UNet(
                checkpoint_path=checkpoint_path,
                model_size=model_size
            )

        elif self.config.model_type == "garden_net":
            print("使用 GARDEN (Frontiers Plant Sci 2025)...")
            try:
                from comparison_models.garden_net import GARDEN
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from garden_net import GARDEN
            return GARDEN(num_classes=1)

        elif self.config.model_type == "emsam_net":
            print("使用 EMSAM (Frontiers Plant Sci 2025)...")
            try:
                from comparison_models.emsam_net import EMSAM
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from emsam_net import EMSAM
            return EMSAM(num_classes=1, img_size=self.args.input_size)

        elif self.config.model_type == "star_net":
            print("使用 STAR-Net (Frontiers Plant Sci 2025)...")
            try:
                from comparison_models.star_net import STARNet
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from star_net import STARNet
            return STARNet(num_classes=1)

        elif self.config.model_type == "segformer_b2":
            print("使用 SegFormer-B2 (NeurIPS 2021)...")
            try:
                from comparison_models.segformer import SegFormer
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from segformer import SegFormer
            return SegFormer(num_classes=1)

        elif self.config.model_type == "vm_unet":
            print("使用 VM-UNet (arXiv 2024, Shanghai Jiao Tong University)...")
            print(f"  - 论文配置: depths=[2,2,9,2], depths_decoder=[2,9,2,2], d_state={self.config.d_state}")
            print(f"  - 预期参数量: ~35M (with mamba_ssm)")
            try:
                from comparison_models.vm_unet import VMUNet
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from vm_unet import VMUNet
            return VMUNet(
                in_channels=3,
                num_classes=1,
                depths=(2, 2, 9, 2),
                depths_decoder=(2, 9, 2, 2),
                d_state=self.config.d_state,
                drop_path_rate=0.2,
            )

        elif self.config.model_type == "lightm_unet":
            print("使用 LightM-UNet (arXiv 2024)...")
            print(f"  - 轻量级Mamba辅助UNet: base_channels=32, d_state={self.config.d_state}")
            print(f"  - 预期参数量: ~1.3M")
            try:
                from comparison_models.lightm_unet import LightMUNet
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from lightm_unet import LightMUNet
            return LightMUNet(
                in_channels=3,
                num_classes=1,
                base_channels=32,
                d_state=self.config.d_state,
            )

        elif self.config.model_type == "attention_unet":
            print("使用 Attention U-Net (MIDL 2018)...")
            try:
                from comparison_models.attention_unet import AttentionUNet
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from attention_unet import AttentionUNet
            return AttentionUNet(num_classes=1)

        elif self.config.model_type == "double_unet":
            print("使用 DoubleU-Net (CBMS 2020)...")
            try:
                from comparison_models.double_unet import DoubleUNet
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from double_unet import DoubleUNet
            return DoubleUNet(num_classes=1)

        elif self.config.model_type == "transunet_agri":
            print("使用 TransUNet-Agri (arXiv 2021)...")
            try:
                from comparison_models.transunet_agri import TransUNetAgri
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from transunet_agri import TransUNetAgri
            return TransUNetAgri(num_classes=1)

        elif self.config.model_type == "swin_unet_agri":
            print("使用 Swin-UNet-Agri (ECCV 2022)...")
            try:
                from comparison_models.swin_unet_agri import SwinUNetAgri
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from swin_unet_agri import SwinUNetAgri
            return SwinUNetAgri(num_classes=1, img_size=self.args.input_size)

        elif self.config.model_type == "gmt":
            print("使用 GMT (Guided Mask Transformer, WACV 2025 Oral)...")
            try:
                from comparison_models.gmt import GMT
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from gmt import GMT
            return GMT(num_classes=1, base_ch=64)

        elif self.config.model_type == "nnwnet":
            print("使用 nnWNet (CVPR 2025): WNet架构，Transformer+CNN混合，连续特征传输")
            try:
                from comparison_models.nnwnet import nnWNet
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from nnwnet import nnWNet
            return nnWNet(num_classes=1, base_ch=32)

        elif self.config.model_type == "emcad":
            print("使用 EMCAD (CVPR 2024): 高效多尺度卷积注意力解码器，参数减少79.4%")
            try:
                from comparison_models.emcad import EMCAD
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from emcad import EMCAD
            return EMCAD(num_classes=1, base_ch=32)

        elif self.config.model_type == "swin_umamba":
            print("使用 Swin-UMamba (MICCAI 2024): Swin Transformer + Mamba SSM 混合架构")
            print("  - 论文: Liu et al., arXiv:2402.03302")
            print("  - 特点: 窗口注意力 + 全局Mamba扫描，O(N)复杂度")
            try:
                from comparison_models.swin_umamba import swin_umamba_tiny
            except ImportError:
                import sys
                sys.path.insert(0, os.path.join(_THIS_DIR, 'comparison_models'))
                from swin_umamba import swin_umamba_tiny
            return swin_umamba_tiny(
                img_size=self.args.input_size,
                in_chans=3,
                num_classes=1
            )

        elif self.config.model_type == "sam2unet":
            print("使用 SAM2UNet...")
            return SAM2UNet(checkpoint_path=checkpoint_path)

        elif self.config.model_type == "mamba_v3":
            if not MAMBA_V3_AVAILABLE:
                raise ImportError("VMambaSAM not available. Check VMamba-SAM.py.")
            print(f"Using VMamba-SAM")
            use_msca          = getattr(self.config, 'use_msca', True)
            use_csfa          = getattr(self.config, 'use_csfa', True)
            use_mamba_adapter = getattr(self.config, 'use_mamba_adapter', True)
            model = VMambaSAM(
                checkpoint_path=checkpoint_path,
                num_classes=1,
                use_mamba=self.config.use_mamba,
                use_skip_connections=self.config.use_skip_connections,
                use_auxiliary_loss=self.config.use_auxiliary_loss,
                use_msca=use_msca,
                use_csfa=use_csfa,
                use_mamba_adapter=use_mamba_adapter,
                d_state=self.config.d_state,
            )
            return model

        else:

            print(f"使用 LatestMambaLightweightSAM2UNet  "
                  f"(use_mamba={self.config.use_mamba}, d_state={self.config.d_state})")
            return LatestMambaLightweightSAM2UNet(
                checkpoint_path=checkpoint_path,
                num_classes=1,
                use_mamba=self.config.use_mamba,
                use_skip_connections=self.config.use_skip_connections,
                use_auxiliary_loss=self.config.use_auxiliary_loss,
                d_state=self.config.d_state,
            )

    def calculate_metrics(self, pred, target, threshold=0.5):

        return ComprehensiveMetrics.calculate_metrics(pred, target, threshold)

    def _apply_encoder_freezing_strategy(self, strategy_name):

        print(f"\n{'='*60}")
        print(f"应用编码器冻结策略: {strategy_name}")
        print(f"{'='*60}")

        total_params_before = sum(p.numel() for p in self.model.parameters())
        trainable_params_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print("\n[调试] 模型参数结构示例（前20个）:")
        for i, (name, param) in enumerate(self.model.named_parameters()):
            if i < 20:
                print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        print()

        if strategy_name == "encoder_all_trainable":

            print("✅ 策略: 所有参数可训练（baseline）")
            print("   - SAM2编码器: 可训练")
            print("   - LoRA: 可训练")
            print("   - MambaAdapter: 可训练")
            print("   - 解码器: 可训练")

        elif strategy_name == "encoder_all_frozen":

            print("🔒 策略: 冻结整个SAM2编码器")
            frozen_count = 0
            for name, param in self.model.named_parameters():

                if name.startswith('encoder.'):
                    param.requires_grad = False
                    frozen_count += 1
            print(f"   - 冻结了 {frozen_count} 个SAM2编码器参数")
            print("   - 解码器: 可训练")

        elif strategy_name == "encoder_lora_only":

            print("🎯 策略: 只训练LoRA适配器")
            frozen_count = 0
            lora_count = 0
            for name, param in self.model.named_parameters():
                if name.startswith('encoder.'):

                    if 'lora_A' in name or 'lora_B' in name:

                        param.requires_grad = True
                        lora_count += 1
                    else:
                        param.requires_grad = False
                        frozen_count += 1
            print(f"   - 冻结了 {frozen_count} 个SAM2编码器参数")
            print(f"   - LoRA参数: {lora_count} 个（可训练）")
            print("   - 解码器: 可训练")

        elif strategy_name == "encoder_adapter_only":

            print("🎯 策略: 只训练MambaAdapter")
            frozen_count = 0
            adapter_count = 0
            for name, param in self.model.named_parameters():
                if name.startswith('encoder.'):

                    if any(keyword in name for keyword in ['mlp_branch', 'mamba_branch', 'alpha', 'beta', 'mamba.', 'A_log', 'D', 'conv1d', 'dt_proj', 'x_proj', 'out_proj']):

                        param.requires_grad = True
                        adapter_count += 1
                    else:
                        param.requires_grad = False
                        frozen_count += 1
            print(f"   - 冻结了 {frozen_count} 个SAM2编码器参数")
            print(f"   - MambaAdapter参数: {adapter_count} 个（可训练）")
            print("   - 解码器: 可训练")

        elif strategy_name == "encoder_lora_adapter":

            print("⭐ 策略: 训练LoRA+Adapter（推荐）")
            frozen_count = 0
            lora_count = 0
            adapter_count = 0
            for name, param in self.model.named_parameters():
                if name.startswith('encoder.'):

                    if 'lora_A' in name or 'lora_B' in name:
                        param.requires_grad = True
                        lora_count += 1
                    elif any(keyword in name for keyword in ['mlp_branch', 'mamba_branch', 'alpha', 'beta', 'mamba.', 'A_log', 'D', 'conv1d', 'dt_proj', 'x_proj', 'out_proj']):
                        param.requires_grad = True
                        adapter_count += 1
                    else:
                        param.requires_grad = False
                        frozen_count += 1
            print(f"   - 冻结了 {frozen_count} 个SAM2编码器参数")
            print(f"   - LoRA参数: {lora_count} 个（可训练）")
            print(f"   - MambaAdapter参数: {adapter_count} 个（可训练）")
            print("   - 解码器: 可训练")

        else:
            print(f"⚠️  未知的冻结策略: {strategy_name}，保持默认（全部可训练）")

        trainable_params_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params_before - trainable_params_after

        print(f"\n参数统计:")
        print(f"  - 总参数: {total_params_before:,}")
        print(f"  - 可训练参数: {trainable_params_after:,} ({trainable_params_after/total_params_before*100:.1f}%)")
        print(f"  - 冻结参数: {frozen_params:,} ({frozen_params/total_params_before*100:.1f}%)")
        print(f"{'='*60}\n")

    def calculate_training_metrics(self, pred, target, threshold=0.5):

        return ComprehensiveMetrics.calculate_metrics(pred, target, threshold)

    def _calculate_comprehensive_metrics(self, pred, target, threshold=0.5):

        pred_binary = (pred > threshold).float()
        target_binary = (target > threshold).float()

        pred_flat = pred_binary.view(-1).cpu().numpy()
        target_flat = target_binary.view(-1).cpu().numpy()

        tp = ((pred_flat == 1) & (target_flat == 1)).sum()
        fp = ((pred_flat == 1) & (target_flat == 0)).sum()
        fn = ((pred_flat == 0) & (target_flat == 1)).sum()
        tn = ((pred_flat == 0) & (target_flat == 0)).sum()

        eps = 1e-8

        accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)

        iou = tp / (tp + fp + fn + eps)

        dice = 2 * tp / (2 * tp + fp + fn + eps)

        specificity = tn / (tn + fp + eps)

        sensitivity = recall

        balanced_accuracy = (sensitivity + specificity) / 2

        mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + eps)

        miou = iou

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'dice': dice,
            'miou': miou,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'balanced_accuracy': balanced_accuracy,
            'mcc': mcc,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }

    def train_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0
        total_metrics = {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'dice': 0, 'miou': 0,
            'specificity': 0, 'macc': 0
        }

        pbar = tqdm(self.train_loader, desc=f'[{self.config.name}] Epoch {epoch+1}/{self.config.epochs}')

        for batch_idx, batch in enumerate(pbar):
            images, targets = batch
            images = images.to(self.device)
            targets = targets.to(self.device)

            if batch_idx % self.accumulation_steps == 0:
                self.optimizer.zero_grad()

            model_output = self.model(images)
            if isinstance(model_output, tuple) and len(model_output) == 4:

                pred, pred1, pred2, pred3 = model_output
            elif isinstance(model_output, tuple) and len(model_output) == 3:
                pred, pred1, pred2 = model_output
                pred3 = None
            elif isinstance(model_output, list):

                pred = model_output[0]
                pred1 = pred2 = pred3 = None
            else:
                pred = model_output
                pred1 = pred2 = pred3 = None

            loss_main = self.train_criterion(pred, targets)
            total_loss_batch = loss_main

            if self.config.use_auxiliary_loss and pred1 is not None and pred2 is not None:
                loss_side1 = self.train_criterion(pred1, targets)
                loss_side2 = self.train_criterion(pred2, targets)
                total_loss_batch += 0.2 * loss_side1 + 0.2 * loss_side2

                if pred3 is not None:
                    loss_side3 = self.train_criterion(pred3, targets)
                    total_loss_batch += 0.2 * loss_side3

            total_loss_batch = total_loss_batch / self.accumulation_steps
            total_loss_batch.backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if hasattr(self, '_use_onecycle') and self._use_onecycle:
                    self.scheduler.step()
            else:

                pass

            metrics = self.calculate_training_metrics(pred, targets)

            total_loss += total_loss_batch.item()
            for key in total_metrics:
                total_metrics[key] += metrics[key]

            pbar.set_postfix({
                'Loss': f'{total_loss_batch.item():.4f}',
                'Acc': f'{metrics["accuracy"]:.4f}',
                'Dice': f'{metrics["dice"]:.4f}',
                'IoU': f'{metrics["iou"]:.4f}',
                'F1': f'{metrics["f1"]:.4f}'
            })

            if self.use_tensorboard and batch_idx % 50 == 0:
                self.tb_writer.add_scalar('train/batch_loss', total_loss_batch.item(),
                                        epoch * len(self.train_loader) + batch_idx)
                self.tb_writer.add_scalar('train/batch_dice', metrics['dice'],
                                        epoch * len(self.train_loader) + batch_idx)
                self.tb_writer.add_scalar('train/batch_iou', metrics['iou'],
                                        epoch * len(self.train_loader) + batch_idx)

        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {key: val / len(self.train_loader) for key, val in total_metrics.items()}

        return avg_loss, avg_metrics

    def validate_epoch(self, epoch):

        self.model.eval()
        total_loss = 0.0
        total_metrics = {
            'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'iou': 0, 'dice': 0, 'miou': 0,
            'specificity': 0, 'macc': 0
        }

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'[{self.config.name}] Validation')

            for batch_idx, batch in enumerate(pbar):
                images, targets = batch
                images = images.to(self.device)
                targets = targets.to(self.device)

                model_output = self.model(images)
                if isinstance(model_output, tuple) and len(model_output) == 4:

                    pred, pred1, pred2, pred3 = model_output
                elif isinstance(model_output, tuple) and len(model_output) == 3:
                    pred, pred1, pred2 = model_output
                    pred3 = None
                elif isinstance(model_output, list):

                    pred = model_output[0]
                    pred1 = pred2 = pred3 = None
                else:
                    pred = model_output
                    pred1 = pred2 = pred3 = None

                loss_main = self.eval_criterion(pred, targets)
                total_loss_batch = loss_main

                if self.config.use_auxiliary_loss and pred1 is not None and pred2 is not None:
                    loss_side1 = self.eval_criterion(pred1, targets)
                    loss_side2 = self.eval_criterion(pred2, targets)
                    total_loss_batch += 0.2 * loss_side1 + 0.2 * loss_side2

                    if pred3 is not None:
                        loss_side3 = self.eval_criterion(pred3, targets)
                        total_loss_batch += 0.2 * loss_side3

                metrics = self.calculate_metrics(pred, targets)

                total_loss += total_loss_batch.item()
                for key in total_metrics:
                    total_metrics[key] += metrics[key]

                pbar.set_postfix({
                    'Loss': f'{total_loss_batch.item():.4f}',
                    'Acc': f'{metrics["accuracy"]:.4f}',
                    'Dice': f'{metrics["dice"]:.4f}',
                    'IoU': f'{metrics["iou"]:.4f}',
                    'F1': f'{metrics["f1"]:.4f}'
                })

        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = {key: val / len(self.val_loader) for key, val in total_metrics.items()}

        return avg_loss, avg_metrics

    def load_checkpoint(self, checkpoint_path):

        print(f"加载检查点: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        if 'val_metrics' in checkpoint:
            self.val_metrics = checkpoint['val_metrics']
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        if 'best_val_dice' in checkpoint:
            self.best_val_dice = checkpoint['best_val_dice']
        if 'best_val_miou' in checkpoint:
            self.best_val_miou = checkpoint['best_val_miou']

        self.start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"检查点加载成功! 从epoch {self.start_epoch}继续训练")
        print(f"最佳验证MIoU: {self.best_val_miou:.4f}")
        print(f"最佳验证Dice: {self.best_val_dice:.4f}")

    def save_checkpoint(self, epoch, is_best=False):

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_dice': self.best_val_dice,
            'best_val_miou': self.best_val_miou,
            'model_complexity': self.model_complexity,
            'config': self.config,
            'model_type': 'mamba_ablation'
        }

        if is_best:
            best_path = os.path.join(self.exp_dir, "checkpoints", 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"新的最佳模型已保存: Dice={self.best_val_dice:.4f}, MIoU={self.best_val_miou:.4f}")

            if (self.config.model_type == "mamba_v3" and
                hasattr(self.model, 'reparameterize') and
                getattr(self.config, 'use_reparam', False)):
                try:
                    print("保存重参数化版本（推理优化）...")

                    import copy
                    model_reparam = copy.deepcopy(self.model)
                    model_reparam.reparameterize()

                    reparam_checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model_reparam.state_dict(),
                        'best_val_dice': self.best_val_dice,
                        'best_val_miou': self.best_val_miou,
                        'model_complexity': self.model_complexity,
                        'config': self.config,
                        'model_type': 'mamba_ablation_v3_reparam',
                        'is_reparameterized': True
                    }
                    reparam_path = os.path.join(self.exp_dir, "checkpoints", 'best_model_reparam.pth')
                    torch.save(reparam_checkpoint, reparam_path)
                    print(f"✓ 重参数化模型已保存: {reparam_path}")
                except Exception as e:
                    print(f"⚠ 重参数化保存失败: {e}")

    def _test_model_only(self):

        print("进行模型测试...")

        test_input = torch.randn(1, 3, self.args.input_size, self.args.input_size).to(self.device)

        self.model.eval()
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(test_input)
            inference_time = time.time() - start_time

        return {
            'success': True,
            'model_complexity': self.model_complexity,
            'inference_time': inference_time,
            'test_passed': True,
            'message': '模型测试成功',
            'best_val_miou': 0.0,
            'best_val_dice': 0.0,
            'final_metrics': {
                'miou': 0.0,
                'dice': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        }

    def train(self):

        print(f"\n开始曼巴消融实验训练: {self.config.name}")

        if not self.dataset_available:
            print("数据集不可用，仅进行模型测试...")
            return self._test_model_only()

        print(f"训练样本: {len(self.train_dataset)}, 验证样本: {len(self.val_dataset)}")

        use_onecycle = True
        config_name = getattr(self.config, 'name', '')
        print(f"[调试] 当前配置名称: '{config_name}'")
        print(f"[调试] 检查是否为 sam2_large: {config_name == 'sam2_large'}")

        if config_name == 'sam2_large':
            use_onecycle = False
            print(f"⚠️  大型模型 {config_name} 禁用 OneCycleLR（使用 CosineAnnealingLR）")

        if getattr(self, '_pending_onecycle', False) and self.train_loader is not None and use_onecycle:
            total_steps = self.config.epochs * len(self.train_loader)
            try:
                from torch.optim.lr_scheduler import OneCycleLR
                self.scheduler = OneCycleLR(
                    self.optimizer,
                    max_lr=[pg['lr'] for pg in self.optimizer.param_groups],
                    total_steps=total_steps,
                    pct_start=0.1,
                    anneal_strategy='cos',
                    div_factor=10.0,
                    final_div_factor=100.0
                )
                self._use_cosine_scheduler = False
                self._use_onecycle = True
                self._pending_onecycle = False
                print(f"OneCycleLR 初始化: total_steps={total_steps} "
                      f"({self.config.epochs} epochs × {len(self.train_loader)} batches)")
            except Exception as e:
                print(f"OneCycleLR 初始化失败，保持 CosineAnnealingLR: {e}")
                self._use_onecycle = False
        elif getattr(self, '_pending_onecycle', False) and not use_onecycle:

            self._use_onecycle = False
            self._pending_onecycle = False
            print(f"大型模型使用 CosineAnnealingLR（禁用 OneCycleLR）")
        print("=" * 80)
        loss_type = getattr(self.config, 'loss_type', 'weighted')
        if loss_type == "bce_only":
            print("训练策略: 纯BCE损失(训练+验证) + 标准MIoU(评估)")
            print("=" * 80)
            print("训练损失: BCEWithLogitsLoss (纯BCE，最稳定)")
            print("验证损失: BCEWithLogitsLoss (纯BCE)")
        else:
            print("训练策略: 加权结构损失(训练) + 组合损失(验证) + 标准MIoU(评估)")
            print("=" * 80)
            weight_factor = getattr(self.config, 'weight_factor', 2.0)
            print(f"训练损失: WeightedStructureLoss (BCE+IoU, 边缘增强, weight_factor={weight_factor})")
            print("验证损失: CombinedLoss (多损失组合)")
        print("=" * 80)

        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                print(f"\nEpoch {epoch+1}/{self.config.epochs}")
                print("-" * 50)

                train_loss, train_metrics = self.train_epoch(epoch)

                val_loss, val_metrics = self.validate_epoch(epoch)

                if self._use_cosine_scheduler:

                    self.scheduler.step()
                elif not (hasattr(self, '_use_onecycle') and self._use_onecycle):

                    self.scheduler.step(val_loss)

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)
                self.val_metrics.append(val_metrics)
                self.train_metrics.append(train_metrics)

                is_best = val_metrics['miou'] > self.best_val_miou
                if is_best:
                    self.best_val_miou = val_metrics['miou']
                    self.best_val_dice = val_metrics['dice']
                    self.best_val_loss = val_loss

                print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_metrics['accuracy']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}")
                print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_metrics['accuracy']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")
                print(f"最佳验证 - MIoU: {self.best_val_miou:.4f}, Dice: {self.best_val_dice:.4f}")

                if self.use_tensorboard:
                    self.tb_writer.add_scalar('epoch/train_loss', train_loss, epoch)
                    self.tb_writer.add_scalar('epoch/train_accuracy', train_metrics['accuracy'], epoch)
                    self.tb_writer.add_scalar('epoch/train_precision', train_metrics['precision'], epoch)
                    self.tb_writer.add_scalar('epoch/train_recall', train_metrics['recall'], epoch)
                    self.tb_writer.add_scalar('epoch/train_f1', train_metrics['f1'], epoch)
                    self.tb_writer.add_scalar('epoch/train_dice', train_metrics['dice'], epoch)
                    self.tb_writer.add_scalar('epoch/train_iou', train_metrics['iou'], epoch)
                    self.tb_writer.add_scalar('epoch/train_miou', train_metrics['miou'], epoch)
                    self.tb_writer.add_scalar('epoch/val_loss', val_loss, epoch)
                    self.tb_writer.add_scalar('epoch/val_accuracy', val_metrics['accuracy'], epoch)
                    self.tb_writer.add_scalar('epoch/val_precision', val_metrics['precision'], epoch)
                    self.tb_writer.add_scalar('epoch/val_recall', val_metrics['recall'], epoch)
                    self.tb_writer.add_scalar('epoch/val_f1', val_metrics['f1'], epoch)
                    self.tb_writer.add_scalar('epoch/val_dice', val_metrics['dice'], epoch)
                    self.tb_writer.add_scalar('epoch/val_iou', val_metrics['iou'], epoch)
                    self.tb_writer.add_scalar('epoch/val_miou', val_metrics['miou'], epoch)
                    self.tb_writer.add_scalar('epoch/val_specificity', val_metrics['specificity'], epoch)
                    self.tb_writer.add_scalar('epoch/val_macc', val_metrics['macc'], epoch)
                    self.tb_writer.add_scalar('epoch/learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

                if is_best:
                    self.save_checkpoint(epoch, is_best)

                if (epoch + 1) % 10 == 0:
                    try:
                        self.save_training_history()
                    except Exception:
                        pass

                if not hasattr(self, '_no_improve_count'):
                    self._no_improve_count = 0
                if is_best:
                    self._no_improve_count = 0
                else:
                    self._no_improve_count += 1
                if epoch > 50 and self._no_improve_count >= 30:
                    print(f"触发早停：连续 {self._no_improve_count} epoch 无改善")
                    break

        except Exception as e:
            print(f"\n训练中断: {e}")

            if self.train_losses:
                try:
                    self.save_training_history()
                    print("已保存中断前的训练日志")
                except Exception:
                    pass
            raise

        finally:

            if self.use_tensorboard:
                try:
                    self.tb_writer.close()
                except Exception:
                    pass

        final_epoch = self.config.epochs - 1
        if final_epoch >= 0:

            best_model_path = os.path.join(self.exp_dir, "checkpoints", 'best_model.pth')
            if not os.path.exists(best_model_path):

                self.save_checkpoint(final_epoch, True)
                print("保存最终模型作为最佳模型")

        self.save_training_history()

        self.plot_training_curves()

        return {
            'config': self.config,
            'best_val_miou': self.best_val_miou,
            'best_val_dice': self.best_val_dice,
            'best_val_loss': self.best_val_loss,
            'model_complexity': self.model_complexity,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'train_metrics': self.train_metrics
        }

    def save_training_history(self):

        def _to_python(obj):

            import numpy as np
            if isinstance(obj, dict):
                return {k: _to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_python(v) for v in obj]
            elif isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):
                return obj.item()
            else:
                return obj

        history = {
            'config': {
                'name': self.config.name,
                'description': self.config.description,
                'use_mamba': self.config.use_mamba,
                'use_skip_connections': self.config.use_skip_connections,
                'use_auxiliary_loss': self.config.use_auxiliary_loss,
                'd_state': self.config.d_state,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size,
                'lr': self.config.lr,
                'weight_decay': self.config.weight_decay,
            },
            'results': {
                'best_val_miou': self.best_val_miou,
                'best_val_dice': self.best_val_dice,
                'best_val_loss': self.best_val_loss,
                'model_complexity': self.model_complexity,
            },
            'training_history': {
                'train_losses': self.train_losses,
                'val_losses':   self.val_losses,
                'val_metrics':  self.val_metrics,
                'train_metrics': self.train_metrics,
            },
        }

        history_path = os.path.join(self.exp_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(_to_python(history), f, indent=2, ensure_ascii=False)

        print(f"Training history saved: {history_path}")

    def plot_training_curves(self):

        epochs = range(1, len(self.train_losses) + 1)

        safe_plot_setup(use_chinese=False)

        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle(f'Mamba SAM2-UNet Ablation Study: {self.config.name}\n{self.config.description}', fontsize=16)

        axes[0, 0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        val_accuracy = [m['accuracy'] for m in self.val_metrics]
        train_accuracy = [m['accuracy'] for m in self.train_metrics]
        axes[0, 1].plot(epochs, train_accuracy, 'b-', label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, val_accuracy, 'r-', label='Val Accuracy', linewidth=2)
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

        val_precision = [m['precision'] for m in self.val_metrics]
        train_precision = [m['precision'] for m in self.train_metrics]
        axes[0, 2].plot(epochs, train_precision, 'b-', label='Train Precision', linewidth=2)
        axes[0, 2].plot(epochs, val_precision, 'r-', label='Val Precision', linewidth=2)
        axes[0, 2].set_title('Training and Validation Precision')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)

        val_recall = [m['recall'] for m in self.val_metrics]
        train_recall = [m['recall'] for m in self.train_metrics]
        axes[1, 0].plot(epochs, train_recall, 'b-', label='Train Recall', linewidth=2)
        axes[1, 0].plot(epochs, val_recall, 'r-', label='Val Recall', linewidth=2)
        axes[1, 0].set_title('Training and Validation Recall')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)

        val_f1 = [m['f1'] for m in self.val_metrics]
        train_f1 = [m['f1'] for m in self.train_metrics]
        axes[1, 1].plot(epochs, train_f1, 'b-', label='Train F1', linewidth=2)
        axes[1, 1].plot(epochs, val_f1, 'r-', label='Val F1', linewidth=2)
        axes[1, 1].set_title('Training and Validation F1 Score')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)

        val_dice = [m['dice'] for m in self.val_metrics]
        train_dice = [m['dice'] for m in self.train_metrics]
        axes[1, 2].plot(epochs, train_dice, 'b-', label='Train Dice', linewidth=2)
        axes[1, 2].plot(epochs, val_dice, 'r-', label='Val Dice', linewidth=2)
        axes[1, 2].set_title('Training and Validation Dice Score')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Dice Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(0, 1)

        val_iou = [m['iou'] for m in self.val_metrics]
        train_iou = [m['iou'] for m in self.train_metrics]
        axes[2, 0].plot(epochs, train_iou, 'b-', label='Train IoU', linewidth=2)
        axes[2, 0].plot(epochs, val_iou, 'r-', label='Val IoU', linewidth=2)
        axes[2, 0].set_title('Training and Validation IoU Score')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('IoU Score')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        axes[2, 0].set_ylim(0, 1)

        val_miou = [m['miou'] for m in self.val_metrics]
        train_miou = [m['miou'] for m in self.train_metrics]
        axes[2, 1].plot(epochs, train_miou, 'b-', label='Train MIoU', linewidth=2)
        axes[2, 1].plot(epochs, val_miou, 'r-', label='Val MIoU', linewidth=2)
        axes[2, 1].set_title('Training and Validation MIoU Score')
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('MIoU Score')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        axes[2, 1].set_ylim(0, 1)

        val_specificity = [m['specificity'] for m in self.val_metrics]
        train_specificity = [m['specificity'] for m in self.train_metrics]
        axes[2, 2].plot(epochs, train_specificity, 'b-', label='Train Specificity', linewidth=2)
        axes[2, 2].plot(epochs, val_specificity, 'r-', label='Val Specificity', linewidth=2)
        axes[2, 2].set_title('Training and Validation Specificity')
        axes[2, 2].set_xlabel('Epoch')
        axes[2, 2].set_ylabel('Specificity')
        axes[2, 2].legend()
        axes[2, 2].grid(True, alpha=0.3)
        axes[2, 2].set_ylim(0, 1)

        plt.tight_layout()

        plot_path = os.path.join(self.exp_dir, "plots", 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved: {plot_path}")

        try:
            plt.show()
        except:
            pass

        plt.close()

def main():

    parser = argparse.ArgumentParser(description='Latest Mamba SAM2-UNet 消融实验')

    parser.add_argument('--train_image_path', type=str,
                       default=r'E:/data/leaf/archive/aug_data/aug_data/leaf/train/images/',
                       help='训练图像路径')
    parser.add_argument('--train_mask_path', type=str,
                       default=r'E:/data/leaf/archive/aug_data/aug_data/leaf/train/masks/',
                       help='训练掩码路径')
    parser.add_argument('--val_image_path', type=str,
                       default=r'E:/data/leaf/archive/aug_data/aug_data/leaf/val/images/',
                       help='验证图像路径')
    parser.add_argument('--val_mask_path', type=str,
                       default=r'E:/data/leaf/archive/aug_data/aug_data/leaf/val/masks/',
                       help='验证掩码路径')

    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='SAM2检查点路径')
    parser.add_argument('--input_size', type=int, default=352,
                       help='输入图像大小')

    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖配置文件中的设置）')

    parser.add_argument('--save_path', type=str, default='./mamba_ablation_experiments/results',
                       help='保存路径')

    parser.add_argument('--use_tensorboard', action='store_true', default=True,
                       help='使用TensorBoard日志')

    parser.add_argument('--config_name', type=str, default='baseline_no_mamba',
                       help='曼巴消融实验配置名称')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从检查点恢复训练')

    args = parser.parse_args()

    config = get_mamba_config_by_name(args.config_name)

    trainer = MambaAblationTrainer(config, args)
    results = trainer.train()

    print(f"\n曼巴消融实验 {config.name} 完成!")

    if results and 'best_val_miou' in results:
        print(f"最佳MIoU: {results['best_val_miou']:.4f}")
    else:
        print("最佳MIoU: N/A (仅进行了模型测试)")

    if results and 'best_val_dice' in results:
        print(f"最佳Dice: {results['best_val_dice']:.4f}")
    else:
        print("最佳Dice: N/A (仅进行了模型测试)")

    if results and 'message' in results:
        print(f"状态: {results['message']}")

    if results and 'model_complexity' in results:
        complexity = results['model_complexity']

        model_size = complexity.get('model_size_mb', complexity.get('params_mb', 0))
        print(f"模型复杂度: {complexity['total_params']:,} 参数, {model_size:.2f} MB")

if __name__ == '__main__':
    main()
