# VMamba-SAM: Leaf Disease Segmentation

VMamba-SAM is a segmentation model for plant leaf disease images built on the SAM2 hierarchical encoder (Hiera). It integrates VMamba Cross-Scan blocks, LoRA fine-tuning, and MambaAdapter into a 4-level U-Net style decoder.

---

## Directory Structure

```
.
├── VMamba-SAM.py              # Main model (VMambaSAM class)
├── mamba_ablation_trainer.py  # Unified training loop for all models
├── run_mamba_ablation.py      # Entry point for running experiments
├── evaluate_leaf_segmentation.py  # Evaluation script
├── models/                    # Comparison baselines
│   ├── unet.py
│   ├── unet_plus_plus.py
│   ├── deeplabv3_plus.py
│   ├── attention_unet.py
│   ├── double_unet.py
│   ├── emsam_net.py
│   ├── emcad.py
│   ├── lightm_unet.py
│   ├── garden_net.py
│   ├── vm_unet.py
│   └── baseline_sam2unet.py
├── sam2/                      # SAM2 library
├── sam2_configs/              # SAM2 YAML configs
└── utils/                     # Loss, metrics, etc.
```

---

## Requirements

```
torch >= 2.1
mamba-ssm
causal-conv1d
hydra-core
sam2
```

Install Mamba:
```bash
pip install causal-conv1d mamba-ssm
```

---

## Usage

### Train VMamba-SAM

```bash
python run_mamba_ablation.py \
    --run_selected mamba_v3 \
    --train_image_path ./leaf/train/images \
    --train_mask_path  ./leaf/train/masks \
    --val_image_path   ./leaf/val/images \
    --val_mask_path    ./leaf/val/masks \
    --checkpoint_path  ./sam2_hiera_small.pt \
    --save_path        ./results \
    --epochs 100
```

### Train all comparison models

```bash
python run_mamba_ablation.py \
    --run_selected \
        garden_net emsam_net emcad lightm_unet \
        attention_unet double_unet vm_unet \
        unet_classic unet_plus_plus deeplabv3_plus \
    --train_image_path ./leaf/train/images \
    --train_mask_path  ./leaf/train/masks \
    --val_image_path   ./leaf/val/images \
    --val_mask_path    ./leaf/val/masks \
    --save_path        ./results \
    --epochs 100
```

### Evaluate a trained model

```bash
python evaluate_leaf_segmentation.py \
    --model_path ./results/mamba_v3_seed42/checkpoints/best_model.pth \
    --model_type mamba_v3 \
    --test_image_path ./leaf/test/images \
    --test_mask_path  ./leaf/test/masks
```

---

## Model Architecture

| Component | Description |
|-----------|-------------|
| **Encoder** | SAM2-Hiera-Small (frozen) |
| **LoRA** | Rank-8 applied to attention layers of blocks 10–15 |
| **MambaAdapter** | α·Mamba(x) + β·MLP(x) injected into all 16 blocks |
| **CS3MambaBlock** | 4-direction scan + deformable path + register tokens |
| **MSCA** | Multi-scale context aggregation (dilated convolutions) |
| **CSFA** | Cross-scale feature aggregation (independent sigmoid weights) |
| **Decoder** | 4-level BoundaryAwareDecoder with PixelShuffle upsampling |
| **Deep supervision** | 3 auxiliary segmentation heads |

---

## Training Config

| Parameter | Value |
|-----------|-------|
| Input size | 352 × 352 |
| Optimizer | AdamW (LoRA lr×2, Adapter lr×0.5, Decoder lr×1) |
| Base lr | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | OneCycleLR (warmup 10%, cosine decay) |
| Loss (train) | WeightedStructureLoss (BCE + IoU, edge-enhanced) |
| Loss (val) | CombinedLoss (α=0.5, β=0.3, γ=0.2) |
| Gradient clip | max_norm = 1.0 |
| Epochs | 100 |
| Batch size | 8 |

Data augmentation (train): horizontal flip, vertical flip, rotation (90/180/270°), color jitter, random grayscale.

---

## Comparison Models

All baselines are trained with the same config and evaluated with the same protocol (threshold = 0.5, metrics averaged per image).

| Model | Reference |
|-------|-----------|
| UNet | Ronneberger et al., MICCAI 2015 |
| UNet++ | Zhou et al., TMI 2018 |
| DeepLabV3+ | Chen et al., ECCV 2018 |
| Attention U-Net | Oktay et al., MIDL 2018 |
| DoubleU-Net | Jha et al., CBMS 2020 |
| LightM-UNet | arXiv 2024 |
| EMCAD | Rahman et al., CVPR 2024 |
| VM-UNet | Ruan et al., arXiv 2024 |
| EMSAM | Li et al., Front. Plant Sci. 2025 |
| GARDEN-Net | Sun et al., Front. Plant Sci. 2025 |
