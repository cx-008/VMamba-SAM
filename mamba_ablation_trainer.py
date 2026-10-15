"""Training engine dedicated to VMamba-SAM and its paper ablations."""
import argparse, importlib.util, json, os, random, sys
from datetime import datetime
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
for path in (HERE, ROOT):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from dataset import FullDataset
from mamba_ablation_configs import MambaAblationConfig, get_mamba_config_by_name
from utils import CombinedLoss, ComprehensiveMetrics
from utils.losses import WeightedStructureLoss
from utils.model_analysis import calculate_model_complexity_safe

def load_vmamba_sam():
    for filename in ("VMamba_SAM.py", "VMamba-SAM.py"):
        path = HERE / filename
        if path.exists():
            spec = importlib.util.spec_from_file_location("vmamba_sam_model", path)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            return module.VMambaSAM
    raise FileNotFoundError("VMamba_SAM.py or VMamba-SAM.py was not found")

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class MambaAblationTrainer:
    def __init__(self, config: MambaAblationConfig, args, seed=42):
        self.config, self.args, self.seed = config, args, seed
        if config.model_type != "vmamba_sam":
            raise ValueError("This streamlined trainer supports VMamba-SAM only")
        if args.epochs is not None: config.epochs = args.epochs
        set_seed(seed); self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.exp_dir = Path(args.save_path) / f"{config.name}_seed{seed}"
        for name in ("checkpoints", "logs"): (self.exp_dir / name).mkdir(parents=True, exist_ok=True)
        self.model = self._create_model().to(self.device)
        self.model_complexity = calculate_model_complexity_safe(self.model, (1, 3, args.input_size, args.input_size))
        self.train_criterion = WeightedStructureLoss(kernel_size=31, padding=15, weight_factor=2.0,
                                                     bce_weight=.5, iou_weight=.5,
                                                     edge_enhancement=True, detail_focus=True)
        self.eval_criterion = CombinedLoss(alpha=.5, beta=.3, gamma=.2)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(config.epochs, 1), eta_min=config.lr * .01)
        self.start_epoch = 0; self.best_val_miou = self.best_val_dice = 0.0; self.best_val_loss = float("inf")
        self.train_loader, self.val_loader = self._load_data()
        self.writer = None
        if args.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.exp_dir / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S"))
            except ImportError: print("TensorBoard is unavailable; continuing without it")
        if getattr(args, "resume_from", None): self.load_checkpoint(args.resume_from)

    def _create_model(self):
        checkpoint = self.config.checkpoint_path or self.args.checkpoint_path
        return load_vmamba_sam()(checkpoint_path=checkpoint, num_classes=1,
            use_mamba=self.config.use_mamba, use_skip_connections=self.config.use_skip_connections,
            use_auxiliary_loss=self.config.use_auxiliary_loss, use_msca=self.config.use_msca,
            use_csfa=self.config.use_csfa, use_mamba_adapter=self.config.use_mamba_adapter,
            d_state=self.config.d_state)

    def _load_data(self):
        paths = [self.args.train_image_path, self.args.train_mask_path,
                 self.args.val_image_path, self.args.val_mask_path]
        if not all(os.path.isdir(path) for path in paths):
            missing = [path for path in paths if not os.path.isdir(path)]
            raise FileNotFoundError("Dataset directories not found: " + ", ".join(missing))
        train = FullDataset(paths[0], paths[1], self.args.input_size, mode="train")
        val = FullDataset(paths[2], paths[3], self.args.input_size, mode="val")
        common = dict(num_workers=self.args.num_workers, pin_memory=self.device.type == "cuda")
        return (DataLoader(train, batch_size=self.config.batch_size, shuffle=True, drop_last=True, **common),
                DataLoader(val, batch_size=self.config.batch_size, shuffle=False, **common))

    @staticmethod
    def split_outputs(output):
        if isinstance(output, dict):
            main = output.get("out", output.get("main")); aux = output.get("aux", [])
        elif isinstance(output, (tuple, list)): main, aux = output[0], list(output[1:])
        else: main, aux = output, []
        return main, [item for item in aux if torch.is_tensor(item)]

    def loss(self, output, target, training=True):
        main, auxiliary = self.split_outputs(output)
        main = F.interpolate(main, target.shape[-2:], mode="bilinear", align_corners=False)
        criterion = self.train_criterion if training else self.eval_criterion
        value = criterion(main, target)
        if training and self.config.use_auxiliary_loss:
            for aux in auxiliary:
                aux = F.interpolate(aux, target.shape[-2:], mode="bilinear", align_corners=False)
                value = value + 0.2 * criterion(aux, target)
        return value, main

    def _epoch(self, loader, training):
        self.model.train(training); totals = {"loss": 0.0}; metric_rows = []
        context = torch.enable_grad() if training else torch.inference_mode()
        with context:
            for images, masks in tqdm(loader, desc="Train" if training else "Val", leave=False):
                images, masks = images.to(self.device), masks.to(self.device)
                if masks.ndim == 3: masks = masks.unsqueeze(1)
                masks = (masks > .5).float()
                if training: self.optimizer.zero_grad(set_to_none=True)
                loss, logits = self.loss(self.model(images), masks, training)
                if training:
                    loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_max_norm)
                    self.optimizer.step()
                totals["loss"] += loss.item(); metric_rows.append(ComprehensiveMetrics.calculate_metrics(logits.detach(), masks))
        count = max(len(loader), 1)
        result = {"loss": totals["loss"] / count}
        if metric_rows:
            result.update({key: float(np.mean([row[key] for row in metric_rows])) for key in metric_rows[0]})
        return result

    def train(self):
        for epoch in range(self.start_epoch, self.config.epochs):
            train_result = self._epoch(self.train_loader, True); val_result = self._epoch(self.val_loader, False)
            self.scheduler.step(); improved = val_result.get("miou", 0) > self.best_val_miou
            self.best_val_miou = max(self.best_val_miou, val_result.get("miou", 0))
            self.best_val_dice = max(self.best_val_dice, val_result.get("dice", 0))
            self.best_val_loss = min(self.best_val_loss, val_result["loss"])
            print(f"Epoch {epoch+1}/{self.config.epochs} train={train_result['loss']:.4f} "
                  f"val={val_result['loss']:.4f} mIoU={val_result.get('miou', 0):.4f} Dice={val_result.get('dice', 0):.4f}")
            if self.writer:
                for key, value in train_result.items(): self.writer.add_scalar(f"train/{key}", value, epoch + 1)
                for key, value in val_result.items(): self.writer.add_scalar(f"val/{key}", value, epoch + 1)
            if improved: self.save_checkpoint(epoch, "best_model.pth")
            if (epoch + 1) % self.config.save_interval == 0: self.save_checkpoint(epoch, f"epoch_{epoch+1}.pth")
        self.save_checkpoint(max(self.config.epochs - 1, 0), "last_model.pth")
        if self.writer: self.writer.close()
        result = {"best_val_miou": self.best_val_miou, "best_val_dice": self.best_val_dice,
                  "best_val_loss": self.best_val_loss, "model_complexity": self.model_complexity}
        with open(self.exp_dir / "results.json", "w", encoding="utf-8") as file: json.dump(result, file, indent=2)
        return result

    def save_checkpoint(self, epoch, filename):
        torch.save({"epoch": epoch, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(), "config": vars(self.config),
                    "best_val_miou": self.best_val_miou, "best_val_dice": self.best_val_dice,
                    "best_val_loss": self.best_val_loss}, self.exp_dir / "checkpoints" / filename)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"]); self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint: self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_epoch = checkpoint.get("epoch", -1) + 1
        self.best_val_miou = checkpoint.get("best_val_miou", 0.0); self.best_val_dice = checkpoint.get("best_val_dice", 0.0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

def main():
    p = argparse.ArgumentParser(description="Train one VMamba-SAM ablation")
    for name in ("train_image_path", "train_mask_path", "val_image_path", "val_mask_path"): p.add_argument(f"--{name}", required=True)
    p.add_argument("--checkpoint_path", default=None); p.add_argument("--input_size", type=int, default=352)
    p.add_argument("--num_workers", type=int, default=4); p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--save_path", default="./vmamba_sam_experiments/results"); p.add_argument("--resume_from", default=None)
    p.add_argument("--use_tensorboard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--config_name", default="vmamba_sam")
    args = p.parse_args(); MambaAblationTrainer(get_mamba_config_by_name(args.config_name), args).train()

if __name__ == "__main__": main()
