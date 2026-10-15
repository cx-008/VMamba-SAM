"""Evaluate a VMamba-SAM checkpoint on a binary leaf-segmentation dataset."""
import argparse, importlib.util, json, os, sys
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np, pandas as pd, torch
import torch.nn.functional as F
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for path in (HERE, ROOT):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from dataset import TestDataset
from utils import ComprehensiveMetrics

def load_model_class():
    """Load VMamba-SAM.py despite the hyphen in its historical filename."""
    for filename in ("VMamba_SAM.py", "VMamba-SAM.py"):
        path = HERE / filename
        if path.exists():
            spec = importlib.util.spec_from_file_location("vmamba_sam_model", path)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            return module.VMambaSAM
    raise FileNotFoundError("VMamba_SAM.py or VMamba-SAM.py was not found")

def unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict): return checkpoint, {}
    meta = checkpoint
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(checkpoint.get(key), dict): return checkpoint[key], meta
    return checkpoint, meta

def strip_prefix(state):
    for prefix in ("module.", "model."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix):]: value for key, value in state.items()}
    return state

class LeafSegmentationEvaluator:
    def __init__(self, model_path, sam2_checkpoint=None, device="cuda", threshold=0.5, strict=False):
        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        raw = torch.load(model_path, map_location="cpu", weights_only=False)
        state, meta = unwrap_checkpoint(raw); state = strip_prefix(state)
        cfg = meta.get("config", {}) if isinstance(meta, dict) else {}
        if not isinstance(cfg, dict): cfg = vars(cfg)
        sam2_checkpoint = sam2_checkpoint or cfg.get("checkpoint_path") or meta.get("sam2_checkpoint")
        VMambaSAM = load_model_class()
        self.model = VMambaSAM(
            checkpoint_path=sam2_checkpoint, num_classes=1,
            use_mamba=cfg.get("use_mamba", True),
            use_skip_connections=cfg.get("use_skip_connections", True),
            use_auxiliary_loss=cfg.get("use_auxiliary_loss", True),
            use_msca=cfg.get("use_msca", True), use_csfa=cfg.get("use_csfa", True),
            use_mamba_adapter=cfg.get("use_mamba_adapter", True), d_state=cfg.get("d_state", 4))
        missing, unexpected = self.model.load_state_dict(state, strict=strict)
        if missing or unexpected: print(f"Checkpoint keys: {len(missing)} missing, {len(unexpected)} unexpected")
        self.model.to(self.device).eval(); self.threshold = threshold

    @staticmethod
    def primary_output(output):
        if isinstance(output, dict): output = output.get("out", output.get("main"))
        if isinstance(output, (tuple, list)): output = output[0]
        return output

    @torch.inference_mode()
    def evaluate(self, dataset, output_dir, visualizations=20):
        output_dir = Path(output_dir); vis_dir = output_dir / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True); vis_dir.mkdir(exist_ok=True)
        dataset.reset(); rows = []
        for index in tqdm(range(dataset.size), desc="Evaluating VMamba-SAM"):
            image, mask, name = dataset.load_data(); image = image.to(self.device)
            logits = self.primary_output(self.model(image))
            target = torch.from_numpy((mask > 127).astype(np.float32))[None, None].to(self.device)
            logits = F.interpolate(logits, target.shape[-2:], mode="bilinear", align_corners=False)
            metrics = ComprehensiveMetrics.calculate_metrics(logits, target, self.threshold)
            rows.append({"image": name, **metrics})
            if index < visualizations:
                pred = (torch.sigmoid(logits)[0, 0].cpu().numpy() > self.threshold).astype(np.uint8)
                self._save_visualization(image[0].cpu(), mask, pred, name, vis_dir)
        frame = pd.DataFrame(rows); frame.to_csv(output_dir / "per_image_metrics.csv", index=False, encoding="utf-8-sig")
        summary = {key: float(frame[key].mean()) for key in frame.columns if key != "image"}
        with open(output_dir / "summary.json", "w", encoding="utf-8") as file: json.dump(summary, file, indent=2)
        print("Mean metrics:", json.dumps(summary, indent=2)); return frame, summary

    @staticmethod
    def _save_visualization(image, mask, pred, name, directory):
        mean = torch.tensor([.485, .456, .406])[:, None, None]; std = torch.tensor([.229, .224, .225])[:, None, None]
        image = ((image * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, data, title in zip(axes, (image, mask, pred), ("Image", "Ground truth", "VMamba-SAM")):
            ax.imshow(data, cmap=None if title == "Image" else "gray"); ax.set_title(title); ax.axis("off")
        fig.tight_layout(); fig.savefig(directory / f"{Path(name).stem}.png", dpi=150); plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="Evaluate VMamba-SAM leaf segmentation")
    p.add_argument("--model_path", required=True); p.add_argument("--sam2_checkpoint", default=None)
    p.add_argument("--model_type", choices=("vmamba_sam", "auto"), default="vmamba_sam")
    p.add_argument("--test_image_path", required=True); p.add_argument("--test_mask_path", required=True)
    p.add_argument("--input_size", type=int, default=352); p.add_argument("--output_dir", default="./evaluation_results")
    p.add_argument("--num_visualizations", type=int, default=20); p.add_argument("--visualize_all", action="store_true")
    p.add_argument("--threshold", type=float, default=.5); p.add_argument("--device", default="cuda")
    p.add_argument("--strict", action="store_true")
    args = p.parse_args(); dataset = TestDataset(args.test_image_path, args.test_mask_path, args.input_size)
    evaluator = LeafSegmentationEvaluator(args.model_path, args.sam2_checkpoint, args.device, args.threshold, args.strict)
    evaluator.evaluate(dataset, args.output_dir, dataset.size if args.visualize_all else args.num_visualizations)

if __name__ == "__main__": main()
