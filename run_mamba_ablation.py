"""Run VMamba-SAM ablations."""
import argparse, csv, json, os, time
from datetime import datetime
from mamba_ablation_configs import MAMBA_ABLATION_ORDER, get_mamba_config_by_name, print_mamba_ablation_summary
from mamba_ablation_trainer import MambaAblationTrainer

class VMambaSAMRunner:
    def __init__(self, args):
        self.args, self.results = args, []
        self.results_dir = os.path.join(args.save_path, "vmamba_sam_ablation_results")
        os.makedirs(self.results_dir, exist_ok=True)

    def run_one(self, name):
        started = time.time()
        try:
            config = get_mamba_config_by_name(name)
            result = MambaAblationTrainer(config, self.args).train() or {}
            row = {"config_name": config.name, "description": config.description, "success": True,
                   "best_val_miou": result.get("best_val_miou", 0.0),
                   "best_val_dice": result.get("best_val_dice", 0.0),
                   "best_val_loss": result.get("best_val_loss", float("inf")),
                   "model_complexity": result.get("model_complexity", {}),
                   "config": vars(config), "training_time": time.time() - started}
        except Exception as exc:
            row = {"config_name": name, "description": "", "success": False, "error": str(exc),
                   "best_val_miou": 0.0, "best_val_dice": 0.0, "best_val_loss": float("inf"),
                   "model_complexity": {}, "config": {}, "training_time": time.time() - started}
            print(f"[FAILED] {name}: {exc}")
        self.results.append(row)
        self.save(False)

    def run(self, names):
        for index, name in enumerate(names, 1):
            print(f"\n[{index}/{len(names)}] VMamba-SAM experiment: {name}")
            self.run_one(name)
        self.save(True)

    def save(self, final):
        filename = "final_results.json" if final else "intermediate_results.json"
        with open(os.path.join(self.results_dir, filename), "w", encoding="utf-8") as file:
            json.dump(self.results, file, indent=2, ensure_ascii=False, allow_nan=True)
        if final:
            cols = ["config_name", "description", "success", "best_val_miou", "best_val_dice",
                    "best_val_loss", "training_time", "error"]
            with open(os.path.join(self.results_dir, "results_summary.csv"), "w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=cols, extrasaction="ignore")
                writer.writeheader(); writer.writerows(self.results)
            print(f"Results: {self.results_dir}")

def build_parser():
    p = argparse.ArgumentParser(description="VMamba-SAM ablation runner")
    p.add_argument("--train_image_path", default="E:/data/leaf/archive/aug_data/aug_data/leaf/train/images/")
    p.add_argument("--train_mask_path", default="E:/data/leaf/archive/aug_data/aug_data/leaf/train/masks/")
    p.add_argument("--val_image_path", default="E:/data/leaf/archive/aug_data/aug_data/leaf/val/images/")
    p.add_argument("--val_mask_path", default="E:/data/leaf/archive/aug_data/aug_data/leaf/val/masks/")
    p.add_argument("--checkpoint_path", default=None); p.add_argument("--input_size", type=int, default=352)
    p.add_argument("--num_workers", type=int, default=4); p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--save_path", default="./vmamba_sam_experiments/results"); p.add_argument("--resume_from", default=None)
    p.add_argument("--use_tensorboard", action=argparse.BooleanOptionalAction, default=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run_all", action="store_true"); mode.add_argument("--run_selected", nargs="+")
    mode.add_argument("--run_core", "--run_v2", "--run_v3", action="store_true")
    return p

def main():
    args = build_parser().parse_args()
    print(f"VMamba-SAM ablation started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print_mamba_ablation_summary()
    names = args.run_selected or (MAMBA_ABLATION_ORDER if args.run_all else
            ["baseline", "vmamba_sam", "no_skip", "no_aux", "no_msca", "no_csfa", "no_adapter"])
    VMambaSAMRunner(args).run(names)

if __name__ == "__main__": main()
