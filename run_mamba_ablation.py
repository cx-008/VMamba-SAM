
import os
import sys
import argparse
import time
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.append('..')
from mamba_ablation_configs import (
    MAMBA_ABLATION_CONFIGS,
    MAMBA_ABLATION_ORDER,
    get_mamba_configs_by_order,
    print_mamba_ablation_summary
)
from mamba_ablation_trainer import MambaAblationTrainer
from utils.font_utils import safe_plot_setup

safe_plot_setup(use_chinese=False)

class MambaAblationRunner:

    def __init__(self, args):
        self.args = args
        self.results = []
        self.start_time = time.time()

        self.results_dir = os.path.join(args.save_path, "mamba_ablation_results")
        os.makedirs(self.results_dir, exist_ok=True)

        print("=" * 80)
        print("Latest Mamba SAM2-UNet 消融实验")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"结果保存路径: {self.results_dir}")
        print(f"实验配置数量: {len(MAMBA_ABLATION_CONFIGS)}")
        print("=" * 80)

    def run_single_experiment(self, config_name):

        print(f"\n{'='*60}")
        print(f"开始实验: {config_name}")
        print(f"{'='*60}")

        try:

            from mamba_ablation_configs import get_mamba_config_by_name
            config = get_mamba_config_by_name(config_name)

            trainer = MambaAblationTrainer(config, self.args)

            result = trainer.train()

            experiment_result = {
                'config_name': config_name,
                'description': config.description,
                'success': True,
                'best_val_miou': result.get('best_val_miou', 0.0),
                'best_val_dice': result.get('best_val_dice', 0.0),
                'best_val_loss': result.get('best_val_loss', float('inf')),
                'model_complexity': result.get('model_complexity', {}),
                'config': {
                    'use_mamba': config.use_mamba,
                    'use_skip_connections': config.use_skip_connections,
                    'use_auxiliary_loss': config.use_auxiliary_loss,
                    'd_state': config.d_state,
                    'epochs': config.epochs,
                    'lr': config.lr,
                    'weight_decay': config.weight_decay,
                },
                'training_time': time.time() - self.start_time
            }

            print(f"✅ 实验 {config_name} 完成!")
            print(f"   最佳MIoU: {experiment_result['best_val_miou']:.4f}")
            print(f"   最佳Dice: {experiment_result['best_val_dice']:.4f}")
            print(f"   模型参数: {experiment_result['model_complexity'].get('total_params', 0):,}")

            return experiment_result

        except Exception as e:
            print(f"❌ 实验 {config_name} 失败: {e}")
            import traceback
            traceback.print_exc()

            return {
                'config_name': config_name,
                'description': config.description if 'config' in locals() else 'Unknown',
                'success': False,
                'error': str(e),
                'best_val_miou': 0.0,
                'best_val_dice': 0.0,
                'best_val_loss': float('inf'),
                'model_complexity': {},
                'config': {},
                'training_time': time.time() - self.start_time
            }

    def run_all_experiments(self):

        print("开始运行所有曼巴消融实验...")

        configs = get_mamba_configs_by_order()

        for i, config in enumerate(configs, 1):
            print(f"\n进度: {i}/{len(configs)} - {config.name}")

            result = self.run_single_experiment(config.name)
            self.results.append(result)

            self.save_intermediate_results()

            completed = len([r for r in self.results if r['success']])
            print(f"已完成: {completed}/{len(configs)} 个实验")

        print(f"\n所有实验完成! 总用时: {(time.time() - self.start_time)/3600:.2f} 小时")

    def run_selected_experiments(self, config_names):

        print(f"开始运行选定的 {len(config_names)} 个实验...")

        for i, config_name in enumerate(config_names, 1):
            print(f"\n进度: {i}/{len(config_names)} - {config_name}")

            result = self.run_single_experiment(config_name)
            self.results.append(result)

            self.save_intermediate_results()

            completed = len([r for r in self.results if r['success']])
            print(f"已完成: {completed}/{len(config_names)} 个实验")

        print(f"\n选定实验完成! 总用时: {(time.time() - self.start_time)/3600:.2f} 小时")

    def save_intermediate_results(self):

        results_path = os.path.join(self.results_dir, "intermediate_results.json")

        serializable_results = []
        for result in self.results:
            serializable_result = {
                'config_name': result['config_name'],
                'description': result['description'],
                'success': result['success'],
                'best_val_miou': result['best_val_miou'],
                'best_val_dice': result['best_val_dice'],
                'best_val_loss': result['best_val_loss'],
                'model_complexity': result['model_complexity'],
                'config': result['config'],
                'training_time': result['training_time']
            }
            if 'error' in result:
                serializable_result['error'] = result['error']
            serializable_results.append(serializable_result)

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    def save_final_results(self):

        print("\n保存最终结果...")

        results_path = os.path.join(self.results_dir, "final_results.json")
        self.save_intermediate_results()

        csv_path = os.path.join(self.results_dir, "results_summary.csv")
        self.save_csv_results(csv_path)

        self.generate_analysis_report()

        self.plot_comparison_charts()

        print(f"结果已保存到: {self.results_dir}")

    def save_csv_results(self, csv_path):

        data = []
        for result in self.results:
            row = {
                'config_name': result['config_name'],
                'description': result['description'],
                'success': result['success'],
                'best_val_miou': result['best_val_miou'],
                'best_val_dice': result['best_val_dice'],
                'best_val_loss': result['best_val_loss'],
                'total_params': result['model_complexity'].get('total_params', 0),
                'trainable_params': result['model_complexity'].get('trainable_params', 0),
                'model_size_mb': result['model_complexity'].get('params_mb', 0),
                'gflops': result['model_complexity'].get('gflops', 0),
                'use_mamba': result['config'].get('use_mamba', False),
                'use_skip_connections': result['config'].get('use_skip_connections', False),
                'use_auxiliary_loss': result['config'].get('use_auxiliary_loss', False),
                'd_state': result['config'].get('d_state', 0),
                'epochs': result['config'].get('epochs', 0),
                'lr': result['config'].get('lr', 0),
                'weight_decay': result['config'].get('weight_decay', 0),
                'training_time': result['training_time']
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"CSV结果已保存: {csv_path}")

    def generate_analysis_report(self):

        report_path = os.path.join(self.results_dir, "mamba_ablation_analysis.md")

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Mamba SAM2-UNet 消融实验分析报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            successful_experiments = [r for r in self.results if r['success']]
            f.write(f"## 实验统计\n")
            f.write(f"- 总实验数: {len(self.results)}\n")
            f.write(f"- 成功实验数: {len(successful_experiments)}\n")
            f.write(f"- 失败实验数: {len(self.results) - len(successful_experiments)}\n\n")

            if successful_experiments:

                f.write("## 性能排名\n\n")
                sorted_results = sorted(successful_experiments, key=lambda x: x['best_val_miou'], reverse=True)

                f.write("### 按 mIoU 排名\n")
                f.write("| 排名 | 配置名称 | mIoU | Dice | 描述 |\n")
                f.write("|------|----------|------|------|------|\n")

                for i, result in enumerate(sorted_results[:10], 1):
                    f.write(f"| {i} | {result['config_name']} | {result['best_val_miou']:.4f} | {result['best_val_dice']:.4f} | {result['description']} |\n")

                f.write("\n### 按 Dice 排名\n")
                f.write("| 排名 | 配置名称 | Dice | mIoU | 描述 |\n")
                f.write("|------|----------|------|------|------|\n")

                sorted_by_dice = sorted(successful_experiments, key=lambda x: x['best_val_dice'], reverse=True)
                for i, result in enumerate(sorted_by_dice[:10], 1):
                    f.write(f"| {i} | {result['config_name']} | {result['best_val_dice']:.4f} | {result['best_val_miou']:.4f} | {result['description']} |\n")

        print(f"分析报告已保存: {report_path}")

    def plot_comparison_charts(self):

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            successful_experiments = [r for r in self.results if r['success']]
            if not successful_experiments:
                print("没有成功的实验可以绘制图表")
                return

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Mamba SAM2-UNet 消融实验对比', fontsize=16)

            config_names = [r['config_name'] for r in successful_experiments]
            mious = [r['best_val_miou'] for r in successful_experiments]
            dices = [r['best_val_dice'] for r in successful_experiments]
            params = [r['model_complexity'].get('total_params', 0) for r in successful_experiments]

            axes[0, 0].bar(range(len(config_names)), mious, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('mIoU 对比')
            axes[0, 0].set_xlabel('实验配置')
            axes[0, 0].set_ylabel('mIoU')
            axes[0, 0].set_xticks(range(len(config_names)))
            axes[0, 0].set_xticklabels(config_names, rotation=45, ha='right')

            axes[0, 1].bar(range(len(config_names)), dices, color='lightgreen', alpha=0.7)
            axes[0, 1].set_title('Dice 对比')
            axes[0, 1].set_xlabel('实验配置')
            axes[0, 1].set_ylabel('Dice')
            axes[0, 1].set_xticks(range(len(config_names)))
            axes[0, 1].set_xticklabels(config_names, rotation=45, ha='right')

            axes[1, 0].bar(range(len(config_names)), params, color='orange', alpha=0.7)
            axes[1, 0].set_title('模型参数数量对比')
            axes[1, 0].set_xlabel('实验配置')
            axes[1, 0].set_ylabel('参数数量')
            axes[1, 0].set_xticks(range(len(config_names)))
            axes[1, 0].set_xticklabels(config_names, rotation=45, ha='right')

            axes[1, 1].scatter(params, mious, s=100, alpha=0.7, c='red')
            axes[1, 1].set_title('mIoU vs 参数数量')
            axes[1, 1].set_xlabel('参数数量')
            axes[1, 1].set_ylabel('mIoU')

            for i, name in enumerate(config_names):
                axes[1, 1].annotate(name, (params[i], mious[i]),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)

            plt.tight_layout()

            chart_path = os.path.join(self.results_dir, "comparison_charts.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            print(f"对比图表已保存: {chart_path}")

            plt.close()

        except ImportError:
            print("警告: matplotlib 或 seaborn 未安装，跳过图表绘制")
        except Exception as e:
            print(f"绘制图表时出错: {e}")

def main():

    parser = argparse.ArgumentParser(description='Latest Mamba SAM2-UNet 消融实验')

    parser.add_argument('--train_image_path', type=str,
                       default='E:/data/leaf/archive/aug_data/aug_data/leaf/train/images/',
                       help='训练图像路径')
    parser.add_argument('--train_mask_path', type=str,
                       default='E:/data/leaf/archive/aug_data/aug_data/leaf/train/masks/',
                       help='训练掩码路径')
    parser.add_argument('--val_image_path', type=str,
                       default='E:/data/leaf/archive/aug_data/aug_data/leaf/val/images/',
                       help='验证图像路径')
    parser.add_argument('--val_mask_path', type=str,
                       default='E:/data/leaf/archive/aug_data/aug_data/leaf/val/masks/',
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

    parser.add_argument('--run_all', action='store_true', default=False,
                       help='运行所有实验')
    parser.add_argument('--run_selected', nargs='+', default=[],
                       help='运行选定的实验（指定配置名称）')
    parser.add_argument('--run_sequential', action='store_true', default=False,
                       help='按顺序运行旧版消融实验')
    parser.add_argument('--run_v3', action='store_true', default=False,
                       help='运行 MambaSAM2UNetV2 核心消融实验（推荐）')
    parser.add_argument('--run_v3_full', action='store_true', default=False,
                       help='运行 V2 消融 + 经典模型对比（完整论文实验）')

    args = parser.parse_args()

    runner = MambaAblationRunner(args)

    print_mamba_ablation_summary()

    if args.run_all:

        runner.run_all_experiments()
    elif args.run_selected:

        runner.run_selected_experiments(args.run_selected)
    elif args.run_sequential:
        sequential_configs = [
            'baseline_no_mamba',
            'with_skip_connections',
            'with_auxiliary_loss',
            'full_mamba_model',
        ]
        runner.run_selected_experiments(sequential_configs)
    elif args.run_v2_full:

        full_configs = [
            'mamba_v2_baseline',
            'mamba_v2_full',
            'mamba_v2_d_state_4',
            'mamba_v2_d_state_16',
            'mamba_v2_no_skip',
            'mamba_v2_no_aux',
            'unet_classic',
            'unet_plus_plus',
            'deeplabv3_plus',
            'sam2unet',
        ]
        runner.run_selected_experiments(full_configs)
    else:
        print("请指定运行模式:")
        print("  --run_v2              : 运行 V2 核心消融（推荐，6个实验）")
        print("  --run_v2_full         : 运行 V2 + 经典模型对比（完整论文实验）")
        print("  --run_selected name1  : 运行单个实验，如 mamba_v2_full")
        print("  --run_sequential      : 运行旧版逐步消融")
        print("  --run_all             : 运行所有实验")
        print("\n可用的 V2 实验名称:")
        print("  mamba_v2_baseline, mamba_v2_full, mamba_v2_d_state_4,")
        print("  mamba_v2_d_state_16, mamba_v2_no_skip, mamba_v2_no_aux")
        return

    runner.save_final_results()

    print("\n" + "="*80)
    print("所有实验完成!")
    print("="*80)

if __name__ == '__main__':
    main()
