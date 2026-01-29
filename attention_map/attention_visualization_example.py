"""
完整的注意力可视化使用示例
适用于3D医学图像分割模型的可解释性分析
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import yaml
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import json
import sys
import os
from pathlib import Path
"""attention_visualization_example.py """
# 获取当前文件的父目录的父目录（即mamba目录）
current_file = Path(__file__).resolve()
mamba_dir = current_file.parent.parent
sys.path.insert(0, str(mamba_dir))

# 现在可以直接导入
from module.GDGMamU_Net_ESAACA import GDGMamU_Net

# 将模型类注册到全局命名空间
globals()['GDGMamU_Net'] = GDGMamU_Net

# 导入改进的可视化器
from improved_attention_visualizer import (
    AttentionVisualizer,
    load_model,
    load_h5_image,
    preprocess_image
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionAnalyzer:
    """注意力分析器，用于批量分析和比较"""

    def __init__(self, config_path: str):
        """从配置文件初始化"""
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = None
        self.visualizer = None

    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    try:
                        return yaml.safe_load(f)
                    except ImportError:
                        logger.warning("PyYAML not found, using default config")
                        return self._get_default_config()
                else:
                    return self._get_default_config()
        else:
            logger.warning(f"Config file {config_path} not found, using default config")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'model': {
            'path': '../results/best_model_WT0.879_ET0.809_TC0.851_AVG0.846.pth',
            'class_name': 'GDGMamU_Net',
            'params': {
                'in_channels': 4,
                'num_classes': 4
            }
        },
        'data': {
            'inference_file': '../dataset_output/inference.txt',
            'data_dir': '../dataset_output/dataset',
            'target_size': [160, 160, 128]
        },
        'visualization': {
            'target_layers': [
                'GDG1.StripPoolingAttention.conv2',
                'GDG1.conv3_2',
                'GDG2.StripPoolingAttention.conv2',
                'Mamba.mamba.stages.0.blocks.0.dwconv1.depth_conv',
                'fusion_modules.0.fusion_gate.1',
                'fusion_modules.0.fusion_gate.5',
            ],
            'colormap': 'viridis_r',
            'alpha': 0.7,
            'modalities': {
                0: 'T1',
                1: 'T1ce',
                2: 'T2',
                3: 'Flair'}
            },
            'output': {
                'base_dir': 'attention_results',
                'save_slices': True,
                'save_projections': True,
                'create_videos': False
            },
            'processing': {
                'num_samples': 5,
                'selected_modalities': [0, 2],  # T1 and T2
                'batch_size': 1
            },
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }

    def setup_model(self):
        """设置模型和可视化器"""
        model_config = self.config['model']

        try:
            # 尝试动态导入模型类
            if '.' in model_config['class_name']:
                module_name, class_name = model_config['class_name'].rsplit('.', 1)
                try:
                    module = __import__(module_name, fromlist=[class_name])
                    model_class = getattr(module, class_name)
                except ImportError:
                    logger.warning(f"Cannot import {model_config['class_name']}, using mock model")
                    model_class = self._create_mock_model_class()
            else:
                try:
                    # 尝试从当前命名空间导入
                    model_class = globals()[model_config['class_name']]
                except KeyError:
                    logger.warning(f"Model class {model_config['class_name']} not found, using mock model")
                    model_class = self._create_mock_model_class()

            # 加载模型
            self.model = load_model(
                model_config['path'],
                model_class,
                self.device,
                **model_config['params']
            )

            # 创建可视化器
            viz_config = self.config['visualization']
            self.visualizer = AttentionVisualizer(
                self.model,
                viz_config['target_layers'],
                viz_config['colormap'],
                self.device
            )

            logger.info("Model and visualizer setup completed")

        except Exception as e:
            logger.error(f"Failed to setup model: {e}")
            logger.info("Using mock model for demonstration")
            # 使用模拟模型
            model_class = self._create_mock_model_class()
            self.model = model_class(**model_config['params']).to(self.device)
            self.model.eval()

            viz_config = self.config['visualization']
            self.visualizer = AttentionVisualizer(
                self.model,
                viz_config['target_layers'],
                viz_config['colormap'],
                self.device
            )

    def _create_mock_model_class(self):
        """创建模拟模型类用于演示"""

        class MockModel(torch.nn.Module):
            def __init__(self, in_channels=4, num_classes=4):
                super().__init__()
                self.conv1 = torch.nn.Conv3d(in_channels, 64, 3, padding=1)

                # 创建注意力层结构
                self.res_attn1 = torch.nn.Module()
                self.res_attn1.sa = torch.nn.Sequential(
                    torch.nn.Conv3d(64, 64, 1),
                    torch.nn.Sigmoid()
                )

                self.res_attn2 = torch.nn.Module()
                self.res_attn2.sa = torch.nn.Sequential(
                    torch.nn.Conv3d(64, 64, 1),
                    torch.nn.Sigmoid()
                )

                self.res_attn3 = torch.nn.Module()
                self.res_attn3.sa = torch.nn.Sequential(
                    torch.nn.Conv3d(64, 64, 1),
                    torch.nn.Sigmoid()
                )

                # COBA模块
                self.COBA = torch.nn.Module()
                self.COBA.esa = torch.nn.Sequential(
                    torch.nn.Conv3d(64, 1, 1),
                    torch.nn.Sigmoid()
                )
                self.COBA.aca = torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool3d(1),
                    torch.nn.Conv3d(64, 64, 1),
                    torch.nn.Sigmoid()
                )

                self.final = torch.nn.Conv3d(64, num_classes, 1)

            def forward(self, x):
                x = self.conv1(x)

                # 注意力层
                attn1 = self.res_attn1.sa(x)
                x = x * attn1

                attn2 = self.res_attn2.sa(x)
                x = x * attn2

                attn3 = self.res_attn3.sa(x)
                x = x * attn3

                # COBA模块
                esa = self.COBA.esa(x)
                aca = self.COBA.aca(x)
                x = x * esa * aca

                return self.final(x)

        return MockModel

    def analyze_single_case(self, h5_path: str, case_name: str) -> Dict:
        """分析单个病例"""
        try:
            # 加载图像
            image = load_h5_image(h5_path)
            input_tensor = preprocess_image(
                image,
                tuple(self.config['data']['target_size'])
            ).to(self.device)

            # 生成可视化
            output_config = self.config['output']
            save_path = Path(output_config['base_dir']) / case_name

            attention_maps = self.visualizer.visualize_attention(
                input_tensor,
                image,
                str(save_path),
                selected_modalities=self.config['processing']['selected_modalities'],
                alpha=self.config['visualization']['alpha'],
                save_individual_slices=output_config['save_slices'],
                save_projections=output_config['save_projections']
            )

            # 创建视频（如果需要）
            if output_config['create_videos']:
                for layer_name in attention_maps.keys():
                    layer_dir = save_path / layer_name.replace('.', '_')
                    video_path = save_path / f"{layer_name.replace('.', '_')}_video.mp4"
                    try:
                        self.visualizer.create_attention_video(
                            str(layer_dir),
                            str(video_path)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create video for {layer_name}: {e}")

            logger.info(f"Successfully analyzed case: {case_name}")
            return attention_maps

        except Exception as e:
            logger.error(f"Failed to analyze case {case_name}: {e}")
            return {}

    def batch_analysis(self):
        """批量分析"""
        data_config = self.config['data']

        # 读取文件列表
        inference_file = Path(data_config['inference_file'])
        if not inference_file.exists():
            logger.error(f"Inference file not found: {inference_file}")
            logger.info("Creating sample data for demonstration...")
            return self._create_sample_analysis()

        with open(inference_file, 'r') as f:
            h5_files = [line.strip() for line in f.readlines()]

        # 处理指定数量的样本
        num_samples = self.config['processing']['num_samples']
        h5_files = h5_files[:num_samples]

        logger.info(f"Starting batch analysis of {len(h5_files)} cases")

        results = {}
        for h5_file in tqdm(h5_files, desc="Processing cases"):
            h5_path = Path(data_config['data_dir']) / h5_file
            if not h5_path.exists():
                logger.warning(f"File not found: {h5_path}")
                continue

            case_name = h5_path.stem
            attention_maps = self.analyze_single_case(str(h5_path), case_name)
            results[case_name] = attention_maps

        logger.info("Batch analysis completed")
        return results

    def _create_sample_analysis(self):
        """创建示例分析数据"""
        logger.info("Creating sample data for demonstration...")

        # 创建示例数据
        sample_data = self._create_sample_h5_data()

        # 分析示例数据
        results = {}
        for i in range(min(3, self.config['processing']['num_samples'])):
            case_name = f"sample_case_{i:03d}"

            # 生成可视化
            attention_maps = self.visualizer.visualize_attention(
                sample_data['input_tensor'],
                sample_data['image'],
                str(Path(self.config['output']['base_dir']) / case_name),
                selected_modalities=self.config['processing']['selected_modalities'],
                alpha=self.config['visualization']['alpha']
            )

            results[case_name] = attention_maps

        return results

    def _create_sample_h5_data(self):
        """创建示例H5数据"""
        import h5py

        # 创建模拟的MRI数据
        np.random.seed(42)
        target_size = tuple(self.config['data']['target_size'])
        C, H, W, D = 4, *target_size

        # 生成具有解剖结构的模拟数据
        image = np.zeros((C, H, W, D), dtype=np.float32)

        # 为每个模态创建不同的信号特征
        for c in range(C):
            # 基础背景
            image[c] = np.random.normal(0.1, 0.05, (H, W, D))

            # 添加脑组织信号
            center_h, center_w, center_d = H // 2, W // 2, D // 2
            radius = min(H, W, D) // 3

            for h in range(H):
                for w in range(W):
                    for d in range(D):
                        dist = np.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + (d - center_d) ** 2)
                        if dist < radius:
                            # 脑组织信号
                            intensity = 0.5 + 0.3 * np.exp(-dist ** 2 / (2 * (radius / 3) ** 2))
                            image[c, h, w, d] += intensity

                            # 添加一些"病变"区域
                            lesion_centers = [(center_h + 20, center_w + 15, center_d),
                                              (center_h - 15, center_w + 10, center_d + 10)]
                            for lh, lw, ld in lesion_centers:
                                lesion_dist = np.sqrt((h - lh) ** 2 + (w - lw) ** 2 + (d - ld) ** 2)
                                if lesion_dist < 15:
                                    lesion_intensity = 0.8 * np.exp(-lesion_dist ** 2 / (2 * 8 ** 2))
                                    if c == 1:  # T1ce模态中增强更明显
                                        lesion_intensity *= 1.5
                                    image[c, h, w, d] += lesion_intensity

        # 标准化
        for c in range(C):
            image[c] = (image[c] - image[c].min()) / (image[c].max() - image[c].min())

        # 保存示例数据
        os.makedirs('sample_data', exist_ok=True)
        sample_path = 'sample_data/sample_brain.h5'
        with h5py.File(sample_path, 'w') as f:
            f.create_dataset('image', data=image)

        # 预处理
        input_tensor = preprocess_image(image, target_size).to(self.device)

        return {
            'image': image,
            'input_tensor': input_tensor,
            'path': sample_path
        }

    def compare_attention_across_layers(self, case_results: Dict) -> Dict:
        """比较不同层的注意力模式"""
        layer_stats = {}

        for case_name, attention_maps in case_results.items():
            for layer_name, attention_map in attention_maps.items():
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {
                        'mean_attention': [],
                        'max_attention': [],
                        'std_attention': [],
                        'min_attention': []
                    }

                # 计算统计信息
                mean_val = np.mean(attention_map)
                max_val = np.max(attention_map)
                min_val = np.min(attention_map)
                std_val = np.std(attention_map)

                layer_stats[layer_name]['mean_attention'].append(mean_val)
                layer_stats[layer_name]['max_attention'].append(max_val)
                layer_stats[layer_name]['min_attention'].append(min_val)
                layer_stats[layer_name]['std_attention'].append(std_val)

        # 计算汇总统计
        summary_stats = {}
        for layer_name, stats in layer_stats.items():
            summary_stats[layer_name] = {
                'avg_mean_attention': np.mean(stats['mean_attention']),
                'avg_max_attention': np.mean(stats['max_attention']),
                'avg_min_attention': np.mean(stats['min_attention']),
                'avg_std_attention': np.mean(stats['std_attention']),
                'case_count': len(stats['mean_attention'])
            }

        # 保存统计结果
        self._save_comparison_results(summary_stats)

        return summary_stats

    def _save_comparison_results(self, stats: Dict):
        """保存比较结果"""
        output_dir = Path(self.config['output']['base_dir'])
        output_dir.mkdir(exist_ok=True)

        # 创建比较图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        layers = list(stats.keys())
        mean_vals = [stats[layer]['avg_mean_attention'] for layer in layers]
        max_vals = [stats[layer]['avg_max_attention'] for layer in layers]
        std_vals = [stats[layer]['avg_std_attention'] for layer in layers]
        min_vals = [stats[layer]['avg_min_attention'] for layer in layers]

        # 平均注意力强度
        axes[0, 0].bar(range(len(layers)), mean_vals, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Average Mean Attention', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(range(len(layers)))
        axes[0, 0].set_xticklabels(layers, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)

        # 最大注意力强度
        axes[0, 1].bar(range(len(layers)), max_vals, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Average Max Attention', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(range(len(layers)))
        axes[0, 1].set_xticklabels(layers, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)

        # 注意力变异性
        axes[1, 0].bar(range(len(layers)), std_vals, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Average Attention Variability (Std)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(range(len(layers)))
        axes[1, 0].set_xticklabels(layers, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)

        # 注意力范围图（最大-最小）
        attention_ranges = [max_vals[i] - min_vals[i] for i in range(len(layers))]
        axes[1, 1].bar(range(len(layers)), attention_ranges, color='gold', alpha=0.7)
        axes[1, 1].set_title('Attention Range (Max - Min)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xticks(range(len(layers)))
        axes[1, 1].set_xticklabels(layers, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'attention_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 创建雷达图比较
        if len(layers) >= 3:
            self._create_radar_chart(layers, mean_vals, output_dir)

        # 保存数值结果
        with open(output_dir / 'attention_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info("Comparison results saved")

        # 打印总结
        self._print_statistics_summary(stats)

    def _create_radar_chart(self, layers: List[str], values: List[float], output_dir: Path):
        """创建雷达图"""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # 归一化值到0-1范围
        max_val = max(values)
        min_val = min(values)
        if max_val > min_val:
            normalized_values = [(v - min_val) / (max_val - min_val) for v in values]
        else:
            normalized_values = [0.5] * len(values)

        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(layers), endpoint=False).tolist()
        normalized_values += normalized_values[:1]  # 闭合图形
        angles += angles[:1]

        # 绘制雷达图
        ax.plot(angles, normalized_values, 'o-', linewidth=2, color='darkblue')
        ax.fill(angles, normalized_values, alpha=0.25, color='skyblue')

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(layers, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Normalized Mean Attention Comparison\n(Radar Chart)',
                     fontsize=14, fontweight='bold', pad=20)

        plt.savefig(output_dir / 'attention_radar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _print_statistics_summary(self, stats: Dict):
        """打印统计摘要"""
        print("\n" + "=" * 60)
        print("ATTENTION LAYER STATISTICS SUMMARY")
        print("=" * 60)

        for layer_name, layer_stats in stats.items():
            print(f"\n📊 {layer_name}:")
            print(f"   Mean Attention: {layer_stats['avg_mean_attention']:.4f}")
            print(f"   Max Attention:  {layer_stats['avg_max_attention']:.4f}")
            print(f"   Min Attention:  {layer_stats['avg_min_attention']:.4f}")
            print(f"   Attention Std:  {layer_stats['avg_std_attention']:.4f}")
            print(f"   Cases Analyzed: {layer_stats['case_count']}")

        # 找出最活跃的层
        max_attention_layer = max(stats.keys(),
                                  key=lambda x: stats[x]['avg_mean_attention'])
        most_variable_layer = max(stats.keys(),
                                  key=lambda x: stats[x]['avg_std_attention'])

        print(f"\n🔥 Most Active Layer: {max_attention_layer}")
        print(f"🌊 Most Variable Layer: {most_variable_layer}")
        print("=" * 60)


def create_config_file(config_path: str):
    """创建示例配置文件"""
    config = {
        'model': {
            'path': '../results/best_model_WT0.879_ET0.809_TC0.851_AVG0.846.pth',
            'class_name': 'GDGMamU_Net',
            'params': {
                'in_channels': 4,
                'num_classes': 4
            }
        },
        'data': {
            'inference_file': '../dataset_output/inference.txt',
            'data_dir': '../dataset_output/dataset',
            'target_size': [160, 160, 128]
        },
        'visualization': {
            'target_layers': [
                'GDG1.StripPoolingAttention.conv2',
                'GDG1.conv3_2',
                'GDG2.StripPoolingAttention.conv2',
                'Mamba.mamba.stages.0.blocks.0.dwconv1.depth_conv',
                'fusion_modules.0.fusion_gate.1',
                'fusion_modules.0.fusion_gate.5',
            ],
            'colormap': 'viridis_r',
            'alpha': 0.7,
            'modalities': {
                0: 'T1',
                1: 'T1ce',
                2: 'T2',
                3: 'Flair'
            }
        },
        'output': {
            'base_dir': 'attention_results',
            'save_slices': True,
            'save_projections': True,
            'create_videos': False
        },
        'processing': {
            'num_samples': 5,
            'selected_modalities': [0, 2],
            'batch_size': 1
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 尝试保存为YAML，如果失败则保存为JSON
    try:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"YAML configuration file created: {config_path}")
    except ImportError:
        # 如果没有PyYAML，保存为JSON
        json_path = config_path.replace('.yaml', '.json').replace('.yml', '.json')
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"JSON configuration file created: {json_path}")
        print("Install PyYAML for YAML support: pip install PyYAML")


def main():
    parser = argparse.ArgumentParser(description='Advanced Attention Visualization for 3D Medical Images')
    parser.add_argument('--config', type=str, default='attention_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--create_config', action='store_true',
                        help='Create example configuration file')
    parser.add_argument('--list_layers', action='store_true',
                        help='List available layers in the model')
    parser.add_argument('--single_case', type=str, default=None,
                        help='Analyze single case (H5 file path)')
    parser.add_argument('--compare_layers', action='store_true',
                        help='Compare attention patterns across layers')

    args = parser.parse_args()

    # 创建配置文件
    if args.create_config:
        create_config_file(args.config)
        return

    # 检查配置文件
    if not os.path.exists(args.config):
        logger.warning(f"Configuration file not found: {args.config}")
        logger.info("Using default configuration")

    # 初始化分析器
    try:
        analyzer = AttentionAnalyzer(args.config)
        analyzer.setup_model()
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return

    # 列出可用层
    if args.list_layers:
        analyzer.visualizer._print_available_layers()
        return

    # 单个病例分析
    if args.single_case:
        if not os.path.exists(args.single_case):
            logger.error(f"File not found: {args.single_case}")
            return

        case_name = Path(args.single_case).stem
        attention_maps = analyzer.analyze_single_case(args.single_case, case_name)
        logger.info(f"Single case analysis completed. Results saved for {case_name}")
        return

    # 批量分析
    results = analyzer.batch_analysis()

    # 层间比较
    if args.compare_layers and results:
        stats = analyzer.compare_attention_across_layers(results)
        logger.info("Layer comparison completed")

    print("\n🎉 Analysis completed successfully!")
    print(f"📁 Results saved in: {analyzer.config['output']['base_dir']}")


if __name__ == "__main__":
    main()