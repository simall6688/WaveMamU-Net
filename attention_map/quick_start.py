"""
完整的批次复制注意力可视化脚本
包含完整的图像保存功能
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import logging
import sys
import yaml
import json
from tqdm import tqdm
import cv2

# 添加路径以导入模块
current_file = Path(__file__).resolve()
mamba_dir = current_file.parent.parent
sys.path.insert(0, str(mamba_dir))

from module.GDGMamU_Net_ESAACA import GDGMamU_Net
from attention_visualization_example import AttentionAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_yaml_config_file():
    """创建配置文件"""
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
                'fusion_modules.0.fusion_gate.5'
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
            'batch_size': 4,
            'duplicate_batches': True
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    config_file = 'complete_batch_config.yaml'
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 配置文件已创建: {config_file}")
        return config_file
    except ImportError:
        config_file = 'complete_batch_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ JSON配置文件已创建: {config_file}")
        return config_file


class CompleteBatchAttentionAnalyzer(AttentionAnalyzer):
    """完整的批次复制注意力分析器"""

    def setup_model(self):
        """设置模型和可视化器"""
        model_config = self.config['model']

        try:
            model_path = model_config['path']
            if not os.path.exists(model_path):
                logger.error(f"模型文件不存在: {model_path}")
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            try:
                model_class = globals()[model_config['class_name']]
            except KeyError:
                logger.warning(f"模型类 {model_config['class_name']} 未找到")
                raise KeyError(f"模型类未找到: {model_config['class_name']}")

            self.model = self._load_model_with_batch_fix(
                model_config['path'],
                model_class,
                self.device,
                **model_config['params']
            )

            viz_config = self.config['visualization']

            self.visualizer = CompleteBatchAttentionVisualizer(
                self.model,
                viz_config['target_layers'],
                viz_config['colormap'],
                self.device,
                batch_size=self.config['processing'].get('batch_size', 4)
            )

            logger.info("完整模型和可视化器设置完成")

        except Exception as e:
            logger.error(f"模型设置失败: {e}")
            logger.info("使用模拟模型进行演示")

            model_class = self._create_mock_model_class()
            self.model = model_class(**model_config['params']).to(self.device)
            self.model.eval()

            viz_config = self.config['visualization']

            self.visualizer = CompleteBatchAttentionVisualizer(
                self.model,
                viz_config['target_layers'],
                viz_config['colormap'],
                self.device,
                batch_size=self.config['processing'].get('batch_size', 4)
            )

    def _load_model_with_batch_fix(self, model_path: str, model_class, device: str = 'cuda', **model_kwargs):
        """加载模型并配置批量归一化层"""
        try:
            model = model_class(**model_kwargs)

            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            logger.info("成功加载模型权重")

            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            try:
                model.load_state_dict(state_dict, strict=True)
                logger.info("严格模式加载权重成功")
            except RuntimeError as e:
                logger.warning(f"严格模式失败，使用非严格模式")
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                logger.info(f"非严格加载完成，缺失键: {len(missing_keys)}, 意外键: {len(unexpected_keys)}")

            model.to(device)
            model.eval()

            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    module.eval()
                    module.track_running_stats = True

            logger.info(f"模型加载完成，已优化批量归一化层配置")
            return model

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def compare_attention_across_layers(self, case_results):
        """安全的层间比较函数"""
        layer_stats = {}
        valid_cases = 0

        for case_name, attention_maps in case_results.items():
            if not attention_maps:
                logger.warning(f"案例 {case_name} 没有注意力图数据")
                continue

            valid_cases += 1
            for layer_name, attention_map in attention_maps.items():
                if attention_map is None or attention_map.size == 0:
                    logger.warning(f"层 {layer_name} 的注意力图为空")
                    continue

                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {
                        'mean_attention': [],
                        'max_attention': [],
                        'std_attention': [],
                        'min_attention': []
                    }

                try:
                    mean_val = float(np.mean(attention_map))
                    max_val = float(np.max(attention_map))
                    min_val = float(np.min(attention_map))
                    std_val = float(np.std(attention_map))

                    layer_stats[layer_name]['mean_attention'].append(mean_val)
                    layer_stats[layer_name]['max_attention'].append(max_val)
                    layer_stats[layer_name]['min_attention'].append(min_val)
                    layer_stats[layer_name]['std_attention'].append(std_val)

                except Exception as e:
                    logger.error(f"计算 {layer_name} 统计信息失败: {e}")

        logger.info(f"处理了 {valid_cases} 个有效案例，{len(layer_stats)} 个层有数据")

        if not layer_stats:
            logger.warning("没有有效的注意力图数据进行比较")
            return {}

        summary_stats = {}
        for layer_name, stats in layer_stats.items():
            if not stats['mean_attention']:
                continue

            summary_stats[layer_name] = {
                'avg_mean_attention': float(np.mean(stats['mean_attention'])),
                'avg_max_attention': float(np.mean(stats['max_attention'])),
                'avg_min_attention': float(np.mean(stats['min_attention'])),
                'avg_std_attention': float(np.mean(stats['std_attention'])),
                'case_count': len(stats['mean_attention'])
            }

        if summary_stats:
            self._save_comparison_results_safe(summary_stats)
            logger.info(f"成功生成 {len(summary_stats)} 个层的统计分析")
        else:
            logger.warning("没有有效的汇总统计数据")

        return summary_stats

    def _save_comparison_results_safe(self, stats):
        """安全的比较结果保存函数"""
        output_dir = Path(self.config['output']['base_dir'])
        output_dir.mkdir(exist_ok=True)

        try:
            with open(output_dir / 'attention_stats.json', 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info("统计数据已保存到JSON文件")
        except Exception as e:
            logger.error(f"保存统计数据失败: {e}")

        self._print_statistics_summary_safe(stats)

    def _print_statistics_summary_safe(self, stats):
        """安全的统计摘要打印函数"""
        if not stats:
            print("没有可用的统计数据进行显示")
            return

        print("\n" + "=" * 60)
        print("注意力层统计摘要")
        print("=" * 60)

        for layer_name, layer_stats in stats.items():
            print(f"\n📊 {layer_name}:")
            print(f"   平均注意力: {layer_stats['avg_mean_attention']:.4f}")
            print(f"   最大注意力: {layer_stats['avg_max_attention']:.4f}")
            print(f"   最小注意力: {layer_stats['avg_min_attention']:.4f}")
            print(f"   注意力标准差: {layer_stats['avg_std_attention']:.4f}")
            print(f"   分析案例数: {layer_stats['case_count']}")

        try:
            if len(stats) > 0:
                max_attention_layer = max(stats.keys(), key=lambda x: stats[x]['avg_mean_attention'])
                most_variable_layer = max(stats.keys(), key=lambda x: stats[x]['avg_std_attention'])

                print(f"\n🔥 最活跃层: {max_attention_layer}")
                print(f"🌊 最具变异性层: {most_variable_layer}")
        except Exception as e:
            logger.warning(f"计算层排名时出错: {e}")

        print("=" * 60)


class CompleteBatchAttentionVisualizer:
    """完整的批次复制注意力可视化器，包含图像保存功能"""

    def __init__(self, model, target_layers, cmap='viridis_r', device='cuda', batch_size=4):
        self.model = model
        self.target_layers = target_layers
        self.cmap = cmap
        self.device = device
        self.batch_size = batch_size
        self.activations = {}
        self.hook_handles = []

        self.model.eval()
        self._validate_and_register_hooks()

    def _validate_and_register_hooks(self):
        """验证目标层并注册hooks"""
        def get_activation(name: str):
            def hook(module, input, output):
                try:
                    if isinstance(output, (list, tuple)):
                        activation = output[0] if len(output) > 0 else output
                    else:
                        activation = output

                    if isinstance(activation, torch.Tensor):
                        self.activations[name] = activation[0:1].detach().clone()
                        logger.debug(f"捕获激活 {name}: {activation.shape} -> {self.activations[name].shape}")
                    else:
                        logger.warning(f"层 {name} 输出不是张量: {type(activation)}")

                except Exception as e:
                    logger.error(f"捕获激活 {name} 时出错: {e}")

            return hook

        self._clear_hooks()
        available_layers = self._get_all_layers()

        registered_count = 0
        for layer_name in self.target_layers:
            layer, layer_type = self._get_layer_by_name(layer_name, available_layers)
            if layer is not None:
                try:
                    handle = layer.register_forward_hook(get_activation(layer_name))
                    self.hook_handles.append(handle)
                    registered_count += 1
                    logger.info(f"成功注册钩子: {layer_name}")
                except Exception as e:
                    logger.error(f"注册钩子失败 {layer_name}: {e}")
            else:
                logger.warning(f"层 {layer_name} 在模型中未找到")

        logger.info(f"成功注册 {registered_count}/{len(self.target_layers)} 个钩子")

    def _get_all_layers(self):
        """获取模型中所有层的字典"""
        layers = {}

        def add_layers_recursive(module, prefix=""):
            for name, child in module.named_children():
                current_name = f"{prefix}.{name}" if prefix else name
                module_type = type(child).__name__
                layers[current_name] = (child, module_type)
                add_layers_recursive(child, current_name)

        add_layers_recursive(self.model)
        return layers

    def _get_layer_by_name(self, name, available_layers):
        """通过名称获取模型中的层"""
        if name in available_layers:
            return available_layers[name]

        try:
            current_module = self.model
            for submodule in name.split('.'):
                current_module = getattr(current_module, submodule)
            return current_module, type(current_module).__name__
        except AttributeError:
            return None, ""

    def _print_available_layers(self):
        """打印模型中所有可用的层"""
        available_layers = self._get_all_layers()
        logger.info("模型中的可用层:")
        for name, (_, layer_type) in sorted(available_layers.items()):
            print(f"  {name} ({layer_type})")

    def _clear_hooks(self):
        """清除所有注册的hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.activations = {}

    def _duplicate_batch(self, input_tensor):
        """复制批次以满足批量归一化层要求"""
        if input_tensor.shape[0] == 1:
            duplicated_tensor = input_tensor.repeat(self.batch_size, 1, 1, 1, 1)
            logger.info(f"批次复制: {input_tensor.shape} -> {duplicated_tensor.shape}")
            return duplicated_tensor
        else:
            logger.info("输入已经有多个批次，无需复制")
            return input_tensor

    def visualize_attention(self, input_tensor, original_image, save_path,
                          selected_modalities=None, alpha=0.7,
                          save_individual_slices=True, save_projections=True):
        """生成注意力可视化并保存图像"""

        input_tensor = input_tensor.to(self.device)
        self.model.eval()
        self.activations = {}

        logger.info(f"原始输入张量形状: {input_tensor.shape}")
        logger.info(f"输入张量数值范围: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")

        duplicated_input = self._duplicate_batch(input_tensor)

        with torch.no_grad():
            try:
                output = self.model(duplicated_input)
                logger.info(f"前向传播成功，输出形状: {output.shape if hasattr(output, 'shape') else type(output)}")

            except Exception as e:
                logger.error(f"前向传播失败: {e}")
                return {}

        if not self.activations:
            logger.warning("没有捕获到任何激活")
            return {}

        logger.info(f"成功捕获 {len(self.activations)} 个激活")

        # 创建保存路径
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # 处理模态选择
        if selected_modalities is None:
            selected_modalities = list(range(original_image.shape[0]))
        elif isinstance(selected_modalities, int):
            selected_modalities = [selected_modalities]

        modality_names = {0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'}
        attention_maps = {}

        # 获取原始图像尺寸
        _, H, W, D = original_image.shape

        # 处理每个捕获的激活并保存可视化
        for layer_name, activation in self.activations.items():
            try:
                logger.info(f"处理层 {layer_name}，激活形状: {activation.shape}")

                # 处理激活生成注意力图
                attention_map = self._process_activation(activation)
                if attention_map is None:
                    continue

                # 调整到原始图像尺寸
                attention_resized = self._resize_attention_map(attention_map, (H, W, D))
                attention_np = attention_resized.cpu().numpy().squeeze()

                # 归一化
                if attention_np.max() > attention_np.min():
                    attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min())
                else:
                    attention_np = np.zeros_like(attention_np)

                # 确保维度正确
                if len(attention_np.shape) == 3:
                    attention_np = attention_np[np.newaxis, ...]

                attention_maps[layer_name] = attention_np

                # 为每个模态生成可视化图像
                for modality_idx in selected_modalities:
                    if modality_idx >= original_image.shape[0]:
                        continue

                    modality_name = modality_names.get(modality_idx, f'Modality_{modality_idx}')
                    layer_save_path = save_path / layer_name.replace('.', '_') / modality_name
                    layer_save_path.mkdir(parents=True, exist_ok=True)

                    selected_image = original_image[modality_idx]  # [H, W, D]

                    # 保存切片
                    if save_individual_slices:
                        logger.info(f"保存切片: {layer_name} - {modality_name}")
                        self._save_attention_slices(
                            selected_image, attention_np[0], layer_save_path, alpha
                        )

                    # 保存投影图
                    if save_projections:
                        logger.info(f"保存投影: {layer_name} - {modality_name}")
                        self._save_projections(
                            selected_image, attention_np[0], layer_save_path, alpha
                        )

                logger.info(f"成功处理层 {layer_name}，最终形状: {attention_maps[layer_name].shape}")

            except Exception as e:
                logger.error(f"处理层 {layer_name} 时出错: {e}")
                continue

        logger.info(f"总共生成 {len(attention_maps)} 个注意力图")
        return attention_maps

    def _process_activation(self, activation):
        """处理激活生成注意力图"""
        try:
            if len(activation.shape) == 5:  # [B, C, H, W, D]
                attention_map = torch.mean(activation, dim=1, keepdim=True)
            elif len(activation.shape) == 4:  # [B, C, H, W]
                attention_map = torch.mean(activation, dim=1, keepdim=True)
                attention_map = attention_map.unsqueeze(-1)
            else:
                logger.warning(f"不支持的激活形状: {activation.shape}")
                return None

            # 确保为正值并归一化
            attention_map = torch.clamp(attention_map, min=0)

            return attention_map

        except Exception as e:
            logger.error(f"处理激活时出错: {e}")
            return None

    def _resize_attention_map(self, attention_map, target_size):
        """调整注意力图大小"""
        H, W, D = target_size

        if len(attention_map.shape) == 5:  # [B, C, H, W, D]
            return F.interpolate(
                attention_map,
                size=(H, W, D),
                mode='trilinear',
                align_corners=False
            )
        elif len(attention_map.shape) == 4:  # [B, C, H, W]
            resized = F.interpolate(
                attention_map,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            return resized.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        else:
            raise ValueError(f"不支持的注意力图形状: {attention_map.shape}")

    def _save_attention_slices(self, original_image, attention_map, save_path, alpha):
        """保存所有切片的注意力可视化"""
        H, W, D = original_image.shape

        # 归一化原始图像
        original_norm = self._normalize_image(original_image)

        # 创建切片目录
        slices_dir = save_path / 'slices'
        slices_dir.mkdir(exist_ok=True)

        logger.info(f"开始保存 {D} 个切片到 {slices_dir}")

        for d in tqdm(range(D), desc=f"保存切片到 {save_path.name}", leave=False):
            try:
                self._save_single_slice(
                    original_norm[:, :, d],
                    attention_map[:, :, d],
                    slices_dir / f'slice_{d:03d}.png',
                    alpha
                )
            except Exception as e:
                logger.error(f"保存切片 {d} 失败: {e}")

        logger.info(f"完成保存切片到 {slices_dir}")

    def _save_projections(self, original_image, attention_map, save_path, alpha):
        """保存最大强度投影图"""
        original_norm = self._normalize_image(original_image)

        projections = {
            'axial': (np.max(original_norm, axis=2), np.max(attention_map, axis=2)),
            'coronal': (np.max(original_norm, axis=1), np.max(attention_map, axis=1)),
            'sagittal': (np.max(original_norm, axis=0), np.max(attention_map, axis=0))
        }

        for direction, (orig_proj, att_proj) in projections.items():
            try:
                self._save_single_slice(
                    orig_proj,
                    att_proj,
                    save_path / f'{direction}_projection.png',
                    alpha
                )
            except Exception as e:
                logger.error(f"保存投影 {direction} 失败: {e}")

    def _normalize_image(self, image):
        """归一化图像到[0,1]范围"""
        image_norm = image - image.min()
        if image_norm.max() > 0:
            image_norm = image_norm / image_norm.max()
        return image_norm

    def _save_single_slice(self, original_slice, attention_slice, save_path, alpha):
        """保存单个切片的注意力可视化"""
        try:
            # 获取颜色映射
            cmap = plt.cm.get_cmap(self.cmap)
            attention_color = cmap(attention_slice)[:, :, :3]

            # 转换原始图像为RGB
            original_rgb = np.stack([original_slice] * 3, axis=-1)

            # 叠加
            overlay = (1 - alpha) * original_rgb + alpha * attention_color
            overlay = np.clip(overlay, 0, 1)

            # 创建图像
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # 原始图像
            axes[0].imshow(original_slice, cmap='gray')
            axes[0].set_title('Original')
            axes[0].axis('off')

            # 注意力图
            im = axes[1].imshow(attention_slice, cmap=self.cmap)
            axes[1].set_title('Attention Map')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

            # 叠加图
            axes[2].imshow(overlay)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 单独保存叠加图
            plt.figure(figsize=(8, 8))
            plt.imshow(overlay)
            plt.axis('off')
            overlay_path = save_path.parent / f"{save_path.stem}_overlay.png"
            plt.savefig(overlay_path, dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()

        except Exception as e:
            logger.error(f"保存切片图像失败: {e}")

    def __del__(self):
        """析构函数，清理hooks"""
        self._clear_hooks()


def main():
    """主执行函数"""
    print("=" * 80)
    print("完整的批次复制注意力可视化分析")
    print("包含图像保存功能")
    print("=" * 80)

    # 创建配置文件
    config_file = create_yaml_config_file()

    # 初始化分析器
    try:
        analyzer = CompleteBatchAttentionAnalyzer(config_file)
        analyzer.setup_model()
        print("✅ 分析器初始化成功（包含图像保存功能）")
    except Exception as e:
        logger.error(f"分析器初始化失败: {e}")
        print("❌ 初始化失败，程序退出")
        return

    # 显示配置信息
    batch_size = analyzer.config['processing'].get('batch_size', 4)
    print(f"📋 配置信息:")
    print(f"   批次复制大小: {batch_size}")
    print(f"   目标层数量: {len(analyzer.config['visualization']['target_layers'])}")
    print(f"   保存切片: {analyzer.config['output']['save_slices']}")
    print(f"   保存投影: {analyzer.config['output']['save_projections']}")

    print(f"\n📋 配置的目标层:")
    for i, layer in enumerate(analyzer.config['visualization']['target_layers'], 1):
        print(f"   {i}. {layer}")

    print(f"\n📋 模型中实际可用的层:")
    analyzer.visualizer._print_available_layers()
    print("-" * 80)

    # 执行分析
    print(f"\n🔄 开始批量分析（使用批次复制技术，包含图像保存）...")
    results = analyzer.batch_analysis()

    if results and any(attention_maps for attention_maps in results.values()):
        print("📊 开始层间比较分析...")
        stats = analyzer.compare_attention_across_layers(results)
        if stats:
            print("✅ 层间比较完成")
        else:
            print("⚠️ 层间比较未产生结果")
    else:
        print("⚠️ 批量分析未返回有效结果")

    # 输出结果位置
    output_dir = analyzer.config['output']['base_dir']
    print(f"\n🎉 分析完成!")
    print(f"📁 结果保存位置: {os.path.abspath(output_dir)}")

    # 验证文件是否生成
    output_path = Path(output_dir)
    if output_path.exists():
        subdirs = list(output_path.iterdir())
        print(f"📊 生成的文件夹数量: {len(subdirs)}")
        for subdir in subdirs:
            if subdir.is_dir():
                files_count = len(list(subdir.rglob('*.png')))
                print(f"   {subdir.name}: {files_count} 个图像文件")

    print("=" * 80)


if __name__ == "__main__":
    main()