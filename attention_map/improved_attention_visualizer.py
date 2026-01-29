import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from typing import List, Dict, Optional, Tuple, Union
import warnings
from pathlib import Path
import cv2
from tqdm import tqdm
import logging
import sys
import os
from pathlib import Path
from collections import defaultdict
# 获取当前文件的父目录的父目录（即mamba目录）
current_file = Path(__file__).resolve()
mamba_dir = current_file.parent.parent
sys.path.insert(0, str(mamba_dir))
"""improved_attention_visualizer.py"""
# 现在可以直接导入
from module.GDGMamU_Net_ESAACA import GDGMamU_Net
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttentionVisualizer:
    """
    改进的注意力可视化器，支持3D医学图像的注意力机制可视化
    """

    def __init__(self, model: torch.nn.Module, target_layers: List[str],
                 cmap: str = 'viridis_r', device: str = 'cuda'):
        """
        初始化注意力可视化器

        Args:
            model: 需要进行可视化的模型
            target_layers: 目标层的列表
            cmap: 用于可视化的颜色映射
            device: 计算设备
        """
        self.model = model.eval()  # 确保模型处于评估模式
        self.target_layers = target_layers
        self.cmap = cmap
        self.device = device
        self.activations = {}
        self.hook_handles = []
        self.layer_info = {}

        # 验证和注册hooks
        self._validate_and_register_hooks()

        # 打印可用层信息
        if not target_layers:  # 如果没有指定层，打印所有可用层
            self._print_available_layers()

    def _validate_and_register_hooks(self):
        """验证目标层并注册hooks"""

        def get_activation(name: str, layer_type: str):
            def hook(module, input, output):
                try:
                    # 根据层类型处理不同的输出
                    if isinstance(output, (list, tuple)):
                        # 如果输出是列表或元组，取第一个元素
                        activation = output[0] if len(output) > 0 else output
                    else:
                        activation = output

                    # 确保输出是张量
                    if isinstance(activation, torch.Tensor):
                        self.activations[name] = activation.detach().clone()
                        self.layer_info[name] = {
                            'type': layer_type,
                            'shape': activation.shape,
                            'device': activation.device
                        }
                    else:
                        logger.warning(f"Layer {name} output is not a tensor: {type(activation)}")

                except Exception as e:
                    logger.error(f"Error capturing activation for {name}: {e}")

            return hook

        # 清除之前的钩子
        self._clear_hooks()

        # 获取所有可用层
        available_layers = self._get_all_layers()

        # 注册目标层的hooks
        registered_count = 0
        for layer_name in self.target_layers:
            layer, layer_type = self._get_layer_by_name(layer_name, available_layers)
            if layer is not None:
                try:
                    handle = layer.register_forward_hook(get_activation(layer_name, layer_type))
                    self.hook_handles.append(handle)
                    registered_count += 1
                    logger.info(f"Successfully registered hook for layer: {layer_name} (type: {layer_type})")
                except Exception as e:
                    logger.error(f"Failed to register hook for {layer_name}: {e}")
            else:
                logger.warning(f"Layer {layer_name} not found in the model")

        logger.info(f"Successfully registered {registered_count}/{len(self.target_layers)} hooks")

    def _get_all_layers(self) -> Dict[str, Tuple[torch.nn.Module, str]]:
        """获取模型中所有层的字典"""
        layers = {}

        def add_layers_recursive(module, prefix=""):
            for name, child in module.named_children():
                current_name = f"{prefix}.{name}" if prefix else name

                # 根据模块类型判断是否为注意力相关层
                module_type = type(child).__name__
                if any(keyword in module_type.lower() for keyword in
                       ['attention', 'attn', 'mish', 'conv', 'norm', 'activation']):
                    layers[current_name] = (child, module_type)

                # 递归添加子层
                add_layers_recursive(child, current_name)

        add_layers_recursive(self.model)
        return layers

    def _get_layer_by_name(self, name: str, available_layers: Dict) -> Tuple[Optional[torch.nn.Module], str]:
        """通过名称获取模型中的层"""
        if name in available_layers:
            return available_layers[name]

        # 尝试通过属性访问
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
        logger.info("Available layers in the model:")
        for name, (_, layer_type) in sorted(available_layers.items()):
            print(f"  {name} ({layer_type})")

    def _clear_hooks(self):
        """清除所有注册的hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.activations = {}
        self.layer_info = {}

    def visualize_attention(self, input_tensor: torch.Tensor, original_image: np.ndarray,
                            save_path: str, selected_modalities: Union[int, List[int]] = None,
                            alpha: float = 0.7, save_individual_slices: bool = True,
                            save_projections: bool = True) -> Dict[str, np.ndarray]:
        """
        生成注意力可视化

        Args:
            input_tensor: 输入张量 [1, C, H, W, D]
            original_image: 原始图像 [C, H, W, D]
            save_path: 保存路径
            selected_modalities: 选择的模态索引，None表示所有模态
            alpha: 透明度
            save_individual_slices: 是否保存单独的切片
            save_projections: 是否保存投影图

        Returns:
            Dict[str, np.ndarray]: 各层的注意力图
        """
        # 确保输入在正确的设备上
        input_tensor = input_tensor.to(self.device)

        # 清除之前的激活
        self.activations = {}

        # 前向传播
        with torch.no_grad():
            try:
                _ = self.model(input_tensor)
            except Exception as e:
                logger.error(f"Forward pass failed: {e}")
                return {}

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

        # 为每个目标层生成可视化
        for layer_name in self.target_layers:
            if layer_name not in self.activations:
                logger.warning(f"No activation captured for layer {layer_name}")
                continue

            try:
                # 处理注意力激活
                attention_map = self._process_attention_map(
                    self.activations[layer_name],
                    self.layer_info.get(layer_name, {})
                )

                if attention_map is None:
                    continue

                logger.info(f"Processing layer: {layer_name}, shape: {attention_map.shape}")

                # 调整注意力图尺寸以匹配原始图像
                attention_resized = self._resize_attention_map(attention_map, (H, W, D))
                attention_np = attention_resized.cpu().numpy().squeeze()

                # 确保维度正确
                if len(attention_np.shape) == 3:
                    attention_np = attention_np[np.newaxis, ...]  # [1, H, W, D]

                attention_maps[layer_name] = attention_np

                # 为每个模态生成可视化
                for modality_idx in selected_modalities:
                    if modality_idx >= original_image.shape[0]:
                        continue

                    modality_name = modality_names.get(modality_idx, f'Modality_{modality_idx}')
                    layer_save_path = save_path / layer_name.replace('.', '_') / modality_name
                    layer_save_path.mkdir(parents=True, exist_ok=True)

                    selected_image = original_image[modality_idx]  # [H, W, D]

                    # 保存切片
                    if save_individual_slices:
                        self._save_attention_slices(
                            selected_image, attention_np[0], layer_save_path, alpha
                        )

                    # 保存投影图
                    if save_projections:
                        self._save_projections(
                            selected_image, attention_np[0], layer_save_path, alpha
                        )

            except Exception as e:
                logger.error(f"Error processing layer {layer_name}: {e}")
                continue

        return attention_maps

    def _process_attention_map(self, activation: torch.Tensor,
                               layer_info: Dict) -> Optional[torch.Tensor]:
        """
        处理注意力图，支持多种类型的激活

        Args:
            activation: 激活张量
            layer_info: 层信息

        Returns:
            处理后的注意力图 [B, 1, H, W, D] 或 None
        """
        try:
            original_shape = activation.shape
            logger.debug(f"Processing activation with shape: {original_shape}")

            # 处理不同维度的激活
            if len(original_shape) == 5:  # [B, C, H, W, D] 或 [B, C, D, H, W]
                B, C = original_shape[:2]

                # 如果通道数大于1，取平均或最大值
                if C > 1:
                    activation = torch.mean(activation, dim=1, keepdim=True)  # 取通道平均

                # 确保空间维度正确
                if original_shape[2] < original_shape[3]:  # 可能是 [B, C, D, H, W]
                    activation = activation.permute(0, 1, 3, 4, 2)  # 转为 [B, C, H, W, D]

            elif len(original_shape) == 4:  # [B, C, H, W] - 2D注意力图
                B, C, H, W = original_shape
                if C > 1:
                    activation = torch.mean(activation, dim=1, keepdim=True)
                # 添加深度维度
                activation = activation.unsqueeze(-1)  # [B, 1, H, W, 1]

            elif len(original_shape) == 3:  # [B, N, N] - 注意力权重矩阵
                B, N, _ = original_shape
                # 尝试重塑为3D空间
                side_length = int(round(N ** (1 / 3)))
                if abs(side_length ** 3 - N) <= 1:  # 近似立方
                    activation = activation.mean(dim=2).view(B, 1, side_length, side_length, side_length)
                else:
                    # 无法重塑，返回None
                    logger.warning(f"Cannot reshape attention matrix with N={N}")
                    return None

            elif len(original_shape) == 2:  # [B, N] - 1D注意力
                B, N = original_shape
                side_length = int(round(N ** (1 / 3)))
                if abs(side_length ** 3 - N) <= 1:
                    activation = activation.view(B, 1, side_length, side_length, side_length)
                else:
                    logger.warning(f"Cannot reshape 1D attention with N={N}")
                    return None
            else:
                logger.warning(f"Unsupported activation shape: {original_shape}")
                return None

            # 确保激活为正值
            activation = torch.clamp(activation, min=0)

            # 归一化到 [0, 1] 范围
            activation = self._normalize_activation(activation)

            return activation

        except Exception as e:
            logger.error(f"Error processing attention map: {e}")
            return None

    def _normalize_activation(self, activation: torch.Tensor) -> torch.Tensor:
        """归一化激活到[0,1]范围"""
        # 计算每个样本的最小值和最大值
        dims = list(range(1, len(activation.shape)))  # 除了batch维度

        min_vals = activation.view(activation.shape[0], -1).min(dim=1, keepdim=True)[0]
        max_vals = activation.view(activation.shape[0], -1).max(dim=1, keepdim=True)[0]

        # 重塑为正确的形状进行广播
        for _ in range(len(activation.shape) - 2):
            min_vals = min_vals.unsqueeze(-1)
            max_vals = max_vals.unsqueeze(-1)

        # 避免除零
        range_vals = max_vals - min_vals
        range_vals = torch.clamp(range_vals, min=1e-8)

        normalized = (activation - min_vals) / range_vals
        return torch.clamp(normalized, 0, 1)

    def _resize_attention_map(self, attention_map: torch.Tensor,
                              target_size: Tuple[int, int, int]) -> torch.Tensor:
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
            # 扩展到3D
            return resized.unsqueeze(-1).expand(-1, -1, -1, -1, D)
        else:
            raise ValueError(f"Unsupported attention map shape: {attention_map.shape}")

    def _save_attention_slices(self, original_image: np.ndarray, attention_map: np.ndarray,
                               save_path: Path, alpha: float):
        """保存所有切片的注意力可视化"""
        H, W, D = original_image.shape

        # 归一化原始图像
        original_norm = self._normalize_image(original_image)

        # 创建切片目录
        slices_dir = save_path / 'slices'
        slices_dir.mkdir(exist_ok=True)

        for d in tqdm(range(D), desc=f"Saving slices to {save_path.name}", leave=False):
            self._save_single_slice(
                original_norm[:, :, d],
                attention_map[:, :, d],
                slices_dir / f'slice_{d:03d}.png',
                alpha
            )

    def _save_projections(self, original_image: np.ndarray, attention_map: np.ndarray,
                          save_path: Path, alpha: float):
        """保存最大强度投影图"""
        # 归一化原始图像
        original_norm = self._normalize_image(original_image)

        # 计算三个方向的最大投影
        projections = {
            'axial': (np.max(original_norm, axis=2), np.max(attention_map, axis=2)),
            'coronal': (np.max(original_norm, axis=1), np.max(attention_map, axis=1)),
            'sagittal': (np.max(original_norm, axis=0), np.max(attention_map, axis=0))
        }

        for direction, (orig_proj, att_proj) in projections.items():
            self._save_single_slice(
                orig_proj,
                att_proj,
                save_path / f'{direction}_projection.png',
                alpha
            )

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """归一化图像到[0,1]范围"""
        image_norm = image - image.min()
        if image_norm.max() > 0:
            image_norm = image_norm / image_norm.max()
        return image_norm

    def _save_single_slice(self, original_slice: np.ndarray, attention_slice: np.ndarray,
                           save_path: Path, alpha: float):
        """保存单个切片的注意力可视化"""
        # 获取颜色映射
        cmap = plt.cm.get_cmap(self.cmap)
        attention_color = cmap(attention_slice)[:, :, :3]  # [H, W, 3]

        # 转换原始图像为RGB
        original_rgb = np.stack([original_slice] * 3, axis=-1)  # [H, W, 3]

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

    def create_attention_video(self, attention_dir: str, output_path: str, fps: int = 8):
        """创建注意力切片的视频"""
        attention_path = Path(attention_dir)
        if not attention_path.exists():
            logger.error(f"Directory not found: {attention_dir}")
            return

        # 获取所有overlay图像
        overlay_images = sorted(list(attention_path.glob('slices/*overlay.png')))
        if not overlay_images:
            logger.error("No overlay images found")
            return

        # 读取第一张图像获取尺寸
        first_img = cv2.imread(str(overlay_images[0]))
        if first_img is None:
            logger.error(f"Could not read image: {overlay_images[0]}")
            return

        height, width = first_img.shape[:2]

        # 创建视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            for img_path in tqdm(overlay_images, desc="Creating video", leave=False):
                img = cv2.imread(str(img_path))
                if img is not None:
                    video_writer.write(img)
        finally:
            video_writer.release()

        logger.info(f"Video saved to: {output_path}")

    def __del__(self):
        """析构函数，清理hooks"""
        self._clear_hooks()


def load_model(model_path: str, model_class, device: str = 'cuda', **model_kwargs):
    """
    加载预训练模型 - 修复PyTorch 2.6兼容性问题

    Args:
        model_path: 模型路径
        model_class: 模型类
        device: 设备
        **model_kwargs: 模型初始化参数

    Returns:
        加载的模型
    """
    try:
        model = model_class(**model_kwargs)

        # 修复PyTorch 2.6的weights_only默认值变化问题
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except Exception as e:
            if "weights_only" in str(e):
                logger.warning("尝试使用weights_only=False加载模型...")
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            else:
                raise e

        # 处理不同的checkpoint格式
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 加载模型权重
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logger.warning(f"严格模式加载失败，尝试非严格模式: {e}")
            model.load_state_dict(state_dict, strict=False)

        model.to(device)
        model.eval()

        logger.info(f"模型成功加载自: {model_path}")
        return model

    except FileNotFoundError:
        logger.error(f"模型文件未找到: {model_path}")
        raise
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

def load_h5_image(h5_path: str) -> np.ndarray:
    """加载H5文件中的图像数据"""
    try:
        with h5py.File(h5_path, 'r') as f:
            if 'image' in f:
                image = f['image'][:]
            else:
                # 尝试其他可能的键名
                keys = list(f.keys())
                logger.warning(f"'image' key not found. Available keys: {keys}")
                if keys:
                    image = f[keys[0]][:]
                else:
                    raise ValueError("No data found in H5 file")

        logger.info(f"Loaded image with shape: {image.shape}")
        return image

    except Exception as e:
        logger.error(f"Failed to load H5 file {h5_path}: {e}")
        raise


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int, int] = (160, 160, 128)) -> torch.Tensor:
    """预处理图像"""
    # 转换为张量并添加batch维度
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).float()
    else:
        image_tensor = image.float()

    if len(image_tensor.shape) == 4:  # [C, H, W, D]
        image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W, D]

    # 调整大小
    if image_tensor.shape[2:] != target_size:
        image_tensor = F.interpolate(
            image_tensor,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )

    return image_tensor