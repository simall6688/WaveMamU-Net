import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from module.GDGMamU_Net_ESAACA import GDGMamU_Net
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionVisualizer:
    def __init__(self, model, target_layers, cmap='viridis_r'):  # 默认使用反转的色谱
        """
        初始化注意力可视化器

        :param model: 需要进行可视化的模型
        :param target_layers: 目标层的列表，例如 ['res_attn1.sa', 'res_attn2.sa', 'COBA.esa']
        :param cmap: 用于可视化的颜色映射，默认为'viridis_r'（反转的蓝绿色调，使得高值为黄绿色）
        """
        self.model = model
        self.target_layers = target_layers
        self.cmap = cmap
        self.activations = {}
        self.hook_handles = []
        self._register_hooks()

        # 打印模型所有可用层
        self._print_available_layers()

    def _register_hooks(self):
        """注册钩子来捕获中间层的激活"""

        def get_activation(name):
            def hook(module, input, output):
                # Handle different output types
                if isinstance(output, tuple):
                    # For modules that return multiple outputs, typically we want the first one
                    # which is usually the main feature output
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        if output[0].dim() > 0:
                            self.activations[name] = output[0][0:1].detach()
                        else:
                            self.activations[name] = output[0].detach()
                    else:
                        # If no valid tensor in tuple, skip this activation
                        logger.warning(f"Layer {name} returned tuple without valid tensor as first element")
                        return
                elif isinstance(output, torch.Tensor):
                    # Original tensor handling
                    if output.dim() > 0:
                        self.activations[name] = output[0:1].detach()
                    else:
                        self.activations[name] = output.detach()
                else:
                    # Handle other types if necessary
                    logger.warning(f"Layer {name} returned unexpected type: {type(output)}")
                    return

            return hook

        # Clear previous hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # Register hooks for target layers
        for name in self.target_layers:
            layer = self._get_layer_by_name(name)
            if layer is not None:
                handle = layer.register_forward_hook(get_activation(name))
                self.hook_handles.append(handle)
            else:
                print(f"Warning: Layer {name} not found in the model.")

    def _get_layer_by_name(self, name):
        """通过名称获取模型中的层"""
        submodules = name.split('.')
        current_module = self.model

        for submodule in submodules:
            if hasattr(current_module, submodule):
                current_module = getattr(current_module, submodule)
            else:
                return None

        return current_module

    def _print_available_layers(self):
        """打印模型中所有可用的层，帮助用户选择要可视化的层"""
        print("Available layers in the model:")
        self._print_layers_recursive(self.model, "")

    def _print_layers_recursive(self, module, prefix):
        """递归打印模型中的层"""
        for name, child in module.named_children():
            current_prefix = f"{prefix}.{name}" if prefix else name
            print(f"  {current_prefix}")
            self._print_layers_recursive(child, current_prefix)

    def visualize_attention(self, input_tensor, original_image, save_path, selected_modality=0, alpha=0.7):
        """
        生成注意力可视化

        :param input_tensor: 输入张量 [1, 4, H, W, D]
        :param original_image: 原始图像 [4, H, W, D]
        :param save_path: 保存路径
        :param selected_modality: 选择模态
        :param alpha: 透明度
        """
        # 确保模型处于评估模式
        self.model.eval()

        # 复制输入张量以增加批次大小，避免GroupNorm错误
        # 从 [1, 4, H, W, D] 复制成 [4, 4, H, W, D]
        batch_size = 4  # 或者使用更大的值如8
        input_tensor_batched = input_tensor.repeat(batch_size, 1, 1, 1, 1)

        # 前向传播
        with torch.no_grad():
            _ = self.model(input_tensor_batched)

        # 由于我们复制了批次，激活值也会被复制
        # 需要只取第一个批次的激活值进行可视化
        for layer_name in self.activations:
            activation = self.activations[layer_name]
            if activation.shape[0] == batch_size:
                # 只保留第一个批次的激活
                self.activations[layer_name] = activation[0:1]

        # 创建保存路径
        os.makedirs(save_path, exist_ok=True)

        # 获取原始图像尺寸
        _, H, W, D = original_image.shape

        # 选择要叠加的模态
        selected_image = original_image[selected_modality]  # [H, W, D]

        # 为每一个目标层生成可视化
        for layer_name in self.target_layers:
            if layer_name not in self.activations:
                print(f"Warning: No activation captured for layer {layer_name}")
                continue

            # 获取注意力激活
            attention = self.activations[layer_name]
            print(f"Layer: {layer_name}, Raw activation shape: {attention.shape}")

            # 处理注意力激活，根据类型处理成统一的格式 [B, 1, D, H, W]
            attention_map = self._process_attention_map(attention)
            print(f"After processing: {attention_map.shape}")

            # 注意：由于模型内部使用 [B, C, D, H, W] 格式，但外部使用 [B, C, H, W, D]
            # 所以需要将注意力图转换为外部格式 [B, 1, H, W, D]
            if len(attention_map.shape) == 5:
                attention_map = attention_map.permute(0, 1, 3, 4, 2)

            # 调整注意力图尺寸以匹配原始图像
            attention_resized = F.interpolate(
                attention_map,
                size=(H, W, D),
                mode='trilinear',
                align_corners=False
            )

            # 转换为numpy数组
            attention_np = attention_resized.cpu().numpy()[0, 0]  # [H, W, D]

            # 创建当前层的保存目录
            layer_save_path = os.path.join(save_path, layer_name.replace('.', '_'))
            os.makedirs(layer_save_path, exist_ok=True)

            # 为每个切片生成可视化
            for d in range(D):
                self._save_attention_slice(
                    selected_image[:, :, d],
                    attention_np[:, :, d],
                    os.path.join(layer_save_path, f'slice_{d:03d}.png'),
                    alpha
                )

            # 保存最大投影图
            proj_orig = np.max(selected_image, axis=2)  # [H, W]
            proj_att = np.max(attention_np, axis=2)  # [H, W]
            self._save_attention_slice(
                proj_orig,
                proj_att,
                os.path.join(layer_save_path, 'max_projection.png'),
                alpha
            )

    def _process_attention_map(self, activation):
        """
        处理注意力图

        :param activation: 激活张量
        :return: 处理后的注意力图，统一格式为 [B, 1, D, H, W]
        """
        # 根据不同层的输出特性进行适配

        # 如果是 SpatialAttention 或类似的空间注意力
        if hasattr(activation, 'shape'):
            shape = activation.shape

            # 模型内部使用 [B, C, D, H, W] 格式
            # 但注意力图可能是多种形式

            # 如果是 [B, C, D, H, W] 形式，取通道平均
            if len(shape) == 5 and shape[1] > 1:
                activation = activation.mean(dim=1, keepdim=True)

            # 如果是注意力权重矩阵 [B, N, N]，需要特殊处理
            elif len(shape) == 3:
                B, N, _ = shape
                # 尝试将其重塑为空间特征
                D = int(N ** (1 / 3) + 0.5)  # 四舍五入
                if abs(D ** 3 - N) < 0.1 * N:  # 近似立方
                    activation = activation.mean(dim=2).view(B, 1, D, D, D)
                else:
                    # 无法重塑，直接平均
                    activation = activation.mean(dim=(1, 2)).view(B, 1, 1, 1, 1)
        else:
            # 无法处理的类型
            print(f"Warning: Unexpected activation type: {type(activation)}")
            return torch.zeros((1, 1, 1, 1, 1), device=activation.device)

        # 确保激活为正值
        activation = torch.relu(activation)

        # 归一化到 [0, 1] 范围
        min_val = activation.min()
        max_val = activation.max()
        if max_val > min_val:
            activation = (activation - min_val) / (max_val - min_val + 1e-8)

        return activation

    def _save_attention_slice(self, original_slice, attention_slice, save_path, alpha=0.7):
        """
        将注意力切片叠加到原始图像切片上并保存

        :param original_slice: 原始图像切片 [H, W]
        :param attention_slice: 注意力切片 [H, W]
        :param save_path: 保存路径
        :param alpha: 透明度
        """
        # 归一化原始图像到 [0,1]
        original_norm = original_slice - original_slice.min()
        if original_norm.max() > 0:
            original_norm = original_norm / original_norm.max()

        # 使用指定的颜色映射
        custom_cmap = plt.cm.get_cmap(self.cmap)

        # 使用颜色映射
        attention_color = custom_cmap(attention_slice)[:, :, :3]  # [H, W, 3]

        # 将原始图像转换为 RGB
        original_rgb = np.stack([original_norm] * 3, axis=-1)  # [H, W, 3]

        # 叠加
        overlay = (1 - alpha) * original_rgb + alpha * attention_color
        overlay = np.clip(overlay, 0, 1)

        # 创建并保存图像，包含原始图像、注意力图和叠加图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 原始图像
        axes[0].imshow(original_norm, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 注意力图
        im = axes[1].imshow(attention_slice, cmap=self.cmap, vmin=0, vmax=1)
        axes[1].set_title('Attention Map')
        axes[1].axis('off')

        # 更新 colorbar 标签以反映 viridis 的实际映射
        cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Attention Value\n1.0 = High Focus (Tumor)\n0.0 = Low Focus (Background)',
                       rotation=270, labelpad=20)

        # 叠加图
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        # 单独保存叠加图（与示例图像相似的版本）
        plt.figure(figsize=(8, 8), dpi=100)
        plt.imshow(overlay)
        plt.axis('off')
        plt.tight_layout()
        overlay_path = save_path.replace('.png', '_overlay.png')
        plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
        plt.close()


def load_model(model_path, device='cuda'):
    try:
        model = GDGMamU_Net(4,4)

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


def load_h5_image(h5_path):
    """
    加载 H5 文件中的图像数据

    :param h5_path: H5 文件路径
    :return: 图像数据 (numpy 数组)
    """
    with h5py.File(h5_path, 'r') as f:
        image = f['image'][:]  # [4, H, W, D]
    return image


def preprocess_image(image, target_size=(160, 160, 128)):
    """
    预处理图像，包括调整大小

    :param image: 原始图像 [4, H, W, D]
    :param target_size: 目标大小 (H, W, D)
    :return: 预处理后的张量 [1, 4, H, W, D]
    """
    image = torch.from_numpy(image).unsqueeze(0)  # [1, 4, H, W, D]
    image = F.interpolate(image, size=target_size, mode='trilinear', align_corners=False)
    return image


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_path =args.model_path  # 预训练模型路径

    # 定义需要可视化的层
    target_layers = [
        'fusion_modules.0.output.2',
        'Mamba.mamba.stages.0.blocks.1',
        'Mamba.mamba.stages.0.blocks.0.dwconv1.depth_conv',
        'GDG1.mish'
    ]

    # 加载模型
    model = load_model(model_path, device)

    # 初始化注意力可视化器 - 使用反转的蓝绿色调
    visualizer = AttentionVisualizer(model, target_layers, cmap='viridis_r')

    # 读取 inference.txt 中的 H5 文件列表
    inference_file = args.inference_file
    with open(inference_file, 'r') as f:
        h5_files = f.read().splitlines()

    # 设置要处理的样本数量和模态
    num_samples = 5  # 处理前5个样本
    modalities = {
        1: 'T1ce',
    }

    # 对每个文件生成注意力图
    for h5_file in tqdm(h5_files[:num_samples], desc='Processing H5 files'):
        h5_path = os.path.join('../dataset_output', 'dataset', h5_file)
        if not os.path.exists(h5_path):
            print(f"文件未找到: {h5_path}")
            continue

        # 加载图像
        image = load_h5_image(h5_path)  # [4, H, W, D]
        input_tensor = preprocess_image(image).to(device)  # [1, 4, H, W, D]

        # 生成注意力可视化
        case_name = os.path.splitext(os.path.basename(h5_file))[0]

        # 对每个模态生成可视化（可选择特定模态）
        for modality_idx, modality_name in modalities.items():
            save_path = os.path.join('attention_results', case_name, modality_name)

            try:
                visualizer.visualize_attention(
                    input_tensor,
                    image,
                    save_path,
                    selected_modality=modality_idx,
                    alpha=0.7
                )
                print(f"成功生成 {case_name} 的 {modality_name} 模态注意力可视化")
            except Exception as e:
                print(f"生成 {case_name} 的 {modality_name} 模态注意力可视化失败: {e}")
                import traceback
                traceback.print_exc()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize attention maps from a neural network model')
    parser.add_argument('--model_path', type=str, default='../results/best_model_WT0.879_ET0.809_TC0.851_AVG0.846.pth', help='Path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../dataset_output/dataset',
                        help='Directory containing H5 data files')
    parser.add_argument('--inference_file', type=str, default='../dataset_output/inference.txt',
                        help='File containing list of H5 files')
    parser.add_argument('--output_dir', type=str, default='attention_results',
                        help='Directory to save attention visualizations')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples to process')
    parser.add_argument('--cmap', type=str, default='viridis',  # 默认使用反转的色谱
                        help='Colormap for attention visualization (e.g., viridis_r, jet_r, coolwarm)')
    parser.add_argument('--alpha', type=float, default=0.7, help='Transparency of the attention overlay (0-1)')
    parser.add_argument('--list_layers', action='store_true', help='List all available layers and exit')
    parser.add_argument('--modality', type=int, default=-1, help='Specific modality to visualize (-1 for all)')
    parser.add_argument('--specific_slices', type=str, default='',
                        help='Comma-separated list of specific slices to visualize (empty for all)')

    args = parser.parse_args()

    # 执行主函数
    if args.list_layers:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = load_model(args.model_path, device)
        # 临时创建可视化器，只为了打印层
        visualizer = AttentionVisualizer(model, [])
        exit(0)

    main()