# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import h5py
# from scipy.ndimage import gaussian_filter, median_filter
# from skimage.morphology import closing, opening, ball
# import logging
# import json
# from tqdm import tqdm
# import gc
# import random
#
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# """生成完整的所有图片"""
#
# def set_reproducible_seed(seed=42):
#     """
#     设置所有随机种子以确保结果可重复
#
#     Args:
#         seed: 随机种子值
#     """
#     logger.info(f"🌱 设置随机种子: {seed}")
#
#     # Python 内置随机数生成器
#     random.seed(seed)
#
#     # NumPy 随机数生成器
#     np.random.seed(seed)
#
#     # PyTorch 随机数生成器 (CPU)
#     torch.manual_seed(seed)
#
#     # PyTorch 随机数生成器 (GPU)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)  # 多GPU情况
#
#     # 确保cuDNN使用确定性算法
#     torch.backends.cudnn.deterministic = True
#
#     # 禁用cuDNN的benchmark功能 (可能会降低性能但确保可重复性)
#     torch.backends.cudnn.benchmark = False
#
#     # 设置PyTorch使用确定性算法 (PyTorch 1.8+)
#     try:
#         torch.use_deterministic_algorithms(True)
#         logger.info("✅ 启用PyTorch确定性算法")
#     except AttributeError:
#         logger.warning("⚠️  PyTorch版本不支持use_deterministic_algorithms")
#
#     # 设置环境变量 (CUDA 10.2+)
#     os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#     os.environ['PYTHONHASHSEED'] = str(seed)
#
#     logger.info("✅ 随机种子设置完成，结果将是可重复的")
#
#
# class CompleteMambaAttentionVisualizer:
#     def __init__(self, model, target_layers, cmap='jet'):
#         """完整的Mamba可视化器"""
#         self.model = model
#         self.target_layers = target_layers
#         self.cmap = cmap
#         self.activations = {}
#         self.hook_handles = []
#         self.input_tensor_shape = None
#
#         # 添加切片保存配置
#         self.save_batch_size = 10  # 每批保存的切片数（内存优化）
#
#         # Mamba特殊层的标识
#         self.mamba_layers = [
#             'Mamba.mamba.stages.0.blocks.0',
#             'Mamba.mamba.stages.0.blocks.1',
#             'Mamba.mamba.stages.1.blocks.0',
#             'Mamba.mamba.stages.1.blocks.1',
#             'Mamba.mamba.stages.2.blocks.0',
#             'Mamba.mamba.stages.2.blocks.1',
#             'Mamba.mamba.feature_enhance.0.2',
#             'Mamba.mamba.feature_enhance.1.2',
#             'Mamba.mamba.feature_enhance.2.2'
#         ]
#
#         self._register_hooks()
#
#     def _register_hooks(self):
#         """注册钩子来捕获中间层的激活"""
#
#         def get_activation(name):
#             def hook(module, input, output):
#                 if isinstance(output, tuple):
#                     if len(output) > 0 and isinstance(output[0], torch.Tensor):
#                         self.activations[name] = output[0][0:1].detach()
#                 elif isinstance(output, torch.Tensor):
#                     self.activations[name] = output[0:1].detach()
#
#             return hook
#
#         for handle in self.hook_handles:
#             handle.remove()
#         self.hook_handles = []
#
#         for name in self.target_layers:
#             layer = self._get_layer_by_name(name)
#             if layer is not None:
#                 handle = layer.register_forward_hook(get_activation(name))
#                 self.hook_handles.append(handle)
#                 logger.info(f"Registered hook for: {name}")
#
#     def _get_layer_by_name(self, name):
#         """通过名称获取模型中的层"""
#         submodules = name.split('.')
#         current_module = self.model
#
#         for submodule in submodules:
#             if hasattr(current_module, submodule):
#                 current_module = getattr(current_module, submodule)
#             else:
#                 return None
#
#         return current_module
#
#     def _fix_dimension_order(self, tensor):
#         """修复张量的维度顺序"""
#         if len(tensor.shape) == 5:  # [B, C, ?, ?, ?]
#             B, C = tensor.shape[:2]
#             remaining_dims = tensor.shape[2:]
#
#             # 找最小维度（通常是深度维度）
#             min_dim_idx = np.argmin(remaining_dims)
#
#             if min_dim_idx == 0:  # [B, C, D, H, W]
#                 tensor = tensor.permute(0, 1, 3, 4, 2)  # -> [B, C, H, W, D]
#
#         return tensor
#
#     def _process_mamba_attention(self, activation, layer_name):
#         """专门处理Mamba激活的函数"""
#         # 确保激活值形状正确
#         if len(activation.shape) == 5:
#             activation = self._fix_dimension_order(activation)
#             if activation.shape[1] > 1:
#                 activation = activation.mean(dim=1, keepdim=True)
#         elif len(activation.shape) == 3:
#             B, N, _ = activation.shape
#             D = int(round(N ** (1 / 3)))
#             if abs(D ** 3 - N) < 0.1 * N:
#                 activation = activation.mean(dim=2).view(B, 1, D, D, D)
#                 activation = activation.permute(0, 1, 3, 4, 2)  # -> [B, 1, H, W, D]
#             else:
#                 logger.warning(f"Cannot reshape {N} into cubic dimensions")
#                 return torch.zeros((1, 1, 1, 1, 1))
#
#         # 转换为numpy进行处理
#         attention_np = activation.cpu().numpy()[0, 0]  # [H, W, D]
#
#         # 噪声抑制处理
#         threshold = np.percentile(attention_np, 80)
#         attention_np[attention_np < threshold] = 0
#
#         # 高斯平滑
#         attention_np = gaussian_filter(attention_np, sigma=1.5)
#
#         # 形态学操作
#         if attention_np.ndim == 3:
#             struct_elem = ball(1)
#             attention_np = opening(attention_np, struct_elem)
#             attention_np = closing(attention_np, struct_elem)
#
#         # 归一化
#         if attention_np.max() > attention_np.min():
#             attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min())
#
#         # 转回tensor
#         processed = torch.from_numpy(attention_np).unsqueeze(0).unsqueeze(0).float()
#
#         return processed
#
#     def _process_attention_map(self, activation, layer_name):
#         """处理注意力图"""
#         is_mamba_layer = any(mamba_id in layer_name for mamba_id in self.mamba_layers)
#
#         if is_mamba_layer:
#             logger.info(f"Using Mamba-specific processing for layer: {layer_name}")
#             processed = self._process_mamba_attention(activation, layer_name)
#         else:
#             processed = self._fix_dimension_order(activation)
#
#             if len(processed.shape) == 5 and processed.shape[1] > 1:
#                 processed = processed.mean(dim=1, keepdim=True)
#
#             processed = torch.relu(processed)
#
#             # 归一化
#             min_val = processed.min()
#             max_val = processed.max()
#             if max_val > min_val:
#                 processed = (processed - min_val) / (max_val - min_val + 1e-8)
#
#         return processed
#
#     def visualize_attention(self, input_tensor, original_image, save_path, selected_modality=0, alpha=0.5):
#         """生成完整的注意力可视化"""
#         self.model.eval()
#
#         # 记录输入形状
#         self.input_tensor_shape = input_tensor.shape
#         logger.info(f"Input tensor shape: {self.input_tensor_shape}")
#
#         # 批处理
#         batch_size = 4
#         input_tensor_batched = input_tensor.repeat(batch_size, 1, 1, 1, 1)
#
#         # 前向传播
#         with torch.no_grad():
#             _ = self.model(input_tensor_batched)
#
#         # 处理激活值
#         for layer_name in self.activations:
#             activation = self.activations[layer_name]
#             if activation.shape[0] == batch_size:
#                 self.activations[layer_name] = activation[0:1]
#
#         os.makedirs(save_path, exist_ok=True)
#
#         # 获取原始图像尺寸
#         _, H_orig, W_orig, D_orig = original_image.shape
#         selected_image = original_image[selected_modality]  # [H, W, D]
#
#         # 获取模型输入尺寸
#         _, _, H_model, W_model, D_model = self.input_tensor_shape
#
#         logger.info(f"Original image shape: {H_orig}x{W_orig}x{D_orig}")
#         logger.info(f"Model input shape: {H_model}x{W_model}x{D_model}")
#
#         # 处理每个目标层
#         for layer_name in self.target_layers:
#             if layer_name not in self.activations:
#                 logger.warning(f"No activation for layer: {layer_name}")
#                 continue
#
#             attention = self.activations[layer_name]
#             logger.info(f"Processing layer: {layer_name}, shape: {attention.shape}")
#
#             # 处理注意力激活
#             attention_map = self._process_attention_map(attention, layer_name)
#
#             # 确保attention_map是 [B, C, H, W, D] 格式
#             if len(attention_map.shape) == 5:
#                 B, C, H_att, W_att, D_att = attention_map.shape
#                 logger.info(f"Attention map shape: {H_att}x{W_att}x{D_att}")
#
#                 # 插值到原始尺寸
#                 attention_resized = F.interpolate(
#                     attention_map,
#                     size=(H_orig, W_orig, D_orig),
#                     mode='trilinear',
#                     align_corners=True
#                 )
#             else:
#                 logger.error(f"Unexpected attention map shape: {attention_map.shape}")
#                 continue
#
#             # 转换为numpy
#             attention_np = attention_resized.cpu().numpy()[0, 0]  # [H, W, D]
#
#             # 验证形状匹配
#             assert attention_np.shape == selected_image.shape, \
#                 f"Shape mismatch: attention {attention_np.shape} vs image {selected_image.shape}"
#
#             # 保存可视化
#             layer_save_path = os.path.join(save_path, layer_name.replace('.', '_'))
#             os.makedirs(layer_save_path, exist_ok=True)
#
#             # 生成完整的可视化（包括切片和投影）
#             self._generate_complete_visualization(
#                 selected_image,
#                 attention_np,
#                 layer_save_path,
#                 layer_name,
#                 alpha
#             )
#
#     def _generate_complete_visualization(self, original_image, attention_map, save_path, layer_name, alpha=0.5):
#         """生成完整的可视化结果，包括代表性切片、所有切片和投影"""
#         H, W, D = original_image.shape
#
#         # 1. 首先生成代表性切片的可视化
#         self._generate_representative_slices(original_image, attention_map, save_path, layer_name, alpha)
#
#         # 2. 【新增】保存所有切片的可视化
#         self._save_all_slices_with_overlay(original_image, attention_map, save_path, layer_name, alpha)
#
#         # 3. 生成三个方向的投影
#         self._generate_3d_projections(original_image, attention_map, save_path, layer_name, alpha)
#
#         # 4. 保存单独的轴向MIP
#         self._save_single_axial_mip(original_image, attention_map, save_path, layer_name, alpha)
#
#         # 5. 保存统计信息
#         self._save_attention_statistics(attention_map, save_path, layer_name)
#
#     def _save_all_slices_with_overlay(self, original_image, attention_map, save_path, layer_name, alpha=0.5):
#         """
#         【新增方法】保存所有切片的注意力叠加图像
#         参考注释代码的save_all_slices_with_overlay方法，但保持新代码的防错位机制
#
#         Args:
#             original_image: 原始图像 [H, W, D]
#             attention_map: 注意力图 [H, W, D]
#             save_path: 保存路径
#             layer_name: 层名称
#             alpha: 透明度
#         """
#         H, W, D = original_image.shape
#
#         # 创建所有切片的子目录
#         all_slices_path = os.path.join(save_path, 'all_slices')
#         os.makedirs(all_slices_path, exist_ok=True)
#
#         print(f"\n📂 保存所有 {D} 个切片到: {all_slices_path}")
#
#         # 分批保存切片以优化内存使用
#         for batch_start in tqdm(range(0, D, self.save_batch_size), desc=f'保存{layer_name}切片批次'):
#             batch_end = min(batch_start + self.save_batch_size, D)
#
#             # 创建当前批次的图形
#             batch_size = batch_end - batch_start
#             fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
#
#             if batch_size == 1:
#                 axes = axes.reshape(1, -1)
#
#             for i, d in enumerate(range(batch_start, batch_end)):
#                 orig_slice = original_image[:, :, d]
#                 attention_slice = attention_map[:, :, d]
#
#                 # 原始图像
#                 axes[i, 0].imshow(orig_slice, cmap='gray')
#                 axes[i, 0].set_title(f'Original - Slice {d}')
#                 axes[i, 0].axis('off')
#
#                 # 注意力热图
#                 im = axes[i, 1].imshow(attention_slice, cmap=self.cmap, vmin=0, vmax=1)
#                 axes[i, 1].set_title(f'Attention - Slice {d}')
#                 axes[i, 1].axis('off')
#
#                 # 叠加图
#                 overlay = self._create_overlay(orig_slice, attention_slice, alpha)
#                 axes[i, 2].imshow(overlay)
#                 axes[i, 2].set_title(f'Overlay - Slice {d}')
#                 axes[i, 2].axis('off')
#
#             # 保存当前批次
#             plt.tight_layout()
#             plt.savefig(os.path.join(all_slices_path, f'slices_{batch_start:03d}-{batch_end - 1:03d}.png'),
#                         dpi=150, bbox_inches='tight')
#             plt.close()
#
#             # 同时保存单独的切片文件
#             for d in range(batch_start, batch_end):
#                 orig_slice = original_image[:, :, d]
#                 attention_slice = attention_map[:, :, d]
#                 overlay = self._create_overlay(orig_slice, attention_slice, alpha)
#
#                 # 保存单个叠加图像
#                 plt.figure(figsize=(6, 6))
#                 plt.imshow(overlay)
#                 plt.axis('off')
#                 plt.title(f'Slice {d} - {layer_name}')
#                 plt.savefig(os.path.join(all_slices_path, f'slice_{d:03d}_overlay.png'),
#                             bbox_inches='tight', pad_inches=0, dpi=150)
#                 plt.close()
#
#             # 清理内存
#             gc.collect()
#
#         print(f"✅ 完成保存所有 {D} 个切片")
#
#     def _generate_representative_slices(self, original_image, attention_map, save_path, layer_name, alpha=0.5):
#         """生成代表性切片的可视化"""
#         H, W, D = original_image.shape
#
#         # 选择代表性切片（肿瘤区域最大的切片）
#         tumor_volume_per_slice = []
#         for d in range(D):
#             # 假设注意力值高的地方是肿瘤（根据您的colormap）
#             tumor_volume = np.sum(attention_map[:, :, d] > 0.3)
#             tumor_volume_per_slice.append((d, tumor_volume))
#
#         # 选择肿瘤最大的8个切片
#         tumor_volume_per_slice.sort(key=lambda x: x[1], reverse=True)
#         representative_slices = [idx for idx, _ in tumor_volume_per_slice[:8]]
#
#         # 创建可视化
#         fig, axes = plt.subplots(4, 4, figsize=(16, 16))
#         axes = axes.flatten()
#
#         for i in range(16):
#             if i < len(representative_slices):
#                 d = representative_slices[i // 2]
#
#                 if i % 2 == 0:
#                     # 原始图像
#                     axes[i].imshow(original_image[:, :, d], cmap='gray')
#                     axes[i].set_title(f'Slice {d} - Original', fontsize=10)
#                 else:
#                     # 注意力叠加
#                     overlay = self._create_overlay(original_image[:, :, d], attention_map[:, :, d], alpha)
#                     axes[i].imshow(overlay)
#                     axes[i].set_title(f'Slice {d} - Attention', fontsize=10)
#             else:
#                 axes[i].axis('off')
#
#             axes[i].axis('off')
#
#         plt.suptitle(f'Layer: {layer_name} - Representative Slices', fontsize=16)
#         plt.tight_layout()
#
#         slice_path = os.path.join(save_path, 'representative_slices.png')
#         plt.savefig(slice_path, dpi=150, bbox_inches='tight')
#         plt.close()
#         logger.info(f"Saved representative slices: {slice_path}")
#
#     def _generate_3d_projections(self, original_image, attention_map, save_path, layer_name, alpha=0.5):
#         """生成三个方向的投影"""
#         fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
#
#         projections = [
#             ('Axial MIP', lambda x: np.max(x, axis=2)),
#             ('Coronal MIP', lambda x: np.max(x, axis=1)),
#             ('Sagittal MIP', lambda x: np.max(x, axis=0))
#         ]
#
#         for idx, (proj_name, proj_func) in enumerate(projections):
#             # 原始图像投影
#             orig_proj = proj_func(original_image)
#             axes[idx, 0].imshow(orig_proj, cmap='gray')
#             axes[idx, 0].set_title(f'{proj_name} - Original')
#             axes[idx, 0].axis('off')
#
#             # 注意力图投影
#             att_proj = proj_func(attention_map)
#             im = axes[idx, 1].imshow(att_proj, cmap=self.cmap, vmin=0, vmax=1)
#             axes[idx, 1].set_title(f'{proj_name} - Attention')
#             axes[idx, 1].axis('off')
#
#             # 叠加投影
#             overlay_proj = self._create_overlay(orig_proj, att_proj, alpha)
#             axes[idx, 2].imshow(overlay_proj)
#             axes[idx, 2].set_title(f'{proj_name} - Overlay')
#             axes[idx, 2].axis('off')
#
#         # 添加colorbar
#         cbar = fig.colorbar(im, ax=axes[:, 1], fraction=0.046, pad=0.04)
#         cbar.set_label('Attention Value', rotation=270, labelpad=20)
#
#         plt.suptitle(f'Layer: {layer_name} - 3D Projections', fontsize=16)
#
#         proj_path = os.path.join(save_path, '3d_projections.png')
#         plt.savefig(proj_path, dpi=150, bbox_inches='tight')
#         plt.close()
#         logger.info(f"Saved 3D projections: {proj_path}")
#
#     def _save_single_axial_mip(self, original_image, attention_map, save_path, layer_name, alpha=0.5):
#         """保存单独的轴向MIP叠加图"""
#         # 计算轴向最大投影
#         orig_mip = np.max(original_image, axis=2)
#         att_mip = np.max(attention_map, axis=2)
#
#         # 创建图形
#         fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
#         # 原始图像
#         axes[0].imshow(orig_mip, cmap='gray')
#         axes[0].set_title('Original MIP')
#         axes[0].axis('off')
#
#         # 注意力图
#         im = axes[1].imshow(att_mip, cmap=self.cmap, vmin=0, vmax=1)
#         axes[1].set_title('Attention MIP')
#         axes[1].axis('off')
#
#         # 叠加
#         overlay = self._create_overlay(orig_mip, att_mip, alpha)
#         axes[2].imshow(overlay)
#         axes[2].set_title('Overlay MIP')
#         axes[2].axis('off')
#
#         # 添加colorbar
#         cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
#         cbar.set_label('Attention Value', rotation=270, labelpad=15)
#
#         plt.tight_layout()
#
#         mip_path = os.path.join(save_path, 'axial_mip_comparison.png')
#         plt.savefig(mip_path, dpi=300, bbox_inches='tight')
#         plt.close()
#
#         # 也保存单独的叠加图
#         plt.figure(figsize=(8, 8))
#         plt.imshow(overlay)
#         plt.axis('off')
#         plt.title(f'Axial MIP Overlay - {layer_name}')
#
#         overlay_path = os.path.join(save_path, 'axial_mip_overlay_only.png')
#         plt.savefig(overlay_path, dpi=300, bbox_inches='tight', pad_inches=0)
#         plt.close()
#
#         logger.info(f"Saved axial MIP: {mip_path}")
#
#     def _create_overlay(self, original_slice, attention_slice, alpha=0.5):
#         """创建注意力叠加图"""
#         # 归一化原始图像
#         original_norm = original_slice - original_slice.min()
#         if original_norm.max() > 0:
#             original_norm = original_norm / original_norm.max()
#
#         # 使用colormap（兼容新版matplotlib）
#         try:
#             # 新版本方式
#             custom_cmap = plt.colormaps[self.cmap]
#         except:
#             # 旧版本方式
#             custom_cmap = plt.cm.get_cmap(self.cmap)
#
#         attention_color = custom_cmap(attention_slice)[:, :, :3]
#
#         # 将原始图像转换为RGB
#         original_rgb = np.stack([original_norm] * 3, axis=-1)
#
#         # 叠加
#         overlay = (1 - alpha) * original_rgb + alpha * attention_color
#         overlay = np.clip(overlay, 0, 1)
#
#         return overlay
#
#     def _save_attention_statistics(self, attention_map, save_path, layer_name):
#         """保存注意力图的统计信息"""
#         stats = {
#             'layer_name': layer_name,
#             'shape': list(attention_map.shape),
#             'mean': float(np.mean(attention_map)),
#             'std': float(np.std(attention_map)),
#             'min': float(np.min(attention_map)),
#             'max': float(np.max(attention_map)),
#             'high_attention_ratio': float(np.sum(attention_map > 0.5) / attention_map.size),
#             'very_high_attention_ratio': float(np.sum(attention_map > 0.7) / attention_map.size)
#         }
#
#         # 保存JSON统计信息
#         json_path = os.path.join(save_path, 'statistics.json')
#         with open(json_path, 'w') as f:
#             json.dump(stats, f, indent=4)
#
#         # 生成直方图
#         plt.figure(figsize=(10, 6))
#         plt.hist(attention_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
#         plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
#         plt.xlabel('Attention Value')
#         plt.ylabel('Frequency')
#         plt.title(f'Attention Value Distribution - {layer_name}')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#
#         hist_path = os.path.join(save_path, 'attention_histogram.png')
#         plt.savefig(hist_path, dpi=150, bbox_inches='tight')
#         plt.close()
#
#         logger.info(f"Saved statistics: {json_path}")
#
#
# def preprocess_image_fixed(image, target_size=(160, 160, 128)):
#     """修复的预处理函数"""
#     image_tensor = torch.from_numpy(image.copy()).float()
#     image_tensor = image_tensor.unsqueeze(0)  # [1, 4, H, W, D]
#
#     image_resized = F.interpolate(
#         image_tensor,
#         size=target_size,
#         mode='trilinear',
#         align_corners=True
#     )
#
#     return image_resized
#
#
# def load_model(model_path, device='cuda'):
#     """加载模型"""
#     from module.GDGMamU_Net_ESAACA import GDGMamU_Net
#
#     model = GDGMamU_Net(4, 4)
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#
#     if 'model' in checkpoint:
#         state_dict = checkpoint['model']
#     elif 'state_dict' in checkpoint:
#         state_dict = checkpoint['state_dict']
#     else:
#         state_dict = checkpoint
#
#     model.load_state_dict(state_dict, strict=False)
#     model.to(device)
#     model.eval()
#
#     return model
#
#
# def main():
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
#     # 定义目标层
#     target_layers = [
#         'Mamba.mamba.stages.0.blocks.0',
#         'Mamba.mamba.stages.0.blocks.1',
#         'fusion_modules.0.aca_mamba',
#     ]
#
#     # 加载模型
#     model = load_model(args.model_path, device)
#
#     # 初始化可视化器
#     visualizer = CompleteMambaAttentionVisualizer(model, target_layers, cmap=args.cmap)
#
#     # 读取文件列表
#     with open(args.inference_file, 'r') as f:
#         h5_files = f.read().splitlines()
#
#     # 处理文件
#     for idx, h5_file in enumerate(h5_files[:args.num_samples]):
#         print(f"\n{'=' * 50}")
#         print(f"Processing {idx + 1}/{args.num_samples}: {h5_file}")
#         print(f"{'=' * 50}")
#
#         h5_path = os.path.join(args.data_dir, h5_file)
#         if not os.path.exists(h5_path):
#             continue
#
#         # 加载图像
#         with h5py.File(h5_path, 'r') as f:
#             image = f['image'][:]  # [4, H, W, D]
#
#         # 使用修复的预处理函数
#         input_tensor = preprocess_image_fixed(image).to(device)
#
#         # 生成可视化
#         case_name = os.path.splitext(os.path.basename(h5_file))[0]
#         save_path = os.path.join(args.output_dir, case_name)
#
#         try:
#             visualizer.visualize_attention(
#                 input_tensor,
#                 image,
#                 save_path,
#                 selected_modality=1,  # T1ce
#                 alpha=args.alpha
#             )
#             print(f"✅ 成功生成 {case_name} 的完整注意力可视化")
#             print(f"   - 代表性切片: representative_slices.png")
#             print(f"   - 所有切片: all_slices/ 目录")
#             print(f"   - 3D投影: 3d_projections.png")
#             print(f"   - 轴向MIP: axial_mip_comparison.png")
#             print(f"   - 统计信息: statistics.json")
#         except Exception as e:
#             print(f"❌ 处理 {case_name} 失败: {e}")
#             import traceback
#             traceback.print_exc()
#
#
# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_path', type=str,
#                         default='../results/best_model_WT0.879_ET0.809_TC0.851_AVG0.846.pth',
#                         help='Path to the model checkpoint')
#     parser.add_argument('--data_dir', type=str, default='../dataset_output/dataset')
#     parser.add_argument('--inference_file', type=str, default='../dataset_output/inference.txt')
#     parser.add_argument('--output_dir', type=str, default='complete_attention_results')
#     parser.add_argument('--num_samples', type=int, default=5)
#     parser.add_argument('--cmap', type=str, default='jet')
#     parser.add_argument('--alpha', type=float, default=0.5)
#     parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
#
#     args = parser.parse_args()
#
#     # 使用参数中的随机种子
#     set_reproducible_seed(seed=args.seed)
#
#     main()

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.ndimage import gaussian_filter, median_filter
from skimage.morphology import closing, opening, ball
import logging
import json
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_reproducible_seed(seed=42):
    """
    设置所有随机种子以确保结果可重复

    Args:
        seed: 随机种子值
    """
    logger.info(f"🌱 设置随机种子: {seed}")

    # Python 内置随机数生成器
    random.seed(seed)

    # NumPy 随机数生成器
    np.random.seed(seed)

    # PyTorch 随机数生成器 (CPU)
    torch.manual_seed(seed)

    # PyTorch 随机数生成器 (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况

    # 确保cuDNN使用确定性算法
    torch.backends.cudnn.deterministic = True

    # 禁用cuDNN的benchmark功能 (可能会降低性能但确保可重复性)
    torch.backends.cudnn.benchmark = False

    # 设置PyTorch使用确定性算法 (PyTorch 1.8+)
    try:
        torch.use_deterministic_algorithms(True)
        logger.info("✅ 启用PyTorch确定性算法")
    except AttributeError:
        logger.warning("⚠️  PyTorch版本不支持use_deterministic_algorithms")

    # 设置环境变量 (CUDA 10.2+)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

    logger.info("✅ 随机种子设置完成，结果将是可重复的")


class CompleteMambaAttentionVisualizer:
    def __init__(self, model, target_layers, cmap='jet'):
        """完整的Mamba可视化器"""
        self.model = model
        self.target_layers = target_layers
        self.cmap = cmap
        self.activations = {}
        self.hook_handles = []
        self.input_tensor_shape = None

        # Mamba特殊层的标识
        self.mamba_layers = [
            'Mamba.mamba.stages.0.blocks.0',
            'Mamba.mamba.stages.0.blocks.1',
            'Mamba.mamba.stages.1.blocks.0',
            'Mamba.mamba.stages.1.blocks.1',
            'Mamba.mamba.stages.2.blocks.0',
            'Mamba.mamba.stages.2.blocks.1',
            'Mamba.mamba.feature_enhance.0.2',
            'Mamba.mamba.feature_enhance.1.2',
            'Mamba.mamba.feature_enhance.2.2'
        ]

        self._register_hooks()

    def _register_hooks(self):
        """注册钩子来捕获中间层的激活"""

        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        self.activations[name] = output[0][0:1].detach()
                elif isinstance(output, torch.Tensor):
                    self.activations[name] = output[0:1].detach()

            return hook

        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        for name in self.target_layers:
            layer = self._get_layer_by_name(name)
            if layer is not None:
                handle = layer.register_forward_hook(get_activation(name))
                self.hook_handles.append(handle)
                logger.info(f"Registered hook for: {name}")

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

    def _fix_dimension_order(self, tensor):
        """修复张量的维度顺序"""
        if len(tensor.shape) == 5:  # [B, C, ?, ?, ?]
            B, C = tensor.shape[:2]
            remaining_dims = tensor.shape[2:]

            # 找最小维度（通常是深度维度）
            min_dim_idx = np.argmin(remaining_dims)

            if min_dim_idx == 0:  # [B, C, D, H, W]
                tensor = tensor.permute(0, 1, 3, 4, 2)  # -> [B, C, H, W, D]

        return tensor

    def _process_mamba_attention(self, activation, layer_name):
        """专门处理Mamba激活的函数"""
        # 确保激活值形状正确
        if len(activation.shape) == 5:
            activation = self._fix_dimension_order(activation)
            if activation.shape[1] > 1:
                activation = activation.mean(dim=1, keepdim=True)
        elif len(activation.shape) == 3:
            B, N, _ = activation.shape
            D = int(round(N ** (1 / 3)))
            if abs(D ** 3 - N) < 0.1 * N:
                activation = activation.mean(dim=2).view(B, 1, D, D, D)
                activation = activation.permute(0, 1, 3, 4, 2)  # -> [B, 1, H, W, D]
            else:
                logger.warning(f"Cannot reshape {N} into cubic dimensions")
                return torch.zeros((1, 1, 1, 1, 1))

        # 转换为numpy进行处理
        attention_np = activation.cpu().numpy()[0, 0]  # [H, W, D]

        # 噪声抑制处理
        threshold = np.percentile(attention_np, 80)
        attention_np[attention_np < threshold] = 0

        # 高斯平滑
        attention_np = gaussian_filter(attention_np, sigma=1.5)

        # 形态学操作
        if attention_np.ndim == 3:
            struct_elem = ball(1)
            attention_np = opening(attention_np, struct_elem)
            attention_np = closing(attention_np, struct_elem)

        # 归一化
        if attention_np.max() > attention_np.min():
            attention_np = (attention_np - attention_np.min()) / (attention_np.max() - attention_np.min())

        # 转回tensor
        processed = torch.from_numpy(attention_np).unsqueeze(0).unsqueeze(0).float()

        return processed

    def _process_attention_map(self, activation, layer_name):
        """处理注意力图"""
        is_mamba_layer = any(mamba_id in layer_name for mamba_id in self.mamba_layers)

        if is_mamba_layer:
            logger.info(f"Using Mamba-specific processing for layer: {layer_name}")
            processed = self._process_mamba_attention(activation, layer_name)
        else:
            processed = self._fix_dimension_order(activation)

            if len(processed.shape) == 5 and processed.shape[1] > 1:
                processed = processed.mean(dim=1, keepdim=True)

            processed = torch.relu(processed)

            # 归一化
            min_val = processed.min()
            max_val = processed.max()
            if max_val > min_val:
                processed = (processed - min_val) / (max_val - min_val + 1e-8)

        return processed

    def visualize_attention(self, input_tensor, original_image, save_path, selected_modality=0, alpha=0.5):
        """生成完整的注意力可视化"""
        self.model.eval()

        # 记录输入形状
        self.input_tensor_shape = input_tensor.shape
        logger.info(f"Input tensor shape: {self.input_tensor_shape}")

        # 批处理
        batch_size = 4
        input_tensor_batched = input_tensor.repeat(batch_size, 1, 1, 1, 1)

        # 前向传播
        with torch.no_grad():
            _ = self.model(input_tensor_batched)

        # 处理激活值
        for layer_name in self.activations:
            activation = self.activations[layer_name]
            if activation.shape[0] == batch_size:
                self.activations[layer_name] = activation[0:1]

        os.makedirs(save_path, exist_ok=True)

        # 获取原始图像尺寸
        _, H_orig, W_orig, D_orig = original_image.shape
        selected_image = original_image[selected_modality]  # [H, W, D]

        # 获取模型输入尺寸
        _, _, H_model, W_model, D_model = self.input_tensor_shape

        logger.info(f"Original image shape: {H_orig}x{W_orig}x{D_orig}")
        logger.info(f"Model input shape: {H_model}x{W_model}x{D_model}")

        # 处理每个目标层
        for layer_name in self.target_layers:
            if layer_name not in self.activations:
                logger.warning(f"No activation for layer: {layer_name}")
                continue

            attention = self.activations[layer_name]
            logger.info(f"Processing layer: {layer_name}, shape: {attention.shape}")

            # 处理注意力激活
            attention_map = self._process_attention_map(attention, layer_name)

            # 确保attention_map是 [B, C, H, W, D] 格式
            if len(attention_map.shape) == 5:
                B, C, H_att, W_att, D_att = attention_map.shape
                logger.info(f"Attention map shape: {H_att}x{W_att}x{D_att}")

                # 插值到原始尺寸
                attention_resized = F.interpolate(
                    attention_map,
                    size=(H_orig, W_orig, D_orig),
                    mode='trilinear',
                    align_corners=True
                )
            else:
                logger.error(f"Unexpected attention map shape: {attention_map.shape}")
                continue

            # 转换为numpy
            attention_np = attention_resized.cpu().numpy()[0, 0]  # [H, W, D]

            # 验证形状匹配
            assert attention_np.shape == selected_image.shape, \
                f"Shape mismatch: attention {attention_np.shape} vs image {selected_image.shape}"

            # 保存可视化
            layer_save_path = os.path.join(save_path, layer_name.replace('.', '_'))
            os.makedirs(layer_save_path, exist_ok=True)

            # 生成完整的可视化（包括切片和投影）
            self._generate_complete_visualization(
                selected_image,
                attention_np,
                layer_save_path,
                layer_name,
                alpha
            )

    def _generate_complete_visualization(self, original_image, attention_map, save_path, layer_name, alpha=0.5):
        """生成完整的可视化结果，包括代表性切片和投影"""
        H, W, D = original_image.shape

        # 1. 首先生成代表性切片的可视化
        self._generate_representative_slices(original_image, attention_map, save_path, layer_name, alpha)

        # 2. 生成三个方向的投影
        self._generate_3d_projections(original_image, attention_map, save_path, layer_name, alpha)

        # 3. 保存单独的轴向MIP
        self._save_single_axial_mip(original_image, attention_map, save_path, layer_name, alpha)

        # 4. 保存统计信息
        self._save_attention_statistics(attention_map, save_path, layer_name)

    def _generate_representative_slices(self, original_image, attention_map, save_path, layer_name, alpha=0.5):
        """生成代表性切片的可视化"""
        H, W, D = original_image.shape

        # 选择代表性切片（肿瘤区域最大的切片）
        tumor_volume_per_slice = []
        for d in range(D):
            # 假设注意力值高的地方是肿瘤（根据您的colormap）
            tumor_volume = np.sum(attention_map[:, :, d] > 0.3)
            tumor_volume_per_slice.append((d, tumor_volume))

        # 选择肿瘤最大的8个切片
        tumor_volume_per_slice.sort(key=lambda x: x[1], reverse=True)
        representative_slices = [idx for idx, _ in tumor_volume_per_slice[:8]]

        # 创建可视化
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for i in range(16):
            if i < len(representative_slices):
                d = representative_slices[i // 2]

                if i % 2 == 0:
                    # 原始图像
                    axes[i].imshow(original_image[:, :, d], cmap='gray')
                    axes[i].set_title(f'Slice {d} - Original', fontsize=10)
                else:
                    # 注意力叠加
                    overlay = self._create_overlay(original_image[:, :, d], attention_map[:, :, d], alpha)
                    axes[i].imshow(overlay)
                    axes[i].set_title(f'Slice {d} - Attention', fontsize=10)
            else:
                axes[i].axis('off')

            axes[i].axis('off')

        plt.suptitle(f'Layer: {layer_name} - Representative Slices', fontsize=16)
        plt.tight_layout()

        slice_path = os.path.join(save_path, 'representative_slices.png')
        plt.savefig(slice_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved representative slices: {slice_path}")

    def _generate_3d_projections(self, original_image, attention_map, save_path, layer_name, alpha=0.5):
        """生成三个方向的投影"""
        fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)

        projections = [
            ('Axial MIP', lambda x: np.max(x, axis=2)),
            ('Coronal MIP', lambda x: np.max(x, axis=1)),
            ('Sagittal MIP', lambda x: np.max(x, axis=0))
        ]

        for idx, (proj_name, proj_func) in enumerate(projections):
            # 原始图像投影
            orig_proj = proj_func(original_image)
            axes[idx, 0].imshow(orig_proj, cmap='gray')
            axes[idx, 0].set_title(f'{proj_name} - Original')
            axes[idx, 0].axis('off')

            # 注意力图投影
            att_proj = proj_func(attention_map)
            im = axes[idx, 1].imshow(att_proj, cmap=self.cmap, vmin=0, vmax=1)
            axes[idx, 1].set_title(f'{proj_name} - Attention')
            axes[idx, 1].axis('off')

            # 叠加投影
            overlay_proj = self._create_overlay(orig_proj, att_proj, alpha)
            axes[idx, 2].imshow(overlay_proj)
            axes[idx, 2].set_title(f'{proj_name} - Overlay')
            axes[idx, 2].axis('off')

        # 添加colorbar
        cbar = fig.colorbar(im, ax=axes[:, 1], fraction=0.046, pad=0.04)
        cbar.set_label('Attention Value', rotation=270, labelpad=20)

        plt.suptitle(f'Layer: {layer_name} - 3D Projections', fontsize=16)

        proj_path = os.path.join(save_path, '3d_projections.png')
        plt.savefig(proj_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved 3D projections: {proj_path}")

    def _save_single_axial_mip(self, original_image, attention_map, save_path, layer_name, alpha=0.5):
        """保存单独的轴向MIP叠加图"""
        # 计算轴向最大投影
        orig_mip = np.max(original_image, axis=2)
        att_mip = np.max(attention_map, axis=2)

        # 创建图形
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(orig_mip, cmap='gray')
        axes[0].set_title('Original MIP')
        axes[0].axis('off')

        # 注意力图
        im = axes[1].imshow(att_mip, cmap=self.cmap, vmin=0, vmax=1)
        axes[1].set_title('Attention MIP')
        axes[1].axis('off')

        # 叠加
        overlay = self._create_overlay(orig_mip, att_mip, alpha)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay MIP')
        axes[2].axis('off')

        # 添加colorbar
        cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Attention Value', rotation=270, labelpad=15)

        plt.tight_layout()

        mip_path = os.path.join(save_path, 'axial_mip_comparison.png')
        plt.savefig(mip_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 也保存单独的叠加图
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.axis('off')
        plt.title(f'Axial MIP Overlay - {layer_name}')

        overlay_path = os.path.join(save_path, 'axial_mip_overlay_only.png')
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

        logger.info(f"Saved axial MIP: {mip_path}")

    def _create_overlay(self, original_slice, attention_slice, alpha=0.5):
        """创建注意力叠加图"""
        # 归一化原始图像
        original_norm = original_slice - original_slice.min()
        if original_norm.max() > 0:
            original_norm = original_norm / original_norm.max()

        # 使用colormap（兼容新版matplotlib）
        try:
            # 新版本方式
            custom_cmap = plt.colormaps[self.cmap]
        except:
            # 旧版本方式
            custom_cmap = plt.cm.get_cmap(self.cmap)

        attention_color = custom_cmap(attention_slice)[:, :, :3]

        # 将原始图像转换为RGB
        original_rgb = np.stack([original_norm] * 3, axis=-1)

        # 叠加
        overlay = (1 - alpha) * original_rgb + alpha * attention_color
        overlay = np.clip(overlay, 0, 1)

        return overlay

    def _save_attention_statistics(self, attention_map, save_path, layer_name):
        """保存注意力图的统计信息"""
        stats = {
            'layer_name': layer_name,
            'shape': list(attention_map.shape),
            'mean': float(np.mean(attention_map)),
            'std': float(np.std(attention_map)),
            'min': float(np.min(attention_map)),
            'max': float(np.max(attention_map)),
            'high_attention_ratio': float(np.sum(attention_map > 0.5) / attention_map.size),
            'very_high_attention_ratio': float(np.sum(attention_map > 0.7) / attention_map.size)
        }

        # 保存JSON统计信息
        json_path = os.path.join(save_path, 'statistics.json')
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=4)

        # 生成直方图
        plt.figure(figsize=(10, 6))
        plt.hist(attention_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
        plt.xlabel('Attention Value')
        plt.ylabel('Frequency')
        plt.title(f'Attention Value Distribution - {layer_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        hist_path = os.path.join(save_path, 'attention_histogram.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved statistics: {json_path}")


def preprocess_image_fixed(image, target_size=(160, 160, 128)):
    """修复的预处理函数"""
    image_tensor = torch.from_numpy(image.copy()).float()
    image_tensor = image_tensor.unsqueeze(0)  # [1, 4, H, W, D]

    image_resized = F.interpolate(
        image_tensor,
        size=target_size,
        mode='trilinear',
        align_corners=True
    )

    return image_resized


def load_model(model_path, device='cuda'):
    """加载模型"""
    from module.GDGMamU_Net_ESAACA import GDGMamU_Net

    model = GDGMamU_Net(4, 4)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model


def main():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # 定义目标层
    target_layers = [
        'Mamba.mamba.stages.0.blocks.1',
        'Mamba.mamba.stages.1.blocks.1',
        'fusion_modules.0.esa.fusion.2',
        'GDG1.mish',
        'GDG2.mish',
        'fusion_modules.0.output.2',
        'fusion_modules.1.output.2',

    ]

    # 加载模型
    model = load_model(args.model_path, device)

    # 初始化可视化器
    visualizer = CompleteMambaAttentionVisualizer(model, target_layers, cmap=args.cmap)

    # 读取文件列表
    with open(args.inference_file, 'r') as f:
        h5_files = f.read().splitlines()

    # 处理文件
    for idx, h5_file in enumerate(h5_files[:args.num_samples]):
        print(f"\n{'=' * 50}")
        print(f"Processing {idx + 1}/{args.num_samples}: {h5_file}")
        print(f"{'=' * 50}")

        h5_path = os.path.join(args.data_dir, h5_file)
        if not os.path.exists(h5_path):
            continue

        # 加载图像
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][:]  # [4, H, W, D]

        # 使用修复的预处理函数
        input_tensor = preprocess_image_fixed(image).to(device)

        # 生成可视化
        case_name = os.path.splitext(os.path.basename(h5_file))[0]
        save_path = os.path.join(args.output_dir, case_name)

        try:
            visualizer.visualize_attention(
                input_tensor,
                image,
                save_path,
                selected_modality=1,  # T1ce
                alpha=args.alpha
            )
            print(f"✅ 成功生成 {case_name} 的完整注意力可视化")
            print(f"   - 代表性切片: representative_slices.png")
            print(f"   - 3D投影: 3d_projections.png")
            print(f"   - 轴向MIP: axial_mip_comparison.png")
            print(f"   - 统计信息: statistics.json")
        except Exception as e:
            print(f"❌ 处理 {case_name} 失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='../results/best_model_WT0.879_ET0.809_TC0.851_AVG0.846.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--data_dir', type=str, default='../dataset_output/dataset')
    parser.add_argument('--inference_file', type=str, default='../dataset_output/inference.txt')
    parser.add_argument('--output_dir', type=str, default='complete_attention_results')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--cmap', type=str, default='jet')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # 🌱 设置随机种子确保结果可重复
    set_reproducible_seed(seed=args.seed)

    main()