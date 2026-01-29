import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from module.GDGMamU_Net_ESAACA import GDGMamU_Net
from tqdm import tqdm
import h5py
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class OptimizedPaperGradCAM3D:
    """优化的3D GradCAM实现 - 结合论文级可视化与完整功能"""

    def __init__(self, model, target_layers=None, paper_mode=False, batch_size=4):
        """
        初始化优化的GradCAM

        Args:
            model: 分割模型
            target_layers: 目标层列表
            paper_mode: 是否启用论文模式
            batch_size: 批次大小（解决归一化层问题）
        """
        self.model = model
        self.paper_mode = paper_mode
        self.batch_size = batch_size

        # 根据模式选择目标层
        if paper_mode:
            self.target_layers = self._get_paper_layers()
        else:
            self.target_layers = target_layers or self._auto_select_layers()

        self.gradients = {}
        self.activations = {}
        self.handles = []

        # 论文展示配置
        self.paper_configs = self._get_paper_figure_configs()

        # 可视化配置
        self.viz_config = {
            'save_all_slices': True,  # 是否保存所有切片
            'save_best_slices': True,  # 是否保存最佳切片
            'num_best_slices': 5,  # 最佳切片数量
            'save_projections': True,  # 是否保存投影视图
            'colormap': 'jet',  # 颜色映射
            'alpha': 0.5,  # 叠加透明度
            'dpi': 300  # 输出分辨率
        }

        self._register_hooks()

    def _get_paper_layers(self):
        """获取论文展示的最佳层配置"""
        # 避免有问题的层，选择稳定的卷积层
        PAPER_LAYERS = [
            'GDG1.conv1_1',  # GDG早期特征
            'GDG1.conv3_2',  # GDG高级特征
            'GDG2.conv3_2',  # GDG2特征
            'decoder1.conv1',  # 解码器第1层
            'decoder2.conv1',  # 解码器第2层
            'decoder4.conv1'  # 最终解码层
        ]

        print(f"🎯 论文模式启用，使用 {len(PAPER_LAYERS)} 个优化层")
        for i, layer in enumerate(PAPER_LAYERS, 1):
            print(f"   {i}. {layer}")

        return PAPER_LAYERS

    def _get_paper_figure_configs(self):
        """获取论文图片的层配置"""
        return {
            'Figure1_Feature_Evolution': {
                'layers': ['GDG1.conv1_1', 'GDG1.conv3_2', 'decoder4.conv1'],
                'description': '特征演化分析',
                'purpose': '展示从浅层到深层的特征变化'
            },
            'Figure2_Multiscale_Analysis': {
                'layers': ['GDG1.conv3_2', 'GDG2.conv3_2'],
                'description': '多尺度特征分析',
                'purpose': '展示不同分辨率下的特征表示'
            },
            'Figure3_Decoder_Progress': {
                'layers': ['decoder1.conv1', 'decoder2.conv1', 'decoder4.conv1'],
                'description': '解码器渐进分析',
                'purpose': '展示特征重建过程'
            }
        }

    def _auto_select_layers(self):
        """自动选择合适的目标层"""
        target_layers = []
        for name, module in self.model.named_modules():
            # 选择没有归一化问题的卷积层
            if isinstance(module, torch.nn.Conv3d) and 'conv' in name:
                if not any(prob in name for prob in ['norm', 'bn', 'pool']):
                    target_layers.append(name)

        # 返回最后6个层
        return target_layers[-6:] if len(target_layers) >= 6 else target_layers

    def _register_hooks(self):
        """注册前向和反向钩子"""

        def get_forward_hook(name):
            def hook(module, input, output):
                # 保存整个批次但后续只使用第一个样本
                self.activations[name] = output.detach()

            return hook

        def get_backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()

            return hook

        # 清理旧钩子
        self._cleanup_hooks()

        # 注册新钩子
        for target_layer in self.target_layers:
            layer_found = False
            for name, module in self.model.named_modules():
                if name == target_layer:
                    fhook = module.register_forward_hook(get_forward_hook(name))
                    bhook = module.register_backward_hook(get_backward_hook(name))
                    self.handles.extend([fhook, bhook])
                    layer_found = True
                    break

            if not layer_found:
                print(f"⚠️ 警告：未找到层 {target_layer}")

    def generate_cam(self, input_tensor, target_class, return_pred=False):
        """
        生成CAM - 使用批次复制策略

        Args:
            input_tensor: 输入张量 [1, C, H, W, D]
            target_class: 目标类别
            return_pred: 是否返回预测结果

        Returns:
            cams: 各层的CAM字典
            pred_mask: 预测掩码（如果return_pred=True）
        """
        self.model.eval()

        # 创建批次输入
        batch_input = self._create_batch_input(input_tensor)
        batch_input.requires_grad_(True)

        # 前向传播
        with torch.set_grad_enabled(True):
            with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度
                output = self.model(batch_input)

            # 只使用第一个样本计算损失
            single_output = output[0:1]
            target_output = single_output[:, target_class, :, :, :]

            # 使用更稳定的目标计算
            target = target_output.mean()

            # 反向传播
            self.model.zero_grad()
            target.backward()

            # 收集CAM
            cams = {}
            for layer_name in self.target_layers:
                if layer_name in self.gradients and layer_name in self.activations:
                    try:
                        cam = self._compute_cam_for_layer(
                            self.gradients[layer_name][0:1],  # 只使用第一个样本
                            self.activations[layer_name][0:1],
                            output.shape[2:]
                        )
                        if cam is not None:
                            cams[layer_name] = cam
                    except Exception as e:
                        print(f"⚠️ 层 {layer_name} 计算CAM失败: {e}")
                        continue

            if return_pred:
                pred_mask = torch.argmax(single_output, dim=1)
                return cams, pred_mask

            return cams

    def _create_batch_input(self, input_tensor):
        """创建批次输入以解决归一化层问题"""
        batch = input_tensor.repeat(self.batch_size, 1, 1, 1, 1)
        # 添加微小噪声避免完全相同
        noise = torch.randn_like(batch) * 0.001
        return batch + noise

    def _compute_cam_for_layer(self, gradients, activations, target_size):
        """计算单个层的CAM"""
        # 全局平均池化计算权重
        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)

        # 生成CAM
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # 调整大小
        if cam.shape[2:] != target_size:
            cam = F.interpolate(cam, size=target_size, mode='trilinear', align_corners=False)

        # 归一化
        cam = self._normalize_cam(cam)

        return cam

    def _normalize_cam(self, cam):
        """归一化CAM"""
        batch_size = cam.shape[0]
        for i in range(batch_size):
            cam_i = cam[i]
            cam_min = cam_i.min()
            cam_max = cam_i.max()
            if cam_max > cam_min:
                cam[i] = (cam_i - cam_min) / (cam_max - cam_min)
            else:
                cam[i] = torch.zeros_like(cam_i)
        return cam

    def generate_comprehensive_analysis(self, input_tensor, case_name, save_dir):
        """
        生成综合分析 - 结合论文级可视化和完整切片保存

        Args:
            input_tensor: 输入张量
            case_name: 病例名称
            save_dir: 保存目录
        """
        print(f"\n{'=' * 80}")
        print(f"🎨 开始综合GradCAM分析 - 案例: {case_name}")
        print(f"{'=' * 80}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(save_dir, f"{case_name}_{timestamp}")

        all_results = {}

        # 1. 生成论文级可视化
        if self.paper_mode:
            print("\n📊 生成论文级可视化...")
            paper_results = self._generate_paper_figures(input_tensor, case_name, save_dir)
            all_results['paper_figures'] = paper_results

        # 2. 生成完整切片分析
        if self.viz_config['save_all_slices']:
            print("\n📂 生成完整切片分析...")
            slice_results = self._generate_all_slices_analysis(input_tensor, case_name, save_dir)
            all_results['all_slices'] = slice_results

        # 3. 生成对比分析
        print("\n📈 生成对比分析...")
        comparison_results = self._generate_comparison_analysis(input_tensor, case_name, save_dir)
        all_results['comparison'] = comparison_results

        # 4. 生成统计报告
        self._generate_statistics_report(all_results, save_dir)

        print(f"\n✅ 分析完成！结果保存在: {save_dir}")
        return all_results

    def _generate_paper_figures(self, input_tensor, case_name, save_dir):
        """生成论文级可视化"""
        paper_results = {}

        for fig_name, config in self.paper_configs.items():
            print(f"\n  📊 {config['description']}")

            # 临时更新目标层
            original_layers = self.target_layers
            self.target_layers = config['layers']
            self._register_hooks()

            fig_results = {}

            # 为每个类别生成CAM
            for class_id in [1, 2, 3]:  # ET, TC, WT
                class_names = {1: 'ET', 2: 'TC', 3: 'WT'}
                class_name = class_names[class_id]

                try:
                    cams = self.generate_cam(input_tensor, class_id)
                    if cams:
                        fig_results[class_id] = cams

                        # 保存最佳切片可视化
                        for layer_name, cam in cams.items():
                            self._save_best_slices_visualization(
                                cam, input_tensor,
                                os.path.join(save_dir, 'paper_figures', fig_name),
                                f"{layer_name}_{class_name}", class_name
                            )

                except Exception as e:
                    print(f"    ❌ 类别 {class_name} 失败: {e}")

            if fig_results:
                paper_results[fig_name] = fig_results
                self._create_figure_comparison(
                    fig_results, config,
                    os.path.join(save_dir, 'paper_figures', f'{fig_name}_comparison.png')
                )

            # 恢复原始层
            self.target_layers = original_layers
            self._register_hooks()

        return paper_results

    def _generate_all_slices_analysis(self, input_tensor, case_name, save_dir):
        """生成所有切片的完整分析"""
        slice_results = {}

        # 选择要分析的层（避免太多）
        layers_to_analyze = self.target_layers[:3] if len(self.target_layers) > 3 else self.target_layers

        for layer_name in layers_to_analyze:
            print(f"\n  📍 分析层: {layer_name}")
            layer_results = {}

            # 临时设置单个目标层
            self.target_layers = [layer_name]
            self._register_hooks()

            for class_id in [1, 2, 3]:
                class_names = {1: 'ET', 2: 'TC', 3: 'WT'}
                class_name = class_names[class_id]

                try:
                    cams = self.generate_cam(input_tensor, class_id)
                    if layer_name in cams:
                        cam = cams[layer_name]
                        layer_results[class_id] = cam

                        # 保存所有切片
                        self._save_all_slices_visualization(
                            cam, input_tensor,
                            os.path.join(save_dir, 'all_slices', layer_name.replace('.', '_')),
                            case_name, class_name
                        )

                except Exception as e:
                    print(f"    ❌ 类别 {class_name} 失败: {e}")

            if layer_results:
                slice_results[layer_name] = layer_results

        return slice_results

    def _generate_comparison_analysis(self, input_tensor, case_name, save_dir):
        """生成层间对比分析"""
        comparison_results = {}

        # 为每个类别生成跨层对比
        for class_id in [1, 2, 3]:
            class_names = {1: 'ET', 2: 'TC', 3: 'WT'}
            class_name = class_names[class_id]

            print(f"\n  🔍 生成 {class_name} 的跨层对比...")

            try:
                # 获取所有层的CAM
                self.target_layers = self._get_paper_layers()[:4]  # 使用前4个层
                self._register_hooks()

                cams = self.generate_cam(input_tensor, class_id)

                if cams:
                    comparison_results[class_id] = cams
                    self._create_cross_layer_comparison(
                        cams, input_tensor,
                        os.path.join(save_dir, 'comparisons'),
                        f"{case_name}_{class_name}_layers", class_name
                    )

            except Exception as e:
                print(f"    ❌ 失败: {e}")

        return comparison_results

    def _save_best_slices_visualization(self, cam, original_image, save_path, name, class_name):
        """保存最佳切片的可视化"""
        cam_np = cam.cpu().numpy()[0, 0]
        orig_np = original_image.cpu().numpy()[0, 1]  # T1ce

        # 计算每个切片的激活强度
        slice_scores = [cam_np[:, :, d].sum() for d in range(cam_np.shape[2])]
        best_indices = sorted(range(len(slice_scores)),
                              key=lambda x: slice_scores[x],
                              reverse=True)[:self.viz_config['num_best_slices']]

        os.makedirs(save_path, exist_ok=True)

        # 创建最佳切片的组合图
        fig, axes = plt.subplots(self.viz_config['num_best_slices'], 3,
                                 figsize=(12, 4 * self.viz_config['num_best_slices']))

        for i, slice_idx in enumerate(best_indices):
            # 原始图像
            axes[i, 0].imshow(orig_np[:, :, slice_idx], cmap='gray')
            axes[i, 0].set_title(f'Original - Slice {slice_idx}')
            axes[i, 0].axis('off')

            # CAM
            im = axes[i, 1].imshow(cam_np[:, :, slice_idx],
                                   cmap=self.viz_config['colormap'])
            axes[i, 1].set_title(f'GradCAM - Slice {slice_idx}')
            axes[i, 1].axis('off')

            # 叠加
            overlay = self._create_overlay(orig_np[:, :, slice_idx],
                                           cam_np[:, :, slice_idx])
            axes[i, 2].imshow(overlay)
            axes[i, 2].set_title(f'Overlay - Slice {slice_idx}')
            axes[i, 2].axis('off')

        plt.suptitle(f'{name} - {class_name} - Best Slices', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{name}_best_slices.png'),
                    dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()

        # 保存投影视图
        if self.viz_config['save_projections']:
            self._save_projection_views(cam_np, orig_np, save_path, name, class_name)

    def _save_all_slices_visualization(self, cam, original_image, save_path, case_name, class_name):
        """保存所有切片的可视化"""
        cam_np = cam.cpu().numpy()[0, 0]
        orig_np = original_image.cpu().numpy()[0, 1]  # T1ce

        full_save_path = os.path.join(save_path, class_name)
        os.makedirs(full_save_path, exist_ok=True)

        # 保存每个切片
        for d in tqdm(range(cam_np.shape[2]), desc=f'    保存 {class_name} 切片', leave=False):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            # 原始图像
            axes[0].imshow(orig_np[:, :, d], cmap='gray')
            axes[0].set_title(f'Original')
            axes[0].axis('off')

            # CAM
            im = axes[1].imshow(cam_np[:, :, d], cmap=self.viz_config['colormap'])
            axes[1].set_title(f'GradCAM')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)

            # 叠加
            overlay = self._create_overlay(orig_np[:, :, d], cam_np[:, :, d])
            axes[2].imshow(overlay)
            axes[2].set_title(f'Overlay')
            axes[2].axis('off')

            plt.suptitle(f'{case_name} - {class_name} - Slice {d}')
            plt.tight_layout()
            plt.savefig(os.path.join(full_save_path, f'slice_{d:03d}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    def _save_projection_views(self, cam_np, orig_np, save_path, name, class_name):
        """保存三个方向的投影视图"""
        projections = {
            'axial': (2, 'Axial View (Top)'),
            'sagittal': (1, 'Sagittal View (Side)'),
            'coronal': (0, 'Coronal View (Front)')
        }

        fig, axes = plt.subplots(len(projections), 3, figsize=(12, 4 * len(projections)))

        for idx, (proj_name, (axis, title)) in enumerate(projections.items()):
            # 计算投影
            cam_proj = np.max(cam_np, axis=axis)
            orig_proj = np.max(orig_np, axis=axis)

            # 原始投影
            axes[idx, 0].imshow(orig_proj, cmap='gray')
            axes[idx, 0].set_title(f'Original - {title}')
            axes[idx, 0].axis('off')

            # CAM投影
            im = axes[idx, 1].imshow(cam_proj, cmap=self.viz_config['colormap'])
            axes[idx, 1].set_title(f'GradCAM - {title}')
            axes[idx, 1].axis('off')
            plt.colorbar(im, ax=axes[idx, 1], fraction=0.046)

            # 叠加投影
            overlay_proj = self._create_overlay(orig_proj, cam_proj)
            axes[idx, 2].imshow(overlay_proj)
            axes[idx, 2].set_title(f'Overlay - {title}')
            axes[idx, 2].axis('off')

        plt.suptitle(f'{name} - {class_name} - Projections', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{name}_projections.png'),
                    dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()

    def _create_overlay(self, orig_slice, cam_slice):
        """创建叠加图像"""
        # 归一化
        if orig_slice.max() > orig_slice.min():
            orig_norm = (orig_slice - orig_slice.min()) / (orig_slice.max() - orig_slice.min())
        else:
            orig_norm = np.zeros_like(orig_slice)

        if cam_slice.max() > cam_slice.min():
            cam_norm = (cam_slice - cam_slice.min()) / (cam_slice.max() - cam_slice.min())
        else:
            cam_norm = np.zeros_like(cam_slice)

        # 应用颜色映射
        cmap = plt.cm.get_cmap(self.viz_config['colormap'])
        cam_colored = cmap(cam_norm)[:, :, :3]

        # 转换原始图像为RGB
        orig_rgb = np.stack([orig_norm] * 3, axis=-1)

        # 叠加
        alpha = self.viz_config['alpha']
        overlay = (1 - alpha) * orig_rgb + alpha * cam_colored

        return np.clip(overlay, 0, 1)

    def _create_figure_comparison(self, results, config, save_path):
        """创建图形对比"""
        layers = config['layers']
        n_layers = len(layers)

        fig, axes = plt.subplots(3, n_layers, figsize=(4 * n_layers, 10))
        if n_layers == 1:
            axes = axes.reshape(-1, 1)

        class_names = {1: 'ET', 2: 'TC', 3: 'WT'}

        for class_idx, class_id in enumerate([1, 2, 3]):
            for layer_idx, layer_name in enumerate(layers):
                if class_id in results and layer_name in results[class_id]:
                    cam = results[class_id][layer_name]
                    cam_np = cam.cpu().numpy()[0, 0]

                    # 选择最佳切片
                    slice_scores = [cam_np[:, :, d].sum() for d in range(cam_np.shape[2])]
                    best_slice = np.argmax(slice_scores)

                    im = axes[class_idx, layer_idx].imshow(
                        cam_np[:, :, best_slice],
                        cmap=self.viz_config['colormap']
                    )
                    axes[class_idx, layer_idx].set_title(
                        f'{layer_name.split(".")[-1]}\n{class_names[class_id]}'
                    )
                    axes[class_idx, layer_idx].axis('off')
                else:
                    axes[class_idx, layer_idx].text(0.5, 0.5, 'No Data',
                                                    ha='center', va='center')
                    axes[class_idx, layer_idx].axis('off')

        fig.suptitle(config['description'], fontsize=16, fontweight='bold')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()

    def _create_cross_layer_comparison(self, cams, original_image, save_path, name, class_name):
        """创建跨层对比可视化"""
        os.makedirs(save_path, exist_ok=True)

        n_layers = len(cams)
        fig, axes = plt.subplots(n_layers, 4, figsize=(16, 4 * n_layers))

        if n_layers == 1:
            axes = axes.reshape(1, -1)

        orig_np = original_image.cpu().numpy()[0, 1]  # T1ce

        for idx, (layer_name, cam) in enumerate(cams.items()):
            cam_np = cam.cpu().numpy()[0, 0]

            # 选择最佳切片
            slice_scores = [cam_np[:, :, d].sum() for d in range(cam_np.shape[2])]
            best_slice = np.argmax(slice_scores)

            # 原始图像
            axes[idx, 0].imshow(orig_np[:, :, best_slice], cmap='gray')
            axes[idx, 0].set_title(f'{layer_name} - Original')
            axes[idx, 0].axis('off')

            # CAM
            im = axes[idx, 1].imshow(cam_np[:, :, best_slice],
                                     cmap=self.viz_config['colormap'])
            axes[idx, 1].set_title(f'{layer_name} - GradCAM')
            axes[idx, 1].axis('off')

            # 叠加
            overlay = self._create_overlay(orig_np[:, :, best_slice],
                                           cam_np[:, :, best_slice])
            axes[idx, 2].imshow(overlay)
            axes[idx, 2].set_title(f'{layer_name} - Overlay')
            axes[idx, 2].axis('off')

            # CAM直方图
            axes[idx, 3].hist(cam_np.flatten(), bins=50, alpha=0.7)
            axes[idx, 3].set_title(f'{layer_name} - Distribution')
            axes[idx, 3].set_xlabel('Activation Value')
            axes[idx, 3].set_ylabel('Frequency')

        plt.suptitle(f'{name} - Cross-Layer Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{name}_comparison.png'),
                    dpi=self.viz_config['dpi'], bbox_inches='tight')
        plt.close()

    def _generate_statistics_report(self, results, save_dir):
        """生成统计报告"""
        report_path = os.path.join(save_dir, 'analysis_report.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GradCAM Analysis Report\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # 配置信息
            f.write("Configuration:\n")
            f.write(f"- Paper Mode: {self.paper_mode}\n")
            f.write(f"- Batch Size: {self.batch_size}\n")
            f.write(f"- Target Layers: {len(self.target_layers)}\n")
            for layer in self.target_layers:
                f.write(f"  * {layer}\n")
            f.write("\n")

            # 可视化配置
            f.write("Visualization Settings:\n")
            for key, value in self.viz_config.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")

            # 结果统计
            f.write("Results Summary:\n")

            if 'paper_figures' in results:
                f.write(f"- Paper Figures Generated: {len(results['paper_figures'])}\n")
                for fig_name, fig_data in results['paper_figures'].items():
                    f.write(f"  * {fig_name}: {len(fig_data)} classes\n")

            if 'all_slices' in results:
                f.write(f"- Complete Slice Analysis: {len(results['all_slices'])} layers\n")

            if 'comparison' in results:
                f.write(f"- Cross-layer Comparisons: {len(results['comparison'])} classes\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("Analysis completed successfully.\n")

        print(f"\n📄 统计报告已保存: {report_path}")

    def _cleanup_hooks(self):
        """清理所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.gradients.clear()
        self.activations.clear()

    def update_visualization_config(self, **kwargs):
        """更新可视化配置"""
        self.viz_config.update(kwargs)
        print("✅ 可视化配置已更新:")
        for key, value in kwargs.items():
            print(f"   - {key}: {value}")

    def set_figure_mode(self, figure_name):
        """设置特定图片模式"""
        if figure_name in self.paper_configs:
            config = self.paper_configs[figure_name]
            self.target_layers = config['layers']
            self._register_hooks()
            print(f"📊 切换到 {config['description']} 模式")
        else:
            print(f"❌ 未知的图片配置: {figure_name}")


def load_model_safe(model_path, device='cuda'):
    """安全加载模型"""
    try:
        model = GDGMamU_Net(in_channels=4, num_classes=4)
        checkpoint = torch.load(model_path, map_location=device,weights_only=False)

        # 灵活处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()
        print(f"✅ 模型加载成功")
        return model

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


def load_and_preprocess_brats(h5_path, target_size=(160, 160, 128)):
    """加载和预处理BraTS数据"""
    try:
        with h5py.File(h5_path, 'r') as f:
            image = f['image'][:]  # [4, H, W, D]
            label = f['label'][:] if 'label' in f else None

        # 标准化处理
        image_normalized = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            img_c = image[c].astype(np.float32)
            mask = img_c > 0
            if mask.any():
                valid_values = img_c[mask]
                mean_val = valid_values.mean()
                std_val = valid_values.std()
                if std_val > 0:
                    img_c[mask] = (img_c[mask] - mean_val) / std_val
            image_normalized[c] = img_c

        # 转换为tensor
        image_tensor = torch.from_numpy(image_normalized).unsqueeze(0)

        # 调整大小
        if image_tensor.shape[2:] != target_size:
            image_tensor = F.interpolate(image_tensor, size=target_size,
                                         mode='trilinear', align_corners=False)

        return image_tensor, label

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return None, None


def main():
    """主函数 - 展示优化整合的GradCAM功能"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 优化整合的GradCAM分析工具")
    print(f"📱 设备: {device}")
    print("=" * 80)

    # 配置
    config = {
        'model_path': '../results/best_model_WT0.879_ET0.809_TC0.851_AVG0.846.pth',
        'data_path': "../dataset_output",
        'inference_file':"../dataset_output/inference.txt",
        'output_path': 'optimized_gradcam_results',
        'target_size': (160, 160, 128)
    }
    # config = {
    #     'model_path': '../results/best_model_WT0.879_ET0.809_TC0.851_AVG0.846.pth',
    #     'data_path': r"C:\Users\smll0\PycharmProjects\pythonProject\code\reappear\CNN_Transformer\dataset_output",
    #     'inference_file': r"C:\Users\smll0\PycharmProjects\pythonProject\code\reappear\CNN_Transformer\dataset_output\inference.txt",
    #     'output_path': 'optimized_gradcam_results',
    #     'target_size': (160, 160, 128)
    # }

    # 加载模型
    print("\n📥 加载模型...")
    model = load_model_safe(config['model_path'], device)

    # 读取文件列表
    with open(config['inference_file'], 'r') as f:
        h5_files = [line.strip() for line in f.readlines() if line.strip()]

    print(f"📁 找到 {len(h5_files)} 个文件")

    # 创建优化的GradCAM实例
    gradcam = OptimizedPaperGradCAM3D(
        model,
        paper_mode=True,  # 启用论文模式
        batch_size=4  # 使用批次策略
    )

    # 更新可视化配置（可选）
    gradcam.update_visualization_config(
        save_all_slices=True,  # 保存所有切片
        save_best_slices=True,  # 保存最佳切片
        num_best_slices=5,  # 最佳切片数量
        save_projections=True,  # 保存投影
        alpha=0.5,  # 叠加透明度
        dpi=300  # 高分辨率输出
    )

    # 处理文件
    max_cases = 2  # 演示用，限制处理数量

    for idx, h5_file in enumerate(h5_files[:max_cases]):
        h5_path = os.path.join(config['data_path'], 'dataset', h5_file)

        if not os.path.exists(h5_path):
            print(f"❌ 文件不存在: {h5_path}")
            continue

        print(f"\n处理文件 {idx + 1}/{max_cases}: {h5_file}")

        # 加载数据
        input_tensor, label = load_and_preprocess_brats(h5_path, config['target_size'])
        if input_tensor is None:
            continue

        input_tensor = input_tensor.to(device)
        case_name = os.path.splitext(os.path.basename(h5_file))[0]

        # 生成综合分析
        try:
            results = gradcam.generate_comprehensive_analysis(
                input_tensor,
                case_name,
                config['output_path']
            )

            print(f"✅ {case_name} 分析完成")

        except Exception as e:
            print(f"❌ 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 清理
    gradcam._cleanup_hooks()

    print(f"\n🎉 所有分析完成！")
    print(f"📁 结果保存在: {config['output_path']}")

    # 提供使用建议
    print("\n💡 使用建议:")
    print("1. 论文展示：使用 paper_figures 目录中的对比图")
    print("2. 详细分析：查看 all_slices 目录中的完整切片")
    print("3. 层间对比：参考 comparisons 目录中的跨层分析")
    print("4. 统计信息：查看每个案例目录中的 analysis_report.txt")


if __name__ == '__main__':
    main()