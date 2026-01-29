import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from module.GDGMamU_Net_ESAACA import GDGMamU_Net
from tqdm import tqdm
import h5py
import warnings

warnings.filterwarnings('ignore')


class ImprovedGradCAM3D:
    """改进的3D GradCAM实现，结合批次策略和完整切片保存"""

    def __init__(self, model, target_layer, batch_size=4):
        """
        初始化 GradCAM

        :param model: 需要进行 GradCAM 的模型
        :param target_layer: 目标层的名称
        :param batch_size: 批次大小（用于解决归一化层问题）
        批次复制策略：有效解决了归一化层的兼容性问题
        """
        self.model = model
        self.target_layer = target_layer
        self.batch_size = batch_size
        self.gradients = None
        self.activations = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        """注册前向和反向钩子"""

        def forward_hook(module, input, output):
            # 保存第一个样本的激活
            self.activations = output[0:1].detach()

        def backward_hook(module, grad_in, grad_out):
            # 保存第一个样本的梯度
            if grad_out[0] is not None:
                self.gradients = grad_out[0][0:1].detach()

        # 查找并注册钩子
        target_found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                fhook = module.register_forward_hook(forward_hook)
                bhook = module.register_backward_hook(backward_hook)
                self.handles.extend([fhook, bhook])
                target_found = True
                break

        if not target_found:
            raise ValueError(f"Layer {self.target_layer} not found in the model.")

    def generate_cam(self, input_tensor, target_class):
        """
        生成 CAM

        :param input_tensor: 输入张量 [1, C, H, W, D]
        :param target_class: 目标类别
        :return: 生成的 CAM [1, 1, H, W, D]
        """
        self.model.eval()

        # 创建批次输入
        batch_input = input_tensor.repeat(self.batch_size, 1, 1, 1, 1)
        # 添加微小噪声避免完全相同
        noise = torch.randn_like(batch_input) * 0.001
        batch_input = batch_input + noise
        batch_input.requires_grad_(True)

        # 前向传播
        with torch.set_grad_enabled(True):
            output = self.model(batch_input)

            # 使用第一个样本的输出计算目标
            target_output = output[0:1, target_class, :, :, :]
            target = target_output.mean()

            # 反向传播
            self.model.zero_grad()
            target.backward()

            # 获取梯度和激活
            if self.gradients is None or self.activations is None:
                raise RuntimeError("Failed to capture gradients or activations")

            # 计算权重（全局平均池化）
            weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)

            # 生成 CAM
            cam = (weights * self.activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            # 调整大小到输出尺寸
            if cam.shape[2:] != output.shape[2:]:
                cam = F.interpolate(cam, size=output.shape[2:],
                                    mode='trilinear', align_corners=False)

            # 归一化 CAM
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            return cam

    def cleanup(self):
        """清理钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.gradients = None
        self.activations = None


def overlay_cam_on_image(original_slice, cam_slice, alpha=0.7):
    """
    将 Grad-CAM 叠加到原始图像上（使用标准的jet色彩映射）

    :param original_slice: 原始图像切片 [H, W]
    :param cam_slice: Grad-CAM 切片 [H, W]
    :param alpha: 透明度
    :return: 叠加后的图像 [H, W, 3]
    """
    # 归一化原始图像到 [0, 1]
    if original_slice.max() > original_slice.min():
        original_norm = (original_slice - original_slice.min()) / (original_slice.max() - original_slice.min())
    else:
        original_norm = np.zeros_like(original_slice)

    # 归一化 CAM 到 [0, 1]
    if cam_slice.max() > cam_slice.min():
        cam_norm = (cam_slice - cam_slice.min()) / (cam_slice.max() - cam_slice.min())
    else:
        cam_norm = np.zeros_like(cam_slice)

    # 使用 jet 颜色映射（蓝色到红色）
    cam_color = plt.cm.jet(cam_norm)[:, :, :3]

    # 将原始图像转换为 RGB
    original_rgb = np.stack([original_norm] * 3, axis=-1)

    # 叠加
    overlay = (1 - alpha) * original_rgb + alpha * cam_color
    overlay = np.clip(overlay, 0, 1)

    return (overlay * 255).astype(np.uint8)


def save_all_slices_cam(cam, original_image, save_path, case_name, class_name,
                        selected_modality=1, alpha=0.5):
    """
    保存所有切片的 Grad-CAM 叠加图像

    :param cam: 生成的 CAM [1, 1, H, W, D]
    :param original_image: 原始图像 [1, 4, H, W, D]
    :param save_path: 保存根路径
    :param case_name: 病例名称
    :param class_name: 类别名称
    :param selected_modality: 选择的模态 (0:T1, 1:T1ce, 2:T2, 3:Flair)
    :param alpha: 透明度
    """
    cam_np = cam.cpu().numpy()[0, 0]  # [H, W, D]
    orig_np = original_image.cpu().numpy()[0, selected_modality]  # [H, W, D]

    # 创建保存目录
    full_save_path = os.path.join(save_path, case_name, class_name)
    os.makedirs(full_save_path, exist_ok=True)

    H, W, D = cam_np.shape
    print(f"  保存 {D} 个切片到: {full_save_path}")

    # 保存每个切片
    for d in tqdm(range(D), desc=f'  保存切片', leave=False):
        orig_slice = orig_np[:, :, d]
        cam_slice = cam_np[:, :, d]

        # 创建叠加图像
        overlay = overlay_cam_on_image(orig_slice, cam_slice, alpha=alpha)

        # 创建包含原始图像、CAM和叠加的完整可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(orig_slice, cmap='gray')
        axes[0].set_title(f'Original - Slice {d}')
        axes[0].axis('off')

        # CAM热图
        im = axes[1].imshow(cam_slice, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title(f'GradCAM - Slice {d}')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # 叠加图
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay - Slice {d}')
        axes[2].axis('off')

        # 保存
        plt.tight_layout()
        plt.savefig(os.path.join(full_save_path, f'slice_{d:03d}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # 保存最大投影图
    save_projection_views(cam_np, orig_np, full_save_path, alpha)

    print(f"  ✅ 完成保存所有切片")


def save_projection_views(cam_np, orig_np, save_path, alpha=0.5):
    """保存三个方向的最大投影图"""
    projections = {
        'axial': (np.max(cam_np, axis=2), np.max(orig_np, axis=2)),
        'sagittal': (np.max(cam_np, axis=1), np.max(orig_np, axis=1)),
        'coronal': (np.max(cam_np, axis=0), np.max(orig_np, axis=0))
    }

    for view_name, (cam_proj, orig_proj) in projections.items():
        overlay_proj = overlay_cam_on_image(orig_proj, cam_proj, alpha=alpha)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始投影
        axes[0].imshow(orig_proj, cmap='gray')
        axes[0].set_title(f'Original - {view_name.capitalize()} Projection')
        axes[0].axis('off')

        # CAM投影
        im = axes[1].imshow(cam_proj, cmap='jet')
        axes[1].set_title(f'GradCAM - {view_name.capitalize()} Projection')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # 叠加投影
        axes[2].imshow(overlay_proj)
        axes[2].set_title(f'Overlay - {view_name.capitalize()} Projection')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'projection_{view_name}.png'),
                    dpi=200, bbox_inches='tight')
        plt.close()


def load_model_safe(model_path, device='cuda'):
    """安全加载模型"""
    model = GDGMamU_Net(in_channels=4, num_classes=4)
    checkpoint = torch.load(model_path, map_location=device,weights_only=False)

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
    return model


def main():
    """主函数 - 生成所有切片的GradCAM可视化"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 完整切片GradCAM生成工具")
    print(f"📱 设备: {device}")
    print("=" * 80)

    # 配置
    config = {
        'model_path': '../results/best_model_WT0.879_ET0.809_TC0.851_AVG0.846.pth',
        'data_path': "../dataset_output",
        'inference_file': "../dataset_output/inference.txt",
        'save_path': 'gradcam_all_slices',
        'target_size': (160, 160, 128)
    }

    # 类别定义
    class_labels = {
        0: 'Background',
        1: 'ET_EnhancingTumor',
        2: 'TC_TumorCore',
        3: 'WT_WholeTumor'
    }

    # 加载模型
    print("加载模型...")
    model = load_model_safe(config['model_path'], device)

    # 读取文件列表
    with open(config['inference_file'], 'r') as f:
        h5_files = [line.strip() for line in f.readlines() if line.strip()]

    print(f"找到 {len(h5_files)} 个待处理文件")

    # 选择要可视化的层
    target_layers = [
        'fusion_modules.0.output.2',
        'fusion_modules.1.output.2',
        'fusion_modules.2.output.2',
        'Mamba.mamba.stages.0.blocks.1',
        'Mamba.mamba.stages.1.blocks.1',
        'Mamba.mamba.stages.0.blocks.0.dwconv1.depth_conv',
        'GDG2.mish',
        'GDG1.mish'
    ]

    # 选择要生成的类别
    classes_to_generate = [1, 2, 3]  # ET, TC, WT

    # 处理每个文件
    for idx, h5_file in enumerate(h5_files[:3]):  # 限制处理前3个文件
        h5_path = os.path.join(config['data_path'], 'dataset', h5_file)

        if not os.path.exists(h5_path):
            print(f"❌ 文件不存在: {h5_path}")
            continue

        print(f"\n处理文件 {idx + 1}/{min(3, len(h5_files))}: {h5_file}")

        # 加载数据
        with h5py.File(h5_path, 'r') as f:
            image = torch.from_numpy(f['image'][:]).float().unsqueeze(0)

        # 调整大小
        if image.shape[2:] != config['target_size']:
            image = F.interpolate(image, size=config['target_size'],
                                  mode='trilinear', align_corners=False)

        image = image.to(device)
        case_name = os.path.splitext(os.path.basename(h5_file))[0]

        # 对每个目标层生成GradCAM
        for target_layer in target_layers:
            print(f"\n目标层: {target_layer}")

            # 初始化GradCAM
            gradcam = ImprovedGradCAM3D(model, target_layer, batch_size=4)

            # 对每个类别生成CAM
            for target_class in classes_to_generate:
                class_name = class_labels[target_class]
                print(f"  生成类别 {target_class} ({class_name}) 的GradCAM...")

                try:
                    # 生成CAM
                    cam = gradcam.generate_cam(image, target_class)

                    # 保存所有切片
                    save_all_slices_cam(
                        cam, image,
                        os.path.join(config['save_path'], target_layer.replace('.', '_')),
                        case_name, class_name,
                        selected_modality=1,  # 使用T1ce
                        alpha=0.5
                    )

                except Exception as e:
                    print(f"  ❌ 生成失败: {e}")
                    continue

            # 清理钩子
            gradcam.cleanup()

    print(f"\n🎉 完成！所有结果保存在: {config['save_path']}")


if __name__ == '__main__':
    main()