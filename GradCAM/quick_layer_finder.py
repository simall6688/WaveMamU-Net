# 修复版层名称调试工具 - 正确处理checkpoint格式
import torch
import os


def debug_model_layers_fixed(model_path):
    """修复版：正确处理checkpoint中的model_state_dict"""
    print("🔍 开始详细分析模型层结构...")
    print("=" * 80)

    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return None

    try:
        # 加载checkpoint
        print(f"📁 正在加载: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu',weights_only=False)
        print(f"✅ 成功加载模型文件")

        # 打印checkpoint的键
        print(f"\n📋 Checkpoint包含的键: {list(checkpoint.keys())}")

        # 🔧 关键修复：正确获取模型的state_dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("🔧 使用 checkpoint['model_state_dict'] ✅")
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            print("🔧 使用 checkpoint['model']")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("🔧 使用 checkpoint['state_dict']")
        else:
            # 如果都没有，可能整个checkpoint就是state_dict
            state_dict = checkpoint
            print("🔧 直接使用 checkpoint")

        # 检查state_dict是否包含模型参数
        if not isinstance(state_dict, dict):
            print(f"❌ state_dict不是字典类型: {type(state_dict)}")
            return None

        # 检查是否包含权重参数
        param_names = list(state_dict.keys())
        weight_params = [name for name in param_names if '.weight' in name]

        if not weight_params:
            print(f"❌ 没有找到.weight参数，可能state_dict格式不正确")
            print(f"📋 State_dict的前10个键: {param_names[:10]}")
            return None

        print(f"📊 State dict包含 {len(param_names)} 个参数")
        print(f"✅ 找到 {len(weight_params)} 个权重参数")

        # 显示前20个参数名称
        print(f"\n📝 前20个参数名称:")
        for i, name in enumerate(param_names[:20]):
            marker = "🔥" if any(x in name.lower() for x in ['gdg', 'mamba', 'fusion', 'attn']) else "  "
            print(f"  {i + 1:2d}. {marker} {name}")

        if len(param_names) > 20:
            print(f"     ... 还有 {len(param_names) - 20} 个参数")

        # 提取层名称（去掉权重后缀）
        print(f"\n🏗️ 提取的层结构:")
        layer_names = set()

        for param_name in param_names:
            # 移除常见的参数后缀
            layer_name = param_name
            suffixes = ['.weight', '.bias', '.running_mean', '.running_var',
                        '.num_batches_tracked', '.A_log', '.D']

            for suffix in suffixes:
                if layer_name.endswith(suffix):
                    layer_name = layer_name[:-len(suffix)]
                    break

            if layer_name and '.' in layer_name:  # 只要有层次结构的
                layer_names.add(layer_name)

        # 排序并显示所有层
        sorted_layers = sorted(layer_names)
        print(f"📊 找到 {len(sorted_layers)} 个不同的层:")

        for i, layer in enumerate(sorted_layers):
            # 标记重要的创新组件
            marker = "  "
            if any(x in layer.lower() for x in ['gdg']):
                marker = "🔥"
            elif any(x in layer.lower() for x in ['mamba', 'stage', 'mixer']):
                marker = "🐍"
            elif any(x in layer.lower() for x in ['fusion']):
                marker = "🔗"
            elif any(x in layer.lower() for x in ['esa', 'aca', 'attn']):
                marker = "👁️"

            print(f"  {i + 1:2d}. {marker} {layer}")

        # 分析层的模式 - 更精确的分类
        print(f"\n🔍 创新组件层名称分析:")
        patterns = {
            '🔥 GDG下采样模块': [],
            '🐍 Mamba编码器': [],
            '🔗 融合模块': [],
            '👁️ ESA空间注意力': [],
            '📊 ACA通道注意力': [],
            '🎯 残差注意力': [],
            '👻 Ghost模块': [],
            '⬆️ 解码器模块': [],
            '📦 其他卷积层': []
        }

        for layer in sorted_layers:
            categorized = False

            # GDG模块 - 核心创新点1
            if any(x in layer.lower() for x in ['gdg']):
                patterns['🔥 GDG下采样模块'].append(layer)
                categorized = True

            # Mamba相关 - 核心创新点2
            elif any(x in layer.lower() for x in ['mamba']):
                patterns['🐍 Mamba编码器'].append(layer)
                categorized = True

            # 融合模块 - 核心创新点3
            elif any(x in layer.lower() for x in ['fusion']):
                patterns['🔗 融合模块'].append(layer)
                categorized = True

            # ESA空间注意力 - 核心创新点4
            elif any(x in layer.lower() for x in ['esa']):
                patterns['👁️ ESA空间注意力'].append(layer)
                categorized = True

            # ACA通道注意力 - 核心创新点5
            elif any(x in layer.lower() for x in ['aca']):
                patterns['📊 ACA通道注意力'].append(layer)
                categorized = True

            # 残差注意力
            elif any(x in layer.lower() for x in ['res_attn', 'sa']):
                patterns['🎯 残差注意力'].append(layer)
                categorized = True

            # Ghost模块
            elif any(x in layer.lower() for x in ['ghost']):
                patterns['👻 Ghost模块'].append(layer)
                categorized = True

            # 解码器
            elif any(x in layer.lower() for x in ['upconv', 'decoder']):
                patterns['⬆️ 解码器模块'].append(layer)
                categorized = True

            # 其他卷积
            elif 'conv' in layer.lower():
                patterns['📦 其他卷积层'].append(layer)
                categorized = True

        # 显示分类结果
        for category, layers in patterns.items():
            if layers:
                print(f"\n{category} ({len(layers)}个):")
                for layer in layers:
                    print(f"   • {layer}")

        # 生成论文级别的GradCAM推荐层
        print(f"\n🎯 论文级别GradCAM推荐层:")
        recommended_layers = []

        # Figure 1: 核心创新对比
        print(f"\n📊 Figure 1 - 核心创新对比:")
        fig1_layers = []

        # GDG模块的最终卷积层
        gdg_layers = patterns['🔥 GDG下采样模块']
        gdg_final = [l for l in gdg_layers if 'conv3_2' in l or l.endswith('conv2')]
        if gdg_final:
            fig1_layers.extend(gdg_final[:3])  # 前3个GDG层
            print(f"   🔥 GDG层: {gdg_final[:3]}")

        # Mamba的关键阶段
        mamba_layers = patterns['🐍 Mamba编码器']
        mamba_stages = [l for l in mamba_layers if 'stage' in l]
        if mamba_stages:
            fig1_layers.extend(mamba_stages[:2])  # 前2个Mamba阶段
            print(f"   🐍 Mamba层: {mamba_stages[:2]}")

        recommended_layers.extend(fig1_layers)

        # Figure 2: 多尺度特征演进
        print(f"\n📈 Figure 2 - 多尺度特征演进:")
        if len(gdg_final) >= 3:
            fig2_layers = gdg_final[:3]
            print(f"   📏 多尺度GDG: {fig2_layers}")
            # 不重复添加，因为已经在fig1中了

        # Figure 3: 注意力机制可视化
        print(f"\n👁️ Figure 3 - 注意力机制可视化:")
        fig3_layers = []

        # ESA注意力
        esa_layers = patterns['👁️ ESA空间注意力']
        if esa_layers:
            fig3_layers.extend(esa_layers[:2])
            print(f"   👁️ ESA层: {esa_layers[:2]}")

        # ACA注意力
        aca_layers = patterns['📊 ACA通道注意力']
        if aca_layers:
            fig3_layers.extend(aca_layers[:2])
            print(f"   📊 ACA层: {aca_layers[:2]}")

        # 残差注意力
        res_attn_layers = patterns['🎯 残差注意力']
        if res_attn_layers:
            fig3_layers.extend(res_attn_layers)
            print(f"   🎯 残差注意力: {res_attn_layers}")

        recommended_layers.extend(fig3_layers)

        # Figure 4: 融合机制分析
        print(f"\n🔗 Figure 4 - 融合机制分析:")
        fusion_layers = patterns['🔗 融合模块']
        if fusion_layers:
            gate_layers = [l for l in fusion_layers if 'gate' in l]
            if gate_layers:
                recommended_layers.extend(gate_layers[:3])
                print(f"   🔗 融合门控: {gate_layers[:3]}")

        # 去重并生成最终配置
        final_layers = list(dict.fromkeys(recommended_layers))  # 保持顺序的去重

        print(f"\n💻 最终GradCAM配置 ({len(final_layers)}个层):")
        print("PAPER_TARGET_LAYERS = [")
        for layer in final_layers:
            print(f"    '{layer}',")
        print("]")

        # 测试层的有效性
        print(f"\n🧪 测试层的GradCAM适用性:")
        valid_layers = []
        for layer in final_layers:
            weight_key = f"{layer}.weight"
            if weight_key in state_dict:
                weight_shape = state_dict[weight_key].shape
                if len(weight_shape) == 5:  # 3D卷积
                    print(f"   ✅ {layer} - 3D卷积, 形状: {weight_shape}")
                    valid_layers.append(layer)
                else:
                    print(f"   ⚠️  {layer} - 非3D卷积, 形状: {weight_shape}")
            else:
                print(f"   ❌ {layer} - 未找到权重")

        # 输出最终有效配置
        print(f"\n🎉 最终有效的GradCAM层 ({len(valid_layers)}个):")
        print("FINAL_GRADCAM_LAYERS = [")
        for layer in valid_layers:
            print(f"    '{layer}',")
        print("]")

        # 保存结果
        with open('gradcam_layer_config.txt', 'w', encoding='utf-8') as f:
            f.write("GDGMamU_Net GradCAM层配置\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"模型文件: {model_path}\n")


            f.write("推荐的论文展示层:\n")
            for i, layer in enumerate(valid_layers, 1):
                f.write(f"{i:2d}. '{layer}',\n")

            f.write(f"\n创新组件分类:\n")
            for category, layers in patterns.items():
                if layers:
                    f.write(f"\n{category}:\n")
                    for layer in layers:
                        f.write(f"  - {layer}\n")

        print(f"\n✅ 配置已保存到: gradcam_layer_config.txt")

        return valid_layers

    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 修改为你的实际模型路径
    model_path = "../results/best_model_WT0.879_ET0.809_TC0.851_AVG0.846.pth"

    print("🚀 GDGMamU_Net 修复版层分析工具")
    print("=" * 80)

    valid_layers = debug_model_layers_fixed(model_path)

    if valid_layers:
        print(f"\n🎊 成功! 找到 {len(valid_layers)} 个适合论文展示的GradCAM层")
        print("\n📋 下一步操作:")
        print("1. 复制上面的 FINAL_GRADCAM_LAYERS 到你的GradCAM代码")
        print("2. 修改 _auto_select_layers 方法返回这些层")
        print("3. 运行GradCAM生成论文级别的可视化")

        print(f"\n💡 论文展示建议:")
        print("• Figure 1: 使用GDG和Mamba层展示核心创新对比")
        print("• Figure 2: 使用不同尺度的GDG层展示多尺度学习")
        print("• Figure 3: 使用注意力层展示聚焦机制")
        print("• Figure 4: 使用融合层展示自适应融合效果")
    else:
        print("❌ 未能找到有效的GradCAM层，请检查模型文件")