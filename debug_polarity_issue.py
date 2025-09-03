#!/usr/bin/env python3
"""
调试输出只有正值的问题
系统地验证每个环节的数据极性保持
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent))
from src.training.training_factory import TrainingFactory
from src.datasets.event_voxel_dataset import EventVoxelDataset

def analyze_tensor_polarity(tensor, name):
    """分析tensor的极性分布"""
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    
    pos_count = (tensor > 0).sum().item()
    neg_count = (tensor < 0).sum().item() 
    zero_count = (tensor == 0).sum().item()
    total = tensor.numel()
    
    print(f"\n{name} 极性分析:")
    print(f"  形状: {tensor.shape}")
    print(f"  范围: [{tensor.min():.6f}, {tensor.max():.6f}]")
    print(f"  均值: {tensor.mean():.6f}, 标准差: {tensor.std():.6f}")
    print(f"  正值: {pos_count:,} ({pos_count/total*100:.1f}%)")
    print(f"  负值: {neg_count:,} ({neg_count/total*100:.1f}%)")
    print(f"  零值: {zero_count:,} ({zero_count/total*100:.1f}%)")
    
    return {
        'pos_ratio': pos_count/total,
        'neg_ratio': neg_count/total,
        'zero_ratio': zero_count/total,
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item()
    }

def test_data_loading():
    """测试数据加载的极性保持"""
    print("="*60)
    print("1. 测试数据加载极性保持")
    
    # 跳过真实数据加载，直接使用合成测试数据
    print("使用合成测试数据（模拟真实voxel特性）:")
    
    # 创建模拟真实voxel的数据：主要为正值但有负值
    input_voxel = torch.abs(torch.randn(8, 480, 640)) * 2  # 主要正值
    # 添加一些负值区域
    neg_mask = torch.rand(8, 480, 640) < 0.2  # 20%为负值
    input_voxel[neg_mask] = -input_voxel[neg_mask]
    
    target_voxel = torch.abs(torch.randn(8, 480, 640)) * 1.5  # 目标通常更干净
    neg_mask_target = torch.rand(8, 480, 640) < 0.15  # 15%为负值  
    target_voxel[neg_mask_target] = -target_voxel[neg_mask_target]
    
    analyze_tensor_polarity(input_voxel, "合成输入Voxel")
    analyze_tensor_polarity(target_voxel, "合成真值Voxel")
    
    return input_voxel, target_voxel

def test_model_output():
    """测试模型输出极性"""
    print("="*60)
    print("2. 测试模型输出极性")
    
    # 配置
    config = {
        'model': {
            'name': 'TrueResidualUNet3D',
            'backbone': 'ResidualUNet3D',
            'in_channels': 1,
            'out_channels': 1,
            'f_maps': [16, 32, 64],
            'num_levels': 3
        }
    }
    
    # 创建模型
    factory = TrainingFactory(config)
    model = factory.create_model()
    model.eval()
    
    print(f"模型类型: {model.__class__.__name__}")
    print(f"是否有get_residual方法: {hasattr(model, 'get_residual')}")
    
    # 测试数据 - 确保有正负值
    test_input = torch.randn(1, 1, 8, 480, 640) * 3  # 放大范围确保有明显正负值
    analyze_tensor_polarity(test_input, "测试输入")
    
    with torch.no_grad():
        # 模型完整输出
        final_output = model(test_input)
        analyze_tensor_polarity(final_output, "模型最终输出")
        
        # 如果是TrueResidualUNet3D，分析残差
        if hasattr(model, 'get_residual'):
            residual_output = model.get_residual(test_input)
            analyze_tensor_polarity(residual_output, "残差输出 (backbone)")
            
            # 验证恒等性: output = input + residual
            manual_output = test_input + residual_output
            identity_check = torch.allclose(final_output, manual_output, atol=1e-6)
            print(f"\n恒等性验证: final_output ≈ input + residual = {identity_check}")
            
            if not identity_check:
                diff = (final_output - manual_output).abs()
                print(f"差异: max={diff.max():.6f}, mean={diff.mean():.6f}")
        else:
            print("⚠️ 不是TrueResidualUNet3D模型，无法分析残差")
    
    return model, test_input, final_output

def test_encoding_decoding():
    """测试编解码极性保持"""
    print("="*60)
    print("3. 测试编解码极性保持")
    
    from src.data_processing.encode import events_to_voxel
    from src.data_processing.decode import voxel_to_events
    
    # 创建包含正负极性的测试events
    n_events = 10000
    test_events = np.column_stack([
        np.random.uniform(0, 20000, n_events),  # timestamps
        np.random.randint(0, 640, n_events),    # x
        np.random.randint(0, 480, n_events),    # y  
        np.random.choice([-1, 1], n_events)     # polarity: 50% positive, 50% negative
    ])
    
    pos_events = (test_events[:, 3] == 1).sum()
    neg_events = (test_events[:, 3] == -1).sum()
    print(f"\n原始Events: {pos_events:,} 正极性, {neg_events:,} 负极性")
    
    # 编码到voxel
    voxel = events_to_voxel(test_events, num_bins=8, sensor_size=(480, 640), fixed_duration_us=20000)
    if isinstance(voxel, np.ndarray):
        voxel_tensor = torch.from_numpy(voxel)
    else:
        voxel_tensor = voxel
    analyze_tensor_polarity(voxel_tensor, "编码后Voxel")
    
    # 解码回events
    decoded_events = voxel_to_events(voxel_tensor, total_duration=20000, sensor_size=(480, 640))
    
    if len(decoded_events) > 0:
        pos_decoded = (decoded_events[:, 3] == 1).sum()
        neg_decoded = (decoded_events[:, 3] == -1).sum()
        print(f"\n解码后Events: {pos_decoded:,} 正极性, {neg_decoded:,} 负极性")
        print(f"极性保持率: 正={pos_decoded/pos_events*100:.1f}%, 负={neg_decoded/neg_events*100:.1f}%")
    else:
        print("⚠️ 解码后无events")

def test_complete_pipeline():
    """测试完整pipeline"""
    print("="*60)
    print("4. 测试完整Pipeline极性流")
    
    # 创建有正负值的voxel
    input_voxel = torch.randn(1, 1, 8, 480, 640) * 2
    analyze_tensor_polarity(input_voxel, "Pipeline输入")
    
    # 模型配置
    config = {
        'model': {
            'name': 'TrueResidualUNet3D', 
            'backbone': 'ResidualUNet3D',
            'in_channels': 1,
            'out_channels': 1,
            'f_maps': [16, 32, 64],
            'num_levels': 3
        }
    }
    
    # 创建模型
    factory = TrainingFactory(config)
    model = factory.create_model()
    model.eval()
    
    with torch.no_grad():
        output_voxel = model(input_voxel)
        analyze_tensor_polarity(output_voxel, "Pipeline输出")
        
        # 解码测试
        from src.data_processing.decode import voxel_to_events
        output_events = voxel_to_events(output_voxel[0, 0], total_duration=20000, sensor_size=(480, 640))
        
        if len(output_events) > 0:
            pos_events = (output_events[:, 3] == 1).sum()
            neg_events = (output_events[:, 3] == -1).sum()
            print(f"\n最终Events: {pos_events:,} 正极性, {neg_events:,} 负极性")
        else:
            print("⚠️ 最终无events输出")

def main():
    """主调试流程"""
    print("🐛 EVENT-VOXEL 极性调试工具")
    print("调试输出只有正值的问题")
    
    try:
        # 1. 数据加载测试
        input_voxel, target_voxel = test_data_loading()
        
        # 2. 模型输出测试  
        model, test_input, final_output = test_model_output()
        
        # 3. 编解码测试
        test_encoding_decoding()
        
        # 4. 完整pipeline测试
        test_complete_pipeline()
        
        print("="*60)
        print("调试完成！请检查上述分析结果。")
        
    except Exception as e:
        print(f"调试过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()