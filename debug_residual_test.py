#!/usr/bin/env python3
"""
ResidualUNet3D 残差行为验证测试
检查模型是否真正实现端到端残差学习
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add project path
sys.path.append(str(Path(__file__).parent))
from src.training.training_factory import TrainingFactory

def test_residual_behavior():
    """测试ResidualUNet3D的初始残差行为"""
    print("=== ResidualUNet3D 残差行为测试 ===")
    
    # 创建模型配置
    config = {
        'model': {
            'name': 'ResidualUNet3D',
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
    
    print(f"Model class: {model.__class__.__name__}")
    print(f"Has final_activation: {hasattr(model, 'final_activation')}")
    if hasattr(model, 'final_activation'):
        print(f"Final activation: {model.final_activation}")
    
    # 测试1: 全零输入
    print("\n--- Test 1: 全零输入 ---")
    zero_input = torch.zeros(1, 1, 8, 480, 640)
    with torch.no_grad():
        zero_output = model(zero_input)
    
    print(f"Input: mean={zero_input.mean():.6f}, std={zero_input.std():.6f}")
    print(f"Output: mean={zero_output.mean():.6f}, std={zero_output.std():.6f}")
    print(f"是否恒等映射 (零输入): {torch.allclose(zero_input, zero_output, atol=1e-6)}")
    
    # 测试2: 小随机输入
    print("\n--- Test 2: 小随机输入 ---")
    small_input = torch.randn(1, 1, 8, 480, 640) * 0.1  # 小幅度随机
    with torch.no_grad():
        small_output = model(small_input)
    
    print(f"Input: mean={small_input.mean():.6f}, std={small_input.std():.6f}")
    print(f"Output: mean={small_output.mean():.6f}, std={small_output.std():.6f}")
    residual = small_output - small_input
    print(f"Residual: mean={residual.mean():.6f}, std={residual.std():.6f}")
    print(f"残差相对大小: {residual.abs().mean() / small_input.abs().mean():.3f}")
    
    # 测试3: 典型voxel输入 (正数为主)
    print("\n--- Test 3: 典型voxel输入 ---")
    voxel_input = torch.abs(torch.randn(1, 1, 8, 480, 640)) * 2.0  # 正数voxel
    with torch.no_grad():
        voxel_output = model(voxel_input)
    
    print(f"Input: mean={voxel_input.mean():.6f}, std={voxel_input.std():.6f}, min={voxel_input.min():.6f}")
    print(f"Output: mean={voxel_output.mean():.6f}, std={voxel_output.std():.6f}, min={voxel_output.min():.6f}")
    
    # 关键测试：输出是否保持输入的主要特征
    input_positive_ratio = (voxel_input > 0).float().mean()
    output_positive_ratio = (voxel_output > 0).float().mean()
    print(f"输入正值比例: {input_positive_ratio:.3f}")
    print(f"输出正值比例: {output_positive_ratio:.3f}")
    
    # 能量保持测试
    input_energy = voxel_input.sum()
    output_energy = voxel_output.sum()
    print(f"输入总能量: {input_energy:.1f}")
    print(f"输出总能量: {output_energy:.1f}")
    print(f"能量保持率: {(output_energy / input_energy):.3f}")
    
    # 测试4: 检查是否真的是残差架构
    print("\n--- Test 4: 残差架构验证 ---")
    
    # 如果是真正的残差学习，对于未训练网络：
    # output ≈ input + small_residual
    # 因此 output - input 应该很小
    
    identity_test = torch.ones(1, 1, 8, 480, 640) * 5.0  # 常数输入
    with torch.no_grad():
        identity_output = model(identity_test)
    
    difference = (identity_output - identity_test).abs()
    print(f"恒等映射测试 (常数5.0):")
    print(f"  |output - input|: mean={difference.mean():.6f}, max={difference.max():.6f}")
    print(f"  相对误差: {(difference.mean() / identity_test.mean()):.3f}")
    
    if difference.mean() / identity_test.mean() < 0.1:
        print("✅ 可能是真正的残差学习架构")
    else:
        print("❌ 不是端到端残差学习，更像传统UNet")
    
    # 最终判断
    print(f"\n=== 结论 ===")
    if (difference.mean() / identity_test.mean()) > 0.5:
        print("❌ ResidualUNet3D不是端到端残差学习!")
        print("   - 初始状态输出与输入差异很大")
        print("   - 背景信息丢失是预期行为")
        print("   - 需要充分训练才能保持有用信息")
    else:
        print("✅ ResidualUNet3D可能实现了残差学习")
        print("   - 初始状态接近恒等映射") 
        print("   - 背景信息丢失可能是其他原因")

if __name__ == "__main__":
    test_residual_behavior()