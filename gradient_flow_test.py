#!/usr/bin/env python3
"""
测试零初始化残差网络的梯度流
验证是否会出现梯度消失问题
"""

import torch
import torch.nn as nn
from true_residual_wrapper import TrueResidualUNet3D

def test_gradient_flow():
    """测试梯度流强度"""
    print("=== 梯度流测试 ===")
    
    # 创建零初始化的残差网络
    model = TrueResidualUNet3D(
        in_channels=1, 
        out_channels=1,
        f_maps=[16, 32, 64],
        backbone='ResidualUNet3D'
    )
    
    # 创建测试数据
    batch_size = 1
    input_data = torch.randn(batch_size, 1, 8, 480, 640, requires_grad=True)
    target_data = torch.randn(batch_size, 1, 8, 480, 640)
    
    criterion = nn.MSELoss()
    
    print(f"输入数据: shape={input_data.shape}, mean={input_data.mean():.4f}")
    print(f"目标数据: shape={target_data.shape}, mean={target_data.mean():.4f}")
    
    # 前向传播
    output = model(input_data)
    residual = model.get_residual(input_data)
    
    print(f"\n前向传播结果:")
    print(f"  残差输出: mean={residual.mean():.6f}, std={residual.std():.6f}")
    print(f"  最终输出: mean={output.mean():.4f}, std={output.std():.4f}")
    print(f"  输出≈输入: {torch.allclose(output, input_data, atol=1e-6)}")
    
    # 计算损失
    loss = criterion(output, target_data)
    print(f"\n损失: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    print(f"\n=== 梯度分析 ===")
    
    # 1. 输入梯度
    if input_data.grad is not None:
        input_grad_norm = input_data.grad.norm().item()
        print(f"输入梯度范数: {input_grad_norm:.4f}")
    
    # 2. 网络各层梯度
    gradient_norms = []
    layer_names = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_norms.append(grad_norm)
            layer_names.append(name)
            
            # 只显示前几层和后几层
            if len(gradient_norms) <= 5 or len(gradient_norms) > len(list(model.parameters())) - 5:
                print(f"{name}: grad_norm={grad_norm:.6f}, param_norm={param.norm().item():.6f}")
    
    print(f"\n=== 梯度统计 ===")
    if gradient_norms:
        print(f"梯度范数统计:")
        print(f"  最小值: {min(gradient_norms):.6f}")
        print(f"  最大值: {max(gradient_norms):.6f}")
        print(f"  平均值: {sum(gradient_norms)/len(gradient_norms):.6f}")
        print(f"  中位数: {sorted(gradient_norms)[len(gradient_norms)//2]:.6f}")
        
        # 梯度消失检测
        small_gradients = [g for g in gradient_norms if g < 1e-6]
        print(f"\n梯度消失检测:")
        print(f"  总参数组数: {len(gradient_norms)}")
        print(f"  小梯度(<1e-6)数量: {len(small_gradients)}")
        print(f"  小梯度比例: {len(small_gradients)/len(gradient_norms)*100:.1f}%")
        
        if len(small_gradients) / len(gradient_norms) > 0.5:
            print("❌ 可能存在梯度消失问题")
        else:
            print("✅ 梯度流正常")
            
    # 3. 特殊检查：最后一层梯度（零初始化层）
    print(f"\n=== 最后一层分析 ===")
    backbone_params = list(model.backbone.parameters())
    if backbone_params:
        last_param = backbone_params[-1]  # 最后一个参数
        if last_param.grad is not None:
            last_grad_norm = last_param.grad.norm().item()
            print(f"最后一层梯度范数: {last_grad_norm:.6f}")
            print(f"最后一层参数范数: {last_param.norm().item():.6f}")
            
            if last_grad_norm > 1e-8:
                print("✅ 最后一层能够接收梯度，可以学习")
            else:
                print("❌ 最后一层梯度过小，可能无法学习")


def compare_with_normal_unet():
    """对比普通UNet和残差UNet的梯度"""
    print(f"\n" + "="*50)
    print("=== 对比测试：普通UNet vs 残差UNet ===")
    
    from pytorch3dunet.unet3d.model import ResidualUNet3D
    
    # 普通ResidualUNet3D (非零初始化)
    normal_model = ResidualUNet3D(in_channels=1, out_channels=1, f_maps=[16, 32, 64])
    
    # 我们的零初始化残差UNet
    residual_model = TrueResidualUNet3D(in_channels=1, out_channels=1, f_maps=[16, 32, 64])
    
    input_data = torch.randn(1, 1, 8, 480, 640)
    target_data = torch.randn(1, 1, 8, 480, 640)
    criterion = nn.MSELoss()
    
    models = [
        ("普通ResidualUNet3D", normal_model),
        ("零初始化残差UNet", residual_model)
    ]
    
    for name, model in models:
        print(f"\n--- {name} ---")
        
        # 前向传播
        output = model(input_data)
        loss = criterion(output, target_data)
        
        print(f"输出与输入差异: {(output - input_data).abs().mean():.6f}")
        print(f"损失值: {loss.item():.4f}")
        
        # 反向传播
        model.zero_grad()
        loss.backward()
        
        # 检查梯度
        gradients = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        if gradients:
            print(f"平均梯度范数: {sum(gradients)/len(gradients):.6f}")
            print(f"梯度范数范围: [{min(gradients):.6f}, {max(gradients):.6f}]")
        
        # 检查最后一层
        last_params = list(model.parameters())[-2:]  # 最后两个参数（权重+偏置）
        for i, param in enumerate(last_params):
            if param.grad is not None:
                print(f"最后层参数{i}: grad_norm={param.grad.norm().item():.6f}")


if __name__ == "__main__":
    test_gradient_flow()
    compare_with_normal_unet()