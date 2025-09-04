#!/usr/bin/env python3
"""
真正的残差学习包装器
包装pytorch-3dunet的UNet3D/ResidualUNet3D，实现端到端残差学习
"""

import torch
import torch.nn as nn
from pytorch3dunet.unet3d.model import ResidualUNet3D, UNet3D

class TrueResidualUNet3D(nn.Module):
    """
    真正的残差学习包装器
    
    架构: output = input + backbone_network(input)
    其中backbone_network学习残差，而不是直接学习目标
    """
    
    def __init__(self, 
                 in_channels=1,
                 out_channels=1, 
                 f_maps=[16, 32, 64],
                 num_levels=3,
                 backbone='ResidualUNet3D',
                 layer_order='gcr',
                 num_groups=8,
                 conv_padding=1,
                 dropout_prob=0.1,
                 **kwargs):
        """
        Args:
            backbone: 'ResidualUNet3D' or 'UNet3D'
        """
        super().__init__()
        
        if backbone == 'ResidualUNet3D':
            self.backbone = ResidualUNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                f_maps=f_maps,
                num_levels=num_levels,
                layer_order=layer_order,
                num_groups=num_groups,
                conv_padding=conv_padding,
                dropout_prob=dropout_prob,
                final_sigmoid=False,  # 我们要学习残差，可以是任意值
                **kwargs
            )
        elif backbone == 'UNet3D':
            self.backbone = UNet3D(
                in_channels=in_channels,
                out_channels=out_channels,
                f_maps=f_maps,
                num_levels=num_levels,
                layer_order=layer_order,
                num_groups=num_groups,
                conv_padding=conv_padding,
                dropout_prob=dropout_prob,
                final_sigmoid=False,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # 移除可能的最终激活
        if hasattr(self.backbone, 'final_activation'):
            self.backbone.final_activation = nn.Identity()
        
        # 关键改进：零初始化最后一层，确保初始时残差≈0
        self._zero_init_residual_layer()
        
        print(f"TrueResidualUNet3D created with {backbone} backbone")
        print(f"Architecture: output = input + {backbone}(input)")
        print(f"Zero-initialized final layer for perfect identity mapping")
    
    def forward(self, x):
        """
        真正的残差前向传播:
        output = input + residual_learned
        """
        # 学习残差
        residual = self.backbone(x)
        
        # 残差连接
        output = x + residual
        
        return output
    
    def get_residual(self, x):
        """获取学习到的残差，用于调试分析"""
        return self.backbone(x)
    
    def _zero_init_residual_layer(self):
        """
        零初始化最后一层，确保初始时网络输出≈0
        这是残差学习的关键：初始时 output = input + 0 = input
        """
        # 查找最后的卷积层并零初始化
        last_conv = None
        
        def find_last_conv(module):
            nonlocal last_conv
            for name, child in module.named_children():
                if isinstance(child, (nn.Conv3d, nn.ConvTranspose3d)):
                    last_conv = child
                else:
                    find_last_conv(child)
        
        find_last_conv(self.backbone)
        
        if last_conv is not None:
            # 零初始化权重和偏置
            nn.init.zeros_(last_conv.weight)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)
            print(f"Zero-initialized final conv layer: {type(last_conv).__name__}")
            print(f"  Weight shape: {last_conv.weight.shape}")
            print(f"  Bias: {'Yes' if last_conv.bias is not None else 'No'}")
        else:
            print("Warning: Could not find final conv layer to zero-initialize")


def test_true_residual():
    """测试真正的残差学习行为"""
    print("=== 真正残差学习测试 ===")
    
    model = TrueResidualUNet3D(
        in_channels=1, 
        out_channels=1,
        f_maps=[16, 32, 64],
        backbone='ResidualUNet3D'
    )
    model.eval()
    
    # 测试1: 零输入应该输出接近零
    zero_input = torch.zeros(1, 1, 8, 480, 640)
    with torch.no_grad():
        zero_output = model(zero_input)
        zero_residual = model.get_residual(zero_input)
    
    print(f"\n零输入测试:")
    print(f"  输入: mean={zero_input.mean():.6f}")
    print(f"  残差: mean={zero_residual.mean():.6f}, std={zero_residual.std():.6f}")
    print(f"  输出: mean={zero_output.mean():.6f}, std={zero_output.std():.6f}")
    print(f"  恒等性: output ≈ input + residual = {torch.allclose(zero_output, zero_input + zero_residual)}")
    
    # 测试2: 常数输入测试
    const_input = torch.ones(1, 1, 8, 480, 640) * 5.0
    with torch.no_grad():
        const_output = model(const_input)
        const_residual = model.get_residual(const_input)
    
    print(f"\n常数输入(5.0)测试:")
    print(f"  输入: mean={const_input.mean():.6f}")
    print(f"  残差: mean={const_residual.mean():.6f}, std={const_residual.std():.6f}")
    print(f"  输出: mean={const_output.mean():.6f}, std={const_output.std():.6f}")
    
    # 关键测试：未训练时输出应该接近输入
    diff = (const_output - const_input).abs().mean()
    relative_diff = diff / const_input.abs().mean()
    print(f"  |输出-输入|平均: {diff:.6f}")
    print(f"  相对差异: {relative_diff:.3f}")
    
    if relative_diff < 0.2:  # 20%以内认为是好的残差学习
        print("✅ 真正的残差学习：未训练时输出≈输入")
    else:
        print("❌ 残差学习不理想")
    
    # 测试3: 实际voxel数据测试
    voxel_input = torch.abs(torch.randn(1, 1, 8, 480, 640)) * 2.0
    with torch.no_grad():
        voxel_output = model(voxel_input)
        voxel_residual = model.get_residual(voxel_input)
    
    print(f"\nVoxel输入测试:")
    print(f"  输入能量: {voxel_input.sum():.1f}")
    print(f"  残差能量: {voxel_residual.sum():.1f}")
    print(f"  输出能量: {voxel_output.sum():.1f}")
    print(f"  能量保持率: {(voxel_output.sum() / voxel_input.sum()):.3f}")
    
    # 背景保持测试
    input_positive_ratio = (voxel_input > 0).float().mean()
    output_positive_ratio = (voxel_output > 0).float().mean()
    print(f"  输入正值比例: {input_positive_ratio:.3f}")
    print(f"  输出正值比例: {output_positive_ratio:.3f}")


if __name__ == "__main__":
    test_true_residual()