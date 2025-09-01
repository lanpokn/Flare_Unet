#!/usr/bin/env python3
"""
测试编码精度问题的专门脚本
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.encode import events_to_voxel, load_h5_events

def test_encode_accuracy():
    """测试编码精度"""
    
    print("=== 编码精度测试 ===")
    
    # 1. 创建简单测试数据
    print("\n1. 简单测试数据:")
    # 4个事件，分布在2个时间bin中
    simple_events = np.array([
        [0, 100, 200, 1],    # t=0, x=100, y=200, p=+1
        [500, 100, 200, -1], # t=500, x=100, y=200, p=-1 (同一像素)
        [1000, 150, 250, 1], # t=1000, x=150, y=250, p=+1
        [1500, 150, 250, 1], # t=1500, x=150, y=250, p=+1 (同一像素)
    ])
    
    voxel_simple = events_to_voxel(simple_events, num_bins=2, sensor_size=(300, 200))
    print(f"简单测试:")
    print(f"  原始事件: 4个")
    print(f"  Voxel总和: {voxel_simple.sum():.0f}")
    print(f"  Bin 0 在 (200,100): {voxel_simple[0, 200, 100]:.0f} (应该是0)")
    print(f"  Bin 1 在 (200,100): {voxel_simple[1, 200, 100]:.0f} (应该是0)")
    print(f"  Bin 0 在 (250,150): {voxel_simple[0, 250, 150]:.0f} (应该是1)")
    print(f"  Bin 1 在 (250,150): {voxel_simple[1, 250, 150]:.0f} (应该是1)")
    
    # 2. 测试真实数据的精度
    print("\n2. 真实数据编码测试:")
    original_events = load_h5_events("testdata/light_source_sequence_00018.h5")
    
    # 只取前1000个事件进行详细测试
    small_sample = original_events[:1000]
    voxel_small = events_to_voxel(small_sample, num_bins=4, sensor_size=(480, 640))
    
    print(f"小样本测试 (前1000个事件):")
    print(f"  原始事件总和: {len(small_sample)}")
    print(f"  原始极性总和: {small_sample[:, 3].sum()}")
    print(f"  Voxel总和: {voxel_small.sum():.0f}")
    print(f"  差异: {small_sample[:, 3].sum() - voxel_small.sum():.0f}")
    
    # 手动验证时间分bin
    ts = small_sample[:, 0]
    t_min, t_max = ts.min(), ts.max()
    dt = (t_max - t_min) / 4
    bin_indices = np.clip(((ts - t_min) / dt).astype(int), 0, 3)
    
    print(f"  时间范围: {t_min} - {t_max} ({(t_max-t_min)/1000:.1f}ms)")
    print(f"  Bin宽度: {dt/1000:.1f}ms")
    
    for bin_idx in range(4):
        bin_mask = bin_indices == bin_idx
        bin_events = small_sample[bin_mask]
        bin_polarity_sum = bin_events[:, 3].sum() if len(bin_events) > 0 else 0
        voxel_bin_sum = torch.sum(voxel_small[bin_idx]).item()
        
        print(f"  Bin {bin_idx}: {len(bin_events)}个事件, 极性总和={bin_polarity_sum}, voxel总和={voxel_bin_sum:.0f}")
    
    # 3. 检查是否有坐标越界问题
    print("\n3. 坐标越界检查:")
    xs, ys = original_events[:, 1].astype(int), original_events[:, 2].astype(int)
    out_of_bounds = (xs < 0) | (xs >= 640) | (ys < 0) | (ys >= 480)
    print(f"越界事件数量: {np.sum(out_of_bounds)} / {len(original_events)}")
    if np.any(out_of_bounds):
        print(f"越界坐标范围: x=[{xs[out_of_bounds].min()}, {xs[out_of_bounds].max()}], y=[{ys[out_of_bounds].min()}, {ys[out_of_bounds].max()}]")
    
    # 4. 测试完整数据集
    print("\n4. 完整数据集编码测试:")
    voxel_full = events_to_voxel(original_events, num_bins=16, sensor_size=(480, 640))
    
    # 手动计算每个bin应该有多少事件
    ts_full = original_events[:, 0]
    t_min_full, t_max_full = ts_full.min(), ts_full.max()
    dt_full = (t_max_full - t_min_full) / 16
    bin_indices_full = np.clip(((ts_full - t_min_full) / dt_full).astype(int), 0, 15)
    
    print(f"完整数据集:")
    print(f"  总事件数: {len(original_events):,}")
    print(f"  总极性和: {original_events[:, 3].sum()}")
    print(f"  Voxel总和: {voxel_full.sum():.0f}")
    print(f"  丢失事件数: {original_events[:, 3].sum() - voxel_full.sum():.0f}")
    print(f"  丢失比例: {(1 - voxel_full.sum() / original_events[:, 3].sum()) * 100:.1f}%")
    
    # 分析丢失事件的空间分布
    print(f"\n5. 丢失事件空间分布分析:")
    
    # 创建一个累积图来查看空间覆盖
    spatial_coverage = torch.zeros((480, 640))
    for bin_idx in range(16):
        spatial_coverage += torch.abs(voxel_full[bin_idx])
    
    # 对比原始事件的空间分布
    original_spatial = np.zeros((480, 640))
    for i in range(len(original_events)):
        x, y = int(original_events[i, 1]), int(original_events[i, 2])
        if 0 <= x < 640 and 0 <= y < 480:
            original_spatial[y, x] += abs(original_events[i, 3])
    
    non_zero_original = np.count_nonzero(original_spatial)
    non_zero_voxel = torch.count_nonzero(spatial_coverage).item()
    
    print(f"  原始非零像素: {non_zero_original:,}")
    print(f"  Voxel非零像素: {non_zero_voxel:,}")
    print(f"  空间覆盖率: {non_zero_voxel / non_zero_original * 100:.1f}%")

if __name__ == "__main__":
    test_encode_accuracy()