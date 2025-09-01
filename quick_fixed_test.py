#!/usr/bin/env python3
"""
快速测试固定时间间隔编码
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.encode import events_to_voxel, load_h5_events

def quick_fixed_test():
    """快速测试固定时间编码"""
    
    print("=== 快速测试固定时间编码 ===")
    
    # 测试新的固定时间编码
    original_events = load_h5_events("testdata/light_source_sequence_00018.h5")
    
    print(f"\n测试数据:")
    print(f"事件数: {len(original_events):,}")
    print(f"时间范围: {original_events[:,0].min():.0f} - {original_events[:,0].max():.0f}μs")
    print(f"实际时长: {(original_events[:,0].max() - original_events[:,0].min())/1000:.1f}ms")
    
    # 测试固定时间编码
    print(f"\n测试固定时间编码 (32 bins, 100ms):")
    voxel = events_to_voxel(original_events, num_bins=32, sensor_size=(480, 640), fixed_duration_us=100000)
    
    print(f"Voxel形状: {voxel.shape}")
    print(f"Voxel总和: {voxel.sum():.0f}")
    print(f"每bin理论时长: {100/32:.2f}ms")
    
    # 测试一致性 - 截取一半数据
    print(f"\n测试时间一致性:")
    half_events = original_events[:len(original_events)//2]
    print(f"截取数据时长: {(half_events[:,0].max() - half_events[:,0].min())/1000:.1f}ms")
    
    voxel_half = events_to_voxel(half_events, num_bins=32, sensor_size=(480, 640), fixed_duration_us=100000)
    print(f"截取数据voxel总和: {voxel_half.sum():.0f}")
    print(f"✅ 两个voxel都使用相同的32×3.125ms时间网格")
    
    # 对比不同分辨率
    print(f"\n不同分辨率对比:")
    for bins in [16, 32, 64]:
        v = events_to_voxel(original_events[:1000], num_bins=bins, sensor_size=(480, 640), fixed_duration_us=100000)
        print(f"{bins:2d} bins: {100/bins:.2f}ms/bin, 小样本总和: {v.sum():.0f}")

if __name__ == "__main__":
    quick_fixed_test()