#!/usr/bin/env python3
"""
修正统计计算错误
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.encode import events_to_voxel, load_h5_events

def fix_stats_calculation():
    """修正统计计算"""
    
    print("=== 修正统计计算 ===")
    
    # 加载数据
    original_events = load_h5_events("testdata/light_source_sequence_00018.h5")
    
    # 基本信息
    ts = original_events[:, 0]
    duration_us = ts.max() - ts.min()  # 微秒
    duration_ms = duration_us / 1000   # 毫秒
    
    print(f"总事件数: {len(original_events):,}")
    print(f"总时长: {duration_us:,} 微秒 = {duration_ms:.1f} ms")
    print(f"事件率: {len(original_events)/(duration_us/1e6):.0f} events/sec")
    
    # 分析不同分辨率
    for num_bins in [8, 16, 32, 64]:
        print(f"\n=== {num_bins} bins 分析 ===")
        
        # 计算正确的每bin时长
        bin_duration_us = duration_us / num_bins
        bin_duration_ms = bin_duration_us / 1000
        
        print(f"每个bin时长: {bin_duration_ms:.2f} ms ({bin_duration_us:.0f} 微秒)")
        print(f"理论每bin事件数: {len(original_events)/num_bins:.0f}")
        
        # 编码到voxel
        voxel = events_to_voxel(original_events, num_bins=num_bins, sensor_size=(480, 640))
        
        # 手动统计每个bin
        dt = duration_us / num_bins  # 微秒
        bin_indices = np.clip(((ts - ts.min()) / dt).astype(int), 0, num_bins - 1)
        
        # 计算正确的信息保留率
        original_total_events = len(original_events)
        original_polarity_sum = original_events[:, 3].sum()  # 净极性
        original_abs_polarity_sum = np.sum(np.abs(original_events[:, 3]))  # 绝对极性总和
        
        voxel_net_sum = voxel.sum().item()  # voxel的净极性
        voxel_abs_sum = torch.sum(torch.abs(voxel)).item()  # voxel的绝对值总和
        
        print(f"原始数据:")
        print(f"  总事件数: {original_total_events:,}")
        print(f"  净极性和: {original_polarity_sum:,}")
        print(f"  绝对极性和: {original_abs_polarity_sum:,}")
        
        print(f"Voxel结果:")
        print(f"  净极性和: {voxel_net_sum:,}")
        print(f"  绝对值和: {voxel_abs_sum:,}")
        
        print(f"保留率:")
        print(f"  净极性保留率: {abs(voxel_net_sum)/abs(original_polarity_sum)*100:.1f}%")
        print(f"  绝对信息保留率: {voxel_abs_sum/original_abs_polarity_sum*100:.1f}%")
        
        # 分析抵消情况
        cancellation_ratios = []
        events_per_bin = []
        
        for bin_idx in range(num_bins):
            bin_mask = bin_indices == bin_idx
            bin_events = original_events[bin_mask]
            
            if len(bin_events) > 0:
                total = len(bin_events)
                pos = np.sum(bin_events[:, 3] > 0)
                neg = np.sum(bin_events[:, 3] < 0)
                net = pos - neg
                cancel_rate = 1 - abs(net)/total
                
                events_per_bin.append(total)
                cancellation_ratios.append(cancel_rate)
            else:
                events_per_bin.append(0)
                cancellation_ratios.append(0)
        
        cancellation_ratios = np.array(cancellation_ratios)
        events_per_bin = np.array(events_per_bin)
        
        print(f"抵消分析:")
        print(f"  平均抵消率: {cancellation_ratios.mean():.1%}")
        print(f"  抵消率>80%的bins: {np.sum(cancellation_ratios > 0.8)} / {num_bins}")
        print(f"  抵消率>90%的bins: {np.sum(cancellation_ratios > 0.9)} / {num_bins}")
        print(f"  平均事件数/bin: {events_per_bin.mean():.0f}")
    
    # 特别分析32 bins的情况
    print(f"\n=== 32 bins 详细分析 (前10个bins) ===")
    voxel_32 = events_to_voxel(original_events, num_bins=32, sensor_size=(480, 640))
    dt_32 = duration_us / 32
    bin_indices_32 = np.clip(((ts - ts.min()) / dt_32).astype(int), 0, 31)
    
    for i in range(10):  # 只显示前10个
        bin_mask = bin_indices_32 == i
        bin_events = original_events[bin_mask]
        
        if len(bin_events) > 0:
            pos = np.sum(bin_events[:, 3] > 0)
            neg = np.sum(bin_events[:, 3] < 0)
            total = len(bin_events)
            cancel_rate = 1 - abs(pos-neg)/total
            
            print(f"Bin {i:2d}: {total:5,} events (+{pos:5,}/-{neg:5,}) 抵消率={cancel_rate:.1%} ({dt_32/1000:.2f}ms)")

if __name__ == "__main__":
    fix_stats_calculation()