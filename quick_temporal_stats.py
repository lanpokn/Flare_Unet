#!/usr/bin/env python3
"""
快速分析时间分辨率统计
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.encode import events_to_voxel, load_h5_events

def quick_temporal_stats():
    """快速统计时间分辨率"""
    
    print("=== 快速时间分辨率统计 ===")
    
    # 加载数据
    original_events = load_h5_events("testdata/light_source_sequence_00018.h5")
    
    # 基本信息
    ts = original_events[:, 0]
    duration = (ts.max() - ts.min()) / 1000  # ms
    
    print(f"总事件数: {len(original_events):,}")
    print(f"总时长: {duration:.1f} ms")
    print(f"事件率: {len(original_events)/(duration/1000):.0f} events/sec")
    
    # 分析16个bins的情况
    print(f"\n=== 16 bins (当前默认) 分析 ===")
    
    voxel_16 = events_to_voxel(original_events, num_bins=16, sensor_size=(480, 640))
    
    # 手动统计每个bin
    dt = (ts.max() - ts.min()) / 16
    bin_indices = np.clip(((ts - ts.min()) / dt).astype(int), 0, 15)
    
    print(f"每个bin时长: {dt/1000:.2f} ms")
    
    # 统计每个bin的信息
    total_events_per_bin = []
    pos_events_per_bin = []
    neg_events_per_bin = []
    net_events_per_bin = []
    cancellation_per_bin = []
    
    for bin_idx in range(16):
        bin_mask = bin_indices == bin_idx
        bin_events = original_events[bin_mask]
        
        if len(bin_events) > 0:
            total = len(bin_events)
            pos = np.sum(bin_events[:, 3] > 0)
            neg = np.sum(bin_events[:, 3] < 0)
            net = pos - neg
            cancel_rate = 1 - abs(net)/total
            
            total_events_per_bin.append(total)
            pos_events_per_bin.append(pos)
            neg_events_per_bin.append(neg)
            net_events_per_bin.append(abs(net))
            cancellation_per_bin.append(cancel_rate)
    
    # 统计摘要
    total_events_per_bin = np.array(total_events_per_bin)
    net_events_per_bin = np.array(net_events_per_bin)
    cancellation_per_bin = np.array(cancellation_per_bin)
    
    print(f"\n每个bin事件数量:")
    print(f"  平均: {total_events_per_bin.mean():.0f} events/bin")
    print(f"  范围: {total_events_per_bin.min():.0f} - {total_events_per_bin.max():.0f}")
    print(f"  标准差: {total_events_per_bin.std():.0f}")
    
    print(f"\n正负抵消情况:")
    print(f"  平均抵消率: {cancellation_per_bin.mean():.1%}")
    print(f"  抵消率范围: {cancellation_per_bin.min():.1%} - {cancellation_per_bin.max():.1%}")
    print(f"  抵消率>80%的bins: {np.sum(cancellation_per_bin > 0.8)} / 16")
    print(f"  抵消率>90%的bins: {np.sum(cancellation_per_bin > 0.9)} / 16")
    
    print(f"\n信息保留:")
    original_sum = abs(original_events[:, 3].sum())
    voxel_sum = torch.sum(torch.abs(voxel_16)).item()
    print(f"  原始事件极性绝对值总和: {original_sum:.0f}")
    print(f"  Voxel保留信息量: {voxel_sum:.0f}")
    print(f"  信息保留率: {voxel_sum/original_sum:.1%}")
    
    # 快速对比其他分辨率
    print(f"\n=== 不同分辨率对比 ===")
    
    for num_bins in [8, 32, 64]:
        voxel = events_to_voxel(original_events, num_bins=num_bins, sensor_size=(480, 640))
        info_retention = torch.sum(torch.abs(voxel)).item() / original_sum
        avg_events_per_bin = len(original_events) / num_bins
        bin_duration = duration / num_bins
        
        print(f"{num_bins:2d} bins: {bin_duration:.2f}ms/bin, {avg_events_per_bin:.0f} events/bin, 信息保留率: {info_retention:.1%}")
    
    # 显示具体的bin分布
    print(f"\n=== 16 bins 详细分布 ===")
    for i in range(16):
        bin_mask = bin_indices == i
        bin_events = original_events[bin_mask]
        
        if len(bin_events) > 0:
            pos = np.sum(bin_events[:, 3] > 0)
            neg = np.sum(bin_events[:, 3] < 0)
            total = len(bin_events)
            cancel_rate = 1 - abs(pos-neg)/total
            
            print(f"Bin {i:2d}: {total:6,} events (+{pos:6,}/-{neg:6,}) 抵消率={cancel_rate:.1%}")

if __name__ == "__main__":
    quick_temporal_stats()