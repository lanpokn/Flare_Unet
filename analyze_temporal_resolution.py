#!/usr/bin/env python3
"""
分析100ms分16份的时间分辨率是否足够
统计每个bin的事件数量和正负抵消情况
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.encode import events_to_voxel, load_h5_events

def analyze_temporal_resolution():
    """分析时间分辨率和正负抵消情况"""
    
    print("=== 时间分辨率分析 ===")
    
    # 加载数据
    testdata_path = "testdata/light_source_sequence_00018.h5"
    sensor_size = (480, 640)
    
    original_events = load_h5_events(testdata_path)
    print(f"总事件数: {len(original_events):,}")
    
    # 时间信息
    ts = original_events[:, 0]
    t_min, t_max = ts.min(), ts.max()
    duration = t_max - t_min
    
    print(f"时间范围: {t_min} - {t_max} 微秒")
    print(f"总时长: {duration:,} 微秒 ({duration/1000:.1f} ms)")
    print(f"平均事件率: {len(original_events)/(duration/1e6):.0f} events/sec")
    
    # 分析不同的时间分辨率
    for num_bins in [8, 16, 32, 64]:
        print(f"\n=== {num_bins} 个时间bins分析 ===")
        
        # 编码到voxel
        voxel = events_to_voxel(original_events, num_bins=num_bins, sensor_size=sensor_size)
        
        # 手动分析每个bin
        dt = duration / num_bins
        bin_indices = np.clip(((ts - t_min) / dt).astype(int), 0, num_bins - 1)
        
        print(f"每个bin时长: {dt/1000:.2f} ms")
        print(f"理论每bin事件数: {len(original_events)/num_bins:.0f}")
        
        # 统计信息
        events_per_bin = []
        positive_per_bin = []
        negative_per_bin = []
        voxel_sums = []
        cancellation_ratios = []
        
        for bin_idx in range(num_bins):
            # 原始事件统计
            bin_mask = bin_indices == bin_idx
            bin_events = original_events[bin_mask]
            
            if len(bin_events) > 0:
                pos_events = np.sum(bin_events[:, 3] > 0)
                neg_events = np.sum(bin_events[:, 3] < 0)
                total_events = len(bin_events)
                
                # Voxel统计  
                voxel_sum = torch.sum(torch.abs(voxel[bin_idx])).item()
                net_polarity = pos_events - neg_events  # 净极性
                
                # 抵消率计算
                cancellation = 1 - (abs(net_polarity) / total_events) if total_events > 0 else 0
                
                events_per_bin.append(total_events)
                positive_per_bin.append(pos_events)
                negative_per_bin.append(neg_events)
                voxel_sums.append(voxel_sum)
                cancellation_ratios.append(cancellation)
                
            else:
                events_per_bin.append(0)
                positive_per_bin.append(0)
                negative_per_bin.append(0)
                voxel_sums.append(0)
                cancellation_ratios.append(0)
        
        # 统计摘要
        events_per_bin = np.array(events_per_bin)
        positive_per_bin = np.array(positive_per_bin)
        negative_per_bin = np.array(negative_per_bin)
        voxel_sums = np.array(voxel_sums)
        cancellation_ratios = np.array(cancellation_ratios)
        
        print(f"每bin事件数统计:")
        print(f"  平均: {events_per_bin.mean():.0f}")
        print(f"  最小: {events_per_bin.min():.0f}")
        print(f"  最大: {events_per_bin.max():.0f}")
        print(f"  标准差: {events_per_bin.std():.0f}")
        
        print(f"正负事件分布:")
        print(f"  平均正事件/bin: {positive_per_bin.mean():.0f}")
        print(f"  平均负事件/bin: {negative_per_bin.mean():.0f}")
        print(f"  正负比例: {positive_per_bin.sum()}/{negative_per_bin.sum()} = {positive_per_bin.sum()/negative_per_bin.sum():.2f}")
        
        print(f"正负抵消分析:")
        print(f"  平均抵消率: {cancellation_ratios.mean():.1%}")
        print(f"  最大抵消率: {cancellation_ratios.max():.1%}")
        print(f"  最小抵消率: {cancellation_ratios.min():.1%}")
        print(f"  抵消率>50%的bins: {np.sum(cancellation_ratios > 0.5)} / {num_bins}")
        print(f"  抵消率>90%的bins: {np.sum(cancellation_ratios > 0.9)} / {num_bins}")
        
        # 信息保留度
        original_total = len(original_events)
        voxel_total = voxel.sum().item()
        information_retention = abs(voxel_total) / abs(original_events[:, 3].sum())
        
        print(f"信息保留:")
        print(f"  原始极性总和: {original_events[:, 3].sum():.0f}")
        print(f"  Voxel极性总和: {voxel_total:.0f}")
        print(f"  信息保留率: {information_retention:.1%}")
        
        # 详细显示前5个和后5个bin
        print(f"\n前5个bins详细:")
        for i in range(min(5, num_bins)):
            print(f"  Bin {i:2d}: {events_per_bin[i]:5.0f} events (+{positive_per_bin[i]:5.0f}/-{negative_per_bin[i]:5.0f}), "
                  f"抵消率={cancellation_ratios[i]:.1%}, voxel_sum={voxel_sums[i]:.0f}")
        
        if num_bins > 10:
            print(f"后5个bins详细:")
            for i in range(max(num_bins-5, 5), num_bins):
                print(f"  Bin {i:2d}: {events_per_bin[i]:5.0f} events (+{positive_per_bin[i]:5.0f}/-{negative_per_bin[i]:5.0f}), "
                      f"抵消率={cancellation_ratios[i]:.1%}, voxel_sum={voxel_sums[i]:.0f}")
    
    # 创建对比图
    create_resolution_comparison_plot(original_events)

def create_resolution_comparison_plot(original_events):
    """创建不同时间分辨率的对比图"""
    
    sensor_size = (480, 640)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('不同时间分辨率的事件分布和抵消分析', fontsize=16)
    
    bin_configs = [8, 16, 32, 64]
    colors = ['blue', 'green', 'red', 'orange']
    
    for idx, (num_bins, color) in enumerate(zip(bin_configs, colors)):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        # 编码
        voxel = events_to_voxel(original_events, num_bins=num_bins, sensor_size=sensor_size)
        
        # 统计每个bin
        ts = original_events[:, 0]
        t_min, t_max = ts.min(), ts.max()
        duration = t_max - t_min
        dt = duration / num_bins
        bin_indices = np.clip(((ts - t_min) / dt).astype(int), 0, num_bins - 1)
        
        events_per_bin = []
        cancellation_ratios = []
        
        for bin_idx in range(num_bins):
            bin_mask = bin_indices == bin_idx
            bin_events = original_events[bin_mask]
            
            if len(bin_events) > 0:
                pos_events = np.sum(bin_events[:, 3] > 0)
                neg_events = np.sum(bin_events[:, 3] < 0)
                total_events = len(bin_events)
                net_polarity = pos_events - neg_events
                cancellation = 1 - (abs(net_polarity) / total_events)
            else:
                total_events = 0
                cancellation = 0
                
            events_per_bin.append(total_events)
            cancellation_ratios.append(cancellation)
        
        # 绘制事件数量
        bars = ax.bar(range(num_bins), events_per_bin, alpha=0.7, color=color)
        ax.set_xlabel(f'Bin Index ({num_bins} bins)')
        ax.set_ylabel('事件数量')
        ax.set_title(f'{num_bins} bins (每bin {dt/1000:.2f}ms)\n平均抵消率: {np.mean(cancellation_ratios):.1%}')
        ax.grid(True, alpha=0.3)
        
        # 添加抵消率信息
        ax2 = ax.twinx()
        line = ax2.plot(range(num_bins), cancellation_ratios, 'r-o', alpha=0.8, markersize=3)
        ax2.set_ylabel('抵消率', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 1)
        
        # 添加平均值线
        avg_events = np.mean(events_per_bin)
        ax.axhline(y=avg_events, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    output_path = 'debug_output/temporal_resolution_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n时间分辨率对比图已保存: {output_path}")

if __name__ == "__main__":
    analyze_temporal_resolution()