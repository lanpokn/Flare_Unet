#!/usr/bin/env python3
"""
诊断时间轴映射问题的专门脚本
检查encode/decode和可视化的时间处理是否一致
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.encode import events_to_voxel, load_h5_events
from data_processing.decode import voxel_to_events

def analyze_time_alignment():
    """详细分析时间轴对齐问题"""
    
    testdata_path = "testdata/light_source_sequence_00018.h5"
    sensor_size = (480, 640)
    
    print("=== 时间轴映射诊断 ===")
    
    # 1. 加载原始数据
    print("\n1. 原始数据时间分析:")
    original_events = load_h5_events(testdata_path)
    original_times = original_events[:, 0]
    
    t_min, t_max = original_times.min(), original_times.max()
    duration = t_max - t_min
    
    print(f"时间范围: {t_min} - {t_max} 微秒")
    print(f"总时长: {duration:,} 微秒 ({duration/1000:.1f} ms)")
    print(f"事件数量: {len(original_events):,}")
    
    # 2. 编码时的时间处理
    print("\n2. 编码过程时间分析:")
    voxel = events_to_voxel(original_events, num_bins=16, sensor_size=sensor_size)
    
    # 手动重现编码逻辑来验证时间分bin
    t_normalized = (original_times - t_min) / duration  # [0, 1]
    bin_indices = np.floor(t_normalized * 16).astype(int)
    bin_indices = np.clip(bin_indices, 0, 15)  # 确保范围
    
    print(f"Voxel形状: {voxel.shape}")
    print(f"时间归一化范围: [{t_normalized.min():.6f}, {t_normalized.max():.6f}]")
    print(f"Bin索引范围: [{bin_indices.min()}, {bin_indices.max()}]")
    
    # 分析每个bin的时间范围和事件数量
    print(f"\n每个Voxel bin的详细分析:")
    for bin_idx in range(16):
        bin_mask = bin_indices == bin_idx
        bin_count = np.sum(bin_mask)
        
        if bin_count > 0:
            bin_times = original_times[bin_mask]
            bin_t_min, bin_t_max = bin_times.min(), bin_times.max()
            theoretical_start = t_min + (duration * bin_idx / 16)
            theoretical_end = t_min + (duration * (bin_idx + 1) / 16)
            
            voxel_sum = torch.sum(torch.abs(voxel[bin_idx])).item()
            
            print(f"  Bin {bin_idx:2d}: {bin_count:6,} events, voxel_sum={voxel_sum:6.0f}")
            print(f"    实际时间: {bin_t_min:8.0f} - {bin_t_max:8.0f} ({(bin_t_max-bin_t_min)/1000:.1f}ms)")  
            print(f"    理论时间: {theoretical_start:8.0f} - {theoretical_end:8.0f} ({(theoretical_end-theoretical_start)/1000:.1f}ms)")
            if bin_count != int(voxel_sum):
                print(f"    ⚠️  事件数量不匹配: 原始={bin_count}, voxel={int(voxel_sum)}")
        else:
            print(f"  Bin {bin_idx:2d}: 0 events")
    
    # 3. 解码时的时间处理
    print(f"\n3. 解码过程时间分析:")
    total_duration_for_decode = int(duration)
    decoded_events = voxel_to_events(voxel, total_duration_for_decode, sensor_size)
    
    if len(decoded_events) > 0:
        decoded_times = decoded_events[:, 0]
        decoded_t_min, decoded_t_max = decoded_times.min(), decoded_times.max()
        
        print(f"解码事件数量: {len(decoded_events):,}")
        print(f"解码时间范围: {decoded_t_min:.0f} - {decoded_t_max:.0f} 微秒")
        print(f"解码总时长: {decoded_t_max - decoded_t_min:.0f} 微秒 ({(decoded_t_max - decoded_t_min)/1000:.1f} ms)")
        
        # 检查解码事件的bin分布
        decoded_normalized = decoded_times / total_duration_for_decode  # [0, 1]
        decoded_bin_indices = np.floor(decoded_normalized * 16).astype(int)
        decoded_bin_indices = np.clip(decoded_bin_indices, 0, 15)
        
        print(f"\n解码事件的bin分布:")
        for bin_idx in range(16):
            original_in_bin = np.sum(bin_indices == bin_idx)
            decoded_in_bin = np.sum(decoded_bin_indices == bin_idx)
            voxel_sum = torch.sum(torch.abs(voxel[bin_idx])).item()
            
            print(f"  Bin {bin_idx:2d}: 原始={original_in_bin:6,}, 解码={decoded_in_bin:6,}, voxel={voxel_sum:6.0f}")
            if decoded_in_bin != int(voxel_sum):
                print(f"    ⚠️  解码数量不匹配!")
    
    # 4. 可视化时间处理分析
    print(f"\n4. 可视化时间切片分析:")
    
    # 模拟可视化的时间切片逻辑（32切片）
    for num_slices in [8, 16, 32]:
        print(f"\n  {num_slices}张时间切片分析:")
        slice_duration = duration / num_slices
        
        for slice_idx in range(min(5, num_slices)):  # 只显示前5个切片
            slice_start = t_min + slice_idx * slice_duration
            slice_end = slice_start + slice_duration
            
            slice_mask = (original_times >= slice_start) & (original_times < slice_end)
            slice_count = np.sum(slice_mask)
            
            # 对应的voxel bin(s)
            if num_slices == 16:
                corresponding_voxel_bin = slice_idx
                voxel_sum = torch.sum(torch.abs(voxel[corresponding_voxel_bin])).item()
                print(f"    切片{slice_idx:2d}: {slice_count:6,} events, 对应voxel bin {corresponding_voxel_bin} (sum={voxel_sum:6.0f})")
                if abs(slice_count - voxel_sum) > slice_count * 0.1:  # 超过10%差异
                    print(f"      ❌ 严重不匹配!")
            else:
                print(f"    切片{slice_idx:2d}: {slice_count:6,} events ({slice_start/1000:.1f}-{slice_end/1000:.1f}ms)")
    
    # 5. 创建对比图
    print(f"\n5. 生成时间对比图...")
    create_time_comparison_plot(original_events, voxel, decoded_events)

def create_time_comparison_plot(original_events, voxel, decoded_events):
    """创建时间对比图"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    fig.suptitle('时间轴映射对比分析', fontsize=16)
    
    # 原始数据时间分布
    ax = axes[0]
    original_times = original_events[:, 0]
    t_min, t_max = original_times.min(), original_times.max()
    duration = t_max - t_min
    
    bins = np.linspace(t_min, t_max, 50)
    ax.hist(original_times, bins=bins, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('时间 (微秒)')
    ax.set_ylabel('事件数量')
    ax.set_title(f'原始事件时间分布 (总计: {len(original_events):,})')
    ax.grid(True, alpha=0.3)
    
    # 添加16个voxel bin的边界线
    for i in range(17):
        bin_time = t_min + (duration * i / 16)
        ax.axvline(bin_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
        if i < 16:
            ax.text(bin_time + duration/32, ax.get_ylim()[1]*0.9, f'B{i}', 
                   color='red', fontsize=8, ha='center')
    
    # Voxel bin分布
    ax = axes[1]
    voxel_sums = [torch.sum(torch.abs(voxel[i])).item() for i in range(16)]
    bin_centers = np.arange(16)
    bars = ax.bar(bin_centers, voxel_sums, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Voxel Bin 索引')
    ax.set_ylabel('事件数量 (绝对值总和)')
    ax.set_title('Voxel Bin 事件分布')
    ax.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 解码数据时间分布  
    ax = axes[2]
    if len(decoded_events) > 0:
        decoded_times = decoded_events[:, 0]
        bins = np.linspace(decoded_times.min(), decoded_times.max(), 50)
        ax.hist(decoded_times, bins=bins, alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('时间 (微秒)')
        ax.set_ylabel('事件数量')
        ax.set_title(f'解码事件时间分布 (总计: {len(decoded_events):,})')
        ax.grid(True, alpha=0.3)
        
        # 添加16个bin的理论边界
        decode_duration = decoded_times.max() - decoded_times.min()
        for i in range(17):
            bin_time = decoded_times.min() + (decode_duration * i / 16)
            ax.axvline(bin_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
    else:
        ax.text(0.5, 0.5, '无解码事件', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('解码事件时间分布 (无数据)')
    
    plt.tight_layout()
    
    output_path = 'debug_output/time_alignment_analysis.png'
    os.makedirs('debug_output', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"时间对比图已保存: {output_path}")

if __name__ == "__main__":
    analyze_time_alignment()