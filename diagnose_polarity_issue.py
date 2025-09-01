#!/usr/bin/env python3
"""
诊断解码数据单色问题的专门脚本
检查是编码器/解码器问题还是可视化问题
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.encode import events_to_voxel, load_h5_events
from data_processing.decode import voxel_to_events

def diagnose_polarity_issue():
    """诊断极性问题"""
    
    testdata_path = "testdata/light_source_sequence_00018.h5"
    sensor_size = (480, 640)
    
    print("=== 诊断极性问题 ===")
    
    # 1. 检查原始数据
    print("\n1. 原始数据分析:")
    original_events = load_h5_events(testdata_path)
    print(f"原始事件数量: {len(original_events):,}")
    
    original_polarities = original_events[:, 3]  # p column
    pos_orig = np.sum(original_polarities > 0)
    neg_orig = np.sum(original_polarities < 0)
    zero_orig = np.sum(original_polarities == 0)
    
    print(f"原始极性分布:")
    print(f"  正极性 (+1): {pos_orig:,} ({pos_orig/len(original_events)*100:.1f}%)")
    print(f"  负极性 (-1): {neg_orig:,} ({neg_orig/len(original_events)*100:.1f}%)")
    print(f"  零极性 (0):  {zero_orig:,} ({zero_orig/len(original_events)*100:.1f}%)")
    print(f"  极性值范围: [{original_polarities.min()}, {original_polarities.max()}]")
    print(f"  极性唯一值: {np.unique(original_polarities)}")
    
    # 2. 编码到voxel
    print("\n2. Voxel编码分析:")
    voxel = events_to_voxel(original_events, num_bins=16, sensor_size=sensor_size)
    print(f"Voxel形状: {voxel.shape}")
    print(f"Voxel总和: {voxel.sum():.0f}")
    print(f"Voxel值范围: [{voxel.min():.3f}, {voxel.max():.3f}]")
    
    # 检查正负voxel值
    pos_voxels = voxel[voxel > 0]
    neg_voxels = voxel[voxel < 0] 
    zero_voxels = voxel[voxel == 0]
    
    print(f"Voxel极性分布:")
    print(f"  正值voxels: {len(pos_voxels):,} (总和: {pos_voxels.sum():.0f})")
    print(f"  负值voxels: {len(neg_voxels):,} (总和: {neg_voxels.sum():.0f})")
    print(f"  零值voxels: {len(zero_voxels):,}")
    
    # 3. 解码分析
    print("\n3. 解码数据分析:")
    total_duration = int(original_events[:, 0].max() - original_events[:, 0].min())
    decoded_events = voxel_to_events(voxel, total_duration, sensor_size)
    print(f"解码事件数量: {len(decoded_events):,}")
    
    if len(decoded_events) > 0:
        decoded_polarities = decoded_events[:, 3]  # p column
        pos_dec = np.sum(decoded_polarities > 0)
        neg_dec = np.sum(decoded_polarities < 0)
        zero_dec = np.sum(decoded_polarities == 0)
        
        print(f"解码极性分布:")
        print(f"  正极性 (+1): {pos_dec:,} ({pos_dec/len(decoded_events)*100:.1f}%)")
        print(f"  负极性 (-1): {neg_dec:,} ({neg_dec/len(decoded_events)*100:.1f}%)")
        print(f"  零极性 (0):  {zero_dec:,} ({zero_dec/len(decoded_events)*100:.1f}%)")
        print(f"  极性值范围: [{decoded_polarities.min()}, {decoded_polarities.max()}]")
        print(f"  极性唯一值: {np.unique(decoded_polarities)}")
        
        # 4. 问题诊断
        print("\n4. 问题诊断:")
        
        # 检查voxel编码是否正确保留了极性信息
        if len(neg_voxels) == 0:
            print("❌ 问题发现: Voxel编码时丢失了负极性信息!")
            print("   所有负极性事件在编码时被转换为正值或丢失")
        elif neg_dec == 0:
            print("❌ 问题发现: 解码器没有正确处理负极性voxel!")
            print(f"   Voxel中有 {len(neg_voxels):,} 个负值，但解码后全部变成正极性")
        else:
            print("✅ 极性在编解码过程中正确保留")
            
        # 检查数量一致性
        voxel_pos_sum = pos_voxels.sum()
        voxel_neg_sum = abs(neg_voxels.sum())
        print(f"\n数量一致性检查:")
        print(f"  原始: +{pos_orig}, -{neg_orig}")
        print(f"  Voxel: +{voxel_pos_sum:.0f}, -{voxel_neg_sum:.0f}")
        print(f"  解码: +{pos_dec}, -{neg_dec}")
        
        # 5. 详细检查解码器逻辑
        print("\n5. 解码器逻辑检查:")
        print("检查前几个非零voxel值...")
        
        nonzero_voxel_values = voxel[voxel != 0].flatten()[:20]  # 前20个非零值
        print(f"前20个非零voxel值: {nonzero_voxel_values.numpy()}")
        
        # 手动模拟解码过程
        print("\n手动模拟解码前几个voxel...")
        count = 0
        for bin_idx in range(min(3, voxel.shape[0])):  # 检查前3个bin
            bin_data = voxel[bin_idx].numpy()
            nonzero_indices = np.nonzero(bin_data)
            if len(nonzero_indices[0]) > 0:
                y_coords, x_coords = nonzero_indices
                values = bin_data[nonzero_indices]
                
                for i, (y, x, val) in enumerate(zip(y_coords[:5], x_coords[:5], values[:5])):
                    if val != 0:
                        polarity = 1 if val > 0 else -1
                        print(f"  Bin {bin_idx}, Pixel ({x},{y}): voxel_val={val:.3f} -> polarity={polarity}")
                        count += 1
                        if count >= 10:
                            break
                if count >= 10:
                    break
    else:
        print("❌ 严重错误: 解码没有生成任何事件!")

if __name__ == "__main__":
    diagnose_polarity_issue()