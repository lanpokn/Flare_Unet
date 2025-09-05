#!/usr/bin/env python3
"""
测试智能缓存Dataset的内存行为
验证修复是否解决5x重复读取问题
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.datasets.event_voxel_dataset import EventVoxelDataset
import torch
from torch.utils.data import DataLoader
import gc
import resource

def get_memory_usage():
    """获取当前内存使用量（MB）- 简化版"""
    # 使用resource模块获取内存使用
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: KB to MB
    except:
        return 0  # 如果获取失败，返回0

def test_smart_cache_behavior():
    """测试智能缓存行为"""
    
    print("=== 智能缓存Dataset测试 ===")
    
    # 创建Dataset
    dataset = EventVoxelDataset(
        noisy_events_dir="data_simu/physics_method/background_with_flare_events",
        clean_events_dir="data_simu/physics_method/background_with_light_events"
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Files: {len(dataset.file_pairs)} files × 5 segments")
    
    # 初始内存
    initial_memory = get_memory_usage()
    print(f"\n初始内存: {initial_memory:.1f}MB")
    
    print("\n=== 测试同一文件多个segments（应该复用缓存）===")
    
    # 同一文件的5个segments（file_idx=0）
    for i in range(5):
        sample = dataset[i]  # file 0, segments 0-4
        current_memory = get_memory_usage()
        print(f"Sample {i} (file 0, seg {i}): shape={sample['raw'].shape}, memory={current_memory:.1f}MB (+{current_memory-initial_memory:.1f}MB)")
    
    memory_after_file0 = get_memory_usage()
    print(f"完成file 0的所有segments: {memory_after_file0:.1f}MB (+{memory_after_file0-initial_memory:.1f}MB)")
    
    print("\n=== 测试切换文件（应该清理缓存）===")
    
    # 切换到file 1
    sample = dataset[5]  # file 1, segment 0
    memory_after_switch = get_memory_usage()
    print(f"切换到file 1: {memory_after_switch:.1f}MB (+{memory_after_switch-initial_memory:.1f}MB)")
    
    # 继续file 1的其他segments
    for i in range(6, 10):
        sample = dataset[i]  # file 1, segments 1-4
        current_memory = get_memory_usage()
        print(f"Sample {i} (file 1, seg {i%5}): memory={current_memory:.1f}MB")
    
    memory_after_file1 = get_memory_usage()
    print(f"完成file 1: {memory_after_file1:.1f}MB")
    
    print("\n=== 测试多文件切换（内存应该稳定）===")
    
    # 测试多个文件切换
    test_indices = [10, 15, 20, 25, 30]  # file 2, 3, 4, 5, 6的第一个segment
    for idx in test_indices:
        sample = dataset[idx]
        file_idx = idx // 5
        current_memory = get_memory_usage()
        print(f"Sample {idx} (file {file_idx}): memory={current_memory:.1f}MB")
    
    final_memory = get_memory_usage()
    print(f"\n最终内存: {final_memory:.1f}MB")
    print(f"总内存增长: {final_memory - initial_memory:.1f}MB")
    
    # 手动清理测试
    dataset.clear_cache()
    gc.collect()
    after_clear = get_memory_usage()
    print(f"清理后内存: {after_clear:.1f}MB (-{final_memory - after_clear:.1f}MB)")
    
    print("\n=== 结果分析 ===")
    if final_memory - initial_memory < 50:  # 小于50MB增长
        print("✅ 内存增长合理，智能缓存工作正常")
        return True
    else:
        print("❌ 内存增长过多，可能仍有问题")
        return False

def test_dataloader_behavior():
    """测试DataLoader中的内存行为"""
    
    print("\n=== DataLoader内存测试 ===")
    
    dataset = EventVoxelDataset(
        noisy_events_dir="data_simu/physics_method/background_with_flare_events",
        clean_events_dir="data_simu/physics_method/background_with_light_events"
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False,  # 不shuffle，测试顺序访问
        num_workers=0
    )
    
    initial_memory = get_memory_usage()
    print(f"DataLoader初始内存: {initial_memory:.1f}MB")
    
    # 测试前几个batches
    for i, batch in enumerate(dataloader):
        current_memory = get_memory_usage()
        print(f"Batch {i}: memory={current_memory:.1f}MB (+{current_memory-initial_memory:.1f}MB)")
        
        if i >= 10:  # 只测试前10个batch
            break
    
    final_memory = get_memory_usage()
    print(f"DataLoader内存增长: {final_memory - initial_memory:.1f}MB")

if __name__ == "__main__":
    print("测试智能缓存Dataset是否解决内存问题...")
    
    # 基础功能测试
    success = test_smart_cache_behavior()
    
    # DataLoader测试
    test_dataloader_behavior()
    
    if success:
        print("\n✅ 智能缓存修复成功！")
        print("现在可以安全地进行训练，内存不会无限增长")
    else:
        print("\n❌ 仍需进一步优化")