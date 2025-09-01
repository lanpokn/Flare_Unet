#!/usr/bin/env python3
"""
检查H5文件原始数据格式
"""

import h5py
import numpy as np

def check_h5_format():
    """检查H5文件的确切格式"""
    
    h5_path = "testdata/light_source_sequence_00018.h5"
    
    print("=== H5文件格式检查 ===")
    
    with h5py.File(h5_path, 'r') as f:
        print(f"\nH5文件结构:")
        def print_structure(name, obj):
            print(f"  {name}: {type(obj).__name__}")
            if hasattr(obj, 'shape'):
                print(f"    shape: {obj.shape}, dtype: {obj.dtype}")
                if obj.size < 50:  # 小数组显示所有值
                    print(f"    values: {obj[:]}")
                else:  # 大数组显示前后几个值
                    print(f"    first 10: {obj[:10]}")
                    print(f"    last 10: {obj[-10:]}")
                    print(f"    min: {obj[:].min()}, max: {obj[:].max()}")
                    print(f"    unique values (first 20): {np.unique(obj[:])[:20]}")
        
        f.visititems(print_structure)
        
        # 专门检查极性数据
        if 'events' in f and 'p' in f['events']:
            p_data = f['events']['p'][:]
            print(f"\n=== 极性数据详细分析 ===")
            print(f"极性数组形状: {p_data.shape}")
            print(f"极性数据类型: {p_data.dtype}")
            print(f"极性值范围: [{p_data.min()}, {p_data.max()}]")
            print(f"极性唯一值: {np.unique(p_data)}")
            
            # 统计分布
            unique_vals, counts = np.unique(p_data, return_counts=True)
            for val, count in zip(unique_vals, counts):
                print(f"  值 {val}: {count:,} 次 ({count/len(p_data)*100:.1f}%)")
                
            # 检查前100个值
            print(f"前100个极性值: {p_data[:100]}")

if __name__ == "__main__":
    check_h5_format()