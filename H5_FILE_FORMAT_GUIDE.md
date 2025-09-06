# H5文件格式详细指南 - Event Data

## 概述

本项目使用HDF5格式存储事件相机数据。H5文件是一种高效的科学数据存储格式，特别适合存储大规模的结构化数值数据。

## 文件结构

### 基本数据组织
```
file.h5
└── events/
    ├── t (时间戳数组)
    ├── x (X坐标数组)  
    ├── y (Y坐标数组)
    └── p (极性数组)
```

### 数据规格
- **时间范围**: 100ms (100,000微秒)
- **传感器分辨率**: 480×640像素
- **数据类型**: 所有数组均为数值类型
- **事件数量**: 通常50万-100万个事件/文件

## 数据字段详解

### 1. 时间戳 (events/t)
```python
# 数据类型: int64 或 float64
# 单位: 微秒 (microseconds)
# 范围: [0, 100000] 对应100ms时间窗口
# 示例: [1234, 1856, 2341, ...]
```

### 2. X坐标 (events/x)  
```python
# 数据类型: int16 或 int32
# 范围: [0, 639] 对应640像素宽度
# 示例: [320, 145, 521, ...]
```

### 3. Y坐标 (events/y)
```python
# 数据类型: int16 或 int32  
# 范围: [0, 479] 对应480像素高度
# 示例: [240, 67, 398, ...]
```

### 4. 极性 (events/p)
```python
# 数据类型: int8 或 int16
# 值: 1(正事件/亮度增加) 或 0/非1(负事件/亮度减少)
# 本项目处理: 1→+1, 非1→-1
# 示例: [1, 0, 1, 1, 0, ...]
```

## Python读取代码示例

### 基础读取方法
```python
import h5py
import numpy as np

def load_h5_events(file_path):
    """
    读取H5事件文件
    
    Returns:
        numpy.ndarray: Shape (N, 4) [t, x, y, p]
    """
    with h5py.File(file_path, 'r') as f:
        # 读取四个数据数组
        t = np.array(f['events']['t'])  # 时间戳
        x = np.array(f['events']['x'])  # X坐标
        y = np.array(f['events']['y'])  # Y坐标  
        p = np.array(f['events']['p'])  # 极性
        
        # 组合成 (N, 4) 格式
        events = np.column_stack((t, x, y, p))
        
    return events

# 使用示例
events = load_h5_events('input_file.h5')
print(f"事件数量: {len(events)}")
print(f"时间范围: {events[:, 0].min():.0f} - {events[:, 0].max():.0f} μs")
print(f"空间范围: X[{events[:, 1].min():.0f}, {events[:, 1].max():.0f}], Y[{events[:, 2].min():.0f}, {events[:, 2].max():.0f}]")
print(f"极性分布: 正={np.sum(events[:, 3] == 1)}, 负={np.sum(events[:, 3] != 1)}")
```

### 高级分析方法
```python
def analyze_h5_events(file_path):
    """详细分析H5事件文件"""
    events = load_h5_events(file_path)
    
    analysis = {
        'total_events': len(events),
        'time_span_ms': (events[:, 0].max() - events[:, 0].min()) / 1000,
        'spatial_resolution': (int(events[:, 1].max()) + 1, int(events[:, 2].max()) + 1),
        'positive_events': np.sum(events[:, 3] == 1),
        'negative_events': np.sum(events[:, 3] != 1),
        'event_rate_khz': len(events) / ((events[:, 0].max() - events[:, 0].min()) / 1000) / 1000,
        'spatial_density': len(events) / ((events[:, 1].max() + 1) * (events[:, 2].max() + 1))
    }
    
    return analysis

# 使用示例
analysis = analyze_h5_events('test_file.h5')
for key, value in analysis.items():
    print(f"{key}: {value}")
```

## 文件比较方法

### 1. 基础统计比较
```python
def compare_h5_files_basic(file1_path, file2_path):
    """基础统计比较"""
    events1 = load_h5_events(file1_path)
    events2 = load_h5_events(file2_path)
    
    comparison = {
        'event_count_diff': len(events1) - len(events2),
        'time_span_diff': (events1[:, 0].max() - events1[:, 0].min()) - (events2[:, 0].max() - events2[:, 0].min()),
        'positive_ratio_diff': (np.sum(events1[:, 3] == 1) / len(events1)) - (np.sum(events2[:, 3] == 1) / len(events2))
    }
    
    return comparison
```

### 2. 时空分布比较
```python
def compare_spatiotemporal_distribution(file1_path, file2_path, bins=50):
    """时空分布比较"""
    events1 = load_h5_events(file1_path)
    events2 = load_h5_events(file2_path)
    
    # 时间分布比较
    hist1_t, _ = np.histogram(events1[:, 0], bins=bins)
    hist2_t, _ = np.histogram(events2[:, 0], bins=bins)
    temporal_similarity = np.corrcoef(hist1_t, hist2_t)[0, 1]
    
    # 空间分布比较  
    hist1_x, _ = np.histogram(events1[:, 1], bins=bins)
    hist2_x, _ = np.histogram(events2[:, 1], bins=bins)
    spatial_x_similarity = np.corrcoef(hist1_x, hist2_x)[0, 1]
    
    hist1_y, _ = np.histogram(events1[:, 2], bins=bins)
    hist2_y, _ = np.histogram(events2[:, 2], bins=bins)
    spatial_y_similarity = np.corrcoef(hist1_y, hist2_y)[0, 1]
    
    return {
        'temporal_correlation': temporal_similarity,
        'spatial_x_correlation': spatial_x_similarity, 
        'spatial_y_correlation': spatial_y_similarity
    }
```

### 3. 基于Voxel表示的比较
```python
def compare_as_voxel_representation(file1_path, file2_path, num_bins=8):
    """转换为voxel表示后比较（类似本项目处理方式）"""
    from src.data_processing.encode import events_to_voxel
    
    # 读取并转换为voxel
    events1 = load_h5_events(file1_path)
    events2 = load_h5_events(file2_path)
    
    voxel1 = events_to_voxel(events1, num_bins=num_bins, sensor_size=(480, 640), fixed_duration_us=100000)
    voxel2 = events_to_voxel(events2, num_bins=num_bins, sensor_size=(480, 640), fixed_duration_us=100000)
    
    # 计算相似性指标
    mse = np.mean((voxel1 - voxel2) ** 2)
    mae = np.mean(np.abs(voxel1 - voxel2))
    cosine_sim = np.sum(voxel1 * voxel2) / (np.linalg.norm(voxel1) * np.linalg.norm(voxel2))
    
    return {
        'voxel_mse': mse,
        'voxel_mae': mae, 
        'voxel_cosine_similarity': cosine_sim
    }
```

## 保存H5文件

```python
def save_h5_events(events, output_path):
    """
    保存事件数据到H5文件
    
    Args:
        events: numpy array (N, 4) [t, x, y, p]
        output_path: 输出文件路径
    """
    with h5py.File(output_path, 'w') as f:
        events_group = f.create_group('events')
        events_group.create_dataset('t', data=events[:, 0].astype(np.int64))
        events_group.create_dataset('x', data=events[:, 1].astype(np.int16)) 
        events_group.create_dataset('y', data=events[:, 2].astype(np.int16))
        events_group.create_dataset('p', data=events[:, 3].astype(np.int8))
```

## 本项目的特定处理

### 输入文件类型
- **含炫光文件**: `background_with_flare_events/*.h5`
- **去炫光目标**: `background_with_light_events/*.h5`  
- **测试输出**: `background_with_flare_events_testoutput/*.h5`

### 处理流程
1. **H5 → Voxel**: 100ms事件 → 5×20ms段 → 8bins voxel
2. **神经网络**: Voxel → 去炫光Voxel  
3. **Voxel → H5**: 重建事件并保存为H5格式

### 质量评估建议
```python
# 推荐的文件比较流程
def evaluate_deflare_quality(input_file, output_file, target_file=None):
    """评估去炫光效果"""
    
    # 1. 基础统计比较
    basic_comparison = compare_h5_files_basic(input_file, output_file)
    
    # 2. Voxel表示比较
    voxel_comparison = compare_as_voxel_representation(input_file, output_file)
    
    # 3. 如果有目标文件，计算与目标的相似性
    if target_file:
        target_comparison = compare_as_voxel_representation(output_file, target_file)
        return basic_comparison, voxel_comparison, target_comparison
    
    return basic_comparison, voxel_comparison
```

## 依赖包

```bash
pip install h5py numpy
```

## 常见问题

1. **内存不足**: 大文件可分块读取
2. **数据类型**: 确保正确的数据类型转换
3. **坐标系**: 注意Y轴可能需要翻转
4. **极性处理**: 本项目将非1值都视为负极性

---

这个指南应该能帮助你的另一个工程正确读取和处理H5事件文件。如果需要更具体的示例或有特定问题，请告诉我。