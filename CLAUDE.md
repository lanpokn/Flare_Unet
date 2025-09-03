# Event-Voxel 炫光去除系统 - 项目记录

## 项目概述

基于ResidualUNet3D的**事件炫光去除(Event Deflare)**训练与推理系统，实现从含炫光事件中去除炫光，保留背景和光源事件。

## 核心设计哲学

基于Linus Torvalds的"好品味"原则：
- **数据结构正确**: Events (N,4) → Voxel (8,H,W) → ResidualUNet3D → 去炫光Voxel
- **消除特殊情况**: 统一20ms/8bins时间分辨率，无多尺度复杂性
- **实用主义**: 解决真实炫光去除问题，不过度设计

## 当前架构 - **2025-01-03最新版本 + 真正残差学习**

### 任务定义
- **输入**: `background_with_flare_events/` (含炫光的背景事件H5文件)  
- **输出**: `background_with_light_events/` (干净的背景+光源事件H5文件)
- **学习目标**: 真正残差学习 `output_voxel = input_voxel + residual_learned`，其中`residual ≈ -flare_voxel`

### 数据流（全在voxel空间）
```
H5 Events (100ms) → 5×20ms Segments → Voxel (1,8,480,640) → TrueResidualUNet3D → Deflared Voxel → Events
                                                                       ↓
                                               input_voxel + backbone_network(input_voxel)
```

### 核心技术突破 - **真正残差学习**
1. **TrueResidualUNet3D**: 端到端残差学习，初始完美恒等映射
2. **零初始化**: 最后一层零权重，确保初始时`output = input + 0 = input`
3. **背景保护**: 初始状态100%保持背景，网络专注学习炫光去除
4. **梯度流畅**: 残差连接确保梯度直接传递，无消失风险
5. **能量保持**: 初始能量保持率100%，正值比例100%→100%

## 项目结构

```
.
├── src/
│   ├── data_processing/            # Events↔Voxel编解码
│   │   ├── encode.py
│   │   ├── decode.py
│   │   └── professional_visualizer.py
│   ├── datasets/
│   │   └── event_voxel_dataset.py  # EventVoxelDataset
│   ├── training/
│   │   ├── training_factory.py     # ResidualUNet3D模型创建
│   │   └── custom_trainer.py       # 自定义训练循环
│   └── utils/
│       └── config_loader.py
├── configs/                        # 训练配置系统
│   ├── train_config.yaml
│   ├── test_config.yaml
│   └── inference_config.yaml
└── main.py                         # 项目入口 (train/test/inference)
```

## 关键技术解决方案

### 1. pytorch-3dunet ResidualUNet3D问题分析 - **已解决**

**发现的根本问题**:
- pytorch-3dunet的ResidualUNet3D **不是真正的端到端残差学习**
- 初始状态：能量保持率-46%，正值比例100%→13.6%，相对差异148.7%
- 大量背景信息在初始状态就丢失，需要充分训练才能恢复

### 2. 真正残差学习实现 - **全新解决方案**

**TrueResidualUNet3D架构**:
```python
class TrueResidualUNet3D(nn.Module):
    def forward(self, x):
        residual = self.backbone(x)  # 学习残差 ≈ -flare_voxel
        output = x + residual        # 真正残差: output = input + residual
        return output
```

**零初始化机制**:
```python
# 最后一层权重和偏置初始化为0
nn.init.zeros_(last_conv.weight)
nn.init.zeros_(last_conv.bias)
# 确保初始时: residual = 0, output = input (完美恒等映射)
```

**测试验证结果**:
- 初始能量保持率: **100.0%** ✅ (vs -46% ❌)
- 初始正值比例: **100%→100%** ✅ (vs 100%→13.6% ❌)  
- 初始相对差异: **0.000%** ✅ (vs 148.7% ❌)
- 梯度流验证: 最后层梯度范数2.3，训练正常

### 3. pytorch-3dunet激活函数问题的完整解决方案 - **关键技术细节**

**问题根源**:
```python
# pytorch-3dunet的激活函数设计缺陷
final_sigmoid=False → Softmax(dim=1) → 单通道时输出全1 ❌
final_sigmoid=True  → Sigmoid()     → 输出限制在[0,1] ❌
```

**为什么需要实数域(R)输出**:
1. **Voxel语义**: 正负事件累积，值域为整个实数轴 `(-∞, +∞)`
2. **残差学习**: 需要学习 `residual ≈ -flare_noise`，必须支持负值
3. **数值精度**: Sigmoid[0,1]压缩破坏voxel的数值含义

**我们的解决方案**:
```python
# 巧妙绕过pytorch-3dunet的限制:
final_sigmoid=True           # 避免Softmax(dim=1)的bug
model.final_activation = nn.Identity()  # 强制替换为恒等映射
# 结果: 输出域 = R，支持任意实数值
```

**实现位置**: `src/training/training_factory.py:68-75`

## 推荐训练配置 - **2025-01-03更新**

### 模型选择
**推荐**: `TrueResidualUNet3D` (真正残差学习)
```yaml
model:
  name: 'TrueResidualUNet3D'  # 替代 'ResidualUNet3D'
  backbone: 'ResidualUNet3D'  # 底层网络
  f_maps: [16, 32, 64]
  num_levels: 3
  # 自动零初始化，无需额外配置
```

**对比选择**:
- `ResidualUNet3D`: 传统pytorch-3dunet实现，背景信息初始丢失
- `TrueResidualUNet3D`: 真正残差学习，背景信息完全保护

### 训练参数
- **数据**: 500个H5文件对 → 2500个训练样本  
- **训练**: batch_size=1, 50 epochs, MSELoss, Adam优化器
- **设备**: RTX 4060 Laptop GPU, CUDA 12.1
- **验证**: 每2个iteration验证一次，快速反馈

## 环境配置

### GPU环境: Umain
- Python 3.9
- PyTorch 2.3.0 + CUDA 12.1 + pytorch-3dunet
- 核心依赖: numpy, h5py, matplotlib, opencv-python, scipy, pandas, pyyaml
- **Debug依赖**: OpenCV 4.10.0 + pandas 1.5.3 (professional_visualizer所需)

### 快速启动
```bash
cd /mnt/e/2025/event_flick_flare/Unet_main && eval "$(conda shell.bash hook)" && conda activate Umain
```

## 使用指南

### 训练
```bash
# 正常训练
python main.py train --config configs/train_config.yaml

# Debug训练模式 (只运行1-2个iteration，生成9个可视化文件夹)
python main.py train --config configs/train_config.yaml --debug
```

### 测试
```bash
python main.py test --config configs/test_config.yaml
```

### 推理
```bash
python main.py inference --config configs/inference_config.yaml \
  --input noisy_events.h5 --output deflared_events.h5
```

## 最新状态 - **2025-01-03**

✅ **生产就绪系统**:
- ResidualUNet3D + final_sigmoid问题修复
- 完整MLOps pipeline (训练→验证→checkpoint→推理)
- 现代化tqdm进度条 + emoji输出
- 分段内存优化 + 固定时间分辨率

🔧 **待解决问题**:
- 之前的"validation loss始终1.109320"问题应该已通过ResidualUNet3D + final_sigmoid修复解决
- 需要重新训练验证新架构效果

### Debug模式 - **2025-01-03最新实现 + 真正残差学习支持**
✅ **增强可视化debug系统**:
- **低耦合设计**: 在数据产生的地方触发可视化钩子，不修改Dataset返回值
- **智能检测**: 自动识别TrueResidualUNet3D，提供残差分析
- **6-8个可视化文件夹**: 标准模型6个，真正残差学习8个
- **快速验证**: 只运行1-2个iteration，快速检查模型和数据流
- **专业可视化**: 复用已有的professional_visualizer模块，每个events文件夹包含3D+2D+temporal全套可视化
- **统一输出**: **所有debug信息都输出到`debug_output`目录**

**真正残差学习的8个可视化文件夹**:
```
debug_output/epoch_000_iter_000/
├── 1_input_events/          # 输入事件综合可视化 (3D+2D+temporal)
├── 3_input_voxel/           # 输入voxel时间bins可视化
├── 4_target_events/         # 真值事件综合可视化 (3D+2D+temporal)
├── 6_target_voxel/          # 真值voxel时间bins可视化
├── 7_output_events/         # 最终输出事件 (input+residual) 可视化
├── 8_residual_voxel/        # 学习残差voxel可视化 ⭐新增
├── 8_residual_events/       # 学习残差events可视化 (如果非零) ⭐新增  
├── 9_output_voxel/          # 最终输出voxel时间bins可视化
└── debug_summary.txt        # 调试总结信息 (含残差分析)
```

**残差分析日志**:
```
🐛 True residual learning detected:
🐛   Input mean: X.XXXX, std: X.XXXX
🐛   Residual mean: X.XXXX, std: X.XXXX  
🐛   Output mean: X.XXXX, std: X.XXXX
🐛   Identity check: output ≈ input + residual = True
```

**使用方法**:
```bash
# 真正残差学习 + debug模式
python main.py train --config configs/train_config.yaml --debug

# 自定义debug目录
python main.py train --config configs/train_config.yaml --debug --debug-dir my_custom_debug
```

**配置文件更新** (configs/train_config.yaml):
```yaml
model:
  name: 'TrueResidualUNet3D'  # 使用真正残差学习
  backbone: 'ResidualUNet3D'   # 或 'UNet3D' 
  in_channels: 1
  out_channels: 1
  f_maps: [16, 32, 64]
  num_levels: 3
```

**重要约定**: **所有debug相关的可视化输出都统一保存到`debug_output`目录**，包括：
- 6-8个综合可视化文件夹（TrueResidualUNet3D多2个残差文件夹）
- 智能残差分析和恒等映射验证  
- debug_summary.txt调试总结文件
- 任何其他debug相关的临时文件和日志

**实现位置**:
- `true_residual_wrapper.py`: TrueResidualUNet3D实现+零初始化
- `src/training/training_factory.py`: 模型创建支持两种架构
- `src/training/custom_trainer.py`: 智能debug可视化钩子

## 验证结果与技术突破

### 端到端测试验证
**测试数据**: 956,728事件 (100ms) → 编码 → 解码 → 重编码
- **完美一致性**: 两次编码voxel完全相同 (L1=0.000000, L2=0.000000, Max=0.000000)
- **信息保持**: 原始voxel sum = 重编码voxel sum (完全匹配)
- **时间分布**: 解码时间戳正确落在对应时间bin内
- **极性保持**: 正负事件比例在编解码中完全一致

### 简化实现突破
1. **数据结构正确**: Events (N,4) ↔ Voxel (8,H,W)，直接对应无特殊情况
2. **模块独立性**: 编解码模块完全独立，可单独调用测试
3. **配置驱动**: YAML统一管理，命令行可覆盖
4. **专业可视化**: 基于event_utils专业库，100+张分析图

### 内存优化效果
- **分段策略**: 100ms → 5×20ms段，显存占用减少80%
- **数据量对比**: 956K事件 → 200K事件/段 (21%内存占用)
- **训练稳定性**: 大幅减少OOM错误，适合GPU训练

### 时间一致性保证
- **固定分辨率**: 所有样本20ms/8bins = 2.5ms/bin
- **泛化性**: 训练推理使用相同时间语义
- **避免问题**: 消除自适应时间间隔导致的泛化失败

---

## 核心模块详解

### 1. 编码器 (src/data_processing/encode.py)

**数据加载**: `load_h5_events(file_path)`
- **输入**: H5文件路径
- **输出**: NumPy数组 (N, 4) [t, x, y, p]
- **极性准则**: **1→正事件(+1), 非1→负事件(-1)** (通用处理各种格式)

**核心编码**: `events_to_voxel(events_np, num_bins=8, sensor_size, fixed_duration_us=20000)`
- **算法**: 简单累积，正负事件分别处理
- **固定时间**: 确保训练一致性，避免泛化问题
- **分段策略**: 100ms → 5×20ms段，每段8个bins

**独立执行**:
```bash
python src/data_processing/encode.py --input_file test.h5 --output_voxel_file output.pt --debug
```

### 2. 解码器 (src/data_processing/decode.py)

**核心解码**: `voxel_to_events(voxel, total_duration, sensor_size)`
- **算法**: 均匀随机分布解码，基于物理意义
- **流程**: 浮点→整数→生成事件→随机时间戳
- **端到端验证**: 编解码pipeline完全可逆 (L1=L2=0.000000)

**独立执行**:
```bash
python src/data_processing/decode.py --input_voxel_file input.pt --output_file output.h5 --debug
```

### 3. 专业可视化系统 (src/data_processing/professional_visualizer.py)

**设计理念** (遵循Linus**"用已有好工具"**原则):
- 基于event_utils-master专业库，不重复造轮子
- 支持任意阶段的events和voxel数据可视化
- 自动处理兼容性和错误容忍

**完整6可视化架构** (2×2+2):
1. **输入Events 3D时空可视化**: 原生3D散点图 + 时间窗口序列
2. **输入Events 2D红蓝时序**: 红蓝极性显示 (红=正事件，蓝=负事件)
3. **输出Events 3D时空可视化**: 解码后events的3D分析
4. **输出Events 2D红蓝时序**: 与输入对比的2D时序图
5. **输入Voxel可视化**: 时间bin展示 + 统计分析
6. **输出Voxel可视化**: 重编码voxel对比分析

**关键接口**:
```python
# 完整pipeline可视化 (推荐)
from src.data_processing.professional_visualizer import visualize_complete_pipeline
visualize_complete_pipeline(
    input_events=input_events_np,
    input_voxel=input_voxel_tensor,
    output_events=output_events_np,
    output_voxel=output_voxel_tensor,
    sensor_size=(480, 640),
    output_dir="debug_output",
    segment_idx=1  # 可视化段索引: 0-4对应不同20ms段
)

# 单独可视化
from src.data_processing.professional_visualizer import visualize_events_and_voxel
visualize_events_and_voxel(events_np, voxel_tensor, sensor_size, "debug_output", "batch_name")
```

**分段内存优化特性**:
- **100ms → 20ms段**: 数据量减少到21%，避免显存爆炸
- **时间一致性**: 输入输出使用相同时间段进行对比
- **专业输出**: 所有可视化统一保存到debug_output目录
- **segment_idx参数**: 选择不同时间段 (0=0-20ms, 1=10-30ms, 2=20-40ms等)

**输出文件结构示例** (Segment 1: 10-30ms):
```
debug_output/
├── input_events_seg1_native_3d_spatiotemporal.png
├── input_events_seg1_2d_temporal/  # 2D红蓝时序图
├── input_voxel_seg1_temporal_bins.png
├── output_events_seg1_native_3d_spatiotemporal.png
├── output_events_seg1_2d_temporal/ # 对比2D时序图
└── output_voxel_seg1_temporal_bins.png
```

### 4. 数据集系统 (src/datasets/event_voxel_dataset.py)

**EventVoxelDataset核心功能**:
- **配对数据**: 自动扫描匹配noisy/clean事件文件对
- **分段处理**: 每个100ms文件产生5个训练样本 (5×20ms段)
- **格式转换**: Events → Voxel → (1,8,H,W) pytorch格式
- **内存友好**: 20ms段处理，避免OOM问题

**使用示例**:
```python
from src.datasets.event_voxel_dataset import EventVoxelDataset

dataset = EventVoxelDataset(
    noisy_events_dir="/path/to/background_with_flare_events",
    clean_events_dir="/path/to/background_with_light_events",
    sensor_size=(480, 640),
    segment_duration_us=20000,  # 20ms段
    num_bins=8,                 # 8个时间bins
    num_segments=5              # 5个段per文件
)

# 返回格式: {'raw': (1,8,480,640), 'label': (1,8,480,640)}
```

## 核心编解码接口

### Events → Voxel 编码
```python
from src.data_processing.encode import events_to_voxel
voxel = events_to_voxel(events_np, num_bins=8, sensor_size=(480,640), fixed_duration_us=20000)
```

### Voxel → Events 解码  
```python
from src.data_processing.decode import voxel_to_events
events = voxel_to_events(voxel, total_duration=20000, sensor_size=(480,640))
```

### 端到端测试验证
```python
# 验证编解码一致性
python -c "
from src.data_processing import encode, decode
events1 = encode.load_h5_events('test.h5')
voxel = encode.events_to_voxel(events1, num_bins=8, sensor_size=(480,640))
events2 = decode.voxel_to_events(voxel, total_duration=20000, sensor_size=(480,640))
voxel2 = encode.events_to_voxel(events2, num_bins=8, sensor_size=(480,640))
print(f'一致性验证: L1={torch.nn.L1Loss()(voxel, voxel2):.6f}')
"
```

## 真正残差学习vs传统方法对比

### 关键指标对比表

| 指标 | pytorch-3dunet ResidualUNet3D | TrueResidualUNet3D (零初始化) |
|-----|------------------------------|------------------------------|
| **架构设计** | 内部skip connections | 端到端残差学习 `output = input + residual` |
| **初始能量保持** | -46% ❌ | **100.0%** ✅ |
| **初始正值保持** | 100%→13.6% ❌ | **100%→100%** ✅ |
| **初始相对差异** | 148.7% ❌ | **0.000%** ✅ |
| **背景保护** | 严重丢失 ❌ | **完全保护** ✅ |
| **梯度流** | 传统UNet梯度 | **残差直接传递** ✅ |
| **训练稳定性** | 需学会重建背景 | **专注学习去炫光** ✅ |

### 推荐迁移路径

1. **立即切换**: 配置文件`model.name: 'TrueResidualUNet3D'`
2. **保持训练代码**: 无需修改训练循环
3. **观察debug结果**: 对比残差学习效果
4. **预期效果**: 初始状态就有合理输出，训练更稳定

---

这个系统实现了**真正残差学习**、**背景信息保护**和**工程简洁性**的统一，基于Linus"好品味"原则解决事件炫光去除的实际问题。

**核心突破**: 让网络从完美恒等映射开始，专注学习需要去除的炫光，而不是重建整个背景。