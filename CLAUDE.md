# Event-Voxel 炫光去除系统 - 项目记录

## 项目概述

基于ResidualUNet3D的**事件炫光去除(Event Deflare)**训练与推理系统，实现从含炫光事件中去除炫光，保留背景和光源事件。

**数据集路径 - 2025-09-04更新**: data_simu已迁移至本地项目目录 `/mnt/e/2025/event_flick_flare/Unet_main/data_simu/`

## 核心设计哲学

基于Linus Torvalds的"好品味"原则：
- **数据结构正确**: Events (N,4) → Voxel (8,H,W) → ResidualUNet3D → 去炫光Voxel
- **消除特殊情况**: 统一20ms/8bins时间分辨率，无多尺度复杂性
- **实用主义**: 解决真实炫光去除问题，不过度设计

## 当前架构 - **2025-01-03最新版本 + 真正残差学习**

### 任务定义
- **输入**: `data_simu/physics_method/background_with_flare_events/` (含炫光的背景事件H5文件)  
- **输出**: `data_simu/physics_method/background_with_light_events/` (干净的背景+光源事件H5文件)
- **学习目标**: 真正残差学习 `output_voxel = input_voxel + residual_learned`，其中`residual ≈ -flare_voxel`
- **数据规模**: 训练集500个文件，测试集50个文件

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
│   ├── tools/                      # 可视化工具
│   │   └── event_video_generator.py # H5事件数据视频生成器
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

## pytorch-3dunet参数理解 - **2025-09-04**

### 核心差异: 自动扩展 vs 精确控制

**官方写法**: `f_maps: 64, num_levels: 5` → 自动扩展为 `[64,128,256,512,1024]` (113M参数)
**我们写法**: `f_maps: [32,64,128,256], num_levels: 4` → 精确控制 (707万参数)

### 关键参数作用

| 参数 | 作用 | 必要性 |
|-----|------|--------|
| `conv_padding: 1` | 保持输出尺寸(8,480,640)不变 | **关键** |
| `dropout_prob: 0.1` | 正则化，防过拟合(小数据集必需) | **重要** |
| `num_groups: 8` | GroupNorm分组，训练稳定性 | **重要** |
| `backbone: ResidualUNet3D` | 指定底层网络类型 | **关键** |
| `num_levels` | 网络深度，影响感受野 | **关键** |
| `f_maps` | 每层特征数，决定参数量 | **关键** |

### 显存vs参数量的重要发现 - **2025-09-04**

**UNet显存分布**:
- **Level 0** (浅层): 75%显存占用 (大分辨率×少通道)
- **Level 3+** (深层): 1-2%显存占用 (小分辨率×多通道)

**关键洞察**: 
- 增加深度(num_levels) **几乎不占显存**，但大幅提升参数量和学习能力
- **深度限制**: 最大4层，5层会导致尺寸过小bug (256x0x30x40)
- 增加f_maps会显著增加显存，特别是浅层

**模型扩展策略**:
```yaml
# 固定深度4层，整体翻倍f_maps来扩大模型
num_levels: 4              # 固定4层深度(5层有bug)
f_maps: [64, 128, 256, 512]  # 整体翻倍扩容(707万→2827万参数)
```

### 参数量对比
```
轻量: [32,64,128] × 3层 = 173万参数
当前: [32,64,128,256] × 4层 = 707万参数 ✅
翻倍: [64,128,256,512] × 4层 = 2827万参数  
官方: [64,128,256,512,1024] × 5层 = 1.13亿参数
医学标准: 5-50M参数 (当前已达标)
限制: 最大4层深度(5层有尺寸bug)
```

**结论**: 我们的"额外参数"都有实际作用，不是冗余而是精确控制。707万参数已达到医学分割标准，模型容量合理。

## 推荐训练配置 - **2025-01-03更新**

### 模型选择与参数量分析
**推荐**: `TrueResidualUNet3D` (真正残差学习)

**当前配置 vs 官方默认对比** - **已升级到医学标准**:
| 参数 | 官方默认 | 之前轻量配置 | **当前标准配置** | 原因 |
|-----|---------|------------|---------------|------|
| `f_maps` | [64,128,256,512,1024] | [32, 64, 128] | **[32, 64, 128, 256]** ✅ | **4层深度，707万参数** |
| `num_levels` | 5 | 3 | **4** | 最大可用深度(5层有bug) |
| `num_groups` | 8 | 8 ✅ | **8** ✅ | GroupNorm标准设置 |
| `layer_order` | 'gcr' | 'gcr' ✅ | **'gcr'** ✅ | GroupNorm+Conv+ReLU |
| `dropout_prob` | 0.1 | 0.1 ✅ | **0.1** ✅ | 标准正则化 |
| **总参数量** | **1.13亿** | 173万 | **707万** ✅ | **达到医学分割标准** |

```yaml
# 当前医学标准配置 (707万参数) - 2025-09-04已部署
model:
  name: 'TrueResidualUNet3D'
  backbone: 'ResidualUNet3D'
  f_maps: [32, 64, 128, 256] # 4层深度，707万参数
  num_levels: 4              # 最大可用深度，平衡性能
  layer_order: 'gcr'         # GroupNorm + Conv + ReLU
  num_groups: 8              # GroupNorm分组
  conv_padding: 1            # 卷积padding
  dropout_prob: 0.1          # Dropout正则化
```

**✅ 升级优势**: 173万参数提供4倍学习能力，适合复杂炫光模式

**进一步扩展建议**:
```yaml
# 大规模配置 (690万参数，复杂炫光场景)
f_maps: [64, 128, 256]      # 16倍特征图容量  
num_levels: 4               # 深度特征学习
```

**参数量对比分析**:
- **之前轻量**: 43万参数 → 可能学习能力不足 ⚠️
- **当前中等**: 173万参数 (4倍) → **平衡性能与效率** ✅ **已部署**
- **大规模**: 690万参数 (16倍) → 强学习能力，需更多数据 

**GPU内存全部友好** (RTX 4060): 所有配置都 <2GB 模型权重

### 训练参数
- **数据**: 500个H5文件对 → 2500个训练样本  
- **训练**: batch_size=1, 50 epochs, MSELoss, Adam优化器
- **设备**: RTX 4060 Laptop GPU, CUDA 12.1
- **验证**: 每2个iteration验证一次，快速反馈

## 环境配置

### 验证工作环境配置
**已验证配置** (2025-09-19):
- **Python**: 3.9  
- **PyTorch**: 2.4.0 + CUDA 支持 ✅
- **pytorch-3dunet**: 1.9.1 ✅ (核心3D UNet模型)
- **安装方式**: 纯conda安装，避免包冲突
- **GPU支持**: CUDA正常工作，模型成功加载和推理

**快速安装** (推荐，纯conda方案):
```bash
# 1. 删除旧环境 (如果存在)
conda deactivate && conda env remove -n Umain2

# 2. 一次性创建环境并安装所有包
conda create -n Umain2 python=3.9 pytorch torchvision pytorch-cuda=12.1 pytorch-3dunet=1.9.1 numpy scipy matplotlib pandas h5py opencv scikit-image pyyaml tqdm tensorboard -c pytorch -c nvidia -c conda-forge

# 3. 激活并验证
conda activate Umain2 && python -c "import torch; from pytorch3dunet.unet3d.model import ResidualUNet3D; print('✅ All OK')"
```

**验证命令**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from pytorch3dunet.unet3d.model import ResidualUNet3D; print('pytorch-3dunet: OK')"
```

### 快速启动 - **2025-12-24更新**
```bash
cd /mnt/e/2025/event_flick_flare/Unet_main && eval "$(conda shell.bash hook)" && conda activate Umain2

# 立即可用 - 无需外部数据
python main.py test --config configs/test_config.yaml --debug     # 评估checkpoint
python main.py test --config configs/test_config.yaml --baseline  # 编解码基线测试
python main.py train --config configs/train_config.yaml --debug   # 调试训练

# 新增：单文件推理 - 2025-12-24
python inference_single.py --input "E:\path\to\file.h5" --mode normal --debug  # 单文件处理
python inference_single.py --input "E:\path\to\file.h5" --mode simple          # Simple权重模式
```

## 使用指南 - **2025-12-24更新**

**重要**: 所有数据已本地化至 `data_simu/physics_method/`，无需外部依赖

### 内存安全单文件推理 - **2025-12-24重大升级** ⭐**关键修复**

**✅ inference_single.py - 内存安全任意长度处理**:
```bash
# Normal权重模式 - 内存安全
python inference_single.py --input "E:\path\to\file.h5" --mode normal

# Simple权重模式 - 内存安全  
python inference_single.py --input "E:\path\to\file.h5" --mode simple

# 带debug可视化
python inference_single.py --input "E:\path\to\file.h5" --mode normal --debug

# 自定义输出路径
python inference_single.py --input "input.h5" --output "custom_output.h5" --mode simple
```

**🛡️ 内存安全核心特性** - **2025-12-24重大修复**:
- ✅ **流式磁盘处理**: 超过50段自动启用磁盘流式处理，避免内存爆炸
- ✅ **智能内存检测**: 自动估算内存需求，预警大文件风险  
- ✅ **分批合并**: 20段为一批进行磁盘合并，避免段数累积
- ✅ **临时文件管理**: 自动创建/清理临时目录，进程终止也会清理
- ✅ **内存监控**: 每50段报告GPU内存使用状况
- ✅ **双权重模式**: normal/simple使用不同checkpoint文件
- ✅ **Windows路径支持**: 自动转换Windows→WSL路径格式  
- ✅ **时空一致性**: 输入输出时间和空间范围完全保持一致
- ✅ **标准格式输出**: 与项目H5格式完全兼容

**🚨 修复的关键问题**:
```python
# ❌ 原始问题: 长文件段数爆炸导致内存耗尽
duration_us = 10,000,000  # 10秒文件  
num_segments = 500        # 500个段
all_processed_events = []  # 累积500个numpy数组 → 内存爆炸 → 终端消失

# ✅ 内存安全解决方案: 流式磁盘处理
max_memory_segments = 50   # 内存段数限制
temp_dir = tempfile.mkdtemp()  # 临时磁盘存储
process_segment_to_disk()  # 逐段处理并保存到磁盘
merge_from_disk()         # 分批从磁盘合并
```

**实测验证结果** - **基于lego2_sequence_new.h5 (1.74M事件, 198.7ms)**:
| 模式 | 输入事件 | 输出事件 | 压缩率 | 处理时间 | 内存安全 | 输出文件 |
|------|----------|----------|--------|----------|----------|----------|  
| Normal | 1,740,864 | 1,270,844 | 73.00% | ~15秒 | ✅安全 | `*_Unet.h5` |
| Simple | 1,740,864 | 1,369,102 | 78.64% | ~15秒 | ✅安全 | `*_Unetsimple.h5` |

**🎯 支持文件规模**:
- **短文件** (<1秒): 直接内存处理
- **中等文件** (1-10秒): 内存处理，监控状态  
- **长文件** (10秒-数分钟): 自动启用流式磁盘处理
- **极长文件** (数分钟-小时): 分批磁盘处理，无内存限制

**批量推理输出**: test模式会创建`输入目录名+output`并行目录，输出去炫光处理后的H5文件

### 训练 - **使用本地数据**
```bash
# 正常训练 (500个本地训练文件)
python main.py train --config configs/train_config.yaml

# Debug训练模式 (生成6-8个可视化文件夹)
python main.py train --config configs/train_config.yaml --debug
```

### 测试 - **批量推理模式** - **2025-09-06更新**
```bash
# 批量推理 (处理所有测试文件，保存去炫光结果)
python main.py test --config configs/test_config.yaml

# 批量推理 + 可视化debug模式 (可选)
python main.py test --config configs/test_config.yaml --debug

# Baseline模式 (仅编解码测试，不经过UNet网络) - **新增**
python main.py test --config configs/test_config.yaml --baseline

# Baseline + debug模式
python main.py test --config configs/test_config.yaml --baseline --debug
```

**核心功能** - **2025-09-06更新**:
- ✅ **批量文件处理**: 自动处理test目录中所有H5文件
- ✅ **输出目录管理**: 
  - 正常模式: 创建`输入目录名+output`并行目录结构
  - Baseline模式: 创建`输入目录名+baseline`并行目录结构
- ✅ **文件名保持**: 输出文件名与输入文件名完全一致
- ✅ **H5格式一致**: 输出H5文件保持与输入相同的events/t,x,y,p结构
- ✅ **时间戳保持**: 保持原始时间范围，确保时序正确性

**Baseline模式特性** - **2025-09-06新增**:
- ✅ **跳过UNet处理**: 不加载模型，不进行网络推理
- ✅ **纯编解码测试**: Events → Voxel → Events，测试编解码保真度
- ✅ **性能基准**: 提供编解码baseline，用于对比UNet改进效果
- ✅ **独立输出**: 保存到`*baseline`目录，与正常模式区分
- ✅ **效率优化**: 跳过模型加载和GPU运算，处理速度更快
- ✅ **格式兼容**: 使用gzip压缩和正确数据类型，与原始文件格式一致

**已知技术债务** - **2025-09-06**:
- **代码重复**: main.py和decode.py中存在两套H5保存函数实现
- **当前状态**: 功能正确，文件大小正常，但代码不够优雅
- **优先级**: 低（纯代码美学问题，可在未来重构中解决）
- **Linus观点**: "代码能用就行，先解决功能问题，再解决优雅问题"

### 推理 - **待测试验证**
```bash
python main.py inference --config configs/inference_config.yaml \
  --input data_simu/some_file.h5 --output results/deflared_events.h5
```

### H5事件视频生成工具集 - **2025-09-09新增**
```bash
# 单文件视频生成（白背景+红蓝映射，2.5ms/帧）
python src/tools/event_video_generator.py --input "path/to/events.h5"

# 快速视频生成 - Lego序列（199帧，15fps，1ms/帧）
python src/tools/event_video_generator.py --input "/mnt/e/2025/event_flick_flare/experiments/3D_reconstruction/datasets/lego/events_h5/lego_sequence.h5" --frame_duration_ms 1.0 --fps 15 --sensor_size 480 640

# Data Simu批量视频生成（physics_method测试集）
python src/tools/datasimu_video_generation.py

# DSEC批量视频生成（DSEC_data真实数据）
python src/tools/dsec_video_generation.py

# EVK4完整可视化生成 - **2025-09-10新增**
python src/tools/evk4_complete_visualization.py  # 自动生成input/target/baseline/inputpfds/unet3d/unet3d_full全部结果对比视频
```

**Data Simu工具特性**:
- ✅ **智能目录识别**: 只处理包含'test'的目录，避免500个文件的训练集
- ✅ **前5文件选择**: 每个test目录自动选择前5个H5文件
- ✅ **分级输出**: debug_output下创建与原目录名一致的子文件夹
- ✅ **完整覆盖**: 5个test目录共25个视频(100%成功率)

**DSEC工具特性**:
- ✅ **真实数据处理**: 处理DSEC_data下4个目录的真实事件数据
- ✅ **全目录覆盖**: 每个目录前5个H5文件，统一视频格式输出

**EVK4 Full权重特性** - **2025-10-01新增**:
- ✅ **unet3d_full目录**: 使用`checkpoint_epoch_0032_iter_076250.pth`full权重处理的EVK4结果
- ✅ **完整对比可视化**: EVK4现支持7种方法对比(input/target/baseline/inputpfds/unet3d/unet3d_simple/unet3d_full)

### 训练日志可视化 - **2025-01-03新增**
```bash
# 基础使用 (推荐) - 自动生成loss曲线
python visualize_training_logs.py

# 自定义日志路径和输出目录
python visualize_training_logs.py --log_dir logs/event_voxel_denoising --output_dir debug_output

# 详细分析模式 (生成额外的详细分析图表)
python visualize_training_logs.py --detailed
```


**功能特性** - **2025-09-04更新**:
- ✅ **自动TensorBoard日志解析**: 支持2594个batch loss + epoch loss数据点
- ✅ **多指标监控**: 训练loss + 学习率曲线 + epoch统计
- ✅ **对数刻度可视化**: 所有loss图表使用对数刻度，更好展示训练动态
- ✅ **平滑曲线分析**: 红色移动平均线突出训练趋势，无多余直线干扰
- ✅ **最佳性能显示**: 淡绿色文本框标注最佳epoch性能
- ✅ **统一输出**: 所有图表保存到`debug_output/training_loss_curves.png`
- ✅ **英文界面**: 避免字体乱码问题，专业展示
- ✅ **无弹窗模式**: 直接保存，不弹出显示窗口

## 最新状态 - **2025-12-24 内存安全单文件推理系统完成** ⭐**重大更新**

✅ **生产就绪系统**:
- **TrueResidualUNet3D**: 真正残差学习 (707万参数医学标准)
- **完整Pipeline**: 训练→验证→批量推理→**内存安全单文件推理** ✅**升级**
- **任意长度处理**: **inference_single.py** 内存安全版本，支持极长文件 ✅**重大修复**
- **流式磁盘处理**: 自动检测内存需求，超过50段启用流式处理 ✅**新技术**
- **双权重模式**: normal/simple模式支持不同checkpoint ✅
- **性能优化**: **4.2倍端到端提升** + 内存安全保障
- **内存泄漏根治**: **段数爆炸问题彻底解决** ⭐**关键修复**
- **本地化部署**: 完全自包含，无外部依赖
- **专业可视化**: 8个文件夹综合debug系统

✅ **核心技术突破**:
- **真正残差学习**: 初始完美恒等映射，专注炫光去除
- **三重I/O优化**: Dataset缓存 + 输出层优化 + 向量化编解码
- **安全向量化**: **PyTorch纯操作**，避免numpy/torch内存交互问题 ⭐**最新修复**
- **内存直接合并**: 消除临时文件，基于100ms固定输入简化

### Debug模式 - **2025-01-03最新实现 + 真正残差学习支持**
✅ **增强可视化debug系统**:
- **低耦合设计**: 在数据产生的地方触发可视化钩子，不修改Dataset返回值
- **智能检测**: 自动识别TrueResidualUNet3D，提供残差分析
- **6-8个可视化文件夹**: 标准模型6个，真正残差学习8个
- **快速验证**: 只运行1-2个iteration，快速检查模型和数据流
- **专业可视化**: 复用已有的professional_visualizer模块，每个events文件夹包含3D+2D+temporal全套可视化
- **统一输出**: **所有debug信息都输出到`debug_output`目录**

**Train/Test Debug可视化文件夹**:
```
# Train模式 debug
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

# Test模式 debug (优化版本标识)
debug_output/test_optimized_iter_000/  # 便于区分优化版本
├── 相同的8个可视化文件夹结构
└── debug_summary.txt
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

**配置文件建议**:

**当前部署配置** (configs/train_config.yaml - **已升级**):
```yaml
model:
  name: 'TrueResidualUNet3D'
  backbone: 'ResidualUNet3D'  
  f_maps: [32, 64, 128]       # 173万参数，4倍学习能力 ✅已部署
  num_levels: 3               # 保持高效深度
  # ... 其他参数
```

**进一步扩展选项** (如需更强学习能力):
```yaml  
model:
  name: 'TrueResidualUNet3D'
  backbone: 'ResidualUNet3D'
  f_maps: [32, 64, 128]       # 当前配置 (173万参数)
  num_levels: 4               # 可选择更深 → ~250万参数
  # ... 其他参数保持不变
```

**大规模配置** (复杂炫光场景):
```yaml
model:
  name: 'TrueResidualUNet3D' 
  backbone: 'ResidualUNet3D'
  f_maps: [64, 128, 256]      # 690万参数，强学习能力
  num_levels: 4
  # ... 其他参数保持不变
```

**重要约定**: **所有debug相关的可视化输出都统一保存到`debug_output`目录**，包括：
- **训练debug**: 6-8个综合可视化文件夹（TrueResidualUNet3D多2个残差文件夹）
- **测试debug**: 智能采样可视化（每5个batch采样1个，覆盖所有文件）
- 智能残差分析和恒等映射验证  
- debug_summary.txt调试总结文件
- **训练loss可视化图表** (`training_loss_curves.png`) - **2025-01-03新增**
- 任何其他debug相关的临时文件和日志

**实现位置**:
- `true_residual_wrapper.py`: TrueResidualUNet3D实现+零初始化
- `src/training/training_factory.py`: 模型创建支持两种架构
- `src/training/custom_trainer.py`: 智能debug可视化钩子（训练模式）
- `main.py`: **测试模式debug可视化** - **2025-09-04新增**
- `visualize_training_logs.py`: **训练loss可视化工具** - **2025-01-03新增**
- **configs/***: **所有配置已本地化** - **2025-09-04更新**

### 测试模式详解 - **2025-09-05批量推理更新**

✅ **批量推理工作流程**:
- **输入目录**: `background_with_flare_events_test/` (50个含炫光的H5文件)
- **输出目录**: `background_with_flare_events_testoutput/` (自动创建)
- **处理流程**: 每个H5文件 → 5个20ms segments → 3D UNet推理 → 合并输出 → 保存H5
- **文件对应**: 输入输出文件名完全一致，便于后续分析对比

✅ **目录结构**:
```
data_simu/physics_method/
├── background_with_flare_events_test/     # 输入: 50个测试文件
└── background_with_flare_events_testoutput/ # 输出: 50个处理结果
    ├── composed_00504_bg_flare.h5           # 与输入同名
    ├── composed_00505_bg_flare.h5           # 去炫光处理后
    └── ...                                  # 保持H5格式一致
```

✅ **技术实现**:
- **分段处理**: 100ms文件分为5×20ms segments，逐段通过UNet处理
- **时间同步**: 输出事件时间戳精确映射回原始时间轴
- **格式保持**: events/t,x,y,p数据结构与输入完全一致
- **内存优化**: 临时segments存储，处理完成后自动清理

✅ **配置要求**:
- `model`: 必须与训练配置完全匹配（TrueResidualUNet3D, f_maps, num_levels等）
- `path`: checkpoint文件路径（如`checkpoint_epoch_0000_iter_002500.pth`）
- `val_noisy_dir`: 测试输入目录路径
- **本地数据路径**: 所有路径已更新为本地`data_simu/physics_method/`结构

### 项目完整性验证 - **2025-09-04新增**

✅ **数据集完整性**:
```
data_simu/physics_method/
├── background_with_flare_events/     # 500个训练文件 (含炫光)
├── background_with_light_events/     # 500个训练文件 (去炫光目标)  
├── background_with_flare_events_test/  # 50个测试文件 (含炫光)
└── background_with_light_events_test/  # 50个测试文件 (去炫光目标)
```

✅ **配置文件完整性**:
- `configs/train_config.yaml` ✅ 已更新本地路径
- `configs/test_config.yaml` ✅ 已更新本地路径  
- `configs/inference_config.yaml` ✅ 已更新本地路径
- 所有配置已验证可正常加载

✅ **Checkpoint可用性**:
- `checkpoint_epoch_0000_iter_002250.pth` ✅ 当前可用 (与优化配置兼容)
- 当前配置: TrueResidualUNet3D, [32,64,128,256], 707万参数 ✅
- 服务器配置: [64,128,256,512], 2827万参数

✅ **参数理解完整性** - **2025-09-04关键更新**:
- **pytorch-3dunet默认行为**: 自动扩展f_maps (官方64→[64,128,256,512,1024])
- **我们的精确控制**: 明确指定f_maps=[32,64,128,256]，禁用自动扩展
- **深度限制发现**: 最大4层，5层会导致尺寸过小bug (256x0x30x40)
- **显存vs深度**: 深度几乎不占显存，扩展模型应该整体翻倍f_maps
- **容量达标**: 707万参数已达到医学标准5-50M范围，模型容量合理

## 核心技术验证

**编解码测试** (956,728事件): 完美一致性 L1/L2/Max = 0.000000 ✅
**内存优化**: 100ms→5×20ms段，显存减少80% ✅
**向量化加速**: Encode 74倍，Decode 1.5倍，端到端7.3倍 ✅

---

## 性能优化总结

**✅ I/O优化** (2025-09-05):
- Dataset智能缓存：5倍I/O减少
- 消除临时文件：内存buffer替代磁盘
- 最终数据流：`H5 → 缓存(1次) → UNet → 内存合并 → H5 (0重复I/O)`

**✅ 向量化加速** (2025-09-06):
- Encode: 15.6s → 0.21s (**74倍提升**)
- Decode: 3.5s → 2.4s (1.5倍)
- 总体: **7.3倍端到端提升**

**✅ 内存安全修复**:
- 纯PyTorch向量化 (避免numpy/torch混合泄漏)
- 智能单文件缓存 (~8MB上限)
- 位置: `encode.py`, `event_voxel_dataset.py`

**GPU内存分析**: 激活内存3.93GB占99.7% vs 参数0.01GB占0.3% (3D卷积瓶颈)

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

### **⚠️ H5文件格式重要约定 - 关键技术要求**

**时间戳数据类型**: **必须使用`np.int64`**，**绝不能用`uint32`或`uint64`**
```python
# ✅ 正确格式 - 项目标准
events_group.create_dataset('t', data=events_np[:, 0].astype(np.int64), compression='gzip', compression_opts=9)

# ❌ 错误格式 - 会导致溢出
events_group.create_dataset('t', data=events_np[:, 0].astype(np.uint32), compression='gzip')
```

**技术原因**:
1. **负时间戳支持**: 事件时间戳可能为负值，具有物理意义(例如相对时间参考点之前发生的事件)
2. **防止溢出**: `uint32`最大值4294967296微秒 ≈ 4295秒，超出后回绕到0导致时间戳错误
3. **完整范围**: `int64`支持±9×10^18微秒范围，足够任何实际应用
4. **项目一致性**: 所有H5文件创建(`main.py`, `decode.py`等)都使用`int64`

**关键Bug案例**: PFDs处理中使用`uint32`导致输出时间戳变为4294967.3ms(接近2^32)而非0-100ms

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

## Inference模式详解 - **2025-09-08全面完成**

### **核心特性**
- ✅ **批量处理**: 自动扫描输入目录中的所有H5文件进行推理
- ✅ **单文件处理**: 支持指定单个H5文件进行推理
- ✅ **无真值标签**: 专为实际应用场景设计，不需要ground truth
- ✅ **灵活配置**: 支持命令行覆盖默认输入输出路径
- ✅ **Debug可视化**: 包含文件名标识的专用debug模式

### **使用方式**

**批量推理** (推荐):
```bash
# 使用配置文件中的默认路径 (DSEC_data/input → DSEC_data/output)
python main.py inference --config configs/inference_config.yaml

# 批量推理 + debug可视化 (包含文件名标识)
python main.py inference --config configs/inference_config.yaml --debug

# 自定义输入输出目录
python main.py inference --config configs/inference_config.yaml \
  --input custom/input_dir --output custom/output_dir --debug
```

**单文件推理**:
```bash
# 单文件处理
python main.py inference --config configs/inference_config.yaml \
  --input DSEC_data/input/real_flare_file.h5 \
  --output DSEC_data/output/deflared_file.h5 --debug
```

### **输出结构**

**文件输出**:
- 输入文件名保持不变
- H5格式与输入完全一致 (events/t,x,y,p)
- 时间戳范围保持一致

**Debug输出** (启用--debug时):
```
debug_output/
├── inference_{filename}_seg_0/     # 第一个时间段(0-20ms)
│   ├── 1_input_events/            # 输入事件可视化
│   ├── 2_output_events/           # 输出事件可视化  
│   └── debug_summary.txt          # 统计信息
├── inference_{filename}_seg_1/     # 第二个时间段(20-40ms)
└── inference_{filename}_seg_2/     # 第三个时间段(40-60ms)
```

**Debug信息内容**:
- 文件名标识: `real_flare_zurich_city_03_a_t1288ms_20250908_173252`
- 事件统计: 输入→输出事件数量和比例
- Voxel统计: 输入输出的均值和标准差
- 完整可视化: 3D时空图、2D时序图、voxel分析图

### **配置文件**

**关键配置** (configs/inference_config.yaml):
```yaml
# 模型配置 (必须与训练配置完全匹配)
model:
  name: TrueResidualUNet3D
  f_maps: [32, 64, 128, 256]  # 707万参数标准配置
  path: checkpoints/event_voxel_deflare/checkpoint_epoch_0027_iter_077500.pth

# 推理配置
inference:
  input_dir: "DSEC_data/input"    # 默认输入目录
  output_dir: "DSEC_data/output"  # 默认输出目录
  sensor_size: [480, 640]
  num_bins: 8
  num_segments: 5
```

### **性能特征**
- **处理速度**: ~7秒/文件 (280万事件)
- **事件压缩率**: ~28% (炫光去除效果)
- **GPU内存**: <2GB (RTX 4060兼容)
- **向量化优化**: 74倍编码加速已应用

### **实际测试结果**
- **输入**: 2,784,964个事件 (100ms数据)
- **输出**: 791,468个事件 (压缩至28%)
- **处理时间**: ~10秒 (包括H5文件读写)
- **Debug可视化**: 3个时间段×8种图表 = 24个可视化文件

---

## PFDs批量处理系统 - **2025-09-09全面完成**

**✅ 完整PFDs Pipeline**:
- **数据流**: H5 → TXT → PFDs_WSL → TXT → H5  
- **批量处理**: 自动扫描目录，处理所有H5文件
- **输出管理**: 输入目录同级创建`输入目录名+pfds`输出目录
- **去噪效果**: 正常压缩率20-30% (1.6M事件 → 400K事件)
- **处理速度**: ~50秒/文件 (含完整可视化)

### **核心特性**
✅ **WSL兼容C++**: 基于PFDs.cpp的WSL版本，逐事件处理  
✅ **Python-C++调度**: 临时文件管理，自动清理  
✅ **inference级可视化**: debug模式生成`pfds_{filename}_seg_0`文件夹  
✅ **路径无关**: 支持任意输入目录，不需要修改代码  

### **PFDs使用指南** - **2025-09-09全面修复完成**

**基础批量处理** (推荐):
```bash
eval "$(conda shell.bash hook)" && conda activate Umain
python3 ext/PFD/batch_pfd_processor.py --input_dir "输入目录路径"
```

**批量处理 + Debug可视化**:
```bash
python3 ext/PFD/batch_pfd_processor.py \
  --input_dir "输入目录路径" --debug --debug_dir debug_output
```

**完整参数版本**:
```bash
python3 ext/PFD/batch_pfd_processor.py \
  --input_dir "data_simu/physics_method/background_with_flare_events_test" \
  --debug --debug_dir debug_output_pfds \
  --delta_t0 20000 --delta_t 20000 --var 1 --neibor 3 --score_select 0
```

**输出结构**:
```
输入目录: background_with_flare_events_test/
输出目录: background_with_flare_events_testpfds/  # 自动创建同级目录+pfds
Debug输出: debug_output_pfds/pfds_{filename}_seg_0/  # 每个文件一个可视化文件夹
```

**PFDs参数说明**:
- `delta_t0`: 第一阶段时间窗口 (默认20000us)
- `delta_t`: 第二阶段时间窗口 (默认20000us)  
- `var`: 第一阶段参数 (默认1)
- `neibor`: 第二阶段邻域参数 (默认3)
- `score_select`: 算法模式 (0=PFDs-B, 1=PFDs-A)

### **技术实现** - **2025-09-09 PFDs核心修复**
- **C++核心**: `ext/PFD/build_wsl/PFDs_WSL` (基于PFDs.cpp逐事件处理版本)
- **编译配置**: WSL环境下自动cmake+make，移除Windows依赖
- **格式转换**: 复用项目现有load_h5_events，兼容当前环境
- **内存管理**: 临时文件自动清理，支持大文件处理
- **可视化集成**: 兼容inference模式的professional_visualizer系统

### **关键技术突破** - **2025-09-09**
✅ **根本问题解决**:
- **WSL版本错误**: PFD_WSL.cpp原创了大量与原版不同的逻辑，导致99.99%压缩率
- **算法差异**: PFD(批处理) vs PFDs(逐事件) - 逐事件处理更适合高密度时间数据
- **数据匹配**: 我们的数据时间密度极高(0.06μs间隔)，PFDs逐事件处理能正常工作

✅ **验证结果**:
- **输入**: 1,689,882 events (100ms)
- **输出**: ~400K events (20-30%压缩率)
- **格式验证**: TXT格式`timestamp x y p`完全符合PFDs_WSL.cpp
- **处理速度**: ~50秒/文件 (大幅改善)

✅ **技术更新**:
- **核心算法**: PFD → PFDs (逐事件处理)
- **输出格式**: `timestamp x y p` (与PFDs.cpp一致)
- **批量处理**: 自动扫描H5文件
- **可视化系统**: 完整debug支持
- **WSL兼容**: 自动编译PFDs_WSL

### **关键Bug修复完成** - **2025-10-11最新**

✅ **时间戳调整逻辑Bug已修复** - **2025-10-11关键修复**:
- **问题**: `if self.min_t_offset < 0` 永远不触发，时间戳未调整到从0开始
- **现象**: 数据时间戳34.86-34.96秒，但EFR处理0-0.2秒窗口，**所有事件被过滤**
- **修复**: 改为 `if self.min_t_offset != 0`，正确调整任意时间范围的数据
- **验证**: 1,850,936 → 505,515 events (27.3%压缩率)，时间戳正确恢复 ✅
- **实现**: `batch_efr_processor.py:175-179`

✅ **极性格式转换Bug已修复** - **2025-09-09**:
- **问题**: PFDs_WSL.cpp错误假设输入为0/1格式，将所有-1极性转换为1极性
- **现象**: 输出事件只有单一极性，丢失了负极性事件
- **修复**: 直接保持输入的-1/1格式，不进行错误转换
- **验证**: 修复后输出包含正确的双极性分布 (62个-1 + 168个1)
- **算法确认**: 使用PFD-B模式 (`score_select=0`) 进行炫光去除

✅ **时间戳数据类型溢出Bug已修复** - **重大修复**:
- **问题**: `batch_pfd_processor.py`使用`uint32`保存时间戳，导致溢出
- **现象**: 输出时间戳变为4294967.3ms(≈2^32)而非正确的0-100ms范围
- **根本原因**: `uint32`最大值4294967296微秒，超出后回绕导致错误时间戳
- **修复**: 改为`np.int64`，支持负时间戳及完整范围(±9×10^18微秒)
- **技术要求**: **所有H5文件时间戳必须使用`int64`，绝不能用`uint32`**
- **物理意义**: 负时间戳有实际意义(相对参考点之前的事件)，必须支持
- **项目一致性**: 与`main.py`、`decode.py`等标准H5格式完全对齐

---

## EFR线性梳状滤波器系统 - **2025-10-01全面分析** ⭐**新增外部方法**

**✅ 完整EFR (Event Flicker Removal) Pipeline**:
- **数据流**: TXT Events → Linear Comb Filter → Filtered TXT → Sort → ZIP
- **核心算法**: ICRA 2022论文"A Linear Comb Filter for Event Flicker Removal"
- **去除对象**: 专门针对荧光灯/LED等周期性炫光 (频率50Hz基准)
- **性能提升**: 相比原始事件流实现4.6倍信噪比改善

### **核心技术原理**

#### **线性梳状滤波器架构**
```cpp
// 四级队列递归滤波系统
struct cell_comb_filter {
    queue<pair<int, double>> q1, q2, q3, q4;  // 四级延迟队列
    double bias = 0.0;         // 像素偏置 (正负事件平衡)
    double sum_p = 0;          // 累积极性和
    int x = -1, y = -1;        // 像素坐标
};
```

**滤波器设计理念**:
- **d1 = 1/f_base**: 主延迟 (50Hz → 20ms = 20000μs)
- **d2 = d1/10**: 次延迟 (2ms = 2000μs)  
- **rho1**: 主反馈系数 (默认0.6)
- **rho2 = 1-(1-rho1)/10**: 次反馈系数 (自动计算)

#### **数据格式规范** - **⚠️关键技术要求**

**输入格式** (events_raw.txt):
```
640 480                    # 第一行: width height
timestamp x y polarity     # 事件格式: t x y p
0 506 294 1               # 正事件: p=1
0 459 294 0               # 负事件: p=0 ⚠️注意：0不是-1
```

**重要数据约定**:
- **时间戳单位**: 微秒 (μs)
- **极性表示**: **智能兼容设计** ✅
  - `p=1` → 正事件 (所有格式通用)
  - `p!=1` (包括0,-1等) → 负事件 (自动兼容多种格式)
- **坐标顺序**: `timestamp x y polarity` ✅**与项目格式一致**
- **坐标系统**: (0,0)在左上角，x向右，y向下

**输出格式** (events_filter.txt):
```cpp
// 输出时自动转换为标准格式
events_output_txt_ << t << " " << ccf_xy.x << " " << ccf_xy.y << " " << 1 << "\n";    // 正事件
events_output_txt_ << t << " " << ccf_xy.x << " " << ccf_xy.y << " " << -1 << "\n";   // 负事件
```

**✅输出极性**: **1=正事件, -1=负事件** (与项目格式一致)

### **配置参数详解**

**EFR_config.yaml关键参数**:
```yaml
base_frequency: 50           # 炫光基频 (Hz) - 针对50Hz荧光灯
process_ts_start: 0          # 处理开始时间 (秒)
process_ts_end: 2.5          # 处理结束时间 (秒)
rho1: 0.6                    # 主反馈系数 (0-1)
delta_t: 10000               # 事件聚合时间窗口 (μs)
sampler_threshold: 0.7       # 输出阈值
load_or_compute_bias: 1      # 1=加载预计算bias, 0=实时计算
img_height: 480              # 图像高度
img_width: 640               # 图像宽度
time_resolution: 1000000     # 时间分辨率 (μs/s)
input_event: "events_raw.txt"     # 输入文件名
output_event: "events_filter.txt" # 输出文件名
data_id: "02"                # 数据集ID
```

### **核心算法流程**

#### **1. 偏置计算与加载**
```cpp
// 计算像素偏置 (正负事件平衡)
if (p == 1) {                    // 正事件
    ccf_xy.event_integ += 1;
} else {                         // 负事件 (p==0)
    ccf_xy.event_integ -= 1;
}
double bias = event_integ / event_num;  // 偏置 = 总极性和 / 总事件数
```

#### **2. 四级递归滤波处理**
```cpp
// Q1: 主延迟处理
void update_q1(int t, int polarity, cell_comb_filter &ccf_xy) {
    ccf_xy.q1.push({t, polarity - ccf_xy.bias});  // 去偏置
    while (t - ccf_xy.q1.front().first >= d1_) {
        update_q2(ccf_xy.q1.front().first + d1_, -ccf_xy.q1.front().second, ccf_xy);
        ccf_xy.q1.pop();
    }
    update_q2(t, polarity - ccf_xy.bias, ccf_xy);  // 处理当前事件
}

// Q2-Q4: 递归反馈滤波
// Q2: rho1反馈 + 时间聚合
// Q3: d2延迟处理  
// Q4: rho2反馈 + 输出采样
```

#### **3. 输出事件采样**
```cpp
void outputEventSampler(int t, cell_comb_filter &ccf_xy) {
    while ((ccf_xy.sum_p >= sampler_thresh_) || (ccf_xy.sum_p <= -sampler_thresh_)) {
        if (ccf_xy.sum_p >= sampler_thresh_) {
            events_output_txt_ << t << " " << ccf_xy.x << " " << ccf_xy.y << " " << 1 << "\n";
            ccf_xy.sum_p -= sampler_thresh_;
        } else {
            events_output_txt_ << t << " " << ccf_xy.x << " " << ccf_xy.y << " " << -1 << "\n";
            ccf_xy.sum_p += sampler_thresh_;
        }
    }
}
```

### **使用流程** - **2025-10-04已验证** ✅

#### **快速启动** (推荐):
```bash
# 激活环境
eval "$(conda shell.bash hook)" && conda activate Umain2

# 单文件处理 + Debug可视化
/home/lanpoknlanpokn/miniconda3/envs/Umain2/bin/python3 \
  ext/EFR-main/batch_efr_processor.py \
  --input_dir testdata/efr_test_input \
  --debug

# 批量处理 (自动创建输入目录+efr输出目录)
/home/lanpoknlanpokn/miniconda3/envs/Umain2/bin/python3 \
  ext/EFR-main/batch_efr_processor.py \
  --input_dir "path/to/h5_files"

# 自定义参数
/home/lanpoknlanpokn/miniconda3/envs/Umain2/bin/python3 \
  ext/EFR-main/batch_efr_processor.py \
  --input_dir testdata/efr_test_input \
  --base_frequency 60 \
  --rho1 0.7 \
  --debug
```

#### **输出结构**:
```
输入: testdata/efr_test_input/
输出: testdata/efr_test_inputefr/        # H5格式去炫光文件
Debug: debug_output/efr/                  # 统一debug输出目录
       └── efr_{filename}_seg_0/
           ├── input_events_seg0_*.png     # 输入事件可视化
           ├── output_events_seg0_*.png    # 输出事件可视化
           ├── input_voxel_seg0_*.png      # 输入voxel分析
           ├── output_voxel_seg0_*.png     # 输出voxel分析
           └── debug_summary.txt           # 统计信息
```

#### **视频生成** (可选):
```bash
/home/lanpoknlanpokn/miniconda3/envs/Umain2/bin/python3 \
  src/tools/event_video_generator.py \
  --input testdata/efr_test_inputefr/composed_00003_bg_flare.h5 \
  --output debug_output/efr/efr_output_video.mp4
```

#### **处理效果验证** - **实测数据**:
| 文件 | 输入事件 | 输出事件 | 压缩率 | 处理时间 |
|-----|---------|---------|--------|---------|
| composed_00003_bg_flare.h5 | 1,689,882 | 175,598 | **10.4%** | 37.3秒 |

**压缩效果**: **90%炫光去除** ⭐**已验证**

### **技术特性与对比**

#### **EFR vs PFDs对比**:
| 特性 | EFR | PFDs |
|-----|-----|------|
| **算法类型** | 线性梳状滤波器 | 基于极性切换检测 |
| **目标炫光** | 周期性炫光 (50Hz LED/荧光灯) | 通用炫光去除 |
| **处理方式** | 每像素独立四级队列滤波 | 逐事件时空邻域分析 |
| **输入极性** | **智能兼容** (1=正,!=1=负) ✅ | **-1/1格式** ✅ |
| **输出极性** | **-1/1格式** ✅ | **-1/1格式** ✅ |
| **参数调节** | 频率相关 (base_frequency) | 时间窗口相关 (delta_t) |
| **计算复杂度** | O(1) per pixel | O(k) per event |
| **内存需求** | 4个队列 × 每像素 | 临时缓存 |

#### **关键优势**:
- ✅ **专业针对性**: 专门设计用于移除周期性炫光
- ✅ **理论基础**: 基于信号处理的线性梳状滤波器理论
- ✅ **像素独立**: 每个像素独立处理，并行友好
- ✅ **自适应偏置**: 自动计算每像素的正负事件偏置
- ✅ **E2VID兼容**: 输出格式直接支持E2VID重建评估

#### **注意事项**:
- ✅ **极性兼容**: EFR智能兼容-1/1格式，无需转换
- ✅ **坐标顺序**: EFR使用t,x,y,p顺序，与项目格式完全一致
- ⚠️ **频率调节**: base_frequency需要根据具体炫光频率调整
- ⚠️ **WSL兼容**: 需要安装OpenCV和yaml-cpp依赖

### **WSL兼容性实现** - **2025-10-04已解决** ✅

#### **核心技术挑战与解决方案**:

**问题**: EFR硬编码相对路径依赖 `../configs/` 和 `../data/`

**✅ 最终解决方案** (copytree策略):
```python
# ext/EFR-main/batch_efr_processor.py
# 1. 创建临时目录结构
temp_dir / "data" / data_id / "events_raw.txt"
temp_dir / "configs" / "EFR_config.yaml"

# 2. 备份原始目录并复制临时数据
for link_path, source_dir in [(Path("../data"), temp_dir / "data"),
                               (Path("../configs"), temp_dir / "configs")]:
    if link_path.exists():
        link_path.rename(link_path.parent / f"{link_path.name}_backup_original")
    shutil.copytree(source_dir, link_path)  # WSL兼容

# 3. 运行EFR (从build/目录)
os.chdir(efr_build_dir)
subprocess.run([str(self.efr_executable)])

# 4. 复制输出文件回临时目录
efr_output_dir = Path("../data") / data_id
for output_file in efr_output_dir.glob("*"):
    shutil.copy2(output_file, temp_dir / "data" / data_id / output_file.name)

# 5. 清理并恢复原始目录
shutil.rmtree(link_path)
backup_path.rename(link_path)
```

**技术要点**:
- ✅ **copytree替代symlink**: 绕过WSL符号链接权限问题
- ✅ **自动备份恢复**: 保护原始EFR data/configs目录
- ✅ **输出文件同步**: 自动复制EFR输出回临时目录
- ✅ **自动清理**: try/finally确保临时文件和备份正确恢复
- ✅ **动态配置**: Python生成YAML配置，无需修改EFR源码
#### **关键技术修复** - **2025-10-04**:

**1. C++编译错误修复**:
```cpp
// comb_filter.h - 添加缺失头文件
#pragma once
#include <string>
#include <fstream>  // ⭐ 新增：修复std::ofstream编译错误
#include <opencv2/opencv.hpp>
```

**2. 自动bias计算**:
```python
# batch_efr_processor.py - 默认参数设置
'load_or_compute_bias': 0,  # 0=自动计算bias (无需预计算文件)
```

**3. 统一debug输出**:
```python
# 默认debug目录: debug_output/efr/ (与项目统一)
def __init__(self, debug: bool = False, debug_dir: str = 'debug_output/efr'):
```

---

这个系统现在实现了**真正残差学习**、**背景信息保护**、**PFDs传统去噪**、**EFR线性滤波**和**工程简洁性**的统一，基于Linus"好品味"原则解决事件炫光去除的实际问题。

**核心突破**:
1. **UNet残差学习**: 让网络从完美恒等映射开始，专注学习需要去除的炫光
2. **PFDs传统去噪**: 基于极性变化的高效逐事件去噪算法，20-30%正常压缩率
3. **EFR线性滤波**: 专业周期性炫光去除，**90%炫光去除率** ✅**已验证** (2025-10-04)
4. **完整Pipeline**: 三种方法全部跑通，支持批量处理+debug可视化+视频输出

---

## 主实验数据集生成工具 - **2025-10-22最新版** ⭐**论文核心工具**

### **核心功能**

**位置**: `src/tools/generate_main_dataset.py`

**✅ 统一处理仿真和真实数据集 + 真实数据专用模式**:
- **数据源**: 固定100ms H5文件（仿真或真实数据）
- **仿真模式**: 4个标准UNet权重 + PFD-A + PFD-B + EFR + Baseline (共8种方法)
- **真实模式** (`--real`): **自动扫描所有40000权重** + 固定old权重 + PFD/EFR/Baseline (共**13种方法**)
- **统一可视化**: 所有方法结果自动生成MP4视频
- **断点续存**: 自动跳过已处理文件

### **UNet权重配置** - **2025-10-23重大更新**

**仿真模式（默认）** - **⭐使用physics_noRandom_noTen数据集**:
```python
# 3个checkpoint: physics_noRandom_noTen_method + simple + old权重
unet_checkpoints = {
    'simple': 'checkpoints/event_voxel_deflare_simple/checkpoint_epoch_0031_iter_040000.pth',
    'physics_noRandom_noTen_method': 'checkpoints/event_voxel_deflare_physics_noRandom_noTen_method/checkpoint_epoch_0031_iter_040000.pth',  # ⭐替换full
    'full_old': 'checkpoints_old/event_voxel_deflare_full/checkpoint_epoch_0032_iter_076250.pth',
    'simple_old': 'checkpoints_old/event_voxel_deflare_simple/checkpoint_epoch_0027_iter_076250.pth',
}

# 数据集路径更新
input_dir: datasimu2/physics_noRandom_noTen_method/background_with_flare_events_test
target_dir: datasimu2/physics_noRandom_noTen_method/background_with_light_events_test
```

**真实模式** (`--real`):
```python
# 自动扫描checkpoints/所有40000权重 + 固定old权重
# 实际发现9个变体：
# - full, simple, nolight, physics, physics_noRandom_method,
#   physics_noRandom_noTen_method, simple_timeRandom_method (7个新版40000权重)
# - full_old, simple_old (2个固定旧版权重)
```

### **使用方法** - **2025-10-22更新**

```bash
# 仿真数据集（默认，使用physics_noRandom_noTen_method + simple + old权重）⭐已更新
python3 src/tools/generate_main_dataset.py --test --num_samples 1

# 真实数据集（EVK4，自动扫描所有40000权重，测试模式）⭐推荐
python3 src/tools/generate_main_dataset.py --real \
  --input_dir /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/EVK4/input \
  --target_dir /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/EVK4/target \
  --test --num_samples 1

# 真实数据集（完整运行，处理所有文件）
python3 src/tools/generate_main_dataset.py --real \
  --input_dir /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/EVK4/input \
  --target_dir /mnt/e/BaiduSyncdisk/2025/event_flick_flare/Unet_main/EVK4/target
```

### **输出目录结构** - **2025-10-22更新**

**仿真模式** (`MainSimu_data/`) - **⭐2025-10-23更新**:
```
MainSimu_data/
├── input/                               # 原始含炫光数据（来自datasimu2/physics_noRandom_noTen_method/）
├── target/                              # 目标去炫光数据
├── output_physics_noRandom_noTen_method/  # UNet physics_noRandom_noTen权重⭐替换原full
├── output_simple/                       # UNet simple权重（新版）
├── output_full_old/                     # UNet full权重（旧版）
├── output_simple_old/                   # UNet simple权重（旧版）
├── inputpfda/                           # PFD-A结果
├── inputpfdb/                           # PFD-B结果
├── inputefr/                            # EFR结果
├── outputbaseline/                      # Baseline结果
└── visualize/{filename}/
    ├── input.mp4
    ├── target.mp4
    ├── unet_physics_noRandom_noTen_method.mp4  ⭐替换原unet_full.mp4
    ├── unet_simple.mp4
    ├── unet_full_old.mp4
    ├── unet_simple_old.mp4
    ├── pfda_output.mp4
    ├── pfdb_output.mp4
    ├── efr_output.mp4
    └── baseline_output.mp4
```

**真实模式** (`MainReal_data/`) - **⭐新增**:
```
MainReal_data/
├── input/                           # 原始含炫光数据
├── target/                          # 目标去炫光数据
├── output_full/                     # UNet full权重
├── output_simple/                   # UNet simple权重
├── output_nolight/                  # UNet nolight权重⭐
├── output_physics/                  # UNet physics权重⭐
├── output_physics_noRandom_method/  # UNet physics_noRandom_method权重⭐
├── output_physics_noRandom_noTen_method/ # UNet physics_noRandom_noTen_method权重⭐
├── output_simple_timeRandom_method/ # UNet simple_timeRandom_method权重⭐
├── output_full_old/                 # UNet full权重（旧版）
├── output_simple_old/               # UNet simple权重（旧版）
├── inputpfda/                       # PFD-A结果
├── inputpfdb/                       # PFD-B结果
├── inputefr/                        # EFR结果
├── outputbaseline/                  # Baseline结果
└── visualize/{filename}/
    ├── input.mp4
    ├── target.mp4
    ├── unet_full.mp4
    ├── unet_simple.mp4
    ├── unet_nolight.mp4            ⭐
    ├── unet_physics.mp4            ⭐
    ├── unet_physics_noRandom_method.mp4 ⭐
    ├── unet_physics_noRandom_noTen_method.mp4 ⭐
    ├── unet_simple_timeRandom_method.mp4 ⭐
    ├── unet_full_old.mp4
    ├── unet_simple_old.mp4
    ├── pfda_output.mp4
    ├── pfdb_output.mp4
    ├── efr_output.mp4
    └── baseline_output.mp4
```

**处理方法总结** - **2025-10-23**:
- **仿真模式**: 4个UNet (physics_noRandom_noTen + simple + 2×old) + 4个传统方法 = **8种方法** ⭐数据集已更新
- **真实模式**: 9个UNet + 4个传统方法 = **13种方法**

---

## DSEC数据集生成工具 - **2025-10-22最新更新** ⭐**真实数据处理**

### **核心功能**

**位置**: `src/tools/generate_dsec_dataset.py`

**✅ 从长炫光文件提取100ms段并处理**:
- **内存安全提取**: 从长炫光文件（1-5GB）中提取100ms段，避免内存溢出
- **时间采样**: 每400ms提取100ms (0-100ms, 400-500ms, 800-900ms, ...)
- **10种处理方法**: **6个UNet权重** + PFD-A + PFD-B + EFR + Baseline ⭐**2025-10-22新增2个UNet**
- **智能断点续存**: 检查所有方法输出是否存在，只处理缺失的 ⭐**2025-10-22重大修复**

### **UNet权重配置** - **2025-10-22新增**

```python
self.unet_checkpoints = {
    'simple': 'checkpoints/event_voxel_deflare_simple/checkpoint_epoch_0031_iter_040000.pth',
    'full': 'checkpoints/event_voxel_deflare_full/checkpoint_epoch_0031_iter_040000.pth',
    'physics_noRandom_method': 'checkpoints/physics_noRandom_method/checkpoint_epoch_0031_iter_040000.pth',  # ⭐新增
    'physics_noRandom_noTen_method': 'checkpoints/event_voxel_deflare_physics_noRandom_noTen_method/checkpoint_epoch_0031_iter_040000.pth',  # ⭐新增
    'full_old': 'checkpoints_old/event_voxel_deflare_full/checkpoint_epoch_0032_iter_076250.pth',
    'simple_old': 'checkpoints_old/event_voxel_deflare_simple/checkpoint_epoch_0027_iter_076250.pth',
}
```

### **技术架构**

**数据流**:
```
长炫光文件 → 内存安全提取100ms → DSEC_data/input/ →
    ├── UNet3D (6个权重) → output_full/, output_simple/, output_physics_noRandom_method/,
    │                       output_physics_noRandom_noTen_method/, output_full_old/, output_simple_old/
    ├── PFD-A → DSEC_data/inputpfda/
    ├── PFD-B → DSEC_data/inputpfdb/
    ├── EFR → DSEC_data/inputefr/
    └── Baseline → DSEC_data/outputbaseline/
→ 统一可视化 → DSEC_data/visualize/{filename}/
```

### **核心技术特性**

**1. 内存安全的100ms提取** - **关键创新**:
```python
def extract_100ms_segment_safe(self, file_path: Path, start_time_us: int) -> np.ndarray:
    """内存安全地提取100ms事件段"""
    # Step 1: 只读取时间戳数组来确定索引范围
    t_all = events_group['t'][:]
    mask = (t_all >= start_time_us) & (t_all < start_time_us + 100000)
    indices = np.where(mask)[0]

    # Step 2: 只读取这个范围的数据（避免加载整个5GB文件）
    idx_start, idx_end = indices[0], indices[-1] + 1
    t = events_group['t'][idx_start:idx_end]
    x = events_group['x'][idx_start:idx_end]
    # ...
```

**优势**:
- ✅ 处理任意大小的文件（测试过5GB文件）
- ✅ 只读取需要的100ms范围
- ✅ 随机起始时间确保多样性

**2. 直接调用处理器类** - **避免subprocess开销**:
```python
# 初始化所有处理器 - 2025-10-21更新
self.pfd_processor_a = BatchPFDProcessor(debug=False)
self.pfd_processor_a.pfds_params['score_select'] = 1  # PFD-A

self.pfd_processor_b = BatchPFDProcessor(debug=False)
self.pfd_processor_b.pfds_params['score_select'] = 0  # PFD-B

self.efr_processor = BatchEFRProcessor(debug=False)

# 直接调用
self.pfd_processor_a.process_single_file(input_h5, pfda_h5, file_idx=0)
self.pfd_processor_b.process_single_file(input_h5, pfdb_h5, file_idx=0)
self.efr_processor.process_single_file(input_h5, efr_h5, file_idx=0)
```

**3. Baseline直接实现** - **无需external调用**:
```python
def run_baseline_processing(self, input_h5: Path, output_h5: Path):
    """Baseline: Events → Voxel → Events（测试编解码保真度）"""
    events_np = load_h5_events(str(input_h5))
    voxel = events_to_voxel(events_np, num_bins=8, sensor_size=(480,640), fixed_duration_us=100000)
    output_events = voxel_to_events(voxel, total_duration=100000, sensor_size=(480,640))
    self.save_h5_events(output_events, output_h5)
```

### **使用方法** - **2025-10-21更新**

```bash
# 完整运行（自动处理所有长H5文件，带断点续存）
python3 src/tools/generate_dsec_dataset.py

# 自定义flare源目录
python3 src/tools/generate_dsec_dataset.py \
  --flare_dir "/mnt/e/2025/event_flick_flare/main/data/flare_events"

# Debug模式
python3 src/tools/generate_dsec_dataset.py --debug
```

**关键说明**:
- **无`--num_samples`参数**: 自动处理所有文件，每400ms提取100ms
- **断点续存**: 自动解析已有文件名，跳过已处理段
- **时间采样**: 0-100ms, 400-500ms, 800-900ms, ... (间隔400ms)

### **关键技术修复** - **2025-10-22**

**1. 智能断点续存**:
```python
def _check_all_outputs_exist(self, filename: str) -> bool:
    """检查所有方法的输出是否都存在，只有全部存在才跳过"""
    # 检查6个UNet变体 + 4个传统方法
    # 只有全部10种方法都完成才返回True
```

**2. 损坏数据段自动跳过**:
```python
# H5文件x坐标损坏时自动捕获并跳过
except OSError as e:
    if "B-tree signature" in str(e):
        print("⏭️ Skipping corrupted segment")
        return None
```

**3. 文件名复用机制**:
```python
def find_existing_filename(self, source_file, start_time):
    """查找已存在的文件（忽略datetime），避免重复生成"""
    pattern = f"real_flare_{source}_t{time}ms_*.h5"
    matches = list(self.input_dir.glob(pattern))
    return matches[0].name if matches else generate_new()
```

### **输出目录结构** - **2025-10-22更新**

```
DSEC_data/
├── input/                    # 提取的100ms段
├── output_full/              # UNet full权重（新版）
├── output_simple/            # UNet simple权重（新版）
├── output_physics_noRandom_method/         # ⭐新增
├── output_physics_noRandom_noTen_method/   # ⭐新增
├── output_simple/            # UNet simple权重（新版）
├── output_full_old/          # UNet full权重（旧版）⭐
├── output_simple_old/        # UNet simple权重（旧版）⭐
├── inputpfda/                # PFD-A结果 ⭐
├── inputpfdb/                # PFD-B结果 ⭐
├── inputefr/                 # EFR线性滤波结果
├── outputbaseline/           # Baseline编解码结果
└── visualize/{filename}/
    ├── input.mp4
    ├── unet_full.mp4
    ├── unet_simple.mp4
    ├── unet_full_old.mp4     ⭐
    ├── unet_simple_old.mp4   ⭐
    ├── pfda_output.mp4       ⭐
    ├── pfdb_output.mp4       ⭐
    ├── efr_output.mp4
    └── baseline_output.mp4
```

### **处理方法总结** - **2025-10-21最终版**

**两个生成器都支持8种方法**:
1-2. UNet full/simple（新版: `epoch_0031_iter_040000.pth`）
3-4. UNet full_old/simple_old（旧版: `epoch_0032/0027_iter_076250.pth`）⭐新增
5-6. PFD-A/B（去噪/去频闪, score_select=1/0）
7-8. EFR + Baseline

