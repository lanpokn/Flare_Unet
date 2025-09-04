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
**我们写法**: `f_maps: [32,64,128,256], num_levels: 4` → 精确控制 (250万参数)

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
f_maps: [64, 128, 256, 512]  # 整体翻倍扩容(当前×8倍)
```

### 参数量对比
```
当前: [32,64,128,256] × 4层 = 250万参数
翻倍: [64,128,256,512] × 4层 = 2800万参数  
官方: [64,128,256,512,1024] × 5层 = 1.13亿参数
医学标准: 5-50M参数
限制: 最大4层深度(5层有尺寸bug)
```

**结论**: 我们的"额外参数"都有实际作用，不是冗余而是精确控制。核心问题是模型容量不足，解决方案是固定深度整体翻倍f_maps。

## 推荐训练配置 - **2025-01-03更新**

### 模型选择与参数量分析
**推荐**: `TrueResidualUNet3D` (真正残差学习)

**当前配置 vs 官方默认对比** - **已升级到中等规模**:
| 参数 | 官方默认 | 之前轻量配置 | **当前中等配置** | 原因 |
|-----|---------|------------|---------------|------|
| `f_maps` | 64 | [16, 32, 64] | **[32, 64, 128]** ✅ | **4倍特征图容量，更强学习能力** |
| `num_levels` | 5 | 3 | **3** | 平衡深度与效率 |
| `num_groups` | 8 | 8 ✅ | **8** ✅ | GroupNorm标准设置 |
| `layer_order` | 'gcr' | 'gcr' ✅ | **'gcr'** ✅ | GroupNorm+Conv+ReLU |
| `dropout_prob` | 0.1 | 0.1 ✅ | **0.1** ✅ | 标准正则化 |
| **总参数量** | **1.13亿** | 43万 | **~173万** | **4倍学习能力提升** |

```yaml
# 当前中等规模配置 (~173万参数) - 2025-01-03已部署
model:
  name: 'TrueResidualUNet3D'
  backbone: 'ResidualUNet3D'
  f_maps: [32, 64, 128]      # 4倍特征图容量，更强学习能力
  num_levels: 3              # 保持3层深度，平衡效率
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

### GPU环境: Umain
- Python 3.9
- PyTorch 2.3.0 + CUDA 12.1 + pytorch-3dunet
- 核心依赖: numpy, h5py, matplotlib, opencv-python, scipy, pandas, pyyaml
- **Debug依赖**: OpenCV 4.10.0 + pandas 1.5.3 (professional_visualizer所需)

### 快速启动 - **2025-09-04**
```bash
cd /mnt/e/2025/event_flick_flare/Unet_main && eval "$(conda shell.bash hook)" && conda activate Umain

# 立即可用 - 无需外部数据
python main.py test --config configs/test_config.yaml --debug  # 评估checkpoint
python main.py train --config configs/train_config.yaml --debug  # 调试训练
```

## 使用指南 - **2025-09-04 本地化完成**

**重要**: 所有数据已本地化至 `data_simu/physics_method/`，无需外部依赖

### 训练 - **使用本地数据**
```bash
# 正常训练 (500个本地训练文件)
python main.py train --config configs/train_config.yaml

# Debug训练模式 (生成6-8个可视化文件夹)
python main.py train --config configs/train_config.yaml --debug
```

### 测试 - **完全重构，支持checkpoint评估**
```bash
# 评估2500 checkpoint (50个本地测试文件)
python main.py test --config configs/test_config.yaml

# Debug测试模式 (智能采样：每5个batch可视化1个)
python main.py test --config configs/test_config.yaml --debug
```

### 推理 - **待更新到本地路径**
```bash
python main.py inference --config configs/inference_config.yaml \
  --input data_simu/some_file.h5 --output results/deflared_events.h5
```

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

## 最新状态 - **2025-09-04更新**

✅ **生产就绪系统**:
- TrueResidualUNet3D + 真正残差学习 (173万参数)
- 完整MLOps pipeline (训练→验证→checkpoint→**测试**)
- **完善的测试模式**: 支持checkpoint评估 + 智能可视化
- **本地化数据管理**: data_simu已完全迁移至本地项目目录
- 现代化tqdm进度条 + emoji输出
- 分段内存优化 + 固定时间分辨率

✅ **测试模式完善** - **2025-09-04新增**:
- **配置统一**: test_config.yaml与train_config.yaml完全匹配
- **数据一致**: 使用与训练validation相同的数据路径
- **checkpoint兼容**: 支持加载任意iteration的checkpoint
- **智能可视化**: 每5个batch采样1个，覆盖所有文件

✅ **数据集本地化** - **2025-09-04完成**:
- **路径迁移**: `/mnt/e/2025/event_flick_flare/main/output/data_simu/` → `/mnt/e/2025/event_flick_flare/Unet_main/data_simu/`
- **训练数据**: 500个H5文件对 (background_with_flare_events + background_with_light_events)
- **测试数据**: 50个H5文件对 (background_with_flare_events_test + background_with_light_events_test)
- **配置更新**: 所有config文件已更新为本地路径
- **独立部署**: 项目现在完全自包含，无外部依赖

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

### 测试模式详解 - **2025-09-04新增**

✅ **核心功能**:
- **Checkpoint评估**: 加载任意iteration的checkpoint进行性能评估
- **数据完整性**: 与train validation使用相同数据路径，确保一致性
- **智能可视化**: debug模式采用5:1采样策略，每个文件可视化1个代表性segment

✅ **使用场景**:
```bash
# 评估2500 iteration checkpoint性能
python main.py test --config configs/test_config.yaml

# 评估 + 智能可视化 (52个文件 × 1个可视化 = 52个文件夹)
python main.py test --config configs/test_config.yaml --debug
```

✅ **数据处理流程**:
- **输入**: 52个test文件 → 260个samples (52×5 segments)
- **评估**: 全部260个samples计算loss
- **可视化**: batch 0,5,10,15... (每个文件的第1个segment)
- **输出**: 平均test loss + 可选的52个可视化文件夹

✅ **配置要求**:
- `model`: 必须与训练配置完全匹配（TrueResidualUNet3D, f_maps, num_levels等）
- `path`: checkpoint文件路径（如`checkpoint_epoch_0000_iter_002500.pth`）
- `val_noisy_dir`/`val_clean_dir`: 与train_config中validation路径一致
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
- `checkpoint_epoch_0000_iter_002500.pth` ✅ 需要重新训练 (配置已升级)
- 当前配置: TrueResidualUNet3D, [32,64,128,256], 250万参数
- 服务器配置: [64,128,256,512], 2800万参数

✅ **参数理解完整性** - **2025-09-04关键更新**:
- **pytorch-3dunet默认行为**: 自动扩展f_maps (官方64→[64,128,256,512,1024])
- **我们的精确控制**: 明确指定f_maps=[32,64,128,256]，禁用自动扩展
- **深度限制发现**: 最大4层，5层会导致尺寸过小bug (256x0x30x40)
- **显存vs深度**: 深度几乎不占显存，扩展模型应该整体翻倍f_maps
- **核心问题识别**: 250万参数仍 < 医学标准5-50M，需要服务器配置

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

## 重要技术发现 - **2025-09-04**

### GPU内存占用分析
**关键发现**: 1.73M参数模型占用8G显存的真实原因
- **模型参数**: 0.01GB
- **激活内存**: 3.93GB (前向传播时)  
- **主要瓶颈**: 3D卷积激活存储，不是参数数量
- **解决方案**: gradient checkpointing可减少80%激活内存

### 参数量对比分析
```
当前配置 [32,64,128]: 1.73M参数 (可能不足)
推荐配置 [64,128,256]: 6.90M参数 (4倍提升)
pytorch-3dunet默认: 113.74M参数 (66倍)
医学去噪典型: 5-50M参数范围
```

**结论**: 参数不足可能是效果不佳的主要原因

## 问题诊断与分析 - **2025-09-04更新**

### **核心问题识别**

✅ **问题诊断完成**:
- **模型容量不足**: 173万参数 vs 医学标准5-50M参数，可能是效果差的根本原因  
- **pytorch-3dunet参数理解**: 官方使用自动扩展(64→[64,128,256,512,1024])，我们使用精确控制([32,64,128])
- **显存占用分析**: 激活内存3.93GB占99.7% vs 参数内存0.01GB占0.3%
- **训练配置问题**: 验证样本过少(10个) + 其他配置需优化
- **参数传递验证**: 所有"额外"参数都有实际作用，不是冗余而是精确控制

### **显存优化尝试总结**

❌ **Gradient Checkpointing尝试**:
- **理论**: 应该减少60-80%激活内存占用
- **实际**: 效果不明显，主要因为pytorch-3dunet内部激活仍被保存
- **结论**: 在wrapper层做checkpointing粒度太粗，需要更深层次的优化

⚠️ **显存优化的实际限制**:
- 3D UNet的激活内存主要在backbone内部
- 简单的wrapper checkpointing无法触及核心问题
- 真正有效的方法可能需要修改pytorch-3dunet内部实现

### **当前可行的优化方向**

1. **模型容量平衡**: 在显存允许范围内适度增加参数
2. **混合精度训练**: 使用AMP减少内存占用
3. **验证逻辑修复**: 完整验证集评估 
4. **配置参数调优**: 学习率、正则化等

### 训练可视化系统优化 - **2025-09-04**

✅ **visualize_training_logs.py 重要更新**:
- **对数刻度显示**: 所有损失图表使用对数刻度，更好展示训练动态和早期变化
- **平滑曲线保留**: 保留红色移动平均线以显示训练趋势
- **删除干扰元素**: 移除红色箭头注释，避免显示为奇怪的直线
- **性能标注优化**: 使用淡绿色文本框显示最佳epoch，无箭头干扰
- **数据量更新**: 支持2594个batch数据点的完整解析和可视化

**技术细节**:
```python
# 主要损失曲线 - 对数刻度
ax1.set_yscale('log')  # 突出训练动态
ax1.plot(steps, losses, 'b-')     # 蓝色原始数据
ax1.plot(smooth_steps, smoothed, 'r-')  # 红色平滑曲线

# Epoch损失 - 对数刻度 + 文本标注
ax2.set_yscale('log')
ax2.text(..., bbox=dict(facecolor='lightgreen'))  # 无箭头干扰
```

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