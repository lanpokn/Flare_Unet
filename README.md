# Voxel编解码工具链

基于Events与Voxel表示法的模块化数据处理工具，用于事件相机数据的编解码转换。

## 项目特色

- **模块化设计**: encode.py和decode.py完全独立，遵循单一职责原则
- **专业可视化**: 10+张详细分析图，支持全方位debug
- **端到端验证**: Events → Voxel → Events 完美重构
- **3DUnet兼容**: 支持后续深度学习训练管道

## 环境安装

### 推荐方法: 纯conda安装 (已验证)

```bash
# 1. 删除旧环境 (如果存在)
conda deactivate && conda env remove -n Umain2

# 2. 一次性创建环境并安装所有包
conda create -n Umain2 python=3.9 pytorch torchvision pytorch-cuda=12.1 pytorch-3dunet=1.9.1 numpy scipy matplotlib pandas h5py opencv scikit-image pyyaml tqdm tensorboard -c pytorch -c nvidia -c conda-forge

# 3. 激活并验证
conda activate Umain2 && python -c "import torch; from pytorch3dunet.unet3d.model import ResidualUNet3D; print('✅ All OK')"
```

### 备用方法: 环境文件克隆

```bash
# 如果有environment.yml文件
conda env create -f environment.yml -n Umain2
conda activate Umain2

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 已验证环境规格

- **Python**: 3.9
- **PyTorch**: 2.4.0 + CUDA支持 ✅  
- **pytorch-3dunet**: 1.9.1 ✅
- **安装方式**: 纯conda，避免版本冲突
- **状态**: GPU推理和权重加载已验证 (2025-09-19)

## 快速开始

### 激活环境 (重要!)

**每次使用前都需要激活虚拟环境:**

```bash
# 进入项目目录
cd /path/to/Unet_main

# 激活虚拟环境
eval "$(conda shell.bash hook)" && conda activate Umain
```

### 基本使用

**1. Events → Voxel 编码**

```bash
python src/data_processing/encode.py \
  --input_file testdata/light_source_sequence_00018.h5 \
  --output_voxel_file output.pt \
  --debug
```

**2. Voxel → Events 解码**

```bash
python src/data_processing/decode.py \
  --input_voxel_file output.pt \
  --output_file decoded_events.h5 \
  --debug
```

**3. 端到端验证**

```bash
# 原始数据编码
python src/data_processing/encode.py \
  --input_file testdata/light_source_sequence_00018.h5 \
  --output_voxel_file step1.pt

# 解码重构
python src/data_processing/decode.py \
  --input_voxel_file step1.pt \
  --output_file step2.h5

# 重新编码验证
python src/data_processing/encode.py \
  --input_file step2.h5 \
  --output_voxel_file step3.pt

# 比较结果
python -c "
import torch
v1 = torch.load('step1.pt')
v3 = torch.load('step3.pt')
print(f'L1误差: {torch.abs(v1-v3).mean():.6f}')
print('验证通过!' if torch.abs(v1-v3).mean() < 1e-6 else '验证失败!')
"
```

## 配置说明

### 默认配置 (config/voxel_config.yaml)

```yaml
# 时间参数
total_duration_ms: 100    # 固定100ms输入
num_bins: 16             # 16个时间bin (6.25ms/bin)

# 传感器参数
sensor_size:
  width: 640             # DSEC标准分辨率
  height: 480

# 算法配置
encoding:
  temporal_bilinear: false  # 简单累积算法
  
decoding:
  algorithm: "uniform_random"  # 均匀随机分布

# Debug设置  
debug:
  save_visualizations: true
  output_dir: "debug_output"
  dpi: 150
```

### 命令行参数覆盖

```bash
# 自定义参数
python src/data_processing/encode.py \
  --input_file input.h5 \
  --output_voxel_file output.pt \
  --num_bins 32 \
  --sensor_size 240 180 \
  --debug
```

## Debug可视化

### 启用Debug模式

添加 `--debug` 参数即可生成详细的可视化分析:

```bash
python src/data_processing/encode.py --input_file input.h5 --output_voxel_file output.pt --debug
```

### 生成的分析图

- **1_original_events_analysis.png**: 原始events详细分析
- **2_voxel_temporal_bins.png**: 16个时间bin可视化  
- **3_voxel_detailed_analysis.png**: Voxel统计分析
- **4_input_voxel_bins.png**: 解码输入voxel分析
- **5_decoded_events_analysis.png**: 解码events综合分析
- **6_comparison_analysis.png**: 原始vs重构对比

### 可视化内容

**原始Events分析**:
- 空间分布散点图
- 时间分布直方图
- 极性分析 (正/负事件)
- 事件率随时间变化
- 空间覆盖范围

**Voxel网格分析**:
- 16个时间bin的图像
- 时间轮廓曲线
- 数值分布直方图
- 稀疏性统计

**解码质量验证**:
- 时间戳随机性检验
- Bin分配准确性
- 重构误差分析
- L1/L2误差统计

## 项目结构

```
Unet_main/
├── README.md                    # 本文档
├── config/
│   └── voxel_config.yaml       # 配置文件
├── src/
│   └── data_processing/
│       ├── encode.py           # Events → Voxel 编码器
│       ├── decode.py           # Voxel → Events 解码器
│       └── debug_visualizer.py # 专业可视化模块
├── debug_output/               # Debug输出目录
├── testdata/
│   └── light_source_sequence_00018.h5
├── event_utils-master/         # 第三方可视化库
├── format.txt                  # H5格式说明
└── CLAUDE.md                   # 项目记录
```

## Python API使用

### 编程接口

```python
# 导入模块
from src.data_processing.encode import events_to_voxel, load_h5_events
from src.data_processing.decode import voxel_to_events, save_h5_events
from src.data_processing.debug_visualizer import EventsVoxelVisualizer

# 加载事件数据
events = load_h5_events("input.h5")
print(f"Loaded {len(events)} events")

# Events → Voxel
voxel = events_to_voxel(events, num_bins=16, sensor_size=(480, 640))
print(f"Voxel shape: {voxel.shape}")

# Voxel → Events  
reconstructed_events = voxel_to_events(voxel, total_duration=100000, sensor_size=(480, 640))
print(f"Reconstructed {len(reconstructed_events)} events")

# 保存结果
save_h5_events(reconstructed_events, "output.h5")

# 专业可视化
visualizer = EventsVoxelVisualizer("debug_output")
visualizer.visualize_original_events(events, (480, 640))
visualizer.visualize_voxel_grid(voxel)
```

### 训练管道集成

```python
# 训练时使用
import torch
from src.data_processing.encode import events_to_voxel

class EventDataset(torch.utils.data.Dataset):
    def __init__(self, event_files):
        self.files = event_files
    
    def __getitem__(self, idx):
        events = load_h5_events(self.files[idx])
        voxel = events_to_voxel(events, num_bins=16, sensor_size=(480, 640))
        return voxel
    
    def __len__(self):
        return len(self.files)

# 创建数据加载器
dataset = EventDataset(event_files)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

# 用于3DUnet训练
for batch_voxels in dataloader:
    # batch_voxels: (B, 16, 480, 640)
    # 可直接输入3DUnet模型
    pass
```

## 故障排除

### 常见问题

**1. 找不到conda命令**
```bash
# 初始化conda
conda init bash
source ~/.bashrc
```

**2. CUDA不可用**
```bash
# 检查CUDA版本
nvidia-smi

# 重新安装正确版本的PyTorch
conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.1
```

**3. 模块导入错误**
```bash
# 确保在项目根目录
cd /path/to/Unet_main

# 使用完整路径运行
python src/data_processing/encode.py --help
```

**4. 内存不足**
```bash
# 减少事件数量进行测试
head -n 100000 input.h5 > small_input.h5
```

### 性能优化

**GPU加速** (如果可用):
```python
# 将voxel转移到GPU
voxel = voxel.cuda()

# GPU上的运算
result = model(voxel.cuda())
```

**批量处理**:
```bash
# 处理多个文件
for file in testdata/*.h5; do
    python src/data_processing/encode.py --input_file "$file" --output_voxel_file "${file%.h5}.pt"
done
```

## 技术规格

- **输入格式**: HDF5 (DSEC标准)
- **输出格式**: PyTorch张量 (.pt) 或 NumPy (.npy)
- **时间精度**: 微秒级 (μs)
- **空间分辨率**: 640×480 (可配置)
- **时间分辨率**: 16 bins @ 6.25ms/bin (可配置)
- **数据类型**: float32 (voxel), int64 (时间戳)

## 许可与引用

该项目基于事件相机数据处理的最佳实践开发。如果在研究中使用，请适当引用。

---

**快速启动命令 (收藏使用)**:
```bash
cd /mnt/e/2025/event_flick_flare/Unet_main && eval "$(conda shell.bash hook)" && conda activate Umain
```