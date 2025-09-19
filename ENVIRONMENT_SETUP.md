# Event-Voxel 炫光去除系统 - 环境安装指南

## 概述

本指南基于已验证的Umain2环境配置，提供完整的环境安装和配置方法。

## 已验证环境配置 ✅

**验证日期**: 2025-09-19  
**验证状态**: GPU推理、权重加载、可视化系统全部正常

### 核心组件版本
```
Python:         3.9.23
PyTorch:        2.5.1 + CUDA 12.1
pytorch-3dunet: 1.9.1 (核心3D UNet模型)
h5py:           3.14.0 (H5文件处理)
numpy:          2.0.2 (数值计算)
opencv-python:  4.12.0.88 (视频生成)
matplotlib:     3.9.4 (可视化)
scipy:          1.13.1 (科学计算)
scikit-image:   0.24.0 (图像处理)
```

## 安装方法

### 方法1: 纯conda安装 (强烈推荐)

**最可靠的方法**，避免pip/conda混用导致的版本冲突：

```bash
# 1. 删除旧环境 (如果存在)
conda deactivate && conda env remove -n Umain2

# 2. 一次性创建环境并安装所有包
conda create -n Umain2 python=3.9 pytorch torchvision pytorch-cuda=12.1 pytorch-3dunet=1.9.1 numpy scipy matplotlib pandas h5py opencv scikit-image pyyaml tqdm tensorboard -c pytorch -c nvidia -c conda-forge

# 3. 激活并验证
conda activate Umain2 && python -c "import torch; from pytorch3dunet.unet3d.model import ResidualUNet3D; print('✅ All OK')"
```

### 方法2: 环境文件克隆 (备用方法)

如果有现成的environment.yml文件：

```bash
# 1. 克隆环境
conda env create -f environment.yml -n Umain2

# 2. 激活环境
conda activate Umain2

# 3. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from pytorch3dunet.unet3d.model import ResidualUNet3D; print('pytorch-3dunet: OK')"
```

### 关键改进点

1. **不限定PyTorch版本**: 让conda自动选择GPU兼容版本，避免意外安装CPU版本
2. **纯conda安装**: 避免pip/conda混用导致的依赖冲突  
3. **一次性安装**: 所有包在同一命令中解决依赖关系
4. **多channel支持**: 使用pytorch、nvidia、conda-forge三个channel确保包可用性

## 安装验证

### 基础验证
```bash
# 验证PyTorch和CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 验证pytorch-3dunet (最关键)
python -c "from pytorch3dunet.unet3d.model import ResidualUNet3D; print('pytorch-3dunet: OK')"

# 验证数据处理依赖
python -c "import h5py, numpy, cv2, matplotlib; print('Data processing: OK')"
```

### 项目功能验证
```bash
# 进入项目目录
cd /mnt/e/2025/event_flick_flare/Unet_main

# 验证配置加载
python -c "from src.utils.config_loader import ConfigLoader; print('Config loader: OK')"

# 验证编解码模块
python -c "from src.data_processing.encode import load_h5_events; print('Encode module: OK')"

# 快速测试 (可选，需要数据)
python main.py test --config configs/test_config.yaml --debug
```

## 常见问题解决

### 1. numpy/h5py兼容性错误
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

**解决方法**:
```bash
pip install --force-reinstall h5py scikit-image
```

### 2. pytorch-3dunet导入失败
**检查安装**:
```bash
conda list | grep pytorch-3dunet
```

**重新安装**:
```bash
conda install -c conda-forge pytorch-3dunet=1.9.1 --force-reinstall
```

### 3. CUDA不可用
**检查NVIDIA驱动**:
```bash
nvidia-smi
```

**重新安装CUDA版本**:
```bash
conda install pytorch=2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia --force-reinstall
```

## 环境文件说明

### environment.yml
包含完整的conda环境导出，包括所有依赖包的精确版本。使用此文件可以创建与验证环境完全相同的环境。

### 关键文件位置
```
项目根目录/
├── environment.yml          # conda环境文件
├── configs/                 # 配置文件
│   ├── train_config.yaml
│   ├── test_config.yaml
│   └── inference_config.yaml
├── checkpoints/             # 训练权重
└── src/                     # 源代码
```

## 使用指南

### 训练
```bash
conda activate Umain
python main.py train --config configs/train_config.yaml
```

### 测试 (带可视化)
```bash
conda activate Umain
python main.py test --config configs/test_config.yaml --debug
```

### 推理
```bash
conda activate Umain
python main.py inference --config configs/inference_config.yaml --input input.h5 --output output.h5
```

## 性能验证结果

- ✅ **GPU加速**: 模型成功加载到GPU进行推理
- ✅ **权重加载**: checkpoint文件正确加载，7百万参数TrueResidualUNet3D
- ✅ **可视化系统**: debug模式生成完整的8模块可视化
- ✅ **数据处理**: H5文件读写、事件编解码正常
- ✅ **真正残差学习**: input + residual → output 数学验证通过

---

**注意**: 此环境配置已在实际项目中验证，包括训练、测试、推理全流程。建议优先使用环境克隆方法以确保最佳兼容性。