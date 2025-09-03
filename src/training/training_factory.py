"""
Training Factory for Event-Voxel Denoising
负责初始化模型、数据加载器等训练组件
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.datasets.event_voxel_dataset import EventVoxelDataset
from src.training.custom_trainer import EventVoxelTrainer


class TrainingFactory:
    """
    训练工厂类，负责创建和配置训练所需的所有组件
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练工厂
        
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_model(self) -> nn.Module:
        """
        创建pytorch-3dunet的UNet3D模型
        
        Returns:
            UNet3D模型实例
        """
        model_config = self.config['model']
        
        if model_config['name'] == 'ResidualUNet3D':
            try:
                # 尝试导入pytorch-3dunet的ResidualUNet3D
                try:
                    from pytorch3dunet.unet3d.model import ResidualUNet3D
                except ImportError:
                    # 备选导入路径
                    from pytorch3dunet.unet3d import ResidualUNet3D
                
                # 创建模型
                model = ResidualUNet3D(
                    in_channels=model_config['in_channels'],
                    out_channels=model_config['out_channels'],
                    f_maps=model_config.get('f_maps', 32),
                    layer_order=model_config.get('layer_order', 'gcr'),
                    num_groups=model_config.get('num_groups', 8),
                    num_levels=model_config.get('num_levels', 4),
                    final_sigmoid=True,  # 设为True避免Softmax bug，下面强制替换为Identity
                    conv_kernel_size=model_config.get('conv_kernel_size', 3),
                    pool_kernel_size=model_config.get('pool_kernel_size', 2)
                )
                
                # 强制替换激活函数为Identity以支持无界voxel输出
                # pytorch-3dunet设计问题: final_sigmoid=False会用Softmax，单通道时输出全1
                if hasattr(model, 'final_activation'):
                    import torch.nn as nn
                    original_activation = str(model.final_activation)
                    model.final_activation = nn.Identity()
                    self.logger.info(f"Forced replacement: {original_activation} -> Identity() for unbounded voxel values")
                else:
                    self.logger.warning("Could not find final_activation layer to replace")
                
                # 计算参数数量
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                self.logger.info(f"Created ResidualUNet3D model:")
                self.logger.info(f"  - Architecture: {model_config['f_maps']} feature maps, {model_config.get('num_levels', 4)} levels")
                self.logger.info(f"  - Input channels: {model_config['in_channels']}")
                self.logger.info(f"  - Output channels: {model_config['out_channels']}")
                self.logger.info(f"  - Residual connections: Enabled for deflare task")
                self.logger.info(f"  - Total parameters: {total_params:,}")
                self.logger.info(f"  - Trainable parameters: {trainable_params:,}")
                
                return model
                
            except ImportError as e:
                self.logger.error(f"Failed to import pytorch-3dunet ResidualUNet3D: {e}")
                self.logger.error("Please install pytorch-3dunet: conda install -c conda-forge pytorch-3dunet")
                raise
            except Exception as e:
                self.logger.error(f"Failed to create ResidualUNet3D model: {e}")
                raise
                
        elif model_config['name'] == 'TrueResidualUNet3D':
            # 真正的残差学习实现
            try:
                import sys
                from pathlib import Path
                sys.path.append(str(Path(__file__).parent.parent.parent))
                from true_residual_wrapper import TrueResidualUNet3D
                
                model = TrueResidualUNet3D(
                    in_channels=model_config['in_channels'],
                    out_channels=model_config['out_channels'],
                    f_maps=model_config.get('f_maps', [16, 32, 64]),
                    num_levels=model_config.get('num_levels', 3),
                    backbone=model_config.get('backbone', 'ResidualUNet3D')
                )
                
                # 计算参数数量
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                self.logger.info(f"Created TrueResidualUNet3D model:")
                self.logger.info(f"  - Architecture: {model_config.get('f_maps', [16, 32, 64])} feature maps, {model_config.get('num_levels', 3)} levels")
                self.logger.info(f"  - Backbone: {model_config.get('backbone', 'ResidualUNet3D')}")
                self.logger.info(f"  - True residual learning: output = input + backbone(input)")
                self.logger.info(f"  - Zero-initialized final layer for perfect identity mapping")
                self.logger.info(f"  - Input channels: {model_config['in_channels']}")
                self.logger.info(f"  - Output channels: {model_config['out_channels']}")
                self.logger.info(f"  - Total parameters: {total_params:,}")
                self.logger.info(f"  - Trainable parameters: {trainable_params:,}")
                
                return model
                
            except ImportError as e:
                self.logger.error(f"Failed to import TrueResidualUNet3D: {e}")
                self.logger.error("Make sure true_residual_wrapper.py is in the project root")
                raise
            except Exception as e:
                self.logger.error(f"Failed to create TrueResidualUNet3D model: {e}")
                raise
        
        else:
            raise ValueError(f"Unsupported model: {model_config['name']}")
    
    def create_datasets(self) -> Tuple[EventVoxelDataset, EventVoxelDataset]:
        """
        创建训练和验证数据集
        
        Returns:
            (train_dataset, val_dataset)
        """
        loader_config = self.config['loaders']
        
        # 提取数据路径（新的deflare配置格式）
        train_noisy_dir = loader_config.get('train_noisy_dir')
        train_clean_dir = loader_config.get('train_clean_dir')
        val_noisy_dir = loader_config.get('val_noisy_dir')
        val_clean_dir = loader_config.get('val_clean_dir')
        
        # 后向兼容：支持旧的train_path/val_path格式
        if not train_noisy_dir:
            train_paths = loader_config.get('train_path', [])
            if isinstance(train_paths, list):
                train_noisy_dir = train_paths[0] + '/noisy' if len(train_paths) > 0 else 'train/noisy'
                train_clean_dir = train_paths[0] + '/clean' if len(train_paths) > 0 else 'train/clean'
            else:
                train_noisy_dir = str(train_paths) + '/noisy'
                train_clean_dir = str(train_paths) + '/clean'
        
        if not val_noisy_dir:
            val_paths = loader_config.get('val_path', [])
            if isinstance(val_paths, list):
                val_noisy_dir = val_paths[0] + '/noisy' if len(val_paths) > 0 else 'val/noisy' 
                val_clean_dir = val_paths[0] + '/clean' if len(val_paths) > 0 else 'val/clean'
            else:
                val_noisy_dir = str(val_paths) + '/noisy'
                val_clean_dir = str(val_paths) + '/clean'
        
        # 传感器尺寸（默认DSEC格式）
        sensor_size = tuple(loader_config.get('sensor_size', [480, 640]))
        
        # 分段参数
        segment_duration_us = loader_config.get('segment_duration_us', 20000)  # 20ms
        num_bins = loader_config.get('num_bins', 8)                            # 8 bins
        num_segments = loader_config.get('num_segments', 5)                    # 5 segments
        
        self.logger.info(f"Creating datasets:")
        self.logger.info(f"  - Sensor size: {sensor_size}")
        self.logger.info(f"  - Segment config: {segment_duration_us/1000}ms/{num_bins}bins = {segment_duration_us/num_bins/1000:.2f}ms per bin")
        self.logger.info(f"  - Segments per file: {num_segments}")
        
        # 创建训练数据集
        try:
            train_dataset = EventVoxelDataset(
                noisy_events_dir=train_noisy_dir,
                clean_events_dir=train_clean_dir,
                sensor_size=sensor_size,
                segment_duration_us=segment_duration_us,
                num_bins=num_bins,
                num_segments=num_segments
            )
            
            self.logger.info(f"Training dataset: {len(train_dataset)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to create training dataset: {e}")
            self.logger.error(f"Train paths - Noisy: {train_noisy_dir}, Clean: {train_clean_dir}")
            raise
        
        # 创建验证数据集
        try:
            val_dataset = EventVoxelDataset(
                noisy_events_dir=val_noisy_dir,
                clean_events_dir=val_clean_dir,
                sensor_size=sensor_size,
                segment_duration_us=segment_duration_us,
                num_bins=num_bins,
                num_segments=num_segments
            )
            
            self.logger.info(f"Validation dataset: {len(val_dataset)} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to create validation dataset: {e}")
            self.logger.error(f"Val paths - Noisy: {val_noisy_dir}, Clean: {val_clean_dir}")
            raise
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset: EventVoxelDataset, 
                          val_dataset: EventVoxelDataset) -> Tuple[DataLoader, DataLoader]:
        """
        创建数据加载器
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            
        Returns:
            (train_loader, val_loader)
        """
        loader_config = self.config['loaders']
        
        batch_size = loader_config.get('batch_size', 1)
        num_workers = loader_config.get('num_workers', 2)
        
        # 创建训练数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # 确保batch大小一致
        )
        
        # 创建验证数据加载器
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        self.logger.info(f"Created data loaders:")
        self.logger.info(f"  - Batch size: {batch_size}")
        self.logger.info(f"  - Num workers: {num_workers}")
        self.logger.info(f"  - Train batches: {len(train_loader)}")
        self.logger.info(f"  - Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def create_trainer(self, model: nn.Module, 
                      train_loader: DataLoader, 
                      val_loader: DataLoader,
                      device: str = 'cuda') -> EventVoxelTrainer:
        """
        创建训练器
        
        Args:
            model: UNet3D模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 训练设备
            
        Returns:
            EventVoxelTrainer实例
        """
        # 确定设备
        if device == 'cuda' and not torch.cuda.is_available():
            self.logger.warning("CUDA not available, falling back to CPU")
            device = 'cpu'
        
        # 创建训练器
        trainer = EventVoxelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=self.config,
            device=device
        )
        
        return trainer
    
    def setup_complete_training(self, device: Optional[str] = None) -> EventVoxelTrainer:
        """
        一站式设置完整训练流程
        
        Args:
            device: 训练设备，None则自动选择
            
        Returns:
            已配置好的EventVoxelTrainer
        """
        self.logger.info("=== Setting up Event-Voxel Denoising Training ===")
        
        # 自动选择设备
        if device is None:
            device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 创建模型
        self.logger.info("Step 1: Creating UNet3D model...")
        model = self.create_model()
        
        # 2. 创建数据集
        self.logger.info("Step 2: Creating datasets...")
        train_dataset, val_dataset = self.create_datasets()
        
        # 3. 创建数据加载器
        self.logger.info("Step 3: Creating data loaders...")
        train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)
        
        # 4. 创建训练器
        self.logger.info("Step 4: Creating trainer...")
        trainer = self.create_trainer(model, train_loader, val_loader, device)
        
        # 5. 加载checkpoint（如果指定）
        resume_path = self.config['trainer'].get('resume')
        if resume_path:
            self.logger.info(f"Step 5: Resuming from checkpoint: {resume_path}")
            success = trainer.load_checkpoint(resume_path)
            if not success:
                self.logger.warning("Failed to load checkpoint, starting from scratch")
        
        self.logger.info("=== Training setup completed ===")
        return trainer