"""
Custom Training System for Event-Voxel Denoising
只使用pytorch-3dunet的UNet3D模型，自定义训练循环
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import logging
from pathlib import Path
import time
from typing import Dict, Any, Optional
import json

class EventVoxelTrainer:
    """
    自定义训练器，只使用pytorch-3dunet的UNet3D模型
    实现完整的训练循环、验证、checkpointing
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: Dict[str, Any],
                 device: str = 'cuda'):
        """
        初始化训练器
        
        Args:
            model: pytorch-3dunet的UNet3D模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器  
            config: 训练配置字典
            device: 训练设备
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # 设置logging
        self.logger = logging.getLogger(__name__)
        
        # 初始化优化器和损失函数
        self._setup_optimizer_and_loss()
        
        # 初始化调度器
        self._setup_scheduler()
        
        # 训练状态跟踪
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_val_loss = float('inf')
        
        # 设置checkpoint目录
        self.checkpoint_dir = Path(config['trainer']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置tensorboard
        if config.get('logger', {}).get('name') == 'TensorBoardLogger':
            log_dir = config['logger']['log_dir']
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        self.logger.info(f"EventVoxelTrainer initialized:")
        self.logger.info(f"  Device: {device}")
        self.logger.info(f"  Model: {model.__class__.__name__}")
        self.logger.info(f"  Train samples: {len(train_loader.dataset)}")
        self.logger.info(f"  Val samples: {len(val_loader.dataset)}")
        self.logger.info(f"  Checkpoint dir: {self.checkpoint_dir}")
    
    def _setup_optimizer_and_loss(self):
        """设置优化器和损失函数"""
        opt_config = self.config['optimizer']
        
        if opt_config['name'] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0)
            )
        elif opt_config['name'] == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 0)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['name']}")
        
        # 损失函数
        loss_config = self.config['loss']
        if loss_config['name'] == 'MSELoss':
            self.criterion = nn.MSELoss()
        elif loss_config['name'] == 'L1Loss':
            self.criterion = nn.L1Loss()
        elif loss_config['name'] == 'SmoothL1Loss':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss: {loss_config['name']}")
        
        self.logger.info(f"Optimizer: {opt_config['name']} (LR: {opt_config['learning_rate']})")
        self.logger.info(f"Loss function: {loss_config['name']}")
    
    def _setup_scheduler(self):
        """设置学习率调度器"""
        if 'lr_scheduler' not in self.config:
            self.scheduler = None
            return
        
        sched_config = self.config['lr_scheduler']
        
        if sched_config['name'] == 'MultiStepLR':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=sched_config['milestones'],
                gamma=sched_config['gamma']
            )
        elif sched_config['name'] == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        else:
            self.scheduler = None
            self.logger.warning(f"Unsupported scheduler: {sched_config['name']}")
        
        if self.scheduler:
            self.logger.info(f"Scheduler: {sched_config['name']}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 数据移动到设备
            inputs = batch['raw'].to(self.device)      # (B, 1, 8, H, W) 
            targets = batch['label'].to(self.device)   # (B, 1, 8, H, W)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)               # (B, 1, 8, H, W)
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            self.current_iteration += 1
            
            # 定期日志
            trainer_config = self.config['trainer']
            if self.current_iteration % trainer_config.get('log_after_iters', 10) == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch:3d} | "
                    f"Batch {batch_idx:3d}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )
                
                # Tensorboard记录
                if self.writer:
                    self.writer.add_scalar('Loss/Train_Batch', loss.item(), self.current_iteration)
                    self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_iteration)
            
            # 早期停止检查（基于迭代数）
            max_iters = trainer_config.get('max_num_iterations')
            if max_iters and self.current_iteration >= max_iters:
                self.logger.info(f"Reached maximum iterations: {max_iters}")
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss, 'num_batches': num_batches}
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['raw'].to(self.device)
                targets = batch['label'].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return {'loss': avg_loss, 'num_batches': num_batches}
    
    def save_checkpoint(self, is_best: bool = False):
        """保存checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 保存最新checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path}")
        
        # 保存epoch checkpoint
        epoch_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch:04d}.pth'
        torch.save(checkpoint, epoch_path)
        
        self.logger.info(f"Saved checkpoint: {latest_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载checkpoint"""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.best_val_loss = checkpoint['best_val_loss']
            
            if self.scheduler and 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
    
    def train(self):
        """完整训练流程"""
        self.logger.info("=== Starting Event-Voxel Denoising Training ===")
        
        trainer_config = self.config['trainer']
        max_epochs = trainer_config.get('max_num_epochs', 100)
        max_iters = trainer_config.get('max_num_iterations', None)
        validate_after_iters = trainer_config.get('validate_after_iters', 100)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            
            self.logger.info(f"\n=== Epoch {epoch + 1}/{max_epochs} ===")
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证（定期进行）
            if (self.current_iteration % validate_after_iters == 0) or (epoch == max_epochs - 1):
                val_metrics = self.validate_epoch()
                
                # 记录日志
                self.logger.info(
                    f"Epoch {epoch + 1:3d} completed | "
                    f"Train Loss: {train_metrics['loss']:.6f} | "
                    f"Val Loss: {val_metrics['loss']:.6f}"
                )
                
                # Tensorboard记录  
                if self.writer:
                    self.writer.add_scalar('Loss/Train_Epoch', train_metrics['loss'], epoch)
                    self.writer.add_scalar('Loss/Val_Epoch', val_metrics['loss'], epoch)
                
                # 保存checkpoint
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                
                self.save_checkpoint(is_best=is_best)
            
            else:
                # 只记录训练loss
                self.logger.info(f"Epoch {epoch + 1:3d} completed | Train Loss: {train_metrics['loss']:.6f}")
                if self.writer:
                    self.writer.add_scalar('Loss/Train_Epoch', train_metrics['loss'], epoch)
                
                self.save_checkpoint(is_best=False)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 早期停止检查（基于迭代数）
            if max_iters and self.current_iteration >= max_iters:
                self.logger.info(f"Training stopped: reached max iterations {max_iters}")
                # 在训练结束前保存final checkpoint
                val_metrics = self.validate_epoch()
                self.logger.info(f"Final validation | Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f}")
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=is_best)
                break
        
        # 训练结束
        total_time = time.time() - start_time
        self.logger.info(f"\n=== Training Completed ===")
        self.logger.info(f"Total training time: {total_time/3600:.2f} hours")
        self.logger.info(f"Final validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Best checkpoint: {self.checkpoint_dir}/best_checkpoint.pth")
        
        # 保存训练摘要
        summary = {
            'total_epochs': self.current_epoch + 1,
            'total_iterations': self.current_iteration,
            'best_val_loss': self.best_val_loss,
            'total_time_hours': total_time / 3600,
            'final_lr': self.optimizer.param_groups[0]['lr']
        }
        
        summary_path = self.checkpoint_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.writer:
            self.writer.close()
        
        return self.best_val_loss