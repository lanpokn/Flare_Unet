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
from tqdm import tqdm

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
        
        # 添加模型架构检查
        self._debug_model_architecture()
    
    def _debug_model_architecture(self):
        """调试模型架构，检查可能导致输出全1的问题"""
        self.logger.info("=== Model Architecture Debug ===")
        
        # 检查final_sigmoid设置
        if hasattr(self.model, 'final_sigmoid'):
            self.logger.info(f"Model final_sigmoid: {self.model.final_sigmoid}")
        
        # 检查模型的最后几层
        model_children = list(self.model.children())
        if model_children:
            last_layer = model_children[-1]
            self.logger.info(f"Last layer type: {type(last_layer)}")
            
        # 快速前向传播测试
        self.model.eval()
        with torch.no_grad():
            # 创建测试输入: 全0
            test_input_zeros = torch.zeros(1, 1, 8, 480, 640).to(self.device)
            test_output_zeros = self.model(test_input_zeros)
            self.logger.info(f"Test with zeros input: output_mean={test_output_zeros.mean():.6f}, output_std={test_output_zeros.std():.6f}")
            
            # 创建测试输入: 全1
            test_input_ones = torch.ones(1, 1, 8, 480, 640).to(self.device)
            test_output_ones = self.model(test_input_ones)
            self.logger.info(f"Test with ones input: output_mean={test_output_ones.mean():.6f}, output_std={test_output_ones.std():.6f}")
            
            # 创建测试输入: 随机
            test_input_random = torch.randn(1, 1, 8, 480, 640).to(self.device)
            test_output_random = self.model(test_input_random)
            self.logger.info(f"Test with random input: output_mean={test_output_random.mean():.6f}, output_std={test_output_random.std():.6f}")
        
        self.model.train()
        self.logger.info("=== End Model Debug ===")
    
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
        
        # 创建tqdm进度条
        progress_bar = tqdm(
            enumerate(self.train_loader), 
            total=len(self.train_loader),
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
            ncols=100
        )
        
        for batch_idx, batch in progress_bar:
            # 检查debug模式是否需要提前退出
            debug_config = self.config.get('debug', {})
            if debug_config.get('enabled', False):
                max_iterations = debug_config.get('max_iterations', 2)
                if batch_idx >= max_iterations:
                    self.logger.info(f"🐛 DEBUG MODE: Stopping after {max_iterations} iterations")
                    break
            
            # 数据移动到设备
            inputs = batch['raw'].to(self.device)      # (B, 1, 8, H, W) 
            targets = batch['label'].to(self.device)   # (B, 1, 8, H, W)
            
            # Debug可视化钩子 - 在数据产生的地方触发
            if debug_config.get('enabled', False) and batch_idx < 2:  # 只对前2个batch做可视化
                self._trigger_debug_visualization(
                    batch_idx, inputs, targets, batch, 
                    debug_config['debug_dir'], self.current_epoch
                )
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)               # (B, 1, 8, H, W)
            
            # Debug可视化钩子 - 模型输出后
            if debug_config.get('enabled', False) and batch_idx < 2:
                self._trigger_model_output_visualization(
                    batch_idx, outputs, debug_config['debug_dir'], self.current_epoch
                )
            
            # 深度调试训练阶段 (每100个batch打印一次)
            if batch_idx % 100 == 0:
                print(f"\n[DEBUG-TRAIN] Batch {batch_idx}: Input mean={inputs.mean():.4f}, Output mean={outputs.mean():.4f}")
                print(f"[DEBUG-TRAIN] Are outputs identical to inputs? {torch.equal(outputs, inputs)}")
                print(f"[DEBUG-TRAIN] Model in training mode? {self.model.training}")
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            num_batches += 1
            self.current_iteration += 1
            
            # 更新进度条显示
            avg_loss_so_far = total_loss / num_batches
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{avg_loss_so_far:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Tensorboard记录
            if self.writer:
                self.writer.add_scalar('Loss/Train_Batch', loss.item(), self.current_iteration)
                self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.current_iteration)
            
            # 检查是否需要验证 (每N个iteration)
            trainer_config = self.config['trainer']
            validate_after_iters = trainer_config.get('validate_after_iters', 100)
            
            if self.current_iteration % validate_after_iters == 0:
                # 调试输出验证触发
                print(f"\n[DEBUG] Validation triggered: iter={self.current_iteration}, validate_after_iters={validate_after_iters}", flush=True)
                
                # 暂停训练进度条，进行验证
                progress_bar.set_description(f"Epoch {self.current_epoch + 1} (Validating...)")
                
                # 检查模型参数是否在变化 (调试)
                model_params_sum = sum([p.sum().item() for p in self.model.parameters()])
                print(f"[DEBUG] Model params sum: {model_params_sum:.6f}", flush=True)
                
                # 执行验证
                val_metrics = self.validate_epoch()
                
                # 检查是否是最佳模型
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                
                # 保存checkpoint (添加调试信息)
                try:
                    self.save_checkpoint(is_best=is_best)
                    checkpoint_name = f"epoch_{self.current_epoch:04d}_iter_{self.current_iteration:06d}"
                    checkpoint_status = f"✅({checkpoint_name})" 
                except Exception as e:
                    checkpoint_status = f"❌({e})"
                
                # 输出验证结果 (强制刷新输出)
                best_indicator = " 🎯" if is_best else ""
                result_msg = f"\n💯 Iter {self.current_iteration:4d}: Val={val_metrics['loss']:.4f}{best_indicator} {checkpoint_status}"
                print(result_msg, flush=True)
                
                # 恢复训练进度条描述
                progress_bar.set_description(f"Epoch {self.current_epoch + 1}")
            
            # 早期停止检查（基于迭代数）
            max_iters = trainer_config.get('max_num_iterations')
            if max_iters and self.current_iteration >= max_iters:
                progress_bar.set_description(f"Epoch {self.current_epoch + 1} (Max iters reached)")
                break
        
        progress_bar.close()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return {'loss': avg_loss, 'num_batches': num_batches}
    
    def validate_epoch(self) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # 限制验证样本数量（快速验证模式）
        # 50个test文件 × 5个segments = 250个验证样本
        # 只验证前2个文件 × 5个segments = 10个样本
        max_val_batches = 10  # 前2个文件的10个数据对
        
        # 修复验证逻辑：不转换为list，直接遍历并计数
        batch_losses = []  # 记录每个batch的loss
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= max_val_batches:
                    break
                inputs = batch['raw'].to(self.device)
                targets = batch['label'].to(self.device)
                
                # 深度调试：检查模型输入输出
                if batch_idx == 0:  # 只在第一个batch打印
                    print(f"[DEBUG] Input stats: min={inputs.min():.4f}, max={inputs.max():.4f}, mean={inputs.mean():.4f}")
                    print(f"[DEBUG] Target stats: min={targets.min():.4f}, max={targets.max():.4f}, mean={targets.mean():.4f}")
                    # 检查voxel值分布
                    unique_values_input = torch.unique(inputs).cpu().numpy()[:10]  # 前10个唯一值
                    unique_values_target = torch.unique(targets).cpu().numpy()[:10]
                    print(f"[DEBUG] Input unique values (first 10): {unique_values_input}")
                    print(f"[DEBUG] Target unique values (first 10): {unique_values_target}")
                
                outputs = self.model(inputs)
                
                if batch_idx == 0:  # 只在第一个batch打印
                    print(f"[DEBUG] Output stats: min={outputs.min():.4f}, max={outputs.max():.4f}, mean={outputs.mean():.4f}")
                    print(f"[DEBUG] Output shape: {outputs.shape}")
                    print(f"[DEBUG] Are outputs identical to targets? {torch.equal(outputs, targets)}")
                    print(f"[DEBUG] Output dtype: {outputs.dtype}, device: {outputs.device}")
                    print(f"[DEBUG] Output requires_grad: {outputs.requires_grad}")
                    
                    # 检查模型的最后几层输出
                    print(f"[DEBUG] Checking model architecture...")
                
                loss = self.criterion(outputs, targets)
                
                batch_loss = loss.item()
                batch_losses.append(batch_loss)
                total_loss += batch_loss
                num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        # 调试输出验证统计
        batch_losses_str = ", ".join([f"{loss:.4f}" for loss in batch_losses])
        print(f"[DEBUG] Validation completed: {num_batches} batches, avg_loss={avg_loss:.6f}", flush=True)
        print(f"[DEBUG] Individual batch losses: [{batch_losses_str}]", flush=True)
        print(f"[DEBUG] Input shape: {inputs.shape}, Target shape: {targets.shape}", flush=True)
        
        # 恢复训练模式 (关键修复!)
        self.model.train()
        
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
            # self.logger.info(f"Saved best checkpoint: {best_path}")  # Reduce verbosity
        
        # 保存iteration checkpoint (避免覆盖)
        iter_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch:04d}_iter_{self.current_iteration:06d}.pth'
        torch.save(checkpoint, iter_path)
        
        # self.logger.info(f"Saved checkpoint: {latest_path}")  # Reduce verbosity
    
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
            
            # 简洁的epoch开始信息
            print(f"\n📊 Epoch {epoch + 1}/{max_epochs}")
            
            # 训练 - 验证逻辑现在在train_epoch内部处理
            train_metrics = self.train_epoch()
            
            # Epoch完成后的总结 - 简化版本，详细验证在train_epoch内部
            print(f"✅ Epoch {epoch + 1:3d}: Train={train_metrics['loss']:.4f}")
            
            # Tensorboard epoch级记录
            if self.writer:
                self.writer.add_scalar('Loss/Train_Epoch', train_metrics['loss'], epoch)
            
            # 每个epoch都保存checkpoint (保证不丢失)
            self.save_checkpoint(is_best=False)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 早期停止检查（基于迭代数）
            if max_iters and self.current_iteration >= max_iters:
                print(f"🛑 Training stopped: reached max iterations {max_iters}")
                # 在训练结束前保存final checkpoint
                val_metrics = self.validate_epoch()
                is_best = val_metrics['loss'] < self.best_val_loss
                best_indicator = " 🎯" if is_best else ""
                print(f"🏁 Final: Train={train_metrics['loss']:.4f}, Val={val_metrics['loss']:.4f}{best_indicator}")
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(is_best=is_best)
                break
        
        # 训练结束
        total_time = time.time() - start_time
        print(f"\n🎉 Training Completed!")
        print(f"⏱️  Total time: {total_time/3600:.2f}h")
        print(f"🎯 Best val loss: {self.best_val_loss:.4f}")
        print(f"💾 Checkpoint: {self.checkpoint_dir}/best_checkpoint.pth")
        
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
    
    def _trigger_debug_visualization(self, batch_idx: int, inputs: torch.Tensor, targets: torch.Tensor, 
                                   batch: dict, debug_dir: str, epoch: int):
        """
        Debug可视化钩子 - 在数据产生的地方触发
        生成9个独立的可视化文件夹：
        1. 输入事件3D可视化
        2. 输入事件2D可视化  
        3. 输入voxel可视化
        4. 真值事件3D可视化
        5. 真值事件2D可视化
        6. 真值voxel可视化
        """
        try:
            import os
            from pathlib import Path
            
            # 创建debug输出结构
            iteration_dir = Path(debug_dir) / f"epoch_{epoch:03d}_iter_{batch_idx:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"🐛 Generating 9 debug visualizations for Epoch {epoch}, Batch {batch_idx}")
            self.logger.info(f"🐛 Output directory: {iteration_dir}")
            
            # 获取第一个样本进行可视化 (batch_size通常为1)
            input_voxel = inputs[0, 0].cpu()   # (8, H, W) - 去除batch和channel维度
            target_voxel = targets[0, 0].cpu() # (8, H, W)
            
            # 从voxel解码回events进行可视化
            from src.data_processing.decode import voxel_to_events
            
            # 解码input和target voxel为events
            # 注意：需要使用正确的duration (20ms = 20000us)
            input_events_np = voxel_to_events(input_voxel, total_duration=20000, sensor_size=(480, 640))
            target_events_np = voxel_to_events(target_voxel, total_duration=20000, sensor_size=(480, 640))
            
            # 使用专业可视化系统 - 修复函数调用
            from src.data_processing.professional_visualizer import visualize_events, visualize_voxel
            
            # === 输入数据可视化 ===
            # 1-2. 输入事件3D+2D可视化 (使用正确的参数顺序)
            input_events_dir = iteration_dir / "1_input_events"
            input_events_dir.mkdir(exist_ok=True)
            visualize_events(input_events_np, sensor_size=(480, 640), output_dir=str(input_events_dir), 
                           name="input_events", num_time_slices=8)
            
            # 3. 输入voxel可视化
            input_voxel_dir = iteration_dir / "3_input_voxel"
            input_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(input_voxel, sensor_size=(480, 640), output_dir=str(input_voxel_dir), 
                          name="input_voxel", duration_ms=20)
            
            # === 真值数据可视化 ===
            # 4-5. 真值事件3D+2D可视化 (使用正确的参数顺序)
            target_events_dir = iteration_dir / "4_target_events"
            target_events_dir.mkdir(exist_ok=True)
            visualize_events(target_events_np, sensor_size=(480, 640), output_dir=str(target_events_dir), 
                           name="target_events", num_time_slices=8)
            
            # 6. 真值voxel可视化
            target_voxel_dir = iteration_dir / "6_target_voxel"
            target_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(target_voxel, sensor_size=(480, 640), output_dir=str(target_voxel_dir), 
                          name="target_voxel", duration_ms=20)
            
            self.logger.info(f"🐛 Generated input and target visualizations (1,3,4,6/9) in {iteration_dir}")
            
        except Exception as e:
            self.logger.warning(f"🐛 Debug visualization failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _trigger_model_output_visualization(self, batch_idx: int, outputs: torch.Tensor, 
                                          debug_dir: str, epoch: int):
        """
        Debug可视化钩子 - 模型输出后触发
        生成9个可视化文件夹中的后3个：
        7. 模型输出事件3D可视化
        8. 模型输出事件2D可视化  
        9. 模型输出voxel可视化
        """
        try:
            from pathlib import Path
            
            iteration_dir = Path(debug_dir) / f"epoch_{epoch:03d}_iter_{batch_idx:03d}"
            
            # 获取第一个样本的模型输出
            output_voxel = outputs[0, 0].cpu()  # (8, H, W)
            
            # 从output voxel解码为events
            from src.data_processing.decode import voxel_to_events
            output_events_np = voxel_to_events(output_voxel, total_duration=20000, sensor_size=(480, 640))
            
            # 使用专业可视化系统 - 修复函数调用
            from src.data_processing.professional_visualizer import visualize_events, visualize_voxel
            
            # === 模型输出可视化 ===
            # 7-8. 模型输出事件3D+2D可视化 (使用正确的参数顺序)
            output_events_dir = iteration_dir / "7_output_events"
            output_events_dir.mkdir(exist_ok=True)
            visualize_events(output_events_np, sensor_size=(480, 640), output_dir=str(output_events_dir), 
                           name="output_events", num_time_slices=8)
            
            # 9. 模型输出voxel可视化
            output_voxel_dir = iteration_dir / "9_output_voxel"
            output_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(output_voxel, sensor_size=(480, 640), output_dir=str(output_voxel_dir), 
                          name="output_voxel", duration_ms=20)
            
            self.logger.info(f"🐛 Generated model output visualizations (7,9/9) in {iteration_dir}")
            self.logger.info(f"🐛 All debug visualizations completed! (6 folders total: 1,3,4,6,7,9)")
            
            # 生成比较总结
            self._generate_debug_summary(iteration_dir, batch_idx, epoch)
            
        except Exception as e:
            self.logger.warning(f"🐛 Model output visualization failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _generate_debug_summary(self, iteration_dir: Path, batch_idx: int, epoch: int):
        """生成debug总结信息"""
        try:
            summary_file = iteration_dir / "debug_summary.txt"
            
            with open(summary_file, 'w') as f:
                f.write(f"Debug Visualization Summary\n")
                f.write(f"========================\n")
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Batch: {batch_idx}\n")
                f.write(f"Model: {self.model.__class__.__name__}\n")
                f.write(f"Device: {self.device}\n")
                f.write(f"\n6 Visualization Folders (comprehensive):\n")
                f.write(f"1. 1_input_events/          - Input events (3D+2D+temporal) comprehensive\n")
                f.write(f"2. 3_input_voxel/           - Input voxel temporal bins\n")
                f.write(f"3. 4_target_events/         - Target events (3D+2D+temporal) comprehensive\n") 
                f.write(f"4. 6_target_voxel/          - Target voxel temporal bins\n")
                f.write(f"5. 7_output_events/         - Model output events (3D+2D+temporal) comprehensive\n")
                f.write(f"6. 9_output_voxel/          - Model output voxel temporal bins\n")
                f.write(f"\nData Format:\n")
                f.write(f"- Events: (N, 4) [t, x, y, p]\n")
                f.write(f"- Voxel: (8, 480, 640) [8 temporal bins]\n")
                f.write(f"- Duration: 20ms per segment\n")
                
            self.logger.info(f"🐛 Debug summary saved to {summary_file}")
            
        except Exception as e:
            self.logger.warning(f"🐛 Failed to generate debug summary: {e}")