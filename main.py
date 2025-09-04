#!/usr/bin/env python3
"""
Main entry point for Event-Voxel Denoising with Custom Training System
Supports three modes: train, test, inference

Usage:
    python main.py train --config configs/train_config.yaml
    python main.py test --config configs/test_config.yaml  
    python main.py inference --config configs/inference_config.yaml --input input.h5 --output output.h5
"""

import argparse
import sys
import logging
from pathlib import Path
import torch
import numpy as np
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.config_loader import ConfigLoader
from src.data_processing.encode import load_h5_events, events_to_voxel
from src.data_processing.decode import voxel_to_events
from src.training.training_factory import TrainingFactory

class EventVoxelPipeline:
    """
    Main pipeline for event-voxel denoising system
    Integrates pytorch-3dunet with custom event processing
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.config_loader = ConfigLoader()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
    
    def train_mode(self, config_path: str, debug: bool = False, debug_dir: str = 'debug_output') -> int:
        """
        Training mode - uses custom training system with pytorch-3dunet UNet3D model
        
        Args:
            config_path: Path to training configuration YAML
            debug: Enable debug mode with detailed visualization
            debug_dir: Directory for debug visualizations
            
        Returns:
            Exit code (0 = success, non-zero = error)
        """
        mode_str = "DEBUG TRAINING" if debug else "CUSTOM TRAINING"
        self.logger.info(f"=== EVENT-VOXEL {mode_str} MODE ===")
        
        try:
            # Load and validate training configuration
            config = self.config_loader.load_train_config(config_path)
            self.logger.info(f"Loaded training config from: {config_path}")
            
            # Inject debug configuration
            if debug:
                config['debug'] = {
                    'enabled': True,
                    'debug_dir': debug_dir,
                    'max_iterations': 2,  # Only run 1-2 iterations in debug mode
                    'max_epochs': 1       # Only 1 epoch in debug mode
                }
                
                # Override training epochs for debug
                config['trainer']['max_num_epochs'] = 1
                
                # Create debug output directory
                debug_path = Path(debug_dir)
                debug_path.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"🐛 DEBUG MODE ENABLED")
                self.logger.info(f"🐛 Debug output: {debug_dir}")
                self.logger.info(f"🐛 Will run max {config['debug']['max_iterations']} iterations only")
                self.logger.info(f"🐛 9 visualization folders will be generated per iteration")
            else:
                config['debug'] = {'enabled': False}
            
            self.logger.info("Starting custom training system...")
            self.logger.info(f"Model: {config['model']['name']} ({config['model']['f_maps']} feature maps)")
            self.logger.info(f"Loss: {config['loss']['name']}")
            self.logger.info(f"Optimizer: {config['optimizer']['name']} (LR: {config['optimizer']['learning_rate']})")
            
            # 创建训练工厂
            training_factory = TrainingFactory(config)
            
            # 设置完整训练流程
            trainer = training_factory.setup_complete_training()
            
            # 开始训练
            best_val_loss = trainer.train()
            
            self.logger.info("Training completed successfully!")
            self.logger.info(f"Best validation loss: {best_val_loss:.6f}")
            self.logger.info(f"Checkpoints saved to: {config['trainer']['checkpoint_dir']}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Training mode error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 1
    
    def test_mode(self, config_path: str, debug: bool = False, debug_dir: str = 'debug_output') -> int:
        """
        Testing mode - evaluates trained model on test dataset with optional debug visualization
        
        Args:
            config_path: Path to test configuration YAML
            debug: Enable debug mode with detailed visualization
            debug_dir: Directory for debug visualizations
            
        Returns:
            Exit code (0 = success, non-zero = error)
        """
        mode_str = "DEBUG TESTING" if debug else "TESTING"
        self.logger.info(f"=== EVENT-VOXEL {mode_str} MODE ===")
        
        try:
            # Load and validate test configuration  
            config = self.config_loader.load_test_config(config_path)
            self.logger.info(f"Loaded test config from: {config_path}")
            
            # Inject debug configuration (same as train mode)
            if debug:
                config['debug'] = {
                    'enabled': True,
                    'debug_dir': debug_dir
                }
                
                # Create debug output directory
                debug_path = Path(debug_dir)
                debug_path.mkdir(parents=True, exist_ok=True)
                
                self.logger.info(f"🐛 DEBUG MODE ENABLED")
                self.logger.info(f"🐛 Debug output: {debug_dir}")
                self.logger.info(f"🐛 Smart sampling: visualizing every 5th batch (1st segment of each file)")
                self.logger.info(f"🐛 Expected visualizations: ~52 folders (1 per file)")
            else:
                config['debug'] = {'enabled': False}
            
            # Verify model checkpoint exists
            model_path = config['model']['path']
            if not Path(model_path).exists():
                self.logger.error(f"Model checkpoint not found: {model_path}")
                return 1
            
            self.logger.info("Starting custom evaluation...")
            self.logger.info(f"Model: {config['model']['name']} ({config['model']['f_maps']} feature maps)")
            self.logger.info(f"Checkpoint: {model_path}")
            
            # 创建训练工厂（仅用于创建模型和数据集）
            training_factory = TrainingFactory(config)
            
            # 创建模型和数据集
            model = training_factory.create_model()
            # For test mode: only create validation dataset (which is our test data)
            try:
                _, test_dataset = training_factory.create_datasets()  # 使用val作为test
            except:
                # If train dataset creation fails, just create val dataset directly
                from src.datasets.event_voxel_dataset import EventVoxelDataset
                loader_config = config.get('loaders', {})
                val_noisy_dir = loader_config.get('val_noisy_dir')
                val_clean_dir = loader_config.get('val_clean_dir')
                sensor_size = tuple(loader_config.get('sensor_size', [480, 640]))
                segment_duration_us = loader_config.get('segment_duration_us', 20000)
                num_bins = loader_config.get('num_bins', 8)
                num_segments = loader_config.get('num_segments', 5)
                
                test_dataset = EventVoxelDataset(
                    noisy_events_dir=val_noisy_dir,
                    clean_events_dir=val_clean_dir,
                    sensor_size=sensor_size,
                    segment_duration_us=segment_duration_us,
                    num_bins=num_bins,
                    num_segments=num_segments
                )
            
            _, test_loader = training_factory.create_dataloaders(test_dataset, test_dataset)
            
            # 加载checkpoint
            device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(model_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            
            # 评估模型
            total_loss = 0.0
            num_batches = 0
            criterion = torch.nn.MSELoss()
            
            # Test模式：处理所有数据，debug模式只影响可视化
            debug_config = config.get('debug', {})
            
            self.logger.info(f"Evaluating on {len(test_dataset)} samples...")
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                        
                    inputs = batch['raw'].to(device)
                    targets = batch['label'].to(device)
                    
                    # Debug可视化钩子 - 每5个batch可视化第1个 (batch_idx % 5 == 0)
                    # 因为每5个batch来自同一个文件，相似性高，只需要可视化每个文件的第1个segment
                    if debug_config.get('enabled', False) and batch_idx % 5 == 0:
                        file_idx = batch_idx // 5  # 文件索引
                        self.logger.info(f"🐛 Visualizing file {file_idx + 1} (batch {batch_idx})")
                        self._trigger_test_debug_visualization(
                            batch_idx, inputs, targets, batch, 
                            debug_config['debug_dir'], batch_idx  # 使用batch_idx作为epoch
                        )
                    
                    outputs = model(inputs)
                    
                    # Debug可视化钩子 - 每5个batch可视化第1个
                    if debug_config.get('enabled', False) and batch_idx % 5 == 0:
                        self._trigger_test_model_output_visualization(
                            batch_idx, inputs, outputs, debug_config['debug_dir'], batch_idx, model
                        )
                    
                    loss = criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0 or (debug_config.get('enabled', False) and batch_idx < 5):
                        self.logger.info(f"Batch {batch_idx}/{len(test_loader)}, Loss: {loss.item():.6f}")
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            
            self.logger.info("Testing completed successfully!")
            self.logger.info(f"Average test loss: {avg_loss:.6f}")
            
            # 保存结果
            results = {
                'test_loss': avg_loss,
                'num_samples': len(test_dataset),
                'model_path': model_path
            }
            
            output_dir = Path(config.get('predictor', {}).get('output_dir', 'test_results'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_dir / 'test_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Results saved to: {output_dir}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Testing mode error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 1
    
    def inference_mode(self, config_path: str, input_file: str, output_file: str) -> int:
        """
        Inference mode - custom end-to-end event denoising
        Pipeline: H5 events → segments → voxels → model → denoised voxels → events → H5
        
        Args:
            config_path: Path to inference configuration YAML
            input_file: Input H5 events file
            output_file: Output H5 events file
            
        Returns:
            Exit code (0 = success, non-zero = error)
        """
        self.logger.info("=== EVENT-VOXEL INFERENCE MODE ===")
        
        try:
            # Load and validate inference configuration
            config = self.config_loader.load_inference_config(config_path)
            self.logger.info(f"Loaded inference config from: {config_path}")
            
            # Verify input file exists
            if not Path(input_file).exists():
                self.logger.error(f"Input file not found: {input_file}")
                return 1
            
            # Verify model checkpoint exists
            model_path = config['model']['path']
            if not Path(model_path).exists():
                self.logger.error(f"Model checkpoint not found: {model_path}")
                return 1
            
            # Extract inference parameters
            inf_config = config['inference']
            sensor_size = tuple(inf_config['sensor_size'])
            segment_duration_us = inf_config['segment_duration_us']
            num_bins = inf_config['num_bins']
            num_segments = inf_config['num_segments']
            device = inf_config['device']
            
            self.logger.info(f"Input: {input_file}")
            self.logger.info(f"Output: {output_file}")
            self.logger.info(f"Model: {model_path}")
            self.logger.info(f"Processing: {num_segments} segments × {segment_duration_us/1000}ms × {num_bins} bins")
            self.logger.info(f"Device: {device}")
            
            # Step 1: Load model
            self.logger.info("Loading trained model...")
            model = self._load_model(config['model'], device)
            
            # Step 2: Load and segment input events
            self.logger.info("Loading input events...")
            events_np = load_h5_events(input_file)
            self.logger.info(f"Loaded {len(events_np)} events")
            
            # Step 3: Process each segment
            self.logger.info("Processing segments through denoising pipeline...")
            denoised_segments = []
            
            for segment_idx in range(num_segments):
                self.logger.info(f"Processing segment {segment_idx + 1}/{num_segments}...")
                
                # Extract segment events
                segment_events = self._extract_segment_events(
                    events_np, segment_idx, num_segments
                )
                
                if len(segment_events) == 0:
                    self.logger.warning(f"Segment {segment_idx} is empty, skipping")
                    continue
                
                # Encode to voxel
                input_voxel = events_to_voxel(
                    segment_events,
                    num_bins=num_bins,
                    sensor_size=sensor_size,
                    fixed_duration_us=segment_duration_us
                )
                
                # Prepare tensor for model (add batch and channel dimensions)
                input_tensor = input_voxel.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 8, H, W)
                
                # Model inference
                model.eval()
                with torch.no_grad():
                    denoised_tensor = model(input_tensor)
                
                # Remove batch and channel dimensions
                denoised_voxel = denoised_tensor.squeeze(0).squeeze(0).cpu()  # (8, H, W)
                
                # Decode back to events
                denoised_events = voxel_to_events(
                    denoised_voxel,
                    total_duration=segment_duration_us,
                    sensor_size=sensor_size
                )
                
                # Adjust timestamps to global timeline
                if len(denoised_events) > 0:
                    t_min_original = events_np[:, 0].min()
                    t_max_original = events_np[:, 0].max()
                    total_duration_original = t_max_original - t_min_original
                    
                    segment_start_global = t_min_original + segment_idx * (total_duration_original / num_segments)
                    
                    # Adjust timestamps
                    denoised_events[:, 0] = denoised_events[:, 0] + segment_start_global - denoised_events[:, 0].min()
                    
                    denoised_segments.append(denoised_events)
                
                self.logger.info(f"Segment {segment_idx + 1}: {len(segment_events)} → {len(denoised_events) if len(denoised_events) > 0 else 0} events")
            
            # Step 4: Merge all segments
            if denoised_segments:
                final_events = np.vstack(denoised_segments)
                # Sort by timestamp
                final_events = final_events[np.argsort(final_events[:, 0])]
            else:
                self.logger.warning("No events produced by denoising pipeline")
                final_events = np.empty((0, 4))
            
            # Step 5: Save output
            self.logger.info(f"Saving {len(final_events)} denoised events to: {output_file}")
            self._save_events_to_h5(final_events, output_file)
            
            # Optional: Visualization
            if config.get('visualization', {}).get('enabled', False):
                self._generate_inference_visualization(
                    events_np, final_events, config['visualization']
                )
            
            self.logger.info("Inference completed successfully!")
            self.logger.info(f"Event count: {len(events_np)} → {len(final_events)}")
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Inference mode error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return 1
    
    def _load_model(self, model_config: dict, device: str):
        """Load trained UNet3D model using training factory"""
        try:
            # Use training factory to create model
            config = {'model': model_config}
            training_factory = TrainingFactory(config)
            model = training_factory.create_model()
            
            # Load checkpoint
            checkpoint = torch.load(model_config['path'], map_location=device)
            
            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict)
            model.to(device)
            
            self.logger.info(f"Model loaded successfully: {model_config.get('name', 'UNet3D')}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _extract_segment_events(self, events_np: np.ndarray, segment_idx: int, num_segments: int) -> np.ndarray:
        """Extract specific segment events from full event array"""
        if len(events_np) == 0:
            return events_np
        
        t_min = events_np[:, 0].min()
        t_max = events_np[:, 0].max()
        total_duration = t_max - t_min
        
        segment_start = t_min + segment_idx * (total_duration / num_segments)
        segment_end = t_min + (segment_idx + 1) * (total_duration / num_segments)
        
        mask = (events_np[:, 0] >= segment_start) & (events_np[:, 0] < segment_end)
        return events_np[mask]
    
    def _save_events_to_h5(self, events_np: np.ndarray, output_path: str):
        """Save events to H5 file in DSEC format"""
        import h5py
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            events_group = f.create_group('events')
            
            if len(events_np) > 0:
                events_group.create_dataset('t', data=events_np[:, 0].astype(np.int64))
                events_group.create_dataset('x', data=events_np[:, 1].astype(np.int16))  
                events_group.create_dataset('y', data=events_np[:, 2].astype(np.int16))
                events_group.create_dataset('p', data=events_np[:, 3].astype(np.int8))
            else:
                # Empty datasets
                events_group.create_dataset('t', data=np.array([], dtype=np.int64))
                events_group.create_dataset('x', data=np.array([], dtype=np.int16))
                events_group.create_dataset('y', data=np.array([], dtype=np.int16))
                events_group.create_dataset('p', data=np.array([], dtype=np.int8))
    
    def _generate_inference_visualization(self, input_events: np.ndarray, output_events: np.ndarray, viz_config: dict):
        """Generate visualization for inference results (optional)"""
        try:
            from src.data_processing.professional_visualizer import visualize_complete_pipeline
            
            output_dir = viz_config.get('output_dir', 'inference_debug')
            segment_idx = viz_config.get('segment_to_visualize', 1)
            
            self.logger.info(f"Generating inference visualization in: {output_dir}")
            
            # Note: This is a simplified version - full visualization would need
            # to reconstruct voxels from events for comparison
            
        except ImportError:
            self.logger.warning("Visualization not available (missing dependencies)")
        except Exception as e:
            self.logger.warning(f"Visualization failed: {e}")
    
    def _trigger_test_debug_visualization(self, batch_idx: int, inputs: torch.Tensor, targets: torch.Tensor, 
                                        batch: dict, debug_dir: str, iteration: int):
        """
        Test模式的Debug可视化钩子 - 输入和目标数据可视化
        复用train模式的可视化代码逻辑
        """
        try:
            from pathlib import Path
            
            # 创建debug输出结构
            iteration_dir = Path(debug_dir) / f"test_iter_{iteration:03d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"🐛 TEST MODE: Generating debug visualizations for Batch {batch_idx}")
            self.logger.info(f"🐛 Output directory: {iteration_dir}")
            
            # 获取第一个样本进行可视化 (batch_size通常为1)
            input_voxel = inputs[0, 0].cpu()   # (8, H, W) - 去除batch和channel维度
            target_voxel = targets[0, 0].cpu() # (8, H, W)
            
            # 从voxel解码回events进行可视化
            from src.data_processing.decode import voxel_to_events
            
            # 解码input和target voxel为events
            input_events_np = voxel_to_events(input_voxel, total_duration=20000, sensor_size=(480, 640))
            target_events_np = voxel_to_events(target_voxel, total_duration=20000, sensor_size=(480, 640))
            
            # 使用专业可视化系统
            from src.data_processing.professional_visualizer import visualize_events, visualize_voxel
            
            # === 输入数据可视化 ===
            # 1. 输入事件3D+2D可视化
            input_events_dir = iteration_dir / "1_input_events"
            input_events_dir.mkdir(exist_ok=True)
            visualize_events(input_events_np, sensor_size=(480, 640), output_dir=str(input_events_dir), 
                           name="test_input_events", num_time_slices=8)
            
            # 3. 输入voxel可视化
            input_voxel_dir = iteration_dir / "3_input_voxel"
            input_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(input_voxel, sensor_size=(480, 640), output_dir=str(input_voxel_dir), 
                          name="test_input_voxel", duration_ms=20)
            
            # === 真值数据可视化 ===
            # 4. 真值事件3D+2D可视化
            target_events_dir = iteration_dir / "4_target_events"
            target_events_dir.mkdir(exist_ok=True)
            visualize_events(target_events_np, sensor_size=(480, 640), output_dir=str(target_events_dir), 
                           name="test_target_events", num_time_slices=8)
            
            # 6. 真值voxel可视化
            target_voxel_dir = iteration_dir / "6_target_voxel"
            target_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(target_voxel, sensor_size=(480, 640), output_dir=str(target_voxel_dir), 
                          name="test_target_voxel", duration_ms=20)
            
            self.logger.info(f"🐛 TEST: Generated input and target visualizations (1,3,4,6/9) in {iteration_dir}")
            
        except Exception as e:
            self.logger.warning(f"🐛 TEST: Debug visualization failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _trigger_test_model_output_visualization(self, batch_idx: int, inputs: torch.Tensor, outputs: torch.Tensor, 
                                               debug_dir: str, iteration: int, model):
        """
        Test模式的Debug可视化钩子 - 模型输出可视化
        复用train模式的可视化代码逻辑，支持真正残差学习
        """
        try:
            from pathlib import Path
            
            iteration_dir = Path(debug_dir) / f"test_iter_{iteration:03d}"
            
            # 获取第一个样本的数据
            input_voxel = inputs[0, 0].cpu()   # (8, H, W)
            output_voxel = outputs[0, 0].cpu()  # (8, H, W) - 这是最终输出
            
            # 检查是否是真正的残差学习模型
            is_true_residual = hasattr(model, 'get_residual') or 'TrueResidualUNet3D' in str(type(model))
            
            if is_true_residual:
                # 对于真正残差学习，计算残差
                residual_voxel = output_voxel - input_voxel  # output = input + residual
                
                self.logger.info(f"🐛 TEST: True residual learning detected:")
                self.logger.info(f"🐛   Input mean: {input_voxel.mean():.4f}, std: {input_voxel.std():.4f}")
                self.logger.info(f"🐛   Residual mean: {residual_voxel.mean():.4f}, std: {residual_voxel.std():.4f}")
                self.logger.info(f"🐛   Output mean: {output_voxel.mean():.4f}, std: {output_voxel.std():.4f}")
                self.logger.info(f"🐛   Identity check: output ≈ input + residual = {torch.allclose(output_voxel, input_voxel + residual_voxel, atol=1e-6)}")
                
                # 残差可视化
                residual_voxel_dir = iteration_dir / "8_residual_voxel"
                residual_voxel_dir.mkdir(exist_ok=True)
                
                from src.data_processing.professional_visualizer import visualize_voxel
                visualize_voxel(residual_voxel, sensor_size=(480, 640), output_dir=str(residual_voxel_dir), 
                              name="test_residual_voxel", duration_ms=20)
                
                # 残差Events可视化 (如果残差有意义)
                if residual_voxel.abs().sum() > 1e-6:
                    from src.data_processing.decode import voxel_to_events
                    residual_events_np = voxel_to_events(residual_voxel, total_duration=20000, sensor_size=(480, 640))
                    
                    residual_events_dir = iteration_dir / "8_residual_events"
                    residual_events_dir.mkdir(exist_ok=True)
                    
                    from src.data_processing.professional_visualizer import visualize_events
                    visualize_events(residual_events_np, sensor_size=(480, 640), output_dir=str(residual_events_dir), 
                                   name="test_residual_events", num_time_slices=8)
            
            # 最终输出可视化 (无论哪种模型)
            from src.data_processing.decode import voxel_to_events
            output_events_np = voxel_to_events(output_voxel, total_duration=20000, sensor_size=(480, 640))
            
            from src.data_processing.professional_visualizer import visualize_events, visualize_voxel
            
            # 7. 最终输出事件可视化
            output_events_dir = iteration_dir / "7_output_events"
            output_events_dir.mkdir(exist_ok=True)
            visualize_events(output_events_np, sensor_size=(480, 640), output_dir=str(output_events_dir), 
                           name="test_output_events", num_time_slices=8)
            
            # 9. 最终输出voxel可视化
            output_voxel_dir = iteration_dir / "9_output_voxel"
            output_voxel_dir.mkdir(exist_ok=True)
            visualize_voxel(output_voxel, sensor_size=(480, 640), output_dir=str(output_voxel_dir), 
                          name="test_output_voxel", duration_ms=20)
            
            folder_count = "8" if is_true_residual else "6"
            self.logger.info(f"🐛 TEST: Generated final output visualizations (7,9/{folder_count}) in {iteration_dir}")
            if is_true_residual:
                self.logger.info(f"🐛 TEST: Generated residual visualizations (8/8) in {iteration_dir}")
                self.logger.info(f"🐛 TEST: All debug visualizations completed! (8 folders total: 1,3,4,6,7,8,9)")
            else:
                self.logger.info(f"🐛 TEST: All debug visualizations completed! (6 folders total: 1,3,4,6,7,9)")
            
            # 生成比较总结
            self._generate_test_debug_summary(iteration_dir, batch_idx, iteration, is_true_residual)
            
        except Exception as e:
            self.logger.warning(f"🐛 TEST: Model output visualization failed: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _generate_test_debug_summary(self, iteration_dir: Path, batch_idx: int, iteration: int, is_true_residual: bool = False):
        """生成test模式的debug总结信息"""
        try:
            summary_file = iteration_dir / "debug_summary.txt"
            
            with open(summary_file, 'w') as f:
                f.write(f"TEST Mode Debug Visualization Summary\n")
                f.write(f"====================================\n")
                f.write(f"Test Iteration: {iteration}\n")
                f.write(f"Batch: {batch_idx}\n")
                f.write(f"Mode: Testing (No Training)\n")
                f.write(f"True Residual Learning: {'Yes' if is_true_residual else 'No'}\n")
                
                if is_true_residual:
                    f.write(f"\n8 Visualization Folders (True Residual Learning):\n")
                    f.write(f"1. 1_input_events/          - Input events (3D+2D+temporal) comprehensive\n")
                    f.write(f"2. 3_input_voxel/           - Input voxel temporal bins\n")
                    f.write(f"3. 4_target_events/         - Target events (3D+2D+temporal) comprehensive\n") 
                    f.write(f"4. 6_target_voxel/          - Target voxel temporal bins\n")
                    f.write(f"5. 7_output_events/         - Model output events (input + residual) comprehensive\n")
                    f.write(f"6. 8_residual_voxel/        - Learned residual voxel temporal bins\n")
                    f.write(f"7. 8_residual_events/       - Learned residual events (if non-zero)\n")
                    f.write(f"8. 9_output_voxel/          - Model output voxel temporal bins\n")
                else:
                    f.write(f"\n6 Visualization Folders (Standard Model):\n")
                    f.write(f"1. 1_input_events/          - Input events (3D+2D+temporal) comprehensive\n")
                    f.write(f"2. 3_input_voxel/           - Input voxel temporal bins\n")
                    f.write(f"3. 4_target_events/         - Target events (3D+2D+temporal) comprehensive\n") 
                    f.write(f"4. 6_target_voxel/          - Target voxel temporal bins\n")
                    f.write(f"5. 7_output_events/         - Model output events (3D+2D+temporal) comprehensive\n")
                    f.write(f"6. 9_output_voxel/          - Model output voxel temporal bins\n")
                
                f.write(f"\nTesting Mode: Evaluate checkpoint performance on test data\n")
                f.write(f"No training - only inference and visualization\n")
                f.write(f"Same data processing pipeline as training validation\n")
                
            self.logger.info(f"🐛 TEST: Debug summary saved to {summary_file}")
            
        except Exception as e:
            self.logger.warning(f"🐛 TEST: Failed to generate debug summary: {e}")


def main():
    """Main entry point with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description='Event-Voxel Denoising with pytorch-3dunet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Training
  python main.py train --config configs/train_config.yaml
  
  # Testing  
  python main.py test --config configs/test_config.yaml
  
  # Inference
  python main.py inference --config configs/inference_config.yaml \\
    --input testdata/noisy_events.h5 --output results/denoised_events.h5
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Training mode
    train_parser = subparsers.add_parser('train', help='Train denoising model')
    train_parser.add_argument('--config', required=True, 
                            help='Path to training configuration YAML')
    train_parser.add_argument('--debug', action='store_true',
                            help='Enable debug mode with detailed visualization (runs only 1-2 iterations)')
    train_parser.add_argument('--debug-dir', type=str, default='debug_output',
                            help='Directory for debug visualizations (default: debug_output)')
    
    # Testing mode
    test_parser = subparsers.add_parser('test', help='Evaluate trained model')
    test_parser.add_argument('--config', required=True,
                           help='Path to test configuration YAML')
    test_parser.add_argument('--debug', action='store_true',
                           help='Enable debug mode with detailed visualization (runs only 1-2 batches)')
    test_parser.add_argument('--debug-dir', type=str, default='debug_output',
                           help='Directory for debug visualizations (default: debug_output)')
    
    # Inference mode
    inference_parser = subparsers.add_parser('inference', help='Denoise event file')
    inference_parser.add_argument('--config', required=True,
                                help='Path to inference configuration YAML')
    inference_parser.add_argument('--input', required=True,
                                help='Input H5 events file')
    inference_parser.add_argument('--output', required=True,
                                help='Output H5 events file')
    
    args = parser.parse_args()
    
    if not args.mode:
        parser.print_help()
        return 1
    
    # Initialize pipeline
    pipeline = EventVoxelPipeline()
    
    # Execute requested mode
    if args.mode == 'train':
        return pipeline.train_mode(args.config, getattr(args, 'debug', False), getattr(args, 'debug_dir', 'debug_output'))
    elif args.mode == 'test':
        return pipeline.test_mode(args.config, getattr(args, 'debug', False), getattr(args, 'debug_dir', 'debug_output'))
    elif args.mode == 'inference':
        return pipeline.inference_mode(args.config, args.input, args.output)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == '__main__':
    sys.exit(main())