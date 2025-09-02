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
    
    def train_mode(self, config_path: str) -> int:
        """
        Training mode - uses custom training system with pytorch-3dunet UNet3D model
        
        Args:
            config_path: Path to training configuration YAML
            
        Returns:
            Exit code (0 = success, non-zero = error)
        """
        self.logger.info("=== EVENT-VOXEL CUSTOM TRAINING MODE ===")
        
        try:
            # Load and validate training configuration
            config = self.config_loader.load_train_config(config_path)
            self.logger.info(f"Loaded training config from: {config_path}")
            
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
    
    def test_mode(self, config_path: str) -> int:
        """
        Testing mode - evaluates trained model on test dataset
        
        Args:
            config_path: Path to test configuration YAML
            
        Returns:
            Exit code (0 = success, non-zero = error)
        """
        self.logger.info("=== EVENT-VOXEL TESTING MODE ===")
        
        try:
            # Load and validate test configuration  
            config = self.config_loader.load_test_config(config_path)
            self.logger.info(f"Loaded test config from: {config_path}")
            
            # Verify model checkpoint exists
            model_path = config['model']['path']
            if not Path(model_path).exists():
                self.logger.error(f"Model checkpoint not found: {model_path}")
                return 1
            
            self.logger.info("Starting custom evaluation...")
            self.logger.info(f"Model: {model_path}")
            
            # 创建训练工厂（仅用于创建模型和数据集）
            training_factory = TrainingFactory(config)
            
            # 创建模型和数据集
            model = training_factory.create_model()
            _, test_dataset = training_factory.create_datasets()  # 使用val作为test
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
            
            self.logger.info(f"Evaluating on {len(test_dataset)} samples...")
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    inputs = batch['raw'].to(device)
                    targets = batch['label'].to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 10 == 0:
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
    
    # Testing mode
    test_parser = subparsers.add_parser('test', help='Evaluate trained model')
    test_parser.add_argument('--config', required=True,
                           help='Path to test configuration YAML')
    
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
        return pipeline.train_mode(args.config)
    elif args.mode == 'test':
        return pipeline.test_mode(args.config)
    elif args.mode == 'inference':
        return pipeline.inference_mode(args.config, args.input, args.output)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == '__main__':
    sys.exit(main())