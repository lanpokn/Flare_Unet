#!/usr/bin/env python3
"""
inference_single.py - Single File Event Deflare Processing
===========================================================

Processes individual H5 event files with variable length using 3D UNet models.
Handles 20ms segmentation automatically, supports both normal and simple model weights.

Usage:
    python inference_single.py --input path/to/input.h5 [--config configs/inference_config.yaml] [--mode simple] [--debug]

Author: Event-Voxel Deflare System
Date: 2025-12-24
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, List, Optional

# Add project paths
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

# Import project modules
from src.utils.config_loader import ConfigLoader
from src.data_processing.encode import load_h5_events, events_to_voxel
from src.data_processing.decode import voxel_to_events
from src.training.training_factory import TrainingFactory


class InferenceSingle:
    """Single file inference processor for event deflare"""
    
    def __init__(self, config_path: str, mode: str = 'normal', device: str = 'cuda'):
        """
        Initialize single file inference processor
        
        Args:
            config_path: Path to inference config YAML
            mode: 'normal' or 'simple' for different model weights
            device: Processing device ('cuda' or 'cpu')
        """
        self.device = device
        self.mode = mode
        
        # Load configuration
        if mode == 'simple':
            # Use simple config if available, otherwise modify normal config
            simple_config_path = config_path.replace('.yaml', '_simple.yaml')
            if Path(simple_config_path).exists():
                config_path = simple_config_path
        
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_config(config_path)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Model will be loaded on first use
        self.model = None
        
        self.logger.info(f"🚀 InferenceSingle initialized - Mode: {mode}, Device: {device}")
    
    def _convert_windows_to_wsl_path(self, windows_path: str) -> str:
        """
        Convert Windows path to WSL path format
        
        Args:
            windows_path: Windows format path (e.g., E:\folder\file.h5)
            
        Returns:
            WSL format path (e.g., /mnt/e/folder/file.h5)
        """
        if not windows_path.startswith(('C:', 'D:', 'E:', 'F:', 'G:')):
            return windows_path  # Already Unix-style path
        
        # Convert E:\path\to\file -> /mnt/e/path/to/file
        drive_letter = windows_path[0].lower()
        rest_path = windows_path[2:].replace('\\', '/')
        wsl_path = f'/mnt/{drive_letter}{rest_path}'
        
        self.logger.info(f"🔄 Path conversion: {windows_path} -> {wsl_path}")
        return wsl_path
    
    def _load_model(self):
        """Load model from checkpoint if not already loaded"""
        if self.model is not None:
            return
        
        self.logger.info("🔧 Loading model...")
        
        # Create model using training factory
        training_factory = TrainingFactory(self.config)
        self.model = training_factory.create_model()
        
        # Load checkpoint
        checkpoint_path = self.config['model']['path']
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"📦 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"✅ Model loaded - Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    def _load_and_validate_events(self, file_path: str) -> np.ndarray:
        """
        Load events from H5 file and validate format
        
        Args:
            file_path: Path to H5 file
            
        Returns:
            events_np: NumPy array (N, 4) with [t, x, y, p] in correct format
        """
        self.logger.info(f"📂 Loading events from: {file_path}")
        
        # Use existing load function which handles polarity conversion automatically
        events_np = load_h5_events(file_path)
        
        # Extract individual components for validation
        t, x, y, p = events_np[:, 0], events_np[:, 1], events_np[:, 2], events_np[:, 3]
        
        # Validate ranges
        duration_ms = (t.max() - t.min()) / 1000
        self.logger.info(f"📊 Events loaded: {len(events_np):,} events, {duration_ms:.1f}ms duration")
        self.logger.info(f"    Spatial range: x=[{x.min():.0f},{x.max():.0f}], y=[{y.min():.0f},{y.max():.0f}]")
        self.logger.info(f"    Polarities: {np.unique(p)} (pos:{(p==1).sum():,}, neg:{(p==-1).sum():,})")
        
        # Validate coordinate ranges
        sensor_size = self.config['inference']['sensor_size']  # [H, W] = [480, 640]
        if x.max() >= sensor_size[1] or y.max() >= sensor_size[0]:
            self.logger.warning(f"⚠️  Events exceed sensor size {sensor_size}: x_max={x.max()}, y_max={y.max()}")
        
        return events_np
    
    def _extract_segment_events(self, events_np: np.ndarray, segment_idx: int, 
                               segment_duration_us: int = 20000) -> np.ndarray:
        """
        Extract events for a specific 20ms segment
        
        Args:
            events_np: Full events array (N, 4)
            segment_idx: Segment index (0, 1, 2, ...)
            segment_duration_us: Duration per segment in microseconds
            
        Returns:
            segment_events: Events in this segment (M, 4) with normalized timestamps
        """
        if len(events_np) == 0:
            return events_np
            
        t_min = events_np[:, 0].min()
        segment_start = t_min + segment_idx * segment_duration_us
        segment_end = t_min + (segment_idx + 1) * segment_duration_us
        
        # Filter events in this time window
        mask = (events_np[:, 0] >= segment_start) & (events_np[:, 0] < segment_end)
        segment_events = events_np[mask].copy()
        
        if len(segment_events) > 0:
            # Normalize timestamps to start from 0 for encoding
            segment_events[:, 0] -= segment_start
        
        self.logger.debug(f"  Segment {segment_idx}: {len(segment_events):,} events in [{segment_start:.0f}, {segment_end:.0f})")
        return segment_events
    
    def _process_segment(self, segment_events: np.ndarray, segment_idx: int) -> np.ndarray:
        """
        Process a single 20ms segment through the UNet model
        
        Args:
            segment_events: Events in this segment (M, 4)
            segment_idx: Segment index for logging
            
        Returns:
            output_events: Processed events (N, 4)
        """
        sensor_size = self.config['inference']['sensor_size']  # [480, 640]
        num_bins = self.config['inference']['num_bins']        # 8
        segment_duration_us = self.config['inference']['segment_duration_us']  # 20000
        
        # Encode to voxel (following main.py inference logic)
        if len(segment_events) == 0:
            # Empty segment - create zero voxel
            input_voxel = torch.zeros((num_bins, sensor_size[0], sensor_size[1]), 
                                     dtype=torch.float32)
            self.logger.debug(f"  Segment {segment_idx}: Empty segment, using zero voxel")
        else:
            input_voxel = events_to_voxel(
                segment_events, 
                num_bins=num_bins,
                sensor_size=sensor_size,
                fixed_duration_us=segment_duration_us
            )
            # events_to_voxel returns torch.Tensor, not numpy
        
        # Process through model (following main.py format)
        input_tensor = input_voxel.unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, 8, H, W)
        
        with torch.no_grad():
            output_tensor = self.model(input_tensor)  # (1, 1, 8, H, W)
        
        # Remove batch and channel dimensions
        output_voxel = output_tensor.squeeze(0).squeeze(0).cpu()  # (8, H, W)
        
        # Decode back to events
        output_events = voxel_to_events(
            output_voxel, 
            total_duration=segment_duration_us,
            sensor_size=sensor_size
        )
        
        input_event_count = len(segment_events)
        output_event_count = len(output_events)
        compression_ratio = output_event_count / input_event_count if input_event_count > 0 else 0
        
        self.logger.debug(f"  Segment {segment_idx}: {input_event_count:,} -> {output_event_count:,} events ({compression_ratio:.2%})")
        
        return output_events
    
    def _merge_segments(self, segment_results: List[Tuple[int, np.ndarray]], 
                       original_events: np.ndarray) -> np.ndarray:
        """
        Merge processed segments back to full timeline
        
        Args:
            segment_results: List of (segment_idx, processed_events) tuples
            original_events: Original events for timestamp reference
            
        Returns:
            merged_events: Final merged events (N, 4)
        """
        if not segment_results:
            self.logger.warning("⚠️  No segments to merge")
            return np.empty((0, 4))
        
        t_min_original = original_events[:, 0].min()
        segment_duration_us = self.config['inference']['segment_duration_us']
        
        all_events = []
        
        for segment_idx, segment_events in segment_results:
            if len(segment_events) == 0:
                continue
            
            # Restore global timestamps
            segment_events_copy = segment_events.copy()
            segment_start_global = t_min_original + segment_idx * segment_duration_us
            segment_events_copy[:, 0] += segment_start_global
            
            all_events.append(segment_events_copy)
        
        if not all_events:
            self.logger.warning("⚠️  All segments are empty")
            return np.empty((0, 4))
        
        # Concatenate all segments
        merged_events = np.vstack(all_events)
        
        # Sort by timestamp
        sort_indices = np.argsort(merged_events[:, 0])
        merged_events = merged_events[sort_indices]
        
        self.logger.info(f"🔗 Merged {len(segment_results)} segments: {len(merged_events):,} total events")
        return merged_events
    
    def _save_events_to_h5(self, events_np: np.ndarray, output_path: str):
        """
        Save processed events to H5 file in standard format
        
        Args:
            events_np: Events array (N, 4)
            output_path: Output file path
        """
        self.logger.info(f"💾 Saving {len(events_np):,} events to: {output_path}")
        
        with h5py.File(output_path, 'w') as f:
            events_group = f.create_group('events')
            
            # Use proper data types matching project standard
            events_group.create_dataset('t', data=events_np[:, 0].astype(np.int64), 
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('x', data=events_np[:, 1].astype(np.uint16), 
                                      compression='gzip', compression_opts=9) 
            events_group.create_dataset('y', data=events_np[:, 2].astype(np.uint16),
                                      compression='gzip', compression_opts=9)
            events_group.create_dataset('p', data=events_np[:, 3].astype(np.int8),
                                      compression='gzip', compression_opts=9)
        
        self.logger.info(f"✅ File saved successfully")
    
    def process_file(self, input_path: str, output_path: Optional[str] = None, debug: bool = False) -> str:
        """
        Process a single H5 file with variable length
        
        Args:
            input_path: Input H5 file path (Windows or WSL format)
            output_path: Output file path (optional, auto-generated if None)
            debug: Enable debug visualization
            
        Returns:
            output_path: Final output file path
        """
        # Convert Windows path to WSL if needed
        input_path = self._convert_windows_to_wsl_path(input_path)
        
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Generate output path if not provided
        if output_path is None:
            input_path_obj = Path(input_path)
            suffix = 'Unetsimple' if self.mode == 'simple' else 'Unet'
            output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_{suffix}.h5")
        else:
            output_path = self._convert_windows_to_wsl_path(output_path)
        
        self.logger.info(f"🎯 Processing: {Path(input_path).name} -> {Path(output_path).name}")
        
        # Load model
        self._load_model()
        
        # Load and validate events
        events_np = self._load_and_validate_events(input_path)
        
        # Calculate number of segments needed
        duration_us = events_np[:, 0].max() - events_np[:, 0].min()
        segment_duration_us = self.config['inference']['segment_duration_us']
        num_segments = int(np.ceil(duration_us / segment_duration_us))
        
        self.logger.info(f"🔢 Processing {num_segments} segments of {segment_duration_us/1000}ms each")
        
        # Process each segment
        segment_results = []
        for segment_idx in range(num_segments):
            self.logger.info(f"⚙️  Processing segment {segment_idx + 1}/{num_segments}")
            
            # Extract segment events
            segment_events = self._extract_segment_events(events_np, segment_idx, segment_duration_us)
            
            # Process through model
            processed_events = self._process_segment(segment_events, segment_idx)
            
            segment_results.append((segment_idx, processed_events))
        
        # Merge all segments
        final_events = self._merge_segments(segment_results, events_np)
        
        # Calculate final statistics
        input_count = len(events_np)
        output_count = len(final_events)
        compression_ratio = output_count / input_count if input_count > 0 else 0
        
        self.logger.info(f"📊 Final result: {input_count:,} -> {output_count:,} events ({compression_ratio:.2%} compression)")
        
        # Save results
        self._save_events_to_h5(final_events, output_path)
        
        # Debug visualization if requested
        if debug:
            self._generate_debug_visualization(events_np, final_events, input_path, output_path)
        
        return output_path
    
    def _generate_debug_visualization(self, input_events: np.ndarray, output_events: np.ndarray,
                                     input_path: str, output_path: str):
        """Generate debug visualization comparing input and output"""
        try:
            debug_dir = Path("debug_output") / f"inference_single_{Path(input_path).stem}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            from src.data_processing.professional_visualizer import visualize_events
            sensor_size = self.config['inference']['sensor_size']
            
            # Input events visualization
            input_dir = debug_dir / "1_input_events"
            input_dir.mkdir(exist_ok=True)
            visualize_events(input_events, sensor_size, str(input_dir), 
                           name="input_events", num_time_slices=8)
            
            # Output events visualization  
            output_dir = debug_dir / "2_output_events"
            output_dir.mkdir(exist_ok=True)
            visualize_events(output_events, sensor_size, str(output_dir),
                           name="output_events", num_time_slices=8)
            
            # Summary statistics
            with open(debug_dir / "debug_summary.txt", 'w') as f:
                f.write(f"Inference Single Debug Summary\n")
                f.write(f"{'='*40}\n")
                f.write(f"Input file: {input_path}\n")
                f.write(f"Output file: {output_path}\n")
                f.write(f"Mode: {self.mode}\n")
                f.write(f"Input events: {len(input_events):,}\n")
                f.write(f"Output events: {len(output_events):,}\n")
                f.write(f"Compression ratio: {len(output_events)/len(input_events):.2%}\n")
            
            self.logger.info(f"🎨 Debug visualization saved to: {debug_dir}")
            
        except Exception as e:
            self.logger.warning(f"⚠️  Debug visualization failed: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Single File Event Deflare Processing")
    parser.add_argument('--input', required=True, help='Input H5 file path')
    parser.add_argument('--output', help='Output H5 file path (optional)')
    parser.add_argument('--config', default='configs/inference_config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['normal', 'simple'], default='normal', 
                       help='Model mode: normal or simple weights')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualization')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'], help='Processing device')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = InferenceSingle(
            config_path=args.config,
            mode=args.mode,
            device=args.device
        )
        
        # Process file
        output_path = processor.process_file(
            input_path=args.input,
            output_path=args.output,
            debug=args.debug
        )
        
        print(f"✅ Processing completed successfully!")
        print(f"📂 Output file: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()