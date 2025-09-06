#!/usr/bin/env python3
"""
Debug script to understand checkpoint saving frequency mismatch
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.utils.config_loader import ConfigLoader
import yaml

def debug_checkpoint_frequency():
    """Debug the checkpoint frequency configuration"""
    
    print("=== Checkpoint Frequency Debug ===\n")
    
    # 1. 直接读取YAML文件
    print("1. Direct YAML file reading:")
    with open('configs/train_config.yaml', 'r') as f:
        raw_config = yaml.safe_load(f)
    print(f"   validate_after_iters: {raw_config['trainer']['validate_after_iters']}")
    print(f"   log_after_iters: {raw_config['trainer']['log_after_iters']}")
    
    # 2. 通过ConfigLoader加载
    print("\n2. ConfigLoader processing:")
    config_loader = ConfigLoader()
    processed_config = config_loader.load_train_config('configs/train_config.yaml')
    print(f"   validate_after_iters: {processed_config['trainer']['validate_after_iters']}")
    print(f"   log_after_iters: {processed_config['trainer']['log_after_iters']}")
    
    # 3. 检查数据集大小
    print("\n3. Dataset size calculation:")
    train_dir = Path('data_simu/physics_method/background_with_flare_events')
    if train_dir.exists():
        train_files = list(train_dir.glob('*.h5'))
        total_samples = len(train_files) * 5
        print(f"   Training files: {len(train_files)}")
        print(f"   Segments per file: 5")
        print(f"   Total training samples: {total_samples}")
        
        # 计算预期的checkpoint保存点
        validate_freq = processed_config['trainer']['validate_after_iters']
        expected_checkpoints = []
        for i in range(validate_freq, total_samples + 1, validate_freq):
            expected_checkpoints.append(i)
        print(f"   Expected checkpoint iterations: {expected_checkpoints}")
    else:
        print(f"   Training directory not found: {train_dir}")
    
    # 4. 分析实际的checkpoint文件
    print("\n4. Actual checkpoint analysis:")
    checkpoint_dirs = [
        'checkpoints/event_voxel_deflare4',  # Most recent
        'checkpoints/event_voxel_deflare_depth3',  # Earlier
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_path = Path(checkpoint_dir)
        if checkpoint_path.exists():
            print(f"\n   {checkpoint_dir}:")
            checkpoint_files = list(checkpoint_path.glob('checkpoint_*.pth'))
            if checkpoint_files:
                # Extract iteration numbers and sort
                iterations = []
                for file in checkpoint_files:
                    if 'iter_' in file.name:
                        iter_part = file.name.split('iter_')[1].split('.')[0]
                        iterations.append(int(iter_part))
                
                iterations.sort()
                print(f"     Checkpoint iterations: {iterations[:10]}{'...' if len(iterations) > 10 else ''}")
                
                if len(iterations) >= 2:
                    # Calculate intervals
                    intervals = [iterations[i] - iterations[i-1] for i in range(1, min(len(iterations), 6))]
                    print(f"     First 5 intervals: {intervals}")
                    
                    # Look for patterns
                    if all(i == 250 for i in intervals):
                        print(f"     Pattern: Every 250 iterations ❌ (Expected every {validate_freq})")
                    elif all(i == 1250 for i in intervals):
                        print(f"     Pattern: Every 1250 iterations ✅")
                    else:
                        print(f"     Pattern: Mixed intervals")
            else:
                print(f"     No checkpoint files found")
        else:
            print(f"   Directory not found: {checkpoint_path}")
    
    # 5. 检查可能的原因
    print("\n5. Potential causes analysis:")
    
    # Check for debug mode remnants
    if 'debug' in processed_config and processed_config['debug'].get('enabled', False):
        print("   ❌ DEBUG mode is enabled - this could affect frequencies")
    else:
        print("   ✅ DEBUG mode not enabled in config")
    
    # Check for multiple save points in code
    print("\n   Code analysis needed:")
    print("   - Check trainer.py line 256: validate_after_iters default value")
    print("   - Check trainer.py line 448: epoch-end checkpoint saving")
    print("   - Check if any debug code overrides the frequency")
    
    print("\n=== Investigation Complete ===")

if __name__ == "__main__":
    debug_checkpoint_frequency()