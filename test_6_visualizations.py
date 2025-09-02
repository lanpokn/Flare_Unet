#!/usr/bin/env python3
"""
Test script for complete 6-visualization pipeline
Tests the 2×2+2 visualization architecture:
- Input Events: 3D + 2D (red/blue)
- Output Events: 3D + 2D (red/blue)  
- Input Voxel + Output Voxel (re-encoded)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'data_processing'))

from encode import load_h5_events, events_to_voxel
from decode import voxel_to_events
from professional_visualizer import visualize_complete_pipeline

def main():
    print("=== Complete 6-Visualization Pipeline Test ===")
    
    # Step 1: Load original events
    input_file = "testdata/composed_00003_bg_flare.h5"
    print(f"Loading input events from: {input_file}")
    input_events = load_h5_events(input_file)
    print(f"Loaded {len(input_events):,} input events")
    
    # Step 2: Encode to voxel
    sensor_size = (480, 640)
    num_bins = 32  # Use 32 bins for consistency
    print(f"Encoding to voxel: {num_bins} bins")
    input_voxel = events_to_voxel(input_events, num_bins=num_bins, sensor_size=sensor_size)
    print(f"Input voxel shape: {input_voxel.shape}, sum: {input_voxel.sum():.0f}")
    
    # Step 3: Decode back to events
    print("Decoding voxel back to events...")
    output_events = voxel_to_events(input_voxel, total_duration=100000, sensor_size=sensor_size)
    print(f"Generated {len(output_events):,} output events")
    
    # Step 4: Re-encode output events to voxel
    print("Re-encoding output events to voxel...")
    output_voxel = events_to_voxel(output_events, num_bins=num_bins, sensor_size=sensor_size)
    print(f"Output voxel shape: {output_voxel.shape}, sum: {output_voxel.sum():.0f}")
    
    # Step 5: Verify consistency
    import torch
    l1_diff = torch.sum(torch.abs(input_voxel - output_voxel)).item()
    l2_diff = torch.sqrt(torch.sum((input_voxel - output_voxel)**2)).item()
    print(f"Pipeline consistency: L1={l1_diff:.6f}, L2={l2_diff:.6f}")
    
    # Step 6: Generate complete 6-visualization pipeline with memory optimization
    print("\n=== Generating 6 Visualization Results (Memory Optimized) ===")
    print("Using segment-based approach: 100ms → 5×20ms, visualizing segment 1 (10-30ms)")
    output_dir = "debug_output"  # Unified output directory
    
    visualize_complete_pipeline(
        input_events=input_events,
        input_voxel=input_voxel,
        output_events=output_events,
        output_voxel=output_voxel,
        sensor_size=sensor_size,
        output_dir=output_dir,
        segment_idx=1  # Visualize segment 1: 10-30ms
    )
    
    print("\n=== Test Complete ===")
    print(f"All visualizations saved to: {output_dir}/")
    
    # Count generated files
    import os
    total_files = 0
    for root, dirs, files in os.walk(output_dir):
        total_files += len(files)
    print(f"Total files generated: {total_files}")

if __name__ == "__main__":
    main()