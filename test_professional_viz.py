#!/usr/bin/env python3
"""
Test script for professional visualizer
Tests three scenarios:
1. Original events from H5 file
2. Voxel encoded from events  
3. Events decoded from voxel
"""

import sys
import os
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.professional_visualizer import ProfessionalEventVisualizer
from data_processing.encode import events_to_voxel, load_h5_events
from data_processing.decode import voxel_to_events

def test_complete_pipeline():
    """Test the complete pipeline with professional visualization"""
    
    testdata_path = "testdata/light_source_sequence_00018.h5"
    sensor_size = (480, 640)
    
    print("=== Testing Professional Visualizer Pipeline ===")
    
    # Create visualizer
    viz = ProfessionalEventVisualizer("debug_output", dpi=120)
    
    # 1. Load original events and visualize
    print("\n1. Loading and visualizing original events...")
    original_events = load_h5_events(testdata_path)
    print(f"Loaded {len(original_events):,} original events")
    
    viz.visualize_events_comprehensive(
        original_events, 
        sensor_size, 
        name_prefix="01_original",
        num_time_slices=32
    )
    
    # 2. Encode to voxel and visualize
    print("\n2. Encoding to voxel and visualizing...")
    voxel = events_to_voxel(original_events, num_bins=16, sensor_size=sensor_size)
    print(f"Created voxel with shape: {voxel.shape}, sum: {voxel.sum()}")
    
    viz.visualize_voxel_comprehensive(
        voxel,
        sensor_size,
        name_prefix="02_encoded_voxel", 
        duration_ms=100
    )
    
    # 3. Decode back to events and visualize
    print("\n3. Decoding voxel to events and visualizing...")
    total_duration = int(original_events[:, 0].max() - original_events[:, 0].min())
    decoded_events = voxel_to_events(voxel, total_duration, sensor_size)
    print(f"Decoded {len(decoded_events):,} events")
    
    viz.visualize_events_comprehensive(
        decoded_events,
        sensor_size,
        name_prefix="03_decoded",
        num_time_slices=32
    )
    
    # 4. Create sliding window analysis for comparison
    print("\n4. Creating sliding window analysis...")
    viz.create_sliding_window_video(
        original_events,
        sensor_size,
        name_prefix="04_original_sliding",
        window_ms=6.25,
        overlap_ms=3.125
    )
    
    viz.create_sliding_window_video(
        decoded_events, 
        sensor_size,
        name_prefix="05_decoded_sliding",
        window_ms=6.25,
        overlap_ms=3.125
    )
    
    # 5. Summary
    print("\n5. Generating final comparison...")
    # Re-encode decoded events for comparison
    reconstructed_voxel = events_to_voxel(decoded_events, num_bins=16, sensor_size=sensor_size)
    
    viz.visualize_voxel_comprehensive(
        reconstructed_voxel,
        sensor_size, 
        name_prefix="06_reconstructed_voxel",
        duration_ms=100
    )
    
    # Print comparison statistics
    print(f"\n=== Pipeline Statistics ===")
    print(f"Original events: {len(original_events):,}")
    print(f"Decoded events: {len(decoded_events):,}")
    print(f"Original voxel sum: {voxel.sum():.0f}")
    print(f"Reconstructed voxel sum: {reconstructed_voxel.sum():.0f}")
    print(f"Voxel difference (L1): {torch.abs(voxel - reconstructed_voxel).mean():.6f}")
    print(f"Voxel difference (L2): {torch.sqrt(torch.pow(voxel - reconstructed_voxel, 2).mean()):.6f}")
    
    viz.print_summary("complete_pipeline")
    
    return viz

if __name__ == "__main__":
    try:
        viz = test_complete_pipeline()
        print("\n✅ Professional visualization testing completed successfully!")
        print(f"Check output in: {viz.output_dir}")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()