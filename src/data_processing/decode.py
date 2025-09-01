import numpy as np
import torch
import h5py
import yaml
import argparse
import os
import sys
from pathlib import Path

# Add event_utils-master to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'event_utils-master'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'event_utils-master', 'lib'))

def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path is None:
        # Get the project root directory (2 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "voxel_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_voxel(voxel_path):
    """Load voxel grid from file
    
    Args:
        voxel_path: Path to voxel file (.pt or .npy)
        
    Returns:
        voxel: PyTorch tensor (B, H, W)
    """
    if voxel_path.endswith('.pt'):
        voxel = torch.load(voxel_path)
    elif voxel_path.endswith('.npy'):
        voxel = torch.from_numpy(np.load(voxel_path))
    else:
        raise ValueError(f"Unsupported voxel file format: {voxel_path}")
    
    return voxel

def voxel_to_events(voxel, total_duration=100000, sensor_size=(480, 640), random_seed=None):
    """Convert voxel grid back to events using uniform random distribution
    
    Args:
        voxel: PyTorch tensor (B, H, W) - voxel grid
        total_duration: Total time duration in microseconds
        sensor_size: Sensor resolution (H, W)
        random_seed: Random seed for reproducibility
        
    Returns:
        events_np: NumPy array (N, 4) with [t, x, y, p]
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    B, H, W = voxel.shape
    bin_duration = total_duration / B  # Duration of each temporal bin
    
    events_list = []
    
    for bin_idx in range(B):
        bin_voxel = voxel[bin_idx]  # (H, W)
        
        # Round to integers to get event counts
        event_counts = torch.round(bin_voxel).int()
        
        # Find all non-zero pixels
        nonzero_indices = torch.nonzero(event_counts, as_tuple=False)  # (N_pixels, 2)
        
        for pixel_idx in range(len(nonzero_indices)):
            y, x = nonzero_indices[pixel_idx].tolist()
            n = event_counts[y, x].item()
            
            if n == 0:
                continue
                
            # Generate abs(n) events
            num_events = abs(n)
            polarity = 1 if n > 0 else -1
            
            # Generate random timestamps within this bin
            bin_start = bin_idx * bin_duration
            bin_end = (bin_idx + 1) * bin_duration
            timestamps = np.random.uniform(bin_start, bin_end, num_events)
            
            # Create events for this pixel
            for i in range(num_events):
                event = [timestamps[i], x, y, polarity]
                events_list.append(event)
    
    if len(events_list) == 0:
        return np.empty((0, 4))
    
    # Convert to numpy array and sort by timestamp
    events_np = np.array(events_list)
    sort_indices = np.argsort(events_np[:, 0])
    events_np = events_np[sort_indices]
    
    return events_np

def save_h5_events(events_np, output_path):
    """Save events to H5 file in DSEC format
    
    Args:
        events_np: NumPy array (N, 4) with [t, x, y, p]
        output_path: Output H5 file path
    """
    with h5py.File(output_path, 'w') as f:
        events_group = f.create_group('events')
        
        if len(events_np) > 0:
            # Convert data types to match DSEC format
            t = events_np[:, 0].astype(np.int64)    # timestamps (μs)
            x = events_np[:, 1].astype(np.uint16)   # x coordinates
            y = events_np[:, 2].astype(np.uint16)   # y coordinates  
            p = events_np[:, 3].astype(np.int8)     # polarities
            
            events_group.create_dataset('t', data=t)
            events_group.create_dataset('x', data=x)
            events_group.create_dataset('y', data=y)
            events_group.create_dataset('p', data=p)
        else:
            # Create empty datasets
            events_group.create_dataset('t', data=np.array([], dtype=np.int64))
            events_group.create_dataset('x', data=np.array([], dtype=np.uint16))
            events_group.create_dataset('y', data=np.array([], dtype=np.uint16))
            events_group.create_dataset('p', data=np.array([], dtype=np.int8))


def main():
    parser = argparse.ArgumentParser(description='Convert voxel grid back to events')
    parser.add_argument('--input_voxel_file', required=True, help='Input voxel file path (.pt or .npy)')
    parser.add_argument('--output_file', required=True, help='Output H5 events file path')
    parser.add_argument('--config', help='Config YAML file path')
    parser.add_argument('--total_duration', type=float, help='Total time duration in microseconds (overrides config)')
    parser.add_argument('--sensor_size', nargs=2, type=int, help='Sensor size H W (overrides config)')
    parser.add_argument('--random_seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with visualizations')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    total_duration = args.total_duration if args.total_duration else config['total_duration_ms'] * 1000  # Convert ms to μs
    sensor_size = tuple(args.sensor_size) if args.sensor_size else (config['sensor_size']['height'], config['sensor_size']['width'])
    random_seed = args.random_seed if args.random_seed is not None else config['decoding'].get('random_seed')
    
    print(f"Loading voxel grid from: {args.input_voxel_file}")
    voxel = load_voxel(args.input_voxel_file)
    print(f"Voxel shape: {voxel.shape}")
    print(f"Voxel range: {voxel.min():.3f} - {voxel.max():.3f}")
    print(f"Non-zero voxels: {torch.count_nonzero(voxel)}")
    
    print(f"Converting to events: {total_duration:.0f}μs duration, {sensor_size} resolution")
    if random_seed is not None:
        print(f"Using random seed: {random_seed}")
    
    events_np = voxel_to_events(voxel, total_duration=total_duration, sensor_size=sensor_size, random_seed=random_seed)
    print(f"Generated {len(events_np)} events")
    
    if len(events_np) > 0:
        print(f"Time range: {events_np[:, 0].min():.0f} - {events_np[:, 0].max():.0f} μs")
        pos_events = np.sum(events_np[:, 3] > 0)
        neg_events = np.sum(events_np[:, 3] < 0)
        print(f"Positive events: {pos_events}, Negative events: {neg_events}")
    
    # Save events
    save_h5_events(events_np, args.output_file)
    print(f"Events saved to: {args.output_file}")
    
    # Debug mode
    if args.debug:
        debug_dir = config['debug']['output_dir']
        print(f"Running debug mode, saving visualizations to: {debug_dir}")
        
        try:
            from .debug_visualizer import EventsVoxelVisualizer
        except ImportError:
            # Fallback for direct script execution
            from debug_visualizer import EventsVoxelVisualizer
        visualizer = EventsVoxelVisualizer(output_dir=debug_dir, dpi=config['debug']['dpi'])
        
        # Create comprehensive visualizations for decoded events
        visualizer.visualize_decoded_events(voxel, events_np, sensor_size)
        visualizer.print_summary()

if __name__ == '__main__':
    main()