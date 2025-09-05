#!/usr/bin/env python3
"""
Latest Training Loss Visualization Tool - Only Latest Log File
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: tensorboard not available")
    TENSORBOARD_AVAILABLE = False

def get_latest_log_file(log_dir):
    """Get the most recent log file by modification time"""
    log_path = Path(log_dir)
    event_files = list(log_path.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No event files found in {log_dir}")
    
    # Sort by file size (larger = more data) and modification time
    latest_file = max(event_files, key=lambda x: (x.stat().st_size, x.stat().st_mtime))
    return latest_file

def parse_latest_log(log_dir):
    """Parse only the latest log file"""
    if not TENSORBOARD_AVAILABLE:
        raise ImportError("tensorboard package is required")
    
    latest_file = get_latest_log_file(log_dir)
    print(f"📊 Using latest log: {latest_file.name}")
    print(f"📅 Size: {latest_file.stat().st_size/1024:.1f}KB")
    
    ea = EventAccumulator(str(latest_file))
    ea.Reload()
    
    scalar_tags = ea.Tags()['scalars']
    print(f"📈 Available metrics: {scalar_tags}")
    
    data = {
        'train_batch_loss': {'steps': [], 'values': []},
        'learning_rate': {'steps': [], 'values': []}
    }
    
    # Parse training batch loss
    if 'Loss/Train_Batch' in scalar_tags:
        events = ea.Scalars('Loss/Train_Batch')
        for event in events:
            data['train_batch_loss']['steps'].append(event.step)
            data['train_batch_loss']['values'].append(event.value)
        print(f"✅ Batch loss: {len(events)} points (step {events[0].step} to {events[-1].step})")
        values = [e.value for e in events]
        print(f"   📊 Range: {min(values):.6f} to {max(values):.6f}")
    
    # Parse learning rate
    if 'LR' in scalar_tags:
        events = ea.Scalars('LR')
        for event in events:
            data['learning_rate']['steps'].append(event.step)
            data['learning_rate']['values'].append(event.value)
        print(f"✅ Learning rate: {len(events)} points")
    
    return data

def create_latest_plots(data, output_dir):
    """Create plots from latest data only"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    if data['train_batch_loss']['values']:
        steps = data['train_batch_loss']['steps']
        losses = data['train_batch_loss']['values']
        
        # 1. Linear scale plot
        ax1.plot(steps, losses, 'b-', alpha=0.7, linewidth=1)
        if len(losses) > 50:
            window_size = min(50, len(losses) // 10)
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            smooth_steps = steps[window_size-1:]
            ax1.plot(smooth_steps, smoothed, 'r-', linewidth=2)
        
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('Loss (Linear)')
        ax1.set_title('Training Loss - Linear Scale')
        ax1.text(0.02, 0.98, f'Min: {min(losses):.4f}\\nFinal: {losses[-1]:.4f}\\nTotal steps: {len(steps)}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 2. Log scale plot
        ax2.plot(steps, losses, 'b-', alpha=0.7, linewidth=1)
        if len(losses) > 50:
            ax2.plot(smooth_steps, smoothed, 'r-', linewidth=2)
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Loss (Log)')
        ax2.set_title('Training Loss - Log Scale')
        ax2.set_yscale('log')
        
        # 3. Loss distribution
        ax3.hist(losses, bins=50, alpha=0.7, color='green')
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Loss Distribution')
        ax3.axvline(np.mean(losses), color='red', linestyle='--', label=f'Mean: {np.mean(losses):.4f}')
        ax3.legend()
        
        # 4. Training progress (last 500 steps)
        if len(steps) > 500:
            recent_steps = steps[-500:]
            recent_losses = losses[-500:]
            ax4.plot(recent_steps, recent_losses, 'purple', linewidth=1.5)
            ax4.set_xlabel('Training Iteration')
            ax4.set_ylabel('Loss')
            ax4.set_title('Recent Training Progress (Last 500 Steps)')
            ax4.text(0.02, 0.98, f'Recent Min: {min(recent_losses):.4f}', 
                    transform=ax4.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'Not enough data for recent progress', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Recent Progress (Insufficient Data)')
    
    plt.suptitle('Latest Training Log Analysis - Event-Voxel DEFLARE', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_file = output_path / 'latest_training_analysis.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Latest analysis saved: {plot_file}")
    plt.close()

def main():
    log_dir = 'logs/event_voxel_denoising'
    output_dir = 'debug_output_latest'
    
    print("🚀 Latest Training Log Analyzer")
    print(f"📂 Log directory: {log_dir}")
    print(f"📊 Output directory: {output_dir}")
    
    try:
        data = parse_latest_log(log_dir)
        create_latest_plots(data, output_dir)
        print("✅ Latest log analysis completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())