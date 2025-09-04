#!/usr/bin/env python3
"""
Training Loss Visualization Tool for Event-Voxel DEFLARE Project

Usage:
    python visualize_training_logs.py
    python visualize_training_logs.py --log_dir logs/event_voxel_denoising
    python visualize_training_logs.py --output_dir plots
"""

import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: tensorboard not available. Install with: pip install tensorboard")
    TENSORBOARD_AVAILABLE = False

def parse_tensorboard_logs(log_dir):
    """解析TensorBoard日志文件，提取loss数据"""
    if not TENSORBOARD_AVAILABLE:
        raise ImportError("tensorboard package is required")
    
    log_path = Path(log_dir)
    if not log_path.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")
    
    # 查找event文件
    event_files = list(log_path.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No tensorboard event files found in {log_dir}")
    
    print(f"📁 Found {len(event_files)} tensorboard files in {log_dir}")
    
    # 解析所有数据
    data = {
        'train_batch_loss': {'steps': [], 'values': []},
        'train_epoch_loss': {'steps': [], 'values': []},
        'learning_rate': {'steps': [], 'values': []}
    }
    
    for event_file in event_files:
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        
        # 获取所有可用的scalar标签
        scalar_tags = ea.Tags()['scalars']
        print(f"📊 Available metrics: {scalar_tags}")
        
        # 解析训练batch loss
        if 'Loss/Train_Batch' in scalar_tags:
            events = ea.Scalars('Loss/Train_Batch')
            for event in events:
                data['train_batch_loss']['steps'].append(event.step)
                data['train_batch_loss']['values'].append(event.value)
        
        # 解析训练epoch loss
        if 'Loss/Train_Epoch' in scalar_tags:
            events = ea.Scalars('Loss/Train_Epoch')
            for event in events:
                data['train_epoch_loss']['steps'].append(event.step)
                data['train_epoch_loss']['values'].append(event.value)
        
        # 解析学习率
        if 'LR' in scalar_tags:
            events = ea.Scalars('LR')
            for event in events:
                data['learning_rate']['steps'].append(event.step)
                data['learning_rate']['values'].append(event.value)
    
    # 数据统计
    for key, values in data.items():
        if values['values']:
            print(f"✅ {key}: {len(values['values'])} data points")
        else:
            print(f"⚠️  {key}: No data found")
    
    return data

def create_loss_plots(data, output_dir):
    """Create loss visualization charts"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set matplotlib style (English only, no Chinese fonts)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['font.family'] = 'serif'  # Use standard English fonts
    
    # Create multi-subplot layout
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Training Batch Loss Curve (Main plot)
    ax1 = fig.add_subplot(gs[0, :])
    if data['train_batch_loss']['values']:
        steps = data['train_batch_loss']['steps']
        losses = data['train_batch_loss']['values']
        
        # Plot raw curve
        ax1.plot(steps, losses, 'b-', alpha=0.6, linewidth=0.8, label='Training Loss (per batch)')
        
        # Add smoothed curve (moving average)
        if len(losses) > 50:
            window_size = min(50, len(losses) // 10)
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            smooth_steps = steps[window_size-1:]
            ax1.plot(smooth_steps, smoothed, 'r-', linewidth=2, label=f'Smoothed (window={window_size})')
        
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve (Batch Level)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.set_yscale('log')  # Log scale for better loss visualization
        
        # Add statistics info
        min_loss = min(losses)
        final_loss = losses[-1]
        ax1.text(0.02, 0.98, f'Min Loss: {min_loss:.4f}\nFinal Loss: {final_loss:.4f}', 
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax1.text(0.5, 0.5, 'No training batch loss data found', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=16)
        ax1.set_title('Training Loss Curve (No Data)', fontsize=14)
    
    # 2. Training Epoch Loss Curve
    ax2 = fig.add_subplot(gs[1, 0])
    if data['train_epoch_loss']['values']:
        steps = data['train_epoch_loss']['steps']
        losses = data['train_epoch_loss']['values']
        
        ax2.plot(steps, losses, 'g-o', linewidth=2, markersize=6, label='Training Loss (per epoch)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Epoch-level Loss', fontsize=12, fontweight='bold')
        ax2.legend()
        
        # Annotate best epoch
        min_idx = np.argmin(losses)
        ax2.annotate(f'Best: {losses[min_idx]:.4f}', 
                    xy=(steps[min_idx], losses[min_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
    else:
        ax2.text(0.5, 0.5, 'No epoch loss data found', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Epoch Loss (No Data)', fontsize=12)
    
    # 3. Learning Rate Curve
    ax3 = fig.add_subplot(gs[1, 1])
    if data['learning_rate']['values']:
        steps = data['learning_rate']['steps']
        lrs = data['learning_rate']['values']
        
        ax3.plot(steps, lrs, 'purple', linewidth=2, label='Learning Rate')
        ax3.set_xlabel('Training Iteration')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.set_yscale('log')  # Log scale for learning rate display
    else:
        ax3.text(0.5, 0.5, 'No learning rate data found', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Learning Rate Schedule (No Data)', fontsize=12)
    
    # Main title
    fig.suptitle('Event-Voxel DEFLARE Training Progress Monitor', fontsize=16, fontweight='bold')
    
    # Save plot
    plot_file = output_path / 'training_loss_curves.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Loss curves saved to: {plot_file}")
    
    # Don't show plot window, just save
    plt.close()

def create_separate_plots(data, output_dir):
    """Create separate detailed plots for analysis"""
    output_path = Path(output_dir)
    
    # 1. Detailed batch loss analysis
    if data['train_batch_loss']['values']:
        plt.figure(figsize=(12, 6))
        steps = data['train_batch_loss']['steps']
        losses = data['train_batch_loss']['values']
        
        plt.subplot(1, 2, 1)
        plt.plot(steps, losses, 'b-', alpha=0.7, linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve (Linear Scale)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(steps, losses, 'b-', alpha=0.7, linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scale)')
        plt.title('Training Loss Curve (Log Scale)')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        detailed_file = output_path / 'detailed_batch_loss.png'
        plt.savefig(detailed_file, dpi=300, bbox_inches='tight')
        print(f"💾 Detailed batch loss saved to: {detailed_file}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize training logs for Event-Voxel DEFLARE')
    parser.add_argument('--log_dir', default='logs/event_voxel_denoising',
                       help='TensorBoard log directory (default: logs/event_voxel_denoising)')
    parser.add_argument('--output_dir', default='debug_output',
                       help='Output directory for plots (default: debug_output)')
    parser.add_argument('--detailed', action='store_true',
                       help='Create additional detailed plots')
    
    args = parser.parse_args()
    
    print("🚀 Event-Voxel DEFLARE Training Log Visualizer")
    print(f"📂 Log directory: {args.log_dir}")
    print(f"📊 Output directory: {args.output_dir}")
    
    try:
        # 解析TensorBoard日志
        data = parse_tensorboard_logs(args.log_dir)
        
        # 创建loss可视化图表
        create_loss_plots(data, args.output_dir)
        
        # 创建详细分析图表
        if args.detailed:
            create_separate_plots(data, args.output_dir)
        
        print("✅ Visualization completed successfully!")
        print(f"📁 Check plots in: {args.output_dir}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Tip: Make sure you have run training with TensorBoard logging enabled")
        return 1
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("💡 Install tensorboard: conda install tensorboard or pip install tensorboard")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())