"""
Debug visualization module for Events and Voxel analysis

This module provides comprehensive visualization tools for debugging 
the Events ↔ Voxel conversion pipeline.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os


class EventsVoxelVisualizer:
    """Comprehensive visualization toolkit for events and voxel debugging"""
    
    def __init__(self, output_dir="debug_output", dpi=150):
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        
    def visualize_original_events(self, events_np, sensor_size, filename="1_original_events_analysis.png"):
        """Create comprehensive analysis of original events data"""
        if len(events_np) == 0:
            print("No events to visualize")
            return
            
        print("Generating original events visualizations...")
        
        t, x, y, p = events_np[:, 0], events_np[:, 1], events_np[:, 2], events_np[:, 3]
        
        plt.figure(figsize=(20, 12))
        
        # 1. Event image using event_utils if available
        plt.subplot(2, 4, 1)
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'event_utils-master', 'lib'))
            from representations.image import events_to_image
            # Convert to integers for event_utils compatibility
            event_img = events_to_image(x.astype(int), y.astype(int), p, sensor_size=sensor_size)
            plt.imshow(event_img, cmap='RdBu')
            plt.title('Event Image (event_utils)')
            plt.colorbar()
        except ImportError:
            # Fallback implementation
            img = np.zeros(sensor_size)
            pos_events = events_np[p > 0]
            neg_events = events_np[p < 0]
            
            if len(pos_events) > 0:
                pos_x, pos_y = pos_events[:, 1].astype(int), pos_events[:, 2].astype(int)
                valid = (pos_x < sensor_size[1]) & (pos_y < sensor_size[0]) & (pos_x >= 0) & (pos_y >= 0)
                img[pos_y[valid], pos_x[valid]] += 1
                
            if len(neg_events) > 0:
                neg_x, neg_y = neg_events[:, 1].astype(int), neg_events[:, 2].astype(int)  
                valid = (neg_x < sensor_size[1]) & (neg_y < sensor_size[0]) & (neg_x >= 0) & (neg_y >= 0)
                img[neg_y[valid], neg_x[valid]] -= 1
                
            plt.imshow(img, cmap='RdBu')
            plt.title('Event Image (fallback)')
            plt.colorbar()
        
        # 2. Temporal distribution
        plt.subplot(2, 4, 2)
        t_ms = t / 1000  # Convert to ms
        bins = np.linspace(t_ms.min(), t_ms.max(), 50)
        plt.hist(t_ms, bins=bins, alpha=0.7, color='blue')
        plt.xlabel('Time (ms)')
        plt.ylabel('Event Count')
        plt.title(f'Temporal Distribution ({len(events_np):,} events)')
        plt.grid(True)
        
        # 3. Spatial distribution (sampled for performance)
        plt.subplot(2, 4, 3)
        sample_size = min(10000, len(events_np))
        indices = np.random.choice(len(events_np), sample_size, replace=False)
        plt.scatter(x[indices], y[indices], c=p[indices], s=0.1, cmap='RdBu', alpha=0.6)
        plt.xlim(0, sensor_size[1])
        plt.ylim(sensor_size[0], 0)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Event Spatial Distribution (sampled)')
        plt.colorbar()
        
        # 4. Polarity analysis
        plt.subplot(2, 4, 4)
        pos_count = np.sum(p > 0)
        neg_count = np.sum(p < 0)
        plt.bar(['Positive', 'Negative'], [pos_count, neg_count], 
                color=['red', 'blue'], alpha=0.7)
        plt.ylabel('Event Count')
        plt.title('Event Polarity Distribution')
        for i, v in enumerate([pos_count, neg_count]):
            plt.text(i, v + max(pos_count, neg_count) * 0.01, f'{v:,}', ha='center')
            
        # 5. Event rate over time
        plt.subplot(2, 4, 5)
        time_windows = np.linspace(t.min(), t.max(), 20)
        event_rates = []
        for i in range(len(time_windows)-1):
            mask = (t >= time_windows[i]) & (t < time_windows[i+1])
            rate = np.sum(mask) / (time_windows[i+1] - time_windows[i]) * 1e6  # events per second
            event_rates.append(rate)
        plt.plot(time_windows[:-1] / 1000, event_rates, 'o-')
        plt.xlabel('Time (ms)')
        plt.ylabel('Event Rate (events/sec)')
        plt.title('Event Rate Over Time')
        plt.grid(True)
        
        # 6. Spatial coverage analysis
        plt.subplot(2, 4, 6)
        x_coverage = np.bincount(x.astype(int), minlength=sensor_size[1])
        y_coverage = np.bincount(y.astype(int), minlength=sensor_size[0])
        plt.plot(x_coverage, label='X coverage', alpha=0.7)
        plt.plot(y_coverage, label='Y coverage', alpha=0.7)
        plt.xlabel('Pixel Index')
        plt.ylabel('Event Count')
        plt.title('Spatial Coverage')
        plt.legend()
        plt.yscale('log')
        plt.grid(True)
        
        # 7. Event statistics
        plt.subplot(2, 4, 7)
        stats_text = f"""Original Events Statistics:
Total events: {len(events_np):,}
Time range: {t.min():.0f} - {t.max():.0f} μs
Duration: {(t.max()-t.min())/1000:.1f} ms
Positive events: {pos_count:,} ({pos_count/len(events_np)*100:.1f}%)
Negative events: {neg_count:,} ({neg_count/len(events_np)*100:.1f}%)
Event rate: {len(events_np)/(t.max()-t.min())*1e6:.0f} events/sec
Spatial coverage: 
  X: [{x.min():.0f}, {x.max():.0f}] (range: {x.max()-x.min():.0f})
  Y: [{y.min():.0f}, {y.max():.0f}] (range: {y.max()-y.min():.0f})
Density: {len(events_np)/(sensor_size[0]*sensor_size[1]):.2f} events/pixel"""
        plt.text(0.05, 0.5, stats_text, fontsize=10, ha='left', va='center', 
                transform=plt.gca().transAxes, fontfamily='monospace')
        plt.axis('off')
        plt.title('Events Statistics')
        
        # 8. Cumulative events
        plt.subplot(2, 4, 8)
        sorted_indices = np.argsort(t)
        cumulative_count = np.arange(1, len(sorted_indices) + 1)
        plt.plot(t[sorted_indices] / 1000, cumulative_count, 'g-', linewidth=2)
        plt.xlabel('Time (ms)')
        plt.ylabel('Cumulative Event Count')
        plt.title('Cumulative Events Over Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def visualize_voxel_grid(self, voxel, filename_bins="2_voxel_temporal_bins.png", 
                           filename_analysis="3_voxel_detailed_analysis.png"):
        """Create comprehensive voxel grid visualizations"""
        print("Generating voxel grid visualizations...")
        
        # Part 1: Show all temporal bins
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle('Voxel Grid Analysis (16 Temporal Bins)', fontsize=16)
        
        for i in range(16):
            ax = axes[i//4, i%4]
            voxel_slice = voxel[i].numpy()
            im = ax.imshow(voxel_slice, cmap='RdBu', vmin=voxel.min(), vmax=voxel.max())
            ax.set_title(f'Bin {i} (t={i*6.25:.1f}-{(i+1)*6.25:.1f}ms)')
            ax.axis('off')
            
        # Add colorbar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.savefig(os.path.join(self.output_dir, filename_bins), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # Part 2: Detailed analysis
        plt.figure(figsize=(20, 12))
        
        # 1. Voxel sum image
        plt.subplot(2, 4, 1)
        voxel_sum = torch.sum(voxel, dim=0).numpy()
        plt.imshow(voxel_sum, cmap='RdBu')
        plt.title('Voxel Sum (All Bins)')
        plt.colorbar()
        
        # 2. Temporal profile
        plt.subplot(2, 4, 2)
        temporal_sum = torch.sum(voxel, dim=[1, 2]).numpy()
        plt.plot(range(16), temporal_sum, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Temporal Bin')
        plt.ylabel('Total Events')
        plt.title('Voxel Temporal Profile')
        plt.grid(True)
        
        # 3. Voxel value distribution
        plt.subplot(2, 4, 3)
        voxel_flat = voxel.flatten().numpy()
        nonzero_voxels = voxel_flat[voxel_flat != 0]
        if len(nonzero_voxels) > 0:
            plt.hist(nonzero_voxels, bins=50, alpha=0.7)
            plt.xlabel('Voxel Value')
            plt.ylabel('Count')
            plt.title(f'Voxel Value Distribution\n({len(nonzero_voxels):,} non-zero)')
            plt.yscale('log')
        
        # 4. Active pixels per bin
        plt.subplot(2, 4, 4)
        bin_events = [torch.count_nonzero(voxel[i]).item() for i in range(16)]
        plt.bar(range(16), bin_events)
        plt.xlabel('Temporal Bin')
        plt.ylabel('Non-zero Pixels')
        plt.title('Active Pixels per Bin')
        plt.grid(True)
        
        # 5. Min/Max values per bin
        plt.subplot(2, 4, 5)
        max_vals = [torch.max(voxel[i]).item() for i in range(16)]
        min_vals = [torch.min(voxel[i]).item() for i in range(16)]
        plt.plot(range(16), max_vals, 'ro-', label='Max', linewidth=2)
        plt.plot(range(16), min_vals, 'bo-', label='Min', linewidth=2)
        plt.xlabel('Temporal Bin')
        plt.ylabel('Voxel Value')
        plt.title('Min/Max Values per Bin')
        plt.legend()
        plt.grid(True)
        
        # 6. Voxel statistics
        plt.subplot(2, 4, 6)
        voxel_stats = f"""Voxel Grid Statistics:
Shape: {voxel.shape}
Total events: {voxel.sum():.0f}
Non-zero voxels: {torch.count_nonzero(voxel).item():,}
Sparsity: {(1-torch.count_nonzero(voxel).item()/voxel.numel())*100:.2f}%
Value range: [{voxel.min():.3f}, {voxel.max():.3f}]
Mean (non-zero): {voxel[voxel!=0].mean():.3f}
Std (non-zero): {voxel[voxel!=0].std():.3f}

Per-bin summary:
Events per bin: {temporal_sum.mean():.0f} ± {temporal_sum.std():.0f}
Active pixels: {np.mean(bin_events):.0f} ± {np.std(bin_events):.0f}"""
        plt.text(0.1, 0.5, voxel_stats, fontsize=10, ha='left', va='center',
                transform=plt.gca().transAxes, fontfamily='monospace')
        plt.axis('off')
        plt.title('Voxel Statistics')
        
        # 7. Cumulative events over time
        plt.subplot(2, 4, 7)
        cumulative_events = np.cumsum(temporal_sum)
        plt.plot(range(16), cumulative_events, 'g-', linewidth=3)
        plt.xlabel('Temporal Bin')
        plt.ylabel('Cumulative Events')
        plt.title('Cumulative Events')
        plt.grid(True)
        
        # 8. Event density map
        plt.subplot(2, 4, 8)
        event_density = voxel_sum / voxel_sum.max() if voxel_sum.max() > 0 else voxel_sum
        plt.imshow(event_density, cmap='viridis')
        plt.title('Event Density (Normalized)')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename_analysis), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def visualize_decoded_events(self, voxel, events_np, sensor_size, 
                               filename_input="4_input_voxel_bins.png",
                               filename_decoded="5_decoded_events_analysis.png",
                               filename_comparison="6_comparison_analysis.png"):
        """Create comprehensive decoded events visualizations"""
        print("Generating decoded events visualizations...")
        
        # Part 1: Input voxel visualization
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle('Input Voxel Grid (16 Temporal Bins)', fontsize=16)
        
        for i in range(16):
            ax = axes[i//4, i%4]
            voxel_slice = voxel[i].numpy()
            im = ax.imshow(voxel_slice, cmap='RdBu', vmin=voxel.min(), vmax=voxel.max())
            ax.set_title(f'Bin {i} (t={i*6.25:.1f}-{(i+1)*6.25:.1f}ms)')
            ax.axis('off')
            
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        plt.savefig(os.path.join(self.output_dir, filename_input), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        if len(events_np) == 0:
            print("No decoded events to visualize")
            return
            
        # Part 2: Decoded events analysis
        t, x, y, p = events_np[:, 0], events_np[:, 1], events_np[:, 2], events_np[:, 3]
        
        plt.figure(figsize=(20, 15))
        
        # 1. Decoded event image
        plt.subplot(3, 4, 1)
        try:
            import sys
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'event_utils-master', 'lib'))
            from representations.image import events_to_image
            # Convert to integers for event_utils compatibility
            decoded_img = events_to_image(x.astype(int), y.astype(int), p, sensor_size=sensor_size)
            plt.imshow(decoded_img, cmap='RdBu')
            plt.title('Decoded Events (event_utils)')
        except ImportError:
            img = np.zeros(sensor_size)
            for i in range(len(events_np)):
                xi, yi, pi = int(x[i]), int(y[i]), p[i]
                if 0 <= xi < sensor_size[1] and 0 <= yi < sensor_size[0]:
                    img[yi, xi] += pi
            plt.imshow(img, cmap='RdBu')
            plt.title('Decoded Events (fallback)')
        plt.colorbar()
        
        # 2. Temporal distribution with bin boundaries
        plt.subplot(3, 4, 2)
        t_ms = t / 1000
        bins = np.linspace(0, 100, 51)
        plt.hist(t_ms, bins=bins, alpha=0.7, color='green', edgecolor='black')
        # Add voxel bin boundaries
        for i in range(17):
            plt.axvline(i * 6.25, color='red', linestyle='--', alpha=0.7, linewidth=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Event Count')
        plt.title(f'Decoded Temporal Distribution\n({len(events_np):,} events)')
        plt.grid(True, alpha=0.3)
        
        # 3. Spatial-temporal distribution
        plt.subplot(3, 4, 3)
        sample_size = min(5000, len(events_np))
        indices = np.random.choice(len(events_np), sample_size, replace=False)
        plt.scatter(x[indices], y[indices], c=t_ms[indices], s=0.5, cmap='plasma', alpha=0.6)
        plt.xlim(0, sensor_size[1])
        plt.ylim(sensor_size[0], 0)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Spatial-Temporal Distribution')
        plt.colorbar(label='Time (ms)')
        
        # 4. Events per temporal bin
        plt.subplot(3, 4, 4)
        bin_assignment = np.floor(t_ms / 6.25).astype(int)
        bin_assignment = np.clip(bin_assignment, 0, 15)
        bin_counts = np.bincount(bin_assignment, minlength=16)
        plt.bar(range(16), bin_counts, alpha=0.7, color='orange')
        plt.xlabel('Temporal Bin')
        plt.ylabel('Decoded Events')
        plt.title('Events per Temporal Bin')
        plt.grid(True, alpha=0.3)
        
        # 5. Randomness check within bins
        plt.subplot(3, 4, 5)
        bin_with_events = np.where(bin_counts > 100)[0][:4]
        if len(bin_with_events) > 0:
            for i, bin_idx in enumerate(bin_with_events):
                bin_mask = bin_assignment == bin_idx
                bin_times = (t_ms[bin_mask] - bin_idx * 6.25) / 6.25
                plt.hist(bin_times, bins=20, alpha=0.6, label=f'Bin {bin_idx}')
            plt.xlabel('Normalized Time within Bin')
            plt.ylabel('Count')
            plt.title('Time Distribution within Bins\n(Should be uniform)')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Continue with remaining subplots...
        # 6. Polarity analysis
        plt.subplot(3, 4, 6)
        pos_count = np.sum(p > 0)
        neg_count = np.sum(p < 0)
        if pos_count + neg_count > 0:
            if neg_count > 0:
                plt.pie([pos_count, neg_count], 
                       labels=['Positive', 'Negative'], 
                       colors=['red', 'blue'], 
                       autopct='%1.1f%%')
            else:
                plt.pie([pos_count], 
                       labels=['Positive'], 
                       colors=['red'], 
                       autopct='%1.1f%%')
        plt.title(f'Polarity Distribution\nPos: {pos_count:,}, Neg: {neg_count:,}')
        
        # Add remaining subplots and save...
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename_decoded), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        # Part 3: Comparison analysis (original vs reconstructed)
        self._create_comparison_analysis(voxel, events_np, sensor_size, filename_comparison)
        
    def _create_comparison_analysis(self, original_voxel, decoded_events, sensor_size, filename):
        """Create detailed comparison between original and reconstructed voxel"""
        print("Generating comparison analysis...")
        
        # Reconstruct voxel from decoded events
        try:
            from . import encode  # Import from current package
        except ImportError:
            # Fallback for direct script execution
            import encode
        reconstructed_voxel = encode.events_to_voxel(decoded_events, num_bins=16, sensor_size=sensor_size)
        
        plt.figure(figsize=(20, 10))
        
        voxel_sum = torch.sum(original_voxel, dim=0).numpy()
        recon_sum = torch.sum(reconstructed_voxel, dim=0).numpy()
        
        # 1. Original voxel
        plt.subplot(2, 4, 1)
        plt.imshow(voxel_sum, cmap='RdBu')
        plt.title('Original Voxel Sum')
        plt.colorbar()
        
        # 2. Reconstructed voxel
        plt.subplot(2, 4, 2)
        plt.imshow(recon_sum, cmap='RdBu')
        plt.title('Reconstructed Voxel Sum')
        plt.colorbar()
        
        # 3. Difference
        plt.subplot(2, 4, 3)
        diff = voxel_sum - recon_sum
        plt.imshow(diff, cmap='RdBu')
        plt.title(f'Difference\n(L1: {np.abs(diff).mean():.6f})')
        plt.colorbar()
        
        # 4. Temporal comparison
        plt.subplot(2, 4, 4)
        original_temporal = torch.sum(original_voxel, dim=[1, 2]).numpy()
        recon_temporal = torch.sum(reconstructed_voxel, dim=[1, 2]).numpy()
        plt.plot(range(16), original_temporal, 'o-', label='Original', linewidth=2)
        plt.plot(range(16), recon_temporal, 's-', label='Reconstructed', linewidth=2)
        plt.xlabel('Temporal Bin')
        plt.ylabel('Total Events')
        plt.title('Temporal Profile Comparison')
        plt.legend()
        plt.grid(True)
        
        # 5. Error analysis
        plt.subplot(2, 4, 5)
        abs_error = np.abs(diff)
        plt.hist(abs_error[abs_error > 0], bins=50, alpha=0.7)
        plt.xlabel('Absolute Error')
        plt.ylabel('Pixel Count')
        plt.title('Error Distribution')
        plt.yscale('log')
        
        # 6. Statistics
        plt.subplot(2, 4, 6)
        comparison_stats = f"""Comparison Statistics:
Original voxel sum: {voxel_sum.sum():.0f}
Reconstructed sum: {recon_sum.sum():.0f}
Difference: {voxel_sum.sum() - recon_sum.sum():.0f}

L1 error: {np.abs(diff).mean():.6f}
L2 error: {np.sqrt((diff**2).mean()):.6f}
Max error: {np.abs(diff).max():.3f}
Relative L1: {np.abs(diff).sum() / max(np.abs(voxel_sum).sum(), 1):.6f}

Non-zero pixels:
Original: {np.count_nonzero(voxel_sum):,}
Reconstructed: {np.count_nonzero(recon_sum):,}"""
        plt.text(0.1, 0.5, comparison_stats, fontsize=10, ha='left', va='center',
                transform=plt.gca().transAxes, fontfamily='monospace')
        plt.axis('off')
        plt.title('Quantitative Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def print_summary(self):
        """Print summary of generated visualizations"""
        print(f"\nComprehensive debug visualizations saved to {self.output_dir}/")
        print("Generated files:")
        print("- 1_original_events_analysis.png: Original events detailed analysis")  
        print("- 2_voxel_temporal_bins.png: All 16 temporal bins visualization")
        print("- 3_voxel_detailed_analysis.png: Voxel statistics and profiles")
        print("- 4_input_voxel_bins.png: Input voxel for decoding")
        print("- 5_decoded_events_analysis.png: Decoded events comprehensive analysis")
        print("- 6_comparison_analysis.png: Original vs reconstructed comparison")