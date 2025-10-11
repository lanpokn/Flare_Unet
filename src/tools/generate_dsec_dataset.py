#!/usr/bin/env python3
"""
DSEC Dataset Generator - 从长炫光文件中智能提取100ms段并处理

基于Linus哲学：
- 数据结构正确: 随机选择 → 安全读取 → 100ms提取 → 多方法处理 → 统一可视化
- 消除特殊情况: 统一处理流程，复用现有工具
- 实用主义: 内存安全，避免溢出

功能：
1. 从flare_events文件夹随机选择长H5文件
2. 智能读取：先读时间戳，再只读取需要的100ms范围（避免内存溢出）
3. 保存到DSEC_data/input（复用现有命名方式）
4. 运行所有处理方法：UNet3D, PFD, Baseline, EFR
5. 生成可视化到DSEC_data/visualize

Usage:
    python src/tools/generate_dsec_dataset.py --num_samples 5
    python src/tools/generate_dsec_dataset.py --num_samples 10 --debug
"""

import os
import sys
import random
import h5py
import hdf5plugin  # 必须import以支持gzip压缩的H5文件
import numpy as np
from pathlib import Path
from datetime import datetime
import subprocess  # 仅UNet3D inference需要
import argparse
from typing import Tuple, Optional

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.event_video_generator import EventVideoGenerator
from src.data_processing.encode import load_h5_events, events_to_voxel
from src.data_processing.decode import voxel_to_events

# 导入处理器类
sys.path.append(str(PROJECT_ROOT / 'ext' / 'PFD'))
sys.path.append(str(PROJECT_ROOT / 'ext' / 'EFR-main'))
from batch_pfd_processor import BatchPFDProcessor
from batch_efr_processor import BatchEFRProcessor


class DSECDatasetGenerator:
    """DSEC数据集生成器 - 内存安全的100ms段提取与处理"""

    def __init__(self,
                 flare_dir: str = "/mnt/e/2025/event_flick_flare/main/data/flare_events",
                 output_base: str = "DSEC_data",
                 debug: bool = False):
        """
        Args:
            flare_dir: 长炫光文件目录（WSL格式）
            output_base: DSEC_data基础目录
            debug: 是否启用debug模式
        """
        self.flare_dir = Path(flare_dir)
        self.output_base = Path(output_base)
        self.debug = debug

        # 创建输出目录结构
        self.input_dir = self.output_base / "input"
        self.inputpfds_dir = self.output_base / "inputpfds"
        self.output_dir = self.output_base / "output"
        self.outputbaseline_dir = self.output_base / "outputbaseline"
        self.inputefr_dir = self.output_base / "inputefr"  # 新增EFR
        self.visualize_dir = self.output_base / "visualize"

        # 创建所有必要的目录
        for dir_path in [self.input_dir, self.inputpfds_dir, self.output_dir,
                         self.outputbaseline_dir, self.inputefr_dir, self.visualize_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 视频生成器
        self.video_generator = EventVideoGenerator(
            sensor_size=(480, 640),
            frame_duration_ms=2.5,
            fps=10
        )

        # 初始化处理器
        self.pfd_processor = BatchPFDProcessor(debug=False)
        self.efr_processor = BatchEFRProcessor(debug=False)

        print(f"🚀 DSEC Dataset Generator initialized")
        print(f"📂 Flare source: {self.flare_dir}")
        print(f"📂 Output base: {self.output_base}")

    def get_random_flare_file(self) -> Path:
        """随机选择一个炫光文件"""
        flare_files = list(self.flare_dir.glob("*.h5"))
        if not flare_files:
            raise FileNotFoundError(f"No H5 files found in {self.flare_dir}")

        selected = random.choice(flare_files)
        print(f"📄 Selected flare file: {selected.name}")
        return selected

    def get_time_range_safe(self, file_path: Path) -> Tuple[int, int]:
        """安全获取H5文件的时间范围（不加载全部数据）"""
        with h5py.File(file_path, 'r') as f:
            t_data = f['events']['t']
            # 只读取首尾元素来确定时间范围
            t_min = int(t_data[0])
            t_max = int(t_data[-1])

        print(f"  Time range: {t_min/1000:.1f}ms - {t_max/1000:.1f}ms (duration: {(t_max-t_min)/1000:.1f}ms)")
        return t_min, t_max

    def extract_100ms_segment_safe(self, file_path: Path, start_time_us: int) -> np.ndarray:
        """
        内存安全地提取100ms事件段

        核心策略：先读时间戳数组，找到索引范围，再只读取该范围的所有数据
        """
        segment_duration_us = 100000  # 100ms
        end_time_us = start_time_us + segment_duration_us

        with h5py.File(file_path, 'r') as f:
            events_group = f['events']

            # Step 1: 只读取时间戳数组来确定索引范围
            t_all = events_group['t'][:]

            # Step 2: 使用布尔索引找到100ms范围内的事件索引
            mask = (t_all >= start_time_us) & (t_all < end_time_us)
            indices = np.where(mask)[0]

            if len(indices) == 0:
                print(f"  ⚠️  No events in selected time window")
                return np.empty((0, 4))

            # Step 3: 只读取这个范围的数据（内存安全）
            idx_start = indices[0]
            idx_end = indices[-1] + 1

            t = events_group['t'][idx_start:idx_end]
            x = events_group['x'][idx_start:idx_end]
            y = events_group['y'][idx_start:idx_end]
            p = events_group['p'][idx_start:idx_end]

            # 极性转换（统一为-1/1格式）
            p_converted = np.where(p == 1, 1, -1)

            # 组合成(N,4)格式
            events_segment = np.column_stack((t, x, y, p_converted))

        print(f"  ✅ Extracted {len(events_segment):,} events from segment")
        return events_segment

    def save_h5_events(self, events: np.ndarray, output_path: Path):
        """保存事件到H5文件（标准DSEC格式）"""
        with h5py.File(output_path, 'w') as f:
            events_group = f.create_group('events')
            events_group.create_dataset('t', data=events[:, 0].astype(np.int64),
                                       compression='gzip', compression_opts=9)
            events_group.create_dataset('x', data=events[:, 1].astype(np.uint16),
                                       compression='gzip', compression_opts=9)
            events_group.create_dataset('y', data=events[:, 2].astype(np.uint16),
                                       compression='gzip', compression_opts=9)
            events_group.create_dataset('p', data=events[:, 3].astype(np.int8),
                                       compression='gzip', compression_opts=9)

    def generate_filename(self, source_file: Path, start_time_us: int) -> str:
        """
        生成DSEC标准文件名

        格式: real_flare_{source}_t{time}ms_{datetime}.h5
        """
        source_name = source_file.stem  # 例如：zurich_city_03_a
        time_ms = int(start_time_us / 1000)
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"real_flare_{source_name}_t{time_ms}ms_{datetime_str}.h5"
        return filename

    def run_unet_inference(self, input_h5: Path, output_h5: Path):
        """运行UNet3D推理"""
        cmd = [
            sys.executable, "main.py", "inference",
            "--config", "configs/inference_config.yaml",
            "--input", str(input_h5),
            "--output", str(output_h5)
        ]

        print(f"  🔧 Running UNet3D inference...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.returncode == 0:
            print(f"  ✅ UNet3D inference completed")
        else:
            print(f"  ❌ UNet3D inference failed: {result.stderr}")

    def run_pfd_processing(self, input_h5: Path, output_h5: Path):
        """运行PFD处理（直接调用）"""
        print(f"  🔧 Running PFD processing...")
        try:
            success = self.pfd_processor.process_single_file(input_h5, output_h5, file_idx=0)
            if success:
                print(f"  ✅ PFD processing completed")
            else:
                print(f"  ❌ PFD processing failed")
        except Exception as e:
            print(f"  ❌ PFD processing failed: {e}")

    def run_efr_processing(self, input_h5: Path, output_h5: Path):
        """运行EFR处理（直接调用）"""
        print(f"  🔧 Running EFR processing...")
        print(f"    Input: {input_h5.name} ({input_h5.stat().st_size/1024/1024:.1f}MB)")
        try:
            success = self.efr_processor.process_single_file(input_h5, output_h5, file_idx=0)
            if success and output_h5.exists():
                output_size = output_h5.stat().st_size / 1024 / 1024
                print(f"  ✅ EFR processing completed - Output: {output_size:.1f}MB")
                if output_size < 0.1:  # Less than 100KB is suspicious
                    print(f"  ⚠️  Warning: EFR output file is unusually small!")
            else:
                print(f"  ❌ EFR processing failed")
        except Exception as e:
            print(f"  ❌ EFR processing failed: {e}")
            import traceback
            traceback.print_exc()

    def run_baseline_processing(self, input_h5: Path, output_h5: Path):
        """运行Baseline（编解码only）处理（直接实现）"""
        print(f"  🔧 Running Baseline processing...")
        try:
            # Baseline: Events → Voxel → Events (测试编解码保真度)
            events_np = load_h5_events(str(input_h5))

            # Encode
            voxel = events_to_voxel(
                events_np,
                num_bins=8,
                sensor_size=(480, 640),
                fixed_duration_us=100000  # 100ms
            )

            # Decode
            output_events = voxel_to_events(
                voxel,
                total_duration=100000,
                sensor_size=(480, 640)
            )

            # Save to H5
            self.save_h5_events(output_events, output_h5)
            print(f"  ✅ Baseline processing completed")
        except Exception as e:
            print(f"  ❌ Baseline processing failed: {e}")

    def generate_visualizations(self, base_filename: str,
                               input_h5: Path,
                               unet_h5: Path,
                               pfd_h5: Path,
                               efr_h5: Path,
                               baseline_h5: Path):
        """生成所有方法的可视化（同一输入的所有结果放在同一子文件夹）"""
        # 创建子文件夹（使用文件基础名）
        vis_subdir = self.visualize_dir / Path(base_filename).stem
        vis_subdir.mkdir(parents=True, exist_ok=True)

        print(f"  🎬 Generating visualizations to: {vis_subdir.name}/")

        # 定义所有需要可视化的文件
        vis_tasks = [
            (input_h5, "input"),
            (unet_h5, "unet_output"),
            (pfd_h5, "pfd_output"),
            (efr_h5, "efr_output"),
            (baseline_h5, "baseline_output")
        ]

        for h5_file, method_name in vis_tasks:
            if h5_file.exists():
                try:
                    output_video = vis_subdir / f"{method_name}.mp4"
                    self.video_generator.process_h5_file(str(h5_file), str(output_video))
                    print(f"    ✅ {method_name}.mp4 generated")
                except Exception as e:
                    print(f"    ❌ {method_name} visualization failed: {e}")

    def generate_single_sample(self):
        """生成单个DSEC样本（完整流程）"""
        print("\n" + "="*80)
        print("🎯 Generating new DSEC sample...")

        # Step 1: 随机选择炫光文件
        source_file = self.get_random_flare_file()

        # Step 2: 安全获取时间范围
        t_min, t_max = self.get_time_range_safe(source_file)

        # Step 3: 随机选择100ms起始时间
        max_start = t_max - 100000  # 确保有完整的100ms
        if max_start <= t_min:
            print(f"  ⚠️  File too short, using entire duration")
            start_time = t_min
        else:
            start_time = random.randint(t_min, max_start)

        print(f"  🎲 Random start time: {start_time/1000:.1f}ms")

        # Step 4: 内存安全地提取100ms段
        events_segment = self.extract_100ms_segment_safe(source_file, start_time)

        if len(events_segment) == 0:
            print(f"  ❌ No events in segment, skipping...")
            return False

        # Step 5: 生成文件名并保存到input
        filename = self.generate_filename(source_file, start_time)
        input_h5 = self.input_dir / filename

        print(f"  💾 Saving to: {filename}")
        self.save_h5_events(events_segment, input_h5)

        # Step 6: 运行所有处理方法
        print(f"\n  🔄 Processing with all methods...")

        # UNet3D
        unet_h5 = self.output_dir / filename
        self.run_unet_inference(input_h5, unet_h5)

        # PFD
        pfd_h5 = self.inputpfds_dir / filename
        self.run_pfd_processing(input_h5, pfd_h5)

        # EFR (新增)
        efr_h5 = self.inputefr_dir / filename
        self.run_efr_processing(input_h5, efr_h5)

        # Baseline
        baseline_h5 = self.outputbaseline_dir / filename
        self.run_baseline_processing(input_h5, baseline_h5)

        # Step 7: 生成可视化
        print(f"\n  📊 Generating visualizations...")
        self.generate_visualizations(
            filename, input_h5, unet_h5, pfd_h5, efr_h5, baseline_h5
        )

        print(f"\n✅ Sample generation completed: {filename}")
        return True

    def generate_batch(self, num_samples: int):
        """批量生成DSEC样本"""
        print(f"\n🚀 Starting batch generation: {num_samples} samples")
        print("="*80)

        success_count = 0
        for i in range(num_samples):
            print(f"\n[Sample {i+1}/{num_samples}]")
            if self.generate_single_sample():
                success_count += 1

        print("\n" + "="*80)
        print(f"🎉 Batch generation completed!")
        print(f"📊 Success: {success_count}/{num_samples} samples")
        print(f"📂 Output: {self.output_base}")


def main():
    parser = argparse.ArgumentParser(description="DSEC Dataset Generator - Memory-safe 100ms extraction and processing")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--flare_dir", default="/mnt/e/2025/event_flick_flare/main/data/flare_events",
                       help="Flare events directory (WSL format)")
    parser.add_argument("--output_base", default="DSEC_data", help="Output base directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    generator = DSECDatasetGenerator(
        flare_dir=args.flare_dir,
        output_base=args.output_base,
        debug=args.debug
    )

    generator.generate_batch(args.num_samples)


if __name__ == "__main__":
    main()
