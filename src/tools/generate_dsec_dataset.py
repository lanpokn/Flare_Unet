#!/usr/bin/env python3
"""
DSEC Dataset Generator - 从长炫光文件中智能提取100ms段并处理

基于Linus哲学：
- 数据结构正确: 顺序处理 → 安全读取 → 100ms提取 → 多方法处理 → 统一可视化
- 消除特殊情况: 统一处理流程，复用现有工具
- 实用主义: 内存安全，避免溢出，断点续存

功能：
1. 从flare_events文件夹按顺序读取长H5文件
2. 每个文件内按时间顺序采样（间隔400ms）：0-100ms, 400-500ms, 800-900ms, ...
3. 智能读取：先读时间戳，再只读取需要的100ms范围（避免内存溢出）
4. 断点续存：解析已有文件名，自动跳过已处理的段
5. 运行所有处理方法：UNet3D, PFD, Baseline, EFR
6. 生成可视化到DSEC_data/visualize

Usage:
    python src/tools/generate_dsec_dataset.py  # 顺序处理所有文件，自动断点续存
    python src/tools/generate_dsec_dataset.py --debug
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
from typing import Tuple, Optional, List, Dict, Set

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

    def get_sorted_flare_files(self) -> List[Path]:
        """获取排序后的炫光文件列表（顺序处理）"""
        flare_files = sorted(list(self.flare_dir.glob("*.h5")))
        if not flare_files:
            raise FileNotFoundError(f"No H5 files found in {self.flare_dir}")

        print(f"📄 Found {len(flare_files)} flare files (sorted)")
        return flare_files

    def get_time_range_safe(self, file_path: Path) -> Tuple[int, int]:
        """安全获取H5文件的时间范围（不加载全部数据）"""
        with h5py.File(file_path, 'r') as f:
            t_data = f['events']['t']
            # 只读取首尾元素来确定时间范围
            t_min = int(t_data[0])
            t_max = int(t_data[-1])

        print(f"  Time range: {t_min/1000:.1f}ms - {t_max/1000:.1f}ms (duration: {(t_max-t_min)/1000:.1f}ms)")
        return t_min, t_max

    def generate_time_samples(self, t_min: int, t_max: int) -> List[int]:
        """
        生成时间采样点列表（间隔400ms）

        采样策略：0-100ms, 400-500ms, 800-900ms, 1200-1300ms, ...

        Args:
            t_min: 文件起始时间（微秒）
            t_max: 文件结束时间（微秒）

        Returns:
            采样起始时间列表（微秒）
        """
        samples = []
        segment_duration = 100000  # 100ms = 100,000μs
        interval = 400000  # 400ms = 400,000μs 间隔

        current_start = t_min
        while current_start + segment_duration <= t_max:
            samples.append(current_start)
            current_start += interval

        print(f"  Generated {len(samples)} time samples (400ms interval)")
        return samples

    def parse_existing_progress(self) -> Dict[str, Set[int]]:
        """
        解析DSEC_data/input中已有文件，推断处理进度（断点续存）

        文件名格式: real_flare_{source}_t{time}ms_{datetime}.h5
        提取信息: source_name, start_time_us

        Returns:
            {source_name: {start_time_us1, start_time_us2, ...}}
        """
        progress = {}

        for h5_file in self.input_dir.glob("real_flare_*.h5"):
            try:
                # 解析文件名
                # 例如: real_flare_zurich_city_03_a_t34867ms_20251011_120721.h5
                stem = h5_file.stem

                # 提取source_name和time
                parts = stem.split('_t')
                if len(parts) >= 2:
                    source_name = parts[0].replace('real_flare_', '')
                    time_part = parts[1].split('ms_')[0]
                    start_time_ms = int(time_part)
                    start_time_us = start_time_ms * 1000

                    if source_name not in progress:
                        progress[source_name] = set()
                    progress[source_name].add(start_time_us)
            except Exception as e:
                print(f"  ⚠️  Warning: Failed to parse {h5_file.name}: {e}")
                continue

        # 打印已有进度
        if progress:
            print(f"📊 Existing progress (断点续存):")
            for source, times in sorted(progress.items()):
                print(f"  {source}: {len(times)} segments processed")
        else:
            print(f"📊 No existing progress found, starting from scratch")

        return progress

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

    def process_single_segment(self, source_file: Path, start_time: int) -> bool:
        """
        处理单个100ms段（完整流程）

        Args:
            source_file: 源H5文件路径
            start_time: 起始时间（微秒）

        Returns:
            是否成功处理
        """
        print(f"  ⏱️  Processing segment: {start_time/1000:.1f}ms - {(start_time+100000)/1000:.1f}ms")

        # Step 1: 内存安全地提取100ms段
        events_segment = self.extract_100ms_segment_safe(source_file, start_time)

        if len(events_segment) == 0:
            print(f"    ❌ No events in segment, skipping...")
            return False

        # Step 2: 生成文件名并保存到input
        filename = self.generate_filename(source_file, start_time)
        input_h5 = self.input_dir / filename

        print(f"    💾 Saving to: {filename}")
        self.save_h5_events(events_segment, input_h5)

        # Step 3: 运行所有处理方法
        print(f"    🔄 Processing with all methods...")

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

        # Step 4: 生成可视化
        print(f"    📊 Generating visualizations...")
        self.generate_visualizations(
            filename, input_h5, unet_h5, pfd_h5, efr_h5, baseline_h5
        )

        print(f"    ✅ Segment completed: {filename}")
        return True

    def generate_batch_sequential(self):
        """
        顺序批量生成DSEC样本（带断点续存）

        处理流程：
        1. 按文件名排序遍历所有长炫光文件
        2. 每个文件内按时间顺序采样（间隔400ms）
        3. 自动跳过已处理的段（断点续存）
        """
        print(f"\n🚀 Starting sequential batch generation with checkpoint resume")
        print("="*80)

        # Step 1: 获取排序后的文件列表
        flare_files = self.get_sorted_flare_files()

        # Step 2: 解析已有进度
        progress = self.parse_existing_progress()

        # Step 3: 遍历每个文件
        total_processed = 0
        total_skipped = 0

        for file_idx, source_file in enumerate(flare_files, 1):
            print(f"\n{'='*80}")
            print(f"📁 File [{file_idx}/{len(flare_files)}]: {source_file.name}")
            print(f"{'='*80}")

            source_name = source_file.stem

            # 获取时间范围
            try:
                t_min, t_max = self.get_time_range_safe(source_file)
            except Exception as e:
                print(f"  ❌ Failed to read time range: {e}")
                continue

            # 生成采样点
            time_samples = self.generate_time_samples(t_min, t_max)

            if len(time_samples) == 0:
                print(f"  ⚠️  No valid time samples, skipping file")
                continue

            # 获取已处理的时间点
            processed_times = progress.get(source_name, set())

            # 遍历每个采样点
            file_processed = 0
            file_skipped = 0

            for sample_idx, start_time in enumerate(time_samples, 1):
                print(f"\n  [Segment {sample_idx}/{len(time_samples)}]")

                # 断点续存：跳过已处理的
                if start_time in processed_times:
                    print(f"    ⏭️  Skipping t={start_time/1000:.1f}ms (already processed)")
                    file_skipped += 1
                    total_skipped += 1
                    continue

                # 处理新的采样点
                try:
                    if self.process_single_segment(source_file, start_time):
                        file_processed += 1
                        total_processed += 1
                        # 更新进度（内存中记录，避免重复处理）
                        if source_name not in progress:
                            progress[source_name] = set()
                        progress[source_name].add(start_time)
                except Exception as e:
                    print(f"    ❌ Failed to process segment: {e}")
                    import traceback
                    traceback.print_exc()

            print(f"\n  📊 File summary: {file_processed} new, {file_skipped} skipped")

        # Final summary
        print("\n" + "="*80)
        print(f"🎉 Sequential batch generation completed!")
        print(f"📊 Total processed: {total_processed} new segments")
        print(f"⏭️  Total skipped: {total_skipped} existing segments")
        print(f"📂 Output: {self.output_base}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="DSEC Dataset Generator - Sequential processing with checkpoint resume"
    )
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

    # 顺序批处理（自动断点续存）
    generator.generate_batch_sequential()


if __name__ == "__main__":
    main()
