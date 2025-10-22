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

        # 创建输出目录结构（基础目录）
        self.input_dir = self.output_base / "input"
        self.inputpfda_dir = self.output_base / "inputpfda"  # PFD-A (score_select=1)
        self.inputpfdb_dir = self.output_base / "inputpfdb"  # PFD-B (score_select=0)
        self.outputbaseline_dir = self.output_base / "outputbaseline"
        self.inputefr_dir = self.output_base / "inputefr"
        self.visualize_dir = self.output_base / "visualize"

        # UNet checkpoint配置 - 2025-10-22新增physics_noRandom和physics_noRandom_noTen
        checkpoint_base = PROJECT_ROOT / "checkpoints"
        checkpoint_old_base = PROJECT_ROOT / "checkpoints_old"
        self.unet_checkpoints = {
            'simple': str(checkpoint_base / 'event_voxel_deflare_simple' / 'checkpoint_epoch_0031_iter_040000.pth'),
            'full': str(checkpoint_base / 'event_voxel_deflare_full' / 'checkpoint_epoch_0031_iter_040000.pth'),
            'physics_noRandom_method': str(checkpoint_base / 'physics_noRandom_method' / 'checkpoint_epoch_0031_iter_040000.pth'),
            'physics_noRandom_noTen_method': str(checkpoint_base / 'event_voxel_deflare_physics_noRandom_noTen_method' / 'checkpoint_epoch_0031_iter_040000.pth'),
            'full_old': str(checkpoint_old_base / 'event_voxel_deflare_full' / 'checkpoint_epoch_0032_iter_076250.pth'),
            'simple_old': str(checkpoint_old_base / 'event_voxel_deflare_simple' / 'checkpoint_epoch_0027_iter_076250.pth'),
        }

        # 为每个UNet变体创建输出目录
        self.unet_output_dirs = {}
        for variant_name in self.unet_checkpoints.keys():
            output_dir = self.output_base / f"output_{variant_name}"
            self.unet_output_dirs[variant_name] = output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

        # 创建基础目录
        for dir_path in [self.input_dir, self.inputpfda_dir, self.inputpfdb_dir,
                         self.outputbaseline_dir, self.inputefr_dir, self.visualize_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 视频生成器
        self.video_generator = EventVideoGenerator(
            sensor_size=(480, 640),
            frame_duration_ms=2.5,
            fps=10
        )

        # 初始化处理器
        self.pfd_processor_a = BatchPFDProcessor(debug=False)
        self.pfd_processor_a.pfds_params['score_select'] = 1  # PFD-A

        self.pfd_processor_b = BatchPFDProcessor(debug=False)
        self.pfd_processor_b.pfds_params['score_select'] = 0  # PFD-B

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

        核心策略：分块二分查找边界索引，避免加载整个时间戳数组
        """
        segment_duration_us = 100000  # 100ms
        end_time_us = start_time_us + segment_duration_us

        with h5py.File(file_path, 'r') as f:
            events_group = f['events']
            t_dataset = events_group['t']
            total_events = len(t_dataset)

            # Step 1: 分块二分查找起始索引 (避免加载全部数据)
            chunk_size = 100000  # 每次读取10万个时间戳
            idx_start = self._binary_search_time_index(
                t_dataset, start_time_us, 0, total_events, chunk_size, find_start=True
            )

            # Step 2: 从起始索引附近查找结束索引
            idx_end = self._binary_search_time_index(
                t_dataset, end_time_us, idx_start, total_events, chunk_size, find_start=False
            )

            if idx_start >= idx_end:
                print(f"  ⚠️  No events in selected time window")
                return np.empty((0, 4))

            # Step 3: 只读取找到的范围（内存安全，带错误处理）
            try:
                t = t_dataset[idx_start:idx_end]
                x = events_group['x'][idx_start:idx_end]
                y = events_group['y'][idx_start:idx_end]
                p = events_group['p'][idx_start:idx_end]
            except OSError as e:
                if "B-tree signature" in str(e) or "filter returned failure" in str(e):
                    print(f"  ❌ H5 data corrupted (x/y/p coordinate): {e}")
                    print(f"  ⏭️  Skipping corrupted segment at {start_time_us/1000:.1f}ms")
                    return None  # 返回None表示损坏段
                else:
                    raise  # 其他错误继续抛出

            # 极性转换（统一为-1/1格式）
            p_converted = np.where(p == 1, 1, -1)

            # 组合成(N,4)格式
            events_segment = np.column_stack((t, x, y, p_converted))

        print(f"  ✅ Extracted {len(events_segment):,} events from segment")
        return events_segment

    def _binary_search_time_index(self, t_dataset, target_time: int,
                                   left: int, right: int, chunk_size: int,
                                   find_start: bool = True) -> int:
        """
        分块二分查找时间索引（内存友好）

        Args:
            t_dataset: H5 dataset对象（不加载到内存）
            target_time: 目标时间戳（微秒）
            left, right: 搜索范围
            chunk_size: 每次读取的事件数量
            find_start: True=查找>=target的第一个索引, False=查找<target的最后一个索引+1

        Returns:
            索引位置
        """
        while left < right:
            mid = (left + right) // 2

            # 分块读取：只读取mid附近的chunk
            chunk_start = max(0, mid - chunk_size // 2)
            chunk_end = min(len(t_dataset), chunk_start + chunk_size)
            t_chunk = t_dataset[chunk_start:chunk_end]

            # 在chunk内找到mid对应的时间戳
            mid_offset = mid - chunk_start
            if mid_offset < 0 or mid_offset >= len(t_chunk):
                # 边界情况：直接读取mid位置
                t_mid = t_dataset[mid]
            else:
                t_mid = t_chunk[mid_offset]

            if find_start:
                # 查找第一个 >= target_time 的位置
                if t_mid < target_time:
                    left = mid + 1
                else:
                    right = mid
            else:
                # 查找第一个 >= target_time 的位置（作为end）
                if t_mid < target_time:
                    left = mid + 1
                else:
                    right = mid

        return left

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

    def _check_all_outputs_exist(self, filename: str) -> bool:
        """
        检查某个文件的所有方法输出是否都存在

        Args:
            filename: 输入文件名

        Returns:
            True if 所有输出都存在, False otherwise
        """
        # 检查input
        if not (self.input_dir / filename).exists():
            return False

        # 检查所有UNet变体
        for variant_name, output_dir in self.unet_output_dirs.items():
            if not (output_dir / filename).exists():
                return False

        # 检查传统方法
        if not (self.inputpfda_dir / filename).exists():
            return False
        if not (self.inputpfdb_dir / filename).exists():
            return False
        if not (self.inputefr_dir / filename).exists():
            return False
        if not (self.outputbaseline_dir / filename).exists():
            return False

        # 所有输出都存在
        return True

    def find_existing_filename(self, source_file: Path, start_time_us: int) -> str:
        """
        查找已存在的文件名（基于source和time，忽略datetime）

        Returns:
            已存在的文件名，如果不存在则生成新文件名
        """
        source_name = source_file.stem
        time_ms = int(start_time_us / 1000)

        # 查找匹配的文件（忽略datetime部分）
        pattern = f"real_flare_{source_name}_t{time_ms}ms_*.h5"

        # 在input目录查找
        matches = list(self.input_dir.glob(pattern))
        if matches:
            # 返回第一个匹配的文件名（basename）
            return matches[0].name

        # 如果不存在，生成新文件名
        datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"real_flare_{source_name}_t{time_ms}ms_{datetime_str}.h5"
        return filename

    def generate_filename(self, source_file: Path, start_time_us: int) -> str:
        """
        生成DSEC标准文件名（优先使用已存在的文件名）

        格式: real_flare_{source}_t{time}ms_{datetime}.h5
        """
        return self.find_existing_filename(source_file, start_time_us)

    def run_unet_inference(self, input_h5: Path, output_h5: Path, checkpoint_path: str, variant_name: str = "standard"):
        """
        运行UNet3D推理

        Args:
            input_h5: 输入H5文件
            output_h5: 输出H5文件
            checkpoint_path: checkpoint文件路径
            variant_name: 权重变体名称（用于日志）
        """
        # 临时修改inference_config.yaml中的checkpoint路径
        import yaml
        config_path = PROJECT_ROOT / "configs" / "inference_config.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        original_path = config['model']['path']
        config['model']['path'] = checkpoint_path

        # 写入临时配置
        temp_config_path = PROJECT_ROOT / f"configs/temp_inference_{variant_name}.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)

        cmd = [
            sys.executable, "main.py", "inference",
            "--config", str(temp_config_path),
            "--input", str(input_h5),
            "--output", str(output_h5)
        ]

        print(f"    🔧 Running UNet3D ({variant_name})...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=300)

            if result.returncode == 0:
                print(f"    ✅ UNet3D ({variant_name}) completed")
                success = True
            else:
                print(f"    ❌ UNet3D ({variant_name}) failed")
                if result.stderr:
                    # 只打印关键错误信息
                    error_lines = result.stderr.strip().split('\n')
                    for line in error_lines[-5:]:  # 只打印最后5行
                        if 'ERROR' in line or 'Error' in line:
                            print(f"       {line}")
                success = False
        except subprocess.TimeoutExpired:
            print(f"    ❌ UNet3D ({variant_name}) timeout (>5min)")
            success = False
        except Exception as e:
            print(f"    ❌ UNet3D ({variant_name}) exception: {e}")
            success = False
        finally:
            # 清理临时配置
            if temp_config_path.exists():
                temp_config_path.unlink()

        return success

    def run_all_unet_variants(self, input_h5: Path, filename: str) -> dict:
        """
        运行所有UNet权重变体（动态支持，带断点续存）

        Returns:
            {variant_name: output_h5_path} (只包含成功的)
        """
        outputs = {}
        total_variants = len(self.unet_checkpoints)

        for variant_name, checkpoint_path in self.unet_checkpoints.items():
            output_dir = self.unet_output_dirs[variant_name]
            output_h5 = output_dir / filename

            # 断点续存：检查输出文件是否已存在
            if output_h5.exists():
                print(f"    ⏭️  UNet3D ({variant_name}) skipped - output exists")
                outputs[variant_name] = output_h5
                continue

            # 验证checkpoint存在
            if not Path(checkpoint_path).exists():
                print(f"    ⚠️  UNet3D ({variant_name}) skipped - checkpoint not found")
                continue

            # 运行推理
            success = self.run_unet_inference(input_h5, output_h5, checkpoint_path, variant_name)

            # 只记录成功的输出
            if success and output_h5.exists():
                outputs[variant_name] = output_h5

        print(f"    📊 UNet variants completed: {len(outputs)}/{total_variants}")
        return outputs

    def run_pfda_processing(self, input_h5: Path, output_h5: Path):
        """运行PFD-A处理（score_select=1，带断点续存）"""
        # 断点续存
        if output_h5.exists():
            print(f"  ⏭️  PFD-A skipped - output exists")
            return

        print(f"  🔧 Running PFD-A processing...")
        try:
            success = self.pfd_processor_a.process_single_file(input_h5, output_h5, file_idx=0)
            if success:
                print(f"  ✅ PFD-A processing completed")
            else:
                print(f"  ❌ PFD-A processing failed")
        except Exception as e:
            print(f"  ❌ PFD-A processing failed: {e}")

    def run_pfdb_processing(self, input_h5: Path, output_h5: Path):
        """运行PFD-B处理（score_select=0，带断点续存）"""
        # 断点续存
        if output_h5.exists():
            print(f"  ⏭️  PFD-B skipped - output exists")
            return

        print(f"  🔧 Running PFD-B processing...")
        try:
            success = self.pfd_processor_b.process_single_file(input_h5, output_h5, file_idx=0)
            if success:
                print(f"  ✅ PFD-B processing completed")
            else:
                print(f"  ❌ PFD-B processing failed")
        except Exception as e:
            print(f"  ❌ PFD-B processing failed: {e}")

    def run_efr_processing(self, input_h5: Path, output_h5: Path):
        """运行EFR处理（直接调用，带断点续存）"""
        # 断点续存
        if output_h5.exists():
            print(f"  ⏭️  EFR skipped - output exists")
            return

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
        """运行Baseline（编解码only）处理（直接实现，带断点续存）"""
        # 断点续存
        if output_h5.exists():
            print(f"  ⏭️  Baseline skipped - output exists")
            return

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
                               unet_outputs: dict,
                               pfda_h5: Path,
                               pfdb_h5: Path,
                               efr_h5: Path,
                               baseline_h5: Path):
        """
        生成所有方法的可视化（同一输入的所有结果放在同一子文件夹）

        Args:
            unet_outputs: {variant_name: h5_path} 字典
        """
        # 创建子文件夹（使用文件基础名）
        vis_subdir = self.visualize_dir / Path(base_filename).stem
        vis_subdir.mkdir(parents=True, exist_ok=True)

        print(f"    🎬 Generating visualizations to: {vis_subdir.name}/")

        # 定义所有需要可视化的文件
        vis_tasks = [(input_h5, "input")]

        # 添加所有UNet变体
        for variant, h5_path in unet_outputs.items():
            vis_tasks.append((h5_path, f"unet_{variant}"))

        # 添加其他方法
        vis_tasks.extend([
            (pfda_h5, "pfda_output"),
            (pfdb_h5, "pfdb_output"),
            (efr_h5, "efr_output"),
            (baseline_h5, "baseline_output")
        ])

        for h5_file, method_name in vis_tasks:
            if h5_file.exists():
                try:
                    output_video = vis_subdir / f"{method_name}.mp4"
                    self.video_generator.process_h5_file(str(h5_file), str(output_video))
                    print(f"      ✅ {method_name}.mp4")
                except Exception as e:
                    print(f"      ❌ {method_name} failed: {e}")

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

        # 检查是否损坏或为空
        if events_segment is None:
            print(f"    ⏭️  Segment corrupted, skipping...")
            return False

        if len(events_segment) == 0:
            print(f"    ❌ No events in segment, skipping...")
            return False

        # Step 2: 生成文件名并保存到input（如果不存在）
        filename = self.generate_filename(source_file, start_time)
        input_h5 = self.input_dir / filename

        if not input_h5.exists():
            print(f"    💾 Saving to: {filename}")
            self.save_h5_events(events_segment, input_h5)
        else:
            print(f"    ✅ Input already exists: {filename}")

        # Step 3: 运行所有处理方法（带断点续存，只处理缺失的）
        print(f"    🔄 Processing with all methods...")

        # UNet3D (所有变体，断点续存在run_all_unet_variants内部)
        print(f"    🧠 Running all UNet variants ({len(self.unet_checkpoints)} models)...")
        unet_outputs = self.run_all_unet_variants(input_h5, filename)

        # PFD-A
        pfda_h5 = self.inputpfda_dir / filename
        self.run_pfda_processing(input_h5, pfda_h5)

        # PFD-B
        pfdb_h5 = self.inputpfdb_dir / filename
        self.run_pfdb_processing(input_h5, pfdb_h5)

        # EFR
        efr_h5 = self.inputefr_dir / filename
        self.run_efr_processing(input_h5, efr_h5)

        # Baseline
        baseline_h5 = self.outputbaseline_dir / filename
        self.run_baseline_processing(input_h5, baseline_h5)

        # Step 4: 生成可视化
        print(f"    📊 Generating visualizations...")
        self.generate_visualizations(
            filename, input_h5, unet_outputs, pfda_h5, pfdb_h5, efr_h5, baseline_h5
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

                # 断点续存优化：检查所有方法的输出是否都存在
                filename = self.generate_filename(source_file, start_time)
                all_outputs_exist = self._check_all_outputs_exist(filename)

                if all_outputs_exist:
                    print(f"    ⏭️  Skipping t={start_time/1000:.1f}ms (all outputs exist)")
                    file_skipped += 1
                    total_skipped += 1
                    continue

                # 处理新的采样点（input可能存在，但某些方法输出缺失）
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
