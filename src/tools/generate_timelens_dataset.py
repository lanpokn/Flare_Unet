#!/usr/bin/env python3
"""
TimeLens数据集生成工具 - 从DSEC数据提取2秒片段并应用各种事件处理方法

功能:
1. 从DSEC训练数据中提取前2秒片段（events + images + timestamps）
2. 创建original基准数据集
3. 应用各种事件处理方法（UNet变体 + PFD-A/B + EFR + Baseline）
4. 生成TimeLens格式的完整数据集

输出结构:
timelens/
├── zurich_city_03_a_0-2s_original/          # 原始数据
├── zurich_city_03_a_0-2s_unet_simple/       # UNet simple权重
├── zurich_city_03_a_0-2s_unet_full/         # UNet full权重
├── zurich_city_03_a_0-2s_pfda/              # PFD-A处理
├── zurich_city_03_a_0-2s_pfdb/              # PFD-B处理
├── zurich_city_03_a_0-2s_efr/               # EFR处理
└── zurich_city_03_a_0-2s_baseline/          # Baseline编解码
"""

import sys
from pathlib import Path
import argparse
import shutil
import numpy as np
import h5py
import hdf5plugin  # 关键：处理DSEC压缩HDF5
from tqdm import tqdm
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_processing.encode import load_h5_events, events_to_voxel
from src.data_processing.decode import voxel_to_events
from src.training.training_factory import TrainingFactory
from src.utils.config_loader import load_test_config
import torch


class TimeLensDatasetGenerator:
    """TimeLens数据集生成器"""

    def __init__(self,
                 dsec_source_dir: str,
                 output_base_dir: str = "timelens",
                 duration_seconds: float = 2.0,  # ⭐默认2秒（DSEC RGB相机20FPS，约40帧）
                 image_subsample: int = 1,  # 图像降采样：1=不降采样（DSEC RGB已经是20FPS）
                 debug: bool = False):
        """
        Args:
            dsec_source_dir: DSEC源目录路径（如zurich_city_03_a）
            output_base_dir: 输出基础目录
            duration_seconds: 提取时长（秒）
            image_subsample: 图像降采样因子（通常保持1，DSEC RGB已经是20FPS）
            debug: Debug模式

        Note:
            DSEC timestamps单位是微秒(μs)，不是纳秒
            RGB相机帧率约20FPS，event camera高速异步
        """
        self.dsec_source = Path(dsec_source_dir)
        self.output_base = Path(output_base_dir)
        self.duration_s = duration_seconds
        self.duration_us = int(duration_seconds * 1e6)  # ⭐修复：DSEC时间戳是微秒(μs)，不是纳秒
        self.image_subsample = image_subsample
        self.debug = debug

        # 提取序列名称
        self.sequence_name = self.dsec_source.name

        # 验证源目录结构
        self._validate_source_structure()

        # UNet权重配置
        self.unet_checkpoints = {
            'simple': 'checkpoints/event_voxel_deflare_simple/checkpoint_epoch_0031_iter_040000.pth',
            'full': 'checkpoints/event_voxel_deflare_full/checkpoint_epoch_0031_iter_040000.pth',
            'physics_noRandom_noTen': 'checkpoints/event_voxel_deflare_physics_noRandom_noTen_method/checkpoint_epoch_0031_iter_040000.pth',
            'simple_old': 'checkpoints_old/event_voxel_deflare_simple/checkpoint_epoch_0027_iter_076250.pth',
        }

        print(f"✅ TimeLens生成器初始化完成")
        print(f"   源序列: {self.sequence_name}")
        print(f"   提取时长: {self.duration_s}秒")
        print(f"   输出目录: {self.output_base}")

    def _validate_source_structure(self):
        """验证DSEC源目录结构"""
        required = {
            'events/left/events.h5': self.dsec_source / 'events/left/events.h5',
            'images/timestamps.txt': self.dsec_source / 'images/timestamps.txt',
            'images/left/distorted': self.dsec_source / 'images/left/distorted',
        }

        for name, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"❌ 缺少必要文件: {name}")

        print(f"✅ DSEC源目录结构验证通过: {self.dsec_source}")

    def generate_all(self):
        """生成完整TimeLens数据集（所有变体）"""
        print("\n" + "="*80)
        print("🚀 开始生成TimeLens数据集")
        print("="*80)

        # Step 1: 提取并创建original数据集
        print(f"\n【Step 1/3】提取前{self.duration_s}秒数据，创建original数据集...")
        original_dir = self._create_original_dataset()

        # Step 2: 创建处理变体
        print("\n【Step 2/3】应用各种事件处理方法...")
        self._create_processed_variants(original_dir)

        # Step 3: 生成摘要
        print("\n【Step 3/3】生成数据集摘要...")
        self._generate_summary()

        print("\n" + "="*80)
        print("✅ TimeLens数据集生成完成！")
        print(f"📁 输出目录: {self.output_base.absolute()}")
        print("="*80)

    def _create_original_dataset(self) -> Path:
        """创建original数据集（前N秒数据）"""
        # 输出目录名: zurich_city_03_a_0-1s_original
        output_name = f"{self.sequence_name}_0-{int(self.duration_s)}s_original"
        output_dir = self.output_base / output_name

        # 如果已存在，跳过
        if (output_dir / 'events/left/events.h5').exists():
            print(f"✅ Original数据集已存在，跳过: {output_name}")
            return output_dir

        # 创建TimeLens目录结构
        events_dir = output_dir / 'events/left'
        images_dir = output_dir / 'images/left/distorted'
        events_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 创建目录: {output_dir}")

        # Step 1: 处理timestamps.txt（确定图像时间范围）
        print("   ├─ 处理timestamps.txt...")
        valid_image_indices, first_t_ns, last_t_ns = self._extract_timestamps(
            self.dsec_source / 'images/timestamps.txt',
            output_dir / 'images/timestamps.txt'
        )

        # Step 2: 复制对应的图像
        print(f"   ├─ 复制{len(valid_image_indices)}张图像...")
        self._copy_images(valid_image_indices, images_dir)

        # Step 3: 提取events（使用图像时间范围）
        print("   └─ 提取events.h5...")
        self._extract_events(
            self.dsec_source / 'events/left/events.h5',
            events_dir / 'events.h5',
            first_t_ns,
            last_t_ns
        )

        print(f"✅ Original数据集创建完成: {output_name}")
        return output_dir

    def _extract_timestamps(self, source_file: Path, output_file: Path) -> Tuple[List[int], int, int]:
        """
        提取前N秒的时间戳并降采样

        Returns:
            (valid_indices, first_t_ns, last_t_ns)
        """
        # 读取所有时间戳
        timestamps_ns = []
        with open(source_file, 'r') as f:
            for line in f:
                timestamps_ns.append(int(line.strip()))

        # 找到指定时间范围内的时间戳（DSEC时间戳单位：微秒）
        first_t_us = timestamps_ns[0]
        cutoff_t_us = first_t_us + self.duration_us

        valid_indices = []
        valid_timestamps = []

        for idx, t in enumerate(timestamps_ns):
            if t <= cutoff_t_us:
                # ⭐ 降采样：每image_subsample帧取1帧
                if idx % self.image_subsample == 0:
                    valid_indices.append(idx)
                    valid_timestamps.append(t)
            else:
                break

        # 保存筛选后的时间戳
        with open(output_file, 'w') as f:
            for t in valid_timestamps:
                f.write(f"{t}\n")

        duration_s = (valid_timestamps[-1] - first_t_us) / 1e6 if valid_timestamps else 0
        print(f"      时间范围: {first_t_us} - {valid_timestamps[-1] if valid_timestamps else first_t_us} μs")
        print(f"      持续时长: {duration_s:.3f}秒")
        print(f"      原始帧数: {len([i for i,t in enumerate(timestamps_ns) if t <= cutoff_t_us])}")
        print(f"      降采样后: {len(valid_indices)} 帧 (每{self.image_subsample}帧取1帧)")

        return valid_indices, first_t_us, valid_timestamps[-1] if valid_timestamps else first_t_us

    def _copy_images(self, valid_indices: List[int], output_dir: Path):
        """复制有效时间范围内的图像"""
        source_dir = self.dsec_source / 'images/left/distorted'

        for new_idx, old_idx in enumerate(valid_indices):
            source_img = source_dir / f"{old_idx:06d}.png"
            target_img = output_dir / f"{new_idx:06d}.png"

            if source_img.exists():
                shutil.copy2(source_img, target_img)
            else:
                print(f"⚠️  图像不存在: {source_img}")

    def _extract_events(self, source_h5: Path, output_h5: Path,
                        start_t_us: int, end_t_us: int):
        """提取指定时间范围的events - 修复时间单位(μs)"""
        with h5py.File(source_h5, 'r') as f_in:
            # 读取t_offset（DSEC关键元数据，单位：微秒）
            t_offset = f_in['t_offset'][()] if 't_offset' in f_in else 0

            print(f"      t_offset: {t_offset} μs")
            print(f"      目标时间范围: {start_t_us} - {end_t_us} μs ({(end_t_us-start_t_us)/1e6:.3f}秒)")

            # ⭐ 关键优化：使用searchsorted快速定位索引
            t_dataset = f_in['events/t']
            total_events = t_dataset.shape[0]

            print(f"      DSEC文件总事件数: {total_events:,}")

            # ⭐ 分块读取时间戳（避免加载1.1亿事件的完整数组）
            print(f"      分块扫描时间戳数组...")
            CHUNK_SIZE = 10_000_000  # 每次10M事件
            idx_start = None
            idx_end = None

            for chunk_idx in range(0, total_events, CHUNK_SIZE):
                chunk_end = min(chunk_idx + CHUNK_SIZE, total_events)
                t_chunk = t_dataset[chunk_idx:chunk_end]
                t_absolute = t_chunk.astype(np.int64) + t_offset

                # 找起始索引
                if idx_start is None and np.any(t_absolute >= start_t_us):
                    idx_start = chunk_idx + np.searchsorted(t_absolute, start_t_us, side='left')

                # 找结束索引
                if np.any(t_absolute <= end_t_us):
                    local_end = np.searchsorted(t_absolute, end_t_us, side='right')
                    idx_end = chunk_idx + local_end

                # 已经超出范围，停止扫描
                if idx_start is not None and np.all(t_absolute > end_t_us):
                    break

                if (chunk_idx // CHUNK_SIZE) % 5 == 0:
                    print(f"        扫描进度: {chunk_idx/total_events*100:.1f}%")

            if idx_start is None or idx_end is None:
                raise ValueError(f"❌ 时间范围内无事件: {start_t_us} - {end_t_us} μs")

            num_events = idx_end - idx_start
            print(f"      提取事件索引: {idx_start:,} - {idx_end:,} (共{num_events:,}个)")

            # 只读取需要的范围
            t = t_dataset[idx_start:idx_end]
            x = f_in['events/x'][idx_start:idx_end]
            y = f_in['events/y'][idx_start:idx_end]
            p = f_in['events/p'][idx_start:idx_end]

            # 转为相对时间（从0开始）
            t_relative = t - t[0]

            # 创建输出H5文件（TimeLens格式）
            with h5py.File(output_h5, 'w') as f_out:
                events_group = f_out.create_group('events')

                # 保存事件数据（保持DSEC原始数据类型）
                events_group.create_dataset('t', data=t_relative, compression='gzip', compression_opts=9)
                events_group.create_dataset('x', data=x, compression='gzip', compression_opts=9)
                events_group.create_dataset('y', data=y, compression='gzip', compression_opts=9)
                events_group.create_dataset('p', data=p, compression='gzip', compression_opts=9)

                # 保存元数据
                f_out.create_dataset('t_offset', data=t_offset)

                print(f"      事件统计:")
                print(f"        - 总数: {len(t_relative):,}")
                print(f"        - 时间范围: {t_relative[0]} - {t_relative[-1]} μs")
                print(f"        - 持续时长: {t_relative[-1] / 1e6:.3f}秒")
                print(f"        - 正事件: {np.sum(p == 1):,}")
                print(f"        - 负事件: {np.sum(p == 0):,}")

    def _create_processed_variants(self, original_dir: Path):
        """创建各种处理变体"""
        original_events_h5 = original_dir / 'events/left/events.h5'

        # 获取所有处理方法
        methods = []

        # UNet变体
        for name in self.unet_checkpoints.keys():
            methods.append(('unet_' + name, self._process_unet, name))

        # 传统方法
        methods.extend([
            ('pfda', self._process_pfda, None),
            ('pfdb', self._process_pfdb, None),
            ('efr', self._process_efr, None),
            ('baseline', self._process_baseline, None),
        ])

        # 处理每种方法
        for method_name, process_func, param in tqdm(methods, desc="处理方法"):
            print(f"\n   处理: {method_name}")
            variant_dir = self._create_variant_structure(original_dir, method_name)

            # 处理events
            output_events_h5 = variant_dir / 'events/left/events.h5'

            # 如果已存在，跳过
            if output_events_h5.exists():
                print(f"   ⏭️  {method_name} 已存在，跳过")
                continue

            process_func(original_events_h5, output_events_h5, param)

            print(f"   ✅ {method_name} 完成")

    def _create_variant_structure(self, original_dir: Path, suffix: str) -> Path:
        """创建变体目录结构并复制图像+时间戳"""
        # 生成变体目录名: zurich_city_03_a_0-2s_unet_simple
        variant_name = original_dir.name.replace('_original', f'_{suffix}')
        variant_dir = self.output_base / variant_name

        # 创建目录结构
        events_dir = variant_dir / 'events/left'
        images_dir = variant_dir / 'images'
        events_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        # 复制图像和时间戳（硬链接避免重复存储）
        shutil.copytree(
            original_dir / 'images/left',
            variant_dir / 'images/left',
            dirs_exist_ok=True
        )
        shutil.copy2(
            original_dir / 'images/timestamps.txt',
            variant_dir / 'images/timestamps.txt'
        )

        return variant_dir

    def _process_unet(self, input_h5: Path, output_h5: Path, checkpoint_name: str):
        """UNet处理 - 内存安全版本（基于inference_single.py）"""
        checkpoint_path = self.unet_checkpoints[checkpoint_name]

        # 加载模型
        config = load_test_config('configs/test_config.yaml')
        config['model']['path'] = checkpoint_path

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        factory = TrainingFactory(config)
        model = factory.create_model().to(device)

        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # 加载events（转换为项目标准格式）
        events_np = self._load_dsec_events_as_standard(input_h5)

        # 分段处理（20ms per segment）
        segment_duration_us = 20000
        duration_us = int(events_np[:, 0].max() - events_np[:, 0].min())  # ⭐DSEC events已经是μs
        num_segments = max(1, duration_us // segment_duration_us)

        # ⭐ 内存安全策略：基于段数和总事件数智能选择
        MAX_MEMORY_SEGMENTS = 30  # 段数阈值
        MEMORY_SAFETY_MARGIN = 0.8  # 内存安全系数（80%）

        # 估算内存需求（每个segment平均事件数 × 段数 × 每事件内存）
        avg_events_per_seg = len(events_np) / max(num_segments, 1)
        estimated_memory_mb = (avg_events_per_seg * num_segments * 32) / (1024**2)  # 32字节/事件(4×float64)

        print(f"      文件时长: {duration_us/1000:.1f}ms, 段数: {num_segments}")
        print(f"      估算内存需求: {estimated_memory_mb:.1f}MB")

        # 智能选择处理模式
        use_streaming = (num_segments > MAX_MEMORY_SEGMENTS) or (estimated_memory_mb > 500)

        if not use_streaming:
            # 小文件：内存处理（快速）
            print(f"      ✅ 内存处理模式（{num_segments}段，{estimated_memory_mb:.0f}MB）")
            final_events = self._process_segments_in_memory(
                events_np, model, device, num_segments, segment_duration_us
            )
        else:
            # 大文件：流式磁盘处理（安全）
            import tempfile
            print(f"      ⚠️  流式磁盘处理（{num_segments}段，{estimated_memory_mb:.0f}MB > 500MB或段数 > {MAX_MEMORY_SEGMENTS}）")
            temp_dir = Path(tempfile.mkdtemp(prefix='timelens_unet_'))
            try:
                final_events = self._process_segments_streaming(
                    events_np, model, device, num_segments, segment_duration_us, temp_dir
                )
            finally:
                # 清理临时文件
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        # 保存为DSEC格式
        self._save_dsec_format_events(final_events, output_h5, input_h5)

    def _process_segments_in_memory(self, events_np: np.ndarray, model, device,
                                    num_segments: int, segment_duration_us: int) -> np.ndarray:
        """内存处理模式（段数 <= 50）"""
        all_processed = []

        for seg_idx in range(num_segments):
            start_us = seg_idx * segment_duration_us
            end_us = start_us + segment_duration_us

            # ⭐ DSEC events时间戳是μs，无需转换
            mask = (events_np[:, 0] >= start_us) & (events_np[:, 0] < end_us)
            seg_events = events_np[mask].copy()

            if len(seg_events) == 0:
                continue

            # Events → Voxel → UNet → Events
            voxel = events_to_voxel(seg_events, num_bins=8, sensor_size=(480, 640),
                                   fixed_duration_us=segment_duration_us)

            with torch.no_grad():
                # ⭐ events_to_voxel已经返回torch.Tensor (T,H,W) → (1,T,H,W) → (1,1,T,H,W)
                voxel_tensor = voxel.unsqueeze(0).unsqueeze(0).to(device)  # Add batch & channel dims
                output_voxel_tensor = model(voxel_tensor).cpu()[0,0]  # Remove batch & channel, keep as Tensor

            # ⭐ voxel_to_events期望Tensor输入
            output_events = voxel_to_events(output_voxel_tensor,
                                           total_duration=segment_duration_us,
                                           sensor_size=(480, 640))
            output_events[:, 0] += start_us  # ⭐ 调整为μs
            all_processed.append(output_events)

        return np.vstack(all_processed) if all_processed else np.zeros((0, 4))

    def _process_segments_streaming(self, events_np: np.ndarray, model, device,
                                    num_segments: int, segment_duration_us: int,
                                    temp_dir: Path) -> np.ndarray:
        """流式磁盘处理模式（段数 > 30）- 避免内存累积"""
        print(f"      ⚠️  段数{num_segments} > 30，启用流式磁盘处理")

        MERGE_BATCH_SIZE = 20  # 每20段合并一次

        # Step 1: 逐段处理并保存到磁盘
        for seg_idx in range(num_segments):
            start_us = seg_idx * segment_duration_us
            end_us = start_us + segment_duration_us

            # ⭐ DSEC events时间戳是μs
            mask = (events_np[:, 0] >= start_us) & (events_np[:, 0] < end_us)
            seg_events = events_np[mask].copy()

            if len(seg_events) == 0:
                continue

            # Events → Voxel → UNet → Events
            voxel = events_to_voxel(seg_events, num_bins=8, sensor_size=(480, 640),
                                   fixed_duration_us=segment_duration_us)

            with torch.no_grad():
                # ⭐ events_to_voxel已经返回torch.Tensor (T,H,W) → (1,T,H,W) → (1,1,T,H,W)
                voxel_tensor = voxel.unsqueeze(0).unsqueeze(0).to(device)  # Add batch & channel dims
                output_voxel_tensor = model(voxel_tensor).cpu()[0,0]  # Remove batch & channel, keep as Tensor

                # ⭐ 关键：立即清理GPU缓存
                del voxel_tensor
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

            # ⭐ voxel_to_events期望Tensor输入
            output_events = voxel_to_events(output_voxel_tensor,
                                           total_duration=segment_duration_us,
                                           sensor_size=(480, 640))
            output_events[:, 0] += start_us  # ⭐ μs

            # 保存到临时文件（避免内存累积）
            np.save(temp_dir / f'seg_{seg_idx:04d}.npy', output_events)

            # 定期报告进度+GPU内存
            if (seg_idx + 1) % 10 == 0 or (seg_idx + 1) == num_segments:
                gpu_info = ""
                if device.type == 'cuda':
                    mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
                    gpu_info = f", GPU: {mem_alloc:.1f}MB"
                print(f"      已处理 {seg_idx + 1}/{num_segments} 段{gpu_info}")

        # Step 2: 分批从磁盘合并（避免一次性加载所有段）
        print(f"      合并 {num_segments} 个临时段文件...")

        all_batches = []
        batch_events = []

        for seg_idx in range(num_segments):
            seg_file = temp_dir / f'seg_{seg_idx:04d}.npy'
            if not seg_file.exists():
                continue

            seg_data = np.load(seg_file)
            batch_events.append(seg_data)

            # 每MERGE_BATCH_SIZE段合并一次
            if len(batch_events) >= MERGE_BATCH_SIZE:
                merged_batch = np.vstack(batch_events)
                all_batches.append(merged_batch)
                batch_events = []  # 清空，释放内存

        # 处理剩余的段
        if batch_events:
            merged_batch = np.vstack(batch_events)
            all_batches.append(merged_batch)

        # 最终合并所有批次
        print(f"      最终合并 {len(all_batches)} 个批次...")
        final_events = np.vstack(all_batches) if all_batches else np.zeros((0, 4))

        return final_events

    def _process_pfda(self, input_h5: Path, output_h5: Path, _):
        """PFD-A处理"""
        import sys
        pfd_path = Path(__file__).parent.parent.parent / 'ext/PFD'
        sys.path.insert(0, str(pfd_path))
        from batch_pfd_processor import BatchPFDProcessor

        processor = BatchPFDProcessor(debug=False)
        processor.pfds_params['score_select'] = 1  # PFD-A

        # 转换为临时标准格式处理
        temp_h5 = input_h5.parent / 'temp_pfda.h5'
        self._convert_dsec_to_standard(input_h5, temp_h5)

        processor.process_single_file(temp_h5, temp_h5.parent / 'temp_pfda_out.h5', file_idx=0)

        # 转换回DSEC格式
        self._convert_standard_to_dsec(temp_h5.parent / 'temp_pfda_out.h5', output_h5, input_h5)

        # 清理
        temp_h5.unlink()
        (temp_h5.parent / 'temp_pfda_out.h5').unlink()

    def _process_pfdb(self, input_h5: Path, output_h5: Path, _):
        """PFD-B处理"""
        import sys
        pfd_path = Path(__file__).parent.parent.parent / 'ext/PFD'
        sys.path.insert(0, str(pfd_path))
        from batch_pfd_processor import BatchPFDProcessor

        processor = BatchPFDProcessor(debug=False)
        processor.pfds_params['score_select'] = 0  # PFD-B

        temp_h5 = input_h5.parent / 'temp_pfdb.h5'
        self._convert_dsec_to_standard(input_h5, temp_h5)

        processor.process_single_file(temp_h5, temp_h5.parent / 'temp_pfdb_out.h5', file_idx=0)

        self._convert_standard_to_dsec(temp_h5.parent / 'temp_pfdb_out.h5', output_h5, input_h5)

        temp_h5.unlink()
        (temp_h5.parent / 'temp_pfdb_out.h5').unlink()

    def _process_efr(self, input_h5: Path, output_h5: Path, _):
        """EFR处理"""
        import sys
        efr_path = Path(__file__).parent.parent.parent / 'ext/EFR-main'
        sys.path.insert(0, str(efr_path))
        from batch_efr_processor import BatchEFRProcessor

        processor = BatchEFRProcessor(debug=False)

        temp_h5 = input_h5.parent / 'temp_efr.h5'
        self._convert_dsec_to_standard(input_h5, temp_h5)

        processor.process_single_file(temp_h5, temp_h5.parent / 'temp_efr_out.h5', file_idx=0)

        self._convert_standard_to_dsec(temp_h5.parent / 'temp_efr_out.h5', output_h5, input_h5)

        temp_h5.unlink()
        (temp_h5.parent / 'temp_efr_out.h5').unlink()

    def _process_baseline(self, input_h5: Path, output_h5: Path, _):
        """Baseline: 编解码测试"""
        events_np = self._load_dsec_events_as_standard(input_h5)

        # Events → Voxel → Events (DSEC时间戳是μs)
        duration_us = int(events_np[:, 0].max() - events_np[:, 0].min())
        voxel = events_to_voxel(events_np, num_bins=8, sensor_size=(480, 640),
                               fixed_duration_us=duration_us)
        output_events = voxel_to_events(voxel, total_duration=duration_us,
                                       sensor_size=(480, 640))

        # 调整时间范围
        output_events[:, 0] += events_np[:, 0].min()

        self._save_dsec_format_events(output_events, output_h5, input_h5)

    def _load_dsec_events_as_standard(self, dsec_h5: Path) -> np.ndarray:
        """加载DSEC格式events并转换为项目标准格式 (t,x,y,p)"""
        with h5py.File(dsec_h5, 'r') as f:
            t = f['events/t'][:]
            x = f['events/x'][:]
            y = f['events/y'][:]
            p = f['events/p'][:]

            # DSEC: p=0为负事件, p=1为正事件
            # 项目标准: p=1为正事件, p=-1为负事件
            p_standard = np.where(p == 1, 1, -1)

            events = np.column_stack([t, x, y, p_standard])
            return events

    def _save_dsec_format_events(self, events_np: np.ndarray, output_h5: Path,
                                 reference_h5: Path):
        """保存为DSEC格式events"""
        # 读取参考文件的t_offset
        with h5py.File(reference_h5, 'r') as f_ref:
            t_offset = f_ref['t_offset'][()]

        # 转换极性: -1 → 0
        p_dsec = np.where(events_np[:, 3] == 1, 1, 0).astype(np.uint8)

        with h5py.File(output_h5, 'w') as f:
            events_group = f.create_group('events')

            events_group.create_dataset('t', data=events_np[:, 0].astype(np.int64),
                                       compression='gzip', compression_opts=9)
            events_group.create_dataset('x', data=events_np[:, 1].astype(np.uint16),
                                       compression='gzip', compression_opts=9)
            events_group.create_dataset('y', data=events_np[:, 2].astype(np.uint16),
                                       compression='gzip', compression_opts=9)
            events_group.create_dataset('p', data=p_dsec,
                                       compression='gzip', compression_opts=9)

            f.create_dataset('t_offset', data=t_offset)

    def _convert_dsec_to_standard(self, dsec_h5: Path, standard_h5: Path):
        """转换DSEC格式到项目标准格式"""
        events = self._load_dsec_events_as_standard(dsec_h5)

        with h5py.File(standard_h5, 'w') as f:
            events_group = f.create_group('events')
            events_group.create_dataset('t', data=events[:, 0].astype(np.int64),
                                       compression='gzip', compression_opts=9)
            events_group.create_dataset('x', data=events[:, 1].astype(np.uint16),
                                       compression='gzip', compression_opts=9)
            events_group.create_dataset('y', data=events[:, 2].astype(np.uint16),
                                       compression='gzip', compression_opts=9)
            events_group.create_dataset('p', data=events[:, 3].astype(np.int8),
                                       compression='gzip', compression_opts=9)

    def _convert_standard_to_dsec(self, standard_h5: Path, dsec_h5: Path,
                                  reference_h5: Path):
        """转换项目标准格式到DSEC格式"""
        events = load_h5_events(str(standard_h5))
        self._save_dsec_format_events(events, dsec_h5, reference_h5)

    def _generate_summary(self):
        """生成数据集摘要"""
        summary_file = self.output_base / 'README.md'

        # 统计所有变体
        variants = sorted(self.output_base.glob(f"{self.sequence_name}_*"))

        with open(summary_file, 'w') as f:
            f.write(f"# TimeLens Dataset - {self.sequence_name}\n\n")
            f.write(f"**源数据**: DSEC - {self.sequence_name}\n")
            f.write(f"**提取时长**: {self.duration_s}秒\n")
            f.write(f"**生成时间**: {Path(__file__).stat().st_mtime}\n\n")

            f.write("## 数据集变体\n\n")

            for variant_dir in variants:
                variant_name = variant_dir.name
                events_h5 = variant_dir / 'events/left/events.h5'

                if events_h5.exists():
                    with h5py.File(events_h5, 'r') as h5f:
                        num_events = h5f['events/t'].shape[0]

                    f.write(f"- **{variant_name}**: {num_events:,} events\n")

            f.write("\n## 目录结构\n\n")
            f.write("```\n")
            f.write("sequence_name/\n")
            f.write("├── events/left/\n")
            f.write("│   └── events.h5\n")
            f.write("├── images/\n")
            f.write("│   ├── timestamps.txt\n")
            f.write("│   └── left/distorted/\n")
            f.write("│       ├── 000000.png\n")
            f.write("│       └── ...\n")
            f.write("```\n")

        print(f"📄 摘要文件: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='TimeLens数据集生成工具')
    parser.add_argument('--source', type=str,
                       default='/mnt/e/2025/event_flick_flare/object_detection/dsec-det-master/data/train/zurich_city_03_a',
                       help='DSEC源目录')
    parser.add_argument('--output', type=str, default='timelens',
                       help='输出基础目录')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='提取时长（秒）DSEC RGB相机20FPS，2秒=40帧')
    parser.add_argument('--debug', action='store_true',
                       help='Debug模式')

    args = parser.parse_args()

    # 创建生成器
    generator = TimeLensDatasetGenerator(
        dsec_source_dir=args.source,
        output_base_dir=args.output,
        duration_seconds=args.duration,
        debug=args.debug
    )

    # 生成数据集
    generator.generate_all()


if __name__ == '__main__':
    main()
