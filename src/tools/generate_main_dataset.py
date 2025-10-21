#!/usr/bin/env python3
"""
主实验数据集生成器 - 统一处理仿真和真实数据集

基于Linus哲学：
- 数据结构正确: 扫描固定100ms H5文件 → 多方法处理 → 统一可视化
- 消除特殊情况: 统一仿真/真实数据处理流程，只需切换输入目录
- 实用主义: 解决论文主实验数据集生成的实际需求

功能：
1. 读取输入目录（仿真或真实数据）和对应的target目录
2. 运行所有处理方法：
   - 5个UNet权重变体 (standard, full, simple, simple_timeRandom, physics_noRandom)
   - PFD-A (score_select=1)
   - PFD-B (score_select=0)
   - EFR (线性梳状滤波器)
   - Baseline (纯encode-decode)
3. 生成所有方法的可视化视频
4. 输出目录结构统一，便于后续分析

输出目录结构:
{output_base}/
├── input/              # 原始含炫光数据
├── target/             # 目标去炫光数据 (可选)
├── output/             # UNet3D standard权重
├── output_full/        # UNet3D full权重
├── output_simple/      # UNet3D simple权重
├── output_simple_timeRandom/
├── output_physics_noRandom/
├── inputpfda/          # PFD-A结果
├── inputpfdb/          # PFD-B结果
├── inputefr/           # EFR结果
├── outputbaseline/     # Baseline结果
└── visualize/          # 所有方法的可视化视频
    └── {filename}/
        ├── input.mp4
        ├── target.mp4 (如果有target)
        ├── unet_standard.mp4
        ├── unet_full.mp4
        ├── unet_simple.mp4
        ├── unet_simple_timeRandom.mp4
        ├── unet_physics_noRandom.mp4
        ├── pfda_output.mp4
        ├── pfdb_output.mp4
        ├── efr_output.mp4
        └── baseline_output.mp4

Usage:
    # 仿真数据集（默认）
    python src/tools/generate_main_dataset.py

    # 真实数据集（DSEC）
    python src/tools/generate_main_dataset.py \
      --input_dir DSEC_data/input \
      --output_base DSEC_results

    # 真实数据集（EVK4）
    python src/tools/generate_main_dataset.py \
      --input_dir EVK4/input \
      --target_dir EVK4/target \
      --output_base EVK4_results

    # 测试模式：只处理前3个文件
    python src/tools/generate_main_dataset.py --test --num_samples 3
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import subprocess

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


class MainDatasetGenerator:
    """主实验数据集生成器 - 统一处理仿真和真实数据"""

    def __init__(self,
                 input_dir: str = None,
                 target_dir: str = None,
                 output_base: str = "Main_data",
                 test_mode: bool = False):
        """
        Args:
            input_dir: 输入目录 (含炫光，默认data_simu/physics_method/background_with_flare_events_test)
            target_dir: 目标目录 (去炫光，可选，默认data_simu/physics_method/background_with_light_events_test)
            output_base: 输出基础目录 (默认Main_data，仿真用MainSimu_data，真实用MainReal_data)
            test_mode: 测试模式（跳过某些耗时操作）
        """
        # 默认使用仿真数据集路径
        if input_dir is None:
            self.input_source_dir = PROJECT_ROOT / "data_simu/physics_method/background_with_flare_events_test"
            # 仿真数据默认输出目录
            if output_base == "Main_data":
                output_base = "MainSimu_data"
        else:
            self.input_source_dir = Path(input_dir)

        # Target目录可选（真实数据可能没有ground truth）
        if target_dir is None and input_dir is None:
            # 仅当使用默认仿真数据时，自动设置target
            self.target_source_dir = PROJECT_ROOT / "data_simu/physics_method/background_with_light_events_test"
        elif target_dir is not None:
            self.target_source_dir = Path(target_dir)
        else:
            self.target_source_dir = None

        self.output_base = Path(output_base)
        self.test_mode = test_mode

        # 创建输出目录结构（只保留需要的方法）
        self.input_dir = self.output_base / "input"
        self.target_dir = self.output_base / "target"
        self.inputpfda_dir = self.output_base / "inputpfda"
        self.inputpfdb_dir = self.output_base / "inputpfdb"
        self.output_full_dir = self.output_base / "output_full"
        self.output_simple_dir = self.output_base / "output_simple"
        self.outputbaseline_dir = self.output_base / "outputbaseline"
        self.inputefr_dir = self.output_base / "inputefr"
        self.visualize_dir = self.output_base / "visualize"

        # UNet checkpoint配置
        checkpoint_base = PROJECT_ROOT / "checkpoints"
        self.unet_checkpoints = {
            'simple': str(checkpoint_base / 'event_voxel_deflare_simple' / 'checkpoint_epoch_0031_iter_040000.pth'),
            'full': str(checkpoint_base / 'event_voxel_deflare_full' / 'checkpoint_epoch_0031_iter_040000.pth'),
        }

        # 创建所有必要的目录
        for dir_path in [self.input_dir, self.target_dir, self.inputpfda_dir, self.inputpfdb_dir,
                         self.output_full_dir, self.output_simple_dir,
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

        print(f"🚀 主实验数据集生成器初始化完成")
        print(f"📂 输入源: {self.input_source_dir}")
        if self.target_source_dir:
            print(f"📂 目标源: {self.target_source_dir}")
        else:
            print(f"📂 目标源: 无 (仅处理input)")
        print(f"📂 输出基础目录: {self.output_base}")
        if test_mode:
            print(f"🧪 测试模式: 已启用")

    def copy_input_target_files(self, num_samples: int = None) -> List[Path]:
        """
        复制input和target文件到输出目录

        Args:
            num_samples: 限制处理的文件数量（测试用）

        Returns:
            复制后的input文件路径列表
        """
        print(f"\n📋 Step 1: 复制输入和目标文件")
        print("=" * 80)

        # 获取所有H5文件
        input_files = sorted(list(self.input_source_dir.glob("*.h5")))

        if not input_files:
            raise FileNotFoundError(f"未找到输入文件: {self.input_source_dir}")

        # Target文件可选
        target_files = []
        if self.target_source_dir and self.target_source_dir.exists():
            target_files = sorted(list(self.target_source_dir.glob("*.h5")))
            if not target_files:
                print(f"⚠️  警告: 未找到目标文件: {self.target_source_dir}")
        else:
            print(f"⚠️  无target目录，仅处理input")

        # 限制样本数量
        if num_samples:
            input_files = input_files[:num_samples]
            print(f"🧪 测试模式: 只处理前 {num_samples} 个文件")

        print(f"📄 找到 {len(input_files)} 个输入文件")

        # 复制input文件
        copied_input_files = []
        for input_file in input_files:
            dest_file = self.input_dir / input_file.name
            if not dest_file.exists():
                shutil.copy2(input_file, dest_file)
                print(f"  ✅ 复制input: {input_file.name}")
            else:
                print(f"  ⏭️  跳过input: {input_file.name} (已存在)")
            copied_input_files.append(dest_file)

        # 复制target文件（匹配的）
        target_copied = 0
        if target_files:
            for input_file in input_files:
                # 查找匹配的target文件（bg_flare → bg_light）
                matching_target = None
                expected_target_name = input_file.name.replace('_bg_flare.h5', '_bg_light.h5')
                for target_file in target_files:
                    if target_file.name == expected_target_name:
                        matching_target = target_file
                        break

                if matching_target:
                    dest_file = self.target_dir / matching_target.name
                    if not dest_file.exists():
                        shutil.copy2(matching_target, dest_file)
                        print(f"  ✅ 复制target: {matching_target.name}")
                    else:
                        print(f"  ⏭️  跳过target: {matching_target.name} (已存在)")
                    target_copied += 1
                else:
                    print(f"  ⚠️  未找到匹配的target: {input_file.name}")

        print(f"\n📊 复制完成: {len(copied_input_files)} input, {target_copied} target")
        return copied_input_files

    def save_h5_events(self, events, output_path: Path):
        """保存事件到H5文件（标准格式）"""
        import h5py
        import numpy as np

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

    def run_unet_inference(self, input_h5: Path, output_h5: Path, checkpoint_path: str, variant_name: str = "standard"):
        """运行UNet3D推理（复用generate_dsec_dataset.py的实现）"""
        import yaml
        config_path = PROJECT_ROOT / "configs" / "inference_config.yaml"

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

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
                success = False
        except subprocess.TimeoutExpired:
            print(f"    ❌ UNet3D ({variant_name}) timeout")
            success = False
        except Exception as e:
            print(f"    ❌ UNet3D ({variant_name}) exception: {e}")
            success = False
        finally:
            if temp_config_path.exists():
                temp_config_path.unlink()

        return success

    def run_all_unet_variants(self, input_h5: Path, filename: str) -> dict:
        """运行所有UNet权重变体（只运行simple和full）"""
        outputs = {}
        variants = [
            ('full', self.output_full_dir),
            ('simple', self.output_simple_dir),
        ]

        for variant_name, output_dir in variants:
            output_h5 = output_dir / filename
            checkpoint_path = self.unet_checkpoints[variant_name]

            if not Path(checkpoint_path).exists():
                print(f"    ⚠️  UNet3D ({variant_name}) skipped - checkpoint not found")
                continue

            success = self.run_unet_inference(input_h5, output_h5, checkpoint_path, variant_name)

            if success and output_h5.exists():
                outputs[variant_name] = output_h5

        print(f"    📊 UNet variants completed: {len(outputs)}/2")
        return outputs

    def run_baseline_processing(self, input_h5: Path, output_h5: Path):
        """运行Baseline（编解码only）处理"""
        print(f"  🔧 Running Baseline processing...")
        try:
            events_np = load_h5_events(str(input_h5))
            voxel = events_to_voxel(events_np, num_bins=8, sensor_size=(480, 640), fixed_duration_us=100000)
            output_events = voxel_to_events(voxel, total_duration=100000, sensor_size=(480, 640))
            self.save_h5_events(output_events, output_h5)
            print(f"  ✅ Baseline processing completed")
        except Exception as e:
            print(f"  ❌ Baseline processing failed: {e}")

    def process_single_file(self, input_h5: Path, filename: str) -> bool:
        """
        处理单个H5文件（所有方法）

        Args:
            input_h5: 输入H5文件路径
            filename: 文件名

        Returns:
            是否成功处理
        """
        print(f"\n📁 Processing: {filename}")
        print("-" * 80)

        try:
            # Step 1: UNet3D (simple和full)
            print(f"  🧠 Running UNet variants (simple, full)...")
            unet_outputs = self.run_all_unet_variants(input_h5, filename)

            # Step 2: PFD-A
            print(f"  🔧 Running PFD-A processing...")
            pfda_h5 = self.inputpfda_dir / filename
            self.pfd_processor_a.process_single_file(input_h5, pfda_h5, file_idx=0)

            # Step 3: PFD-B
            print(f"  🔧 Running PFD-B processing...")
            pfdb_h5 = self.inputpfdb_dir / filename
            self.pfd_processor_b.process_single_file(input_h5, pfdb_h5, file_idx=0)

            # Step 4: EFR
            print(f"  🔧 Running EFR processing...")
            efr_h5 = self.inputefr_dir / filename
            self.efr_processor.process_single_file(input_h5, efr_h5, file_idx=0)

            # Step 5: Baseline
            baseline_h5 = self.outputbaseline_dir / filename
            self.run_baseline_processing(input_h5, baseline_h5)

            # Step 6: 生成可视化
            print(f"  🎬 Generating visualizations...")
            target_h5 = self.target_dir / filename
            self.generate_visualizations(
                filename, input_h5, target_h5, unet_outputs,
                pfda_h5, pfdb_h5, efr_h5, baseline_h5
            )

            print(f"  ✅ File completed: {filename}")
            return True

        except Exception as e:
            print(f"  ❌ File failed: {filename} - {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_visualizations(self, base_filename: str,
                               input_h5: Path,
                               target_h5: Path,
                               unet_outputs: dict,
                               pfda_h5: Path,
                               pfdb_h5: Path,
                               efr_h5: Path,
                               baseline_h5: Path):
        """生成所有方法的可视化"""
        vis_subdir = self.visualize_dir / Path(base_filename).stem
        vis_subdir.mkdir(parents=True, exist_ok=True)

        print(f"    🎬 Generating visualizations to: {vis_subdir.name}/")

        # 定义所有需要可视化的文件
        vis_tasks = [
            (input_h5, "input"),
        ]

        # Target可选
        if target_h5.exists():
            vis_tasks.append((target_h5, "target"))

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

    def generate_all(self, num_samples: int = None):
        """
        生成完整的主实验数据集

        Args:
            num_samples: 限制处理的文件数量（测试用）
        """
        print(f"\n🚀 开始生成主实验数据集")
        print("=" * 80)
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # Step 1: 复制input和target文件
        input_files = self.copy_input_target_files(num_samples)

        if not input_files:
            print("❌ 没有文件需要处理")
            return

        # Step 2: 处理每个文件
        print(f"\n🔄 Step 2: 处理所有文件（{len(input_files)} 个）")
        print("=" * 80)

        success_count = 0
        for idx, input_file in enumerate(input_files, 1):
            print(f"\n[{idx}/{len(input_files)}]")
            if self.process_single_file(input_file, input_file.name):
                success_count += 1

        # Final summary
        print("\n" + "=" * 80)
        print(f"🎉 主实验数据集生成完成!")
        print(f"📊 处理结果: {success_count}/{len(input_files)} 文件成功")
        print(f"📂 输出目录: {self.output_base}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        # 输出目录结构说明
        print(f"\n📁 输出目录结构:")
        print(f"  • input/               含炫光数据")
        if self.target_source_dir:
            print(f"  • target/              目标去炫光数据")
        print(f"  • output_full/         UNet3D full权重结果")
        print(f"  • output_simple/       UNet3D simple权重结果")
        print(f"  • inputpfda/           PFD-A结果")
        print(f"  • inputpfdb/           PFD-B结果")
        print(f"  • inputefr/            EFR结果")
        print(f"  • outputbaseline/      Baseline结果")
        print(f"  • visualize/           所有方法的可视化视频")


def main():
    parser = argparse.ArgumentParser(
        description="主实验数据集生成器 - 统一处理仿真和真实数据集"
    )
    parser.add_argument("--input_dir", help="输入目录 (默认: data_simu/physics_method/background_with_flare_events_test)")
    parser.add_argument("--target_dir", help="目标目录 (可选，默认仿真数据使用background_with_light_events_test)")
    parser.add_argument("--output_base", default="Main_data", help="输出基础目录 (默认Main_data，仿真自动改为MainSimu_data)")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--num_samples", type=int, help="限制处理的文件数量（测试用）")

    args = parser.parse_args()

    generator = MainDatasetGenerator(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        output_base=args.output_base,
        test_mode=args.test
    )

    # 生成完整数据集
    generator.generate_all(num_samples=args.num_samples)


if __name__ == "__main__":
    main()
