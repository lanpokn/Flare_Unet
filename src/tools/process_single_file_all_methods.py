#!/usr/bin/env python3
"""
单文件全方法处理工具 - 任意长度文件支持

基于Linus哲学：
- 数据结构正确: 单个H5文件 → 多种方法 → 同目录带后缀输出
- 消除特殊情况: 统一处理任意长度文件（不限100ms）
- 实用主义: 100%复用现有处理器，零修改现有代码

功能：
1. 读取单个H5文件（任意长度）
2. 运行所有处理方法：
   - 多个UNet权重变体（通过inference_single.py内存安全处理）
   - PFD-A (复用BatchPFDProcessor)
   - PFD-B (复用BatchPFDProcessor)
   - EFR (复用BatchEFRProcessor)
   - Baseline (新实现，支持任意长度)
3. 输出到同目录，带后缀标识方法

输出文件命名:
- input.h5 → input_unet_full.h5, input_unet_simple.h5, input_unet_nolight.h5,
             input_unet_physics.h5, input_unet_physics_noRandom_method.h5,
             input_unet_physics_noRandom_noTen_method.h5, input_unet_simple_timeRandom_method.h5,
             input_unet_full_old.h5, input_unet_simple_old.h5,
             input_pfda.h5, input_pfdb.h5, input_efr.h5, input_baseline.h5

使用方法:
    # 处理单个文件（所有方法，默认使用所有9个UNet权重）
    python src/tools/process_single_file_all_methods.py \
      --input "E:\BaiduSyncdisk\2025\event_flick_flare\experiments\3D_reconstruction\datasets\lego2\events_h5\lego2_sequence_new.h5"

    # 只处理指定方法
    python src/tools/process_single_file_all_methods.py \
      --input "path/to/file.h5" \
      --methods pfda pfdb efr baseline

    # 指定特定UNet权重（可选9个：full, simple, nolight, physics,
    #                       physics_noRandom_method, physics_noRandom_noTen_method,
    #                       simple_timeRandom_method, full_old, simple_old）
    python src/tools/process_single_file_all_methods.py \
      --input "path/to/file.h5" \
      --unet_checkpoints full simple physics_noRandom_noTen_method
"""

import argparse
import sys
import subprocess
import h5py
import numpy as np
import yaml
import tempfile
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入现有处理器（100%复用）
sys.path.append(str(PROJECT_ROOT / 'ext' / 'PFD'))
sys.path.append(str(PROJECT_ROOT / 'ext' / 'EFR-main'))
from batch_pfd_processor import BatchPFDProcessor
from batch_efr_processor import BatchEFRProcessor

# 导入项目模块
from src.data_processing.encode import load_h5_events, events_to_voxel
from src.data_processing.decode import voxel_to_events


class SingleFileAllMethodsProcessor:
    """单文件全方法处理器 - 支持任意长度文件"""

    def __init__(self, input_file: str, methods: Optional[List[str]] = None,
                 unet_checkpoints: Optional[List[str]] = None):
        """
        Args:
            input_file: 输入H5文件路径（任意长度）
            methods: 指定处理方法列表（None=全部）
            unet_checkpoints: 指定UNet权重变体（None=默认使用所有9个可用权重）
        """
        self.input_file = Path(input_file)

        # 转换Windows路径到WSL路径
        self.input_file_wsl = self._convert_to_wsl_path(self.input_file)

        if not self.input_file_wsl.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # 输出目录（同目录）
        self.output_dir = self.input_file_wsl.parent
        self.base_name = self.input_file_wsl.stem

        # 可用方法
        self.available_methods = ['unet', 'pfda', 'pfdb', 'efr', 'baseline']
        self.methods = methods if methods else self.available_methods

        # UNet权重配置（默认：full + simple）
        checkpoint_base = PROJECT_ROOT / "checkpoints"
        checkpoint_old_base = PROJECT_ROOT / "checkpoints_old"

        # 所有可用的UNet权重（新版40000 + 旧版76250）
        # ⭐ 注意：physics_noRandom_method目录名没有event_voxel_deflare_前缀
        all_checkpoints = {
            # 新版40000权重（7个变体）
            'full': str(checkpoint_base / 'event_voxel_deflare_full' / 'checkpoint_epoch_0031_iter_040000.pth'),
            'simple': str(checkpoint_base / 'event_voxel_deflare_simple' / 'checkpoint_epoch_0031_iter_040000.pth'),
            'nolight': str(checkpoint_base / 'event_voxel_deflare_nolight' / 'checkpoint_epoch_0031_iter_040000.pth'),
            'physics': str(checkpoint_base / 'event_voxel_deflare_physics' / 'checkpoint_epoch_0031_iter_040000.pth'),
            'physics_noRandom_method': str(checkpoint_base / 'physics_noRandom_method' / 'checkpoint_epoch_0031_iter_040000.pth'),  # ✅ 直接目录名
            'physics_noRandom_noTen_method': str(checkpoint_base / 'event_voxel_deflare_physics_noRandom_noTen_method' / 'checkpoint_epoch_0031_iter_040000.pth'),
            'simple_timeRandom_method': str(checkpoint_base / 'event_voxel_deflare_simple_timeRandom_method' / 'checkpoint_epoch_0031_iter_040000.pth'),
            # 旧版76250权重（2个变体）
            'full_old': str(checkpoint_old_base / 'event_voxel_deflare_full' / 'checkpoint_epoch_0032_iter_076250.pth'),
            'simple_old': str(checkpoint_old_base / 'event_voxel_deflare_simple' / 'checkpoint_epoch_0027_iter_076250.pth'),
        }

        # 过滤可用的checkpoint
        if unet_checkpoints:
            self.unet_checkpoints = {k: v for k, v in all_checkpoints.items()
                                    if k in unet_checkpoints and Path(v).exists()}
        else:
            # 默认使用所有可用的权重（9个变体）
            self.unet_checkpoints = {k: v for k, v in all_checkpoints.items()
                                    if Path(v).exists()}

        # 初始化处理器（复用现有）
        self.pfd_processor_a = BatchPFDProcessor(debug=False)
        self.pfd_processor_a.pfds_params['score_select'] = 1  # PFD-A

        self.pfd_processor_b = BatchPFDProcessor(debug=False)
        self.pfd_processor_b.pfds_params['score_select'] = 0  # PFD-B

        self.efr_processor = BatchEFRProcessor(debug=False)

        print(f"🎯 单文件全方法处理器初始化")
        print(f"📂 输入文件: {self.input_file_wsl.name}")
        print(f"📊 处理方法: {', '.join(self.methods)}")
        if 'unet' in self.methods:
            print(f"🧠 UNet权重: {', '.join(self.unet_checkpoints.keys())}")

    def _convert_to_wsl_path(self, windows_path: Path) -> Path:
        """转换Windows路径到WSL路径"""
        path_str = str(windows_path)
        if path_str.startswith(('E:', 'C:', 'D:', 'F:', 'G:')):
            drive = path_str[0].lower()
            rest = path_str[2:].replace('\\', '/')
            wsl_path = f'/mnt/{drive}{rest}'
            return Path(wsl_path)
        return windows_path

    def _save_h5_events(self, events: np.ndarray, output_path: Path):
        """保存事件到H5文件（标准格式，支持任意长度）"""
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

    def process_unet(self) -> Dict[str, Path]:
        """
        处理UNet所有权重变体（复用main.py inference模式，支持任意长度）

        Returns:
            {variant_name: output_path}
        """
        if 'unet' not in self.methods:
            return {}

        print(f"\n🧠 Running UNet inference ({len(self.unet_checkpoints)} variants)...")
        outputs = {}

        for variant, checkpoint_path in self.unet_checkpoints.items():
            output_file = self.output_dir / f"{self.base_name}_unet_{variant}.h5"

            if output_file.exists():
                print(f"  ⏭️  UNet {variant} skipped (output exists)")
                outputs[variant] = output_file
                continue

            if not Path(checkpoint_path).exists():
                print(f"  ⚠️  UNet {variant} skipped (checkpoint not found)")
                continue

            temp_config_path = None
            try:
                # 创建临时配置文件，指定checkpoint路径
                config_path = PROJECT_ROOT / "configs" / "inference_config.yaml"
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                # 修改checkpoint路径
                config['model']['path'] = checkpoint_path

                # 写入临时配置
                temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                                         delete=False, dir=PROJECT_ROOT / "configs")
                yaml.dump(config, temp_config)
                temp_config.close()
                temp_config_path = temp_config.name

                # 使用main.py inference模式（支持任意长度文件）
                cmd = [
                    sys.executable, "main.py", "inference",
                    "--config", temp_config_path,
                    "--input", str(self.input_file_wsl),
                    "--output", str(output_file)
                ]

                print(f"  🔧 Processing UNet {variant}...")
                result = subprocess.run(cmd, capture_output=True, text=True,
                                      cwd=PROJECT_ROOT, timeout=600)

                if result.returncode == 0 and output_file.exists():
                    print(f"  ✅ UNet {variant} completed")
                    outputs[variant] = output_file
                else:
                    print(f"  ❌ UNet {variant} failed")
                    if result.stderr:
                        print(f"     Error: {result.stderr[:200]}")

            except subprocess.TimeoutExpired:
                print(f"  ❌ UNet {variant} timeout (10 min)")
            except Exception as e:
                print(f"  ❌ UNet {variant} error: {e}")
            finally:
                # 清理临时配置文件
                if temp_config_path and Path(temp_config_path).exists():
                    try:
                        Path(temp_config_path).unlink()
                    except:
                        pass  # 忽略清理失败

        return outputs

    def process_pfda(self) -> Optional[Path]:
        """处理PFD-A（100%复用BatchPFDProcessor）"""
        if 'pfda' not in self.methods:
            return None

        output_file = self.output_dir / f"{self.base_name}_pfda.h5"

        if output_file.exists():
            print(f"⏭️  PFD-A skipped (output exists)")
            return output_file

        print(f"🔧 Running PFD-A...")
        try:
            success = self.pfd_processor_a.process_single_file(
                self.input_file_wsl, output_file, file_idx=0
            )
            if success:
                print(f"✅ PFD-A completed")
                return output_file
            else:
                print(f"❌ PFD-A failed")
                return None
        except Exception as e:
            print(f"❌ PFD-A error: {e}")
            return None

    def process_pfdb(self) -> Optional[Path]:
        """处理PFD-B（100%复用BatchPFDProcessor）"""
        if 'pfdb' not in self.methods:
            return None

        output_file = self.output_dir / f"{self.base_name}_pfdb.h5"

        if output_file.exists():
            print(f"⏭️  PFD-B skipped (output exists)")
            return output_file

        print(f"🔧 Running PFD-B...")
        try:
            success = self.pfd_processor_b.process_single_file(
                self.input_file_wsl, output_file, file_idx=0
            )
            if success:
                print(f"✅ PFD-B completed")
                return output_file
            else:
                print(f"❌ PFD-B failed")
                return None
        except Exception as e:
            print(f"❌ PFD-B error: {e}")
            return None

    def process_efr(self) -> Optional[Path]:
        """处理EFR（100%复用BatchEFRProcessor）"""
        if 'efr' not in self.methods:
            return None

        output_file = self.output_dir / f"{self.base_name}_efr.h5"

        if output_file.exists():
            print(f"⏭️  EFR skipped (output exists)")
            return output_file

        print(f"🔧 Running EFR...")
        try:
            success = self.efr_processor.process_single_file(
                self.input_file_wsl, output_file, file_idx=0
            )
            if success:
                print(f"✅ EFR completed")
                return output_file
            else:
                print(f"❌ EFR failed")
                return None
        except Exception as e:
            print(f"❌ EFR error: {e}")
            return None

    def process_baseline(self) -> Optional[Path]:
        """
        处理Baseline（编解码only，新实现支持任意长度）

        关键修复：动态计算文件实际时长，不硬编码100ms
        """
        if 'baseline' not in self.methods:
            return None

        output_file = self.output_dir / f"{self.base_name}_baseline.h5"

        if output_file.exists():
            print(f"⏭️  Baseline skipped (output exists)")
            return output_file

        print(f"🔧 Running Baseline...")
        try:
            # 加载事件
            events_np = load_h5_events(str(self.input_file_wsl))

            # ⭐ 动态计算实际时长（不硬编码100ms）
            actual_duration_us = int(events_np[:, 0].max() - events_np[:, 0].min())

            # Encode → Decode（支持任意长度）
            voxel = events_to_voxel(
                events_np,
                num_bins=8,
                sensor_size=(480, 640),
                fixed_duration_us=actual_duration_us  # ⭐使用实际时长
            )

            output_events = voxel_to_events(
                voxel,
                total_duration=actual_duration_us,  # ⭐使用实际时长
                sensor_size=(480, 640)
            )

            # 保存
            self._save_h5_events(output_events, output_file)

            print(f"✅ Baseline completed (duration: {actual_duration_us/1000:.1f}ms)")
            return output_file

        except Exception as e:
            print(f"❌ Baseline error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_all(self) -> Dict[str, any]:
        """
        运行所有处理方法

        Returns:
            处理结果字典
        """
        print(f"\n{'='*80}")
        print(f"🚀 开始处理: {self.input_file_wsl.name}")
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        results = {}

        # 1. UNet所有变体
        if 'unet' in self.methods:
            results['unet'] = self.process_unet()

        # 2. PFD-A
        if 'pfda' in self.methods:
            results['pfda'] = self.process_pfda()

        # 3. PFD-B
        if 'pfdb' in self.methods:
            results['pfdb'] = self.process_pfdb()

        # 4. EFR
        if 'efr' in self.methods:
            results['efr'] = self.process_efr()

        # 5. Baseline
        if 'baseline' in self.methods:
            results['baseline'] = self.process_baseline()

        # 最终总结
        print(f"\n{'='*80}")
        print(f"🎉 处理完成!")
        print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 输出目录: {self.output_dir}")
        print(f"\n📊 输出文件:")

        if 'unet' in results and results['unet']:
            for variant, path in results['unet'].items():
                if path and path.exists():
                    print(f"  ✅ {path.name}")

        for method in ['pfda', 'pfdb', 'efr', 'baseline']:
            if method in results and results[method] and results[method].exists():
                print(f"  ✅ {results[method].name}")

        print(f"{'='*80}\n")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="单文件全方法处理工具 - 支持任意长度H5文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件（所有方法）
  python src/tools/process_single_file_all_methods.py
    --input "E:/path/to/file.h5"

  # 只处理特定方法
  python src/tools/process_single_file_all_methods.py
    --input "file.h5" --methods pfda pfdb efr baseline

  # 指定UNet权重
  python src/tools/process_single_file_all_methods.py
    --input "file.h5" --unet_checkpoints full simple physics_noRandom_noTen_method
        """
    )

    parser.add_argument('--input', required=True,
                       help='输入H5文件路径（支持Windows和WSL路径）')
    parser.add_argument('--methods', nargs='+',
                       choices=['unet', 'pfda', 'pfdb', 'efr', 'baseline'],
                       help='指定处理方法（默认：全部）')
    parser.add_argument('--unet_checkpoints', nargs='+',
                       choices=['full', 'simple', 'nolight', 'physics',
                               'physics_noRandom_method', 'physics_noRandom_noTen_method',
                               'simple_timeRandom_method', 'full_old', 'simple_old'],
                       help='指定UNet权重变体（默认：所有9个可用权重）')

    args = parser.parse_args()

    try:
        processor = SingleFileAllMethodsProcessor(
            input_file=args.input,
            methods=args.methods,
            unet_checkpoints=args.unet_checkpoints
        )

        results = processor.process_all()

        # 检查是否有失败
        failed = []
        if 'unet' in results and isinstance(results['unet'], dict):
            for variant in processor.unet_checkpoints.keys():
                if variant not in results['unet'] or not results['unet'][variant]:
                    failed.append(f"unet_{variant}")

        for method in ['pfda', 'pfdb', 'efr', 'baseline']:
            if method in results and (not results[method] or not results[method].exists()):
                failed.append(method)

        if failed:
            print(f"\n⚠️  部分方法失败: {', '.join(failed)}")
            return 1
        else:
            print(f"\n✅ 所有方法处理成功!")
            return 0

    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
