#!/usr/bin/env python3
"""
EFR专用重新生成脚本 - 仅重新生成EFR相关结果

基于Linus哲学：
- 数据结构正确: 遍历现有input文件，只重新生成EFR
- 消除特殊情况: 不依赖generate_main_dataset.py，直接调用EFR处理器
- 实用主义: 解决EFR更新后的重新生成需求，保护已筛选的good/bad数据

功能：
1. 遍历现有input目录中的H5文件（不新增不删除input）
2. 删除对应的旧EFR输出（inputefr/*.h5 + visualize/*/efr_output.mp4）
3. 重新运行EFR处理器生成新输出
4. 重新生成EFR视频

适用场景：
- MainSimu_data/good/ (已筛选，35个文件)
- MainSimu_data/bad/  (已筛选)
- MainReal_data/     (标准结构)

Usage:
    # 重新生成MainSimu_data/good的EFR
    python regenerate_efr_only.py --dir MainSimu_data/good

    # 重新生成MainReal_data的EFR
    python regenerate_efr_only.py --dir MainReal_data

    # 测试模式：只处理前1个文件
    python regenerate_efr_only.py --dir MainSimu_data/good --test --num_samples 1

    # 批量处理多个目录
    python regenerate_efr_only.py --dir MainSimu_data/good --dir MainReal_data
"""

import argparse
import sys
from pathlib import Path
from typing import List

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.event_video_generator import EventVideoGenerator

# 导入EFR处理器
sys.path.append(str(PROJECT_ROOT / 'ext' / 'EFR-main'))
from batch_efr_processor import BatchEFRProcessor


class EFRRegenator:
    """EFR专用重新生成器 - 只处理EFR，不影响其他方法"""

    def __init__(self, base_dir: Path, test_mode: bool = False):
        """
        Args:
            base_dir: 基础目录 (如 MainSimu_data/good 或 MainReal_data)
            test_mode: 测试模式
        """
        self.base_dir = Path(base_dir)
        self.test_mode = test_mode

        if not self.base_dir.exists():
            raise FileNotFoundError(f"目录不存在: {self.base_dir}")

        self.input_dir = self.base_dir / "input"
        self.inputefr_dir = self.base_dir / "inputefr"
        self.visualize_dir = self.base_dir / "visualize"

        if not self.input_dir.exists():
            raise FileNotFoundError(f"input目录不存在: {self.input_dir}")

        # 创建inputefr目录（如果不存在）
        self.inputefr_dir.mkdir(parents=True, exist_ok=True)

        # 初始化EFR处理器
        self.efr_processor = BatchEFRProcessor(debug=False)

        # 初始化视频生成器
        self.video_generator = EventVideoGenerator(
            sensor_size=(480, 640),
            frame_duration_ms=2.5,
            fps=10
        )

        print(f"🚀 EFR重新生成器初始化完成")
        print(f"📂 基础目录: {self.base_dir}")
        print(f"📂 Input目录: {self.input_dir}")
        print(f"📂 EFR输出目录: {self.inputefr_dir}")

    def regenerate_all(self, num_samples: int = None):
        """
        重新生成所有EFR输出

        Args:
            num_samples: 限制处理的文件数量（测试用）
        """
        print(f"\n🔄 开始重新生成EFR输出")
        print("=" * 80)

        # 获取所有input文件
        input_files = sorted(list(self.input_dir.glob("*.h5")))

        if not input_files:
            print(f"❌ 未找到输入文件: {self.input_dir}")
            return

        # 限制样本数量
        if num_samples:
            input_files = input_files[:num_samples]
            print(f"🧪 测试模式: 只处理前 {num_samples} 个文件")

        print(f"📄 找到 {len(input_files)} 个输入文件")

        success_count = 0
        for idx, input_file in enumerate(input_files, 1):
            print(f"\n[{idx}/{len(input_files)}] {input_file.name}")
            print("-" * 80)

            # Step 1: 删除旧的EFR H5文件
            efr_h5 = self.inputefr_dir / input_file.name
            if efr_h5.exists():
                efr_h5.unlink()
                print(f"  🗑️  删除旧EFR H5: {efr_h5.name}")

            # Step 2: 删除旧的EFR视频
            if self.visualize_dir.exists():
                vis_subdir = self.visualize_dir / input_file.stem
                efr_video = vis_subdir / "efr_output.mp4"
                if efr_video.exists():
                    efr_video.unlink()
                    print(f"  🗑️  删除旧EFR视频: {vis_subdir.name}/efr_output.mp4")

            # Step 3: 运行EFR处理
            try:
                print(f"  🔧 运行EFR处理...")
                self.efr_processor.process_single_file(input_file, efr_h5, file_idx=idx-1)
                print(f"  ✅ EFR处理完成")

                # Step 4: 重新生成EFR视频
                if efr_h5.exists():
                    vis_subdir = self.visualize_dir / input_file.stem
                    vis_subdir.mkdir(parents=True, exist_ok=True)
                    efr_video = vis_subdir / "efr_output.mp4"

                    print(f"  🎬 生成EFR视频...")
                    try:
                        self.video_generator.process_h5_file(str(efr_h5), str(efr_video))
                        print(f"  ✅ EFR视频生成完成")
                    except Exception as e:
                        print(f"  ❌ EFR视频生成失败: {e}")

                success_count += 1

            except Exception as e:
                print(f"  ❌ 处理失败: {e}")
                import traceback
                traceback.print_exc()

        # 最终总结
        print("\n" + "=" * 80)
        print(f"🎉 EFR重新生成完成!")
        print(f"📊 处理结果: {success_count}/{len(input_files)} 文件成功")
        print(f"📂 输出目录: {self.inputefr_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="EFR专用重新生成脚本 - 仅重新生成EFR相关结果"
    )
    parser.add_argument(
        "--dir",
        action="append",
        required=True,
        help="要处理的目录 (可多次指定，如 MainSimu_data/good, MainReal_data)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="测试模式"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        help="限制处理的文件数量（测试用）"
    )

    args = parser.parse_args()

    print(f"🎯 要处理的目录: {args.dir}")
    if args.test:
        print(f"🧪 测试模式: 启用")
    if args.num_samples:
        print(f"📊 限制样本数: {args.num_samples}")

    # 处理每个目录
    for dir_path in args.dir:
        print(f"\n{'='*80}")
        print(f"📂 处理目录: {dir_path}")
        print(f"{'='*80}")

        try:
            regenerator = EFRRegenator(
                base_dir=dir_path,
                test_mode=args.test
            )
            regenerator.regenerate_all(num_samples=args.num_samples)
        except Exception as e:
            print(f"❌ 目录处理失败: {dir_path} - {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
