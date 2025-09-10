"""
EVK4 Video Generation - 批量生成EVK4 input/target成对数据可视化视频

基于Linus哲学：
- 数据结构正确: 扫描EVK4/input和EVK4/target → 调用event_video_generator → 成对输出
- 消除特殊情况: 统一处理input和target文件
- 实用主义: 解决EVK4成对数据可视化和对齐检查的实际需求

Usage:
    python src/tools/evk4_video_generation.py
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Tuple

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.event_video_generator import EventVideoGenerator


class EVK4VideoGenerator:
    """EVK4批量视频生成器 - 专门处理EVK4 input/target成对数据"""
    
    def __init__(self, evk4_dir: str = None, output_base_dir: str = "debug_output"):
        """
        Args:
            evk4_dir: EVK4目录路径
            output_base_dir: 输出基础目录
        """
        if evk4_dir is None:
            self.evk4_dir = Path(PROJECT_ROOT) / "EVK4"
        else:
            self.evk4_dir = Path(evk4_dir)
            
        self.input_dir = self.evk4_dir / "input"
        self.target_dir = self.evk4_dir / "target"
        self.output_base_dir = Path(output_base_dir)
        
        # 创建视频生成器实例
        self.video_generator = EventVideoGenerator(
            sensor_size=(480, 640),
            frame_duration_ms=2.5,
            fps=10
        )
    
    def scan_paired_h5_files(self) -> List[Tuple[Path, Path]]:
        """扫描EVK4 input/target成对H5文件"""
        paired_files = []
        
        print(f"🔍 Scanning paired files in EVK4:")
        print(f"  📁 Input: {self.input_dir}")
        print(f"  📁 Target: {self.target_dir}")
        
        if not self.input_dir.exists():
            print(f"❌ Input directory not found: {self.input_dir}")
            return paired_files
            
        if not self.target_dir.exists():
            print(f"❌ Target directory not found: {self.target_dir}")
            return paired_files
        
        # 获取input文件
        input_files = sorted(list(self.input_dir.glob("*.h5")))
        target_files = sorted(list(self.target_dir.glob("*.h5")))
        
        print(f"📊 Found {len(input_files)} input files, {len(target_files)} target files")
        
        # 匹配成对文件（按文件名匹配或按索引匹配）
        for input_file in input_files:
            # 尝试按文件名匹配
            corresponding_target = self.target_dir / input_file.name
            if corresponding_target.exists():
                paired_files.append((input_file, corresponding_target))
                print(f"✅ Paired: {input_file.name} ↔ {corresponding_target.name}")
            else:
                print(f"⚠️  No target found for: {input_file.name}")
        
        # 如果按文件名匹配失败，尝试按索引匹配
        if not paired_files and len(input_files) == len(target_files):
            print("🔄 Falling back to index-based pairing")
            for i, (input_file, target_file) in enumerate(zip(input_files, target_files)):
                paired_files.append((input_file, target_file))
                print(f"✅ Paired by index [{i}]: {input_file.name} ↔ {target_file.name}")
        
        return paired_files
    
    def create_output_structure(self) -> Path:
        """创建EVK4输出目录结构"""
        output_dir = self.output_base_dir / "EVK4"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def generate_paired_videos(self, paired_files: List[Tuple[Path, Path]]):
        """为成对文件生成视频"""
        print(f"\n🎬 Processing {len(paired_files)} paired files")
        
        # 创建输出目录
        output_dir = self.create_output_structure()
        
        success_count = 0
        for i, (input_file, target_file) in enumerate(paired_files, 1):
            try:
                print(f"\n📹 [{i}/{len(paired_files)}] Processing pair:")
                print(f"  📥 Input: {input_file.name}")
                print(f"  🎯 Target: {target_file.name}")
                
                # 生成输出文件名
                base_name = input_file.stem
                input_video = output_dir / f"{base_name}_input.mp4"
                target_video = output_dir / f"{base_name}_target.mp4"
                
                # 生成input视频
                print(f"  🎬 Generating input video...")
                self.video_generator.process_h5_file(
                    str(input_file), 
                    str(input_video)
                )
                
                # 生成target视频
                print(f"  🎬 Generating target video...")
                self.video_generator.process_h5_file(
                    str(target_file), 
                    str(target_video)
                )
                
                success_count += 1
                print(f"✅ [{i}/{len(paired_files)}] Success:")
                print(f"  📥 {input_video.name}")
                print(f"  🎯 {target_video.name}")
                
            except Exception as e:
                print(f"❌ [{i}/{len(paired_files)}] Failed: {input_file.name} - {str(e)}")
                continue
        
        print(f"\n📊 EVK4 paired processing summary: {success_count}/{len(paired_files)} pairs completed")
        return success_count
    
    def generate_all_videos(self):
        """生成所有EVK4成对视频"""
        print("🚀 EVK4 Paired Video Generation Started")
        print(f"📂 Source: {self.evk4_dir}")
        print(f"📂 Output: {self.output_base_dir}/EVK4")
        print("-" * 80)
        
        # 扫描成对文件
        paired_files = self.scan_paired_h5_files()
        
        if not paired_files:
            print("❌ No paired H5 files found in EVK4 directory")
            return
        
        print(f"\n🎯 Found {len(paired_files)} paired files for processing")
        
        # 处理成对文件
        success_count = self.generate_paired_videos(paired_files)
        
        # 最终统计
        print("\n" + "=" * 80)
        print("🎉 EVK4 Paired Video Generation Completed!")
        print(f"📊 Overall Summary:")
        print(f"  • Processed pairs: {len(paired_files)}")
        print(f"  • Successful pairs: {success_count}")
        print(f"  • Total videos generated: {success_count * 2}")
        print(f"  • Success rate: {success_count/len(paired_files)*100:.1f}%")
        print(f"  • Output location: {self.output_base_dir}/EVK4")
        print("=" * 80)
    
    def list_available_files(self):
        """列出可用的EVK4文件"""
        paired_files = self.scan_paired_h5_files()
        
        print("Available EVK4 paired files:")
        if not paired_files:
            print("  No paired files found")
            return
            
        for i, (input_file, target_file) in enumerate(paired_files, 1):
            print(f"  {i}. {input_file.name} ↔ {target_file.name}")


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="Batch generate paired visualization videos from EVK4 input/target H5 data")
    
    parser.add_argument("--evk4_dir", default=None,
                       help="EVK4 directory path (default: auto-detect)")
    parser.add_argument("--output_dir", default="debug_output",
                       help="Output base directory (default: debug_output)")
    parser.add_argument("--list", action='store_true',
                       help="List available paired files and exit")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = EVK4VideoGenerator(
        evk4_dir=args.evk4_dir,
        output_base_dir=args.output_dir
    )
    
    # 列出文件选项
    if args.list:
        generator.list_available_files()
        return
    
    # 生成视频
    generator.generate_all_videos()


if __name__ == "__main__":
    main()