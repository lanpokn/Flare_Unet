"""
EVK4 Complete Visualization - 批量生成EVK4所有处理结果的可视化视频

基于Linus哲学：
- 数据结构正确: 扫描EVK4下所有子目录 → 匹配文件 → 生成对比视频
- 消除特殊情况: 统一处理input/target/baseline/inputpfds/unet3d
- 实用主义: 解决EVK4多种方法结果对比可视化的实际需求

Usage:
    python src/tools/evk4_complete_visualization.py
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.event_video_generator import EventVideoGenerator


class EVK4CompleteVisualizer:
    """EVK4完整可视化器 - 处理所有方法的结果对比"""
    
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
            
        self.output_base_dir = Path(output_base_dir)
        
        # 预期的子目录列表
        self.expected_dirs = [
            "input",      # 原始含炫光数据
            "target",     # 目标去炫光数据  
            "baseline",   # Baseline结果(编解码only)
            "inputpfds",  # PFD处理结果
            "unet3d"      # UNet3D处理结果
        ]
        
        # 创建视频生成器实例
        self.video_generator = EventVideoGenerator(
            sensor_size=(480, 640),
            frame_duration_ms=2.5,
            fps=10
        )
    
    def scan_all_directories(self) -> Dict[str, List[Path]]:
        """扫描EVK4下所有子目录的H5文件"""
        all_files = {}
        
        print(f"🔍 Scanning EVK4 directories in: {self.evk4_dir}")
        
        if not self.evk4_dir.exists():
            print(f"❌ EVK4 directory not found: {self.evk4_dir}")
            return all_files
        
        # 扫描所有预期的子目录
        for dir_name in self.expected_dirs:
            dir_path = self.evk4_dir / dir_name
            if dir_path.exists():
                h5_files = sorted(list(dir_path.glob("*.h5")))
                all_files[dir_name] = h5_files
                print(f"📁 {dir_name}: {len(h5_files)} files")
            else:
                print(f"⚠️  Directory not found: {dir_name}")
        
        return all_files
    
    def match_files_across_directories(self, all_files: Dict[str, List[Path]]) -> Dict[str, Dict[str, Path]]:
        """匹配所有目录中的同名文件"""
        # 获取所有文件的基础名称
        all_basenames = set()
        for dir_name, files in all_files.items():
            for file in files:
                all_basenames.add(file.name)
        
        print(f"\n🎯 Found {len(all_basenames)} unique file basenames")
        
        # 为每个基础名称匹配所有目录中的文件
        matched_files = {}
        for basename in sorted(all_basenames):
            file_group = {}
            for dir_name in self.expected_dirs:
                if dir_name in all_files:
                    # 查找匹配的文件
                    matching_file = None
                    for file in all_files[dir_name]:
                        if file.name == basename:
                            matching_file = file
                            break
                    if matching_file:
                        file_group[dir_name] = matching_file
            
            if file_group:  # 至少有一个目录包含该文件
                matched_files[basename] = file_group
                dirs_found = list(file_group.keys())
                print(f"  📄 {basename}: {dirs_found}")
        
        return matched_files
    
    def create_output_structure(self) -> Path:
        """创建EVK4完整可视化输出目录"""
        output_dir = self.output_base_dir / "EVK4_complete"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def generate_videos_for_file_group(self, basename: str, file_group: Dict[str, Path], output_dir: Path):
        """为单个文件组生成所有方法的视频"""
        print(f"\n🎬 Processing file group: {basename}")
        
        # 获取基础名称（去除.h5扩展名）
        base_name = Path(basename).stem
        
        success_count = 0
        total_count = 0
        
        # 为每个方法生成视频
        for method_name in self.expected_dirs:
            if method_name in file_group:
                try:
                    input_file = file_group[method_name]
                    output_video = output_dir / f"{base_name}_{method_name}.mp4"
                    
                    print(f"  🎥 Generating {method_name} video...")
                    
                    self.video_generator.process_h5_file(
                        str(input_file), 
                        str(output_video)
                    )
                    
                    success_count += 1
                    print(f"  ✅ {method_name}: {output_video.name}")
                    
                except Exception as e:
                    print(f"  ❌ {method_name}: Failed - {str(e)}")
                
                total_count += 1
            else:
                print(f"  ⏭️  {method_name}: File not found")
        
        print(f"  📊 File group summary: {success_count}/{total_count} videos generated")
        return success_count, total_count
    
    def generate_all_videos(self):
        """生成所有EVK4处理结果的可视化视频"""
        print("🚀 EVK4 Complete Visualization Started")
        print(f"📂 Source: {self.evk4_dir}")
        print(f"📂 Output: {self.output_base_dir}/EVK4_complete")
        print("-" * 80)
        
        # 扫描所有目录
        all_files = self.scan_all_directories()
        
        if not all_files:
            print("❌ No H5 files found in any EVK4 directory")
            return
        
        # 匹配文件
        matched_files = self.match_files_across_directories(all_files)
        
        if not matched_files:
            print("❌ No matching files found across directories")
            return
        
        print(f"\n🎯 Processing {len(matched_files)} file groups")
        
        # 创建输出目录
        output_dir = self.create_output_structure()
        
        # 处理每个文件组
        total_success = 0
        total_videos = 0
        
        for basename, file_group in matched_files.items():
            success_count, video_count = self.generate_videos_for_file_group(
                basename, file_group, output_dir
            )
            total_success += success_count
            total_videos += video_count
        
        # 最终统计
        print("\n" + "=" * 80)
        print("🎉 EVK4 Complete Visualization Completed!")
        print(f"📊 Overall Summary:")
        print(f"  • Processed file groups: {len(matched_files)}")
        print(f"  • Total videos generated: {total_success}/{total_videos}")
        print(f"  • Success rate: {total_success/total_videos*100:.1f}%")
        print(f"  • Output location: {self.output_base_dir}/EVK4_complete")
        print(f"  • Expected methods: {', '.join(self.expected_dirs)}")
        print("=" * 80)
    
    def list_available_files(self):
        """列出可用的EVK4文件和目录"""
        all_files = self.scan_all_directories()
        matched_files = self.match_files_across_directories(all_files)
        
        print("Available EVK4 processing results:")
        print(f"Directories: {', '.join(self.expected_dirs)}")
        print("\nFile groups:")
        if not matched_files:
            print("  No matching file groups found")
            return
            
        for i, (basename, file_group) in enumerate(matched_files.items(), 1):
            methods = list(file_group.keys())
            print(f"  {i}. {basename} → {methods}")
    
    def generate_method_comparison_summary(self):
        """生成方法对比总结"""
        all_files = self.scan_all_directories()
        
        print("\n📈 EVK4 Method Comparison Summary:")
        print("-" * 60)
        
        for method in self.expected_dirs:
            if method in all_files:
                files = all_files[method]
                print(f"{method:12}: {len(files):2d} files")
                
                # 显示文件大小统计（如果可以快速获取）
                if files:
                    try:
                        sizes = [f.stat().st_size / (1024*1024) for f in files[:3]]  # 前3个文件的MB大小
                        avg_size = sum(sizes) / len(sizes)
                        print(f"            Average size: ~{avg_size:.1f}MB")
                    except:
                        pass
            else:
                print(f"{method:12}: Not found")


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="Complete visualization for all EVK4 processing results")
    
    parser.add_argument("--evk4_dir", default=None,
                       help="EVK4 directory path (default: auto-detect)")
    parser.add_argument("--output_dir", default="debug_output",
                       help="Output base directory (default: debug_output)")
    parser.add_argument("--list", action='store_true',
                       help="List available files and exit")
    parser.add_argument("--summary", action='store_true',
                       help="Show method comparison summary and exit")
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = EVK4CompleteVisualizer(
        evk4_dir=args.evk4_dir,
        output_base_dir=args.output_dir
    )
    
    # 处理选项
    if args.list:
        visualizer.list_available_files()
        return
    
    if args.summary:
        visualizer.generate_method_comparison_summary()
        return
    
    # 生成完整可视化
    visualizer.generate_all_videos()


if __name__ == "__main__":
    main()