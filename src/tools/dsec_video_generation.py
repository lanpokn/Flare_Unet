"""
DSEC Video Generation - 批量处理DSEC_data中真实事件数据生成可视化视频

基于Linus哲学：
- 数据结构正确: 扫描DSEC目录 → 选取前5个H5 → 调用event_video_generator → 按文件夹组织输出
- 消除特殊情况: 统一处理DSEC_data下的所有文件夹
- 实用主义: 解决真实数据批量可视化的实际需求

Usage:
    python src/tools/dsec_video_generation.py
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List, Dict

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools.event_video_generator import EventVideoGenerator


class DSECVideoGenerator:
    """DSEC批量视频生成器 - 专门处理DSEC_data下的真实事件数据"""
    
    def __init__(self, dsec_data_dir: str = None, output_base_dir: str = "debug_output"):
        """
        Args:
            dsec_data_dir: DSEC_data目录路径
            output_base_dir: 输出基础目录
        """
        if dsec_data_dir is None:
            self.dsec_data_dir = Path(PROJECT_ROOT) / "DSEC_data"
        else:
            self.dsec_data_dir = Path(dsec_data_dir)
            
        self.output_base_dir = Path(output_base_dir)
        
        # 创建视频生成器实例
        self.video_generator = EventVideoGenerator(
            sensor_size=(480, 640),
            frame_duration_ms=2.5,
            fps=10
        )
    
    def scan_dsec_directories(self) -> Dict[str, List[Path]]:
        """扫描DSEC_data下包含H5文件的所有目录"""
        h5_dirs = {}
        
        print(f"🔍 Scanning DSEC directories in: {self.dsec_data_dir}")
        
        if not self.dsec_data_dir.exists():
            print(f"❌ DSEC data directory not found: {self.dsec_data_dir}")
            return h5_dirs
        
        # 扫描所有子目录
        for subdir in self.dsec_data_dir.iterdir():
            if subdir.is_dir():
                # 查找H5文件
                h5_files = list(subdir.glob("*.h5"))
                if h5_files:
                    # 按名称排序并取前5个
                    h5_files_sorted = sorted(h5_files)[:5]
                    h5_dirs[subdir.name] = h5_files_sorted
                    print(f"📁 Found {len(h5_files)} H5 files in '{subdir.name}' (taking first {len(h5_files_sorted)})")
                else:
                    print(f"⚠️  Directory '{subdir.name}' has no H5 files")
        
        return h5_dirs
    
    def create_output_structure(self, folder_name: str) -> Path:
        """创建输出目录结构"""
        output_dir = self.output_base_dir / f"dsec_{folder_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def generate_videos_for_folder(self, folder_name: str, h5_files: List[Path]):
        """为单个文件夹生成视频"""
        print(f"\n🎬 Processing DSEC folder: {folder_name}")
        print(f"📋 Files to process: {len(h5_files)}")
        
        # 创建输出目录
        output_dir = self.create_output_structure(folder_name)
        
        success_count = 0
        for i, h5_file in enumerate(h5_files, 1):
            try:
                print(f"\n📹 [{i}/{len(h5_files)}] Processing: {h5_file.name}")
                
                # 生成输出文件名
                output_video = output_dir / f"{h5_file.stem}_visualization.mp4"
                
                # 生成视频
                self.video_generator.process_h5_file(
                    str(h5_file), 
                    str(output_video)
                )
                
                success_count += 1
                print(f"✅ [{i}/{len(h5_files)}] Success: {output_video.name}")
                
            except Exception as e:
                print(f"❌ [{i}/{len(h5_files)}] Failed: {h5_file.name} - {str(e)}")
                continue
        
        print(f"\n📊 DSEC folder '{folder_name}' summary: {success_count}/{len(h5_files)} videos generated")
        return success_count
    
    def generate_all_videos(self):
        """生成所有DSEC视频"""
        print("🚀 DSEC Video Generation Started")
        print(f"📂 Source: {self.dsec_data_dir}")
        print(f"📂 Output: {self.output_base_dir}")
        print("-" * 80)
        
        # 扫描目录
        h5_dirs = self.scan_dsec_directories()
        
        if not h5_dirs:
            print("❌ No H5 files found in any DSEC directory")
            return
        
        print(f"\n🎯 Found {len(h5_dirs)} DSEC directories with H5 files:")
        for folder_name, files in h5_dirs.items():
            print(f"  • {folder_name}: {len(files)} files")
        
        # 处理每个文件夹
        total_success = 0
        total_files = 0
        
        for folder_name, h5_files in h5_dirs.items():
            success_count = self.generate_videos_for_folder(folder_name, h5_files)
            total_success += success_count
            total_files += len(h5_files)
        
        # 最终统计
        print("\n" + "=" * 80)
        print("🎉 DSEC Batch Video Generation Completed!")
        print(f"📊 Overall Summary:")
        print(f"  • Processed DSEC folders: {len(h5_dirs)}")
        print(f"  • Total videos generated: {total_success}/{total_files}")
        print(f"  • Success rate: {total_success/total_files*100:.1f}%")
        print(f"  • Output location: {self.output_base_dir}")
        print("=" * 80)
    
    def list_available_folders(self):
        """列出可用的DSEC文件夹"""
        h5_dirs = self.scan_dsec_directories()
        
        print("Available DSEC folders with H5 files:")
        if not h5_dirs:
            print("  No DSEC folders found with H5 files")
            return
            
        for i, (folder_name, files) in enumerate(h5_dirs.items(), 1):
            print(f"  {i}. {folder_name} ({len(files)} files)")
    
    def generate_for_specific_folders(self, folder_names: List[str]):
        """为指定的DSEC文件夹生成视频"""
        print("🎯 Generating videos for specific DSEC folders:")
        for folder in folder_names:
            print(f"  • {folder}")
        print("-" * 80)
        
        h5_dirs = self.scan_dsec_directories()
        
        # 过滤指定文件夹
        filtered_dirs = {name: files for name, files in h5_dirs.items() if name in folder_names}
        
        if not filtered_dirs:
            print("❌ No matching DSEC folders found")
            return
        
        # 处理指定文件夹
        total_success = 0
        total_files = 0
        
        for folder_name, h5_files in filtered_dirs.items():
            success_count = self.generate_videos_for_folder(folder_name, h5_files)
            total_success += success_count
            total_files += len(h5_files)
        
        print(f"\n🎉 Specific DSEC folder processing completed: {total_success}/{total_files} videos")


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="Batch generate visualization videos from DSEC H5 event data")
    
    parser.add_argument("--dsec_dir", default=None,
                       help="DSEC data directory path (default: auto-detect)")
    parser.add_argument("--output_dir", default="debug_output",
                       help="Output base directory (default: debug_output)")
    parser.add_argument("--folders", nargs='+', default=None,
                       help="Specific folder names to process (default: all folders)")
    parser.add_argument("--list", action='store_true',
                       help="List available DSEC folders and exit")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = DSECVideoGenerator(
        dsec_data_dir=args.dsec_dir,
        output_base_dir=args.output_dir
    )
    
    # 列出文件夹选项
    if args.list:
        generator.list_available_folders()
        return
    
    # 生成视频
    if args.folders:
        generator.generate_for_specific_folders(args.folders)
    else:
        generator.generate_all_videos()


if __name__ == "__main__":
    main()