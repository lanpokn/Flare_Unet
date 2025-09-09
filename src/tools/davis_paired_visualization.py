#!/usr/bin/env python3
"""
DAVIS成对可视化工具 - 比较input和target的H5事件数据

基于Linus哲学：
- 数据结构正确: Input/Target H5 Files → Side-by-side Visualization → Paired Comparison
- 消除特殊情况: 统一DAVIS分辨率346×260，自动匹配同名文件
- 实用主义: 直观判断input和target是否成对匹配

用法:
    python src/tools/davis_paired_visualization.py
    python src/tools/davis_paired_visualization.py --debug
    python src/tools/davis_paired_visualization.py --sample_count 5
    
功能:
- 自动扫描DAVIS/input和DAVIS/target目录
- 生成成对比较的可视化视频
- 支持并排(side-by-side)和上下(top-bottom)布局
- 输出到debug_output/davis_paired/文件夹
- 生成配对检查报告
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import numpy as np
import cv2
import h5py

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.encode import load_h5_events
from src.tools.event_video_generator import EventVideoGenerator


class DAVISPairedVisualizer:
    """DAVIS成对可视化器"""
    
    def __init__(self, davis_dir: str = None, output_dir: str = "debug_output", debug: bool = False):
        """
        Args:
            davis_dir: DAVIS数据目录路径
            output_dir: 输出目录
            debug: 调试模式
        """
        if davis_dir is None:
            self.davis_dir = Path(PROJECT_ROOT) / "DAVIS"
        else:
            self.davis_dir = Path(davis_dir)
            
        self.input_dir = self.davis_dir / "input"
        self.target_dir = self.davis_dir / "target"
        self.output_dir = Path(output_dir) / "davis_paired"
        self.debug = debug
        
        # DAVIS相机分辨率: 346×260 (您提供的规格)
        self.davis_sensor_size = (260, 346)  # (height, width)
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 单独的视频生成器（用于单个文件）
        self.single_video_generator = EventVideoGenerator(
            sensor_size=self.davis_sensor_size,
            frame_duration_ms=2.5,
            fps=30
        )
        
        print(f"🔍 DAVIS成对可视化器初始化完成")
        print(f"📁 输入目录: {self.input_dir}")
        print(f"📁 目标目录: {self.target_dir}")
        print(f"📁 输出目录: {self.output_dir}")
        print(f"📏 DAVIS分辨率: {self.davis_sensor_size}")
    
    def scan_paired_files(self) -> List[Tuple[Path, Path]]:
        """
        扫描input和target目录，找到匹配的文件对
        
        Returns:
            List[Tuple[Path, Path]]: (input_file, target_file) 文件对列表
        """
        if not self.input_dir.exists() or not self.target_dir.exists():
            print(f"❌ DAVIS目录不存在: {self.input_dir} 或 {self.target_dir}")
            return []
        
        # 获取所有H5文件
        input_files = {f.name: f for f in self.input_dir.glob("*.h5")}
        target_files = {f.name: f for f in self.target_dir.glob("*.h5")}
        
        # 找到匹配的文件对
        paired_files = []
        for filename in sorted(input_files.keys()):
            if filename in target_files:
                paired_files.append((input_files[filename], target_files[filename]))
                if self.debug:
                    print(f"✅ 找到文件对: {filename}")
            else:
                print(f"⚠️  input中的文件无匹配target: {filename}")
        
        # 检查target中未匹配的文件
        for filename in target_files:
            if filename not in input_files:
                print(f"⚠️  target中的文件无匹配input: {filename}")
        
        print(f"📊 找到 {len(paired_files)} 个文件对")
        return paired_files
    
    def analyze_file_pair(self, input_file: Path, target_file: Path) -> Dict:
        """
        分析单个文件对的基本统计信息
        
        Args:
            input_file: 输入文件路径
            target_file: 目标文件路径
            
        Returns:
            dict: 分析结果
        """
        try:
            # 加载事件数据
            input_events = load_h5_events(str(input_file))
            target_events = load_h5_events(str(target_file))
            
            stats = {
                "filename": input_file.name,
                "input_event_count": len(input_events),
                "target_event_count": len(target_events),
                "event_count_ratio": len(target_events) / len(input_events) if len(input_events) > 0 else 0,
                "input_time_range": None,
                "target_time_range": None,
                "time_overlap": False
            }
            
            # 时间范围分析
            if len(input_events) > 0:
                input_t_min, input_t_max = input_events[:, 0].min(), input_events[:, 0].max()
                stats["input_time_range"] = (int(input_t_min), int(input_t_max))
                stats["input_duration_ms"] = (input_t_max - input_t_min) / 1000
            
            if len(target_events) > 0:
                target_t_min, target_t_max = target_events[:, 0].min(), target_events[:, 0].max()
                stats["target_time_range"] = (int(target_t_min), int(target_t_max))
                stats["target_duration_ms"] = (target_t_max - target_t_min) / 1000
                
                # 检查时间重叠
                if stats["input_time_range"] and stats["target_time_range"]:
                    input_range = stats["input_time_range"]
                    target_range = stats["target_time_range"]
                    stats["time_overlap"] = not (input_range[1] < target_range[0] or target_range[1] < input_range[0])
            
            return stats
            
        except Exception as e:
            return {
                "filename": input_file.name,
                "error": str(e),
                "input_event_count": 0,
                "target_event_count": 0
            }
    
    def create_side_by_side_video(self, input_file: Path, target_file: Path, output_file: Path) -> bool:
        """
        创建并排比较的视频
        
        Args:
            input_file: 输入文件路径
            target_file: 目标文件路径  
            output_file: 输出视频路径
            
        Returns:
            bool: 是否成功
        """
        try:
            print(f"🎬 创建并排视频: {input_file.name}")
            
            # 加载事件数据
            input_events = load_h5_events(str(input_file))
            target_events = load_h5_events(str(target_file))
            
            if len(input_events) == 0 or len(target_events) == 0:
                print(f"❌ 事件数据为空")
                return False
            
            # 计算共同的时间范围
            input_t_min, input_t_max = input_events[:, 0].min(), input_events[:, 0].max()
            target_t_min, target_t_max = target_events[:, 0].min(), target_events[:, 0].max()
            
            # 使用重叠时间范围，或使用较小的时间范围
            t_min = max(input_t_min, target_t_min)
            t_max = min(input_t_max, target_t_max)
            
            if t_max <= t_min:
                # 如果没有重叠，使用input的时间范围
                t_min, t_max = input_t_min, input_t_max
                print(f"⚠️  时间范围无重叠，使用input时间范围")
            
            total_duration = t_max - t_min
            frame_duration_us = 2500  # 2.5ms per frame
            num_frames = int(np.ceil(total_duration / frame_duration_us))
            
            print(f"📊 时间范围: {t_min:.0f} - {t_max:.0f} μs ({total_duration/1000:.1f} ms)")
            print(f"🎞️  生成 {num_frames} 帧")
            
            # 创建视频写入器
            # 并排布局: 两个346×260 → 692×260
            video_width = self.davis_sensor_size[1] * 2  # 692
            video_height = self.davis_sensor_size[0]     # 260
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_file), fourcc, 30.0, (video_width, video_height))
            
            for frame_idx in range(num_frames):
                # 计算当前帧的时间窗口
                t_start = t_min + frame_idx * frame_duration_us
                t_end = t_start + frame_duration_us
                
                # 生成input帧
                input_mask = (input_events[:, 0] >= t_start) & (input_events[:, 0] < t_end)
                input_frame_events = input_events[input_mask]
                input_frame = self._create_davis_event_frame(input_frame_events)
                
                # 生成target帧
                target_mask = (target_events[:, 0] >= t_start) & (target_events[:, 0] < t_end)
                target_frame_events = target_events[target_mask]
                target_frame = self._create_davis_event_frame(target_frame_events)
                
                # 创建并排帧
                side_by_side_frame = np.hstack([input_frame, target_frame])
                
                # 添加标题
                self._add_titles_to_frame(side_by_side_frame, "INPUT", "TARGET", frame_idx, num_frames)
                
                # 写入视频
                out.write(side_by_side_frame)
                
                if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
                    print(f"  生成帧 {frame_idx + 1}/{num_frames} ({(frame_idx + 1)/num_frames*100:.1f}%)")
            
            out.release()
            
            # 检查输出文件
            if output_file.exists() and output_file.stat().st_size > 0:
                file_size_mb = output_file.stat().st_size / (1024 * 1024)
                print(f"✅ 视频生成成功: {output_file}")
                print(f"📊 视频信息: {num_frames} frames, {file_size_mb:.1f}MB")
                return True
            else:
                print(f"❌ 视频生成失败")
                return False
                
        except Exception as e:
            print(f"❌ 生成并排视频异常: {e}")
            return False
    
    def _create_davis_event_frame(self, events: np.ndarray) -> np.ndarray:
        """创建DAVIS事件可视化帧"""
        # 创建白色背景的RGB图像
        frame = np.full((*self.davis_sensor_size, 3), 255, dtype=np.uint8)  # 白色背景
        
        # 绘制事件
        for event in events:
            x, y, pol = int(event[1]), int(event[2]), int(event[3])
            if 0 <= x < self.davis_sensor_size[1] and 0 <= y < self.davis_sensor_size[0]:
                if pol > 0.5:  # Positive event (ON)
                    frame[y, x] = [0, 0, 255]  # 红色 (BGR格式)
                else:  # Negative event (OFF)  
                    frame[y, x] = [255, 0, 0]  # 蓝色 (BGR格式)
        
        return frame
    
    def _add_titles_to_frame(self, frame: np.ndarray, left_title: str, right_title: str, 
                           frame_idx: int, total_frames: int):
        """在并排帧上添加标题"""
        height, width = frame.shape[:2]
        half_width = width // 2
        
        # 字体设置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (0, 0, 0)  # 黑色
        
        # 左侧标题
        (text_width, text_height), _ = cv2.getTextSize(left_title, font, font_scale, thickness)
        left_x = (half_width - text_width) // 2
        cv2.putText(frame, left_title, (left_x, text_height + 10), font, font_scale, color, thickness)
        
        # 右侧标题
        (text_width, text_height), _ = cv2.getTextSize(right_title, font, font_scale, thickness)
        right_x = half_width + (half_width - text_width) // 2
        cv2.putText(frame, right_title, (right_x, text_height + 10), font, font_scale, color, thickness)
        
        # 帧数信息
        progress_text = f"Frame {frame_idx + 1}/{total_frames}"
        (text_width, text_height), _ = cv2.getTextSize(progress_text, font, 0.5, 1)
        cv2.putText(frame, progress_text, (width - text_width - 10, height - 10), 
                   font, 0.5, color, 1)
    
    def generate_individual_videos(self, input_file: Path, target_file: Path, output_subdir: Path) -> Tuple[bool, bool]:
        """
        生成单独的input和target视频
        
        Args:
            input_file: 输入文件路径
            target_file: 目标文件路径
            output_subdir: 输出子目录
            
        Returns:
            Tuple[bool, bool]: (input视频成功, target视频成功)
        """
        try:
            file_stem = input_file.stem
            
            # 生成input视频
            input_video_file = output_subdir / f"{file_stem}_input.mp4"
            print(f"  🎬 生成input视频: {input_video_file.name}")
            
            self.single_video_generator.process_h5_file(str(input_file), str(input_video_file))
            input_success = input_video_file.exists() and input_video_file.stat().st_size > 0
            
            # 生成target视频
            target_video_file = output_subdir / f"{file_stem}_target.mp4"
            print(f"  🎬 生成target视频: {target_video_file.name}")
            
            self.single_video_generator.process_h5_file(str(target_file), str(target_video_file))
            target_success = target_video_file.exists() and target_video_file.stat().st_size > 0
            
            return input_success, target_success
            
        except Exception as e:
            print(f"❌ 生成单独视频异常: {e}")
            return False, False
    
    def process_all_pairs(self, sample_count: int = None) -> Dict:
        """
        处理所有文件对
        
        Args:
            sample_count: 限制处理的文件对数量
            
        Returns:
            dict: 处理结果统计
        """
        print(f"\n🚀 开始DAVIS成对可视化")
        print(f"{'='*50}")
        
        # 扫描文件对
        paired_files = self.scan_paired_files()
        
        if not paired_files:
            print("❌ 未找到匹配的文件对")
            return {"success": False, "pairs_processed": 0}
        
        # 限制处理数量
        if sample_count:
            paired_files = paired_files[:sample_count]
            print(f"📊 限制处理前 {sample_count} 个文件对")
        
        # 处理结果统计
        results = {
            "start_time": datetime.now().isoformat(),
            "total_pairs": len(paired_files),
            "successful_pairs": 0,
            "failed_pairs": 0,
            "pair_analyses": [],
            "summary": {}
        }
        
        for i, (input_file, target_file) in enumerate(paired_files):
            print(f"\n📁 处理文件对 {i+1}/{len(paired_files)}: {input_file.name}")
            
            try:
                # 分析文件对
                pair_stats = self.analyze_file_pair(input_file, target_file)
                results["pair_analyses"].append(pair_stats)
                
                # 创建输出子目录
                file_stem = input_file.stem
                pair_output_dir = self.output_dir / file_stem
                pair_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 生成并排视频
                side_by_side_video = pair_output_dir / f"{file_stem}_comparison.mp4"
                comparison_success = self.create_side_by_side_video(input_file, target_file, side_by_side_video)
                
                # 生成单独视频
                individual_success = self.generate_individual_videos(input_file, target_file, pair_output_dir)
                
                if comparison_success:
                    results["successful_pairs"] += 1
                    print(f"✅ 文件对处理成功")
                else:
                    results["failed_pairs"] += 1
                    print(f"❌ 文件对处理失败")
                    
                # 保存单个文件对的统计
                pair_stats["comparison_video_success"] = comparison_success
                pair_stats["individual_videos_success"] = individual_success
                
                pair_stats_file = pair_output_dir / f"{file_stem}_stats.json"
                with open(pair_stats_file, 'w', encoding='utf-8') as f:
                    json.dump(pair_stats, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                print(f"❌ 处理文件对异常: {e}")
                results["failed_pairs"] += 1
        
        # 生成总结报告
        results["end_time"] = datetime.now().isoformat()
        results["success_rate"] = results["successful_pairs"] / results["total_pairs"] if results["total_pairs"] > 0 else 0
        
        # 保存全局统计
        global_stats_file = self.output_dir / "davis_paired_analysis.json"
        with open(global_stats_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印总结
        print(f"\n📊 DAVIS成对可视化完成")
        print(f"{'='*50}")
        print(f"总文件对: {results['total_pairs']}")
        print(f"成功处理: {results['successful_pairs']}")
        print(f"失败处理: {results['failed_pairs']}")
        print(f"成功率: {results['success_rate']*100:.1f}%")
        print(f"输出目录: {self.output_dir}")
        print(f"全局统计: {global_stats_file}")
        
        # 显示配对分析摘要
        if results["pair_analyses"]:
            event_ratios = [p.get("event_count_ratio", 0) for p in results["pair_analyses"] if "event_count_ratio" in p]
            if event_ratios:
                avg_ratio = sum(event_ratios) / len(event_ratios)
                print(f"📈 平均事件比率 (target/input): {avg_ratio:.3f}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DAVIS成对可视化工具")
    parser.add_argument("--davis_dir", type=str, help="DAVIS数据目录路径")
    parser.add_argument("--output_dir", type=str, default="debug_output", help="输出目录")
    parser.add_argument("--sample_count", type=int, help="限制处理的文件对数量")
    parser.add_argument("--debug", action="store_true", help="启用debug模式")
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = DAVISPairedVisualizer(
        davis_dir=args.davis_dir,
        output_dir=args.output_dir,
        debug=args.debug
    )
    
    # 处理所有文件对
    results = visualizer.process_all_pairs(sample_count=args.sample_count)
    
    if results["success_rate"] > 0.5:
        print(f"\n🎉 DAVIS成对可视化大部分成功!")
        return 0
    else:
        print(f"\n💥 DAVIS成对可视化大部分失败!")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())