"""
Event Video Generator - 基于H5事件数据生成可视化视频

基于Linus哲学：
- 数据结构正确: Events (N,4) → Time Windows → RGB Frames → Video
- 消除特殊情况: 统一2.5ms时间窗口，白背景+红蓝映射
- 实用主义: 解决H5事件数据可视化的实际需求

Usage:
    python src/tools/event_video_generator.py \
        --input "path/to/events.h5" \
        --output "output_video.mp4" \
        --frame_duration_ms 2.5
"""

import numpy as np
import h5py
import cv2
import argparse
from pathlib import Path
import os
import sys

# 添加项目路径以导入现有模块
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.encode import load_h5_events


class EventVideoGenerator:
    """H5事件数据视频生成器"""
    
    def __init__(self, sensor_size=(480, 640), frame_duration_ms=2.5, fps=30):
        """
        Args:
            sensor_size: 传感器尺寸 (height, width)
            frame_duration_ms: 每帧时间间隔 (默认2.5ms)
            fps: 输出视频帧率
        """
        self.sensor_size = sensor_size  # (H, W)
        self.frame_duration_us = frame_duration_ms * 1000  # 转换为微秒
        self.fps = fps
        
    def load_events(self, h5_path: str) -> np.ndarray:
        """加载H5事件数据"""
        print(f"Loading events from: {h5_path}")
        events_np = load_h5_events(h5_path)
        print(f"Loaded {len(events_np):,} events")
        
        # 时间范围统计
        if len(events_np) > 0:
            t_min, t_max = events_np[:, 0].min(), events_np[:, 0].max()
            duration_ms = (t_max - t_min) / 1000
            print(f"Time range: {t_min:.0f} - {t_max:.0f} μs ({duration_ms:.1f} ms)")
        
        return events_np
    
    def _create_event_visualization_iebcs(self, events: np.ndarray) -> np.ndarray:
        """创建IEBCS事件可视化 - 白背景版本"""
        # 🎯 关键：创建白色背景的RGB图像
        event_img = np.full((*self.sensor_size, 3), 255, dtype=np.uint8)  # 白色背景
        
        # 🎨 核心颜色映射逻辑
        for event in events:
            x, y, pol = int(event[1]), int(event[2]), int(event[3])
            if 0 <= x < self.sensor_size[1] and 0 <= y < self.sensor_size[0]:
                if pol > 0.5:  # Positive event (ON)
                    event_img[y, x] = [0, 0, 255]  # 🔴 Red (RGB格式)
                else:  # Negative event (OFF)  
                    event_img[y, x] = [255, 0, 0]  # 🔵 Blue (RGB格式)
        
        return event_img
    
    def generate_frames(self, events_np: np.ndarray):
        """生成视频帧序列"""
        if len(events_np) == 0:
            print("No events to generate frames")
            return []
            
        # 计算时间范围和帧数
        t_min, t_max = events_np[:, 0].min(), events_np[:, 0].max()
        total_duration = t_max - t_min
        num_frames = int(np.ceil(total_duration / self.frame_duration_us))
        
        print(f"Generating {num_frames} frames with {self.frame_duration_us/1000:.1f}ms per frame")
        
        frames = []
        
        for frame_idx in range(num_frames):
            # 计算当前帧的时间窗口
            t_start = t_min + frame_idx * self.frame_duration_us
            t_end = t_start + self.frame_duration_us
            
            # 选择当前时间窗口内的事件
            mask = (events_np[:, 0] >= t_start) & (events_np[:, 0] < t_end)
            frame_events = events_np[mask]
            
            # 生成可视化图像
            frame_img = self._create_event_visualization_iebcs(frame_events)
            frames.append(frame_img)
            
            # 进度显示
            if (frame_idx + 1) % 100 == 0 or frame_idx == num_frames - 1:
                progress = (frame_idx + 1) / num_frames * 100
                print(f"Generated frame {frame_idx + 1}/{num_frames} ({progress:.1f}%) - {len(frame_events)} events")
        
        return frames
    
    def save_video(self, frames: list, output_path: str):
        """保存视频文件"""
        if not frames:
            print("No frames to save")
            return
            
        print(f"Saving video to: {output_path}")
        
        # 创建输出目录
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            self.fps, 
            (self.sensor_size[1], self.sensor_size[0])  # OpenCV expects (width, height)
        )
        
        # 写入帧
        for frame_idx, frame in enumerate(frames):
            # OpenCV使用BGR格式，需要转换
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
            
            if (frame_idx + 1) % 100 == 0 or frame_idx == len(frames) - 1:
                progress = (frame_idx + 1) / len(frames) * 100
                print(f"Saved frame {frame_idx + 1}/{len(frames)} ({progress:.1f}%)")
        
        video_writer.release()
        print(f"✅ Video saved successfully: {output_path}")
        
        # 输出视频信息
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        duration_s = len(frames) / self.fps
        print(f"📊 Video info: {len(frames)} frames, {duration_s:.1f}s, {file_size_mb:.1f}MB")
    
    def process_h5_file(self, h5_path: str, output_path: str = None, output_dir: str = "debug_output"):
        """处理单个H5文件生成视频"""
        # 生成默认输出路径（包含文件名）
        if output_path is None:
            h5_file = Path(h5_path)
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{h5_file.stem}_visualization.mp4"
        
        print(f"🎬 Event Video Generation Started")
        print(f"📁 Input: {h5_path}")
        print(f"📁 Output: {output_path}")
        print(f"⚙️ Config: {self.sensor_size}, {self.frame_duration_us/1000:.1f}ms/frame, {self.fps}fps")
        print("-" * 60)
        
        # 加载事件数据
        events_np = self.load_events(h5_path)
        
        if len(events_np) == 0:
            print("❌ No events found in H5 file")
            return
        
        # 生成帧序列
        frames = self.generate_frames(events_np)
        
        if not frames:
            print("❌ No frames generated")
            return
            
        # 保存视频
        self.save_video(frames, output_path)
        
        print("-" * 60)
        print(f"🎉 Video generation completed!")


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description="Generate visualization video from H5 event data")
    
    # 默认路径（Windows格式，Python会自动处理）
    default_input = r"E:\2025\event_flick_flare\Unet_main\data_simu\physics_method\background_with_flare_events_testoutput\composed_00504_bg_flare.h5"
    
    parser.add_argument("--input", "-i", default=default_input,
                       help="Input H5 event file path")
    parser.add_argument("--output", "-o", default=None,
                       help="Output video file path (auto-generated if not specified)")
    parser.add_argument("--output_dir", default="debug_output",
                       help="Output directory (default: debug_output)")
    parser.add_argument("--frame_duration_ms", "-d", type=float, default=2.5,
                       help="Frame duration in milliseconds (default: 2.5ms)")
    parser.add_argument("--fps", type=int, default=10,
                       help="Output video FPS (default: 10)")
    parser.add_argument("--sensor_size", nargs=2, type=int, default=[480, 640],
                       help="Sensor size as height width (default: 480 640)")
    
    args = parser.parse_args()
    
    # 转换Windows路径格式
    input_path = args.input.replace('\\', '/')
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ Input file not found: {input_path}")
        return
    
    # 创建视频生成器
    generator = EventVideoGenerator(
        sensor_size=tuple(args.sensor_size),
        frame_duration_ms=args.frame_duration_ms,
        fps=args.fps
    )
    
    # 处理文件
    generator.process_h5_file(input_path, args.output, args.output_dir)


if __name__ == "__main__":
    main()