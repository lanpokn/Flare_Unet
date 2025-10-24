#!/usr/bin/env python3
"""
视频帧提取工具 - Video to Frames Extractor

功能：
- 读取visualize文件夹中的所有MP4视频
- 将每个视频的所有帧提取为图片
- 存储在同级同名子文件夹中（视频名_frames/）

使用方法：
    python src/tools/video_to_frames.py --input DSEC_data/visualize
    python src/tools/video_to_frames.py --input MainReal_data/visualize --format png
    python src/tools/video_to_frames.py --input DSEC_data/visualize/real_flare_xxx --single
"""

import argparse
import cv2
from pathlib import Path
from typing import List
import sys


class VideoToFramesExtractor:
    """视频帧提取器"""

    def __init__(self, image_format: str = 'jpg', quality: int = 95):
        """
        Args:
            image_format: 图片格式 ('jpg' or 'png')
            quality: JPG质量 (1-100, 仅jpg格式有效)
        """
        self.image_format = image_format.lower()
        self.quality = quality

        if self.image_format not in ['jpg', 'png']:
            raise ValueError("image_format must be 'jpg' or 'png'")

    def extract_frames(self, video_path: Path) -> int:
        """
        提取单个视频的所有帧

        Args:
            video_path: 视频文件路径

        Returns:
            提取的帧数
        """
        # 创建输出目录: 视频名_frames/
        video_name = video_path.stem  # 去掉.mp4后缀
        output_dir = video_path.parent / f"{video_name}_frames"
        output_dir.mkdir(exist_ok=True)

        # 打开视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return 0

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"📹 Processing: {video_path.name}")
        print(f"   Total frames: {total_frames}, FPS: {fps:.2f}")

        # 提取每一帧
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 保存帧
            if self.image_format == 'jpg':
                frame_path = output_dir / f"frame_{frame_count:05d}.jpg"
                cv2.imwrite(str(frame_path), frame,
                           [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            else:  # png
                frame_path = output_dir / f"frame_{frame_count:05d}.png"
                cv2.imwrite(str(frame_path), frame)

            frame_count += 1

            # 进度显示（每10帧显示一次）
            if frame_count % 10 == 0:
                print(f"   Progress: {frame_count}/{total_frames}", end='\r')

        cap.release()
        print(f"   ✅ Extracted {frame_count} frames → {output_dir}")
        return frame_count

    def process_directory(self, visualize_dir: Path) -> dict:
        """
        处理visualize目录下的所有视频

        Args:
            visualize_dir: visualize根目录路径

        Returns:
            处理统计信息
        """
        if not visualize_dir.exists():
            print(f"❌ Directory not found: {visualize_dir}")
            return {}

        # 查找所有MP4文件
        video_files = list(visualize_dir.rglob("*.mp4"))

        if not video_files:
            print(f"⚠️  No MP4 files found in {visualize_dir}")
            return {}

        print(f"\n🎬 Found {len(video_files)} video files")
        print(f"📁 Output format: {self.image_format.upper()}")
        if self.image_format == 'jpg':
            print(f"🎨 JPEG quality: {self.quality}")
        print("-" * 60)

        # 处理每个视频
        stats = {
            'total_videos': len(video_files),
            'total_frames': 0,
            'success': 0,
            'failed': 0
        }

        for idx, video_path in enumerate(video_files, 1):
            print(f"\n[{idx}/{len(video_files)}]")
            try:
                frame_count = self.extract_frames(video_path)
                if frame_count > 0:
                    stats['success'] += 1
                    stats['total_frames'] += frame_count
                else:
                    stats['failed'] += 1
            except Exception as e:
                print(f"❌ Error processing {video_path.name}: {e}")
                stats['failed'] += 1

        return stats

    def process_single_folder(self, folder_path: Path) -> dict:
        """
        处理单个文件夹中的所有视频（非递归）

        Args:
            folder_path: 单个visualize子文件夹路径

        Returns:
            处理统计信息
        """
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"❌ Directory not found: {folder_path}")
            return {}

        # 只查找当前目录下的MP4文件
        video_files = list(folder_path.glob("*.mp4"))

        if not video_files:
            print(f"⚠️  No MP4 files found in {folder_path}")
            return {}

        print(f"\n🎬 Found {len(video_files)} video files in {folder_path.name}")
        print(f"📁 Output format: {self.image_format.upper()}")
        if self.image_format == 'jpg':
            print(f"🎨 JPEG quality: {self.quality}")
        print("-" * 60)

        # 处理每个视频
        stats = {
            'total_videos': len(video_files),
            'total_frames': 0,
            'success': 0,
            'failed': 0
        }

        for idx, video_path in enumerate(video_files, 1):
            print(f"\n[{idx}/{len(video_files)}]")
            try:
                frame_count = self.extract_frames(video_path)
                if frame_count > 0:
                    stats['success'] += 1
                    stats['total_frames'] += frame_count
                else:
                    stats['failed'] += 1
            except Exception as e:
                print(f"❌ Error processing {video_path.name}: {e}")
                stats['failed'] += 1

        return stats


def print_summary(stats: dict):
    """打印处理总结"""
    print("\n" + "=" * 60)
    print("📊 Processing Summary")
    print("=" * 60)
    print(f"Total videos: {stats.get('total_videos', 0)}")
    print(f"Success: {stats.get('success', 0)}")
    print(f"Failed: {stats.get('failed', 0)}")
    print(f"Total frames extracted: {stats.get('total_frames', 0)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from MP4 videos in visualize directory'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to visualize directory or single folder'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='jpg',
        choices=['jpg', 'png'],
        help='Output image format (default: jpg)'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG quality 1-100 (default: 95, only for jpg format)'
    )
    parser.add_argument(
        '--single',
        action='store_true',
        help='Process single folder (non-recursive)'
    )

    args = parser.parse_args()

    # 验证输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Path not found: {input_path}")
        sys.exit(1)

    # 创建提取器
    extractor = VideoToFramesExtractor(
        image_format=args.format,
        quality=args.quality
    )

    # 处理视频
    if args.single:
        # 单个文件夹模式
        stats = extractor.process_single_folder(input_path)
    else:
        # 递归处理模式
        stats = extractor.process_directory(input_path)

    # 打印总结
    print_summary(stats)


if __name__ == '__main__':
    main()
