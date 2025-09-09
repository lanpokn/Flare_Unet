#!/usr/bin/env python3
"""
递归推理实验工具 - 验证UNet3D网络的递归处理效果

基于Linus哲学：
- 数据结构正确: Input H5 → UNet → Output1 → UNet → Output2 → ... → OutputN
- 消除特殊情况: 统一递归处理pipeline，无需手动管理中间步骤
- 实用主义: 观察网络递归处理对事件数据的累积影响

用法:
    python src/tools/repeat_inference_experiment.py --iterations 10
    python src/tools/repeat_inference_experiment.py --iterations 5 --debug
    
功能:
- 对指定的两个H5文件进行递归inference处理
- 每次输出作为下次输入，形成处理链
- 每个中间结果生成MP4视频
- 输出整理到debug_output/repeat/文件夹
- 分析递归处理过程中事件数量和分布的变化
"""

import os
import sys
import tempfile
import shutil
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import h5py

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.encode import load_h5_events
from src.tools.event_video_generator import EventVideoGenerator


class RecursiveInferenceExperiment:
    """递归推理实验管理器"""
    
    def __init__(self, iterations=10, debug=False):
        """
        Args:
            iterations: 每个文件递归处理次数
            debug: 是否启用debug模式
        """
        self.iterations = iterations
        self.debug = debug
        
        # 目标文件路径 (基于CLAUDE.md记录的路径)
        self.target_files = [
            "data_simu/physics_method/background_with_flare_events_test/composed_00504_bg_flare.h5",
            "DSEC_data/input/real_flare_zurich_city_03_a_t1288ms_20250908_173252.h5"
        ]
        
        # 输出目录结构
        self.base_output_dir = Path("debug_output/repeat")
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 临时目录用于inference输出
        self.temp_dir = Path("temp_recursive")
        
        # 视频生成器
        self.video_generator = EventVideoGenerator(sensor_size=(480, 640), frame_duration_ms=2.5, fps=30)
        
        # 实验统计
        self.experiment_stats = {
            "start_time": datetime.now().isoformat(),
            "iterations": iterations,
            "target_files": self.target_files,
            "results": {}
        }
        
        print(f"🔄 递归推理实验初始化完成")
        print(f"📊 递归次数: {iterations}")
        print(f"📁 输出目录: {self.base_output_dir}")
        print(f"🎯 目标文件: {len(self.target_files)}个")
        
    def run_single_inference(self, input_file: str, output_file: str) -> bool:
        """
        运行单次inference
        
        Args:
            input_file: 输入H5文件路径
            output_file: 输出H5文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 使用main.py的inference模式
            cmd = [
                sys.executable, "main.py", "inference",
                "--config", "configs/inference_config.yaml",
                "--input", input_file,
                "--output", output_file
            ]
            
            if self.debug:
                print(f"🔧 运行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
            
            if result.returncode == 0:
                if self.debug:
                    print(f"✅ Inference成功: {input_file} → {output_file}")
                return True
            else:
                print(f"❌ Inference失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Inference异常: {e}")
            return False
    
    def generate_video_from_h5(self, h5_file: str, video_file: str) -> bool:
        """
        从H5文件生成MP4视频
        
        Args:
            h5_file: 输入H5文件路径
            video_file: 输出MP4文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            if self.debug:
                print(f"🎬 生成视频: {h5_file} → {video_file}")
            
            # 确保输出目录存在
            Path(video_file).parent.mkdir(parents=True, exist_ok=True)
            
            # 使用现有的视频生成器
            self.video_generator.process_h5_file(h5_file, video_file)
            success = Path(video_file).exists()
            
            if success:
                if self.debug:
                    print(f"✅ 视频生成成功: {video_file}")
                return True
            else:
                print(f"❌ 视频生成失败: {video_file}")
                return False
                
        except Exception as e:
            print(f"❌ 视频生成异常: {e}")
            return False
    
    def analyze_recursive_progression(self, h5_files: list) -> dict:
        """
        分析递归处理过程中的数据变化
        
        Args:
            h5_files: 按递归顺序排列的H5文件路径列表
            
        Returns:
            dict: 递归分析结果
        """
        try:
            stats = {
                "iteration_count": len(h5_files),
                "event_counts": [],
                "event_count_changes": [],
                "compression_ratios": [],
                "total_compression_ratio": 0.0
            }
            
            # 加载所有文件的事件计数并计算变化
            previous_count = None
            for i, h5_file in enumerate(h5_files):
                if os.path.exists(h5_file):
                    events = load_h5_events(h5_file)
                    current_count = len(events)
                    stats["event_counts"].append(current_count)
                    
                    if previous_count is not None:
                        # 计算相对于上一次的变化
                        change_ratio = current_count / previous_count if previous_count > 0 else 0
                        stats["event_count_changes"].append(change_ratio)
                        stats["compression_ratios"].append(change_ratio)
                    
                    previous_count = current_count
            
            # 计算总体压缩比率
            if len(stats["event_counts"]) >= 2:
                initial_count = stats["event_counts"][0]
                final_count = stats["event_counts"][-1]
                stats["total_compression_ratio"] = final_count / initial_count if initial_count > 0 else 0
                stats["initial_count"] = initial_count
                stats["final_count"] = final_count
                stats["total_change"] = final_count - initial_count
            
            return stats
            
        except Exception as e:
            print(f"❌ 递归分析异常: {e}")
            return {"error": str(e)}
    
    def process_single_file(self, target_file: str) -> bool:
        """
        处理单个目标文件的递归inference实验
        
        递归处理流程: 
        原始文件 → UNet → 结果1 → UNet → 结果2 → ... → 结果N
        
        Args:
            target_file: 目标H5文件路径
            
        Returns:
            bool: 是否成功完成所有递归步骤
        """
        file_name = Path(target_file).stem
        print(f"\n🎯 开始递归处理: {file_name}")
        
        # 为此文件创建输出目录
        file_output_dir = self.base_output_dir / file_name
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建临时目录
        temp_file_dir = self.temp_dir / file_name
        temp_file_dir.mkdir(parents=True, exist_ok=True)
        
        successful_iterations = 0
        output_files = []
        current_input_file = target_file  # 初始输入是原始文件
        
        try:
            for i in range(self.iterations):
                print(f"  🔄 递归 {i+1}/{self.iterations}", end="")
                
                # 定义当前递归步骤的输出文件
                temp_h5_file = temp_file_dir / f"iteration_{i+1:02d}.h5"
                
                # 运行inference: 当前输入 → UNet → 当前输出
                success = self.run_single_inference(current_input_file, str(temp_h5_file))
                
                if success and temp_h5_file.exists():
                    # 移动到最终输出目录
                    final_h5_file = file_output_dir / f"iteration_{i+1:02d}.h5"
                    shutil.move(str(temp_h5_file), str(final_h5_file))
                    output_files.append(str(final_h5_file))
                    
                    # 生成视频
                    video_file = file_output_dir / f"iteration_{i+1:02d}.mp4"
                    video_success = self.generate_video_from_h5(str(final_h5_file), str(video_file))
                    
                    if video_success:
                        successful_iterations += 1
                        print(f" ✅")
                        
                        # 🔄 关键: 将当前输出设为下次输入，形成递归链
                        current_input_file = str(final_h5_file)
                        
                    else:
                        print(f" ⚠️ (H5成功,视频失败)")
                        # 即使视频失败，也继续递归链
                        current_input_file = str(final_h5_file)
                        successful_iterations += 1
                else:
                    print(f" ❌ (停止递归)")
                    break  # 如果inference失败，停止递归链
            
            # 添加原始文件到分析序列的开头
            all_files_for_analysis = [target_file] + output_files
            
            # 分析递归过程
            recursive_stats = self.analyze_recursive_progression(all_files_for_analysis)
            
            # 保存统计结果
            stats_file = file_output_dir / "recursive_experiment_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "file_name": file_name,
                    "target_file": target_file,
                    "successful_iterations": successful_iterations,
                    "total_iterations": self.iterations,
                    "success_rate": successful_iterations / self.iterations,
                    "recursive_analysis": recursive_stats
                }, f, indent=2, ensure_ascii=False)
            
            # 保存到全局统计
            self.experiment_stats["results"][file_name] = {
                "successful_iterations": successful_iterations,
                "total_iterations": self.iterations,
                "success_rate": successful_iterations / self.iterations,
                "recursive_analysis": recursive_stats
            }
            
            print(f"  📋 完成递归: {successful_iterations}/{self.iterations} 成功")
            
            # 显示递归效果摘要
            if recursive_stats.get("initial_count") and recursive_stats.get("final_count"):
                initial = recursive_stats["initial_count"]
                final = recursive_stats["final_count"]
                ratio = recursive_stats["total_compression_ratio"]
                change = recursive_stats["total_change"]
                print(f"  📊 事件变化: {initial:,} → {final:,} (比率: {ratio:.3f}, 变化: {change:+,})")
            
            return successful_iterations > 0
            
        except Exception as e:
            print(f"❌ 递归处理失败 {target_file}: {e}")
            return False
            
        finally:
            # 清理临时目录
            if temp_file_dir.exists():
                shutil.rmtree(temp_file_dir)
    
    def run_experiment(self) -> bool:
        """运行完整递归实验"""
        print(f"\n🚀 开始递归推理实验")
        print(f"{'='*50}")
        
        overall_success = True
        
        try:
            # 确保临时目录存在
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            
            # 处理每个目标文件
            for target_file in self.target_files:
                # 检查文件是否存在
                if not Path(target_file).exists():
                    print(f"❌ 目标文件不存在: {target_file}")
                    overall_success = False
                    continue
                
                file_success = self.process_single_file(target_file)
                if not file_success:
                    overall_success = False
            
            # 保存全局统计报告
            self.experiment_stats["end_time"] = datetime.now().isoformat()
            self.experiment_stats["overall_success"] = overall_success
            
            global_stats_file = self.base_output_dir / "global_recursive_stats.json"
            with open(global_stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.experiment_stats, f, indent=2, ensure_ascii=False)
            
            # 打印总结
            print(f"\n📊 递归实验总结")
            print(f"{'='*50}")
            print(f"总体成功: {'✅' if overall_success else '❌'}")
            print(f"处理文件: {len(self.target_files)}个")
            print(f"递归次数: {self.iterations}")
            print(f"输出目录: {self.base_output_dir}")
            print(f"全局统计: {global_stats_file}")
            
            for file_name, result in self.experiment_stats["results"].items():
                success_rate = result["success_rate"]
                recursive_analysis = result["recursive_analysis"]
                print(f"  📁 {file_name}: {success_rate*100:.1f}% 成功")
                
                if "total_compression_ratio" in recursive_analysis:
                    ratio = recursive_analysis["total_compression_ratio"]
                    initial = recursive_analysis.get("initial_count", 0)
                    final = recursive_analysis.get("final_count", 0)
                    print(f"    🔄 递归效果: {initial:,} → {final:,} (比率: {ratio:.3f})")
            
            return overall_success
            
        except Exception as e:
            print(f"❌ 实验异常: {e}")
            return False
            
        finally:
            # 清理临时目录
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="递归推理实验工具")
    parser.add_argument("--iterations", type=int, default=10, help="每个文件的递归次数 (默认: 10)")
    parser.add_argument("--debug", action="store_true", help="启用debug模式")
    
    args = parser.parse_args()
    
    # 创建实验管理器
    experiment = RecursiveInferenceExperiment(
        iterations=args.iterations,
        debug=args.debug
    )
    
    # 运行实验
    success = experiment.run_experiment()
    
    if success:
        print(f"\n🎉 递归实验完成成功!")
        return 0
    else:
        print(f"\n💥 递归实验部分失败!")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())