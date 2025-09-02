#!/usr/bin/env python3
"""
调试数据加载，确保读取到了正确的physics_method数据
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.utils.config_loader import ConfigLoader
from src.datasets.event_voxel_dataset import EventVoxelDataset
from src.data_processing.encode import load_h5_events

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("=== 调试数据加载过程 ===")
    
    # 1. 加载配置
    config_loader = ConfigLoader()
    config = config_loader.load_train_config("configs/train_config.yaml")
    
    # 2. 提取数据路径
    train_noisy_dir = config['loaders']['train_noisy_dir']
    train_clean_dir = config['loaders']['train_clean_dir']
    
    logger.info(f"输入数据路径: {train_noisy_dir}")
    logger.info(f"真值数据路径: {train_clean_dir}")
    
    # 3. 验证路径存在
    if not Path(train_noisy_dir).exists():
        logger.error(f"输入路径不存在: {train_noisy_dir}")
        return 1
    if not Path(train_clean_dir).exists():
        logger.error(f"真值路径不存在: {train_clean_dir}")
        return 1
    
    # 4. 查看目录内容
    noisy_files = list(Path(train_noisy_dir).glob("*.h5"))
    clean_files = list(Path(train_clean_dir).glob("*.h5"))
    
    logger.info(f"输入文件数量: {len(noisy_files)}")
    logger.info(f"真值文件数量: {len(clean_files)}")
    
    if len(noisy_files) > 0:
        logger.info(f"输入文件示例: {noisy_files[0].name}")
    if len(clean_files) > 0:
        logger.info(f"真值文件示例: {clean_files[0].name}")
    
    # 5. 创建数据集并验证文件对匹配
    try:
        dataset = EventVoxelDataset(
            noisy_events_dir=train_noisy_dir,
            clean_events_dir=train_clean_dir,
            sensor_size=(480, 640),
            segment_duration_us=20000,
            num_bins=8,
            num_segments=5
        )
        
        logger.info(f"数据集创建成功: {len(dataset)} 个样本")
        
        # 6. 测试读取第一个样本
        if len(dataset) > 0:
            logger.info("测试读取第一个样本...")
            
            # 获取第一个文件对的路径
            first_file_pair = dataset.file_pairs[0]
            noisy_path, clean_path = first_file_pair
            
            logger.info(f"第一个文件对:")
            logger.info(f"  输入文件: {noisy_path}")
            logger.info(f"  真值文件: {clean_path}")
            
            # 验证这确实是physics_method数据
            if "physics_method" in noisy_path and "physics_method" in clean_path:
                logger.info("✅ 确认读取的是physics_method数据")
            else:
                logger.warning("⚠️  读取的可能不是physics_method数据")
            
            # 读取events数据查看内容
            logger.info("读取events数据...")
            noisy_events = load_h5_events(noisy_path)
            clean_events = load_h5_events(clean_path)
            
            logger.info(f"输入events数量: {len(noisy_events)}")
            logger.info(f"真值events数量: {len(clean_events)}")
            
            if len(noisy_events) > 0:
                logger.info(f"输入events时间范围: {noisy_events[:, 0].min():.0f} - {noisy_events[:, 0].max():.0f} μs")
                logger.info(f"输入events空间范围: x=[{noisy_events[:, 1].min():.0f}, {noisy_events[:, 1].max():.0f}], y=[{noisy_events[:, 2].min():.0f}, {noisy_events[:, 2].max():.0f}]")
                logger.info(f"输入events极性分布: {(noisy_events[:, 3] == 1).sum()} 正事件, {(noisy_events[:, 3] != 1).sum()} 负事件")
            
            if len(clean_events) > 0:
                logger.info(f"真值events时间范围: {clean_events[:, 0].min():.0f} - {clean_events[:, 0].max():.0f} μs")
                logger.info(f"真值events空间范围: x=[{clean_events[:, 1].min():.0f}, {clean_events[:, 1].max():.0f}], y=[{clean_events[:, 2].min():.0f}, {clean_events[:, 2].max():.0f}]")
                logger.info(f"真值events极性分布: {(clean_events[:, 3] == 1).sum()} 正事件, {(clean_events[:, 3] != 1).sum()} 负事件")
            
            # 7. 测试数据集__getitem__
            logger.info("测试数据集采样...")
            sample = dataset[0]  # 第一个20ms段
            
            logger.info(f"采样结果:")
            logger.info(f"  raw voxel shape: {sample['raw'].shape}")
            logger.info(f"  label voxel shape: {sample['label'].shape}")
            logger.info(f"  raw voxel value range: [{sample['raw'].min():.3f}, {sample['raw'].max():.3f}]")
            logger.info(f"  label voxel value range: [{sample['label'].min():.3f}, {sample['label'].max():.3f}]")
            
            logger.info("✅ 数据加载测试完成，一切正常！")
        
    except Exception as e:
        logger.error(f"数据集创建失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())