#!/usr/bin/env python3
"""
将MainDatasimu2_data按照序号分类为good和bad两个子集
"""
import shutil
from pathlib import Path

# 不好的序号列表
bad_indices = {
    473, 475, 476, 485, 486, 489, 495, 499,
    979, 980, 981, 984, 985, 987, 990, 992, 993, 997, 998
}

# 基础目录
base_dir = Path("MainDatasimu2_data")

# 所有子目录
subdirs = [
    "input", "target",
    "output_full", "output_simple", "output_full_old", "output_simple_old",
    "inputpfda", "inputpfdb", "inputefr", "outputbaseline"
]

# 创建目标目录结构
bad_dir = base_dir / "bad"
good_dir = base_dir / "good"

for target_base in [bad_dir, good_dir]:
    for subdir in subdirs:
        (target_base / subdir).mkdir(parents=True, exist_ok=True)

print("创建目录结构完成")

# 遍历所有子目录，按序号分类移动文件
for subdir in subdirs:
    source_dir = base_dir / subdir
    if not source_dir.exists():
        print(f"⚠️ 跳过不存在的目录: {subdir}")
        continue

    files = sorted(source_dir.glob("*.h5"))
    print(f"\n处理目录: {subdir} ({len(files)} 个文件)")

    bad_count = 0
    good_count = 0

    for file in files:
        # 从文件名提取序号 (例如: composed_00473_bg_flare.h5 → 473)
        try:
            # 尝试多种文件名格式
            parts = file.stem.split('_')
            idx = None
            for part in parts:
                if part.isdigit():
                    idx = int(part)
                    break

            if idx is None:
                print(f"⚠️ 无法解析序号: {file.name}")
                continue

            # 判断分类
            if idx in bad_indices:
                target = bad_dir / subdir / file.name
                bad_count += 1
            else:
                target = good_dir / subdir / file.name
                good_count += 1

            # 移动文件
            shutil.move(str(file), str(target))

        except Exception as e:
            print(f"❌ 处理文件失败 {file.name}: {e}")

    print(f"  ✅ Bad: {bad_count}, Good: {good_count}")

print("\n" + "="*50)
print("分类完成！")
print(f"Bad目录: {bad_dir}")
print(f"Good目录: {good_dir}")
