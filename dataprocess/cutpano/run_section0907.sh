#!/usr/bin/env bash
# 用途：为已复制的 section0907 全景图生成标定后的 pose、18 视角 PNG，并验证投影。
# 输入：85 服务器 B3_Full/pano_section0907、imu_pose.txt 及本目录原始 pose 参考。
# 输出：B3_Full/section0907_poses.json、section0907_perspective_2048/。
# 运行（任意目录）：bash /home/zc/code/aibee/gsplat/dataprocess/cutpano/run_section0907.sh
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR=/media/zc/SSD/ar/ABXBJQ_BJ/B3/B3_Full
python3 "$SCRIPT_DIR/prepare_section_poses.py" \
  --imu "$DATA_DIR/imu_pose.txt" \
  --reference "$SCRIPT_DIR/pano_pose_reference.txt" \
  --section-images "$SCRIPT_DIR/images.txt" \
  --output "$DATA_DIR/section0907_poses.json"
python3 "$SCRIPT_DIR/main.py" \
  --input "$DATA_DIR/pano_section0907" \
  --output "$DATA_DIR/section0907_perspective_2048" \
  --size 2048 --workers 6 \
  --pose-file "$DATA_DIR/section0907_poses.json"
python3 "$SCRIPT_DIR/validate_section_views.py" \
  --output "$DATA_DIR/section0907_perspective_2048" \
  --poses "$DATA_DIR/section0907_poses.json" \
  --legacy-images /media/zc/SSD/ar/ABXBJQ_BJ/B3/section_0907/images
