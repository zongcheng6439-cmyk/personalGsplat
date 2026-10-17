#!/usr/bin/env bash
# 用途：挂载 cutpano 数据，用 LetsGo 官方光栅器生成深度，不训练 GS。
# 输入：数据根目录(images、sparse/0 的 txt 和 extracted.ply)；输出：depths_render_color/imageN/*.png。
# 在任意目录运行：bash /home/zc/code/aibee/gsplat/dataprocess/renderDepth/run_depth.sh /media/zc/SSD/ar/ABXBJQ_BJ/B3/B3_Full/section0907_perspective_2048
# 可选：在目录后追加 --limit 3（试跑）、--scale-multiplier 1.0、--normalize-alpha。
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_dir="$(realpath "${1:?请提供 cutpano 数据根目录}")"
shift
docker run --rm --gpus all --shm-size 8g \
  --user "$(id -u):$(id -g)" \
  -e PYTHONDONTWRITEBYTECODE=1 -e OPENBLAS_NUM_THREADS=1 \
  -v "$script_dir:/code:ro" -v "$data_dir:/data" \
  gsplat-letsgo-depth:local render_depth.py --data-dir /data "$@"
