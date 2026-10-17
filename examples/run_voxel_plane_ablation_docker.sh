#!/usr/bin/env bash
# 用服务器 Docker 环境依次执行五组损失消融实验。
# 输入：本项目代码与 HOST_DATA_DIR 中已准备好的停车场数据。
# 输出：数据目录下 ablation_voxel_plane_original/{A_base,...,E_all}。
# 在 zc@192.168.88.85 上本项目根目录执行（不会自动同步代码）：
#   GPU=0 bash examples/run_voxel_plane_ablation_docker.sh
# 可覆盖 IMAGE、HOST_DATA_DIR、OUTPUT_DIR（容器内路径）、GPU。
# IMAGE 默认沿用项目 docker_run.bash；运行前检查本机是否存在该镜像。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
IMAGE="${IMAGE:-harbor.aibee.cn/robot/gsplat:V2}"
HOST_DATA_DIR="${HOST_DATA_DIR:-/media/zc/SSD/GS/lutu/parking01_section}"
OUTPUT_DIR="${OUTPUT_DIR:-/lutu/parking01_section/ablation_voxel_plane_original}"
GPU="${GPU:-0}"

[[ -f "$HOST_DATA_DIR/cgal_voxel_plane_anchors.ply" ]] || {
    echo "Missing prepared scene: $HOST_DATA_DIR" >&2
    exit 1
}
docker image inspect "$IMAGE" >/dev/null

docker run --rm --gpus "device=$GPU" --shm-size 8g \
    --mount "type=bind,src=$REPO_DIR,dst=/code/gsplat" \
    --mount "type=bind,src=$HOST_DATA_DIR,dst=/lutu/parking01_section" \
    --workdir /code/gsplat/examples \
    --env PYTHONPATH=/code/gsplat:/code/gsplat/examples \
    --env CUDA_VISIBLE_DEVICES=0 \
    --env DATA_DIR=/lutu/parking01_section \
    --env "OUTPUT_DIR=$OUTPUT_DIR" \
    --entrypoint /bin/bash "$IMAGE" -c '
        set -euo pipefail
        source /opt/gsplat-venv/bin/activate
        python -c "import gsplat; from pathlib import Path; p=Path(gsplat.__file__).resolve(); print(p); assert p.is_relative_to(Path(\"/code/gsplat\")), p"
        exec bash /code/gsplat/examples/run_voxel_plane_ablation.sh
    '
