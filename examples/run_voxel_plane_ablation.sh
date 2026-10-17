#!/usr/bin/env bash
# 依次运行停车场 original voxel-plane MCMC 的五组损失消融实验。
# 输入：图像/COLMAP、深度、weight map、cgal_voxel_plane_anchors.ply。
# 输出：OUTPUT_DIR 下五个独立目录，包含模型、train.log 和 command.sh。
# 将本文件放在 gsplat/examples，与 train_voxel_plane_mcmc.py 同目录。
# 在已配置 CUDA/gsplat 的训练环境运行：
#   cd /code/gsplat/examples
#   CUDA_VISIBLE_DEVICES=0 bash run_voxel_plane_ablation.sh
# 可选覆盖：DATA_DIR=/lutu/parking01_section OUTPUT_DIR=/lutu/ablation_v2
#           PYTHON_BIN=python TRAINER=/code/gsplat/examples/train_voxel_plane_mcmc.py
# 任一实验失败立即停止；使用新的 OUTPUT_DIR 重跑，避免覆盖旧结果。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-/lutu/parking01_section}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_DIR/ablation_voxel_plane_original}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAINER="${TRAINER:-$SCRIPT_DIR/train_voxel_plane_mcmc.py}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

[[ -f "$TRAINER" ]] || { echo "Missing trainer: $TRAINER" >&2; exit 1; }
[[ -f "$DATA_DIR/cgal_voxel_plane_anchors.ply" ]] || {
    echo "Missing plane anchors under $DATA_DIR" >&2
    exit 1
}
# 检查全部目标目录后再开始，防止执行到中途才发现已有结果。
for name in A_base B_knn C_pgsr_normal D_pgsr_ncc E_all; do
    [[ ! -e "$OUTPUT_DIR/$name" ]] || {
        echo "Result already exists: $OUTPUT_DIR/$name; choose a new OUTPUT_DIR." >&2
        exit 1
    }
done

# 基础训练器内部固定 seed=42（单 GPU），每组启动独立 Python 进程。
COMMON=(
    original
    --data-dir "$DATA_DIR"
    --data-factor 1 --port 8000 --disable-viewer
    --sh-degree 0 --max-steps 30000 --save-ply
    --depth-loss --depth-refine-lambda 0.01 --depth-lambda 0.05
    --strategy.cap-max 500000
    --strategy.refine-start-iter 1000
    --strategy.refine-every 300
    --strategy.refine-stop-iter 27000
    --strategy.plane-anchor-ply "$DATA_DIR/cgal_voxel_plane_anchors.ply"
    --opacity-reg 0.01 --scale-reg 0.01
    --anchor-normal-lambda 0.1
    --plane-scale-lambda 0.01 --anchor-scale-lambda 0.02
    --pgsr-start-iter 5000 --pgsr-warmup-steps 5000
    --pgsr-every 2 --pgsr-samples 4096
)

run() {
    local name="$1"
    shift
    local result="$OUTPUT_DIR/$name"
    local started=$SECONDS
    local command=("$PYTHON_BIN" -u "$TRAINER" "${COMMON[@]}"
                   --result-dir "$result" "$@")
    mkdir -p "$result"
    printf '%q ' "${command[@]}" > "$result/command.sh"
    printf '\n' >> "$result/command.sh"
    printf '\nStarting %s\n' "$name"
    "${command[@]}" 2>&1 | tee "$result/train.log"
    printf 'Completed %s in %s seconds\n' "$name" "$((SECONDS - started))" \
        | tee -a "$result/train.log"
}

run A_base \
    --knn-normal-lambda 0 --knn-plane-lambda 0 --no-pgsr-loss

run B_knn \
    --knn-normal-lambda 5 --knn-plane-lambda 0.5 --no-pgsr-loss

run C_pgsr_normal \
    --knn-normal-lambda 0 --knn-plane-lambda 0 --pgsr-loss \
    --pgsr-sv-lambda 0.015 --pgsr-mv-lambda 0

run D_pgsr_ncc \
    --knn-normal-lambda 0 --knn-plane-lambda 0 --pgsr-loss \
    --pgsr-sv-lambda 0 --pgsr-mv-lambda 0.15

run E_all \
    --knn-normal-lambda 5 --knn-plane-lambda 0.5 --pgsr-loss \
    --pgsr-sv-lambda 0.015 --pgsr-mv-lambda 0.15

printf '\nAll five experiments completed: %s\n' "$OUTPUT_DIR"
