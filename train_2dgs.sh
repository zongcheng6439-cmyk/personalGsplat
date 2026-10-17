#!/usr/bin/env bash
set -euo pipefail

# Run this script inside the gsplat V2 container. All paths can be overridden:
#   DATA_DIR=/lutu/parking01_section/ RESULT_DIR=/lutu/parking01_section//result ANCHOR_PLY=/data/anchors.ply \
#     bash /code/gsplat/train_2dgs.sh
GSPLAT_DIR="${GSPLAT_DIR:-/code/gsplat}"
DATA_DIR="${DATA_DIR:-/B3/section_0907}"
RESULT_DIR="${RESULT_DIR:-${DATA_DIR}/gsplat_voxel_plane_2dgs_mcmc}"
ANCHOR_PLY="${ANCHOR_PLY:-${DATA_DIR}/cgal_voxel_plane_anchors.ply}"
GPU_ID="${GPU_ID:-0}"

if [[ -f /opt/gsplat-venv/bin/activate ]]; then
    source /opt/gsplat-venv/bin/activate
fi

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Data directory does not exist: ${DATA_DIR}" >&2
    exit 1
fi
if [[ ! -f "${ANCHOR_PLY}" ]]; then
    echo "Plane-anchor PLY does not exist: ${ANCHOR_PLY}" >&2
    exit 1
fi

cd "${GSPLAT_DIR}"
mkdir -p "${RESULT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -u examples/train_voxel_plane_2dgs_mcmc.py \
    --data-dir "${DATA_DIR}" \
    --result-dir "${RESULT_DIR}" \
    --data-factor 1 \
    --test-every 0 \
    --sh-degree 0 \
    --max-steps 30000 \
    --ply-every 3000 \
    --port 8300 \
    --depth-loss \
    --depth-lambda 0.05 \
    --normal-loss \
    --normal-lambda 0.01 \
    --normal-start-iter 4000 \
    --dist-loss \
    --dist-lambda 0.01 \
    --dist-start-iter 2000 \
    --strategy.plane-anchor-ply "${ANCHOR_PLY}" \
    --strategy.plane-position-lambda 0.1 \
    --strategy.plane-normal-lambda 0.01 \
    --strategy.plane-position-beta 0.02 \
    --strategy.initial-tangent-scale-multiplier 1.5 \
    --strategy.cap-max 500000 \
    --strategy.min-opacity 0.005 \
    --strategy.refine-start-iter 1000 \
    --strategy.refine-every 300 \
    --strategy.refine-stop-iter 0 \
    --strategy.noise-injection-stop-iter -1 \
    --strategy.active-noise-steps 30000 \
    --strategy.noise-scale 10 \
    --strategy.use-opacity-score \
    --strategy.use-grad-score \
    --strategy.no-use-texture-score \
    --strategy.opacity-weight 1 \
    --strategy.grad-weight 1 \
    --strategy.texture-weight 0 \
    --strategy.keep-one-per-anchor
