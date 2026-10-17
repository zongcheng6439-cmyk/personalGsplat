#!/usr/bin/env bash
set -euo pipefail

# Run inside the gsplat Docker container. Environment variables can override
# the defaults without editing this file.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${GSPLAT_DATA_DIR:-/lutu/parking01_section}"
RESULT_DIR="${GSPLAT_RESULT_DIR:-/lutu/parking01_section/gsplat_2dgs_edge_depth}"
MAX_STEPS="${GSPLAT_MAX_STEPS:-30000}"
SH_DEGREE="${GSPLAT_SH_DEGREE:-0}"
DEPTH_LAMBDA="${GSPLAT_DEPTH_LAMBDA:-0.05}"
DIST_LAMBDA="${GSPLAT_DIST_LAMBDA:-0.01}"
DIST_START_ITER="${GSPLAT_DIST_START_ITER:-2000}"
NORMAL_LAMBDA="${GSPLAT_NORMAL_LAMBDA:-0.01}"
NORMAL_START_ITER="${GSPLAT_NORMAL_START_ITER:-4000}"
PLY_EVERY="${GSPLAT_PLY_EVERY:-3000}"

exec python train_2dgs_parking.py \
    --data-dir "${DATA_DIR}" \
    --result-dir "${RESULT_DIR}" \
    --data-factor 1 \
    --max-steps "${MAX_STEPS}" \
    --sh-degree "${SH_DEGREE}" \
    --depth-lambda "${DEPTH_LAMBDA}" \
    --dist-lambda "${DIST_LAMBDA}" \
    --dist-start-iter "${DIST_START_ITER}" \
    --normal-lambda "${NORMAL_LAMBDA}" \
    --normal-start-iter "${NORMAL_START_ITER}" \
    --ply-every "${PLY_EVERY}" \
    "$@"
