#!/usr/bin/env bash
set -euo pipefail

# Large-scene pipeline: prepare edge-filtered depth supervision and voxel-plane
# anchors, run two coarse steps per image, then train every block with the
# latest Voxel Plane Texture MCMC before merging the block PLYs.
# Input: prepared scene images/COLMAP/depth and plane anchors; output: coarse,
# per-block models/logs and merged PLY under RESULT_DIR.
# From the repository root in the training environment:
#   bash utils/train/train_citygs_mcmc.sh /lutu/parking01 /lutu/parking01/run_pgsr all
# Only train existing partitions, with PGSR NCC and no single-view normal loss:
#   PGSR_SV_LAMBDA=0 bash utils/train/train_citygs_mcmc.sh /lutu/parking01 /lutu/parking01/run_pgsr blocks
# PGSR_LOSS=0 disables PGSR; otherwise its default settings match the dispatcher.

TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UTIL_DIR="$(cd "$TRAIN_DIR/.." && pwd)"
REPO_DIR="$(cd "$UTIL_DIR/.." && pwd)"
DATA_PROCESS_DIR="$UTIL_DIR/data_process"
PARTITION_DIR="$UTIL_DIR/partition"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$REPO_DIR/examples:$PARTITION_DIR:$TRAIN_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 DATA_DIR [RESULT_DIR] [all|prepare|coarse|partition|blocks|merge] [PLANE_ANCHOR_PLY]" >&2
    exit 2
fi
DATA_DIR="$1"
RESULT_DIR="${2:-$DATA_DIR/citygs_voxel_plane}"
STAGE="${3:-all}"
PLANE_ANCHOR_PLY="${4:-${PLANE_ANCHOR_PLY:-$DATA_DIR/cgal_voxel_plane_anchors.ply}}"

DATA_FACTOR="${DATA_FACTOR:-1}"
COARSE_GPU="${COARSE_GPU:-0}"
TARGET_GS="${TARGET_GS:-10000000}"
BLOCK_X="${BLOCK_X:-2}"
BLOCK_Y="${BLOCK_Y:-2}"
BLOCK_DIR="${BLOCK_DIR:-$DATA_DIR/partition/partitions-dim_${BLOCK_X}_${BLOCK_Y}_visibility_0.2}"
COARSE_DIR="$RESULT_DIR/coarse"


run_stage() {
    [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]
}

count_images() {
    local image_dir="$DATA_DIR/images"
    [[ -d "$image_dir" ]] || {
        echo "Missing image directory: $image_dir" >&2
        return 1
    }
    find "$image_dir" -type f \( \
        -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \
        -o -iname '*.tif' -o -iname '*.tiff' \
    \) -print | wc -l
}

if run_stage prepare; then
    prepare_args=(
        --data-dir "$DATA_DIR"
        --data-factor "$DATA_FACTOR"
        --target-gs "$TARGET_GS"
        --voxel-size "${ANCHOR_VOXEL_SIZE:-0.1}"
        --depth-dir "${RAW_DEPTH_DIR:-depths_render_color}"
        --depth-edge-abs-threshold "${DEPTH_EDGE_ABS_THRESHOLD:-0.2}"
        --depth-edge-relative-threshold "${DEPTH_EDGE_RELATIVE_THRESHOLD:-0.02}"
        --depth-edge-dilation-kernel "${DEPTH_EDGE_DILATION_KERNEL:-7}"
    )
    prepare_args+=(--plane-anchor-ply "$PLANE_ANCHOR_PLY")
    "$PYTHON_BIN" "$DATA_PROCESS_DIR/prepare_citygs_data.py" "${prepare_args[@]}"
fi

if run_stage coarse; then
    IMAGE_COUNT="$(count_images)"
    COARSE_STEPS="$((IMAGE_COUNT * 2))"
    ((COARSE_STEPS > 0)) || {
        echo "No training images found under $DATA_DIR/images" >&2
        exit 1
    }
    echo "Coarse steps: $COARSE_STEPS (2 x $IMAGE_COUNT images)"

    latest_ply=""
    if [[ -d "$COARSE_DIR/ply" ]]; then
        latest_ply="$(find "$COARSE_DIR/ply" -maxdepth 1 -type f -size +0c -name 'point_cloud_*.ply' -print | sort -V | tail -n 1)"
    fi

    if [[ -n "$latest_ply" && "${FORCE_COARSE:-0}" != "1" ]]; then
        echo "Reuse existing coarse PLY: $latest_ply"
    else
        CUDA_VISIBLE_DEVICES="$COARSE_GPU" "$PYTHON_BIN" "$REPO_DIR/examples/simple_trainer.py" mcmc \
            --data-dir "$DATA_DIR" \
            --result-dir "$COARSE_DIR" \
            --data-factor "$DATA_FACTOR" \
            --test-every 0 \
            --max-steps "$COARSE_STEPS" \
            --save-steps "$COARSE_STEPS" \
            --ply-steps "$COARSE_STEPS" \
            --sh-degree 0 \
            --depth-loss \
            --depth-lambda "${DEPTH_LAMBDA:-0.05}" \
            --depth-refine-lambda "${DEPTH_LAMBDA:-0.05}" \
            --disable-viewer \
            --save-ply \
            --means-lr 1.6e-5 \
            --strategy.refine-start-iter "$COARSE_STEPS" \
            --strategy.refine-stop-iter "$COARSE_STEPS" \
            --strategy.noise-injection-stop-iter 0

        latest_ply="$(find "$COARSE_DIR/ply" -maxdepth 1 -type f -size +0c -name 'point_cloud_*.ply' -print | sort -V | tail -n 1)"
    fi
    [[ -n "$latest_ply" ]] || { echo "No coarse PLY found" >&2; exit 1; }
    echo "Coarse PLY: $latest_ply"
fi

if run_stage partition; then
    camera_block_count=0
    init_block_count=0
    if [[ -d "$BLOCK_DIR" ]]; then
        camera_block_count="$(find "$BLOCK_DIR" -maxdepth 1 -type f -name 'cameras-*.json' | wc -l)"
    fi
    if [[ -d "$BLOCK_DIR/block_init" ]]; then
        init_block_count="$(find "$BLOCK_DIR/block_init" -maxdepth 1 -type f -name 'point_cloud_*.ply' 2>/dev/null | wc -l)"
    fi
    if [[ -f "$BLOCK_DIR/partitions.pt" \
          && "$camera_block_count" -gt 0 \
          && "$camera_block_count" -eq "$init_block_count" \
          && "${PARTITION_OVERWRITE:-0}" != "1" ]]; then
        echo "Reuse existing partition: $BLOCK_DIR"
    else
        CUDA_VISIBLE_DEVICES="$COARSE_GPU" "$PYTHON_BIN" "$PARTITION_DIR/partition.py" \
            --data_dir "$DATA_DIR" \
            --data_factor "$DATA_FACTOR" \
            --block_dim "$BLOCK_X" "$BLOCK_Y" \
            --ckpt_dir "$COARSE_DIR/ckpts" \
            --gs_occlusion_cell "${GS_OCCLUSION_CELL:-4}" \
            --gs_depth_tolerance "${GS_DEPTH_TOLERANCE:-0.1}" \
            --gs_visibility_chunk_size "${GS_VISIBILITY_CHUNK_SIZE:-1000000}"
    fi
fi

if run_stage blocks; then
    case "${PGSR_LOSS:-1}" in
        1) pgsr_flag=--pgsr_loss ;;
        0) pgsr_flag=--no-pgsr_loss ;;
        *) echo "PGSR_LOSS must be 0 or 1" >&2; exit 2 ;;
    esac
    "$PYTHON_BIN" "$TRAIN_DIR/train_all_partitions.py" \
        --method voxel_plane \
        --data_dir "$DATA_DIR" \
        --block_dir "$BLOCK_DIR" \
        --coarse_dir "$COARSE_DIR" \
        --result_dir "$RESULT_DIR" \
        --plane_anchor_ply "$PLANE_ANCHOR_PLY" \
        --anchor_voxel_size "${ANCHOR_VOXEL_SIZE:-0.1}" \
        --initial_plane_thickness "${INITIAL_PLANE_THICKNESS:-0.0001}" \
        --plane_normal_tolerance "${PLANE_NORMAL_TOLERANCE:-0.05}" \
        --anchor_max_distance "${ANCHOR_MAX_DISTANCE:-0.5}" \
        --scene_target_gs "$TARGET_GS" \
        --bbox_enlarge "${BLOCK_BBOX_ENLARGE:-0.0}" \
        --python "$PYTHON_BIN" \
        --data_factor "$DATA_FACTOR" \
        --steps_per_image "${STEPS_PER_IMAGE:-100}" \
        --means_lr "${BLOCK_MEANS_LR:-1.6e-5}" \
        --noise_scale "${NOISE_SCALE:-10.0}" \
        --sh_degree 0 \
        --depth_lambda "${DEPTH_LAMBDA:-0.1}" \
        --depth_refine_lambda "${DEPTH_REFINE_LAMBDA:-0.05}" \
        --texture_weight "${TEXTURE_WEIGHT:-2.0}" \
        --grad_weight "${GRAD_WEIGHT:-1.0}" \
        "$pgsr_flag" \
        --pgsr_sv_lambda "${PGSR_SV_LAMBDA:-0.015}" \
        --pgsr_mv_lambda "${PGSR_MV_LAMBDA:-0.15}" \
        --pgsr_start_iter "${PGSR_START_ITER:-5000}" \
        --pgsr_warmup_steps "${PGSR_WARMUP_STEPS:-5000}" \
        --pgsr_every "${PGSR_EVERY:-2}" \
        --pgsr_samples "${PGSR_SAMPLES:-4096}" \
        --min_free_gb "${MIN_FREE_GB:-6}"
fi

if run_stage merge; then
    "$PYTHON_BIN" "$PARTITION_DIR/merge_block_ply.py" \
        --path "$RESULT_DIR" \
        --block_dir "$BLOCK_DIR" \
        --block_dim "$BLOCK_X" "$BLOCK_Y" \
        --sh 0
fi
