#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-/data}"
RESULT_DIR="${2:-$DATA_DIR/score_ablation_fastgs}"
cd "$(dirname "$0")/.."

python simple_trainer_texture.py texture_mcmc  \
  --data_dir "$DATA_DIR" --result_dir "$RESULT_DIR" \
  --data_factor 1 --sh_degree 0 --save_ply --disable_viewer --max_steps 30000 \
  --depth_loss --depth_refine_lambda 0.01 --depth_lambda 0.05 --means_lr 1.6e-6 \
  --top_k 3 --loss_threshold 0.1 \
  --strategy.cap_max 2000000 --strategy.refine_start_iter 2000 \
  --strategy.refine_stop_iter 25000 --strategy.refine_every 100 \
  --strategy.noise_injection_stop_iter -1 --strategy.active_noise_steps 80 \
  --strategy.add_hot_fraction 0.1 --strategy.add_hot_probability 0.7 \
  --strategy.texture_update_every 2 \
  --strategy.use_fastgs_score --strategy.fastgs_weight 1.0 \
  --strategy.surface_anchor_constraint --strategy.surface_anchor_max_distance 0.1 \
  --strategy.score_debug_dir "$RESULT_DIR/mcmc_debug"
