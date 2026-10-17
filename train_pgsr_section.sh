#!/usr/bin/env bash
# parking01_section + PGSR 几何损失（--pgsr-loss）。
# 基于 train_section.sh 中 parking01_section 的 original 命令，仅新增 PGSR
# 参数并更换结果目录，便于与 gsplat_voxel_plane_mcmc 基线 A/B。
#
# 运行方式（与 train_section.sh 相同，先起容器）：
#   sudo docker run -it --rm --shm-size 8G --gpus all \
#       --entrypoint /bin/bash \
#       -v /media/zc/SSD/GS/lutu/:/lutu/ \
#       -v /home/zc/code/aibee/gsplat:/code/gsplat \
#       harbor.aibee.cn/robot/gsplat:V2
#   source /opt/gsplat-venv/bin/activate && cd /code/gsplat/ && python setup.py install
#   bash train_pgsr_section.sh
#
# 注意：训练器要求恰好一张可见 GPU，按需设置 CUDA_VISIBLE_DEVICES。
# PGSR 多视图项首次运行会在数据目录生成 ncc_source_views.json 共视缓存。

set -euo pipefail
cd "$(dirname "$0")/examples"

# ---------- original 固定锚点 + PGSR ----------
# python train_voxel_plane_mcmc.py original \
#    --data-dir /lutu/parking01_section/ \
#    --result-dir /lutu/parking01_section/gsplat_voxel_plane_mcmc_pgsr \
#    --data-factor 1 \
#    --sh-degree 0 \
#    --max-steps 30000 \
#    --save-ply \
#    --depth-loss \
#    --depth-refine-lambda 0.01 \
#    --depth-lambda 0.05 \
#    --strategy.cap-max 300000 \
#    --strategy.refine-start-iter 1000 \
#    --strategy.refine-every 300 \
#    --strategy.refine-stop-iter 30000 \
#    --strategy.plane-anchor-ply /lutu/parking01_section/cgal_voxel_plane_anchors.ply \
#    --pgsr-loss \
#    --pgsr-sv-lambda 0.015 \
#    --pgsr-mv-lambda 0.15 \
#    --pgsr-start-iter 5000 \
#    --pgsr-warmup-steps 5000 \
#    --pgsr-every 2 \
#    --pgsr-samples 4096 \
#    --pgsr-vis \
#    --pgsr-vis-count 10

python train_voxel_plane_mcmc.py original \
   --data-dir /B3/section_0907/ \
   --result-dir /B3/section_0907/gsplat_voxel_plane_mcmc_pgsr \
   --data-factor 2 \
   --sh-degree 0 \
   --max-steps 200000 \
   --save-ply \
   --depth-loss \
   --depth-refine-lambda 0.01 \
   --depth-lambda 0.05 \
   --strategy.cap-max 300000 \
   --strategy.refine-start-iter 2000 \
   --strategy.refine-every 500 \
   --strategy.refine-stop-iter 180000 \
   --strategy.plane-anchor-ply /B3/section_0907/cgal_voxel_plane_anchors.ply \
   --pgsr-loss \
   --pgsr-sv-lambda 0.015 \
   --pgsr-mv-lambda 0.15 \
   --pgsr-start-iter 10000 \
   --pgsr-warmup-steps 20000 \
   --pgsr-every 2 \
   --pgsr-samples 4096 \
   --pgsr-vis \
   --pgsr-vis-count 10
# ---------- A/B 对照 1：同参数关闭 PGSR（基线，可选）----------
# python train_voxel_plane_mcmc.py original \
#    --data-dir /lutu/parking01_section/ \
#    --result-dir /lutu/parking01_section/gsplat_voxel_plane_mcmc_ab_base \
#    --data-factor 1 --sh-degree 0 --max-steps 30000 --save-ply \
#    --depth-loss --depth-refine-lambda 0.01 --depth-lambda 0.05 \
#    --strategy.cap-max 300000 \
#    --strategy.refine-start-iter 1000 \
#    --strategy.refine-every 300 \
#    --strategy.refine-stop-iter 30000 \
#    --strategy.plane-anchor-ply /lutu/parking01_section/cgal_voxel_plane_anchors.ply

# ---------- A/B 对照 2：PGSR 只开单视图法向（可选）----------
# 在上面主命令基础上改两行：
#    --pgsr-mv-lambda 0 \
#    --result-dir /lutu/parking01_section/gsplat_voxel_plane_mcmc_pgsr_sv_only

# ---------- A/B 对照 3：PGSR 与原 NCC 同时开启对比（可选）----------
# 在主命令基础上追加（多视图两项会同时生效，通常只用于对比）：
#    --ncc-loss --ncc-lambda 0.01 --ncc-start-iter 5000 \
#    --ncc-warmup-steps 5000 --ncc-every 4 --ncc-samples 1024
