cd /code/gsplat
source /opt/gsplat-venv/bin/activate

python train_block_densification.py voxel_plane \
    --data-dir /lutu/parking01/ \
    --coarse-dir /lutu/parking01/citygs_weighted/coarse \
    --block-cameras-json /lutu/parking01/partition/partitions-dim_2_2_visibility_0.2/cameras-000_000.json \
    --result-dir /lutu/parking01/citygs_weighted/block/000_000 \
    --data-factor 1 \
    --test-every 0 \
    --max-steps 300000 \
    --save-ply \
    --disable-viewer \
    --sh-degree 0 \
    --depth-loss \
    --depth-refine-lambda 0.01 \
    --depth-lambda 0.05 \
    --strategy.noise_scale 30 \
    --strategy.cap-max 3004855 \
    --strategy.refine-start-iter 3000 \
    --strategy.refine-stop-iter 270000 \
    --strategy.refine-every 1500 \
    --strategy.plane-anchor-ply /lutu/parking01/cgal_voxel_plane_anchors.ply 
