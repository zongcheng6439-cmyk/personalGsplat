 python3 utils/data_process/create_validation_ids.py \
        --data-dir /media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/ \
        --per-group 50 \
        --texture-threshold 0.1


python utils/data_process/cgal_plane_segmentation/run.py \
        --input /media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/sparse/0/extracted.ply \
        --output /media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/cgal_planes.ply \
        --anchor-output /media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/cgal_voxel_plane_anchors.ply \
        --anchor-voxel-size 0.1 \
        --overwrite
       