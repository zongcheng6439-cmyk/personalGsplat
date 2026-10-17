#!/usr/bin/env bash

# run docker 
sudo docker run -it --rm --shm-size 8G  --gpus all \
   --entrypoint /bin/bash \
    -v /media/zc/SSD/GS/lutu/:/lutu/ \
    -v /media/zc/SSD/ar/AIBEE_16/L16/:/L16/ \
    -v /media/zc/SSD/ar/ABXBJQ_BJ/B3/:/B3/ \
    -v /media/zc/SSD/ar/ABXBJQ_BJ/B3/B3_Full/section0907_perspective_2048/:/B3_section/ \
    -v /home/zc/code/aibee/gsplat:/code/gsplat \
    -p 8300:8300 \
    harbor.aibee.cn/robot/gsplat:V2

    source /opt/gsplat-venv/bin/activate && cd /code/gsplat/ && pip uninstall gsplat && python setup.py install



 python simple_trainer_texture_default.py texture_default \
    --data_dir /small_section/ \
    --result_dir /small_section/gsplat_latest_texture_mcmc_depth_anchor/ \
    --data_factor 2 --sh_degree 0 --save_ply --max_steps 100000 --port 8090 \
    --depth_loss --depth_refine_lambda 0.01 --depth_lambda 0.05 \
    --strategy.prune_opa 0.005 \
    --strategy.refine_every 300 \
    --strategy.refine_start_iter 3000 \
    --strategy.refine_stop_iter 85000 \
    --strategy.texture_weight 2 \
    --strategy.grad_weight 0.25 \
    --strategy.surface_anchor_constraint \
    --strategy.surface_anchor_max_distance 0.1 \
    --strategy.score_debug_dir /small_section/gsplat_latest_texture_mcmc_depth_anchor/mcmc_debug


 python simple_trainer_texture.py mcmc \
    --data_dir /B3/section_0907/ \
    --result_dir /B3/section_0907/gsplat_latest_mcmc/ \
    --data_factor 1 --sh_degree 0 --save_ply --max_steps 200000 \
    --strategy.cap_max 500000 \
    --strategy.noise_injection_stop_iter -1 \
    --strategy.refine_every 2000 \
    --strategy.refine_start_iter 2000 \
    --strategy.refine_stop_iter 180000 

## train lutu parking 
    python simple_trainer_texture.py texture_mcmc \
    --data_dir /lutu/parking01_section/ \
    --result_dir /lutu/parking01_section/gsplat_latest_texture_mcmc_constraint/ \
    --data_factor 1 --sh_degree 0 --save_ply --max_steps 30000 \
    --depth_loss --depth_refine_lambda 0.01 --depth_lambda 0.05 \
    --strategy.cap_max 500000 \
    --strategy.noise_injection_stop_iter -1 \
    --strategy.active_noise_steps 30000 \
    --strategy.refine_every 300 \
    --strategy.refine_start_iter 1000 \
    --strategy.refine_stop_iter 25000 \
    --strategy.use_texture_score \
    --strategy.use_grad_score \
    --strategy.texture_weight 2 \
    --strategy.grad_weight 1 \
    --strategy.surface_anchor_constraint \
    --strategy.surface_anchor_max_distance 0.3  


  python simple_trainer_texture.py texture_mcmc \
    --data_dir /showroom/ \
    --result_dir /showroom/gsplat_latest_texture_mcmc_constraint/ \
    --data_factor 2 --sh_degree 0 --save_ply --max_steps 30000 --port 8080 \
    --depth_loss --depth_refine_lambda 0.01 --depth_lambda 0.05 \
    --strategy.cap_max 500000 \
    --strategy.noise_injection_stop_iter -1 \
    --strategy.active_noise_steps 100 \
    --strategy.refine_every 300 \
    --strategy.refine_start_iter 1000 \
    --strategy.refine_stop_iter 25000 \
    --strategy.texture_weight 2 \
    --strategy.grad_weight 1 \
    --strategy.surface_anchor_constraint \
    --strategy.surface_anchor_max_distance 0.3 




# weight anchor mcmc 
 python train_weighted_anchor_mcmc.py \
    --data_dir /lutu/parking01_section/ \
    --result_dir /lutu/parking01_section/gsplat_weight_anchor_mcmc/ \
    --data_factor 1 --sh_degree 0 --save_ply --max_steps 30000 \
    --depth_loss --depth_refine_lambda 0.01 --depth_lambda 0.05 \
    --strategy.cap_max 500000 \
    --strategy.noise_injection_stop_iter -1 \
    --strategy.refine_every 300 \
    --strategy.refine_start_iter 1000 \
    --strategy.refine_stop_iter 25000 \
    --strategy.anchor_ply /lutu/parking01_section/voxel_target_count.ply \
    --strategy.surface_anchor_constraint \
    --strategy.surface_anchor_max_distance 0.3 





 python train_weighted_anchor_mcmc.py \
    --data_dir /B3/section_0907/ \
    --result_dir /B3/section_0907/gsplat_weight_anchor_mcmc/ \
    --data_factor 2 --sh_degree 0 --save_ply --max_steps 200000 \
    --depth_loss --depth_refine_lambda 0.05 --depth_lambda 0.2 --port 8000\
    --strategy.cap_max 2000000 \
    --strategy.noise_injection_stop_iter -1 \
    --strategy.refine_every 1000 \
    --strategy.refine_start_iter 2000 \
    --strategy.refine_stop_iter 160000 \
    --strategy.anchor_ply /B3/section_0907/voxel_target_count.ply \
    --strategy.surface_anchor_max_distance 0.2


# plane gs 
 python train_voxel_plane_mcmc.py \
    --data_dir /B3/section_0907/ \
    --result_dir /B3/section_0907/gsplat_voxel_plane_mcmc/ \
    --data_factor 2 --sh_degree 0 --save_ply --max_steps 200000 \
    --depth_loss --depth_refine_lambda 0.05 --depth_lambda 0.2 --port 8000\
    --strategy.cap_max 2000000 \
    --strategy.noise_injection_stop_iter -1 \
    --strategy.refine_every 1000 \
    --strategy.refine_start_iter 2000 \
    --strategy.refine_stop_iter 160000 \
    --strategy.anchor_ply /B3/section_0907/voxel_target_count.ply \
    --strategy.surface_anchor_max_distance 0.2



python train_voxel_plane_mcmc.py original\
   --data-dir /B3/small_section/ \
   --result-dir /B3/small_section/gsplat_voxel_plane_mcmc \
   --data-factor 2  \
   --sh-degree 0 \
   --max-steps 200000 \
   --save-ply \
   --depth-loss \
   --depth-refine-lambda 0.01 \
   --depth-lambda 0.05 \
   --strategy.cap-max 3000000 \
   --strategy.refine-start-iter 2000 \
   --strategy.refine-every 700 \
   --strategy.refine-stop-iter 180000 \
   --strategy.anchor_max_distance 0.4 \
   --strategy.plane-normal-tolerance 0.03 \
   --strategy.plane-anchor-ply /B3/small_section/cgal_voxel_plane_anchors.ply 
   

python train_voxel_plane_mcmc.py original\
   --data-dir /L16/showroom/ \
   --result-dir /L16/showroom/gsplat_voxel_plane_mcmc \
   --data-factor 2  \
   --sh-degree 0 \
   --max-steps 30000 \
   --save-ply \
   --depth-loss \
   --depth-refine-lambda 0.01 \
   --depth-lambda 0.05 \
   --strategy.cap-max 1500000 \
   --strategy.refine-start-iter 1500 \
   --strategy.refine-every 500 \
   --strategy.refine-stop-iter 27000 \
   --strategy.anchor_max_distance 0.4 \
   --strategy.plane-normal-tolerance 0.03 \
   --strategy.plane-anchor-ply /L16/showroom/cgal_voxel_plane_anchors.ply 
   
   






python train_voxel_plane_mcmc.py original \
   --data-dir /lutu/parking01_section/\
   --result-dir  /lutu/parking01_section/gsplat_voxel_plane_mcmc_pgsr_ncc \
   --data-factor 1  \
   --sh-degree 0 \
   --max-steps 30000 \
   --save-ply \
   --depth-loss \
   --depth-refine-lambda 0.01 \
   --depth-lambda 0.05 \
   --strategy.cap-max 300000 \
   --strategy.refine-start-iter 1000 \
   --strategy.refine-every 300 \
   --strategy.refine-stop-iter 30000 \
   --strategy.plane-anchor-ply /lutu/parking01_section/cgal_voxel_plane_anchors.ply 


python train_voxel_plane_mcmc.py original\
   --data-dir /B3_section/ \
   --result-dir /B3_section/gsplat_voxel_plane_mcmc \
   --port 8300 \
   --data-factor 2  \
   --sh-degree 0 \
   --max-steps 200000 \
   --save-ply \
   --depth-loss \
   --depth-refine-lambda 0.01 \
   --depth-lambda 0.05 \
   --strategy.cap-max 3000000 \
   --strategy.refine-start-iter 5000 \
   --strategy.refine-every 1000 \
   --strategy.refine-stop-iter 180000 \
   --strategy.anchor_max_distance 0.3 \
   --strategy.plane-normal-tolerance 0.03 \
   --strategy.plane-anchor-ply /B3_section/cgal_voxel_plane_anchors.ply 
   


python train_voxel_plane_mcmc.py dynamic\
   --data-dir /lutu/parking01_section/\
   --result-dir /lutu/parking01_section/gsplat_voxel_plane_mcmc_dynamic_v1 \
   --data-factor 1 --port 8000 \
   --sh-degree 0 \
   --max-steps 30000 \
   --save-ply \
   --depth-loss \
   --depth-refine-lambda 0.01 \
   --depth-lambda 0.05 \
   --strategy.cap-max 500000 \
   --strategy.refine-start-iter 1000 \
   --strategy.refine-every 300 \
   --strategy.refine-stop-iter 27000 \
   --strategy.plane-anchor-ply /lutu/parking01_section/cgal_voxel_plane_anchors.ply 




python examples/train_voxel_plane_mcmc.py \
  --data-dir /B3/section_0907/ \
  --result-dir /B3/section_0907/ppisp_quick_test \
  --data-factor 4 \
  --sh-degree 0 \
  --max-steps 5000 \
  --disable-viewer \
  --post-processing ppisp \
  --no-ppisp-controller-distillation \
  --tb-save-image \
  --tb-every 100 \
  --strategy.plane-anchor-ply /B3/section_0907/cgal_voxel_plane_anchors.ply \
  --strategy.initial-plane-thickness 0.0001 \
  --strategy.plane-normal-tolerance 0.05 \
  --strategy.anchor-max-distance 0.5 \
  --strategy.cap-max 3000000 \
  --strategy.refine-start-iter 6000 \
  --strategy.refine-stop-iter 6000 \
  --strategy.noise-injection-stop-iter 0
