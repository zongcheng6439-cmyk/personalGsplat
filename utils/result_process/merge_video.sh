
ffmpeg \
-i /media/zc/SSD/GS/lutu/parking01_section/gsplat_latest_mcmc/video_100w.mp4 \
-i /media/zc/SSD/GS/lutu/parking01_section/gsplat_voxel_plane_mcmc_pgsr/video.mp4 \
-filter_complex \
"[0:v]drawtext=text='mcmc':x=10:y=10:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[l]; \
 [1:v]drawtext=text='ours':x=10:y=10:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[r]; \
 [l][r]hstack=inputs=2" \
-c:v libx264 -crf 18 /media/zc/SSD/GS/lutu/parking01_section/gsplat_voxel_plane_mcmc_pgsr/compare_mcmc.mp4


ffmpeg \
-i /media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/gsplat_latest_mcmc/video.mp4 \
-i /media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/gsplat_voxel_plane_mcmc/video.mp4 \
-filter_complex \
"[0:v]drawtext=text='mcmc':x=10:y=10:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[l]; \
 [1:v]drawtext=text='Ours':x=10:y=10:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[r]; \
 [l][r]hstack=inputs=2" \
-c:v libx264 -crf 18 /media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/gsplat_voxel_plane_mcmc/compare_mcmc.mp4


ffmpeg \
-i /media/zc/SSD/GS/lutu/parking01/gsplat_0717/video.mp4 \
-i /media/zc/SSD/GS/lutu/parking01/gsplat_0917/video.mp4 \
-filter_complex \
"[0:v]drawtext=text='mcmc':x=10:y=10:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[l]; \
 [1:v]drawtext=text='Ours':x=10:y=10:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[r]; \
 [l][r]hstack=inputs=2" \
-c:v libx264 -crf 18 /media/zc/SSD/GS/lutu/parking01/gsplat_0917/result_compare.mp4



ffmpeg \
-i /media/zc/SSD/GS/lutu/parking01_section/gsplat_weight_anchor_mcmc/video_50w.mp4 \
-i /media/zc/SSD/GS/lutu/parking01_section/ltu.mp4 \
-filter_complex \
"[0:v]drawtext=text='Ours':x=10:y=10:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[l]; \
 [1:v]drawtext=text='Ltu':x=10:y=10:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[r]; \
 [l][r]hstack=inputs=2" \
-c:v libx264 -crf 18 /media/zc/SSD/GS/lutu/parking01_section/gsplat_weight_anchor_mcmc/compare_lutu.mp4


ffmpeg \
-i /media/zc/SSD/ar/AIBEE_16/L16/showroom/mcmc_depth_1024/point_cloud.mp4 \
-i /media/zc/SSD/ar/AIBEE_16/L16/showroom/mcmc_depth_1024_edge_penalty/point_cloud.mp4 \
-i /media/zc/SSD/ar/AIBEE_16/L16/showroom/mcmc_depth_1024_edge/point_cloud.mp4 \
-filter_complex \
"[0:v]drawtext=text='mcmc':x=10:y=10:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[l]; \
 [1:v]drawtext=text='mcmc_edge_penalty':x=10:y=10:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[r]; \
 [2:v]drawtext=text='mcmc_edge':x=10:y=10:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.5[t]; \
 [l][r][t]vstack=inputs=3" \
-c:v libx264 -crf 18 /media/zc/SSD/ar/AIBEE_16/L16/showroom/mcmc_depth_1024_edge_penalty/three_way_compare.mp4