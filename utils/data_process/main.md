
cd /home/zc/code/aibee/gsplat

python utils/data_process/prepare_citygs_data.py \
  --data-dir /media/zc/SSD/ar/ABXBJQ_BJ/B3/B3_Full/section0907_perspective_2048 \
  --depth-dir depths_render_color \
  --device cuda:0 \
  --workers 8 \
  --voxel-size 0.1 \
  --plane-anchor-ply /media/zc/SSD/ar/ABXBJQ_BJ/B3/B3_Full/section0907_perspective_2048/cgal_voxel_plane_anchors.ply

这会依次生成：
1. 人物 mask：masks/
2. 深度边缘过滤结果：depths_render_color_edge/
3. 纹理权重：weight_map/
4. CGAL 平面及 voxel-plane anchors。
必须先完成 depths_render_color/ 的深度渲染；这个脚本不会生成原始深度，缺少深度文件会退出。已完整生成的结果默认跳过。

渲染深度的脚本：
/home/zc/code/aibee/gsplat/dataprocess/renderDepth/run_depth.sh 