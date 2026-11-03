# 深度格式 tag

`prepare_citygs_data.py` 使用 `--depth-tag` 显式选择格式，不按文件名猜测：

| tag | 输入 PNG | 解码方式 |
| --- | --- | --- |
| `la`（默认） | 8 位亮度 L + 有效性 A | 现有分段深度编码；A > 127 且 0 < L < 255 有效 |
| `uint16` | 单通道 16 位整数 | 深度值 / `--depth-scale`，默认 1000，即毫米转米；0 无效 |

`--depth-dir` 控制目录，`--depth-tag` 控制格式，二者独立。世界坐标和相机位姿不进行归一化。

## ScanNet++ 068ba2946c

```bash
cd /home/zc/code/gsplat-self/personalGsplat
python utils/data_process/prepare_citygs_data.py \
  --data-dir /media/zc/SSD/GS/scannetTpp/068ba2946c \
  --depth-dir depth \
  --depth-tag uint16 \
  --depth-scale 1000 \
  --device cuda:0 \
  --workers 8 \
  --voxel-size 0.05 \
  --point-ply /media/zc/SSD/GS/scannetTpp/068ba2946c/sparse/0/init.ply \
  --plane-anchor-ply /media/zc/SSD/GS/scannetTpp/068ba2946c/cgal_voxel_plane_anchors.ply
```

`--point-ply` 指定平面提取输入，省略时仍使用 `sparse/0/extracted.ply`。它不改变其他训练初始化点云的配置。人物检测和 CGAL 仍需各自环境依赖。

## 输出与训练

边缘检测先把深度解码为米，按绝对/相对深度差检测不连续区域，再膨胀剔除区域。输出仍在 `depths_render_color_edge/`：LA 更新 alpha，uint16 将无效位置清零，其余深度数值保持原样，不压缩到 8 位。

输出目录的 `depth_format.json` 保存格式、单位、来源和过滤参数。`colmap_adc.py`、`colmap_depth.py` 以及法向量预计算读取同一配置，得到米制深度与有效掩码。缩放深度时采用最近邻，防止插值混合无效像素。复制数据时须同时保留 JSON。

未附带 JSON 的旧数据默认按 LA 读取；16 位深度不会被静默转换成 LA，而会明确报错。修改格式、单位或过滤参数，或旧输出缺少 JSON 时，必须重新生成。可给总入口增加 `--overwrite-derived`（也会重算纹理权重和指定的 anchors），或者只重算深度：

```bash
python utils/data_process/depth_process/generate_depth_edge_masks.py \
  --data-dir /media/zc/SSD/GS/scannetTpp/068ba2946c \
  --input-dir depth --depth-tag uint16 --depth-scale 1000 \
  --workers 8 --overwrite
```

`train_citygs_mcmc.sh` 的 prepare 阶段也支持环境变量 `DEPTH_TAG=uint16 DEPTH_SCALE=1000 RAW_DEPTH_DIR=depth`；后续训练从过滤深度目录的 JSON 获取格式。
