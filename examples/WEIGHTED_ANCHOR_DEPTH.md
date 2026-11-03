# Weighted Anchor MCMC 深度输入

在仓库根目录使用 `examples/train_weighted_anchor_mcmc.py`。新增参数由共享 texture trainer 传入 `colmap_adc.Parser`，再由统一深度读取模块解码；不改变世界坐标。

| 参数 | 行为 |
| --- | --- |
| `--depth-tag auto` | 默认，沿用优先选择过滤目录的逻辑，从 depth_format.json 读取格式；无 JSON 时为 LA |
| `--depth-tag uint16` | 默认读取场景 `depth/`，16 位整数除以 1000 得到米，0 无效 |
| `--depth-tag la` | 原有 8 位 LA 深度编码 |
| `--depth-dir PATH` | 覆盖目录，相对路径相对于 data-dir，也可指定绝对路径 |
| `--depth-scale 1000` | 覆盖 uint16 原始单位/米的比例；显式 uint16 默认 1000，auto 默认沿用 JSON |
| `--depth-loss` | 开启深度损失，默认关闭；只传 tag 不会自动开启 |

显式 tag 和 scale 优先于目录中的 JSON。深度图缩放使用最近邻，防止无效像素污染有效深度。启动时打印实际目录、格式和 scale。

## 068ba2946c

在现有训练命令中加入以下参数即可：

```bash
--depth-loss --depth-tag uint16 --depth-dir depth --depth-scale 1000
```

完整示例（把 `/path/to/weighted_anchors.ply` 换成该策略所需的实际 anchor 文件）：

```bash
cd /home/zc/code/gsplat-self/personalGsplat
CUDA_VISIBLE_DEVICES=0 python examples/train_weighted_anchor_mcmc.py \
  --data-dir /media/zc/SSD/GS/scannetTpp/068ba2946c \
  --result-dir results/068ba2946c_weighted \
  --strategy.anchor-ply /path/to/weighted_anchors.ply \
  --depth-loss \
  --depth-tag uint16 \
  --depth-dir depth \
  --depth-scale 1000
```

若已完成深度边缘过滤，只将 `--depth-dir depth` 换成 `--depth-dir depths_render_color_edge`。场景仍需具备训练所需的 COLMAP 相机、图像、masks、weight_map 和有效 anchors。tag 不会生成这些数据，也不会转换 anchors 格式。
