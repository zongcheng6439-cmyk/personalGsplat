# 点云预处理

- 深度图融合：[`tsdf_fusion.py`](tsdf_fusion.py)，运行方式见 [TSDF.md](TSDF.md)。
- 点云下采样：[`downsample.py`](downsample.py)，说明如下。

## GTLR 点云下采样

`downsample.py` 从原仓库 `dataprocess/gtlr/sample_points.py` 迁移几何/纹理加权下采样部分，可独立运行，不依赖训练代码、PyTorch 或 CUDA。输入为带 XYZ 和 RGB 的 PLY 点云，输出为选中的原始点子集，以及同名 `.json` 参数记录。

## 原理

1. SciPy `cKDTree` 在完整输入点云中寻找每个点的 K 个最近邻，按点索引排除自身，默认 K=64。
2. 对邻居坐标计算协方差，曲率分数为最小特征值除以全部特征值之和（分母加 1e-12）。
3. 对邻居 RGB 分别计算总体方差，再对三个通道取平均，得到纹理分数。
4. 两类分数分别做全局 min-max 归一化，按各 0.5 混合，归一化为抽样概率，按指定数量无放回抽样。

只有**分数**归一化；不缩放、不平移、不归一化世界坐标，也不进行体素中心替换。边缘、曲率较大或颜色变化明显的区域更容易保留，但这是概率采样，并不保证每个区域都保留点。RGB 无变化时纹理贡献为零；所有分数均为零时均匀采样；正权重点不足目标数量时沿用原实现的 1% 均匀概率兜底。

迁移保留了原实现的评分公式与默认纯加权采样方式，近邻查询改为全量 CPU cKDTree，分块计算协方差。原实现部分路径会将参考点限制为 50,000，因此结果不保证与旧脚本逐点一致。此版本不迁移上采样、均匀覆盖混合模式或参考点截取选项。

## 安装和运行

以下命令在仓库根目录运行。可使用已经装有依赖的 Python 环境，也可以建立独立环境：

```bash
cd /Users/zongcheng/code/3dgs/personalGsplat
python3 -m venv .venv-pointprocess
source .venv-pointprocess/bin/activate
python -m pip install numpy scipy open3d
```

例如，将 ScanNet++ 场景的 TSDF 点云下采样到 40 万点（输入文件须存在，且至少包含 40 万点）：

```bash
python dataprocess/pointprocess/downsample.py \
  --input_ply /Users/zongcheng/code/3dgs/data/scannet/068ba2946c/tsdf_output/point_cloud.ply \
  --output_ply outputs/068ba2946c/init.ply \
  --num_samples 400000
```

服务器上进入实际同步的仓库目录，换成服务器数据路径：

```bash
cd /home/zc/code/gsplat-self/personalGsplat
python dataprocess/pointprocess/downsample.py \
  --input_ply /media/zc/SSD/GS/scannetTpp/068ba2946c/tsdf_output/point_cloud.ply \
  --output_ply outputs/068ba2946c/init.ply \
  --num_samples 400000
```

在 Docker 中运行时，输入和输出必须使用容器实际挂载的路径。服务器或容器里的 Python 同样需要安装上述依赖。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--input_ply` | 必填 | 带 RGB 的输入点云 |
| `--output_ply` | 必填 | 输出 PLY，父目录自动创建；已有输出会被覆盖 |
| `--num_samples` | 必填 | 输出点数，不能大于输入点数 |
| `--knn` | 64 | 邻居数量，小点云自动减小 |
| `--chunk_size` | 4096 | 每批评分点数，内存紧张可设为 1024 |
| `--workers` | 4 | 近邻查询线程数，-1 使用全部 CPU |
| `--seed` | 42 | 抽样随机种子 |

日志输出读取、评分进度、采样和总耗时。固定输入顺序、参数和软件环境时，相同种子可复现抽样结果。KDTree 和输入仍常驻内存，分块只限制邻居和协方差中间数组的大小。

输出仅包含 XYZ 和 RGB，不保留输入的法向量、强度、语义或其他自定义字段；RGB 由 Open3D 写为 8 位颜色。这是用于初始化的普通点云，不是带 SH、opacity、scale 等属性的训练后 Gaussian PLY。下采样仅是预处理的一步，不会自动生成相机、深度图或启动训练。
