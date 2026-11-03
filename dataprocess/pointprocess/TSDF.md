# 深度图融合为点云

`tsdf_fusion.py` 从原仓库 `dataprocess/tsdf_fusion.py` 迁移，使用 Open3D 的 `ScalableTSDFVolume` 在 CPU 上融合 ScanNet++ iPhone RGB-D 帧，提取带颜色和法向量的表面点云。保持原来的相机坐标约定、融合参数及世界坐标，不进行世界坐标归一化。

## 输入格式

```text
068ba2946c/
├── scene_iphone_metadata.npz
├── images/                         # RGB 图像
└── depth/                          # 同名 PNG 深度图
```

NPZ 必须包含：`images`（N 个图像文件名）、`trajectories`（N×4×4 的 OpenCV camera-to-world 位姿）、`intrinsics`（N×3×3 内参）。例如 `images/frame_000001.jpg` 对应 `depth/frame_000001.png`。

RGB 与深度图须同尺寸并已对齐，深度为 uint16 相机 Z 深度，默认单位为毫米（1000 个深度单位对应 1 米），0 表示无效深度；位姿平移单位须为米。脚本适配此前使用的第三方预处理数据格式，不自动读取其他 ScanNet++ 原始数据布局或转换其他位姿约定。

## 安装与执行

在已创建的 Python 环境中安装依赖：

```bash
cd /Users/zongcheng/code/3dgs/personalGsplat
python -m pip install -r dataprocess/pointprocess/requirements-tsdf.txt

python dataprocess/pointprocess/tsdf_fusion.py \
  --scene /Users/zongcheng/code/3dgs/data/scannet/068ba2946c \
  --output outputs/068ba2946c/tsdf
```

服务器执行示例（须先在服务器的 Python 环境安装依赖）：

```bash
cd /home/zc/code/gsplat-self/personalGsplat
python dataprocess/pointprocess/tsdf_fusion.py \
  --scene /media/zc/SSD/GS/scannetTpp/068ba2946c \
  --output outputs/068ba2946c/tsdf
```

Docker 内请将上述目录替换为容器实际挂载的路径。省略 `--output` 会写入输入场景的 `tsdf_output/`。已有 `point_cloud.ply` 时脚本拒绝覆盖，请指定新的输出目录。

## 参数和融合原理

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--scene` | 必填 | 场景目录 |
| `--output` | `<scene>/tsdf_output` | 输出目录 |
| `--voxel-size` | 0.01 | 体素边长，单位米 |
| `--sdf-trunc` | 0.04 | TSDF 表面附近的截断距离，单位米 |
| `--depth-scale` | 1000 | 原始深度值除以该值后得到米 |
| `--depth-max` | 10 | 超过该距离的深度不参与融合，单位米 |
| `--stride` | 1 | 每隔多少帧融合一次；设为 10 可先快速试跑 |

每帧 RGB-D 按内参投影到体素空间，通过位姿对应到同一世界坐标系，累积表面附近的截断有符号距离及颜色。Open3D 接收 world-to-camera 外参，所以脚本对 NPZ 中的 camera-to-world 矩阵求逆。最后提取零等值面附近的表面点，而不是简单拼接所有帧的反投影点。

`--sdf-trunc` 是 TSDF 融合本身的距离范围，`--depth-max` 是输入深度过滤范围；它们都不是 GTLR 训练中的深度损失截断。体素越小，通常细节越多、内存和计算开销越大。融合质量受输入深度、RGB 对齐和位姿精度影响。

输出：

- `point_cloud.ply`：世界坐标中的彩色表面点云，含法向量。
- `fusion_report.json`：输入路径、帧数、点数、参数、包版本、坐标范围和耗时。

## 接着下采样

需要安装下采样脚本的 SciPy 依赖，目标数量不能超过融合后的点数：

```bash
python -m pip install scipy
python dataprocess/pointprocess/downsample.py \
  --input_ply outputs/068ba2946c/tsdf/point_cloud.ply \
  --output_ply outputs/068ba2946c/init.ply \
  --num_samples 400000
```

下采样保留选中点的 XYZ/RGB，当前不保留 TSDF 点云的法向量。融合与下采样都不会启动训练。
