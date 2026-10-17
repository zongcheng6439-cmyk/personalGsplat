# Voxel-plane：可选 PGSR 几何损失（2026-09-20）

适用入口：`examples/train_voxel_plane_mcmc.py original` 与 `dynamic`。
默认关闭，追加 `--pgsr-loss` 启用，可用 `--no-pgsr-loss` 显式关闭。

## 新增开关

```bash
--pgsr-loss \
--pgsr-sv-lambda 0.015 \
--pgsr-mv-lambda 0.15 \
--pgsr-start-iter 5000 \
--pgsr-warmup-steps 5000 \
--pgsr-every 2 \
--pgsr-samples 4096
```

调度与 NCC 相同：start 之后 warmup 线性升权重，每 `pgsr_every` 步计算一次；
`steps_scaler` 同步缩放 start/warmup。λ 默认取 PGSR 论文值（λ2=0.015、λ3=0.15）。

## 渲染：PGSR 平面参数与无偏深度

实现集中在 `examples/pgsr_geom.py`：

1. 每个高斯按最短尺度轴取法向（世界→相机系），按视线方向翻转消歧；
   平面距离 `d_i = n_i · μ_i^cam`（相机原点到高斯切平面，相机前方为负，
   与 `ncc_loss.py` 的 PCA 平面符号约定一致）。
2. `(nx, ny, nz, d)` 四个通道经 gsplat `rasterization` 的 `extra_signals`
   与 3 个哑零颜色通道同一次前向渲染（共 8 通道，是 CUDA kernel 已编译的
   通道数；3+4=7 未编译）。α 累积权重在除法中约掉，得到无偏光线-平面
   z 深度 `D(p) = 𝒟(p) / (N(p)·K⁻¹p̃)`（PGSR 式 2/3/4）。
3. 梯度经 `meta["render_extra_signals"]` 回传：法向图 → quats，
   距离图 → means + quats。几何渲染只在 PGSR 步执行，代价是该步多一次
   前向光栅化；不使用主渲染的深度通道，与 `depth_loss`/NCC 互不干扰。

## 两个损失

- 单视图（`single_view_normal_loss`，PGSR 式 5/6）：`depth_to_normal`
  从无偏深度反推法向（朝向相机翻转），与渲染法向图做 L1；权重
  `(1-∇Ī)²` 用参考图灰度梯度做边缘感知。有效像素要求 mask 内、
  α≥0.8、深度有限为正、深度法向模长 ≥0.5（自动剔除零填充边界）。
- 多视图（`PatchNCCLoss` 渲染平面模式）：`plane_normals/plane_distances`
  直接取渲染图的逐像素平面，跳过 11×11 PCA 拟合与其三重筛选，
  法向和平面距离都保留梯度；patch 候选、ZNCC、源视图选择、LiDAR
  遮挡检查等原有门控全部保留。混合法向模长 <0.5 的中心被剔除。

`--pgsr-loss` 且 `pgsr_mv_lambda>0` 时会构建 `PatchNCCLoss`（即使
`--ncc-loss` 未开），样本数取 `pgsr_samples`。两个多视图项可同时开启，
但注意它们各自独立采样 patch，通常二选一即可。

## 运行约束

在原有 voxel-plane 约束（单 GPU、`packed=False`、`sparse_grad=False`、
`visible_adam=False`、`normalize_world_space=False`）之上，PGSR 额外要求：
`batch_size=1`、COLMAP 数据集、pinhole 相机、无 `pose_noise`、
`with_ut=False`、`with_eval3d=False`（几何渲染必须与主渲染同一求值路径）。

## 文件与移除

- `examples/pgsr_geom.py`：平面参数信号、像素光线、无偏深度、联合渲染、
  单视图损失。
- `examples/ncc_loss.py`：`__call__` 新增可选 `plane_normals/plane_distances`
  （必须成对传入），及 `_rendered_planes` 收集分支；PCA 路径不变。
- `examples/train_voxel_plane_mcmc.py`：flag、校验、调度、`_pgsr_extra_loss`，
  `extra_training_loss` 改为 NCC/PGSR 调度器（原 NCC 行为不变）。

临时禁用只需去掉 `--pgsr-loss`。彻底删除时移除 `pgsr_geom.py`、
`train_voxel_plane_mcmc.py` 中标记的 PGSR 配置/校验/方法、`adjust_steps()`
中的两个 PGSR 字段缩放，以及 `ncc_loss.py` 的渲染平面分支（也可保留该
分支，不影响 PCA 路径）。

## 验证

新增 `tests/test_pgsr_geom.py`（12 项，CPU/CUDA）：信号朝向与符号、
光线-平面求交性质、合成平面上无偏深度≈真值且梯度可达 means/quats、
单视图损失零/正值对照与梯度、渲染平面模式下 NCC 的 patch 数与
plane map 梯度、成对参数校验。

容器内执行（无需安装包、不重编 CUDA，使用 `gsplat/csrc.so` 符号链接指向
`build/lib.linux-x86_64-3.10/gsplat/csrc.so`）：

```bash
docker run --rm -i --gpus all --entrypoint /opt/gsplat-venv/bin/python \
  -v /home/zc/code/aibee/gsplat:/code/gsplat -w /code/gsplat \
  harbor.aibee.cn/robot/gsplat:V2 \
  -m pytest -q -o addopts="" tests/test_pgsr_geom.py \
  tests/test_ncc_warp_refactor.py tests/test_ncc_view_selection.py
```

结果：PGSR 12 项通过；NCC 相关回归全部通过。
`tests/test_voxel_plane_mcmc.py` 有 3 项失败（`test_gs_knn_has_eight_nonself_neighbors`
等），为本次修改前已存在的测试与 `_find_gs_knn` 现行返回值签名漂移，
与本特性无关，未改动该文件。`original`/`dynamic` 的 `--help` 均已确认
暴露 `pgsr-*` 参数。未进行整场景训练，尚未声明实际 NVS/几何指标提升。

训练环境注意：训练入口每次 `python setup.py install` 后才生效，源码上传
不等于安装包更新；本次只改 examples/tests，不触碰 gsplat 包与 CUDA。

## 可视化（2026-09-20 补充）

`--pgsr-vis`（可独立于 `--pgsr-loss` 使用）：每次 val 评估后，对验证集前
`--pgsr-vis-count`（默认 10）张图像渲染平面参数，保存到
`result_dir/pgsr_vis/step_XXXXXX/`：

- `val_XXXX_depth.png`：无偏深度，2%~98% 百分位拉伸 + viridis 着色，
  无效像素（α≤0.5、深度非正、mask 外）置黑；
- `val_XXXX_normal.png`：渲染法向图（单位化后 (n+1)/2 编码，相机系）；
- `val_XXXX_alpha.png`：渲染 α 图。

要求 pinhole 相机、`with_ut=False`、`with_eval3d=False`。可视化覆盖
`eval()`：先调用基类评估（RGB 指标不受影响），再追加保存，训练主循环
无改动。
