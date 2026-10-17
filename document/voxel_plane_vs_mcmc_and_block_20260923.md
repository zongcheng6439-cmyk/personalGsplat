# Voxel-Plane MCMC 与分块训练的改进点总结

日期：2026-09-23
范围：`gsplat/strategy/voxel_plane_mcmc.py`（及其基类链）、`utils/train/` 分块管线
对照：原版 3DGS-MCMC（`gsplat/strategy/mcmc.py`）、CityGaussian 官方分块方案

---

## 一、Voxel-Plane MCMC 相比原版 MCMC 的改进

继承链：`MCMCStrategy`（原版）→ `TextureMCMCStrategy`（纹理感知）→ `VoxelPlaneMCMCStrategy`（LiDAR 平面锚点）。改进分两层。

### 1.1 原版 MCMC 的机制（对照基线）

- **reloc­ate**：把 opacity ≤ `min_opacity` 的"死"高斯按**不透明度分布**采样瞬移；
- **sample_add**：按**不透明度分布**采样新增高斯（二项式系数近似），每轮总数 ×1.05 直到 `cap_max`；
- **噪声注入**：噪声幅度 = `lr × noise_lr`，与 means 优化器学习率耦合，全程无独立日程。

问题：增密位置完全由不透明度决定——不透明度高的地方未必是细节/纹理复杂的地方；低不透明度的重要区域直接被排除；噪声与学习率耦合，后期想保留探索能力时无法独立控制。

### 1.2 TextureMCMC 层：增密引导信号从"单一不透明度"变为"多信号加权"

| 改进 | 原版 MCMC | TextureMCMC |
|---|---|---|
| donor 重要性 | 仅 opacity 分布 | opacity + **纹理权重图** + **absgrad 屏幕梯度**（可选 FastGS 分数）加权：`opacity_weight=0.25 / texture_weight=0.5 / grad_weight=0.25` |
| 低 opacity 处理 | 直接不参与采样 | opacity 只作**软可靠性门控**（`opacity_weight` 降权而非排除），重要区域的低不透明度高斯仍能接收新点 |
| 新增位置 | 全局按分布采样 | `add_hot_fraction=0.3` / `add_hot_probability=0.7`：70% 概率把新点定向投放到分数 top-30% 的"热点"高斯附近 |
| 梯度统计 | 无 | 屏幕空间**绝对梯度**（absgrad）累积 + 分位数门控（`texture_quantile/grad_quantile=0.9`），抗噪 |
| 噪声日程 | `lr × noise_lr`，与优化器耦合 | **独立噪声日程**：`noise_scale=10`，按训练进度指数衰减到 10%（`noise_scale_at_step`），`active_noise_steps` 可控；噪声门控参数 `noise_opacity_t=0.054` |
| 表面约束 | 无 | `surface_anchor_constraint`：新增/瞬移的高斯可选约束在锚点附近（`surface_anchor_max_distance`） |

### 1.3 VoxelPlaneMCMC 层：LiDAR 体素平面锚点（GeoGaussian 式几何先验）

这是相对原版 MCMC 最核心的增量——把 LiDAR 检测出的**体素平面锚点**（`build_voxel_plane_anchors.py` 生成的 PLY：`xyz / match_xyz / normals / is_plane`）绑定到每个高斯上：

1. **一次性锚点绑定**（`set_plane_anchors`）：
   - cKDTree 把每个初始高斯匹配到最近锚点（按 `match_xyz`）；
   - 无任何高斯匹配的空锚点被丢弃并压缩索引；
   - **每个锚点保留一个 keeper 高斯**（`keep_one_per_anchor`），在 `_dead_mask` 中保护不被 prune——保证每个 LiDAR 平面体素始终有至少一个高斯覆盖，这是原版 MCMC 没有的几何覆盖保证。
2. **平面初始化**（`initialize_plane_gaussians`）：
   - 旋转用**最短弧四元数**对齐到锚点法向；
   - 两个切向轴取 **8-NN 距离中位数**推出的尺度（`sqrt(clip(d²,1e-7,1e-2)/2)`），避免重复点导致切向轴塌缩；
   - 法向轴压扁到 `initial_plane_thickness`（1e-4 m）——平面区域的高斯天生"薄而贴面"，原版 MCMC 只能靠优化慢慢学会。
3. **锚点继承**：新增/瞬移的高斯继承 donor 的锚点绑定，增密过程不破坏几何结构。
4. **运行期几何约束**（优化器自由更新之外的硬约束）：
   - 平面高斯限制在法向 ±`plane_normal_tolerance`（0.03 m）的"平板"内；
   - **三角波径向反射**把高斯限制在锚点 `anchor_max_distance`（0.2 m）内——原版 MCMC 的高斯可以漂到任意远；
   - 平面形状约束：scale ≤ `anchor_voxel_size`，平面高斯第三轴 ≤ `plane_s3_max`。
5. **锚点梯度 hook**（注册在优化器 `register_step_pre_hook`，每步生效）：
   - `_add_anchor_normal_gradient`（quats）：可见平面高斯旋转 z 轴与锚点法向 L1 对齐（自动处理法向正负号）；
   - `_add_knn_plane_gradients`（means/quats）：对**所有可见高斯**做 8-NN 局部平面共识（邻居 ≤ 2×voxel、法向两两夹角 <10°、≥6 票、共识平面厚度 ≤ tolerance 才生效），贴平面 + 法向对齐双重损失；
   - `_add_anchor_scale_gradient`（scales）：两档尺度上限——全体 ≤ 2×voxel_size，平面高斯第三轴 ≤ 0.005，log 空间 ReLU 平方惩罚。
6. **训练器配套损失**（`train_voxel_plane_mcmc.py`）：LetsGo 式压缩 LiDAR 深度损失（带 ramp 日程）、法向损失、NCC 多视图一致性、以及 2026-09-20 新增的 **PGSR 无偏深度/法向量渲染与平面几何损失**（`--pgsr-loss`）。

### 1.4 一句话对比

原版 MCMC 把增密当作"纯采样问题"（不透明度即概率）；本方法把它变成"**先验引导的受约束采样**"——纹理/梯度决定**哪里该加密**，LiDAR 平面锚点决定**新点长什么样、能去哪、不能去哪**，再用锚点梯度 hook 和几何损失在优化期持续把高斯"按"在平面上。

---

## 二、`utils/train/` 分块训练相比 CityGaussian 分块的改进

CityGaussian 官方流程：全图训 coarse(30k) → coarse 高斯 XY 分位数切块 → 相机按"中心落块 ∪ SSIM 可见性"分配 → 每块加载**全量** coarse 高斯、只用本块图像 finetune → 合并时按块 bbox 丢弃块外高斯直接拼接。

本管线（`train_all_partitions.py` + `block_utils.py` + `train_voxel_plane_block.py` + `train_citygs_mcmc.sh`）沿用了 CityGS 的 partitions.pt 格式与"coarse → 分块 → 合并"骨架，但有以下实质改动：

### 2.1 块初始化：可见性扩张的预选点云，而非 bbox 裁剪

- CityGS：每块从同一 coarse ckpt 加载**全部高斯**训练，块外高斯当"背景上下文"，合并时才按 bbox 丢弃。
- 本管线：partition 阶段（`utils/partition/partition.py`）就预计算好每块的 **`block_init/point_cloud_RRR_CCC.ply`**；训练时直接整文件加载（`find_preselected_block_ply`），`bbox_enlarge` 默认 0，bbox 过滤只是 fallback。
- block_init 的构成 = 块 bbox 内核心高斯 ∪ **逐相机遮挡感知可见高斯**：对每个分配到该块的相机，用粗 z-buffer（4×4 cell 取最近深度 + `gs_depth_tolerance=0.1` 判遮挡、chunk 1M 控显存）求可见高斯，OR 进块掩码（`visible_gs_centers`，partition.py:35-85）。
- 好处：初始点集贴合每块实际可见内容（块外但被看到的高斯也进来了），块间重复更少，合并几乎不用裁边；显存从"每块背全场景"降下来。

### 2.2 相机分配：SSIM 判据不变，加两级廉价预过滤

- 分配逻辑与 CityGS-V2 相同：位置分配（相机中心落块内，不外扩）∪ 投影可见性分配（块内高斯 opacity 置 0 前后各渲染一次，`1-SSIM > content_threshold` 即归入）。
- 差异点：
  1. **阈值更严格**：`content_threshold=0.2`（CityGS 默认 0.05~0.08）→ 每块相机更少、训练更快，漏分风险由 block_init 的可见性扩张兜底；
  2. **两级预过滤**（`camera_filter.py` 的 `filter_camera_block_pairs`）：先用块包围球（含 3σ 高斯外扩）与相机视锥做保守相交测试（只读元数据），幸存对再做 1/8 分辨率块级 alpha 渲染过滤（`alpha.max()>0.01`、可见占比 ≥0.0005、`alpha.sum()≥0.25`），只对最终幸存对跑全分辨率双重 SSIM 渲染——CityGS 对所有未分配对都要两次全分辨率渲染，partition 开销大；
  3. 渲染后端换成 **gsplat rasterization**（`sh_degree=0`），与训练器一致；
  4. 只对 **trainset 相机**分配（partition.py:186），评估划分不会导致分配矩阵错位。

### 2.3 预算与步数日程：按块自适应

- **预算**：场景级总额度（默认 8M，shell 给 10M）**均分到各块**（`assign_cap_max`，前余数块 +1；初始点数超额度的块自动抬 cap）——而不是 CityGS 那样每块共用同一个 `cap_max` 配置。
- **步数**：`max_steps = 图像数 × steps_per_image`（100）；`refine_every = refine_start = 0.3×图像数`；`refine_stop = 0.8×max_steps`；深度损失 ramp `4×图像数`——全套日程随块图像数缩放。CityGS-V2 每块固定 `max_steps`（30k/60k），V1 才有按 300 张基准的线性放大。
- **coarse 阶段极轻量化**：只训 `2×图像数` 步、`sh_degree=0`、**densification 完全关闭**（`refine_start=refine_stop=max_steps`）、噪声立即关闭——coarse 只学粗略几何/外观先验，不承担增密职责；同时 coarse ckpt 兼职做 partition 可见性计算的输入。

### 2.4 块训练的正则与策略配置

- 显式打开 `opacity_reg=0.01 / scale_reg=0.01`（基础配置默认关闭），`large_scale_reg` 的职责由锚点 scale 惩罚取代；
- 全程开 `use_texture_score + use_grad_score`（texture 2.0 / grad 1.0）+ absgrad，噪声**全程不关**（`noise_injection_stop_iter=-1`、`active_noise_steps=-1`）配合独立衰减日程；
- LiDAR 深度损失默认开（`depth_loss=True`），带 refine 停止后的 ramp；
- 块评估完全关闭（`test_every=0`、`eval_steps=[]`）：全部图像进训练集、零评估开销（CityGS 每块仍留 val split）。

### 2.5 与平面锚点的分块级结合（voxel_plane 独有）

- 锚点按块裁剪：有 `block_init` 时**全量锚点送入**（匹配不上的空锚点自动丢弃）；否则用 `match_xyz` 做 bbox 裁剪；
- 块级超参更激进：`initial_plane_thickness=1e-4`（比 standalone 的 1e-3 薄 10 倍）、`plane_normal_tolerance=0.05`、`anchor_max_distance=0.5`；
- 三个锚点梯度 hook 原样继承到块 Runner。

### 2.6 工程编排（train_all_partitions.py）

- **多 GPU 调度**：`ThreadPoolExecutor(GPU数)`，只往空闲显存 ≥ 阈值的卡派任务，无闲卡 30s 轮询；每块独占一卡；
- **启动错峰** 30s，避免多块同时加载 coarse PLY；
- **断点续训**：按 `gsplat_point_cloud_{max_steps}.ply` 精确匹配已完成块并跳过；
- 每块独立 `train.log` + `block_training_info.json`，全局 `training_summary.json` / `block_training_complete.json` 完成标记。

### 2.7 相同点（沿用 CityGS 的设计）

- 只在 **XY 平面**分块（Z 不切），块 bbox 为 AABB；
- `partitions.pt` 格式与 CityGS 一致（partition_coordinates / scene_config / rotation_transform）；
- 支持 contract 收缩空间分块（`_contract_xy`，NDC 式 unisphere）与重力对齐 reorient；
- 相机可同时属于多个块（可见性重叠），保证边界处监督连续；
- 合并为"各块最终 PLY 直接拼接"（`merge_block_ply.py`，因 bbox_enlarge=0 重复点更少）。

### 2.8 对比一览表

| 环节 | CityGaussian | 本管线 |
|---|---|---|
| 相机分配 | 中心落块 ∪ 两次全分辨率渲染比 SSIM（阈值 0.05~0.08） | 判据相同但阈值更严（0.2），渲染前加包围球视锥 + 1/8 低分辨率 alpha 两级预过滤，后端换 gsplat |
| 块初始化 | 全量 coarse 高斯，块外当背景 | 可见性扩张的预选 block_init PLY（bbox 核心 ∪ 逐相机 z-buffer 遮挡感知可见高斯） |
| 预算 | 每块统一 cap_max | 场景总额度均分 + 初始点数兜底 |
| 步数 | V2 固定 / V1 按 300 张线性 | 全套日程按块图像数缩放 |
| coarse 阶段 | 30k 步完整训练 | 2×图像数步、SH0、不增密 |
| 块内评估 | 保留 val split | 完全关闭，全部图像训练 |
| 正则 | trim 渲染器按贡献剪 floater | opacity/scale reg + 锚点法向/KNN/尺度三重 hook |
| 增密策略 | 2DGS 梯度增密 | 纹理/梯度分数引导 MCMC，噪声全程独立日程 |
| 几何先验 | Depth-Anything-V2 估计深度 | LiDAR 实测深度 + 体素平面锚点 |
| 调度 | 每卡一块轮询/SLURM | 显存门槛调度 + 错峰 + 断点续训 + 完成标记 |

---

## 备注

- `merge_block_ply.py`、`prepare_citygs_data.py`、`datasets/colmap_adc.py`（带 `block_cameras_json` 的改造版）不在本地快照中；`partition.py`/`citygs_gsplat.py`/`camera_filter.py` 已于 2026-09-23 从服务器补齐到 `utils/partition/`，相机分配与 block_init 的结论来自源码实读。
- 已知疑点：`train_texture_anchor_block.py:163` 对 `find_latest_coarse_ply` 返回值取 `[0]` 疑似 bug（取到字符串首字符），fallback 分支才触发。
