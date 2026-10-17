# Voxel Plane MCMC 与当前 MCMC 训练的差异

本文以服务器当前源码为准，对比以下三层实现：

1. `MCMCStrategy`：gsplat 的基础 MCMC；
2. `TextureMCMCStrategy`：当前 texture MCMC；
3. `VoxelPlaneMCMCStrategy` + `train_voxel_plane_mcmc.py`：在 texture MCMC 上增加 voxel、平面初始化、空间/形状约束和几何损失。

继承关系为：

```text
MCMCStrategy
└── TextureMCMCStrategy
    └── VoxelPlaneMCMCStrategy
```

因此，Voxel Plane MCMC **仍然使用原有 MCMC 的 CUDA relocate、sample_add 和 noise 算法**，没有复制或修改 `ops.py` 中的底层结构操作。它主要改变 donor 的选择，并在结构操作外围增加 anchor 状态、约束和训练损失。

## 1. 总体差异

| 项目 | 基础 MCMC | Texture MCMC | Voxel Plane MCMC |
|---|---|---|---|
| 初始化几何 | 输入点云、KNN 各向同性 scale、随机 quaternion | 同基础 MCMC | 平面 anchor 内的初始 GS 对齐 anchor normal，并压薄第三轴 |
| relocate 判定 | `opacity <= min_opacity` | 相同 | 相同，但 keeper 永远不进入 dead mask |
| donor 概率 | opacity | opacity、texture、2D grad、FastGS score 的可选加权和 | 完全继承 Texture MCMC |
| add 数量 | 每次增长到当前数量的 105%，不超过 `cap_max` | 相同 | 相同 |
| add 分布 | opacity 分布 | 将 top-score 10% 的总采样概率至少提高到 90% | 完全继承 Texture MCMC |
| donor 属性处理 | donor 与新 GS 的 opacity/scale 由原始 MCMC relocation 公式共同重算 | 相同 | 相同 |
| noise 对象 | 默认所有 GS、每一步 | 仅最近 add/relocate 且当前图像可见的 active GS | 相同，并在 noise 后施加 voxel/平面边界约束 |
| 空间归属 | 无 | 可选“初始位置作为 surface anchor” | 每个 GS 归属一个 voxel-plane anchor；新 GS 继承 donor 的 anchor ID |
| 空间边界 | 无 | 可选球形反射边界 | 平面 anchor 使用有限厚度 slab；可同时使用球形 anchor 半径 |
| 最低覆盖 | 无 | 无 | 可保护每个已占用 anchor 的一个 keeper GS |
| 几何损失 | 无专用平面损失 | 无专用平面损失 | anchor normal、平面第三轴、全局大 scale、KNN normal、KNN plane |
| KNN | 不需要 | 不需要 | 初始化及每次 refine 后在 CPU 上重建 8-NN |

## 2. 当前基础训练损失

`simple_trainer_texture.py` 的基础图像损失是：

\[
L_{rgb}=(1-\lambda_{ssim})L_1+\lambda_{ssim}L_{ssim},
\qquad \lambda_{ssim}=0.2.
\]

另外提供以下可选损失：

- opacity regularization：由 `opacity_reg` 控制；
- scale regularization：由 `scale_reg` 控制；
- needle/large-scale regularization：由 `needle_scale_reg` 控制，同时惩罚最大轴超过 `needle_scale_max`，以及最大轴与中间轴之比超过 `needle_scale_ratio`；
- LiDAR depth loss：由 `depth_loss` 控制。

当前 MCMC 和 Texture MCMC 的命令入口默认使用：

- `init_opa=0.5`；
- `init_scale=0.1`；
- `opacity_reg=0.01`；
- `scale_reg=0.01`；
- `needle_scale_reg=0.01`。

这些属于训练器配置，不是 MCMC 策略内部的 relocate 规则。

### 2.1 当前深度损失

深度首先经过 LetsGo 风格的压缩映射：

\[
f(d)=
\begin{cases}
d/(2\beta), & d<\beta,\\
1-\beta/(2d), & d\ge\beta,
\end{cases}
\qquad \beta=10\text{ m（默认）}.
\]

在 densification 停止前，使用较弱权重 `depth_refine_lambda=0.01`，并对前后方向误差使用对称 L1：

\[
L_d=|f(d_{render})-f(d_{gt})|.
\]

在 `refine_stop_iter` 后，深度权重在 `depth_final_ramp_steps=3000` 步内由 0.01 增加到 `depth_lambda=0.05`。此时重点惩罚位于 LiDAR 表面前方、靠近相机的 floater：

\[
L_d=\operatorname{ReLU}(-e)+w_b\operatorname{ReLU}(e),
\quad e=f(d_{render})-f(d_{gt}),\quad w_b=0.1.
\]

其中 `pred < target` 表示渲染表面比 LiDAR 表面更靠近相机。有效深度还必须通过有限值、near/far plane、图像 mask 和 `depth_gs_mask`。深度边界是否被排除由预处理生成的 `depth_gs_mask` 决定，训练器本身不重复计算边界。

深度损失是基础训练器能力，Voxel Plane MCMC 可以使用，但 `train_voxel_plane_mcmc.py` 中的 `depth_loss` 仍继承默认值 `False`，必须通过命令行显式打开。

## 3. Texture MCMC 相对基础 MCMC 的修改

### 3.1 donor score

Texture MCMC 的 donor importance 是启用项的加权和：

\[
I_i=w_o O_i+w_t\hat T_i+w_g\hat G_i+w_f\hat F_i.
\]

- `O`：sigmoid 后的 opacity；
- `T`：texture score；
- `G`：屏幕空间绝对梯度 score；
- `F`：FastGS high-loss 命中次数。

每项只有对应的 `use_*_score` 开关打开时才参与。Texture 和 grad 使用正数样本的分位数归一化，并截断到 `[0, 2]`；如果最终 importance 总和为 0，则退化为均匀采样。

texture 统计来自当前像素 top-K contributor：

\[
T_i=
\frac{\sum_{p,k:\,g_{p,k}=i} W_p\,r_{p,k}}
     {\max(N_i,10^{-4})},
\]

其中 `W_p` 是 weight map，`r_{p,k}` 是 GS 对像素的渲染贡献权重，`N_i` 是该 GS 进入 top-K 的次数。注意：当前实现的分母是**命中次数**，不是渲染权重之和。

grad score 是 GS 可见时 `means2d.absgrad` 范数的累计均值。FastGS score 是归一化图像 L1 大于 `loss_threshold` 的像素中，GS 进入 top-K 的次数；累计次数少于 5 时暂不使用。

### 3.2 relocate 与 add

- dead GS 仍由 `opacity <= min_opacity` 决定；
- relocate 使用上述 importance 在 live GS 中采 donor；
- add 同样使用事件前的 importance，但会把 top `add_hot_fraction=10%` 候选的总概率至少提升到 `add_hot_probability=90%`；
- 每个 refine 事件仍最多将 GS 数量增长到原来的 105%，并受 `cap_max` 限制；
- refine 完成后，texture、grad 和 FastGS 的累计统计全部清零，下一事件重新累计。

`relocate()` 和 `sample_add()` 仍调用原始 MCMC relocation 公式。一个 donor 被复用时，其 opacity 和 scale 会按复用次数重新计算，然后属性复制给 dead/new GS；这不是简单地只复制位置。

### 3.3 active、visible noise

基础 MCMC 默认每步给所有 GS 注入：

\[
\Delta\mu=\Sigma\epsilon\,
\sigma(-k(O-t))\,s_{noise}.
\]

因此 scale/covariance 越大、opacity 越低，随机位移通常越大。

Texture MCMC 将它改为：

1. add/relocate 后将目标 GS 标记为 active；
2. active 状态保持 `active_noise_steps`；
3. 每一步只对 active 且在当前图像产生非零屏幕梯度的 GS 注入 noise；
4. `noise_scale` 在整个训练中指数衰减到初值的 10%；
5. 可选 surface anchor 时，noise 后用反射映射把这些 GS 放回 anchor 球内。

这减少了对已经稳定 GS 的持续扰动，也避免未被当前图像监督的 GS 随机游走。

## 4. Voxel Plane MCMC 的 MCMC 修改

### 4.1 anchor 输入和一次性匹配

`plane_anchor_ply` 至少需要：

```text
x y z nx ny nz is_plane
```

如果 PLY 同时有 `source_x/source_y/source_z`，初始 GS 用 source 坐标匹配；否则用 anchor XYZ。初始化时通过 CPU `cKDTree` 为每个 GS 找最近 anchor，并要求距离不超过 `anchor_match_max_distance`。

没有任何初始 GS 归属的 anchor 会被删除并压紧 ID。这意味着代码保证的是“每个**保留下来的已占用 anchor**至少有一个初始 GS”，不是强制给空 anchor 创建 GS。

### 4.2 keeper 保证 anchor 不被清空

每个 anchor 选择匹配顺序中的第一个初始 GS 作为 keeper。开启 `keep_one_per_anchor` 后：

```text
dead_mask[keeper_id] = False
```

所以 keeper 即使 opacity 低于 `min_opacity` 也不会被 relocate。它不一定是 opacity 最高或贡献最大的 GS，代价是可能永久保留少量低贡献 GS；好处是 relocate 不会让已占用 voxel 完全失去表示。

### 4.3 anchor ID 继承

relocate 和 add 的目标 GS 都继承 donor 的 `anchor_id`。因此结构操作后，GS 属于 donor 所在 voxel，而不是再次用 KNN 查找 anchor。这既保持归属一致，也避免每次结构操作都做 GS-to-anchor 最近邻匹配。

### 4.4 位置硬约束

对 add、relocate 和 active noise 的目标 GS 执行约束；**普通 Adam 对 means 的梯度更新不会每步投影回 anchor**。

平面 anchor：只限制法向位移在

\[
[-\delta,+\delta],\qquad \delta=\texttt{plane\_normal\_tolerance}
\]

内，越界部分使用周期反射，不截断到同一个边界。因此 GS 仍可在平面切向方向移动，也允许在默认 ±3 cm 的薄层中调整。

如果 `limit_anchor_distance=True`，还会将三维距离用反射映射限制在 `anchor_max_distance` 内。非平面 anchor 只使用这个球形距离约束。

### 4.5 add/relocate 后的形状硬约束

所有新加或刚 relocate 的 GS：

\[
s_x,s_y,s_z\le \texttt{anchor\_voxel\_size}.
\]

属于平面 anchor 的目标 GS还会：

- 用最短旋转将 quaternion 的局部第三轴对齐 anchor normal；
- 将第三轴限制到 `plane_s3_max`；
- 位置立即应用平面 slab 和可选球形边界。

这里直接修改参数，是硬约束；下一步优化器仍可继续改变它们，训练损失负责长期拉回合理形状。

### 4.6 GeoGaussian 风格的初始平面 GS

若 `align_initial_plane_gs=True`，所有归属平面 anchor 的初始 GS 会：

- quaternion 第三轴对齐 anchor normal；
- 两个切向 scale 取最近邻点距推导值
  \(s_t=\sqrt{\operatorname{clip}(d_{NN}^2,10^{-7},10^{-2})/2}\)；
- 第三轴初始化为 `initial_plane_thickness=1e-4 m`。

非平面 anchor 内的 GS 保留基础初始化方式。

### 4.7 KNN 更新

每个 GS 保存 `plane_knn=8` 个最近 GS。KNN 在初始化时建立，并在每次 relocate/add refine 事件完成后用全部当前 means 重建。它不在每个训练 step 重建。

当前实现需要把全部 means 传到 CPU 并运行 SciPy `cKDTree`，因此百万级 GS 下，refine 事件会产生明显同步和 CPU KNN 开销。

## 5. Voxel Plane 额外几何损失

这些损失没有写进 MCMC strategy，而是由 `train_voxel_plane_mcmc.py` 在 optimizer 的 `step_pre_hook` 中把梯度直接叠加到已有梯度。终端打印的是未乘 lambda 的损失值；它们不会出现在主 `loss.item()` 或 TensorBoard 的总 loss 中。

训练脚本要求单 GPU，并关闭 `packed`、`sparse_grad` 和 `visible_adam`，以便访问完整参数梯度并正确映射 GS ID。

### 5.1 全局 anchor scale loss

对所有 GS 的每一个轴，若实体 scale 超过

\[
s_{max}=\texttt{anchor\_scale\_max\_ratio}\times
\texttt{anchor\_voxel\_size},
\]

则惩罚 log-scale 超限量的平方。只对超限轴求平均，不被正常 GS 稀释。`large_scale_gs` 输出至少一个轴超限的 GS 数量。

该软损失作用于所有 GS；add/relocate 后的 scale clamp 是对应的即时硬限制。

### 5.2 plane S3 thickness loss

对所有平面 anchor GS 的第三轴：

\[
L_{s3}=\operatorname{mean}
\left[\operatorname{ReLU}(\log s_3-\log s_{3,max})^2\right].
\]

它只惩罚第三轴过厚，不要求三个轴相等，也不限制平面内两个切向轴缩小。

### 5.3 anchor normal loss

只作用于平面 anchor 内的全部 GS。令 quaternion 第三轴方向为 `n`，anchor normal 为 `a`。先用停止梯度的符号选择将 `a` 翻到与 `n` 同半球，再计算：

\[
L_{anchor\_normal}=\operatorname{mean}|n-a'|.
\]

使用显式正负方向选择而不是 `1-|n\cdot a|`，是为了避免两者恰好 90° 时 `abs(dot)` 在 PyTorch 中给出零梯度。由于这些平面 GS 的第三轴由初始化和结构操作保证为薄轴，此处明确把第三轴当法向量。

### 5.4 KNN 的 6/8 主方向投票

KNN normal/plane loss 面向当前可见的中心 GS，但邻居可来自全部 GS，而且梯度同时回传给中心和被选中的邻居。

对中心和 8 个邻居，均从 quaternion 旋转矩阵中选择**当前最短 scale 对应的轴**作为 GS 法向量。8 个邻居互相比较：

- 两个法向量的无向夹角小于 10°，视为同方向；
- 得票最多的邻居作为候选主方向；
- 至少 6/8 个邻居同意时，该中心才参与 KNN 损失；
- 同意的邻居经正负号对齐后求均值，得到目标法向 `n*`。

投票只由邻居完成，中心自身不参与，因此即使中心是突然竖起或倾斜的异常 GS，也能被周围多数平面方向拉回。

### 5.5 KNN normal loss

对投票通过的中心与同组邻居计算：

\[
L_{knn\_normal}=
\operatorname{mean}\left(1-n_c\cdot n_j'\right).
\]

邻居方向先做无向符号对齐。该损失同时更新中心和同组邻居的 quaternion。它不是只约束 plane anchor，因此也能处理未被 anchor 预处理标成平面的局部平面 GS。

### 5.6 KNN plane loss

同一投票组的邻居位置均值为 `mu_N`，主法向为 `n*`：

\[
L_{knn\_plane}=\operatorname{mean}
\left| (\mu_c-\mu_N)\cdot n^* \right|.
\]

它只惩罚中心相对邻域平面的**法向高度差**，不会约束平面内切向移动。梯度同时作用于中心 means 和所选邻居 means，所以局部平面会共同靠拢，而不是把邻居当完全固定的目标。

## 6. 当前 Voxel Plane 默认配置

`train_voxel_plane_mcmc.py` 的关键默认值为：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `means_lr` | `1.6e-5` | 较低的位置学习率 |
| `init_opa` | `0.5` | 初始 opacity |
| `init_scale` | `0.1` | 非 PLY scale 的初始化倍率 |
| `texture_weight / grad_weight` | `2 / 1` | donor importance 权重 |
| `active_noise_steps` | `1000000` | 结构操作后的 GS 基本持续 active 到训练结束 |
| `anchor_voxel_size` | `0.1 m` | voxel 尺寸及 add/relocate 最大轴硬限制 |
| `anchor_max_distance` | `0.2 m` | 球形 anchor 反射半径，当前默认开启 |
| `plane_normal_tolerance` | `0.03 m` | 平面上下允许的半厚度 |
| `initial_plane_thickness` | `0.0001 m` | 初始平面 GS 第三轴厚度 |
| `plane_s3_max` | `0.005 m` | 训练及结构操作中的平面第三轴上限 |
| `anchor_normal_lambda` | `0.1` | anchor normal 梯度权重 |
| `knn_plane_lambda` | `0.1` | KNN 平面位置梯度权重 |
| `knn_normal_lambda` | `1.0` | KNN 法向梯度权重 |
| `plane_scale_lambda` | `0.01` | 平面第三轴超限梯度权重 |
| `anchor_scale_lambda` | `0.02` | 所有 GS 大轴超限梯度权重 |

其中 `active_noise_steps=1000000` 不等于每步移动所有 GS：仍必须是 add/relocate 后的 active GS，并且当前图像可见。`noise_injection_stop_iter=-1` 表示训练期间不主动停止这套 active-visible noise。

## 7. 每一步的实际执行顺序

```text
渲染 RGB（可选 ED depth）
  -> 计算 texture / FastGS 统计
  -> Voxel Plane 记录当前可见 GS 与全部 plane-anchor GS
  -> RGB、可选 depth、基础 regularization backward
  -> optimizer step_pre_hook 叠加：
       means: KNN plane 梯度
       quats: anchor normal + KNN normal 梯度
       scales: 全局大 scale + plane S3 梯度
  -> optimizer.step()
  -> Texture MCMC 累计 grad/texture/FastGS 状态
  -> 到 refine 事件时：relocate -> add
       -> 继承 donor anchor ID
       -> 目标位置反射约束
       -> 目标 scale/quaternion 硬修正
  -> 对 active 且当前可见 GS 注入 noise
  -> noise 后再次做位置反射约束
  -> refine 事件后重建全部 GS 的 8-NN
  -> 清零本轮 texture/grad/FastGS 累计统计
```

需要注意：几何 loss 的 hook 在本步使用的是上一次 refine 后建立的 KNN；本步 refine 完成后才为后续 step 更新 KNN。

## 8. 主要收益、代价和已知限制

### 收益

- 保留 Texture MCMC 将 GS 向高 weight/high-gradient 区域补充的能力；
- keeper 防止平面或墙面 voxel 被 relocate 清空；
- LiDAR 平面初始 GS 更薄且朝向正确；
- KNN normal 可以修正未标记为 plane anchor 的局部异常朝向；
- KNN plane 只限制法向高度，不妨碍平面内位置优化；
- active-visible noise 减少无监督随机游走。

### 额外开销

- texture top-K contributor 的额外 rasterization/统计；
- 每个可见中心读取 8 个邻居，并对中心和邻居构建局部 autograd 图；
- quaternion-to-rotation 和 8×8 法向投票；
- 每次 refine 后将全部 means 同步到 CPU 并重建 KNN；
- 对全部 plane GS 的 anchor normal/S3 梯度，以及对全部 GS 的大 scale 检查。

这些是 Voxel Plane 训练明显慢于基础 MCMC 的主要来源，其中大规模场景最敏感的是全量 CPU KNN 重建和 KNN 几何梯度。

### 已知限制

- 效果高度依赖 `plane_anchor_ply` 的平面分类和法向质量；
- keeper 只按初始稳定顺序选第一个，不按贡献选，可能保留低 opacity GS；
- 6/8 投票判断的是主方向一致，并不能严格证明 8 个邻居属于同一物理平面；
- GS 三轴接近相等时，“最短轴”可能随优化切换，KNN 法向会不稳定；
- 普通 means 梯度没有硬投影，位置约束只在 add、relocate 和 active noise 后执行；
- anchor ID 始终继承 donor，不会因普通优化越过 voxel 而自动重新匹配；
- Texture score 当前除以 top-K 命中次数，而不是 `sum(render_weight)`，其含义是平均加权贡献，不是严格的条件期望；
- `depth_loss`、needle loss 等基础损失是否生效取决于训练命令，不能仅凭策略类名判断。

## 9. 对照源码

- 基础 MCMC：`gsplat/strategy/mcmc.py`
- Texture MCMC：`gsplat/strategy/texture_mcmc.py`
- Voxel Plane MCMC：`gsplat/strategy/voxel_plane_mcmc.py`
- 基础训练器与 RGB/depth/regularization：`examples/simple_trainer_texture.py`
- Voxel Plane 独立训练入口与几何梯度：`examples/train_voxel_plane_mcmc.py`
- texture/FastGS 投影统计：`examples/render_utils.py`
- 原始 relocate/add/noise：`gsplat/strategy/ops.py`

