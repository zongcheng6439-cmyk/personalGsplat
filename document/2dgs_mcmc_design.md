# 面向 2D Gaussian Splatting 的 MCMC 设计方案

> 状态：第一版最小实现，范围限定为 MCMC + 静态 plane anchor。
>
> 目标：新增独立的 2DGS MCMC 策略，不修改或替换现有的 3D
> `voxel_plane_mcmc` 和相应训练方法。

## 1. 设计目标

当前 3D MCMC 将 Gaussian 当作三维椭球处理，包括三维位置噪声、三个
scale 的更新和基于完整三维协方差的探索。2DGS 中的 Gaussian 是有方向的
椭圆面片，因此需要单独处理以下内容：

1. MCMC noise 只在面片切平面内移动，不沿法向随机漂移。
2. relocate 和 add 继续复用现有 CUDA 操作，避免修改 `ops.py`。
3. 第三个 scale 固定为很小的 surfel 厚度，不参与有效几何表达。
4. 平面 anchor 通过可微的位置投影误差和法向误差监督面片。
5. 第一版只使用静态 anchor ID；不做动态 anchor 和 free GS，但每个有效
   anchor 至少保留一个 GS。
6. 使用 2DGS 原生 normal consistency 和 distortion loss，不重复加入复杂
   的 3D KNN/needle 约束。

建议新增而不是覆盖现有文件：

```text
gsplat/strategy/voxel_plane_2dgs_mcmc.py
examples/train_voxel_plane_2dgs_mcmc.py
```

## 2. 2DGS 参数含义

gsplat 的 2DGS 仍然使用与 3DGS 相同的参数形状：

```text
means       [N, 3]
scales      [N, 3]
quats       [N, 4]
opacities   [N]
```

但在 2DGS 中：

- `scales[:, 0]` 和 `scales[:, 1]` 表示椭圆面片的两个切向半轴；
- quaternion 旋转矩阵的前两列表示面片切向基；
- quaternion 旋转矩阵的第三列表示面片法向；
- `scales[:, 2]` 不参与主要的 2DGS 渲染，应固定为很小的物理厚度，例如
  `1e-4 m`。

因此无需改变参数格式，也无需改变 2DGS rasterizer 的接口。

## 3. 初始化

### 3.1 平面 anchor 内的初始 GS

对于属于可靠平面 anchor 的初始化 GS：

```text
GS normal = anchor normal
scale_x   = 局部点间距估计
scale_y   = 局部点间距估计
scale_z   = surfel_thickness
```

使用 quaternion 将 GS 的局部 Z 轴对齐到 anchor normal。局部点间距只用于
初始化两个切向 scale。

当前 2DGS 实现会在最近 anchor 匹配完成后立即执行该法向初始化：属于平面
anchor 的 GS quaternion 被设置为从局部 Z 轴旋转到 anchor normal 的最短
旋转；同时将已有初始化结果的两个切向 scale 乘以
`initial_tangent_scale_multiplier`（默认 `1.5`）以覆盖较稀疏的初始点间隔。
非平面 GS 的 quaternion 和 scale 保持普通初始化结果。该操作只扩大 surfel
覆盖范围，不会改变 GS 中心位置或增加初始 GS 数量。

### 3.2 非平面 GS

非平面 GS 不强制使用 anchor normal，保留普通 2DGS 初始化，并由图像监督、
normal consistency 和 distortion loss 继续优化。

## 4. relocate 的适配

现有 CUDA `relocate()` 可以继续复用，因为它已经负责：

- 根据 donor 权重选择源 GS；
- 复制位置和属性；
- 重新计算 donor 与 relocated GS 的 opacity；
- 调整 scale；
- 同步优化器状态。

relocate 完成后，2DGS 策略额外执行：

1. 将第三个 scale 恢复为 `log(surfel_thickness)`；
2. relocated GS 继承 donor 的 `anchor_id`；
3. 如果目标属于平面 anchor，立即把 GS 法向对齐 anchor normal；
4. 将 relocated GS 标记为 active，允许之后在可见视角中继续探索；
5. 根据动态 anchor 规则更新越界或进入其他 voxel 的 GS。

relocate 也会改变 donor 的第三个 scale。为了实现简单且不遗漏 donor，可以在
每次 refine 事件结束后统一恢复：

```python
params["scales"][:, 2].fill_(math.log(surfel_thickness))
```

该操作只在 refine 时执行，不是每个训练 step 执行。

## 5. add 的适配

现有 CUDA `sample_add()` 继续负责采样 donor、复制属性、调整 opacity/scale
以及扩展优化器状态。

add 完成后：

1. 新 GS 继承 donor 的 `anchor_id`；
2. 固定第三个 scale；
3. 属于平面 anchor 的新 GS 对齐 anchor normal；
4. 新 GS 被标记为 active；
5. 不立即加入较大的随机 jitter，由后续可见视角下的 MCMC noise 将其逐渐与
   donor 分开。

这样可以降低白色 donor 被一次性移动到周边区域形成环状伪影的风险。

## 6. 2D 切平面 noise

这是与 3D MCMC 最重要的区别。

3D MCMC 的位置噪声近似为：

\[
\Delta\boldsymbol\mu
=
\mathbf R\operatorname{diag}(s_x,s_y,s_z)\boldsymbol\epsilon.
\]

2DGS 应改为：

\[
\Delta\boldsymbol\mu
=
\mathbf r_1s_x\epsilon_x
+
\mathbf r_2s_y\epsilon_y,
\]

其中 \(\mathbf r_1,\mathbf r_2\) 是 quaternion 旋转矩阵的两个切向轴，
不生成沿法向 \(\mathbf r_3\) 的随机位移。

最终位移还要乘以 opacity gate 和随训练衰减的 noise scale：

\[
\Delta\boldsymbol\mu
\leftarrow
\Delta\boldsymbol\mu\;g(\alpha)\;\eta(t).
\]

这样可以保证：

- 地面 GS 主要沿地面探索；
- 墙面 GS 主要沿墙面探索；
- 低 opacity GS 移动较明显；
- 高 opacity GS 基本保持稳定；
- 不容易因为随机噪声产生离开表面的漂浮 GS。

仍然只对以下交集注入 noise：

```text
当前训练图像中可见的 GS
∩ 刚刚 add/relocate 的 GS
∩ 仍处于 active 时间的 GS
```

## 7. plane anchor 可微约束

对于平面 anchor，将 GS 相对 anchor 的位移分成平面内分量和法向分量：

\[
\mathbf d=\mathbf d_{\parallel}+\mathbf d_{\perp}.
\]

使用法向分量构造位置 loss：

\[
L_{plane-pos}=\operatorname{SmoothL1}(\mathbf d\cdot\mathbf n, 0).
\]

其中建议初始设置 `SmoothL1 beta = 0.02 m`。

法向使用符号无关的 loss，避免 CGAL normal 正负方向不一致：

\[
L_{plane-normal}=1-|\mathbf n_{GS}\cdot\mathbf n_{anchor}|.
\]

二者在前向之后加入总 loss，随后执行 `backward()` 和 `optimizer.step()`；
第一版不对优化后的参数做硬投影。平面内移动不使用球形距离截断。

非平面 anchor 可以不施加平面约束；free GS 完全跳过 anchor 约束。

## 8. 第一版静态 anchor ID 与 keeper

初始化 GS 只进行一次最近 anchor 匹配。relocate 和 add 后的新 GS 继承 donor
的 anchor ID。初始化时删除没有匹配到任何 GS 的空 anchor，因此每个有效
anchor 从训练开始就至少包含一个 GS。每次 relocate 前，在各 anchor 内选择
opacity 最高的一个 GS 作为 keeper；即使其 opacity 低于 `min_opacity`，也不
允许被 relocate。keeper 每次重新计算，不是永久绑定某个 GS。

第一版仍不根据位置动态更新 anchor ID，也不加入 free GS 或
`pending_relocate`。

## 9. donor score

2DGS donor 梯度只读取原生 densification signal：

```python
info["gradient_2dgs"]
```

并启用：

```text
absgrad = True
```

当前 score 使用 opacity 和 2DGS 原生梯度，texture 默认关闭：

\[
score=opacity+grad.
\]

如后续重新开启 texture，它通过 2DGS rasterizer 返回的真实 GS/像素相交
关系统计。对每个 GS 累加其覆盖像素的 weight-map 值，再除以覆盖次数；
不乘 alpha 或渲染贡献权重。

## 10. 训练器接口

`simple_trainer_2dgs.py` 向 strategy 的 `step_post_backward()` 传入
`packed`，而当前 Texture MCMC 接口传入 `lr`。新策略增加一层接口适配即可：

```python
def step_post_backward(
    self,
    params,
    optimizers,
    state,
    step,
    info,
    packed=False,
    scene=None,
):
    del packed
    super().step_post_backward(
        params,
        optimizers,
        state,
        step,
        info,
        lr=0.0,
        scene=scene,
    )
```

Texture MCMC 使用独立的 `noise_scale`，所以这里不依赖 means optimizer 的
学习率。

## 11. 损失函数

第一版保留 2DGS 原生训练损失：

\[
L=
L_{RGB}
+\lambda_dL_{depth}
+\lambda_nL_{normal}
+\lambda_{dist}L_{distortion}.
\]

暂时不加入：

- 3D needle scale loss；
- 第三轴 scale loss；
- 每次 refine 重建全体 GS KNN 的 normal/plane loss；
- 普通梯度更新后对所有 GS 强制投影。

这些约束与 2DGS 的面片表达、normal consistency 和 distortion loss 存在重复，
还会增加训练时间。KNN plane/normal loss 是否保留，应在单独讨论其目标、梯度
对象和开销后决定。

## 12. 第一版建议参数

```text
surfel_thickness          = 1e-4
plane_position_beta      = 0.02
plane_position_lambda    = 0.1
plane_normal_lambda      = 0.01
min_opacity              = 0.005
active_noise_steps       = 500
use_opacity_score        = True
use_grad_score           = True
use_texture_score        = False
opacity_weight           = 1.0
grad_weight              = 1.0
texture_weight           = 0.0
texture_update_every     = 10
keep_one_per_anchor      = True
absgrad                  = True
normal_loss              = True
dist_loss                = True
```

## 13. 预期代码结构

```text
VoxelPlaneMCMC2DStrategy
├── 现有 CUDA relocate
├── 现有 CUDA sample_add
├── opacity/grad donor score
├── 2D tangent-plane noise
├── static inherited anchor_id
├── differentiable plane-normal loss
└── differentiable point-to-plane loss

2DGS trainer
├── rasterization_2dgs
├── RGB loss
├── LiDAR depth loss
├── normal consistency loss
└── distortion loss
```

## 14. 实验顺序

建议按以下顺序验证，避免同时修改太多因素：

1. 标准 2DGS + DefaultStrategy；
2. 2DGS + MCMC relocate/add，但关闭 noise；
3. 启用可见 active GS 的切平面 noise；
4. 启用静态 plane anchor 位置与法向 loss；
5. 最后再比较是否需要动态 anchor、free GS、keeper、KNN loss 和 texture score。

重点观察：PSNR/SSIM/LPIPS、地面和墙面平整度、边界清晰度、漂浮 GS、平面
区域的 GS 分布，以及单步训练时间。
