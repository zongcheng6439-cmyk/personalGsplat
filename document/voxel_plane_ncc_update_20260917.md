# Voxel-plane 修正与可选 NCC（2026-09-17）

适用入口：`examples/train_voxel_plane_mcmc.py original` 与 `dynamic`。

## 第三轴厚度

`_add_anchor_scale_gradient()` 分开处理两种范围：

- 全部 GS（包括没有 anchor 的 free GS）：逐轴惩罚超过 `anchor_scale_max_ratio * voxel_size` 的尺度。该项是软损失，不是每步硬裁剪；`large_scale_gs` 统计全体 GS 中任意轴超限的点数。
- `plane_ids`：其中 `anchor_is_plane=True` 的 GS，只对这批 GS 的第三轴施加 `plane_s3_max` 厚度损失。

厚度损失仍覆盖全部 plane GS，而不是仅当前可见 GS。非平面区域不被强制压薄；这次不添加全体 GS 最短轴正则。dynamic 中没有有效 anchor 的 GS 只接受全局尺度上限惩罚，不接受平面第三轴厚度约束。

## KNN 法向与位置

保留原有实际最短轴法向、8 邻居、6 票、10 度和 `2 * voxel_size` 距离筛选。

法向目标先按中心法向停止梯度地选择正负方向，然后计算 `mean(abs(normal - oriented_target))`。这消除了原 `1 - abs(dot)` 在精确垂直处的零梯度。

位置约束增加一道保守筛选：把同组邻居投影到投票主法向，计算 `max(height) - min(height)`。只有厚度不超过 `plane_normal_tolerance` 才计算中心到邻居均值的法向距离损失。默认阈值为 0.02 m；命令行若覆盖为 0.03 m，则使用 0.03 m。

双层高度差超限时，跳过位置损失但保留法向指导。它不会自动选出“正确一层”；只是避免把两层平均到中间。厚度在容差内的层仍可能合并。邻居与投票结果停止梯度，仍只优化可见中心。

## Noise 与位置约束

`TextureMCMCStrategy._inject_active_noise()` 返回本次实际改变位置的 GS ID。只有 active 且具有当前视角梯度、并且实际发生位移的 GS 才出现在返回值中。

两个 plane 策略覆盖 `_apply_active_noise()`：

1. 达到 noise stop 时直接返回。
2. 获取 active GS，由 noise 内部筛选当前视角并执行随机移动。
3. original 对返回的 moved IDs 执行原有 slab/半径反射。
4. dynamic 先为 moved IDs 更新 anchor，再对这些 IDs 执行原有约束。

未被 noise 移动的 GS 不再因该步骤被反射。add/relocate 后的单独位置约束保留。Texture、weighted、2D 等其他策略沿用默认调度；这次没有更改它们的采样权重和参数设置。当前服务器入口的 `noise_scale=30` 被原样保留。

## NCC：开关与执行

实现集中在 `examples/ncc_loss.py`。在现有 original/dynamic 命令后添加：

```bash
--ncc-loss \
--ncc-lambda 0.01 \
--ncc-start-iter 5000 \
--ncc-warmup-steps 5000 \
--ncc-every 4 \
--ncc-samples 1024
```

不添加 `--ncc-loss` 即关闭，默认关闭。也可以显式使用 `--no-ncc-loss`。

默认 5000 步开始，5000–10000 步线性升至 0.01，每四步计算一次，不对稀疏执行的权重额外乘 4。每次最多 1024 个 5x5 patch。`steps_scaler` 同步缩放开始步数与 warmup 步数。原有 depth loss、scale loss、noise、SH 和学习率参数不变。

当前版本要求 COLMAP、pinhole、batch_size=1、不开 pose_noise。源视图需有数据加载器支持的稠密深度与 mask；即使关闭 depth loss，NCC 仍需要这些源深度进行遮挡筛选。

数据流：

1. COLMAP 共享点轨迹选择一个训练集内源视图：共享点至少 50、基线 0.05–30 m、朝向差不超过 75 度。
2. 当前真实图像挑选有纹理的 patch；过滤无效 mask、低 alpha 和深度跳变边缘。
3. 用当前渲染深度的 11x11 邻域拟合局部平面：有效点 >=90%、RMSE <=2 cm、最小/次小特征值比例 <=0.05。
4. 用平面单应变换把参考 patch 投影到源图，检查源深度遮挡，容差 `max(0.15 m, 2% * projected_depth)`。
5. 计算灰度零均值归一化相关系数 rho，损失为 `mean(1-rho)`。

PCA 筛选与拟合法向停止梯度，以避免平面重复特征值导致不稳定反向传播。平面距离保留渲染深度梯度，因此仍有：`NCC -> 采样位置 -> 平面距离 -> 渲染深度 -> GS参数`。

不开启 NCC 时不建立共视表、不读取源图、不产生 NCC 深度渲染。开启时只有计算 NCC 的 step 才额外请求深度；若原有 depth loss 已请求深度，则直接复用。源图采用最多八张 CPU 缓存。NCC 信息只输出到终端，不写入 TensorBoard。

## 文件位置与移除

- `examples/ncc_loss.py`：共视选择、筛选、平面拟合、warp、NCC。
- `examples/train_voxel_plane_mcmc.py`：flag、调度、调用，以及本轮 plane/KNN 损失修正。
- `examples/simple_trainer_texture.py`：仅补充通用额外损失的上下文和按需深度接口。
- `examples/train_weighted_anchor_normal_mcmc.py`：仅接受多余上下文参数，保持旧 hook 兼容，损失逻辑不变。
- `gsplat/strategy/{texture_mcmc,voxel_plane_mcmc,plane_mcmc}.py`：noise 返回 ID 及 plane 约束调度。

临时禁用只需关闭 flag。彻底删除时移除 NCC 模块及 voxel-plane 入口中标记的 NCC 配置、校验、初始化、调度方法，并去掉 `adjust_steps()` 中的两个 NCC 字段缩放。通用 hook 可以保留。不要删除本轮独立的 plane/KNN 修正。

## 验证与运行路径

18 项小规模测试通过，覆盖 CPU/CUDA 厚度范围、垂直法向梯度、跨层位置过滤、noise ID、遮挡过滤、NCC 深度扰动、original/dynamic CLI 和经现有 CUDA 渲染器回传到 GS 位置。没有启动完整场景训练，尚未证明实际 NVS 指标提升。

当前检查容器默认从 `/opt/gsplat-venv/lib/python3.10/site-packages/gsplat-1.5.3-py3.10-linux-x86_64.egg/gsplat` 导入。源码上传不等于安装包自动更新。训练前在实际训练环境检查：

```bash
python -c 'import gsplat.strategy.voxel_plane_mcmc as m; print(m.__file__)'
```

使用已更新的安装包，或按现有工作流程配置源码导入路径。此次没有覆盖容器安装包、没有更换 CUDA 二进制，也没有修改训练 shell。验证使用已有 `csrc.so`，不需要重新编译本轮 Python 改动。
