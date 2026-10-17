# Voxel / Plane MCMC：局部平滑修改记录

日期：2026-09-17。

## 当前版本：统一 KNN，不再区分模式

按用户要求，将原 `all` 的计算统一为唯一实现，移除 `legacy/nonplane` 分支及
`knn_loss_mode` 配置，不保留模式别名。`original` 与 `dynamic` 共用这套损失：

- 所有类别的当前可见 GS 都可作为中心，邻居保持固定参考。
- 保留距离、6/8 法向投票和共面检查，法向及位置损失共用通过筛选的中心集合。
- 不再保留“邻居跨层时仍算 normal loss”的旧行为。
- 删除冗余 `same_plane` / `plane_valid` 中间掩码，位置损失直接对已通过筛选的中心求平均。
- KNN 失败仍不触发 relocate，不改变 opacity、noise、donor 或 anchor 自身损失。

训练命令不再需要 `--knn-loss-mode`；旧命令中的 `all`、`legacy`、`nonplane` 参数行均应删除。
服务器 `train_section.sh` 仅删除一行 `--knn-loss-mode legacy`，结果目录和其余内容不改。
结果目录名称中的 `legacy` 只是历史命名，不再代表一种可选算法；做新旧对照时应自行另设结果目录。

仍可用 `--knn-plane-lambda 0 --knn-normal-lambda 0` 关闭两项 KNN 损失。
本轮 CPU/CUDA 回归：**41 passed in 6.02s**；Python 语法和 `train_section.sh` 的 Bash
语法检查通过。覆盖统一作用范围、跨层双损失拒绝、旧模式参数移除及原有尺度/noise/NCC。
没有进行整场景质量对照。下面保留之前两轮修改记录和论文链接，仅用于追溯，
不代表当前参数接口或计算分支。

---

## 历史修订：所有类别的可见中心都参加 KNN

用户确认移除“排除平面 anchor 中心”的筛选。本次已直接删除该筛选代码，不再根据
`anchor_is_plane` 排除 KNN 中心，平面、非平面和 free GS 一视同仁；仍只对当前可见
中心施加梯度，邻居作为固定参考。

- 新默认参数改为 `--knn-loss-mode all`，不再使用 `nonplane` 这个名称。
- 保留 `2 * voxel_size` 距离、6/8 法向投票和邻居高度跨度的共面检查。
- `all` 中，共面检查失败时，法向与位置损失一起跳过，不触发 relocate。
- `legacy` 原样保留：所有可见中心也参加，但跨层时只跳过位置损失，仍计算法向损失。
- 平面 anchor 自身的法向和 S3 损失、其他超参数、策略与训练脚本均不改。
- 检查发现服务器 `train_section.sh` 显式指定 `--knn-loss-mode legacy`；没有改动该文件。
  运行新模式时需把该次命令中的参数改成 `--knn-loss-mode all`。

本轮 CPU/CUDA 回归：**43 passed in 2.98s**，包含平面中心接收梯度、混合可见中心、
平面/非平面跨层拒绝和新旧 CLI 参数；本地语法检查通过，未进行整场景质量对照。
以下内容是上一轮 `nonplane` 实验的历史记录，不再代表当前默认模式；其论文参考仍然适用。

---

## 上一轮记录：仅非平面中心的 KNN 实验

## 本次修改范围

修改 `examples/train_voxel_plane_mcmc.py` 中的 KNN 损失，`original` 和
`dynamic` 子命令共用。先读取并比较 `zc@192.168.88.85` 的最新文件，再制作增量补丁。

只增加一个配置：`knn_loss_mode`，默认 `nonplane`；设置 `legacy` 可恢复本次修改前
的 KNN 计算对象和损失行为。两个模式共用查询结果、投票和梯度代码，不复制整套函数。

以下内容不变：

- 原有 anchor normal、plane S3、全体 GS 的各轴尺度上限损失。
- 原有深度损失、NCC、初始化、学习率、noise、donor score、add/relocate 和 keeper。
- 原有 KNN 查询方法与更新时机，不增加查询。
- `noise_scale=30`、`ncc_samples=2048` 等服务器已有参数。
- MCMC 策略文件、普通 texture MCMC 和其他训练脚本。

本次没有实现 CGAL 跨 anchor 共享 plane_id、新的 anchor 软位置损失或自动 floater
检测。平面 GS 继续使用现有 anchor 约束，并不意味着每步新增一次位置投影。

## 新旧模式

| 条件 | `nonplane`：新默认 | `legacy`：保留旧行为 |
| --- | --- | --- |
| 可见的平面 anchor GS，作为中心 | 跳过 KNN，继续原有 anchor 约束 | 参加 KNN |
| 可见的非平面 / free GS，作为中心 | 参加候选筛选 | 参加候选筛选 |
| 邻居候选 | 全体 GS，包括平面 anchor GS | 全体 GS |
| 邻居法向不满足投票 | 两项 KNN 损失都跳过 | 两项 KNN 损失都跳过 |
| 法向通过，但邻居分成平行多层 | 两项 KNN 损失都跳过 | 跳过位置损失，保留法向损失 |
| 方向、距离、共面性均通过 | 计算法向和位置损失 | 计算法向和位置损失 |
| 不可见中心 | 不施加 KNN 梯度 | 不施加 KNN 梯度 |
| KNN 未通过后是否加入 dead / pending | 否 | 否 |

`nonplane` 是按 anchor 的平面标志划分约束责任，不是物体语义分类。
它使用 `anchor_is_plane[anchor_ids]`。动态版已有的 free anchor 哨兵标志为 False，
因此 free GS 也能作为 KNN 中心，不会因缺少真实 anchor 而被漏掉。

## 计算过程

1. 读取当前 `visible_gs_ids`；新模式只保留非平面 anchor / free 的中心 ID。
2. 用这些中心 ID 索引已有 `gs_knn_ids`。只筛中心，不筛掉平面邻居。
3. 使用每个 GS 的实际最短尺度轴作为法向，不假定所有 GS 的第三轴都是最短轴。
4. 保留距离中心不超过 `2 * anchor_voxel_size` 的邻居，并排除中心自身。
5. 沿用向量化投票：默认 8 个邻居，至少 6 个法向与某候选方向夹角小于 10°。
   法向正负视为同一方向；符号对齐后求主法向。`plane_knn` 和投票阈值本次不改。
6. 同组邻居沿主法向的高度跨度必须不超过 `plane_normal_tolerance`。
   当前训练脚本默认 0.02 m，新模式要求法向和位置损失共同通过这个条件。
7. 筛选、投票与邻居目标全部停止梯度；仅给通过筛选的可见中心累加梯度。

设同组邻居的平均位置为 `mu_ref`，投票法向为 `n_ref`：

```text
position_loss = mean(abs(dot(mu_center - mu_ref, n_ref)))
normal_loss   = mean(abs(n_center - sign_aligned(n_ref)))
```

位置损失只沿法向纠正，不把中心拉向邻居均值的平面内位置。法向损失沿用停止梯度
的正负方向选择后 L1 差，以保留 90° 偏差时的修正梯度。

本次没有给 KNN 位置损失增加死区：0.02 m 在这里是“邻居是否近似共面”的筛选阈值，
不是“中心离面 0.02 m 内没有损失”。正常通过的中心仍按上面的原有 L1 公式优化。

所有候选都被筛掉时，两项 KNN 记录为 0，不进行空张量均值反向传播。
沿用终端现有 `knn_plane_loss` / `knn_normal_loss` 输出，不添加 writer 项或每 GS 状态。

## 为什么失败不触发 relocate

没有形成可靠局部平面，可能来自墙角、曲面、细小物体、邻居不足或暂未优化好，
不能直接证明 GS 悬浮。一团共同悬浮的 GS 也可能通过 KNN。

因此本次不记录 KNN 失败 ID，不修改 opacity，不设置 `pending_relocate`，也不把失败
结果并入 `dead_mask`。这些 GS 继续接受原有图像、深度和全局尺度优化。
已有动态策略自己的 pending / relocate 行为保持不变。

## 使用与回退

在现有 `original` 或 `dynamic` 训练命令中追加下面一行即可，不修改数据路径等参数：

```bash
--knn-loss-mode nonplane
```

这是新默认。复现修改前的 KNN：

```bash
--knn-loss-mode legacy
```

只关闭 KNN 损失、保留其他约束：

```bash
--knn-plane-lambda 0 --knn-normal-lambda 0
```

注意最后一种只关闭损失；策略中现有 KNN 初始化和 refine 后更新仍执行，不代表已消除
全部查询开销。本次没有修改策略查询，也没有做整场景训练耗时基准。

建议新旧两组只切换 `knn_loss_mode`，固定数据、随机种子、训练步数与其他参数。
观察平面接缝、车辆边缘、文字细节和新视角漂浮情况。固定训练图指标只能衡量拟合，
不能单独证明 NVS 提升。

## 验证记录

回归测试位于本地 `tests/test_plane_ncc_regression.py`，使用服务器隔离临时目录执行，
不覆盖训练数据，不安装包，不重新编译 CUDA，不启动整场景训练。覆盖：

- 平面中心在新模式不接收 KNN 梯度，旧模式仍接收。
- 非平面与 free 中心可使用平面邻居，邻居属性不接收该中心损失的梯度。
- 多层、距离超限、方向不一致时，新模式两项损失均跳过，不触发 relocate。
- 混合可见中心索引、实际最短轴选择、90° 法向梯度和两种 CLI 配置。
- 原有全 GS 尺度、plane S3、noise 对象及 NCC 回归。

实际结果：CPU/CUDA 回归 **35 passed in 2.42s**；本地 Python 语法检查通过。
尚未进行整场景 A/B 训练，没有宣称已改善真实场景的 NVS。

## 与 floaters 相关的四篇论文

下列是论文方法参考，不是本次已经实现的功能；不把预印本当作已确认的顶会录用。

### 1. FreeSplat++（2025）：优先看 Weighted Floater Removal

[论文，重点看 §IV-D](https://arxiv.org/html/2503.22986v1)

通过跨视角投影比较 Gaussian 深度与参考深度，并结合累积权重调整候选的 opacity。
它不是简单地把所有前方 GS 一次性删掉。对本项目最有启发的是“深度冲突 + 可靠性权重”，
但原方法面向前馈融合模型，其权重不能直接等同于我们的 texture score。

### 2. TIDI-GS（2026，arXiv）：看候选筛选与细节保护

[论文](https://arxiv.org/html/2601.09291v1)

结合持续的可见性、优化活动、opacity / importance 和空间孤立性筛选候选，并设置细节
保护与周期清理。借鉴重点是“多个证据持续支持再处理”，而非直接把 KNN 失败等同于
floater；完整实现会增加状态和判断，不建议直接全部移植。

### 3. StableGS（2025，arXiv）：看为什么 photometric loss 不一定消除 floaters

[论文](https://arxiv.org/abs/2503.18458)

分析漂浮伪影为何可能停在错误局部解，提出分离几何正则与外观渲染的 dual-opacity
路径，并结合深度一致性及外部几何先验。适合理解几何约束与 NVS 外观之间的冲突；
不是一个只增加少量 KNN 判断就能直接复现的方法。

### 4. SparseGS（2023 起）：看显式 Floater Pruning

[论文 §3 Floater Removal](https://arxiv.org/html/2312.00206v1)
 / [作者代码](https://github.com/ForMyCat/SparseGS)

比较 alpha-blended depth 与 mode depth（最大 alpha*T 贡献项的深度），利用差异分布
识别可疑像素，再映射回 Gaussian 清理。适合研究显式回收，但面向稀疏视角，且需要
相应深度输出和像素到 GS 的映射；不能把当前普通 ED 输出直接当成 mode depth。

以上各方法都不意味着“没有平面邻居 = 悬浮”。本次仅调整局部平滑责任与可靠性门控。
