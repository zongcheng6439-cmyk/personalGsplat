# Gaussian opacity 二值化正则

适用入口：`train_voxel_plane_mcmc.py original` 和 `dynamic`。

参考 SuGaR（CVPR 2024）：[论文 §5.1](https://openaccess.thecvf.com/content/CVPR2024/papers/Guedon_SuGaR_Surface-Aligned_Gaussian_Splatting_for_Efficient_3D_Mesh_Reconstruction_and_CVPR_2024_paper.pdf)，[官方代码](https://github.com/Anttwo/SuGaR/blob/main/sugar_trainers/coarse_sdf.py)。论文先训练 7000 次，再进行 2000 次 opacity 熵正则，随后剔除 opacity < 0.5 的点。这里借鉴熵正则，不复制整套训练/剔除流程。

对当前视图投影有效（radii > 0）的 Gaussian，以参数 logits 经 sigmoid 得到 alpha，最小化：

```text
L_entropy = mean(-alpha * log(alpha) - (1-alpha) * log(1-alpha))
L_total += effective_weight * L_entropy
```

多相机 batch 取可见点并集，每点计一次；这里的可见是投影半径有效，并不意味着没有被其他点遮挡。不作用于逐像素累积 alpha。二元熵在 alpha=0.5 时最大，在 0/1 两端趋近零。采用 logits 的稳定等价式计算，避免 log(0) 和硬截断；空可见集合返回可微的零。

## 使用

在已有的完整训练命令后追加（权重是建议的初始实验值，并非已调优结果）：

```bash
--opacity-entropy-lambda 0.01 \
--opacity-entropy-start-iter 7000 \
--opacity-entropy-warmup-steps 1000
```

容器中 068ba2946c 示例（anchors 须已生成，体素尺寸须匹配）：

```bash
cd /code/gsplat
python examples/train_voxel_plane_mcmc.py original \
  --data-dir /data \
  --result-dir /data/results/voxel_plane_entropy \
  --strategy.plane-anchor-ply /data/cgal_voxel_plane_anchors.ply \
  --strategy.anchor-voxel-size 0.05 \
  --depth-loss --depth-tag uint16 --depth-dir depth --depth-scale 1000 \
  --opacity-entropy-lambda 0.01 \
  --opacity-entropy-start-iter 7000 \
  --opacity-entropy-warmup-steps 1000
```

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `--opacity-entropy-lambda` | 0 | 最大权重；0 关闭，保持原有训练行为 |
| `--opacity-entropy-start-iter` | 7000 | 开始步骤，使用训练内部从 0 计数的 step |
| `--opacity-entropy-warmup-steps` | 1000 | 从 0 线性增加到最大权重；0 表示立即全权重 |
| `--opacity-entropy-stop-iter` | -1 | 独立停止步骤，不包含该步；-1 持续到训练结束 |

以上默认时，step 7000 权重为 0，step 8000 达到完整权重。步骤参数随 `--steps-scaler` 同步缩放。若只想在一个阶段启用，可另外指定 `--opacity-entropy-stop-iter 10000`。

## 与现有损失的区别

`--opacity-reg` 是 mean(alpha)，只偏向透明；熵正则同时偏向 0 和 1。本次保留已有 `opacity_reg=0.01`，没有替换它。二者同时使用会使综合正则偏向 0，最终还需重建损失支持有贡献的点。可在对照实验中单独调低 opacity-reg，但它也参与 MCMC 的稀疏化行为，不应未经验证直接认为可以去掉。

恰好 alpha=0.5 时，熵项梯度为 0，需要其他损失打破对称；不保证每个点达到精确 0/1，也不会直接删除点。权重过大或启用过早可能影响细小结构、透明区域和新点优化。优先在重建已稳定后启用，比较 PSNR、深度以及可见点 opacity 分布。

## 日志与验证

激活期间按 `--tb-every` 输出终端日志及 TensorBoard：`train/opacity_entropy`、`train/opacity_entropy_weighted`、`train/opacity_entropy_weight`、`train/opacity_entropy_visible_count`、`train/opacity_visible_below_001`、`train/opacity_visible_above_099`。比例统计针对当前可见集合，视角改变时可能波动。

回归检查：在训练环境执行 `python tests/test_opacity_entropy.py`，覆盖公式、梯度方向、极端 logits、空集合、可见性、启停/warmup、PGSR 损失叠加及共享配置的参数缩放（通过抽取实际 hook 源码测试，不加载渲染器）。
