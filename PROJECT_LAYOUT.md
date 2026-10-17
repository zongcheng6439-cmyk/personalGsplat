# Project layout

The maintained gsplat checkout is the repository root:

```text
/home/zc/code/aibee/gsplat/
```

Main entry points:

```text
utils/train/train_citygs_mcmc.sh
utils/train/train_all_partitions.py
utils/train/train_block_densification.py
utils/train/train_texture_anchor_block.py
utils/train/train_weighted_anchor_block.py
utils/train/train_classified_texture_anchor_block.py
```

Utilities are grouped under:

```text
utils/data_process/
utils/data_process/depth_process/
utils/partition/
utils/result_process/
utils/train/
```

`third/` contains third-party dependencies only.

The root-level build, distribution, egg-info and CUDA artifacts are intentionally
retained to avoid unnecessary recompilation.

Typical large-scene command:

```bash
cd /home/zc/code/aibee/gsplat
bash utils/train/train_citygs_mcmc.sh DATA_DIR RESULT_DIR all
```

The source package, examples, utilities and build outputs are all located at the
repository root, alongside `third/`.

# weight anchor mcmc 
1. voxel has min gs one and no redundant  

2. add normal loss 

1. control noise number 

2. through depth loss to 

3. normal loss and depth loss 

4. gs only random go in lidar surface   



