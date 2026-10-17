#!/usr/bin/env python3
"""Train one block with MCMC or a texture-guided MCMC variant.

Examples:

    CUDA_VISIBLE_DEVICES=0 python utils/train/train_block_densification.py \
        mcmc --data-dir /data --coarse-dir /result/coarse \
        --block-cameras-json /data/partition/cameras-000_001.json \
        --result-dir /result/mcmc/block/000_001

    CUDA_VISIBLE_DEVICES=0 python utils/train/train_block_densification.py \
        texture_mcmc --data-dir /data --coarse-dir /result/coarse \
        --block-cameras-json /data/partition/cameras-000_001.json \
        --result-dir /result/texture_mcmc/block/000_001

    CUDA_VISIBLE_DEVICES=0 python utils/train/train_block_densification.py \
        texture_voxel --data-dir /data --coarse-dir /result/coarse \
        --block-cameras-json /data/partition/cameras-000_000.json \
        --result-dir /result/texture_voxel/block/000_000 \
        --strategy.anchor-ply /data/voxel_target_count.ply

    CUDA_VISIBLE_DEVICES=0 python utils/train/train_block_densification.py \
        voxel_plane --data-dir /data --coarse-dir /result/coarse \
        --block-cameras-json /data/partition/cameras-000_000.json \
        --result-dir /result/voxel_plane/block/000_000 \
        --strategy.plane-anchor-ply /data/voxel_plane_anchors.ply

"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import torch
import tyro

REPO_DIR = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_DIR / "examples"
TRAIN_DIR = REPO_DIR / "utils" / "train"
for path in (EXAMPLE_DIR, TRAIN_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_texture_anchor_block as anchor_block  # noqa: E402
import train_voxel_plane_block as plane_block  # noqa: E402
import train_weighted_anchor_block as voxel_block  # noqa: E402
from gsplat.distributed import cli  # noqa: E402
from gsplat.strategy import (  # noqa: E402
    MCMCStrategy,
    TextureMCMCStrategy,
)
from gsplat.strategy.weighted_anchor_mcmc import (  # noqa: E402
    WeightedAnchorMCMCStrategy,
)
from gsplat.strategy.voxel_plane_mcmc import VoxelPlaneMCMCStrategy  # noqa: E402


Strategy = Union[
    MCMCStrategy,
    TextureMCMCStrategy,
    WeightedAnchorMCMCStrategy,
    VoxelPlaneMCMCStrategy,
]


@dataclass
class Config(anchor_block.Config):
    """Shared block configuration; dense depth supervision is mandatory."""

    depth_loss: bool = True
    large_scale_reg: float = 0.0
    strategy: Strategy = field(default_factory=TextureMCMCStrategy)


CONFIGS = {
    "mcmc": (
        "Original MCMC densification without texture scores or anchors.",
        Config(
            init_opa=0.5,
            init_scale=0.1,
            opacity_reg=0.01,
            scale_reg=0.01,
            strategy=MCMCStrategy(verbose=True),
        ),
    ),
    "texture_mcmc": (
        "Texture MCMC with one surface anchor per initialized coarse GS.",
        Config(
            strategy=TextureMCMCStrategy(
                absgrad=True,
                texture_update_every=1,
                use_texture_score=True,
                use_grad_score=True,
                active_noise_steps=-1,
                noise_injection_stop_iter=-1,
                surface_anchor_constraint=True,
                surface_anchor_max_distance=0.1,
                verbose=True,
            )
        ),
    ),
    "texture_voxel": (
        "Texture MCMC with weighted voxel anchors; redundant relocation disabled.",
        Config(
            strategy=WeightedAnchorMCMCStrategy(
                anchor_ply=None,
                anchor_voxel_size=0.1,
                redundant_move=False,
                absgrad=True,
                texture_update_every=1,
                use_texture_score=True,
                use_grad_score=True,
                active_noise_steps=-1,
                noise_injection_stop_iter=-1,
                surface_anchor_constraint=True,
                surface_anchor_max_distance=0.1,
                verbose=True,
            )
        ),
    ),
    "voxel_plane": (
        "Texture MCMC with block-filtered voxel-plane anchors.",
        # Reuse the standalone voxel-plane defaults through the block config.
        # Keeping a second parameter list here caused the two entry points to
        # silently diverge (notably donor weights and active-noise duration).
        plane_block.Config(),
    ),
}


def main() -> None:
    cfg = tyro.extras.overridable_config_cli(CONFIGS)
    cfg.adjust_steps(cfg.steps_scaler)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one GPU")

    if isinstance(cfg.strategy, VoxelPlaneMCMCStrategy):
        train = plane_block.main
    elif isinstance(cfg.strategy, WeightedAnchorMCMCStrategy):
        train = voxel_block.main
    else:
        train = anchor_block.main
    cli(train, cfg, verbose=True)


if __name__ == "__main__":
    main()
