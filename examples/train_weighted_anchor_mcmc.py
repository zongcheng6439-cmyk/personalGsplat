"""Weighted-anchor MCMC training; input COLMAP/RGB/depth/anchors, output models.

From /Users/zongcheng/code/3dgs/personalGsplat (replace example data paths):
CUDA_VISIBLE_DEVICES=0 python examples/train_weighted_anchor_mcmc.py --data-dir /path/to/068ba2946c --strategy.anchor-ply /path/to/anchors.ply --depth-loss --depth-tag uint16 --result-dir results/068ba2946c
uint16 selects scene/depth/ and converts millimeters to meters (zero invalid).
Use --depth-dir depths_render_color_edge for prefiltered depth PNGs.
auto (default) reads the prepared directory metadata; la selects legacy encoding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import tyro

from gsplat.distributed import cli
from gsplat.strategy.weighted_anchor_mcmc import WeightedAnchorMCMCStrategy
import simple_trainer_texture as base


def _default_strategy() -> WeightedAnchorMCMCStrategy:
    return WeightedAnchorMCMCStrategy(
        absgrad=True,
        texture_update_every=1,
        use_texture_score=True,
        use_grad_score=True,
        texture_weight=2,
        grad_weight=1,
        noise_injection_stop_iter=-1,
        surface_anchor_constraint=False,
        surface_anchor_max_distance=0.1,
        verbose=True,
    )


@dataclass
class Config(base.Config):
    normalize_world_space: bool = False
    depth_loss: bool = False
    means_lr: float = 1.6e-5

    init_opa: float = 0.5
    init_scale: float = 0.1
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    needle_scale_reg: float = 0
    strategy: WeightedAnchorMCMCStrategy = field(default_factory=_default_strategy)


def _validate_config(cfg: Config, world_size: int) -> None:
    if world_size != 1:
        raise ValueError("Weighted Anchor MCMC currently supports one GPU")
    if cfg.normalize_world_space:
        raise ValueError("anchor coordinates require normalize_world_space=False")
    if cfg.packed:
        raise ValueError("texture statistics require packed=False")
    if not cfg.strategy.anchor_ply:
        raise ValueError("pass --strategy.anchor-ply")
    anchor_path = Path(cfg.strategy.anchor_ply).expanduser()
    if not anchor_path.is_file():
        raise FileNotFoundError(f"anchor PLY does not exist: {anchor_path}")
    cfg.strategy.anchor_ply = str(anchor_path)


class Runner(base.Runner):
    def __init__(self, local_rank, world_rank, world_size, cfg):
        _validate_config(cfg, world_size)
        super().__init__(local_rank, world_rank, world_size, cfg)


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    original_runner = base.Runner
    base.Runner = Runner
    try:
        return base.main(local_rank, world_rank, world_size, cfg)
    finally:
        base.Runner = original_runner


if __name__ == "__main__":
    config = tyro.cli(Config)
    if config.backend == "dgx":
        import torch_dgx  # noqa: F401
    config.adjust_steps(config.steps_scaler)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one GPU")
    cli(main, config, verbose=True)
