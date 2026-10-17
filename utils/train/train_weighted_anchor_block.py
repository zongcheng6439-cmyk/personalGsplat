#!/usr/bin/env python3
"""Train one CityGS block with the latest WeightedAnchorMCMC strategy."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import tyro

REPO_DIR = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_DIR / "examples"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import simple_trainer_texture as base  # noqa: E402
import block_utils as block  # noqa: E402
import train_texture_anchor_block as common  # noqa: E402
from gsplat.distributed import cli  # noqa: E402
from gsplat.strategy import read_anchor_capacity_ply  # noqa: E402
from gsplat.strategy.weighted_anchor_mcmc import (  # noqa: E402
    WeightedAnchorMCMCStrategy,
)


def default_strategy() -> WeightedAnchorMCMCStrategy:
    return WeightedAnchorMCMCStrategy(
        absgrad=True,
        texture_update_every=1,
        use_texture_score=True,
        use_grad_score=True,
        noise_scale=10.0,
        redundant_move=False,
        surface_anchor_constraint=True,
        surface_anchor_max_distance=0.3,
        verbose=True,
    )


@dataclass
class Config(base.Config):
    test_every: int = 0
    eval_steps: list[int] = field(default_factory=list)
    coarse_dir: str = ""
    block_cameras_json: str = ""
    partitions_path: Optional[str] = None
    init_block_id: Optional[str] = None
    init_block_bbox_enlarge: float = 0.0
    init_block_bbox_margin: Optional[float] = None
    init_ply_path: Optional[str] = None

    normalize_world_space: bool = False
    depth_loss: bool = True
    init_opa: float = 0.5
    init_scale: float = 0.1
    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    large_scale_reg: float = 0.1
    large_scale_threshold: float = 0.01
    strategy: WeightedAnchorMCMCStrategy = field(default_factory=default_strategy)


def validate(cfg: Config, world_size: int) -> None:
    common.validate(cfg, world_size)
    anchor_path = Path(cfg.strategy.anchor_ply or "").expanduser()
    if not anchor_path.is_file():
        raise FileNotFoundError(f"anchor_ply does not exist: {anchor_path}")
    cfg.strategy.anchor_ply = str(anchor_path)


def find_latest_coarse_ply(coarse_dir: str) -> str:
    candidates = []
    for path in (Path(coarse_dir) / "ply").glob("point_cloud_*.ply"):
        match = re.search(r"(\d+)(?=\.ply$)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if candidates:
        return str(max(candidates, key=lambda item: item[0])[1])
    return block.find_latest_coarse_ply(coarse_dir)[0]


BaseRunner = base.Runner


class Runner(BaseRunner):
    def bind_strategy_state(self) -> None:
        cfg = self.cfg
        xyz, weights = read_anchor_capacity_ply(
            cfg.strategy.anchor_ply,
            expected_voxel_size=cfg.strategy.anchor_voxel_size,
        )
        partitions_path = cfg.partitions_path or str(
            Path(cfg.block_cameras_json).parent / "partitions.pt"
        )
        block_id = block._parse_block_id(
            cfg.block_cameras_json, cfg.init_block_id
        )
        if block.find_preselected_block_ply(partitions_path, block_id):
            selected = {
                "means": torch.as_tensor(
                    xyz, dtype=self.splats["means"].dtype
                ),
                "weights": torch.as_tensor(weights, dtype=torch.float32),
            }
        else:
            selected = block._filter_splats_for_block(
                {
                    "means": torch.as_tensor(
                        xyz, dtype=self.splats["means"].dtype
                    ),
                    "weights": torch.as_tensor(weights, dtype=torch.float32),
                },
                partitions_path,
                block_id,
                cfg.init_block_bbox_enlarge,
                cfg.init_block_bbox_margin,
            )
        cfg.strategy.set_anchors(
            self.strategy_state,
            self.splats["means"],
            selected["means"],
            selected["weights"],
        )


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    validate(cfg, world_size)
    cfg.init_type = "ply"
    cfg.init_ply_path = find_latest_coarse_ply(cfg.coarse_dir)

    original_parser = base.Parser
    original_create = base.create_splats_with_optimizers
    original_runner = base.Runner

    def parser_factory(**kwargs):
        return common.create_block_parser(original_parser, cfg, **kwargs)

    def create_factory(parser, **kwargs):
        return block.create_splats_with_optimizers(
            parser,
            **kwargs,
            init_ply_path=cfg.init_ply_path,
            partitions_path=cfg.partitions_path,
            block_cameras_json=cfg.block_cameras_json,
            init_block_id=cfg.init_block_id,
            init_block_bbox_enlarge=cfg.init_block_bbox_enlarge,
            init_block_bbox_margin=cfg.init_block_bbox_margin,
        )

    base.Parser = parser_factory
    base.create_splats_with_optimizers = create_factory
    base.Runner = Runner
    try:
        return base.main(local_rank, world_rank, world_size, cfg)
    finally:
        base.Parser = original_parser
        base.create_splats_with_optimizers = original_create
        base.Runner = original_runner


if __name__ == "__main__":
    config = tyro.cli(Config)
    config.adjust_steps(config.steps_scaler)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one GPU")
    cli(main, config, verbose=True)
