#!/usr/bin/env python3
"""Train one CityGS block with voxel-plane Texture MCMC."""

from __future__ import annotations

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
import train_voxel_plane_mcmc as plane  # noqa: E402
import block_utils as block  # noqa: E402
import train_texture_anchor_block as common  # noqa: E402
from gsplat.distributed import cli  # noqa: E402
from gsplat.strategy.voxel_plane_mcmc import read_voxel_plane_ply  # noqa: E402


@dataclass
class Config(plane.Config):
    test_every: int = 0
    eval_steps: list[int] = field(default_factory=list)
    coarse_dir: str = ""
    block_cameras_json: str = ""
    partitions_path: Optional[str] = None
    init_block_id: Optional[str] = None
    init_block_bbox_enlarge: float = 0.0
    init_block_bbox_margin: Optional[float] = None
    init_ply_path: Optional[str] = None

    depth_loss: bool = True
    init_opa: float = 0.5
    init_scale: float = 0.1
    opacity_reg: float = 0.01
    scale_reg: float = 0.01


class Runner(plane.Runner):
    def bind_strategy_state(self) -> None:
        cfg = self.cfg
        xyz, match_xyz, normals, is_plane = read_voxel_plane_ply(
            cfg.strategy.plane_anchor_ply
        )
        partitions_path = cfg.partitions_path or str(
            Path(cfg.block_cameras_json).parent / "partitions.pt"
        )
        block_id = block._parse_block_id(
            cfg.block_cameras_json, cfg.init_block_id
        )
        all_anchors = {
            "means": torch.as_tensor(match_xyz, dtype=torch.float32),
            "xyz": torch.as_tensor(xyz, dtype=torch.float32),
            "normals": torch.as_tensor(normals, dtype=torch.float32),
            "is_plane": torch.as_tensor(is_plane),
        }
        if block.find_preselected_block_ply(partitions_path, block_id):
            selected = all_anchors
        else:
            selected = block._filter_splats_for_block(
                all_anchors,
                partitions_path,
                block_id,
                cfg.init_block_bbox_enlarge,
                cfg.init_block_bbox_margin,
            )
        cfg.strategy.set_plane_anchors(
            self.strategy_state,
            self.splats["means"],
            selected["xyz"].numpy(),
            selected["means"].numpy(),
            selected["normals"].numpy(),
            selected["is_plane"].numpy(),
        )
        self._geometry_loss_values = {}
        cfg.strategy.initialize_plane_gaussians(
            self.splats, self.strategy_state
        )
        self.optimizers["quats"].register_step_pre_hook(
            self._add_anchor_normal_gradient
        )
        self.optimizers["means"].register_step_pre_hook(
            self._add_knn_plane_gradients
        )
        self.optimizers["scales"].register_step_pre_hook(
            self._add_anchor_scale_gradient
        )


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    common.validate(cfg, world_size)
    cfg.init_type = "ply"
    cfg.init_ply_path = common.find_latest_coarse_ply(cfg.coarse_dir)

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
