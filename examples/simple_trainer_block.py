#!/usr/bin/env python3
"""Train one CityGS block with the densification choices from texture trainer."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import tyro

REPO_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = Path(__file__).resolve().parent
TRAIN_UTIL_DIR = REPO_DIR / "utils" / "train"
for import_dir in (REPO_DIR, EXAMPLE_DIR, TRAIN_UTIL_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import simple_trainer_texture as trainer  # noqa: E402
from block_utils import (  # noqa: E402
    create_splats_with_optimizers,
    find_latest_coarse_ply,
)
from gsplat.distributed import cli  # noqa: E402
from gsplat.strategy import (  # noqa: E402
    DefaultStrategy,
    MCMCStrategy,
    TextureDefaultStrategy,
    TextureMCMCStrategy,
)


Strategy = Union[
    DefaultStrategy,
    TextureDefaultStrategy,
    MCMCStrategy,
    TextureMCMCStrategy,
]


@dataclass
class Config(trainer.Config):
    """Block-only options layered on the shared texture trainer."""

    coarse_dir: str = ""
    block_cameras_json: str = ""
    partitions_path: Optional[str] = None
    init_block_id: Optional[str] = None
    init_block_bbox_enlarge: float = 0.1
    init_block_bbox_margin: Optional[float] = None
    init_ply_path: Optional[str] = None

    test_every: int = 0
    eval_steps: list[int] = field(default_factory=list)
    normalize_world_space: bool = False
    depth_loss: bool = True
    disable_viewer: bool = True
    sh_degree: int = 0
    means_lr: float = 1.6e-6
    strategy: Strategy = field(default_factory=DefaultStrategy)


def _validate_config(cfg: Config, world_size: int) -> None:
    if world_size != 1:
        raise ValueError("block training supports exactly one GPU per process")
    if cfg.data_type != "colmap":
        raise ValueError("block training currently requires --data-type colmap")
    if not cfg.depth_loss:
        raise ValueError("block training requires --depth-loss")
    if cfg.normalize_world_space:
        raise ValueError("coarse PLY block training requires unnormalized coordinates")
    if cfg.packed:
        raise ValueError("block training requires --packed False")
    if cfg.batch_size != 1:
        raise ValueError("block training requires --batch-size 1")
    if cfg.max_steps <= 0:
        raise ValueError("max_steps must be positive")

    for name in ("coarse_dir", "block_cameras_json"):
        path = Path(getattr(cfg, name)).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    partitions = cfg.partitions_path or str(
        Path(cfg.block_cameras_json).expanduser().parent / "partitions.pt"
    )
    if not Path(partitions).is_file():
        raise FileNotFoundError(f"partitions_path does not exist: {partitions}")
    cfg.partitions_path = partitions


def _validate_block_assets(parser, cfg: Config) -> None:
    with Path(cfg.block_cameras_json).open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or "img_name" not in row for row in rows
    ):
        raise ValueError("block camera JSON must contain objects with img_name")

    requested = {str(row["img_name"]) for row in rows}
    missing_images = sorted(requested - set(parser.image_names))
    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} block images were not loaded; "
            f"first: {', '.join(missing_images[:5])}"
        )

    required_assets = [("mask", "mask_paths"), ("depth", "depth_paths")]
    if isinstance(cfg.strategy, (TextureDefaultStrategy, TextureMCMCStrategy)):
        required_assets.append(("weight map", "weight_paths"))
    for label, attribute in required_assets:
        paths = getattr(parser, attribute, None)
        if paths is None or len(paths) != len(parser.image_names):
            raise RuntimeError(f"Parser.{attribute} is not aligned with images")
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)} block {label} files; first: {missing[0]}"
            )

    print(f"Block assets validated: images={len(parser.image_names)}")


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    _validate_config(cfg, world_size)
    cfg.init_type = "ply"
    cfg.init_ply_path = find_latest_coarse_ply(cfg.coarse_dir)

    original_parser = trainer.Parser
    original_create = trainer.create_splats_with_optimizers

    def parser_factory(**kwargs):
        parser = original_parser(
            **kwargs, block_cameras_json=cfg.block_cameras_json
        )
        _validate_block_assets(parser, cfg)
        return parser

    def create_factory(parser, **kwargs):
        return create_splats_with_optimizers(
            parser,
            **kwargs,
            init_ply_path=cfg.init_ply_path,
            partitions_path=cfg.partitions_path,
            block_cameras_json=cfg.block_cameras_json,
            init_block_id=cfg.init_block_id,
            init_block_bbox_enlarge=cfg.init_block_bbox_enlarge,
            init_block_bbox_margin=cfg.init_block_bbox_margin,
        )

    trainer.Parser = parser_factory
    trainer.create_splats_with_optimizers = create_factory
    try:
        return trainer.main(local_rank, world_rank, world_size, cfg)
    finally:
        trainer.Parser = original_parser
        trainer.create_splats_with_optimizers = original_create


CONFIGS = {
    "default": (
        "Original 3DGS duplicate/split/prune densification for one block.",
        Config(strategy=DefaultStrategy(verbose=True)),
    ),
    "texture_default": (
        "Default densification guided by the texture weight map.",
        Config(
            strategy=TextureDefaultStrategy(
                verbose=True,
                surface_anchor_constraint=True,
                surface_anchor_max_distance=0.1,
            )
        ),
    ),
    "mcmc": (
        "Original MCMC add/relocate/noise densification for one block.",
        Config(
            opacity_reg=0.01,
            scale_reg=0.01,
            strategy=MCMCStrategy(verbose=True),
        ),
    ),
    "texture_mcmc": (
        "MCMC densification guided by the texture weight map.",
        Config(
            opacity_reg=0.01,
            scale_reg=0.01,
            strategy=TextureMCMCStrategy(verbose=True, use_fastgs_score=False),
        ),
    ),
}


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(CONFIGS)
    config.adjust_steps(config.steps_scaler)
    cli(main, config, verbose=True)
