#!/usr/bin/env python3
"""Train one CityGS block with Texture MCMC and surface anchors."""

from __future__ import annotations

import json
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
from gsplat.distributed import cli  # noqa: E402
from gsplat.strategy import TextureMCMCStrategy  # noqa: E402


def default_strategy() -> TextureMCMCStrategy:
    return TextureMCMCStrategy(
        absgrad=True,
        texture_update_every=1,
        use_texture_score=True,
        use_grad_score=True,
        active_noise_steps=-1,
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
    # Inherit large_scale_reg=0.0; enable it only with an explicit CLI option.
    strategy: TextureMCMCStrategy = field(default_factory=default_strategy)


def validate(cfg: Config, world_size: int) -> None:
    if world_size != 1:
        raise ValueError("Texture MCMC block training supports one GPU per block")
    if cfg.data_type != "colmap":
        raise ValueError("block training currently requires --data-type colmap")
    if not cfg.depth_loss:
        raise ValueError("block training requires --depth-loss")
    if cfg.normalize_world_space or cfg.packed:
        raise ValueError("Texture MCMC block training requires unnormalized, unpacked data")
    if cfg.batch_size != 1:
        raise ValueError("Texture MCMC block training requires --batch-size 1")
    if cfg.depth_lambda < 0 or cfg.depth_refine_lambda < 0:
        raise ValueError("depth loss weights must be non-negative")
    if cfg.depth_final_ramp_steps < 1:
        raise ValueError("depth_final_ramp_steps must be at least 1")
    for label, path in (
        ("coarse_dir", cfg.coarse_dir),
        ("block_cameras_json", cfg.block_cameras_json),
    ):
        if not path or not Path(path).expanduser().exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    partitions_path = cfg.partitions_path or str(
        Path(cfg.block_cameras_json).expanduser().parent / "partitions.pt"
    )
    if not Path(partitions_path).is_file():
        raise FileNotFoundError(f"partitions_path does not exist: {partitions_path}")
    cfg.partitions_path = str(Path(partitions_path).expanduser())


def validate_block_assets(parser, cfg: Config) -> None:
    """Fail before model creation when a selected image lacks training assets."""
    with Path(cfg.block_cameras_json).open("r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or "img_name" not in row for row in rows
    ):
        raise ValueError(
            "block_cameras_json must be a list of objects containing img_name"
        )

    requested = {str(row["img_name"]) for row in rows}
    matched = set(parser.image_names)
    missing_images = sorted(requested - matched)
    if missing_images:
        preview = ", ".join(missing_images[:5])
        raise FileNotFoundError(
            f"{len(missing_images)} block images were not found in COLMAP/images: "
            f"{preview}"
        )

    if parser.test_every <= 0:
        train_indices = list(range(len(parser.image_names)))
    else:
        train_indices = [
            index
            for index in range(len(parser.image_names))
            if index % parser.test_every != 0
        ]
    if not train_indices:
        raise ValueError("the selected block has no training images")

    for label, attribute in (
        ("mask", "mask_paths"),
        ("depth map", "depth_paths"),
        ("weight map", "weight_paths"),
    ):
        paths = getattr(parser, attribute, None)
        if paths is None or len(paths) != len(parser.image_names):
            raise RuntimeError(
                f"Parser.{attribute} must align one-to-one with image_names"
            )
        missing = [paths[index] for index in train_indices if not Path(paths[index]).is_file()]
        if missing:
            preview = ", ".join(map(str, missing[:5]))
            raise FileNotFoundError(
                f"missing {len(missing)} selected {label} files; first: {preview}"
            )

    print(
        f"Block assets validated: train_images={len(train_indices)}, "
        "depth_maps=ok, weight_maps=ok, masks=ok"
    )


def create_block_parser(parser_class, cfg: Config, **kwargs):
    parser = parser_class(
        **kwargs,
        block_cameras_json=cfg.block_cameras_json,
    )
    validate_block_assets(parser, cfg)
    return parser


def find_latest_coarse_ply(coarse_dir: str) -> str:
    candidates = []
    for path in (Path(coarse_dir) / "ply").glob("point_cloud_*.ply"):
        match = re.search(r"(\d+)(?=\.ply$)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if candidates:
        return str(max(candidates, key=lambda item: item[0])[1])
    return block.find_latest_coarse_ply(coarse_dir)[0]


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    validate(cfg, world_size)
    cfg.init_type = "ply"
    cfg.init_ply_path = find_latest_coarse_ply(cfg.coarse_dir)

    original_parser = base.Parser
    original_create = base.create_splats_with_optimizers

    def parser_factory(**kwargs):
        return create_block_parser(original_parser, cfg, **kwargs)

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
    try:
        return base.main(local_rank, world_rank, world_size, cfg)
    finally:
        base.Parser = original_parser
        base.create_splats_with_optimizers = original_create


if __name__ == "__main__":
    config = tyro.cli(Config)
    config.adjust_steps(config.steps_scaler)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one GPU")
    cli(main, config, verbose=True)
