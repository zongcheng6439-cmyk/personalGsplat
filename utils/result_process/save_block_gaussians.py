"""Save the Gaussians whose centers lie inside one core partition block."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _load_partitions(path: Path) -> dict[str, Any]:
    path = path / "partitions.pt" if path.is_dir() else path
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _contract_xy(xy: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    aabb_min, aabb_max = torch.chunk(aabb, 2)
    xy = (xy - aabb_min) / (aabb_max - aabb_min) * 2.0 - 1.0
    magnitude = torch.linalg.vector_norm(xy, ord=torch.inf, dim=-1, keepdim=True)
    outside = magnitude[:, 0] > 1.0
    xy[outside] = (2.0 - 1.0 / magnitude[outside]) * (
        xy[outside] / magnitude[outside]
    )
    return xy / 4.0 + 0.5


def block_mask(
    xyz: np.ndarray,
    partitions: dict[str, Any],
    block_id: tuple[int, int],
) -> np.ndarray:
    """Return the core-block mask using the same coordinates as partitioning."""
    coordinates = partitions["partition_coordinates"]
    ids = torch.as_tensor(coordinates["id"], dtype=torch.long)
    target = torch.tensor(block_id, dtype=torch.long)
    matches = torch.where(torch.all(ids == target, dim=1))[0]
    if len(matches) != 1:
        raise ValueError(f"Block {block_id} occurs {len(matches)} times in partitions.pt")
    index = int(matches.item())

    points = torch.as_tensor(xyz, dtype=torch.float32)
    rotation = torch.as_tensor(
        partitions.get("extra_data", {}).get("rotation_transform", torch.eye(4)),
        dtype=points.dtype,
    )[:3, :3].T
    xy = (points @ rotation)[:, :2]

    scene = partitions["scene_config"]
    if scene.get("contract", False):
        xy = _contract_xy(xy, torch.as_tensor(scene["aabb"], dtype=xy.dtype))

    origin = torch.as_tensor(coordinates["xy"], dtype=xy.dtype)[index]
    size = torch.as_tensor(scene["partition_size"], dtype=xy.dtype)
    if size.ndim > 1:
        size = size[index]
    if size.ndim == 0:
        size = size.repeat(2)
    return torch.all((xy >= origin) & (xy <= origin + size), dim=1).numpy()


def save_block_gaussians(
    input_ply: str | Path,
    output_ply: str | Path,
    partitions_path: str | Path,
    block_id: tuple[int, int],
) -> int:
    """Filter a Gaussian PLY to one core block and preserve every vertex field."""
    input_ply = Path(input_ply)
    output_ply = Path(output_ply)
    ply = PlyData.read(str(input_ply))
    vertex = ply["vertex"]
    xyz = np.stack([vertex[name] for name in ("x", "y", "z")], axis=1)
    mask = block_mask(xyz, _load_partitions(Path(partitions_path)), block_id)
    selected = vertex.data[mask].copy()

    output_ply.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_ply.with_name(output_ply.name + ".part")
    result = PlyData(
        [PlyElement.describe(selected, "vertex")],
        text=ply.text,
        byte_order=ply.byte_order,
        comments=list(ply.comments),
        obj_info=list(ply.obj_info),
    )
    result.write(str(temporary))
    os.replace(temporary, output_ply)
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, default="True")
    parser.add_argument("--output-ply", type=Path, default='True')
    parser.add_argument(
        "--partitions",
        type=Path,
        default='True',
        help="partitions.pt or the directory containing it",
    )
    parser.add_argument("--block-id", type=int, nargs=2, required=True, metavar=("X", "Y"))
    args = parser.parse_args()
    count = save_block_gaussians(
        args.input_ply,
        args.output_ply,
        args.partitions,
        tuple(args.block_id),
    )
    print(f"Saved {count:,} Gaussians for block {tuple(args.block_id)} to {args.output_ply}")


if __name__ == "__main__":
    main()
