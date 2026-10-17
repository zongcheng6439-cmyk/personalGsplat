"""Coarse-Ply initialization helpers shared by block trainers."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData

from gsplat.optimizers import SelectiveAdam


def find_latest_coarse_ply(coarse_dir: str) -> str:
    root = Path(coarse_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"coarse_dir does not exist: {root}")

    candidates = []
    for path in root.rglob("*.ply"):
        relative = str(path.relative_to(root))
        iteration = re.search(r"iteration[_-]?(\d+)", relative)
        numbers = re.findall(r"\d+", path.stem)
        if iteration:
            candidates.append((int(iteration.group(1)), path))
        elif numbers and path.parent.name == "ply":
            candidates.append((int(numbers[-1]), path))

    if not candidates:
        raise FileNotFoundError(
            f"no coarse PLY found under {root}; expected point_cloud/"
            "iteration_<step>/point_cloud.ply or ply/*_<step>.ply"
        )
    step, path = max(candidates, key=lambda item: (item[0], str(item[1])))
    print(f"Using coarse PLY: step={step}, path={path}")
    return str(path)


def _numbered_properties(vertex, prefix: str) -> List[str]:
    names = [prop.name for prop in vertex.properties if prop.name.startswith(prefix)]
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[-1]))


def _load_trained_splats(ply_path: str) -> Dict[str, torch.Tensor]:
    vertex = PlyData.read(ply_path)["vertex"]
    names = {prop.name for prop in vertex.properties}
    required = {
        "x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    }
    missing = sorted(required - names)
    if missing:
        raise ValueError(f"coarse PLY is missing trained GS fields: {missing}")

    def stack(properties: List[str]) -> torch.Tensor:
        values = np.stack([np.asarray(vertex[name]) for name in properties], axis=-1)
        return torch.from_numpy(values.astype(np.float32, copy=False))

    means = stack(["x", "y", "z"])
    rest_names = _numbered_properties(vertex, "f_rest_")
    if len(rest_names) % 3 != 0:
        raise ValueError("the number of f_rest fields must be divisible by three")
    shN = (
        stack(rest_names).reshape(len(means), 3, -1).transpose(1, 2).contiguous()
        if rest_names
        else torch.empty((len(means), 0, 3), dtype=torch.float32)
    )
    return {
        "means": means,
        "scales": stack(_numbered_properties(vertex, "scale_")),
        "quats": stack(_numbered_properties(vertex, "rot_")),
        "opacities": torch.from_numpy(
            np.asarray(vertex["opacity"]).astype(np.float32, copy=False)
        ),
        "sh0": stack(["f_dc_0", "f_dc_1", "f_dc_2"]).unsqueeze(1),
        "shN": shN,
    }


def _parse_block_id(path: str, explicit: Optional[str]) -> Tuple[int, int]:
    text = explicit
    if text is None:
        match = re.search(r"cameras-(\d+)_(\d+)$", Path(path).stem)
        if match is None:
            raise ValueError(
                "cannot infer block id; set --init-block-id row_col explicitly"
            )
        return int(match.group(1)), int(match.group(2))
    match = re.fullmatch(r"(\d+)[_,](\d+)", text)
    if match is None:
        raise ValueError(f"invalid init_block_id {text!r}; expected row_col")
    return int(match.group(1)), int(match.group(2))


def find_preselected_block_ply(
    partitions_path: str, block_id: Tuple[int, int]
) -> Optional[Path]:
    """Find the visibility-expanded initialization saved by partition.py."""
    path = Path(partitions_path).expanduser()
    root = path if path.is_dir() else path.parent
    candidate = (
        root
        / "block_init"
        / f"point_cloud_{block_id[0]:03d}_{block_id[1]:03d}.ply"
    )
    return candidate if candidate.is_file() else None


def _contract_xy(xy: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    aabb = torch.as_tensor(aabb, dtype=xy.dtype)
    xy = (xy - aabb[:2]) / (aabb[2:] - aabb[:2]) * 2.0 - 1.0
    magnitude = torch.linalg.vector_norm(
        xy, ord=float("inf"), dim=-1, keepdim=True
    )
    outside = magnitude.squeeze(-1) > 1.0
    xy[outside] = (2.0 - 1.0 / magnitude[outside]) * (
        xy[outside] / magnitude[outside]
    )
    return xy / 4.0 + 0.5


def _filter_splats_for_block(
    splats: Dict[str, torch.Tensor],
    partitions_path: str,
    block_id: Tuple[int, int],
    bbox_enlarge: float,
    bbox_margin: Optional[float],
) -> Dict[str, torch.Tensor]:
    if bbox_enlarge < 0:
        raise ValueError("init_block_bbox_enlarge must be non-negative")
    try:
        partitions = torch.load(
            partitions_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        partitions = torch.load(partitions_path, map_location="cpu")

    coordinates = partitions["partition_coordinates"]
    ids = torch.as_tensor(coordinates["id"], dtype=torch.long)
    target = torch.tensor(block_id, dtype=torch.long)
    matches = torch.where(torch.all(ids == target, dim=-1))[0]
    if len(matches) != 1:
        raise ValueError(
            f"block {block_id[0]}_{block_id[1]} occurs {len(matches)} times"
        )
    block_index = int(matches.item())
    block_xy = torch.as_tensor(coordinates["xy"], dtype=torch.float32)[block_index]
    partition_size = torch.as_tensor(
        partitions["scene_config"]["partition_size"], dtype=torch.float32
    )
    if partition_size.ndim == 0:
        block_size = partition_size.repeat(2)
    elif partition_size.ndim == 1:
        block_size = partition_size
    else:
        block_size = partition_size[block_index]

    if bbox_margin is not None:
        if bbox_margin < 0:
            raise ValueError("init_block_bbox_margin must be non-negative")
        bbox_min = block_xy - bbox_margin
        bbox_max = block_xy + block_size + bbox_margin
    else:
        bbox_min = block_xy - bbox_enlarge * block_size
        bbox_max = block_xy + (1.0 + bbox_enlarge) * block_size

    means = splats["means"]
    rotation = torch.as_tensor(
        partitions.get("extra_data", {}).get("rotation_transform", torch.eye(4)),
        dtype=means.dtype,
    )[:3, :3]
    xy = (means @ rotation.T)[:, :2]
    scene_config = partitions["scene_config"]
    if scene_config.get("contract", False):
        xy = _contract_xy(xy, scene_config["aabb"])
    mask = torch.all((xy >= bbox_min) & (xy <= bbox_max), dim=-1)
    if not mask.any():
        raise ValueError(f"no coarse GS lies inside block {block_id[0]}_{block_id[1]}")
    print(
        f"Block GS selection: {block_id[0]}_{block_id[1]}, "
        f"selected={int(mask.sum())}/{len(mask)}"
    )
    return {name: value[mask] for name, value in splats.items()}


def create_splats_with_optimizers(
    parser,
    *,
    init_type: str,
    means_lr: float,
    scales_lr: float,
    opacities_lr: float,
    quats_lr: float,
    sh0_lr: float,
    shN_lr: float,
    scene_scale: float,
    sh_degree: int,
    sparse_grad: bool,
    visible_adam: bool,
    batch_size: int,
    feature_dim: Optional[int],
    device: str,
    world_rank: int,
    world_size: int,
    init_ply_path: str,
    partitions_path: str,
    block_cameras_json: str,
    init_block_id: Optional[str],
    init_block_bbox_enlarge: float,
    init_block_bbox_margin: Optional[float],
    **unused,
):
    del parser, unused
    if init_type != "ply":
        raise ValueError("block training requires init_type=ply")
    if feature_dim is not None:
        raise ValueError("coarse PLY initialization currently requires app_opt=False")

    block_id = _parse_block_id(block_cameras_json, init_block_id)
    block_ply = find_preselected_block_ply(partitions_path, block_id)
    if block_ply is not None:
        splats = _load_trained_splats(str(block_ply))
        print(
            f"Using visibility-expanded block PLY: {block_ply}, "
            f"selected={len(splats['means']):,}"
        )
    else:
        splats = _filter_splats_for_block(
            _load_trained_splats(init_ply_path),
            partitions_path,
            block_id,
            init_block_bbox_enlarge,
            init_block_bbox_margin,
        )
    loaded_sh = 1 + splats["shN"].shape[1]
    required_sh = (sh_degree + 1) ** 2
    if loaded_sh < required_sh:
        raise ValueError(
            f"coarse PLY has {loaded_sh} SH coefficients, but degree "
            f"{sh_degree} requires {required_sh}"
        )
    splats = {
        name: value[world_rank::world_size] for name, value in splats.items()
    }
    params = [
        ("means", splats["means"], means_lr * scene_scale),
        ("scales", splats["scales"], scales_lr),
        ("quats", splats["quats"], quats_lr),
        ("opacities", splats["opacities"], opacities_lr),
        ("sh0", splats["sh0"], sh0_lr),
        ("shN", splats["shN"], shN_lr),
    ]
    parameters = torch.nn.ParameterDict(
        {name: torch.nn.Parameter(value) for name, value, _ in params}
    ).to(device)
    optimizer_class = (
        torch.optim.SparseAdam
        if sparse_grad
        else SelectiveAdam if visible_adam else torch.optim.Adam
    )
    batch_scale = batch_size * world_size
    optimizers = {
        name: optimizer_class(
            [{"params": parameters[name], "lr": lr * math.sqrt(batch_scale), "name": name}],
            eps=1e-15 / math.sqrt(batch_scale),
            betas=(0.9**batch_scale, 0.999**batch_scale),
            fused=True,
        )
        for name, _, lr in params
    }
    return parameters, optimizers
