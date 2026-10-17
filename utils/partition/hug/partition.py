#!/usr/bin/env python3
"""HUG-style partitioning directly from the input point cloud.

The script divides the point cloud into quantile-balanced XY blocks and assigns
images by counting the block points projected inside each camera.  It does not
load a coarse checkpoint, render Gaussians, or generate per-image masks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from tqdm import tqdm


PARTITION_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_DIR / "examples"
for path in (EXAMPLE_DIR, PARTITION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from citygs_gsplat import CityGSSceneConfig, CityGSPartitionableScene  # noqa: E402
from partitioning_utils import MinMaxBoundingBox  # noqa: E402
from datasets.colmap_adc import Parser  # noqa: E402
from datasets.colmap_custom_utils import fetchPly  # noqa: E402


def assign_point_blocks(points_xy: torch.Tensor, scene) -> torch.Tensor:
    boxes = scene.partition_coordinates.get_bounding_boxes(
        scene.scene_config.partition_size
    )
    block_ids = torch.full(
        (len(points_xy),), -1, dtype=torch.long, device=points_xy.device
    )
    for block_id in range(len(scene.partition_coordinates)):
        inside = torch.all(
            (points_xy >= boxes.min[block_id])
            & (points_xy <= boxes.max[block_id]),
            dim=-1,
        )
        block_ids[inside & (block_ids < 0)] = block_id
    if torch.any(block_ids < 0):
        raise RuntimeError(
            f"{int((block_ids < 0).sum())} projected points are outside all blocks"
        )
    return block_ids


@torch.inference_mode()
def projection_counts(
    parser: Parser,
    points: torch.Tensor,
    point_block_ids: torch.Tensor,
    block_count: int,
    chunk_size: int,
) -> torch.Tensor:
    """Count in-image projected points for every block-camera pair."""
    counts = torch.zeros(
        (block_count, len(parser.image_names)), dtype=torch.long, device="cpu"
    )
    for camera_index in tqdm(range(len(parser.image_names)), desc="Project cameras"):
        camera_id = parser.camera_ids[camera_index]
        width, height = parser.imsize_dict[camera_id]
        K = torch.as_tensor(
            parser.Ks_dict[camera_id], dtype=points.dtype, device=points.device
        )
        c2w = torch.as_tensor(
            parser.camtoworlds[camera_index],
            dtype=points.dtype,
            device=points.device,
        )
        w2c = torch.linalg.inv(c2w)
        camera_counts = torch.zeros(block_count, dtype=torch.long, device=points.device)

        for start in range(0, len(points), chunk_size):
            stop = min(start + chunk_size, len(points))
            camera_points = points[start:stop] @ w2c[:3, :3].T + w2c[:3, 3]
            projected = camera_points @ K.T
            depth = camera_points[:, 2]
            uv = projected[:, :2] / depth[:, None].clamp_min(1e-8)
            visible = (
                (depth > 0)
                & torch.isfinite(uv).all(dim=-1)
                & (uv[:, 0] >= 0)
                & (uv[:, 0] < width)
                & (uv[:, 1] >= 0)
                & (uv[:, 1] < height)
            )
            camera_counts += torch.bincount(
                point_block_ids[start:stop][visible], minlength=block_count
            )
        counts[:, camera_index] = camera_counts.cpu()
    return counts


def select_cameras(
    counts: torch.Tensor,
    block_point_counts: torch.Tensor,
    min_visible_points: int,
    min_visible_ratio: float,
    min_cameras: int,
) -> torch.Tensor:
    required = torch.maximum(
        torch.full_like(block_point_counts, min_visible_points),
        torch.ceil(block_point_counts * min_visible_ratio).long(),
    )
    selected = counts >= required[:, None]
    for block_id in range(len(selected)):
        positive = torch.nonzero(counts[block_id] > 0, as_tuple=False).flatten()
        missing = min(min_cameras, len(positive)) - int(selected[block_id].sum())
        if missing > 0:
            candidates = positive[~selected[block_id, positive]]
            top = candidates[torch.topk(counts[block_id, candidates], missing).indices]
            selected[block_id, top] = True
    return selected


def write_camera_files(
    output_dir: Path,
    scene,
    parser: Parser,
    selected: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    for block_index in range(len(scene.partition_coordinates)):
        name = scene.partition_coordinates.get_str_id(block_index)
        camera_indices = torch.nonzero(
            selected[block_index], as_tuple=False
        ).flatten().tolist()
        rows = []
        for camera_index in camera_indices:
            c2w = parser.camtoworlds[camera_index]
            rows.append(
                {
                    "id": camera_index,
                    "img_name": parser.image_names[camera_index],
                    "position": c2w[:3, 3].tolist(),
                    "rotation": c2w[:3, :3].tolist(),
                    "visible_points": int(counts[block_index, camera_index]),
                }
            )
        (output_dir / f"{name}.txt").write_text(
            "".join(f"{row['img_name']}\n" for row in rows), encoding="utf-8"
        )
        (output_dir / f"cameras-{name}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def _block_boxes(scene) -> tuple[np.ndarray, np.ndarray]:
    boxes = scene.partition_coordinates.get_bounding_boxes(
        scene.scene_config.partition_size
    )
    return boxes.min.detach().cpu().numpy(), boxes.max.detach().cpu().numpy()


def _set_equal_limits(ax, points_xy: np.ndarray, camera_xy: np.ndarray) -> None:
    coordinates = np.concatenate([points_xy, camera_xy], axis=0)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    margin = max(float((upper - lower).max()) * 0.03, 1e-3)
    ax.set_xlim(lower[0] - margin, upper[0] + margin)
    ax.set_ylim(lower[1] - margin, upper[1] + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world X")
    ax.set_ylabel("world Y")


def save_visualizations(
    output_dir: Path,
    scene,
    parser: Parser,
    points: torch.Tensor,
    point_block_ids: torch.Tensor,
    selected: torch.Tensor,
    max_plot_points: int,
) -> None:
    points_xy = points[:, :2].detach().cpu().numpy()
    block_ids = point_block_ids.detach().cpu().numpy()
    if len(points_xy) > max_plot_points:
        indices = np.linspace(0, len(points_xy) - 1, max_plot_points).astype(int)
        points_xy = points_xy[indices]
        block_ids = block_ids[indices]

    c2w = np.asarray(parser.camtoworlds)
    camera_xy = c2w[:, :2, 3]
    box_min, box_max = _block_boxes(scene)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(points_xy[:, 0], points_xy[:, 1], s=0.15, c="0.65", label="points")
    ax.scatter(
        camera_xy[:, 0], camera_xy[:, 1], s=2, c="#377eb8", alpha=0.45,
        label="cameras",
    )
    colors = plt.get_cmap("tab20", len(box_min))
    for block_index, (lower, upper) in enumerate(zip(box_min, box_max)):
        size = upper - lower
        ax.add_patch(
            patches.Rectangle(
                lower, size[0], size[1], fill=False,
                linewidth=2, color=colors(block_index),
            )
        )
        name = scene.partition_coordinates.get_str_id(block_index)
        ax.text(
            lower[0] + size[0] * 0.03,
            upper[1] - size[1] * 0.08,
            f"{name}\n{int(selected[block_index].sum())} cameras",
            color=colors(block_index),
            fontsize=9,
            weight="bold",
        )
    _set_equal_limits(ax, points_xy, camera_xy)
    ax.legend(markerscale=4, loc="best")
    ax.set_title("HUG point-cloud partitions")
    fig.tight_layout()
    fig.savefig(output_dir / "partitions.png", dpi=200)
    plt.close(fig)

    for block_index in range(len(box_min)):
        name = scene.partition_coordinates.get_str_id(block_index)
        camera_mask = selected[block_index].cpu().numpy()
        point_mask = block_ids == block_index
        camera_track = camera_xy.copy()
        step_distance = np.linalg.norm(np.diff(camera_xy, axis=0), axis=1)
        nonzero_steps = step_distance[step_distance > 0]
        if len(nonzero_steps):
            scene_extent = np.ptp(camera_xy, axis=0).max()
            jump_threshold = max(
                float(np.median(nonzero_steps)) * 10.0,
                float(scene_extent) * 0.02,
            )
            jump_indices = np.flatnonzero(step_distance > jump_threshold) + 1
            camera_track[jump_indices] = np.nan
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(points_xy[:, 0], points_xy[:, 1], s=0.1, c="0.85")
        ax.scatter(
            points_xy[point_mask, 0], points_xy[point_mask, 1],
            s=0.3, c="#ff7f00", label="block points",
        )
        ax.plot(
            camera_track[:, 0], camera_track[:, 1],
            color="#377eb8", linewidth=1.0, alpha=0.8,
            label="full camera trajectory",
        )
        ax.scatter(
            camera_xy[:, 0], camera_xy[:, 1],
            s=2, c="#377eb8", alpha=0.35,
        )
        ax.scatter(
            camera_xy[camera_mask, 0], camera_xy[camera_mask, 1],
            s=6, c="#e41a1c", alpha=0.9, label="selected images",
        )
        lower, upper = box_min[block_index], box_max[block_index]
        size = upper - lower
        ax.add_patch(
            patches.Rectangle(
                lower, size[0], size[1], fill=False,
                linewidth=3, color="#4daf4a", label="block boundary",
            )
        )
        _set_equal_limits(ax, points_xy, camera_xy)
        ax.legend(markerscale=4, loc="best")
        ax.set_title(
            f"HUG block {name}: {int(point_mask.sum()):,} plotted points, "
            f"{int(camera_mask.sum()):,} selected cameras"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=200)
        plt.close(fig)


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("HUG partition currently requires CUDA")
    if len(args.block_dim) != 2 or min(args.block_dim) <= 0:
        raise ValueError("--block_dim requires two positive integers")

    parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=False,
        test_every=0,
        load_exposure=False,
    )
    extracted = Path(args.data_dir) / "sparse/0/extracted.ply"
    if not extracted.is_file():
        extracted = Path(args.data_dir) / "sparse/extracted.ply"
    point_source = args.point_ply or (
        str(extracted) if extracted.is_file() else "COLMAP points3D"
    )
    if args.point_ply:
        point_xyz, _, _ = fetchPly(args.point_ply)
    else:
        point_xyz = parser.points
    if len(point_xyz) == 0:
        raise ValueError(f"Point cloud is empty: {point_source}")

    device = torch.device("cuda")
    all_points = torch.as_tensor(point_xyz, dtype=torch.float32, device=device)
    camera_centers = torch.as_tensor(
        parser.camtoworlds[:, :3, 3], dtype=torch.float32, device=device
    )
    scene_config = CityGSSceneConfig(
        origin=torch.zeros(2, device=device),
        block_dim=args.block_dim,
        content_threshold=args.min_visible_ratio,
    )
    scene = CityGSPartitionableScene(
        scene_config,
        camera_centers[:, :2],
        reoriented_points=all_points[:, :2],
    )
    scene.point_based_bounding_box = MinMaxBoundingBox(
        min=all_points[:, :2].amin(dim=0),
        max=all_points[:, :2].amax(dim=0),
    )
    scene_config.origin = (
        scene.point_based_bounding_box.min + scene.point_based_bounding_box.max
    ) / 2
    scene.get_scene_bounding_box()
    scene.build_partition_coordinates()

    if args.max_project_points > 0 and len(all_points) > args.max_project_points:
        indices = torch.linspace(
            0,
            len(all_points) - 1,
            args.max_project_points,
            device=device,
        ).long()
        points = all_points[indices]
    else:
        points = all_points
    point_block_ids = assign_point_blocks(points[:, :2], scene)
    block_point_counts = torch.bincount(
        point_block_ids, minlength=len(scene.partition_coordinates)
    ).cpu()
    counts = projection_counts(
        parser,
        points,
        point_block_ids,
        len(scene.partition_coordinates),
        args.chunk_size,
    )
    selected = select_cameras(
        counts,
        block_point_counts,
        args.min_visible_points,
        args.min_visible_ratio,
        args.min_cameras,
    )

    output_dir = Path(args.output_dir or (
        Path(args.data_dir)
        / "partition"
        / f"partitions-hug-dim_{args.block_dim[0]}_{args.block_dim[1]}"
    ))
    output_dir.mkdir(parents=True, exist_ok=True)
    scene.camera_visibilities = counts
    scene.is_camera_in_partition = torch.zeros_like(selected)
    scene.is_partitions_visible_to_cameras = selected
    scene.save(
        str(output_dir),
        extra_data={
            "mode": "hug",
            "point_cloud": point_source,
            "point_counts": block_point_counts,
            "min_visible_points": args.min_visible_points,
            "min_visible_ratio": args.min_visible_ratio,
            "rotation_transform": torch.eye(4),
        },
    )
    write_camera_files(output_dir, scene, parser, selected, counts)
    save_visualizations(
        output_dir,
        scene,
        parser,
        points,
        point_block_ids,
        selected,
        args.max_plot_points,
    )

    print(f"HUG partition written to: {output_dir}")
    for block_index in range(len(scene.partition_coordinates)):
        name = scene.partition_coordinates.get_str_id(block_index)
        print(
            f"  block={name} points={int(block_point_counts[block_index]):,} "
            f"cameras={int(selected[block_index].sum())}"
        )


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--data_dir", required=True)
    argument_parser.add_argument("--data_factor", type=int, default=1)
    argument_parser.add_argument("--block_dim", type=int, nargs=2, default=[2, 2])
    argument_parser.add_argument(
        "--point_ply",
        help="Optional LiDAR PLY; defaults to Parser points (extracted.ply when present)",
    )
    argument_parser.add_argument("--output_dir")
    argument_parser.add_argument("--min_visible_points", type=int, default=64)
    argument_parser.add_argument("--min_visible_ratio", type=float, default=0.001)
    argument_parser.add_argument("--min_cameras", type=int, default=8)
    argument_parser.add_argument("--max_project_points", type=int, default=2_000_000)
    argument_parser.add_argument("--max_plot_points", type=int, default=100_000)
    argument_parser.add_argument("--chunk_size", type=int, default=500_000)
    main(argument_parser.parse_args())
