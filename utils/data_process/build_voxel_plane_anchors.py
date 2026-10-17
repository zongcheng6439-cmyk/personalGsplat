"""Build plane-aware voxel anchors from the LiDAR initialization PLY.

Normals are estimated on the original points. A point is planar only when its
local PCA is planar and neighboring normals agree, following GeoGaussian's
normal-consistency filtering idea. Reliable point normals are then aggregated
into occupied voxel anchors. An existing voxel PLY may be supplied to preserve
its exact positions and order.

Parking01 example (the layer-aware balanced parameters below are defaults)::

    python utils/data_process/build_voxel_plane_anchors.py \
        --point-ply /lutu/parking01/sparse/0/extracted.ply \
        --output /lutu/parking01/voxel_plane_anchors.ply \
        --voxel-size 0.1 --overwrite

To preserve an existing voxel layout, additionally pass::

        --anchor-ply /data/voxel_target_count.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree


def read_xyz(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    names = vertex.data.dtype.names or ()
    if not all(name in names for name in ("x", "y", "z")):
        raise ValueError(f"PLY has no XYZ properties: {path}")
    points = np.stack([np.asarray(vertex[name]) for name in ("x", "y", "z")], 1)
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0 or not np.isfinite(points).all():
        raise ValueError(f"PLY contains empty or invalid XYZ data: {path}")
    return points


def voxel_representatives(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Choose the point nearest each occupied voxel centroid as its anchor."""
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, count = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    centroid = np.zeros((len(count), 3), dtype=np.float64)
    np.add.at(centroid, inverse, points)
    centroid /= count[:, None]
    distance2 = ((points - centroid[inverse]) ** 2).sum(1)
    order = np.lexsort((distance2, inverse))
    first = np.r_[True, inverse[order][1:] != inverse[order][:-1]]
    return points[order[first]]


def estimate_point_planes(
    points: np.ndarray,
    normal_k: int,
    consistency_k: int,
    max_angle_deg: float,
    planarity_threshold: float,
    curvature_threshold: float,
    chunk_size: int,
):
    """Estimate PCA normals, then reject noisy/inconsistent point normals."""
    tree = cKDTree(np.asarray(points, dtype=np.float64))
    normals = np.empty_like(points)
    planarity = np.empty(len(points), dtype=np.float32)
    curvature = np.empty(len(points), dtype=np.float32)

    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        ids = tree.query(points[start:end], k=normal_k, workers=-1)[1]
        neighbors = points[ids].astype(np.float64, copy=False)
        centered = neighbors - neighbors.mean(1, keepdims=True)
        covariance = np.einsum("nki,nkj->nij", centered, centered) / (normal_k - 1)
        values, vectors = np.linalg.eigh(covariance)
        scale = np.maximum(values[:, 2], 1e-12)
        planarity[start:end] = (values[:, 1] - values[:, 0]) / scale
        curvature[start:end] = values[:, 0] / np.maximum(values.sum(1), 1e-12)
        normal = vectors[:, :, 0]
        dominant = np.abs(normal).argmax(1)
        normal[normal[np.arange(len(normal)), dominant] < 0] *= -1
        normals[start:end] = normal.astype(np.float32)

    consistent = np.empty(len(points), dtype=bool)
    cosine_threshold = np.cos(np.deg2rad(max_angle_deg))
    for start in range(0, len(points), chunk_size):
        end = min(start + chunk_size, len(points))
        ids = tree.query(points[start:end], k=consistency_k, workers=-1)[1]
        cosine = np.abs(
            np.einsum("ni,nki->nk", normals[start:end], normals[ids])
        )
        consistent[start:end] = (cosine >= cosine_threshold).all(1)

    is_plane = (
        consistent
        & (planarity >= planarity_threshold)
        & (curvature <= curvature_threshold)
    )
    return normals, is_plane


def aggregate_to_anchors(
    points: np.ndarray,
    point_normals: np.ndarray,
    point_is_plane: np.ndarray,
    anchors: np.ndarray,
    min_plane_ratio: float,
):
    anchor_ids = cKDTree(anchors).query(points, k=1, workers=-1)[1]
    count = np.bincount(anchor_ids, minlength=len(anchors))
    plane_count = np.bincount(
        anchor_ids, weights=point_is_plane, minlength=len(anchors)
    )
    normal_sum = np.zeros((len(anchors), 3), dtype=np.float64)
    np.add.at(normal_sum, anchor_ids[point_is_plane], point_normals[point_is_plane])
    normal_norm = np.linalg.norm(normal_sum, axis=1)
    plane_ratio = plane_count / np.maximum(count, 1)
    coherence = normal_norm / np.maximum(plane_count, 1)
    is_plane = (
        (count > 0)
        & (plane_ratio >= min_plane_ratio)
        & (coherence >= np.cos(np.deg2rad(10.0)))
    )
    normals = np.zeros((len(anchors), 3), dtype=np.float32)
    normals[is_plane] = (
        normal_sum[is_plane] / normal_norm[is_plane, None]
    ).astype(np.float32)
    return normals, is_plane, plane_ratio.astype(np.float32)


def smooth_anchor_planes(
    anchors: np.ndarray,
    normals: np.ndarray,
    is_plane: np.ndarray,
    voxel_size: float,
    neighbor_k: int,
    radius_ratio: float,
    angle_deg: float,
    plane_distance_ratio: float,
    iterations: int,
    chunk_size: int,
):
    """Locally align coplanar anchors without moving them in-plane."""
    if iterations == 0 or not is_plane.any():
        return anchors, normals
    source = anchors.astype(np.float64)
    positions = source.copy()
    directions = normals.astype(np.float64)
    tree = cKDTree(source)
    k = min(neighbor_k, len(source))
    radius = radius_ratio * voxel_size
    plane_distance = plane_distance_ratio * voxel_size
    cosine_threshold = np.cos(np.deg2rad(angle_deg))

    for iteration in range(iterations):
        new_positions = positions.copy()
        new_directions = directions.copy()
        for start in range(0, len(source), chunk_size):
            end = min(start + chunk_size, len(source))
            distances, ids = tree.query(
                source[start:end],
                k=k,
                distance_upper_bound=radius,
                workers=-1,
            )
            if k == 1:
                distances, ids = distances[:, None], ids[:, None]
            valid_distance = np.isfinite(distances)
            safe_ids = np.where(valid_distance, ids, 0)
            center_position = positions[start:end]
            center_normal = directions[start:end]
            neighbor_position = positions[safe_ids]
            neighbor_normal = directions[safe_ids]
            delta = neighbor_position - center_position[:, None]
            cosine = np.einsum("ni,nki->nk", center_normal, neighbor_normal)
            aligned_normal = neighbor_normal * np.where(
                cosine[..., None] < 0, -1.0, 1.0
            )
            same_plane = (
                valid_distance
                & is_plane[start:end, None]
                & is_plane[safe_ids]
                & (np.abs(cosine) >= cosine_threshold)
                & (
                    np.abs(np.einsum("nki,ni->nk", delta, center_normal))
                    <= plane_distance
                )
                & (
                    np.abs(np.einsum("nki,nki->nk", delta, neighbor_normal))
                    <= plane_distance
                )
            )
            weight = np.exp(-((distances / radius) ** 2)) * same_plane
            weight_sum = weight.sum(1, keepdims=True)
            valid_center = is_plane[start:end] & (weight_sum[:, 0] > 0)
            smoothed_normal = np.einsum("nk,nki->ni", weight, aligned_normal)
            smoothed_normal /= np.linalg.norm(
                smoothed_normal, axis=1, keepdims=True
            ).clip(1e-12)
            normal_offset = (
                np.einsum("nki,ni->nk", delta, smoothed_normal) * weight
            ).sum(1) / weight_sum[:, 0].clip(1e-12)
            chunk_directions = new_directions[start:end]
            chunk_positions = new_positions[start:end]
            chunk_directions[valid_center] = smoothed_normal[valid_center]
            chunk_positions[valid_center] += (
                normal_offset[valid_center, None]
                * smoothed_normal[valid_center]
            )
        movement = np.linalg.norm(new_positions - positions, axis=1)
        positions, directions = new_positions, new_directions
        print(
            f"Plane smoothing {iteration + 1}/{iterations}: "
            f"mean movement={movement[is_plane].mean():.6f} m, "
            f"max={movement[is_plane].max():.6f} m",
            flush=True,
        )
    return positions.astype(np.float32), directions.astype(np.float32)


def write_anchors(
    path: Path,
    anchors: np.ndarray,
    source_anchors: np.ndarray,
    normals: np.ndarray,
    is_plane: np.ndarray,
    plane_ratio: np.ndarray,
    voxel_size: float,
) -> None:
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("source_x", "f4"), ("source_y", "f4"), ("source_z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("is_plane", "u1"), ("plane_ratio", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    vertex = np.empty(len(anchors), dtype=dtype)
    for name, values in zip(("x", "y", "z"), anchors.T):
        vertex[name] = values
    for name, values in zip(("source_x", "source_y", "source_z"), source_anchors.T):
        vertex[name] = values
    for name, values in zip(("nx", "ny", "nz"), normals.T):
        vertex[name] = values
    vertex["is_plane"] = is_plane
    vertex["plane_ratio"] = plane_ratio

    # X/Y/Z-dominant planes are bright red/green/blue; non-planes are gray.
    colors = np.full((len(anchors), 3), 40, dtype=np.uint8)
    dominant = np.abs(normals[is_plane]).argmax(1)
    colors[np.where(is_plane)[0], dominant] = 255
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertex, "vertex")],
        text=False,
        comments=[
            "GeoGaussian-style planar voxel anchors from LiDAR",
            f"voxel_size {voxel_size:.17g}",
        ],
    ).write(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-ply", type=Path, required=True)
    parser.add_argument(
        "--anchor-ply",
        type=Path,
        help="optional existing voxel PLY whose positions/order are preserved",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--normal-k", type=int, default=16)
    parser.add_argument("--consistency-k", type=int, default=6)
    parser.add_argument("--max-normal-angle", type=float, default=8.0)
    parser.add_argument("--planarity-threshold", type=float, default=0.55)
    parser.add_argument("--curvature-threshold", type=float, default=0.06)
    parser.add_argument("--min-plane-ratio", type=float, default=0.6)
    parser.add_argument("--smooth-neighbor-k", type=int, default=12)
    parser.add_argument("--smooth-radius-ratio", type=float, default=1.5)
    parser.add_argument("--smooth-angle", type=float, default=5.0)
    parser.add_argument("--smooth-plane-distance-ratio", type=float, default=0.08)
    parser.add_argument("--smooth-iterations", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {args.output}; pass --overwrite")
    if args.voxel_size <= 0 or args.chunk_size <= 0:
        raise ValueError("voxel-size and chunk-size must be positive")
    if not 3 <= args.consistency_k <= args.normal_k:
        raise ValueError("consistency-k must be in [3, normal-k]")
    if not 0 < args.max_normal_angle < 90:
        raise ValueError("max-normal-angle must be in (0, 90)")
    if (
        args.smooth_neighbor_k < 2
        or args.smooth_radius_ratio <= 0
        or not 0 < args.smooth_angle < 90
        or args.smooth_plane_distance_ratio <= 0
        or args.smooth_iterations < 0
    ):
        raise ValueError("plane-smoothing arguments are invalid")
    for name in ("planarity_threshold", "curvature_threshold", "min_plane_ratio"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name.replace('_', '-')} must be in [0, 1]")

    points = read_xyz(args.point_ply)
    if len(points) < args.normal_k:
        raise ValueError(
            f"point cloud has {len(points)} points, fewer than normal-k={args.normal_k}"
        )
    source_anchors = (
        read_xyz(args.anchor_ply)
        if args.anchor_ply
        else voxel_representatives(points, args.voxel_size)
    )
    point_normals, point_is_plane = estimate_point_planes(
        points,
        args.normal_k,
        args.consistency_k,
        args.max_normal_angle,
        args.planarity_threshold,
        args.curvature_threshold,
        args.chunk_size,
    )
    normals, is_plane, plane_ratio = aggregate_to_anchors(
        points,
        point_normals,
        point_is_plane,
        source_anchors,
        args.min_plane_ratio,
    )
    anchors, normals = smooth_anchor_planes(
        source_anchors,
        normals,
        is_plane,
        args.voxel_size,
        args.smooth_neighbor_k,
        args.smooth_radius_ratio,
        args.smooth_angle,
        args.smooth_plane_distance_ratio,
        args.smooth_iterations,
        args.chunk_size,
    )
    write_anchors(
        args.output,
        anchors,
        source_anchors,
        normals,
        is_plane,
        plane_ratio,
        args.voxel_size,
    )
    print(
        f"Saved {len(anchors):,} anchors to {args.output}; "
        f"point planes={point_is_plane.mean():.1%}, "
        f"anchor planes={is_plane.mean():.1%}."
    )


if __name__ == "__main__":
    main()
