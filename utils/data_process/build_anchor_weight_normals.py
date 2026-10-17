#!/usr/bin/env python3
"""Build voxel weights and ground/wall normals from a detail point cloud."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_voxel_anchor_budget import (  # noqa: E402
    limit_target_count_categories,
    write_target_count_ply,
)


def read_xyz(path):
    vertex = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
        np.float32
    )
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("input PLY contains no finite XYZ points")
    return xyz


def voxelize(points, voxel_size):
    """Use each voxel centroid as anchor and point count as its weight."""
    origin = np.floor(points.min(0) / voxel_size) * voxel_size
    keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    keys = keys[order]
    starts = np.r_[0, np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1]
    counts = np.diff(np.r_[starts, len(points)]).astype(np.int64)
    centers = np.add.reduceat(points[order].astype(np.float64), starts, axis=0)
    return (centers / counts[:, None]).astype(np.float32), counts


def estimate_normals(points, anchors, radius, up):
    """PCA normals; type 1 is ground, type 2 is wall, 0 is ignored."""
    tree = cKDTree(points, leafsize=32)
    normals = np.zeros_like(anchors)
    surface_type = np.zeros(len(anchors), dtype=np.uint8)
    k, batch_size = min(48, len(points)), 32768

    for start in range(0, len(anchors), batch_size):
        stop = min(start + batch_size, len(anchors))
        distance, index = tree.query(
            anchors[start:stop], k=k, distance_upper_bound=radius, workers=-1
        )
        if k == 1:
            distance, index = distance[:, None], index[:, None]
        valid = np.isfinite(distance) & (index < len(points))
        neighbor = points[np.where(valid, index, 0)].astype(np.float64)
        count = valid.sum(1)
        weight = valid[..., None]
        mean = (neighbor * weight).sum(1) / np.maximum(count[:, None], 1)
        centered = (neighbor - mean[:, None]) * weight
        covariance = np.einsum("bki,bkj->bij", centered, centered)
        covariance /= np.maximum(count[:, None, None], 1)
        eigenvalue, eigenvector = np.linalg.eigh(covariance)
        normal = eigenvector[:, :, 0]

        variation = eigenvalue[:, 0] / np.maximum(eigenvalue.sum(1), 1e-12)
        tangent_ratio = eigenvalue[:, 1] / np.maximum(eigenvalue[:, 2], 1e-12)
        planar = (count >= 10) & (variation < 0.02) & (tangent_ratio > 0.1)
        up_dot = np.abs(normal @ up)
        ground = planar & (up_dot > np.cos(np.deg2rad(20)))
        wall = planar & (up_dot < np.sin(np.deg2rad(20)))

        row = np.arange(len(normal))
        normal[normal[row, np.abs(normal).argmax(1)] < 0] *= -1
        keep = ground | wall
        normals[start:stop][keep] = normal[keep]
        surface_type[start:stop][ground] = 1
        surface_type[start:stop][wall] = 2
        print(f"Normals: {stop:,}/{len(anchors):,}", flush=True)

    return normals, surface_type


def write_normal_ply(path, anchors, normals, surface_type):
    """Write only valid planes; anchor_id indexes anchor_weight.ply."""
    anchor_id = np.flatnonzero(surface_type).astype(np.int32)
    xyz, normal = anchors[anchor_id], normals[anchor_id]
    point_type = surface_type[anchor_id]
    color = np.zeros((len(anchor_id), 3), dtype=np.uint8)
    color[point_type == 1] = (0, 0, 255)  # ground, dominant Z
    wall = point_type == 2
    wall_x = wall & (np.abs(normal[:, 0]) >= np.abs(normal[:, 1]))
    color[wall_x] = (255, 0, 0)           # wall, dominant X
    color[wall & ~wall_x] = (0, 255, 0)   # wall, dominant Y
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("anchor_id", "i4"), ("surface_type", "u1"),
    ]
    vertex = np.empty(len(anchor_id), dtype=dtype)
    vertex["x"], vertex["y"], vertex["z"] = xyz.T
    vertex["nx"], vertex["ny"], vertex["nz"] = normal.T
    vertex["red"], vertex["green"], vertex["blue"] = color.T
    vertex["anchor_id"], vertex["surface_type"] = anchor_id, point_type
    PlyData(
        [PlyElement.describe(vertex, "vertex")],
        text=False,
        comments=["normal axes: X=red, Y=green, Z=blue", "anchor_id indexes weight PLY"],
    ).write(str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", type=Path, default='/media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/sparse/0/extracted_detail.ply')
    parser.add_argument("--output-dir", type=Path, default='/media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/anchor')
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--normal-radius", type=float)
    parser.add_argument("--up-axis", choices="xyz", default="z")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.voxel_size <= 0 or (args.normal_radius is not None and args.normal_radius <= 0):
        raise ValueError("voxel-size and normal-radius must be positive")
    output_dir = args.output_dir.expanduser().resolve()
    weight_path = output_dir / "anchor_weight.ply"
    normal_path = output_dir / "anchor_surface_normals.ply"
    if not args.overwrite:
        for path in (weight_path, normal_path):
            if path.exists():
                raise FileExistsError(f"output exists: {path}; pass --overwrite")

    points = read_xyz(args.input_ply.expanduser().resolve())
    anchors, weight = voxelize(points, args.voxel_size)
    weight = limit_target_count_categories(weight, 256)
    up = np.eye(3)["xyz".index(args.up_axis)]
    normals, surface_type = estimate_normals(
        points, anchors, args.normal_radius or 3 * args.voxel_size, up
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_target_count_ply(weight_path, anchors, weight, args.voxel_size)
    write_normal_ply(normal_path, anchors, normals, surface_type)
    print(json.dumps({
        "input_points": len(points),
        "anchors": len(anchors),
        "weight": "detail point count per voxel",
        "ground_anchors": int(np.sum(surface_type == 1)),
        "wall_anchors": int(np.sum(surface_type == 2)),
        "weight_ply": str(weight_path),
        "normal_ply": str(normal_path),
    }, indent=2))


if __name__ == "__main__":
    main()
