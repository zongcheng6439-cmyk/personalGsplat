#!/usr/bin/env python3
"""Expand a section point cloud with visible points from a full-scene PLY.

Visibility uses selected COLMAP cameras, positive depth, image bounds and a
coarse z-buffer. The original section PLY is never modified.

Test run:

    cd /home/zc/code/aibee/gsplat
    python3 utils/data_process/reconstruction_section/expand_point_cloud_by_visibility.py \
        --data-dir /media/zc/SSD/GS/lutu/parking01_section/ \
        --output /media/zc/SSD/GS/lutu/parking01_section/sparse/0/extracted_visible_test.ply \
        --frame-stride 20 \
        --overwrite

"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_section_anchor import camera_matrix, read_cameras  # noqa: E402
from prepare_section_from_box import qvec_to_rotation, stream_colmap_images  # noqa: E402


PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def ply_layout(path: Path):
    ply_format = None
    properties = []
    vertex_count = None
    in_vertices = False
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"incomplete PLY header: {path}")
            fields = line.decode("ascii").strip().split()
            if fields[:1] == ["format"]:
                if len(fields) != 3 or fields[2] != "1.0":
                    raise ValueError(f"unsupported PLY format line: {fields}")
                ply_format = fields[1]
            elif fields[:2] == ["element", "vertex"]:
                vertex_count, in_vertices = int(fields[2]), True
            elif fields[:1] == ["element"]:
                in_vertices = False
            elif in_vertices and fields[:1] == ["property"]:
                if len(fields) != 3 or fields[1] not in PLY_TYPES:
                    raise ValueError(f"unsupported vertex property: {fields}")
                properties.append((fields[2], PLY_TYPES[fields[1]]))
            elif fields == ["end_header"]:
                break
        offset = handle.tell()
    if ply_format not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError(f"unsupported PLY encoding: {ply_format}")
    if vertex_count is None or not properties:
        raise ValueError(f"PLY has no vertex data: {path}")
    return vertex_count, offset, ply_format, properties


def load_vertices(path: Path) -> np.ndarray:
    count, offset, ply_format, properties = ply_layout(path)
    names = [name for name, _ in properties]
    for axis in ("x", "y", "z"):
        if axis not in names:
            raise ValueError(f"PLY is missing {axis}: {path}")
    with path.open("rb") as handle:
        handle.seek(offset)
        if ply_format == "ascii":
            data = np.loadtxt(handle, dtype=np.float32, max_rows=count)
            if data.ndim == 1:
                data = data[None]
            columns = {name: data[:, index] for index, name in enumerate(names)}
        else:
            endian = "<" if ply_format == "binary_little_endian" else ">"
            dtype = np.dtype([(name, endian + kind) for name, kind in properties])
            data = np.fromfile(handle, dtype=dtype, count=count)
            columns = {name: data[name] for name in names}
    if len(data) != count:
        raise ValueError(f"PLY vertex count mismatch: {len(data)} != {count}")
    rows = np.empty((count, 6), dtype=np.float32)
    rows[:, :3] = np.column_stack([columns[axis] for axis in ("x", "y", "z")])
    color_aliases = (("red", "r"), ("green", "g"), ("blue", "b"))
    for index, aliases in enumerate(color_aliases, start=3):
        name = next((name for name in aliases if name in columns), None)
        rows[:, index] = 127 if name is None else columns[name]
    return rows


def selected_frames(model_dir: Path, image_root: Path, stride: int):
    by_camera = defaultdict(list)
    for frame in stream_colmap_images(model_dir / "images.bin"):
        if (image_root / frame.name).is_file():
            by_camera[frame.camera_id].append(frame)
    selected = []
    for frames in by_camera.values():
        frames.sort(key=lambda frame: frame.name)
        selected.extend(frames[::stride])
    return sorted(selected, key=lambda frame: (frame.name, frame.camera_id))


def visible_from_camera(
    xyz: np.ndarray,
    candidate_ids: np.ndarray,
    frame,
    camera,
    occlusion_cell: int,
    depth_tolerance: float,
) -> np.ndarray:
    rotation = qvec_to_rotation(frame.qvec)
    camera_xyz = xyz[candidate_ids] @ rotation.T + np.asarray(frame.tvec)
    depth = camera_xyz[:, 2]
    _, width, height, _ = camera
    projected = camera_xyz @ camera_matrix(camera, width, height).T
    u = projected[:, 0] / np.maximum(projected[:, 2], 1e-12)
    v = projected[:, 1] / np.maximum(projected[:, 2], 1e-12)
    valid = (
        np.isfinite(camera_xyz).all(axis=1)
        & (depth > 1e-6)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    if not np.any(valid):
        return np.empty(0, dtype=np.int64)
    ids = candidate_ids[valid]
    depth = depth[valid]
    cell_width = (width + occlusion_cell - 1) // occlusion_cell
    cell_height = (height + occlusion_cell - 1) // occlusion_cell
    cell_x = (u[valid].astype(np.int64) // occlusion_cell)
    cell_y = (v[valid].astype(np.int64) // occlusion_cell)
    cell = cell_y * cell_width + cell_x
    nearest = np.full(cell_width * cell_height, np.inf, dtype=np.float32)
    np.minimum.at(nearest, cell, depth)
    return ids[depth <= nearest[cell] + depth_tolerance]


def write_expanded_ply(
    output: Path,
    section_rows: np.ndarray,
    added_rows: np.ndarray,
) -> None:
    count = len(section_rows) + len(added_rows)
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
         ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as destination:
        destination.write(header)
        for rows in (section_rows, added_rows):
            for start in range(0, len(rows), 1_000_000):
                chunk = rows[start : start + 1_000_000]
                vertices = np.empty(len(chunk), dtype=dtype)
                vertices["x"], vertices["y"], vertices["z"] = chunk[:, :3].T
                colors = np.clip(np.rint(chunk[:, 3:6]), 0, 255).astype(np.uint8)
                vertices["red"], vertices["green"], vertices["blue"] = colors.T
                vertices.tofile(destination)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-distance", type=float, default=20.0)
    parser.add_argument("--occlusion-cell", type=int, default=4)
    parser.add_argument("--depth-tolerance", type=float, default=0.10)
    parser.add_argument("--dedup-distance", type=float, default=0.005)
    parser.add_argument("--min-visible-views", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    model_dir = data_dir / "sparse" / "0"
    full_scene = data_dir / "full_scene.ply"
    section_ply = model_dir / "extracted.ply"
    output = (args.output or model_dir / "extracted_visible.ply").resolve()
    if output in (full_scene, section_ply):
        raise ValueError("output must not overwrite an input PLY")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}")
    if min(
        args.frame_stride,
        args.occlusion_cell,
        args.min_visible_views,
    ) <= 0 or min(args.max_distance, args.depth_tolerance, args.dedup_distance) < 0:
        raise ValueError("invalid non-positive visibility parameter")

    print(f"Loading full scene: {full_scene}", flush=True)
    full_rows = load_vertices(full_scene)
    print(f"Loading section: {section_ply}", flush=True)
    section_rows = load_vertices(section_ply)
    full_xyz = np.ascontiguousarray(full_rows[:, :3])
    section_xyz = np.ascontiguousarray(section_rows[:, :3])
    cameras = read_cameras(model_dir / "cameras.bin")
    frames = selected_frames(model_dir, data_dir / "images", args.frame_stride)
    if not frames:
        raise ValueError("no selected COLMAP image exists on disk")

    full_tree = cKDTree(full_xyz)
    visible_count = np.zeros(len(full_xyz), dtype=np.uint16)
    for index, frame in enumerate(frames, start=1):
        candidate_ids = np.asarray(
            full_tree.query_ball_point(frame.center, args.max_distance),
            dtype=np.int64,
        )
        visible_ids = visible_from_camera(
            full_xyz,
            candidate_ids,
            frame,
            cameras[frame.camera_id],
            args.occlusion_cell,
            args.depth_tolerance,
        )
        visible_count[visible_ids] += 1
        if index == 1 or index % 10 == 0 or index == len(frames):
            print(
                f"[{index}/{len(frames)}] candidates={len(candidate_ids):,} "
                f"visible={len(visible_ids):,}",
                flush=True,
            )

    visible_ids = np.flatnonzero(visible_count >= args.min_visible_views)
    if len(visible_ids):
        nearest, _ = cKDTree(section_xyz).query(
            full_xyz[visible_ids], k=1, distance_upper_bound=args.dedup_distance
        )
        visible_ids = visible_ids[~np.isfinite(nearest)]
    added_rows = full_rows[visible_ids]
    write_expanded_ply(
        output,
        section_rows,
        added_rows,
    )
    report = {
        "full_scene_points": len(full_rows),
        "section_points": len(section_rows),
        "sampled_images": len(frames),
        "visible_full_scene_points": int(np.count_nonzero(visible_count)),
        "added_points_after_dedup": len(added_rows),
        "output_points": len(section_rows) + len(added_rows),
        "output": str(output),
        "parameters": {
            "frame_stride_per_camera": args.frame_stride,
            "max_distance": args.max_distance,
            "occlusion_cell": args.occlusion_cell,
            "depth_tolerance": args.depth_tolerance,
            "dedup_distance": args.dedup_distance,
            "min_visible_views": args.min_visible_views,
        },
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
