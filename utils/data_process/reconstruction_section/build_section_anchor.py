#!/usr/bin/env python3
"""Compute voxel anchor weights from COLMAP BIN files without pycolmap."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PROCESS_DIR = SCRIPT_DIR.parent
if str(DATA_PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_PROCESS_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_voxel_anchor_budget import (  # noqa: E402
    _load_weight_map,
    allocate_integer_budget,
    limit_target_count_categories,
    read_target_count_ply,
    sample_frame_weights,
    score_to_priority,
    voxel_downsample,
    write_target_count_ply,
)
from prepare_section_from_box import (  # noqa: E402
    qvec_to_rotation,
    read_ascii_ply_xyz,
    stream_colmap_images,
)


UINT64 = struct.Struct("<Q")
CAMERA_FIXED = struct.Struct("<iiQQ")
CAMERA_PARAM_COUNTS = {
    0: 3,
    1: 4,
    2: 4,
    3: 5,
    4: 8,
    5: 8,
    6: 12,
    7: 5,
    8: 4,
    9: 5,
    10: 12,
}


def read_cameras(path: Path) -> dict[int, tuple[int, int, int, np.ndarray]]:
    cameras: dict[int, tuple[int, int, int, np.ndarray]] = {}
    with path.open("rb") as handle:
        count = UINT64.unpack(handle.read(UINT64.size))[0]
        for _ in range(count):
            data = handle.read(CAMERA_FIXED.size)
            if len(data) != CAMERA_FIXED.size:
                raise EOFError("cameras.bin ended early")
            camera_id, model_id, width, height = CAMERA_FIXED.unpack(data)
            if model_id not in CAMERA_PARAM_COUNTS:
                raise ValueError(f"unknown COLMAP camera model ID {model_id}")
            param_count = CAMERA_PARAM_COUNTS[model_id]
            raw = handle.read(8 * param_count)
            if len(raw) != 8 * param_count:
                raise EOFError("camera parameters ended early")
            params = np.asarray(struct.unpack(f"<{param_count}d", raw))
            if model_id not in (0, 1, 2):
                raise ValueError(
                    f"camera {camera_id} uses distorted/unsupported model ID "
                    f"{model_id}; undistort the model first"
                )
            cameras[int(camera_id)] = (
                int(model_id), int(width), int(height), params
            )
    return cameras


def camera_matrix(
    camera: tuple[int, int, int, np.ndarray], image_width: int, image_height: int
) -> np.ndarray:
    model_id, width, height, params = camera
    if model_id in (0, 2):
        # Match the existing gsplat COLMAP loader: SIMPLE_RADIAL uses its
        # pinhole part (f, cx, cy), with the radial coefficient ignored.
        fx = fy = params[0]
        cx, cy = params[1:3]
    elif model_id == 1:
        fx, fy, cx, cy = params
    else:  # guarded by read_cameras; retained for defensive use.
        raise ValueError(f"unsupported camera model ID {model_id}")
    scale_x = image_width / width
    scale_y = image_height / height
    return np.asarray(
        [
            [fx * scale_x, 0.0, cx * scale_x],
            [0.0, fy * scale_y, cy * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--target-gs", type=int, required=True)
    parser.add_argument("--min-target-per-voxel", type=int, default=1)
    parser.add_argument("--score-gamma", type=float, default=1.0)
    parser.add_argument("--score-clip-quantile", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--depth-tolerance", type=float, default=-1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.voxel_size <= 0 or args.target_gs <= 0 or args.frame_stride <= 0:
        raise ValueError("voxel-size, target-gs and frame-stride must be positive")
    if args.min_target_per_voxel < 0 or args.max_frames < 0:
        raise ValueError("minimum target and max-frames must be non-negative")

    data_dir = args.data_dir.expanduser().resolve()
    model_dir = data_dir / "sparse" / "0"
    cameras = read_cameras(model_dir / "cameras.bin")
    frames = sorted(
        stream_colmap_images(model_dir / "images.bin"), key=lambda frame: frame.name
    )[:: args.frame_stride]
    if args.max_frames and len(frames) > args.max_frames:
        indices = np.linspace(0, len(frames) - 1, args.max_frames, dtype=np.int64)
        frames = [frames[int(index)] for index in indices]
    if not frames:
        raise ValueError("no frames selected")

    points = read_ascii_ply_xyz(model_dir / "extracted.ply")
    voxels = voxel_downsample(points, args.voxel_size)
    minimum_total = len(voxels) * args.min_target_per_voxel
    if args.target_gs < max(minimum_total, len(points)):
        raise ValueError(
            f"target-gs={args.target_gs} must be at least "
            f"max(voxels*minimum={minimum_total}, input_points={len(points)})"
        )

    score_sum = np.zeros(len(voxels), dtype=np.float64)
    visible_views = np.zeros(len(voxels), dtype=np.int32)
    tolerance = args.voxel_size * 0.5 if args.depth_tolerance < 0 else args.depth_tolerance
    for position, frame in enumerate(frames, start=1):
        image_path = data_dir / "images" / frame.name
        weight_path = data_dir / "weight_map" / Path(frame.name).with_suffix(".npy")
        with Image.open(image_path) as image:
            width, height = image.size
        weight = _load_weight_map(str(weight_path), width, height)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = qvec_to_rotation(frame.qvec)
        world_to_camera[:3, 3] = np.asarray(frame.tvec)
        camera_to_world = np.linalg.inv(world_to_camera)
        ids, values = sample_frame_weights(
            voxels,
            camera_to_world,
            camera_matrix(cameras[frame.camera_id], width, height),
            weight,
            tolerance,
            valid_mask=None,
        )
        score_sum[ids] += values
        visible_views[ids] += 1
        if position == 1 or position % 10 == 0 or position == len(frames):
            print(f"[{position}/{len(frames)}] {frame.name}: scored voxels={len(ids):,}", flush=True)

    score = np.divide(
        score_sum,
        np.maximum(visible_views, 1),
        out=np.zeros_like(score_sum),
        where=visible_views > 0,
    )
    priority = score_to_priority(score, args.score_clip_quantile, args.score_gamma)
    target = allocate_integer_budget(priority, args.target_gs, args.min_target_per_voxel)
    categories_before = len(np.unique(target))
    target = limit_target_count_categories(target, max_categories=256)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    write_target_count_ply(output, voxels, target, voxel_size=args.voxel_size)
    saved_xyz, saved_target = read_target_count_ply(output)
    if not np.array_equal(saved_xyz, voxels) or not np.array_equal(saved_target, target):
        raise RuntimeError("anchor PLY round-trip verification failed")
    print(
        json.dumps(
            {
                "output": str(output),
                "input_points": len(points),
                "voxels": len(voxels),
                "processed_frames": len(frames),
                "visible_voxel_ratio": float(np.mean(visible_views > 0)),
                "requested_target_gs": args.target_gs,
                "stored_target_gs": int(target.sum()),
                "target_count_min": int(target.min()),
                "target_count_max": int(target.max()),
                "target_categories_before": int(categories_before),
                "target_categories_after": int(len(np.unique(target))),
                "verification": "PASS",
                "pycolmap": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
