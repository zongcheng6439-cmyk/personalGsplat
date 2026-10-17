#!/usr/bin/env python3
"""Compact a section in place using the AABB of extracted.ply.

No pycolmap is used. Without ``--apply`` this command only prints a plan.
With ``--apply`` it keeps box cameras, clears SfM observations/points, removes
out-of-box image/depth/mask files, then builds weight maps and voxel anchors.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np


IMAGE_FIXED = struct.Struct("<i4d3di")
UINT64 = struct.Struct("<Q")
MEDIA_SUFFIXES = {
    ".bmp", ".jpeg", ".jpg", ".npy", ".png", ".tif", ".tiff", ".webp"
}


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]
    camera_id: int
    name: str
    center: tuple[float, float, float]


def read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"binary ended early: wanted {size}, got {len(data)}")
    return data


def read_c_string(handle: BinaryIO) -> str:
    data = bytearray()
    while True:
        value = read_exact(handle, 1)
        if value == b"\0":
            return data.decode("utf-8")
        data.extend(value)


def qvec_to_rotation(qvec: Iterable[float]) -> np.ndarray:
    w, x, y, z = qvec
    return np.asarray(
        [
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*z*x + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*z*x - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
        ],
        dtype=np.float64,
    )


def stream_colmap_images(path: Path) -> Iterable[ColmapImage]:
    """Read poses and skip the large Point2D payload without loading it."""
    with path.open("rb") as handle:
        count = UINT64.unpack(read_exact(handle, UINT64.size))[0]
        for _ in range(count):
            values = IMAGE_FIXED.unpack(read_exact(handle, IMAGE_FIXED.size))
            image_id = int(values[0])
            qvec = tuple(float(v) for v in values[1:5])
            tvec = tuple(float(v) for v in values[5:8])
            camera_id = int(values[8])
            name = read_c_string(handle)
            point_count = UINT64.unpack(read_exact(handle, UINT64.size))[0]
            handle.seek(int(point_count) * 24, os.SEEK_CUR)
            center = -qvec_to_rotation(qvec).T @ np.asarray(tvec)
            yield ColmapImage(
                image_id, qvec, tvec, camera_id, name, tuple(center.tolist())
            )


def read_ascii_ply_xyz(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        if (
            handle.readline().strip() != b"ply"
            or handle.readline().strip() != b"format ascii 1.0"
        ):
            raise ValueError("extracted.ply must be ASCII PLY 1.0")
        count = None
        properties: list[str] = []
        in_vertices = False
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("incomplete PLY header")
            fields = line.decode("ascii").strip().split()
            if fields[:2] == ["element", "vertex"]:
                count, in_vertices = int(fields[2]), True
            elif fields[:1] == ["element"]:
                in_vertices = False
            elif in_vertices and fields[:1] == ["property"]:
                properties.append(fields[-1])
            elif fields == ["end_header"]:
                break
        if count is None:
            raise ValueError("PLY has no vertex element")
        xyz_columns = [properties.index(axis) for axis in ("x", "y", "z")]
        points = np.empty((count, 3), dtype=np.float32)
        for index in range(count):
            fields = handle.readline().split()
            if not fields:
                raise EOFError(f"PLY ended at vertex {index}/{count}")
            points[index] = [float(fields[column]) for column in xyz_columns]
    if not np.isfinite(points).all():
        raise ValueError("extracted.ply contains non-finite XYZ")
    return points


def in_box(center, lower: np.ndarray, upper: np.ndarray, axes: str) -> bool:
    count = 2 if axes == "xy" else 3
    center = np.asarray(center)
    return bool(
        np.all(center[:count] >= lower[:count])
        and np.all(center[:count] <= upper[:count])
    )


def write_compact_images(path: Path, images: list[ColmapImage]) -> None:
    with path.open("wb") as handle:
        handle.write(UINT64.pack(len(images)))
        for image in images:
            handle.write(
                IMAGE_FIXED.pack(
                    image.image_id, *image.qvec, *image.tvec, image.camera_id
                )
            )
            handle.write(image.name.encode("utf-8") + b"\0")
            handle.write(UINT64.pack(0))


def path_key(path: Path) -> str:
    return path.with_suffix("").as_posix()


def files_to_remove(
    root: Path, keep: set[str], keep_basenames: set[str], exact: bool
) -> list[Path]:
    if not root.is_dir():
        return []
    result = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
            continue
        relative = path_key(path.relative_to(root))
        retained = relative in keep or (
            not exact and Path(relative).name in keep_basenames
        )
        if not retained:
            result.append(path)
    return result


def matching(root: Path, name: str) -> bool:
    relative = Path(name)
    candidates = (
        root / relative,
        root / relative.with_suffix(".png"),
        root / relative.name,
        root / relative.with_suffix(".png").name,
    )
    return any(path.is_file() for path in candidates)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default="/media/zc/SSD/ar/ABXBJQ_BJ/B3/section_0907/")
    parser.add_argument("--box-margin", type=float, default=0.0)
    parser.add_argument("--box-axes", choices=("xy", "xyz"), default="xyz")
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--target-gs", type=int, default=10_000_000)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.box_margin < 0 or args.voxel_size <= 0 or args.target_gs <= 0:
        raise ValueError("margin must be non-negative; voxel-size/target-gs positive")
    data_dir = args.data_dir.expanduser().resolve()
    model = data_dir / "sparse" / "0"
    points = read_ascii_ply_xyz(model / "extracted.ply")
    lower = points.min(axis=0).astype(np.float64) - args.box_margin
    upper = points.max(axis=0).astype(np.float64) + args.box_margin
    selected = sorted(
        (
            image
            for image in stream_colmap_images(model / "images.bin")
            if in_box(image.center, lower, upper, args.box_axes)
        ),
        key=lambda image: image.image_id,
    )
    keep = {path_key(Path(image.name)) for image in selected}
    keep_basenames = {Path(value).name for value in keep}
    roots = {
        "images": (data_dir / "images", True),
        "depths": (data_dir / "depths_render_color", False),
        "masks": (data_dir / "masks", False),
        "weight_maps": (data_dir / "weight_map", False),
    }
    removals = {
        label: files_to_remove(root, keep, keep_basenames, exact)
        for label, (root, exact) in roots.items()
    }
    missing_images = [
        image.name
        for image in selected
        if not matching(data_dir / "images", image.name)
    ]
    missing_depths = [
        image.name
        for image in selected
        if not matching(data_dir / "depths_render_color", image.name)
    ]
    mask_root = data_dir / "masks"
    missing_masks = (
        [image.name for image in selected if not matching(mask_root, image.name)]
        if mask_root.is_dir()
        else []
    )
    report = {
        "data_dir": str(data_dir),
        "point_count": len(points),
        "box_lower": lower.tolist(),
        "box_upper": upper.tolist(),
        "box_axes": args.box_axes,
        "selected_cameras": len(selected),
        "missing_selected_images": len(missing_images),
        "missing_selected_depths": len(missing_depths),
        "missing_selected_masks": len(missing_masks),
        "remove": {label: len(paths) for label, paths in removals.items()},
        "mode": "apply" if args.apply else "preview",
        "pycolmap": False,
    }
    print(json.dumps(report, indent=2), flush=True)
    if not selected:
        raise ValueError("point-cloud box selected no cameras")
    if not args.apply:
        print("Preview only. Add --apply after image copying finishes.")
        return 0
    incomplete = {
        "images": missing_images,
        "depths": missing_depths,
        "masks": missing_masks,
    }
    incomplete = {name: paths for name, paths in incomplete.items() if paths}
    if incomplete:
        summary = ", ".join(f"{name}={len(paths)}" for name, paths in incomplete.items())
        first = {name: paths[:3] for name, paths in incomplete.items()}
        raise RuntimeError(
            f"copy is incomplete ({summary}); first missing entries: {first}"
        )

    with tempfile.TemporaryDirectory(prefix=".compact_sfm.", dir=model) as temp:
        temp = Path(temp)
        write_compact_images(temp / "images.bin", selected)
        (temp / "points3D.bin").write_bytes(UINT64.pack(0))
        os.replace(temp / "images.bin", model / "images.bin")
        os.replace(temp / "points3D.bin", model / "points3D.bin")
    for paths in removals.values():
        for path in paths:
            path.unlink()
    (data_dir / "section_box_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    data_process = Path(__file__).resolve().parent.parent
    weight_command = [
        sys.executable,
        str(data_process / "generate_weight_maps.py"),
        "--project_dir",
        str(data_dir),
        "--workers",
        str(args.workers),
    ]
    if not (data_dir / "masks").is_dir():
        weight_command.append("--without_mask")
    run(weight_command)
    run(
        [
            sys.executable,
            str(Path(__file__).with_name("build_section_anchor.py")),
            "--data-dir",
            str(data_dir),
            "--output",
            str(data_dir / "voxel_target_count.ply"),
            "--voxel-size",
            str(args.voxel_size),
            "--target-gs",
            str(args.target_gs),
            "--min-target-per-voxel",
            "1",
            "--frame-stride",
            str(args.frame_stride),
        ]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
