#!/usr/bin/env python3
"""Clean broken references in a COLMAP sparse model."""

from __future__ import annotations

import argparse
import shutil
import struct
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from depth_process.colmap import (
    CAMERA_MODEL_IDS,
    Camera,
    Image,
    Point3D,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    write_cameras_binary,
    write_images_binary,
    write_points3D_binary,
)


BIN_NAMES = ("cameras.bin", "images.bin", "points3D.bin")


def read_camera_model_ids(path: Path) -> dict[int, int]:
    """Read camera IDs and raw model IDs without changing their meaning."""
    result = {}
    with path.open("rb") as handle:
        count = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(count):
            raw = handle.read(24)
            if len(raw) != 24:
                raise EOFError("cameras.bin ended before all camera headers")
            camera_id, model_id, _, _ = struct.unpack("<iiQQ", raw)
            model = CAMERA_MODEL_IDS.get(model_id)
            if model is None:
                raise ValueError(f"Unknown COLMAP camera model ID: {model_id}")
            handle.seek(8 * model.num_params, 1)
            result[camera_id] = model_id
    return result


def strip_camera_distortion(cameras, model_ids):
    """Drop distortion coefficients while preserving pinhole intrinsics.

    This is appropriate only when image pixels have already been undistorted.
    It does not resample images.
    """
    simple_models = {0, 2, 3, 8, 9}
    pinhole_models = {1, 4, 5, 6, 7, 10}
    converted = {}
    changed = []
    for camera_id, camera in cameras.items():
        model_id = model_ids[camera_id]
        params = np.asarray(camera.params, dtype=np.float64)
        if model_id in simple_models:
            model, pinhole_params = "SIMPLE_PINHOLE", params[:3]
        elif model_id in pinhole_models:
            model, pinhole_params = "PINHOLE", params[:4]
        else:
            raise ValueError(f"Unsupported camera model ID: {model_id}")
        expected = 3 if model == "SIMPLE_PINHOLE" else 4
        if len(pinhole_params) != expected:
            raise ValueError(
                f"Camera {camera_id} model {model_id} has too few parameters"
            )
        converted[camera_id] = Camera(
            id=camera.id,
            model=model,
            width=camera.width,
            height=camera.height,
            params=pinhole_params.copy(),
        )
        if model_id not in (0, 1):
            changed.append(camera_id)
    return converted, changed


def load_model(model_dir: Path):
    """Read the three COLMAP BIN files without pycolmap validation."""
    return (
        read_cameras_binary(model_dir / "cameras.bin"),
        read_images_binary(model_dir / "images.bin"),
        read_points3D_binary(model_dir / "points3D.bin"),
    )


def clean_model(cameras, images, points, min_track_length: int):
    """Remove invalid tracks and return a consistent model."""
    input_image_count = len(images)
    images = {
        image_id: image
        for image_id, image in images.items()
        if image.camera_id in cameras
    }

    clean_points = {}
    observations = set()
    removed_tracks = 0

    for point_id, point in points.items():
        valid_tracks = []
        seen = set()

        for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs):
            image_id, point2d_idx = int(image_id), int(point2d_idx)
            pair = (image_id, point2d_idx)
            image = images.get(image_id)

            valid = (
                image is not None
                and 0 <= point2d_idx < len(image.point3D_ids)
                and int(image.point3D_ids[point2d_idx]) == point_id
                and pair not in seen
            )
            if valid:
                valid_tracks.append(pair)
                seen.add(pair)
            else:
                removed_tracks += 1

        if len(valid_tracks) < min_track_length:
            continue

        image_ids, point2d_idxs = zip(*valid_tracks)
        clean_points[point_id] = Point3D(
            id=point.id,
            xyz=point.xyz,
            rgb=point.rgb,
            error=point.error,
            image_ids=np.asarray(image_ids, dtype=np.int32),
            point2D_idxs=np.asarray(point2d_idxs, dtype=np.int32),
        )
        observations.update(
            (image_id, point2d_idx, point_id)
            for image_id, point2d_idx in valid_tracks
        )

    cleared_image_refs = 0
    clean_images = {}
    for image_id, image in images.items():
        point_ids = image.point3D_ids.copy()
        for point2d_idx, point_id in enumerate(point_ids):
            point_id = int(point_id)
            if point_id >= 0 and (image_id, point2d_idx, point_id) not in observations:
                point_ids[point2d_idx] = -1
                cleared_image_refs += 1
        clean_images[image_id] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=image.xys,
            point3D_ids=point_ids,
        )

    stats = {
        "input_images": input_image_count,
        "input_points": len(points),
        "removed_images_missing_camera": input_image_count - len(clean_images),
        "removed_track_elements": removed_tracks,
        "removed_points": len(points) - len(clean_points),
        "cleared_image_point_refs": cleared_image_refs,
        "output_images": len(clean_images),
        "output_points": len(clean_points),
    }
    return clean_images, clean_points, stats


def verify_model(cameras, images, points):
    """Check every image/point observation in both directions."""
    observations = set()

    for point_id, point in points.items():
        for image_id, point2d_idx in zip(point.image_ids, point.point2D_idxs):
            image_id, point2d_idx = int(image_id), int(point2d_idx)
            image = images.get(image_id)
            if image is None:
                raise ValueError(f"Point {point_id} references missing image {image_id}")
            if not 0 <= point2d_idx < len(image.point3D_ids):
                raise ValueError(f"Point {point_id} has invalid point2D index")
            if int(image.point3D_ids[point2d_idx]) != point_id:
                raise ValueError(f"Point {point_id} is not linked back by image {image_id}")
            observations.add((image_id, point2d_idx, point_id))

    for image_id, image in images.items():
        if image.camera_id not in cameras:
            raise ValueError(f"Image {image_id} references missing camera")
        for point2d_idx, point_id in enumerate(image.point3D_ids):
            point_id = int(point_id)
            if point_id >= 0 and (image_id, point2d_idx, point_id) not in observations:
                raise ValueError(f"Image {image_id} references missing point {point_id}")

    return {
        "cameras": len(cameras),
        "images": len(images),
        "points": len(points),
        "track_elements": len(observations),
    }


def write_and_verify(
    model_dir: Path, source: Path, cameras, images, points, rewrite_cameras: bool
):
    """Write a staged model, read it back, and validate it."""
    if rewrite_cameras:
        write_cameras_binary(cameras, model_dir / "cameras.bin")
    else:
        shutil.copy2(source / "cameras.bin", model_dir / "cameras.bin")
    write_images_binary(images, model_dir / "images.bin")
    write_points3D_binary(points, model_dir / "points3D.bin")
    verification = verify_model(*load_model(model_dir))
    return verification


def verify_with_pycolmap(model_dir: Path, expected_images: int, label: str):
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_dir))
    if reconstruction.num_images() != expected_images:
        raise RuntimeError("pycolmap image count differs from binary verification")
    print(f"{label}: pycolmap {pycolmap.__version__} load OK")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Remove invalid cross-references from a COLMAP sparse model."
    )
    parser.add_argument("--input", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--in_place", action="store_true")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--min_track_length", type=int, default=1)
    parser.add_argument(
        "--strip_distortion",
        action="store_true",
        help=(
            "Rewrite camera models as PINHOLE/SIMPLE_PINHOLE by dropping "
            "distortion coefficients. Use only if images are already undistorted."
        ),
    )
    parser.add_argument("--verify_pycolmap", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.input.resolve()

    if args.min_track_length < 1:
        raise ValueError("--min_track_length must be at least 1")
    if args.backup and not args.in_place:
        raise ValueError("--backup requires --in_place")
    missing = [name for name in BIN_NAMES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {source}: {', '.join(missing)}")

    output = args.output.resolve() if args.output else None
    backup = None
    if output and output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if args.in_place:
        backup = (
            args.backup.resolve()
            if args.backup
            else source.with_name(
                f"{source.name}_before_filter_{datetime.now():%Y%m%d_%H%M%S}"
            )
        )
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
        if backup == source or source in backup.parents:
            raise ValueError("Backup must be outside the input directory")

    cameras, images, points = load_model(source)
    converted_camera_ids = []
    if args.strip_distortion:
        model_ids = read_camera_model_ids(source / "cameras.bin")
        cameras, converted_camera_ids = strip_camera_distortion(cameras, model_ids)
    images, points, stats = clean_model(
        cameras, images, points, args.min_track_length
    )

    parent = source.parent if args.in_place else output.parent
    parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{source.name}.filtered.", dir=parent))

    try:
        verification = write_and_verify(
            staged,
            source,
            cameras,
            images,
            points,
            rewrite_cameras=args.strip_distortion,
        )
        if args.verify_pycolmap:
            verify_with_pycolmap(staged, verification["images"], "staged model")

        if args.in_place:
            shutil.copytree(source, backup, copy_function=shutil.copy2)
            try:
                for name in BIN_NAMES:
                    (staged / name).replace(source / name)
            except Exception:
                for name in BIN_NAMES:
                    shutil.copy2(backup / name, source / name)
                raise
            staged.rmdir()
            result = source
        else:
            staged.rename(output)
            result = output
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise

    # Verify the files at their final location too.
    verify_model(*load_model(result))
    if args.verify_pycolmap:
        verify_with_pycolmap(result, verification["images"], "final model")

    results = {
        **stats,
        "converted_distorted_cameras": len(converted_camera_ids),
        **{f"verified_{k}": v for k, v in verification.items()},
    }
    for name, value in results.items():
        print(f"{name}: {value}")
    if backup:
        print(f"Complete backup saved to: {backup}")
    print(f"Filtered model saved to: {result}")


if __name__ == "__main__":
    main()
