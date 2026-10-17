"""Shared helpers for the customized COLMAP dataset variants."""

import os
from typing import List

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm


def fetchPly(path):
    """Load XYZ, RGB, and optional normals from a PLY point cloud."""
    print(f"Read lidar pc: {path}")
    vertices = PlyData.read(path)["vertex"]
    names = vertices.data.dtype.names or ()
    positions = np.stack(
        [vertices["x"], vertices["y"], vertices["z"]], axis=-1
    ).astype(np.float32)
    colors = (
        np.stack(
            [vertices["red"], vertices["green"], vertices["blue"]], axis=-1
        )
        if all(name in names for name in ("red", "green", "blue"))
        else np.full_like(positions, 128)
    )
    normals = (
        np.stack([vertices["nx"], vertices["ny"], vertices["nz"]], axis=-1)
        if all(name in names for name in ("nx", "ny", "nz"))
        else np.zeros_like(positions)
    )
    return positions, colors, normals


def count_files(path, recursive=True):
    if recursive:
        return sum(len(files) for _, _, files in os.walk(path))
    return sum(
        os.path.isfile(os.path.join(path, filename)) for filename in os.listdir(path)
    )


def _get_rel_paths(path_dir: str) -> List[str]:
    return [
        os.path.relpath(os.path.join(root, filename), path_dir)
        for root, _, filenames in os.walk(path_dir)
        for filename in filenames
    ]


def _resize_image_folder(
    image_dir: str,
    resized_dir: str,
    factor: int,
    resize_mask: bool = False,
    skip_existing: bool = False,
) -> str:
    """Resize an image or mask tree while preserving relative subdirectories."""
    print(f"Downscaling images by {factor}x from {image_dir} to {resized_dir}.")
    os.makedirs(resized_dir, exist_ok=True)
    for relative_path in tqdm(_get_rel_paths(image_dir)):
        source = os.path.join(image_dir, relative_path)
        destination = os.path.join(
            resized_dir, os.path.splitext(relative_path)[0] + ".png"
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if (
            os.path.isfile(destination)
            and (
                skip_existing
                or os.path.getmtime(destination) >= os.path.getmtime(source)
            )
        ):
            continue
        image = imageio.imread(source)
        if not resize_mask:
            image = image[..., :3]
        size = (
            max(1, int(round(image.shape[1] / factor))),
            max(1, int(round(image.shape[0] / factor))),
        )
        interpolation = Image.Resampling.NEAREST if resize_mask else Image.Resampling.BICUBIC
        resized = Image.fromarray(image).resize(size, interpolation)
        imageio.imwrite(destination, np.asarray(resized))
    return resized_dir


def reconstruction_points(reconstruction, as_dict):
    """Convert pycolmap points3D into contiguous NumPy arrays."""
    points3D = as_dict(reconstruction.points3D)
    point_ids = sorted(points3D)
    points = np.asarray(
        [points3D[point_id].xyz for point_id in point_ids], dtype=np.float32
    ).reshape(-1, 3)
    errors = np.asarray(
        [points3D[point_id].error for point_id in point_ids], dtype=np.float32
    )
    colors = np.asarray(
        [points3D[point_id].color for point_id in point_ids], dtype=np.uint8
    ).reshape(-1, 3)
    return points, errors, colors
