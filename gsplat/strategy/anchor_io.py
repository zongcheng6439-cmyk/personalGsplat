from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


_ANCHOR_PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)

_DENSE_ANCHOR_INIT_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("anchor_id", "<i4"),
        ("base_scale", "<f4"),
    ]
)


def read_anchor_capacity_ply(
    path: str | Path, *, expected_voxel_size: float | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Read the grayscale anchor-budget PLY produced by stage one.

    The RGB value is an index into ``comment target_count_lut`` records in the
    header; it is not the capacity itself.
    """
    path = Path(path)
    header_lines = []
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("incomplete anchor PLY header")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                payload_offset = handle.tell()
                break

    vertex_count = None
    expected_total = None
    gray_levels = []
    count_levels = []
    voxel_sizes = []
    for line in header_lines:
        fields = line.split()
        if len(fields) == 3 and fields[:2] == ["element", "vertex"]:
            vertex_count = int(fields[2])
        elif len(fields) == 4 and fields[:2] == ["comment", "target_count_lut"]:
            gray_levels.append(int(fields[2]))
            count_levels.append(int(fields[3]))
        elif len(fields) == 3 and fields[:2] == ["comment", "target_count_total"]:
            expected_total = int(fields[2])
        elif len(fields) == 3 and fields[:2] == ["comment", "voxel_size"]:
            voxel_sizes.append(float(fields[2]))

    expected_properties = [
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    actual_properties = [
        line for line in header_lines if line.startswith("property ")
    ]
    if header_lines[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError("anchor PLY must use binary little-endian format")
    if actual_properties != expected_properties:
        raise ValueError("anchor PLY vertex schema does not match the budget format")
    if vertex_count is None or expected_total is None or not gray_levels:
        raise ValueError("anchor PLY is missing target-count metadata")
    if len(voxel_sizes) > 1 or (
        voxel_sizes and (not np.isfinite(voxel_sizes[0]) or voxel_sizes[0] <= 0)
    ):
        raise ValueError("anchor PLY contains invalid voxel_size metadata")
    if (
        expected_voxel_size is not None
        and voxel_sizes
        and not np.isclose(
            voxel_sizes[0], float(expected_voxel_size), rtol=1e-6, atol=1e-9
        )
    ):
        raise ValueError(
            f"anchor PLY voxel_size={voxel_sizes[0]} does not match "
            f"strategy anchor_voxel_size={expected_voxel_size}"
        )

    gray_levels_np = np.asarray(gray_levels, dtype=np.int64)
    count_levels_np = np.asarray(count_levels, dtype=np.int64)
    if (
        len(gray_levels_np) != len(count_levels_np)
        or len(gray_levels_np) > 256
        or np.any((gray_levels_np < 0) | (gray_levels_np > 255))
        or np.any(count_levels_np < 0)
        or np.any(np.diff(gray_levels_np) <= 0)
        or np.any(np.diff(count_levels_np) <= 0)
    ):
        raise ValueError("anchor PLY contains an invalid target-count lookup table")

    expected_size = payload_offset + vertex_count * _ANCHOR_PLY_DTYPE.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError("anchor PLY payload size does not match its vertex count")
    with path.open("rb") as handle:
        handle.seek(payload_offset)
        vertices = np.fromfile(handle, dtype=_ANCHOR_PLY_DTYPE, count=vertex_count)
    if not (
        np.array_equal(vertices["red"], vertices["green"])
        and np.array_equal(vertices["red"], vertices["blue"])
    ):
        raise ValueError("anchor PLY capacity colors must be grayscale")

    lookup = np.full(256, -1, dtype=np.int64)
    lookup[gray_levels_np] = count_levels_np
    capacity = lookup[vertices["red"]]
    if np.any(capacity < 0):
        raise ValueError("anchor PLY uses a gray level absent from its lookup table")
    if int(capacity.sum()) != expected_total:
        raise ValueError("anchor PLY capacities do not match target_count_total")

    xyz = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=1)
    if not np.isfinite(xyz).all():
        raise ValueError("anchor PLY contains non-finite coordinates")
    return np.ascontiguousarray(xyz), capacity


def read_dense_anchor_init_ply(
    path: str | Path,
    *,
    expected_anchor_count: int | None = None,
    expected_voxel_size: float | None = None,
    expected_anchor_budget_sha256: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read XYZ, RGB, and explicit anchor IDs from a dense initialization PLY."""
    path = Path(path)
    header_lines = []
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("incomplete dense-anchor PLY header")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                payload_offset = handle.tell()
                break

    vertex_count = None
    metadata_anchor_count = None
    metadata_gaussian_count = None
    voxel_sizes = []
    anchor_budget_hashes = []
    dense_marker = False
    for line in header_lines:
        fields = line.split()
        if len(fields) == 3 and fields[:2] == ["element", "vertex"]:
            vertex_count = int(fields[2])
        elif fields == ["comment", "dense_anchor_init", "1"]:
            dense_marker = True
        elif len(fields) == 3 and fields[:2] == ["comment", "anchor_count"]:
            metadata_anchor_count = int(fields[2])
        elif len(fields) == 3 and fields[:2] == ["comment", "gaussian_count"]:
            metadata_gaussian_count = int(fields[2])
        elif len(fields) == 3 and fields[:2] == ["comment", "voxel_size"]:
            voxel_sizes.append(float(fields[2]))
        elif len(fields) == 3 and fields[:2] == [
            "comment",
            "anchor_budget_sha256",
        ]:
            anchor_budget_hashes.append(fields[2].lower())

    expected_properties = [
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property int anchor_id",
        "property float base_scale",
    ]
    actual_properties = [
        line for line in header_lines if line.startswith("property ")
    ]
    if header_lines[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError("dense-anchor PLY must use binary little-endian format")
    if not dense_marker or actual_properties != expected_properties:
        raise ValueError("dense-anchor PLY schema or marker is invalid")
    if (
        vertex_count is None
        or vertex_count <= 0
        or metadata_gaussian_count != vertex_count
        or metadata_anchor_count is None
        or metadata_anchor_count <= 0
    ):
        raise ValueError("dense-anchor PLY count metadata is invalid")
    if expected_anchor_count is not None and metadata_anchor_count != int(
        expected_anchor_count
    ):
        raise ValueError(
            f"dense-anchor PLY anchor_count={metadata_anchor_count} does not "
            f"match capacity anchor count {expected_anchor_count}"
        )
    if len(voxel_sizes) != 1 or not np.isfinite(voxel_sizes[0]) or voxel_sizes[0] <= 0:
        raise ValueError("dense-anchor PLY voxel_size metadata is invalid")
    if expected_voxel_size is not None and not np.isclose(
        voxel_sizes[0], float(expected_voxel_size), rtol=1e-6, atol=1e-9
    ):
        raise ValueError(
            f"dense-anchor PLY voxel_size={voxel_sizes[0]} does not match "
            f"strategy anchor_voxel_size={expected_voxel_size}"
        )
    if (
        len(anchor_budget_hashes) != 1
        or len(anchor_budget_hashes[0]) != 64
        or any(character not in "0123456789abcdef" for character in anchor_budget_hashes[0])
    ):
        raise ValueError("dense-anchor PLY anchor budget SHA256 metadata is invalid")
    if (
        expected_anchor_budget_sha256 is not None
        and anchor_budget_hashes[0] != expected_anchor_budget_sha256.lower()
    ):
        raise ValueError("dense-anchor PLY was built from a different anchor budget")

    expected_size = payload_offset + vertex_count * _DENSE_ANCHOR_INIT_DTYPE.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError("dense-anchor PLY payload size does not match vertex count")
    with path.open("rb") as handle:
        handle.seek(payload_offset)
        vertices = np.fromfile(
            handle, dtype=_DENSE_ANCHOR_INIT_DTYPE, count=vertex_count
        )

    xyz = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=1)
    rgb = np.stack(
        (vertices["red"], vertices["green"], vertices["blue"]), axis=1
    )
    anchor_ids = vertices["anchor_id"].astype(np.int64)
    base_scale = vertices["base_scale"].astype(np.float32)
    if not np.isfinite(xyz).all():
        raise ValueError("dense-anchor PLY contains non-finite coordinates")
    if np.any(anchor_ids < 0) or np.any(anchor_ids >= metadata_anchor_count):
        raise ValueError("dense-anchor PLY contains out-of-range anchor IDs")
    if not np.all(np.isfinite(base_scale) & (base_scale > 0)):
        raise ValueError("dense-anchor PLY contains invalid base scales")
    return (
        np.ascontiguousarray(xyz, dtype=np.float32),
        np.ascontiguousarray(rgb, dtype=np.uint8),
        np.ascontiguousarray(anchor_ids),
        np.ascontiguousarray(base_scale),
    )
