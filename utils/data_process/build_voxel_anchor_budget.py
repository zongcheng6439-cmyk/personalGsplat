"""Build one grayscale PLY that stores the target GS count per voxel.

This is the deliberately small first stage of the voxel-budget experiment:

1. Downsample ``sparse/0/extracted.ply`` to one representative point per voxel.
2. Project those voxel points into the training cameras.
3. Average the visible weight-map values for every voxel.
4. Allocate a user-specified total, then quantize excessive count categories.
5. Save only ``x, y, z, red, green, blue`` in one PLY file.

The color is grayscale (``red == green == blue``).  The smallest target count
is black and the largest is white.  The PLY header stores a gray-to-count
lookup table, so :func:`read_target_count_ply` recovers the exact integer
target from the grayscale plus that table.  Pure binary black/white could
represent only two target counts; grayscale supports values in between.
When every voxel has the same target, all points are black because there is no
range to visualize.  The decoder still recovers that one target exactly.

There is no manually configured common per-voxel maximum: the allocated target
is the maximum capacity for that anchor.  A grayscale lookup table in the PLY
header maps each used gray level back to its exact integer capacity.  A value may be
larger than 255; the color-only limit is at most 256 distinct target values in
one file.  If allocation produces more than 256 distinct target counts, 256
representatives are retained and every other count maps to its numerically
nearest representative.  This category quantization may change the final total
GS count; the report records both requested and saved totals and their difference.

For anchor scoring, projection intentionally uses only the pinhole calibration
matrix (``f/fx/fy, cx, cy``). Distortion coefficients are ignored.


"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


def voxel_downsample_with_indices(
    points: np.ndarray, voxel_size: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Return one real point and its input index per world-aligned voxel."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N,3], got {points.shape}")
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be finite and positive")

    finite = np.isfinite(points).all(axis=1)
    if not finite.all():
        raise ValueError(
            f"point cloud contains {int((~finite).sum())} non-finite XYZ points; "
            "clean extracted.ply before preprocessing and training"
        )
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 0:
        raise ValueError("point cloud is empty")

    # The grid boundaries are integer multiples of voxel_size in scene space.
    origin = np.floor(points.min(axis=0) / voxel_size) * voxel_size

    point_keys = np.floor((points - origin) / voxel_size).astype(np.int64)
    # 只保留该格子第一个碰到的点，实现体素降采样
    _, first_index = np.unique(point_keys, axis=0, return_index=True)
    return (
        np.ascontiguousarray(points[first_index]),
        np.ascontiguousarray(first_index, dtype=np.int64),
    )


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return one real input point per occupied, world-aligned voxel."""
    return voxel_downsample_with_indices(points, voxel_size)[0]


def project_visible_voxels(
    xyz: np.ndarray,
    camtoworld: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    depth_tolerance: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project voxel points and retain the front layer at each image pixel."""
    xyz = np.asarray(xyz, dtype=np.float64)
    camtoworld = np.asarray(camtoworld, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape [V,3]")
    if camtoworld.shape != (4, 4) or K.shape != (3, 3):
        raise ValueError("camtoworld and K must be [4,4] and [3,3]")
    if width <= 0 or height <= 0 or depth_tolerance < 0:
        raise ValueError("image size must be positive and depth tolerance non-negative")

    worldtocam = np.linalg.inv(camtoworld)
    camera_xyz = xyz @ worldtocam[:3, :3].T + worldtocam[:3, 3]
    depth = camera_xyz[:, 2]
    homogeneous = camera_xyz @ K.T
    denom = homogeneous[:, 2]

    u = np.full(len(xyz), np.nan, dtype=np.float64)
    v = np.full(len(xyz), np.nan, dtype=np.float64)
    valid_depth = (
        np.isfinite(camera_xyz).all(axis=1)
        & np.isfinite(denom)
        & (depth > 1e-6)
        & (np.abs(denom) > 1e-12)
    )
    u[valid_depth] = homogeneous[valid_depth, 0] / denom[valid_depth]
    v[valid_depth] = homogeneous[valid_depth, 1] / denom[valid_depth]
    in_frame = (
        valid_depth
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    ids = np.flatnonzero(in_frame)
    if len(ids) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    pixel_x = np.floor(u[ids]).astype(np.int64)
    pixel_y = np.floor(v[ids]).astype(np.int64)
    flat_pixel = pixel_y * width + pixel_x
    nearest_depth = np.full(width * height, np.inf, dtype=np.float64)
    np.minimum.at(nearest_depth, flat_pixel, depth[ids])
    visible = depth[ids] <= nearest_depth[flat_pixel] + depth_tolerance
    return ids[visible], pixel_x[visible], pixel_y[visible]


def sample_frame_weights(
    xyz: np.ndarray,
    camtoworld: np.ndarray,
    K: np.ndarray,
    weight_map: np.ndarray,
    depth_tolerance: float,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return voxel IDs and finite weight values sampled from one view."""
    weight_map = np.asarray(weight_map, dtype=np.float32)
    if weight_map.ndim != 2:
        raise ValueError(f"weight_map must be [H,W], got {weight_map.shape}")
    height, width = weight_map.shape
    ids, pixel_x, pixel_y = project_visible_voxels(
        xyz, camtoworld, K, width, height, depth_tolerance
    )
    values = weight_map[pixel_y, pixel_x]
    valid = np.isfinite(values)
    if valid_mask is not None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if valid_mask.shape != weight_map.shape:
            raise ValueError(
                f"valid_mask shape {valid_mask.shape} does not match "
                f"weight_map shape {weight_map.shape}"
            )
        # Invalid/person pixels contribute neither a zero score nor a view to
        # the denominator; they are observations we intentionally ignore.
        valid &= valid_mask[pixel_y, pixel_x]
    return ids[valid], values[valid]


def score_to_priority(
    score: np.ndarray, clip_quantile: float, gamma: float
) -> np.ndarray:
    """Robustly convert non-negative multi-view scores into allocation weights."""
    if not 0 < clip_quantile <= 1:
        raise ValueError("clip_quantile must be in (0,1]")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("gamma must be finite and positive")
    score = np.nan_to_num(
        np.asarray(score, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    score = np.clip(score, 0.0, None)
    positive = score[score > 0]
    if len(positive) == 0:
        raise ValueError("all visible weight-map scores are zero")
    clip_value = float(np.quantile(positive, clip_quantile))
    normalized = np.clip(score / max(clip_value, 1e-12), 0.0, 1.0)
    return np.power(normalized, gamma)


def allocate_integer_budget(
    priority: np.ndarray,
    target_total: int,
    min_count: int = 1,
) -> np.ndarray:
    """Allocate an uncapped integer budget using deterministic largest remainder."""
    priority = np.nan_to_num(
        np.asarray(priority, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    priority = np.clip(priority, 0.0, None)
    if priority.ndim != 1 or len(priority) == 0:
        raise ValueError("priority must be a non-empty 1D array")
    if min_count < 0:
        raise ValueError("min_count must be non-negative")

    count = len(priority)
    minimum_total = count * min_count
    if target_total < minimum_total:
        raise ValueError(
            f"target_total={target_total} is smaller than the required minimum "
            f"{minimum_total}"
        )

    target = np.full(count, min_count, dtype=np.int64)
    remaining = int(target_total - minimum_total)
    if remaining == 0:
        return target

    weights = priority if priority.sum() > 0 else np.ones_like(priority)
    shares = remaining * weights / weights.sum()
    base = np.floor(shares).astype(np.int64)
    target += base
    leftover = int(target_total - target.sum())
    if leftover:
        voxel_ids = np.arange(count, dtype=np.int64)
        fraction = shares - base
        # 后面参数是主排序键，前面是次要键
        order = np.lexsort((voxel_ids, -fraction))
        # 小数部分最高的前leftover个元素多拿 1 份
        target[order[:leftover]] += 1

    if int(target.sum()) != int(target_total):
        raise RuntimeError("integer allocation did not conserve target_total")
    return target


def limit_target_count_categories(
    target_count: np.ndarray, max_categories: int = 256
) -> np.ndarray:
    """Map excess count levels to their nearest retained category."""
    target_count = np.asarray(target_count, dtype=np.int64)
    if target_count.ndim != 1 or len(target_count) == 0:
        raise ValueError("target_count must be a non-empty 1D array")
    if int(target_count.min()) < 0:
        raise ValueError("target_count cannot be negative")
    if max_categories < 2 or max_categories > 256:
        raise ValueError("max_categories must be in [2,256]")
    unique_count, inverse = np.unique(target_count, return_inverse=True)
    if len(unique_count) <= max_categories:
        return target_count.copy()

    representative_indices = np.rint(
        np.linspace(0, len(unique_count) - 1, max_categories)
    ).astype(np.int64)
    representatives = unique_count[representative_indices]

    upper_index = np.searchsorted(representatives, unique_count, side="left")
    upper_index = np.clip(upper_index, 0, len(representatives) - 1)
    lower_index = np.maximum(upper_index - 1, 0)
    lower_distance = unique_count - representatives[lower_index]
    upper_distance = representatives[upper_index] - unique_count
    use_upper = upper_distance < lower_distance
    mapped_count = np.where(
        use_upper,
        representatives[upper_index],
        representatives[lower_index],
    )
    return mapped_count[inverse]


def encode_target_count(
    target_count: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode arbitrary integer counts with an ordered 8-bit grayscale LUT."""
    target_count = np.asarray(target_count, dtype=np.int64)
    if target_count.ndim != 1 or len(target_count) == 0:
        raise ValueError("target_count must be a non-empty 1D array")
    count_levels = np.unique(target_count)
    if int(count_levels[0]) < 0:
        raise ValueError("target_count cannot be negative")
    if len(count_levels) > 256:
        raise ValueError(
            "color-only encoding supports at most 256 distinct target counts"
        )
    if len(count_levels) == 1:
        gray_levels = np.asarray([0], dtype=np.uint8)
    else:
        values = count_levels.astype(np.float64)
        gray_levels = np.rint(
            255.0 * (values - values[0]) / (values[-1] - values[0])
        ).astype(np.int64)
        # Rounding can map nearby counts to the same gray.  Make the levels
        # strictly increasing while staying as close as possible to the
        # count-proportional brightness.
        for index in range(1, len(gray_levels)):
            gray_levels[index] = max(
                gray_levels[index], gray_levels[index - 1] + 1
            )
        gray_levels[-1] = 255
        for index in range(len(gray_levels) - 2, -1, -1):
            gray_levels[index] = min(
                gray_levels[index], gray_levels[index + 1] - 1
            )
        gray_levels = gray_levels.astype(np.uint8)
    level_index = np.searchsorted(count_levels, target_count)
    return gray_levels[level_index], gray_levels, count_levels


def decode_target_count(
    gray: np.ndarray, gray_levels: np.ndarray, count_levels: np.ndarray
) -> np.ndarray:
    """Recover target counts using the grayscale lookup table."""
    gray = np.asarray(gray, dtype=np.int64)
    gray_levels = np.asarray(gray_levels, dtype=np.int64)
    count_levels = np.asarray(count_levels, dtype=np.int64)
    if (
        np.any((gray < 0) | (gray > 255))
        or gray_levels.ndim != 1
        or count_levels.ndim != 1
        or len(gray_levels) == 0
        or len(gray_levels) != len(count_levels)
        or len(gray_levels) > 256
        or np.any((gray_levels < 0) | (gray_levels > 255))
        or np.any(np.diff(gray_levels) <= 0)
        or np.any(np.diff(count_levels) <= 0)
        or int(count_levels[0]) < 0
    ):
        raise ValueError("invalid target-count grayscale lookup table")
    lookup = np.full(256, -1, dtype=np.int64)
    lookup[gray_levels] = count_levels
    decoded = lookup[gray]
    if np.any(decoded < 0):
        raise ValueError("PLY contains a gray level absent from its lookup table")
    return decoded


_PLY_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def write_target_count_ply(
    path: Path,
    xyz: np.ndarray,
    target_count: np.ndarray,
    voxel_size: Optional[float] = None,
) -> None:
    """Write one standard grayscale PLY; no NPZ/CSV/extra PLY is created."""
    path = Path(path)
    xyz = np.asarray(xyz, dtype=np.float32)
    target_count = np.asarray(target_count, dtype=np.int64)
    if xyz.shape != (len(target_count), 3):
        raise ValueError("xyz and target_count lengths do not match")
    gray, gray_levels, count_levels = encode_target_count(target_count)

    vertices = np.empty(len(xyz), dtype=_PLY_DTYPE)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"] = gray
    vertices["green"] = gray
    vertices["blue"] = gray
    lookup_comments = [
        f"comment target_count_lut {int(gray_level)} {int(count_level)}"
        for gray_level, count_level in zip(gray_levels, count_levels)
    ]
    voxel_comment = (
        []
        if voxel_size is None
        else [f"comment voxel_size {float(voxel_size):.17g}"]
    )
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment grayscale stores the target GS count",
            f"comment target_count_total {int(target_count.sum())}",
            *voxel_comment,
            *lookup_comments,
            "comment voxel_id is the zero-based vertex index",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        vertices.tofile(handle)


def read_target_count_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read XYZ and recover target counts only from grayscale plus PLY comments."""
    path = Path(path)
    header_lines = []
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("incomplete PLY header")
            text = line.decode("ascii").strip()
            header_lines.append(text)
            if text == "end_header":
                payload_offset = handle.tell()
                break

    vertex_count = None
    expected_total = None
    gray_levels = []
    count_levels = []
    for line in header_lines:
        fields = line.split()
        if len(fields) == 3 and fields[:2] == ["element", "vertex"]:
            vertex_count = int(fields[2])
        elif len(fields) == 4 and fields[:2] == ["comment", "target_count_lut"]:
            gray_levels.append(int(fields[2]))
            count_levels.append(int(fields[3]))
        elif len(fields) == 3 and fields[:2] == ["comment", "target_count_total"]:
            expected_total = int(fields[2])
    expected_properties = [
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
    ]
    actual_properties = [line for line in header_lines if line.startswith("property ")]
    if header_lines[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError("PLY must use the expected binary little-endian format")
    if actual_properties != expected_properties:
        raise ValueError("PLY vertex property schema does not match the target format")
    if (
        vertex_count is None
        or not gray_levels
        or expected_total is None
    ):
        raise ValueError("PLY is missing target-count encoding metadata")
    expected_size = payload_offset + vertex_count * _PLY_DTYPE.itemsize
    if path.stat().st_size != expected_size:
        raise ValueError("PLY payload size does not match its vertex count")

    with path.open("rb") as handle:
        handle.seek(payload_offset)
        vertices = np.fromfile(handle, dtype=_PLY_DTYPE, count=vertex_count)
    if len(vertices) != vertex_count:
        raise ValueError("PLY vertex payload is truncated")
    if not (
        np.array_equal(vertices["red"], vertices["green"])
        and np.array_equal(vertices["red"], vertices["blue"])
    ):
        raise ValueError("target-count PLY is not grayscale")
    xyz = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=1)
    target_count = decode_target_count(
        vertices["red"],
        np.asarray(gray_levels, dtype=np.int64),
        np.asarray(count_levels, dtype=np.int64),
    )
    if int(target_count.sum()) != expected_total:
        raise ValueError("PLY colors do not match target_count_total")
    return xyz, target_count


def _load_weight_map(path: str, width: int, height: int) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"weight map does not exist: {path}")
    weight = np.squeeze(np.asarray(np.load(path), dtype=np.float32))
    if weight.ndim == 3:
        weight = weight.mean(axis=-1)
    if weight.ndim != 2:
        raise ValueError(f"weight map must be 2D, got {weight.shape}: {path}")
    if weight.shape != (height, width):
        image = Image.fromarray(weight, mode="F")
        weight = np.asarray(
            image.resize((width, height), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    return weight


def _load_valid_mask(path: str, width: int, height: int) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"mask does not exist: {path}")
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != (width, height):
            raise ValueError(
                f"mask/image size mismatch for {path}: {mask.size} != "
                f"{(width, height)}"
            )
        return np.asarray(mask, dtype=np.uint8) > 128


def _find_extracted_ply(data_dir: Path) -> Path:
    # Match colmap_adc.Parser exactly: if sparse/0 exists, that model wins and
    # a PLY next to sparse/ (the other model) must not be reported as the input.
    sparse_root = data_dir / "sparse"
    model_dir = sparse_root / "0" if (sparse_root / "0").is_dir() else sparse_root
    path = model_dir / "extracted.ply"
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"expected the radar/LiDAR point cloud used by the selected COLMAP "
        f"model at {path}"
    )


def _validate_output_path(
    output: Path, point_cloud_path: Path, overwrite: bool
) -> Path:
    """Resolve output and prevent replacing the initialization point cloud."""
    output = output.expanduser().resolve()
    point_cloud_path = point_cloud_path.expanduser().resolve()
    if output == point_cloud_path:
        raise ValueError(
            "--output must not be the input sparse/.../extracted.ply point cloud"
        )
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output}; pass --overwrite to replace it"
        )
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--test-every", type=int, default=100000)
    parser.add_argument("--voxel-size", type=float, default=0.10)
    parser.add_argument("--target-gs", type=int, required=True)
    parser.add_argument("--min-target-per-voxel", type=int, default=1)
    parser.add_argument("--score-gamma", type=float, default=1.0)
    parser.add_argument("--score-clip-quantile", type=float, default=1.0)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--depth-tolerance",
        type=float,
        default=-1.0,
        help="negative means half of voxel_size",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.data_factor <= 0 or args.test_every <= 0 or args.frame_stride <= 0:
        raise ValueError("data_factor, test_every and frame_stride must be positive")
    if args.max_frames < 0:
        raise ValueError("max_frames must be non-negative")
    if args.min_target_per_voxel < 0:
        raise ValueError("min_target_per_voxel must be non-negative")
    example_dir = Path(__file__).resolve().parents[2] / "examples"
    if not example_dir.is_dir():
        raise FileNotFoundError(f"cannot locate gsplat examples at {example_dir}")
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))
    from datasets.colmap_adc import Parser

    data_dir = args.data_dir.resolve()
    point_cloud_path = _find_extracted_ply(data_dir)
    args.output = _validate_output_path(
        args.output, point_cloud_path, args.overwrite
    )
    parser: Any = Parser(
        data_dir=str(data_dir),
        factor=args.data_factor,
        normalize=False,
        test_every=args.test_every,
        load_exposure=False,
    )
    ignored_distortion = [
        int(camera_id)
        for camera_id, params in parser.params_dict.items()
        if len(params) > 0
    ]
    if ignored_distortion:
        print(
            "Ignoring distortion coefficients for anchor projection; using "
            "only f/fx/fy, cx and cy for camera IDs: "
            f"{ignored_distortion}",
            flush=True,
        )

    voxel_xyz = voxel_downsample(parser.points, args.voxel_size)
    voxel_count = len(voxel_xyz)
    minimum_total = voxel_count * args.min_target_per_voxel
    if args.target_gs < minimum_total:
        raise ValueError(
            f"target_gs={args.target_gs} is infeasible for {voxel_count} voxels; "
            f"it must be at least {minimum_total}"
        )
    if args.target_gs < len(parser.points):
        raise ValueError(
            f"target_gs={args.target_gs} is smaller than the "
            f"{len(parser.points)} initial LiDAR Gaussians used by the trainer"
        )

    all_indices = np.arange(len(parser.image_names), dtype=np.int64)
    train_indices = all_indices[all_indices % args.test_every != 0]
    selected = train_indices[:: args.frame_stride]
    if args.max_frames and len(selected) > args.max_frames:
        selected = selected[
            np.linspace(0, len(selected) - 1, args.max_frames, dtype=np.int64)
        ]
    if len(selected) == 0:
        raise ValueError("no training frames were selected")

    score_sum = np.zeros(voxel_count, dtype=np.float64)
    visible_views = np.zeros(voxel_count, dtype=np.int32)
    depth_tolerance = (
        args.voxel_size * 0.5
        if args.depth_tolerance < 0
        else args.depth_tolerance
    )
    for position, index in enumerate(selected, start=1):
        image_path = parser.image_paths[int(index)]
        with Image.open(image_path) as image:
            width, height = image.size
        weight = _load_weight_map(
            parser.weight_paths[int(index)], width=width, height=height
        )
        valid_mask = _load_valid_mask(
            parser.mask_paths[int(index)], width=width, height=height
        )
        camera_id = parser.camera_ids[int(index)]
        ids, values = sample_frame_weights(
            voxel_xyz,
            parser.camtoworlds[int(index)],
            parser.Ks_dict[camera_id],
            weight,
            depth_tolerance,
            valid_mask,
        )
        score_sum[ids] += values
        visible_views[ids] += 1
        if position == 1 or position % 10 == 0 or position == len(selected):
            print(
                f"[{position}/{len(selected)}] {parser.image_names[int(index)]}: "
                f"scored voxels={len(ids):,}",
                flush=True,
            )

    # multi view average score 
    score = np.divide(
        score_sum,
        np.maximum(visible_views, 1),
        out=np.zeros_like(score_sum),
        where=visible_views > 0,
    )
    priority = score_to_priority(
        score, args.score_clip_quantile, args.score_gamma
    )
    target_count = allocate_integer_budget(
        priority,
        args.target_gs,
        min_count=args.min_target_per_voxel,
    )
    category_count_before = int(len(np.unique(target_count)))
    target_count = limit_target_count_categories(target_count, max_categories=256)
    category_count_after = int(len(np.unique(target_count)))
    target_count = np.asarray(target_count, dtype=np.int64)
    if int(target_count.sum()) < len(parser.points):
        raise ValueError(
            "capacity-category quantization reduced the saved total below the "
            f"{len(parser.points)} initial LiDAR Gaussians; increase --target-gs"
        )
    write_target_count_ply(
        args.output, voxel_xyz, target_count, voxel_size=args.voxel_size
    )
    print(f"max gs num in voxel : {target_count.max()}")

    # Immediate color-only round-trip verification.
    saved_xyz, saved_count = read_target_count_ply(args.output)
    if not np.array_equal(saved_xyz, voxel_xyz):
        raise RuntimeError("saved PLY XYZ failed round-trip verification")
    if not np.array_equal(saved_count, target_count):
        raise RuntimeError("saved PLY colors failed target-count verification")
    saved_target_gs = int(saved_count.sum())

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "point_cloud": str(point_cloud_path.resolve()),
                "input_points": int(len(parser.points)),
                "voxels": voxel_count,
                "processed_frames": int(len(selected)),
                "visible_voxel_ratio": float(np.mean(visible_views > 0)),
                "requested_target_gs": int(args.target_gs),
                "stored_target_gs": saved_target_gs,
                "target_gs_delta": saved_target_gs - int(args.target_gs),
                "target_count_min": int(saved_count.min()),
                "target_count_max": int(saved_count.max()),
                "target_categories_before": category_count_before,
                "target_categories_after": category_count_after,
                "verification": "PASS",
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0

"""
Example:

    python -B build_voxel_anchor_budget.py \
        --data-dir /B3/section_0907// \
        --output /B3/section_0907/voxel_target_count.ply \
        --voxel-size 0.10 \
        --target-gs 4000000 --overwrite
"""

if __name__ == "__main__":
    raise SystemExit(main())
