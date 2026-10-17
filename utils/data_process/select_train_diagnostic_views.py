"""Select fixed plane-heavy and detail-heavy training views.

The script projects reliable CGAL voxel-plane anchors into every training view,
uses the training depth map for occlusion filtering, and ranks views by visible
plane coverage. Detail views are ranked by their valid-pixel weight-map mean.

It writes a reusable JSON list, text lists, masks/overlays for the selected
plane views, and two contact sheets for manual inspection.

Parking01 example::

    python utils/data_process/select_train_diagnostic_views.py \
        --data-dir /media/zc/SSD/GS/lutu/parking01 \
        --plane-anchor-ply /media/zc/SSD/GS/lutu/parking01/cgal_voxel_plane_anchors.ply \
        --output-dir /media/zc/SSD/GS/lutu/parking01/train_diagnostic_views \
        --data-factor 1 --overwrite
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from plyfile import PlyData


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples"))
from datasets.colmap_adc import Dataset, Parser  # noqa: E402


def read_plane_anchors(path: Path) -> np.ndarray:
    vertex = PlyData.read(str(path))["vertex"]
    names = vertex.data.dtype.names or ()
    required = ("x", "y", "z", "is_plane")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"anchor PLY is missing properties: {missing}")
    plane = np.asarray(vertex["is_plane"]).astype(bool)
    xyz = np.stack([np.asarray(vertex[name]) for name in ("x", "y", "z")], 1)
    xyz = np.asarray(xyz[plane], dtype=np.float32)
    if len(xyz) == 0 or not np.isfinite(xyz).all():
        raise ValueError("anchor PLY contains no finite planar anchors")
    return xyz


def resize_tensor(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        values[None, None].float(), size=size, mode="nearest"
    )[0, 0]


@torch.no_grad()
def project_plane_mask(
    plane_xyz: torch.Tensor,
    data: dict,
    downsample: int,
    abs_tolerance: float,
    relative_tolerance: float,
    dilation_kernel: int,
    chunk_size: int,
) -> tuple[np.ndarray, float]:
    image = data["image"]
    height, width = image.shape[:2]
    small_height = math.ceil(height / downsample)
    small_width = math.ceil(width / downsample)
    device = plane_xyz.device

    depth = resize_tensor(data["depth_gs"], (small_height, small_width)).to(device)
    depth_valid = resize_tensor(
        data["depth_gs_mask"], (small_height, small_width)
    ).bool().to(device)
    image_valid = resize_tensor(
        data["mask"], (small_height, small_width)
    ).bool().to(device)
    valid_pixels = depth_valid & image_valid & torch.isfinite(depth) & (depth > 0)

    world_to_camera = torch.linalg.inv(data["camtoworld"].to(device))
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    K = data["K"].to(device)
    flat_mask = torch.zeros(
        small_height * small_width, dtype=torch.bool, device=device
    )
    flat_depth = depth.flatten()
    flat_valid = valid_pixels.flatten()

    for start in range(0, len(plane_xyz), chunk_size):
        xyz = plane_xyz[start : start + chunk_size]
        camera_xyz = xyz @ rotation.T + translation
        z = camera_xyz[:, 2]
        positive = z > 1e-4
        if not positive.any():
            continue
        camera_xyz = camera_xyz[positive]
        z = z[positive]
        u = (K[0, 0] * camera_xyz[:, 0] / z + K[0, 2]) / downsample
        v = (K[1, 1] * camera_xyz[:, 1] / z + K[1, 2]) / downsample
        x = u.round().long()
        y = v.round().long()
        inside = (x >= 0) & (x < small_width) & (y >= 0) & (y < small_height)
        if not inside.any():
            continue
        x, y, z = x[inside], y[inside], z[inside]
        pixel_ids = y * small_width + x
        target_depth = flat_depth[pixel_ids]
        tolerance = torch.maximum(
            torch.full_like(target_depth, abs_tolerance),
            target_depth * relative_tolerance,
        )
        visible = flat_valid[pixel_ids] & ((z - target_depth).abs() <= tolerance)
        flat_mask[pixel_ids[visible]] = True

    mask = flat_mask.view(small_height, small_width).cpu().numpy().astype(np.uint8)
    if dilation_kernel > 1:
        kernel = np.ones((dilation_kernel, dilation_kernel), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel)
    valid = image_valid.cpu().numpy()
    mask = mask.astype(bool) & valid
    ratio = float(mask.sum() / max(valid.sum(), 1))
    return mask, ratio


def detail_score(data: dict) -> float:
    weight = data["weight_map"].numpy()
    valid = data["mask"].numpy() & np.isfinite(weight)
    return float(weight[valid].mean()) if valid.any() else 0.0


def select_diverse(
    records: list[dict],
    score_key: str,
    count: int,
    excluded: set[int],
    min_frame_gap: int,
) -> list[dict]:
    ranked = sorted(records, key=lambda row: (-row[score_key], row["parser_index"]))
    for gap in range(min_frame_gap, -1, -1):
        selected: list[dict] = []
        for row in ranked:
            index = row["parser_index"]
            if index in excluded:
                continue
            if any(abs(index - item["parser_index"]) <= gap for item in selected):
                continue
            selected.append(row)
            if len(selected) == count:
                return selected
    raise RuntimeError(f"cannot select {count} views from {len(records)} candidates")


def save_mask_and_overlay(
    output_dir: Path,
    image_name: str,
    image: np.ndarray,
    small_mask: np.ndarray,
) -> None:
    height, width = image.shape[:2]
    mask = cv2.resize(
        small_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    relative = Path(image_name).with_suffix(".png")
    mask_path = output_dir / "plane_masks" / relative
    overlay_path = output_dir / "plane_overlays" / relative
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
    overlay = np.asarray(image, dtype=np.uint8).copy()
    overlay[mask] = (
        0.45 * overlay[mask] + 0.55 * np.array([30, 100, 255])
    ).astype(np.uint8)
    Image.fromarray(overlay).save(overlay_path)


def make_contact_sheet(
    path: Path,
    selected: list[dict],
    dataset: Dataset,
    title_key: str,
    overlay_root: Path | None = None,
) -> None:
    columns, tile_width, tile_height, label_height = 10, 240, 150, 24
    rows = math.ceil(len(selected) / columns)
    canvas = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    for slot, row in enumerate(selected):
        if overlay_root is None:
            image = Image.fromarray(dataset[row["dataset_index"]]["image"].numpy().astype(np.uint8))
        else:
            overlay_path = overlay_root / Path(row["image_name"]).with_suffix(".png")
            image = Image.open(overlay_path).convert("RGB")
        image.thumbnail((tile_width, tile_height), resampling)
        x = (slot % columns) * tile_width
        y = (slot // columns) * (tile_height + label_height)
        canvas.paste(image, (x + (tile_width - image.width) // 2, y))
        label = f"{Path(row['image_name']).name} {row[title_key]:.3f}"
        draw.text((x + 3, y + tile_height + 4), label, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--plane-anchor-ply", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--num-plane", type=int, default=50)
    parser.add_argument("--num-detail", type=int, default=50)
    parser.add_argument("--projection-downsample", type=int, default=4)
    parser.add_argument("--depth-abs-tolerance", type=float, default=0.15)
    parser.add_argument("--depth-relative-tolerance", type=float, default=0.02)
    parser.add_argument("--mask-dilation", type=int, default=3)
    parser.add_argument("--projection-chunk-size", type=int, default=500000)
    parser.add_argument("--min-frame-gap", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=0, help="0 uses all images")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir / "train_diagnostic_views"
    json_path = output_dir / "train_diagnostic_views.json"
    if json_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {json_path}; pass --overwrite")
    if min(args.data_factor, args.num_plane, args.num_detail) <= 0:
        raise ValueError("data-factor and selection counts must be positive")
    if args.projection_downsample <= 0 or args.projection_chunk_size <= 0:
        raise ValueError("projection sizes must be positive")
    if args.mask_dilation <= 0 or args.mask_dilation % 2 == 0:
        raise ValueError("mask-dilation must be a positive odd number")
    if min(args.depth_abs_tolerance, args.depth_relative_tolerance) < 0:
        raise ValueError("depth tolerances must be non-negative")

    dataset_parser = Parser(
        data_dir=str(args.data_dir),
        factor=args.data_factor,
        normalize=False,
        test_every=0,
        load_exposure=False,
    )
    dataset = Dataset(
        dataset_parser,
        split="train",
        load_depths=True,
        load_weight_maps=True,
    )
    image_count = len(dataset) if args.max_images <= 0 else min(len(dataset), args.max_images)
    if image_count < args.num_plane + args.num_detail:
        raise ValueError(
            f"need at least {args.num_plane + args.num_detail} images, got {image_count}"
        )

    device = torch.device(args.device)
    plane_xyz = torch.from_numpy(read_plane_anchors(args.plane_anchor_ply)).to(device)
    print(f"Projecting {len(plane_xyz):,} planar anchors into {image_count:,} views on {device}")
    records: list[dict] = []
    for dataset_index in range(image_count):
        data = dataset[dataset_index]
        _, plane_ratio = project_plane_mask(
            plane_xyz,
            data,
            args.projection_downsample,
            args.depth_abs_tolerance,
            args.depth_relative_tolerance,
            args.mask_dilation,
            args.projection_chunk_size,
        )
        records.append(
            {
                "image_name": dataset.image_names[dataset_index],
                "dataset_index": dataset_index,
                "parser_index": int(dataset.indices[dataset_index]),
                "plane_ratio": plane_ratio,
                "detail_score": detail_score(data),
            }
        )
        if (dataset_index + 1) % 25 == 0 or dataset_index + 1 == image_count:
            print(f"Scored {dataset_index + 1}/{image_count} views", flush=True)

    plane_views = select_diverse(
        records, "plane_ratio", args.num_plane, set(), args.min_frame_gap
    )
    plane_ids = {row["parser_index"] for row in plane_views}
    detail_views = select_diverse(
        records, "detail_score", args.num_detail, plane_ids, args.min_frame_gap
    )
    selected_plane_ids = {row["dataset_index"] for row in plane_views}
    selected_detail_ids = {row["dataset_index"] for row in detail_views}
    for row in records:
        row["group"] = (
            "plane" if row["dataset_index"] in selected_plane_ids else
            "detail" if row["dataset_index"] in selected_detail_ids else None
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    for row in plane_views:
        data = dataset[row["dataset_index"]]
        small_mask, _ = project_plane_mask(
            plane_xyz,
            data,
            args.projection_downsample,
            args.depth_abs_tolerance,
            args.depth_relative_tolerance,
            args.mask_dilation,
            args.projection_chunk_size,
        )
        save_mask_and_overlay(
            output_dir, row["image_name"], data["image"].numpy(), small_mask
        )

    make_contact_sheet(
        output_dir / "plane_views_contact_sheet.jpg",
        plane_views,
        dataset,
        "plane_ratio",
        output_dir / "plane_overlays",
    )
    make_contact_sheet(
        output_dir / "detail_views_contact_sheet.jpg",
        detail_views,
        dataset,
        "detail_score",
    )

    result = {
        "version": 1,
        "data_dir": str(args.data_dir.resolve()),
        "plane_anchor_ply": str(args.plane_anchor_ply.resolve()),
        "selection": {
            "plane": plane_views,
            "detail": detail_views,
        },
        "config": {
            "data_factor": args.data_factor,
            "projection_downsample": args.projection_downsample,
            "depth_abs_tolerance": args.depth_abs_tolerance,
            "depth_relative_tolerance": args.depth_relative_tolerance,
            "mask_dilation": args.mask_dilation,
            "min_frame_gap": args.min_frame_gap,
        },
        "all_views": records,
    }
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    for name, rows in (("plane", plane_views), ("detail", detail_views)):
        (output_dir / f"{name}_images.txt").write_text(
            "".join(f"{row['image_name']}\n" for row in rows)
        )
    print(
        f"Saved {len(plane_views)} plane and {len(detail_views)} detail views to {output_dir}"
    )


if __name__ == "__main__":
    main()
