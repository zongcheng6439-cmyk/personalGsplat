#!/usr/bin/env python3
"""Prepare masks, depths, weight maps and anchors for the training pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def images_below(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def matching_file(root: Path, relative: Path, suffix: str = ".png") -> Path | None:
    candidates = (
        root / relative,
        root / relative.with_suffix(suffix),
        root / f"{relative.stem}{suffix}",
    )
    return next((path for path in candidates if path.is_file()), None)


def missing_inputs(
    images: list[Path], image_root: Path, target_root: Path, suffix: str = ".png"
) -> list[Path]:
    return [
        image.relative_to(image_root)
        for image in images
        if matching_file(target_root, image.relative_to(image_root), suffix) is None
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default='/media/zc/SSD/GS/letsgo/')
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--depth-dir",
        default="depths_render_color",
        help="raw LA depth PNG directory relative to data-dir",
    )
    parser.add_argument("--depth-edge-abs-threshold", type=float, default=0.2)
    parser.add_argument("--depth-edge-relative-threshold", type=float, default=0.02)
    parser.add_argument("--depth-edge-dilation-kernel", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--target-gs", type=int, default=4_000_000)
    parser.add_argument("--voxel-size", type=float, default=0.1)
    parser.add_argument("--anchor-ply", type=Path)
    parser.add_argument("--plane-anchor-ply", type=Path)
    parser.add_argument("--cgal-plane-ply", type=Path)
    parser.add_argument("--skip-mask-generation", action="store_true")
    parser.add_argument("--overwrite-masks", action="store_true")
    parser.add_argument("--overwrite-derived", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    image_root = data_dir / "images"
    mask_root = data_dir / "masks"
    depth_root = data_dir / args.depth_dir
    filtered_depth_root = data_dir / "depths_render_color_edge"
    weight_root = data_dir / "weight_map"
    anchor_ply = args.anchor_ply.resolve() if args.anchor_ply else None
    plane_anchor_ply = (
        args.plane_anchor_ply.resolve() if args.plane_anchor_ply else None
    )
    cgal_plane_ply = (
        args.cgal_plane_ply.resolve()
        if args.cgal_plane_ply
        else data_dir / "cgal_planes.ply"
    )
    images = images_below(image_root)
    if not images or not (data_dir / "sparse").is_dir():
        parser.error(f"Expected populated images/ and sparse/ under {data_dir}")

    missing_masks = missing_inputs(images, image_root, mask_root)
    if (missing_masks or args.overwrite_masks) and not args.skip_mask_generation:
        command = [
            sys.executable,
            str(Path(__file__).with_name("remove_person_v1.py")),
            "--project_dir",
            str(data_dir),
            "--device",
            args.device,
            "--batch_size",
            "4",
        ]
        if args.overwrite_masks:
            command.append("--overwrite")
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        missing_masks = missing_inputs(images, image_root, mask_root)

    missing_depths = missing_inputs(images, image_root, depth_root)
    if missing_masks or missing_depths:
        for label, missing in (("mask", missing_masks), ("depth", missing_depths)):
            if missing:
                print(f"Missing {label} files ({len(missing)}), first entries:")
                for path in missing[:10]:
                    print(f"  {path}")
        raise SystemExit("Training inputs are incomplete")

    missing_filtered_depths = missing_inputs(
        images, image_root, filtered_depth_root
    )
    if missing_filtered_depths or args.overwrite_derived:
        command = [
            sys.executable,
            str(
                Path(__file__).parent
                / "depth_process"
                / "generate_depth_edge_masks.py"
            ),
            "--data-dir",
            str(data_dir),
            "--input-dir",
            args.depth_dir,
            "--abs-threshold",
            str(args.depth_edge_abs_threshold),
            "--relative-threshold",
            str(args.depth_edge_relative_threshold),
            "--dilation-kernel",
            str(args.depth_edge_dilation_kernel),
            "--workers",
            str(args.workers),
        ]
        if args.overwrite_derived:
            command.append("--overwrite")
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        missing_filtered_depths = missing_inputs(
            images, image_root, filtered_depth_root
        )
    if missing_filtered_depths:
        raise SystemExit(
            f"Missing {len(missing_filtered_depths)} filtered depth maps"
        )

    missing_weights = missing_inputs(images, image_root, weight_root, ".npy")
    if missing_weights or args.overwrite_derived:
        command = [
            sys.executable,
            str(Path(__file__).with_name("generate_weight_maps.py")),
            "--project_dir",
            str(data_dir),
            "--workers",
            str(args.workers),
        ]
        if args.overwrite_derived:
            command.append("--overwrite")
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        missing_weights = missing_inputs(images, image_root, weight_root, ".npy")
    if missing_weights:
        raise SystemExit(f"Missing {len(missing_weights)} weight maps")

    if anchor_ply is not None and (
        not anchor_ply.is_file() or args.overwrite_derived
    ):
        command = [
            sys.executable,
            str(Path(__file__).with_name("build_voxel_anchor_budget.py")),
            "--data-dir",
            str(data_dir),
            "--output",
            str(anchor_ply),
            "--data-factor",
            str(args.data_factor),
            "--voxel-size",
            str(args.voxel_size),
            "--target-gs",
            str(args.target_gs),
            "--min-target-per-voxel",
            "1",
            "--frame-stride",
            "4",
        ]
        if args.overwrite_derived:
            command.append("--overwrite")
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    if plane_anchor_ply is not None and (
        not plane_anchor_ply.is_file() or args.overwrite_derived
    ):
        point_ply = data_dir / "sparse" / "0" / "extracted.ply"
        if not point_ply.is_file():
            raise FileNotFoundError(
                f"voxel-plane anchors require initialization PLY: {point_ply}"
            )
        command = [
            sys.executable,
            str(
                Path(__file__).parent
                / "cgal_plane_segmentation"
                / "run.py"
            ),
            "--input",
            str(point_ply),
            "--output",
            str(cgal_plane_ply),
            "--anchor-output",
            str(plane_anchor_ply),
            "--anchor-voxel-size",
            str(args.voxel_size),
            "--overwrite",
        ]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    print(
        f"Ready: {len(images)} images, {len(images)} masks, "
        f"{len(images)} raw depth maps, {len(images)} filtered depth maps, "
        f"{len(images)} weight maps, "
        f"anchor={anchor_ply if anchor_ply is not None else 'skipped'}, "
        f"plane_anchor={plane_anchor_ply if plane_anchor_ply is not None else 'skipped'}"
    )


if __name__ == "__main__":
    main()
