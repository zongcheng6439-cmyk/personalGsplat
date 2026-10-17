"""Generate per-pixel weight maps and precomputed image sampling scores."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import find_images
from texture_score import compute_weight_map
from weight_map_texture_score import normalize_scores, score_weight_map


def load_image(path: Path, max_size: int) -> np.ndarray:
    with Image.open(path) as image_file:
        image_file = image_file.convert("RGB")
        if max_size > 0:
            scale = min(1.0, float(max_size) / max(image_file.size))
            if scale < 1.0:
                size = tuple(max(1, round(v * scale)) for v in image_file.size)
                image_file = image_file.resize(size, Image.Resampling.BOX)
        return np.asarray(image_file)


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    with Image.open(path) as mask_file:
        mask_file = mask_file.convert("L")
        if mask_file.size != (width, height):
            mask_file = mask_file.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(mask_file) > 0


def process_image(
    image_path,
    *,
    image_root,
    mask_root,
    output_root,
    max_size,
    color_threshold,
    gamma,
    top_fraction,
    without_mask,
    overwrite,
) -> tuple[str, str, str, float]:
    relative = image_path.relative_to(image_root)
    output_base = output_root / relative
    npy_path = output_base.with_suffix(".npy")
    png_path = output_base.with_suffix(".png")
    mask_path = None if without_mask else mask_root / relative
    if mask_path is not None and not mask_path.is_file():
        raise FileNotFoundError(f"missing mask for {relative}: {mask_path}")

    inputs = [image_path] + ([] if mask_path is None else [mask_path])
    outputs_are_fresh = (
        npy_path.is_file()
        and png_path.is_file()
        and min(npy_path.stat().st_mtime_ns, png_path.stat().st_mtime_ns)
        >= max(path.stat().st_mtime_ns for path in inputs)
    )
    if outputs_are_fresh and not overwrite:
        weight_map = np.load(npy_path)
        return (
            "skipped",
            relative.as_posix(),
            npy_path.relative_to(output_root).as_posix(),
            score_weight_map(weight_map, top_fraction),
        )

    image = load_image(image_path, max_size)
    valid_mask = None
    if mask_path is not None:
        valid_mask = load_mask(mask_path, image.shape[:2])

    weight_map = compute_weight_map(
        image,
        valid_mask=valid_mask,
        color_threshold=color_threshold,
        gamma=gamma,
    )
    output_base.parent.mkdir(parents=True, exist_ok=True)
    # Store at half precision to halve disk usage. Training converts each
    # loaded map to float32 before interpolation and score accumulation.
    np.save(npy_path, weight_map.astype(np.float16))

    peak = float(weight_map.max())
    preview = (
        np.zeros(weight_map.shape, dtype=np.uint8)
        if peak <= 0.0
        else np.round(np.clip(weight_map / peak, 0.0, 1.0) * 255).astype(np.uint8)
    )
    Image.fromarray(preview).save(png_path)
    return (
        "written",
        relative.as_posix(),
        npy_path.relative_to(output_root).as_posix(),
        score_weight_map(weight_map, top_fraction),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, default=Path("images"))
    parser.add_argument("--mask_dir", type=Path, default=Path("masks"))
    parser.add_argument("--output_dir", type=Path, default=Path("weight_map"))
    parser.add_argument(
        "--max_size",
        type=int,
        default=0,
        help="Longest side; 0 preserves the exact training-image resolution.",
    )
    parser.add_argument("--color_threshold", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--top_fraction", type=float, default=0.1)
    parser.add_argument("--min_samples", type=int, default=60)
    parser.add_argument("--max_samples", type=int, default=120)
    parser.add_argument("--score_output", type=Path, default=Path("texture_score.json"))
    parser.add_argument("--without_mask", action="store_true")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not 0.0 < args.top_fraction <= 1.0:
        parser.error("--top_fraction must be in (0, 1]")
    if args.min_samples <= 0 or args.max_samples < args.min_samples:
        parser.error("require 0 < min_samples <= max_samples")

    image_root = args.project_dir / args.image_dir
    mask_root = args.project_dir / args.mask_dir
    output_root = args.project_dir / args.output_dir
    images = find_images(image_root)
    if not images:
        parser.error(f"no images found below {image_root}")

    process = partial(
        process_image,
        image_root=image_root,
        mask_root=mask_root,
        output_root=output_root,
        max_size=args.max_size,
        color_threshold=args.color_threshold,
        gamma=args.gamma,
        top_fraction=args.top_fraction,
        without_mask=args.without_mask,
        overwrite=args.overwrite,
    )
    counts = {"written": 0, "skipped": 0}
    processed = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for status, image_name, weight_name, raw_score in tqdm(
            executor.map(process, images), total=len(images)
        ):
            counts[status] += 1
            processed.append((image_name, weight_name, raw_score))

    normalized = normalize_scores([row[2] for row in processed])
    sample_counts = np.rint(
        args.min_samples
        + normalized * (args.max_samples - args.min_samples)
    ).astype(np.int64)
    sample_counts = np.clip(sample_counts, args.min_samples, args.max_samples)
    score_rows = [
        {
            "name": image_name,
            "weight_map": weight_name,
            "raw_score": float(raw_score),
            "score": float(score),
            "sampling_count": int(sample_count),
        }
        for (image_name, weight_name, raw_score), score, sample_count in zip(
            processed, normalized, sample_counts
        )
    ]
    score_path = args.project_dir / args.score_output
    score_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.write_text(json.dumps(score_rows, indent=2, ensure_ascii=False))
    print(
        f"Weight maps: {len(images)}, written: {counts['written']}, "
        f"skipped: {counts['skipped']}, output: {output_root.resolve()}"
    )
    print(
        f"Image scores: {len(score_rows)}, samples/epoch: {int(sample_counts.sum())}, "
        f"output: {score_path.resolve()}"
    )


if __name__ == "__main__":
    main()
