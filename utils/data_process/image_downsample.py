#!/usr/bin/env python3
"""Resize an image tree in parallel while preserving relative paths."""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from common import find_images


def resize_one(source, *, source_root, output_root, factor, overwrite):
    destination = output_root / source.relative_to(source_root)
    if (
        destination.is_file()
        and not overwrite
        and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns
    ):
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        width, height = image.size
        size = (max(1, round(width / factor)), max(1, round(height / factor)))
        image.resize(size, Image.Resampling.LANCZOS).save(destination, quality=95)
    return "written"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path)
    parser.add_argument("--factor", type=float, default=2.0)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.factor <= 0:
        parser.error("--factor must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    source = args.src.resolve()
    factor_name = f"{args.factor:g}"
    destination = (args.dst or source.with_name(f"{source.name}_{factor_name}_png")).resolve()
    if source == destination:
        parser.error("--src and --dst must be different")

    images = find_images(source)
    resize = partial(
        resize_one,
        source_root=source,
        output_root=destination,
        factor=args.factor,
        overwrite=args.overwrite,
    )

    counts = {"written": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for status in tqdm(executor.map(resize, images), total=len(images)):
            counts[status] += 1

    print(
        f"Images: {len(images)}, written: {counts['written']}, "
        f"skipped: {counts['skipped']}, output: {destination}"
    )


if __name__ == "__main__":
    main()
