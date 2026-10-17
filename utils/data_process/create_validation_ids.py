#!/usr/bin/env python3
"""Select 50 low-score and 50 high-score registered images from weight maps.

Run on the server:
    cd /home/zc/code/aibee/gsplat/utils/data_process
    python3 create_validation_ids.py \
        --data-dir /media/zc/SSD/ar/ABXBJQ_BJ/B3/section_0907/ \
        --per-group 50 \
        --texture-threshold 0.1

The JSON list and contact-sheet preview are saved in ``data-dir`` by default.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from depth_process.colmap import read_images_binary


def texture_score(data_dir: Path, name: str, threshold: float) -> int:
    """Count pixels whose generated texture score exceeds the threshold."""
    relative = Path(name).with_suffix(".npy")
    roots = (data_dir / "weight_map", data_dir / "weight_maps")
    candidates = [root / relative for root in roots]
    candidates += [root / relative.name for root in roots]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"No weight map found for {name}")
    weight = np.load(path)
    return int(np.count_nonzero(np.isfinite(weight) & (weight > threshold)))


def camera_group(name: str) -> str:
    """Use the filename prefix before its final frame number as camera ID."""
    return re.sub(r"_\d+$", "", Path(name).stem)


def diverse_extremes(items, count: int, high: bool, excluded=frozenset()):
    """Pick score extremes while suppressing adjacent frames."""
    if count <= 0:
        return []
    candidates = sorted(items, key=lambda item: item[0], reverse=high)
    gap = max(1, len(items) // max(2 * count, 1))
    chosen = []
    for score, name, frame_index in candidates:
        if name in excluded:
            continue
        if all(abs(frame_index - old_index) >= gap for _, _, old_index in chosen):
            chosen.append((score, name, frame_index))
            if len(chosen) == count:
                return chosen
    for item in candidates:
        if item[1] not in excluded and item not in chosen:
            chosen.append(item)
            if len(chosen) == count:
                break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=50)
    parser.add_argument("--texture-threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()

    sparse = args.data_dir / "sparse" / "0"
    if not (sparse / "images.bin").is_file():
        sparse = args.data_dir / "sparse"
    images = read_images_binary(sparse / "images.bin")
    names = sorted(
        image.name
        for image in images.values()
        if (args.data_dir / "images" / image.name).is_file()
    )
    needed = 2 * args.per_group
    if len(names) < needed:
        raise ValueError(f"Need at least {needed} registered images, found {len(names)}")

    groups = defaultdict(list)
    for name in names:
        groups[camera_group(name)].append(name)
    groups = dict(sorted(groups.items()))
    quotas = {
        group: args.per_group // len(groups) + (i < args.per_group % len(groups))
        for i, group in enumerate(groups)
    }

    scored_groups = {}
    for group, group_names in groups.items():
        scored_groups[group] = [
            (texture_score(args.data_dir, name, args.texture_threshold), name, index)
            for index, name in enumerate(group_names)
        ]
    low = [
        item[:2]
        for group, items in scored_groups.items()
        for item in diverse_extremes(items, quotas[group], high=False)
    ]
    low_names = {name for _, name in low}
    high = [
        item[:2]
        for group, items in scored_groups.items()
        for item in diverse_extremes(
            items, quotas[group], high=True, excluded=low_names
        )
    ]
    low.sort()
    high.sort(reverse=True)
    selected = [("plane", score, name) for score, name in low]
    selected += [("high_frequency", score, name) for score, name in high]

    output = args.output or args.data_dir / "validation_images.json"
    preview = args.preview or args.data_dir / "validation_images_preview.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "selection": "camera_balanced_low_and_high_texture_pixel_count",
                "texture_threshold": args.texture_threshold,
                "total_registered_images": len(names),
                "camera_groups": {group: quotas[group] for group in groups},
                "plane_images": [
                    {"name": name, "high_texture_pixels": score}
                    for score, name in low
                ],
                "high_frequency_images": [
                    {"name": name, "high_texture_pixels": score}
                    for score, name in high
                ],
                "images": [name for _, _, name in selected],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    columns, cell_width, image_height, label_height = 10, 240, 160, 24
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * cell_width, rows * (image_height + label_height)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for item, (group, score, name) in enumerate(selected):
        with Image.open(args.data_dir / "images" / name) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((cell_width, image_height))
        x = item % columns * cell_width
        y = item // columns * (image_height + label_height)
        sheet.paste(image, (x + (cell_width - image.width) // 2, y))
        color = "blue" if group == "plane" else "red"
        draw.text((x + 3, y + image_height + 4), f"{name} {score}", fill=color)
    preview.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(preview, quality=88)
    print(f"Selected {args.per_group} plane and {args.per_group} high-frequency images")
    print(f"Saved IDs to {output}")
    print(f"Saved preview to {preview}")


if __name__ == "__main__":
    main()
