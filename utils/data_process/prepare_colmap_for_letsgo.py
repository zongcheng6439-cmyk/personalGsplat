#!/usr/bin/env python3
"""Repair and validate a COLMAP text model for the gsplat simple trainer.

Some SLAM exports use ``No Content`` as the second line of every COLMAP image
record and store the initialization cloud only as ``extracted.ply``.  COLMAP
does not accept that layout.  This script replaces the placeholder with one
untriangulated observation (``0 0 -1``) and creates an empty ``points3D.txt``;
the customized gsplat loader then obtains initialization points from
``extracted.ply``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


PLACEHOLDERS = {"no content", "none", "null"}


def is_image_record(line: str) -> bool:
    """Return whether a non-comment line looks like a COLMAP image pose."""
    fields = line.split()
    if len(fields) != 10:
        return False
    try:
        int(fields[0])
        for value in fields[1:8]:
            float(value)
        int(fields[8])
    except ValueError:
        return False
    return bool(fields[9])


def normalize_images_text(text: str) -> tuple[str, list[str], int]:
    """Return valid COLMAP images.txt content, image names and replacement count."""
    output: list[str] = []
    image_names: list[str] = []
    waiting_for_points = False
    replacements = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("#"):
            output.append(raw_line.rstrip("\r\n"))
            continue

        if is_image_record(line):
            if waiting_for_points:
                output.append("0 0 -1")
                replacements += 1
            output.append(line)
            image_names.append(line.split(maxsplit=9)[9])
            waiting_for_points = True
            continue

        if not waiting_for_points:
            if line:
                raise ValueError(f"Unexpected line outside an image record: {line!r}")
            continue

        if not line or line.lower() in PLACEHOLDERS:
            output.append("0 0 -1")
            replacements += 1
        else:
            fields = line.split()
            if len(fields) % 3:
                raise ValueError(f"Invalid POINTS2D line: {line!r}")
            output.append(line)
        waiting_for_points = False

    if waiting_for_points:
        output.append("0 0 -1")
        replacements += 1
    if not image_names:
        raise ValueError("No COLMAP image records found")
    if len(image_names) != len(set(image_names)):
        raise ValueError("Duplicate image names found in images.txt")
    return "\n".join(output) + "\n", image_names, replacements


def find_matching_file(root: Path, relative_image: Path, suffix: str) -> Path | None:
    candidates = (
        root / relative_image,
        root / relative_image.with_suffix(suffix),
        root / f"{relative_image.stem}{suffix}",
    )
    return next((path for path in candidates if path.is_file()), None)


def validate_assets(data_dir: Path, image_names: list[str], require_depths: bool) -> None:
    specifications = [
        ("image", data_dir / "images", None),
        ("mask", data_dir / "masks", ".png"),
        ("weight map", data_dir / "weight_map", ".npy"),
    ]
    if require_depths:
        specifications.append(("depth", data_dir / "depths_render_color", ".png"))

    failures: list[tuple[str, str]] = []
    for image_name in image_names:
        relative = Path(image_name)
        for label, root, suffix in specifications:
            path = root / relative if suffix is None else find_matching_file(root, relative, suffix)
            if path is None or not path.is_file():
                failures.append((label, image_name))

    if failures:
        examples = ", ".join(f"{label}:{name}" for label, name in failures[:10])
        raise FileNotFoundError(
            f"Missing {len(failures)} required files. First entries: {examples}"
        )
    print(
        f"Assets OK: {len(image_names)} images, masks and weight maps"
        + (", including depth maps" if require_depths else "")
    )



def atomic_write(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(content)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Back up and repair sparse/0/images.txt, and add points3D.txt.",
    )
    parser.add_argument(
        "--require-depths",
        action="store_true",
        help="Also require depths_render_color files (needed with --depth-loss).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    sparse_dir = data_dir / "sparse" / "0"
    cameras_path = sparse_dir / "cameras.txt"
    images_path = sparse_dir / "images.txt"
    extracted_path = sparse_dir / "extracted.ply"
    for required in (cameras_path, images_path, extracted_path):
        if not required.is_file():
            parser.error(f"Required file does not exist: {required}")

    source_text = images_path.read_text(encoding="utf-8-sig")
    normalized_text, image_names, replacements = normalize_images_text(source_text)
    validate_assets(data_dir, image_names, args.require_depths)

    has_points_model = (sparse_dir / "points3D.bin").is_file() or (
        sparse_dir / "points3D.txt"
    ).is_file()
    needs_images_update = normalized_text != source_text.replace("\r\n", "\n")
    print(
        f"COLMAP model: {len(image_names)} images; "
        f"{replacements} empty/placeholder observation lines need normalization"
    )

    if not args.apply:
        print(
            "Dry run only. Re-run with --apply to update images.txt and create "
            "points3D.txt."
        )
        return

    if needs_images_update:
        backup = images_path.with_name(images_path.name + ".pre_simple_trainer.bak")
        if not backup.exists():
            shutil.copy2(images_path, backup)
            print(f"Backup: {backup}")
        atomic_write(images_path, normalized_text)
        print(f"Updated: {images_path}")
    else:
        print(f"Already normalized: {images_path}")

    if not has_points_model:
        (sparse_dir / "points3D.txt").touch()
        print(f"Created: {sparse_dir / 'points3D.txt'}")
    else:
        print("Standard points3D model already exists")


if __name__ == "__main__":
    main()
"""
python prepare_colmap_for_simple_trainer.py \
    --data-dir /media/zc/SSD/GS/letsgo/ \
    --apply  --require-depths
"""