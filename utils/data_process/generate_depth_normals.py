"""Precompute planar normals from the rendered depth maps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

REPO_DIR = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_DIR / "examples"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from datasets.colmap_adc import Parser, PILtoTorch, warpped_depth_inv  # noqa: E402
from utils import (  # noqa: E402
    depth_to_normal,
    load_depth_normals,
    save_depth_normals,
)


def load_depth(parser: Parser, index: int):
    """Load depth exactly as the training dataset does."""
    with Image.open(parser.image_paths[index]) as image:
        width, height = image.size
    with Image.open(parser.depth_paths[index]) as image:
        encoded = PILtoTorch(image.convert("LA"), (width, height))

    depth = warpped_depth_inv(encoded[0])
    valid = encoded[1].bool() & torch.isfinite(depth)
    depth = torch.nan_to_num(depth)
    camera_id = parser.camera_ids[index]
    K = torch.from_numpy(parser.Ks_dict[camera_id].copy()).float()
    if len(parser.params_dict[camera_id]) > 0:
        x, y, width, height = parser.roi_undist_dict[camera_id]
        depth = depth[y : y + height, x : x + width]
        valid = valid[y : y + height, x : x + width]
    return depth, valid, K


def load_image(parser: Parser, index: int) -> np.ndarray:
    """Load the RGB image with the same undistortion as training."""
    with Image.open(parser.image_paths[index]) as source:
        image = np.asarray(source.convert("RGB"))
    camera_id = parser.camera_ids[index]
    if len(parser.params_dict[camera_id]) > 0:
        image = cv2.remap(
            image,
            parser.mapx_dict[camera_id],
            parser.mapy_dict[camera_id],
            cv2.INTER_CUBIC,
        )
        x, y, width, height = parser.roi_undist_dict[camera_id]
        image = image[y : y + height, x : x + width]
    return image


def save_preview(
    path: Path,
    image: np.ndarray,
    normal: torch.Tensor,
    plane_mask: torch.Tensor,
):
    """Save RGB, colored normals and their mask side by side."""
    normal = normal.cpu().numpy()
    plane_mask = plane_mask.cpu().numpy()
    color = np.round(np.clip(normal * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
    color[~plane_mask] = 0
    mask = np.repeat((plane_mask[..., None] * 255).astype(np.uint8), 3, axis=-1)
    if image.shape[:2] != normal.shape[:2]:
        raise ValueError(f"Preview shapes do not match: {image.shape} vs {normal.shape}")

    height, width = plane_mask.shape
    preview = Image.new("RGB", (width * 3, height + 24))
    preview.paste(Image.fromarray(image), (0, 24))
    preview.paste(Image.fromarray(color), (width, 24))
    preview.paste(Image.fromarray(mask), (width * 2, 24))
    draw = ImageDraw.Draw(preview)
    for x, label in zip((0, width, width * 2), ("RGB", "Normal", "Plane mask")):
        draw.text((x + 5, 5), label, fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default='/data/')
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=10)
    parser.add_argument("--preview_every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_parser = Parser(args.data_dir, factor=args.factor, test_every=0)
    output_dir = args.output_dir or (
        args.data_dir / f"depth_normals_w{args.window_size}_f{args.factor}"
    )
    if not output_dir.is_absolute():
        output_dir = args.data_dir / output_dir
    preview_dir = output_dir / "preview"

    written = skipped = 0
    for index, image_name in enumerate(tqdm(dataset_parser.image_names)):
        relative = Path(image_name).with_suffix(".npz")
        output_path = output_dir / relative
        show = args.preview_every > 0 and index % args.preview_every == 0

        if output_path.is_file() and not args.overwrite:
            skipped += 1
            if show:
                normal, plane_mask = load_depth_normals(output_path)
        else:
            depth, valid, K = load_depth(dataset_parser, index)
            with torch.no_grad():
                normal, plane_mask = depth_to_normal(
                    depth.to(args.device),
                    K.to(args.device),
                    valid.to(args.device),
                    window_size=args.window_size,
                    find_planes=True,
                )
            normal, plane_mask = normal.cpu(), plane_mask.cpu()
            save_depth_normals(output_path, normal, plane_mask)
            written += 1

        if show:
            preview_path = preview_dir / f"{index:06d}_{relative.stem}.png"
            save_preview(
                preview_path,
                load_image(dataset_parser, index),
                normal,
                plane_mask,
            )

    print(f"Saved to {output_dir}: written={written}, skipped={skipped}")


if __name__ == "__main__":
    main()
