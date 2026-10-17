"""Remove depth discontinuities from the alpha mask of LA depth PNGs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def resolve_dir(data_dir: Path, path: Path | None, default: str) -> Path:
    path = path or Path(default)
    return path if path.is_absolute() else data_dir / path


def depth_edges(
    encoded: np.ndarray,
    valid: np.ndarray,
    abs_threshold: float,
    relative_threshold: float,
) -> np.ndarray:
    """Mark both pixels of horizontal/vertical depth discontinuities."""
    value = encoded.astype(np.float32) / 255.0
    depth = np.where(
        value < 0.5,
        value * 20.0,
        5.0 / np.maximum(1.0 - value, 1e-6),
    )
    edge = np.zeros(valid.shape, dtype=bool)

    def mark(a, b, valid_pair, edge_a, edge_b):
        threshold = np.maximum(abs_threshold, relative_threshold * np.minimum(a, b))
        jump = valid_pair & (np.abs(a - b) > threshold)
        edge_a |= jump
        edge_b |= jump

    mark(
        depth[:, :-1],
        depth[:, 1:],
        valid[:, :-1] & valid[:, 1:],
        edge[:, :-1],
        edge[:, 1:],
    )
    mark(
        depth[:-1],
        depth[1:],
        valid[:-1] & valid[1:],
        edge[:-1],
        edge[1:],
    )
    return edge


def process_depth_map(
    source: Path,
    input_dir: Path,
    output_dir: Path,
    abs_threshold: float,
    relative_threshold: float,
    kernel: np.ndarray,
    overwrite: bool,
) -> tuple[bool, int, int]:
    """Process one depth map and return written, removed, valid counts."""
    target = output_dir / source.relative_to(input_dir)
    if target.is_file() and not overwrite:
        return False, 0, 0

    with Image.open(source) as image:
        depth_la = np.asarray(image.convert("LA"), dtype=np.uint8)
    encoded, alpha = depth_la[..., 0], depth_la[..., 1]
    valid = (alpha > 127) & (encoded > 0) & (encoded < 255)
    edge = depth_edges(encoded, valid, abs_threshold, relative_threshold)
    edge = cv2.dilate(edge.astype(np.uint8), kernel).astype(bool)
    filtered = valid & ~edge

    result = np.stack((encoded, filtered.astype(np.uint8) * 255), axis=-1)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="LA").save(target)
    valid_count = int(valid.sum())
    return True, valid_count - int(filtered.sum()), valid_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default='/media/zc/SSD/ar/ABXBJQ_BJ/B3/section_0907/')
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--abs-threshold", type=float, default=0.2)
    parser.add_argument("--relative-threshold", type=float, default=0.02)
    parser.add_argument("--dilation-kernel", type=int, default=7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.abs_threshold < 0 or args.relative_threshold < 0:
        raise ValueError("depth thresholds must be non-negative")
    if args.dilation_kernel < 1 or args.dilation_kernel % 2 == 0:
        raise ValueError("dilation-kernel must be a positive odd number")
    if args.workers < 1:
        raise ValueError("workers must be positive")

    input_dir = resolve_dir(args.data_dir, args.input_dir, "depths_render_color")
    output_dir = resolve_dir(
        args.data_dir, args.output_dir, "depths_render_color_edge"
    )
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("output-dir must differ from input-dir")

    paths = sorted(input_dir.rglob("*.png"))
    if not paths:
        raise FileNotFoundError(f"no PNG depth maps found in {input_dir}")

    kernel = np.ones((args.dilation_kernel, args.dilation_kernel), np.uint8)
    written = skipped = removed = valid_total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_depth_map,
                source,
                input_dir,
                output_dir,
                args.abs_threshold,
                args.relative_threshold,
                kernel,
                args.overwrite,
            )
            for source in paths
        ]
        update_every = max(1, len(futures) // 200)
        for completed, future in enumerate(as_completed(futures), start=1):
            was_written, removed_count, valid_count = future.result()
            written += int(was_written)
            skipped += int(not was_written)
            removed += removed_count
            valid_total += valid_count
            if completed == len(futures) or completed % update_every == 0:
                print(
                    f"\rDepth edge masks: {completed}/{len(futures)} "
                    f"({completed / len(futures):6.2%}) "
                    f"written={written} skipped={skipped}",
                    end="\n" if completed == len(futures) else "",
                    flush=True,
                )

    ratio = removed / max(valid_total, 1)
    print(
        f"Saved to {output_dir}: written={written}, skipped={skipped}, "
        f"removed={ratio:.2%} of valid depth pixels"
    )


if __name__ == "__main__":
    main()
