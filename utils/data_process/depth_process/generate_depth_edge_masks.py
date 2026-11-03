"""Filter depth edges in LA or uint16 PNGs; preserve encoding and record format.

From the repository root:
python utils/data_process/depth_process/generate_depth_edge_masks.py --data-dir /path/to/scene --input-dir depth --depth-tag uint16 --depth-scale 1000
Input: PNG depth directory. Output: depths_render_color_edge/ and depth_format.json.
Replace /path/to/scene with the actual scene directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))
from datasets.depth_io import read_depth


def resolve_dir(data_dir: Path, path: Path | None, default: str) -> Path:
    path = path or Path(default)
    return path if path.is_absolute() else data_dir / path


def depth_edges(
    depth: np.ndarray,
    valid: np.ndarray,
    abs_threshold: float,
    relative_threshold: float,
) -> np.ndarray:
    """Mark both pixels of horizontal/vertical depth discontinuities."""
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
    depth_tag: str = "la",
    depth_scale: float = 1000.0,
) -> tuple[bool, int, int]:
    """Process one depth map and return written, removed, valid counts."""
    target = output_dir / source.relative_to(input_dir)
    if target.is_file() and not overwrite:
        return False, 0, 0

    depth, valid = read_depth(source, depth_tag, depth_scale)
    edge = depth_edges(depth, valid, abs_threshold, relative_threshold)
    edge = cv2.dilate(edge.astype(np.uint8), kernel).astype(bool)
    filtered = valid & ~edge
    with Image.open(source) as image:
        raw = np.array(image)
    if depth_tag == "la":
        raw[..., 1] = filtered.astype(np.uint8) * 255
    else:
        raw = np.where(filtered, raw, 0).astype(np.uint16)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(raw).save(target)
    valid_count = int(valid.sum())
    return True, valid_count - int(filtered.sum()), valid_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default='/media/zc/SSD/ar/ABXBJQ_BJ/B3/section_0907/')
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--depth-tag", choices=("la", "uint16"), default="la")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
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

    # Never reuse outputs made with a different encoding, unit, or filter.
    metadata = dict(tag=args.depth_tag, scale=args.depth_scale,
                    source=str(input_dir.resolve()), abs_threshold=args.abs_threshold,
                    relative_threshold=args.relative_threshold, dilation_kernel=args.dilation_kernel)
    config_path = output_dir / "depth_format.json"
    if not np.isfinite(args.depth_scale) or args.depth_scale <= 0:
        parser.error("depth-scale must be finite and positive")
    if config_path.exists():
        if json.loads(config_path.read_text()) != metadata and not args.overwrite:
            parser.error("Depth settings changed; use --overwrite to regenerate filtered depths")
    elif output_dir.exists() and any(output_dir.rglob("*.png")) and not args.overwrite:
        parser.error("Existing depths have no format metadata; use --overwrite to regenerate")
    # Validate the format before submitting work or replacing the metadata.
    read_depth(paths[0], args.depth_tag, args.depth_scale)
    if config_path.exists():
        config_path.unlink()
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
                args.depth_tag,
                args.depth_scale,
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

    config_path.write_text(json.dumps(metadata, indent=2) + "\n")
    ratio = removed / max(valid_total, 1)
    print(
        f"Saved to {output_dir}: written={written}, skipped={skipped}, "
        f"removed={ratio:.2%} of valid depth pixels"
    )


if __name__ == "__main__":
    main()
