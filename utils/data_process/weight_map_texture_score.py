"""Precompute per-image texture scores and 60--120 sampling counts from NPY maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def score_weight_map(weight_map: np.ndarray, top_fraction: float = 0.1) -> float:
    """Combine whole-map texture coverage with the strongest local responses."""
    values = np.asarray(weight_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("weight_map must be a non-empty HxW array")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    values = np.maximum(
        np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    flattened = values.ravel()
    count = max(1, int(np.ceil(flattened.size * top_fraction)))
    top_mean = float(np.partition(flattened, -count)[-count:].mean())
    return 0.65 * float(values.mean()) + 0.35 * top_mean


def normalize_scores(raw_scores: list[float]) -> np.ndarray:
    """Robustly normalize all image scores to [0, 1]."""
    scores = np.asarray(raw_scores, dtype=np.float64)
    minimum, maximum = float(scores.min()), float(scores.max())
    if maximum <= minimum + 1e-12:
        return np.zeros(scores.shape, dtype=np.float32)
    low, high = np.percentile(scores, [5.0, 95.0])
    if high <= low + 1e-12:
        return ((scores - minimum) / (maximum - minimum)).astype(np.float32)
    normalized = 0.05 + 0.90 * (scores - low) / (high - low)
    below, above = scores < low, scores > high
    normalized[below] = 0.05 * (scores[below] - minimum) / max(
        low - minimum, 1e-12
    )
    normalized[above] = 0.95 + 0.05 * (scores[above] - high) / max(
        maximum - high, 1e-12
    )
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=Path, required=True)
    parser.add_argument("--weight_map_dir", type=Path, default=Path("weight_map"))
    parser.add_argument("--output", type=Path, default=Path("texture_score.json"))
    parser.add_argument("--top_fraction", type=float, default=0.1)
    parser.add_argument("--min_samples", type=int, default=60)
    parser.add_argument("--max_samples", type=int, default=120)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    if args.min_samples <= 0 or args.max_samples < args.min_samples:
        parser.error("require 0 < min_samples <= max_samples")
    map_root = args.project_dir / args.weight_map_dir
    map_paths = sorted(map_root.rglob("*.npy"))
    if not map_paths:
        parser.error(f"no NPY weight maps found below {map_root}")

    raw_scores = []
    for path in map_paths:
        weight_map = np.squeeze(np.load(path))
        if weight_map.ndim == 3:
            weight_map = weight_map.mean(axis=-1)
        raw_scores.append(score_weight_map(weight_map, args.top_fraction))

    scores = normalize_scores(raw_scores)
    span = args.max_samples - args.min_samples
    sample_counts = np.rint(args.min_samples + scores * span).astype(np.int64)
    sample_counts = np.clip(sample_counts, args.min_samples, args.max_samples)

    result = []
    for path, raw_score, score, count in zip(
        map_paths, raw_scores, scores, sample_counts
    ):
        relative = path.relative_to(map_root)
        result.append(
            {
                "name": relative.as_posix(),
                "weight_map": relative.as_posix(),
                "raw_score": float(raw_score),
                "score": float(score),
                "sampling_count": int(count),
            }
        )
    result.sort(key=lambda row: (-row["score"], row["name"]))

    output = args.project_dir / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved {len(result)} precomputed scores to {output.resolve()}")
    print(f"Total samples per epoch: {sum(row['sampling_count'] for row in result)}")
    for rank, row in enumerate(result[: args.top_k], start=1):
        print(
            f"#{rank:3d} score={row['score']:.6f} "
            f"samples={row['sampling_count']:3d} {row['name']}"
        )


if __name__ == "__main__":
    main()
