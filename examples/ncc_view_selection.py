"""LiDAR-depth source-view selection for NCC; no SfM feature tracks required."""

import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from tqdm import tqdm


def _cache_signature(dataset, settings):
    parser = dataset.parser
    entries = []
    for index in dataset.indices:
        files = []
        for paths in (parser.image_paths, parser.depth_paths, parser.mask_paths):
            path = Path(paths[index])
            stat = path.stat()
            files.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        camera = parser.camera_ids[index]
        entries.append({
            "name": parser.image_names[index], "files": files,
            "pose": parser.camtoworlds[index].tolist(),
            "K": parser.Ks_dict[camera].tolist(),
            "distortion": np.asarray(parser.params_dict[camera]).tolist(),
        })
    payload = {"version": 1, "factor": parser.factor,
               "settings": settings, "images": entries}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _small_depth(data, max_size):
    depth = data["depth_gs"].cpu().numpy()
    mask = (data["mask"] & data["depth_gs_mask"]).cpu().numpy()
    stride = max(1, math.ceil(max(depth.shape) / max_size))
    # Striding keeps the exact sampled pixel coordinates: x_full = stride*x.
    depth = depth[::stride, ::stride].copy()
    mask = mask[::stride, ::stride] & np.isfinite(depth) & (depth > 0)
    K = data["K"].cpu().numpy().astype(np.float64)
    K[:2] /= stride
    pose = data["camtoworld"].cpu().numpy().astype(np.float64)
    return {"depth": depth, "mask": mask, "K": K,
            "pose": pose, "w2c": np.linalg.inv(pose)}


def _sample_world_points(view, samples):
    y, x = np.nonzero(view["mask"])
    if len(x) > samples:
        ids = np.linspace(0, len(x) - 1, samples, dtype=np.int64)
        x, y = x[ids], y[ids]
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1)
    points = (pixels @ np.linalg.inv(view["K"]).T) * view["depth"][y, x, None]
    return points @ view["pose"][:3, :3].T + view["pose"][:3, 3]


def _depth_overlap(points_world, source, absolute=0.15, relative=0.02):
    if not len(points_world):
        return 0.0
    points = points_world @ source["w2c"][:3, :3].T + source["w2c"][:3, 3]
    z = points[:, 2]
    pixels = points @ source["K"].T
    uv = pixels[:, :2] / np.maximum(z[:, None], 1e-8)
    height, width = source["depth"].shape
    inside = ((z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1)
              & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1))
    # Use nearest depth, not an average mixing foreground and background.
    x = np.rint(uv[:, 0]).clip(0, width - 1).astype(np.int64)
    y = np.rint(uv[:, 1]).clip(0, height - 1).astype(np.int64)
    consistent = np.abs(source["depth"][y, x] - z) <= np.maximum(absolute, relative * z)
    return float((inside & source["mask"][y, x] & consistent).mean())


def _pose_candidates(poses, count, min_distance, max_distance, max_angle):
    centers, forward = poses[:, :3, 3], poses[:, :3, 2]
    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
    result = []
    # One row at a time avoids an N*N allocation for large scenes.
    for item, center in enumerate(centers):
        distance = np.linalg.norm(centers - center, axis=-1)
        valid = ((distance >= min_distance) & (distance <= max_distance)
                 & (forward @ forward[item] >= math.cos(math.radians(max_angle))))
        valid[item] = False
        ids = np.flatnonzero(valid)
        result.append(ids[np.argsort(distance[ids], kind="stable")[:count]])
    return result


def choose_depth_sources(dataset, data_dir, candidates=20, keep=5, samples=2048,
                         max_size=320, min_overlap=0.1):
    """Return dataset-item -> source-items, ranked by visible depth overlap.

    Build once using existing depth/mask loading, then persist only names and
    scores. LiDAR is used for selection only; NCC still uses rendered depth.
    """
    settings = dict(candidates=candidates, keep=keep, samples=samples,
                    max_size=max_size, min_overlap=min_overlap,
                    min_distance=0.05, max_distance=30.0, max_angle=75.0,
                    depth_absolute=0.15, depth_relative=0.02)
    parser = dataset.parser
    names = [parser.image_names[i] for i in dataset.indices]
    name_to_item = {name: item for item, name in enumerate(names)}
    signature = _cache_signature(dataset, settings)
    cache_path = Path(data_dir) / "ncc_source_views.json"
    try:
        cached = json.loads(cache_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        cached = {}
    if cached.get("signature") == signature:
        print(f"NCC: loading LiDAR source views from {cache_path}", flush=True)
        return {name_to_item[name]: [name_to_item[row["name"]] for row in rows]
                for name, rows in cached["sources"].items() if rows}

    poses = np.asarray(parser.camtoworlds)[dataset.indices]
    candidate_ids = _pose_candidates(poses, candidates, settings["min_distance"],
                                    settings["max_distance"], settings["max_angle"])
    print("NCC: building LiDAR source views (once; no SfM tracks)", flush=True)
    views = [_small_depth(dataset[item], max_size)
             for item in tqdm(range(len(dataset)), desc="NCC depth thumbnails")]
    records, result = {}, {}
    for item, ids in enumerate(tqdm(candidate_ids, desc="NCC depth overlap")):
        points = _sample_world_points(views[item], samples)
        scored = [(int(source), _depth_overlap(points, views[source],
                   settings["depth_absolute"], settings["depth_relative"])) for source in ids]
        scored = sorted((row for row in scored if row[1] >= min_overlap),
                        key=lambda row: row[1], reverse=True)[:keep]
        records[names[item]] = [{"name": names[source], "overlap": score}
                               for source, score in scored]
        if scored:
            result[item] = [source for source, _ in scored]

    payload = {"signature": signature, "settings": settings, "sources": records}
    # Atomic replacement also makes simultaneous training ranks safe.
    with NamedTemporaryFile(mode="w", dir=cache_path.parent,
                            prefix=".ncc_source_views_", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        json.dump(payload, file, indent=2)
    try:
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result
