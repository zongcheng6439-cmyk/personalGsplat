"""Shared metric-depth reader, called by CityGS preprocessing and datasets.

Callers: generate_depth_edge_masks.py, colmap_adc.py, colmap_depth.py,
generate_depth_normals.py. Example: read_depth(path, tag='uint16', scale=1000).
Input: LA encoded PNG or uint16 PNG; output: float32 meters and boolean mask.
Directory depth_format.json records tag/scale for subsequent training reads.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def depth_settings(directory):
    """Read folder metadata; old unlabelled datasets default to LA."""
    path = Path(directory) / 'depth_format.json'
    return json.loads(path.read_text()) if path.is_file() else {'tag': 'la', 'scale': 1000.0}


def resolve_depth_source(data_dir, directory=None, tag="auto", scale=None):
    """Resolve training directory and explicit tag overrides without changing files."""
    root = Path(data_dir)
    if directory is not None:
        folder = Path(directory).expanduser()
        if not folder.is_absolute():
            folder = root / folder
    elif tag == "uint16":
        folder = root / "depth"
    else:
        folder = root / "depths_render_color_edge"
        if not folder.is_dir():
            folder = root / "depths_render_color"
    settings = depth_settings(folder)
    if tag not in ("auto", "la", "uint16"):
        raise ValueError(f"Unknown depth tag: {tag}")
    if tag != "auto":
        settings = dict(settings, tag=tag)
        # Explicit formats have explicit defaults, independent of stale metadata.
        settings["scale"] = 1000.0
    if scale is not None:
        settings = dict(settings, scale=scale)
    if not np.isfinite(settings["scale"]) or settings["scale"] <= 0:
        raise ValueError("depth-scale must be finite and positive")
    return str(folder), settings


def read_depth(path, tag='la', scale=1000.0, size=None):
    """Decode to meters, resizing with nearest neighbors to preserve invalid pixels."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('depth scale must be finite and positive')
    with Image.open(path) as image:
        raw = np.asarray(image)
        if tag == 'uint16':
            if image.mode not in ('I', 'I;16', 'I;16L', 'I;16B') or raw.ndim != 2:
                raise ValueError(f'{path}: uint16 tag requires a 16-bit integer PNG')
            depth = raw.astype(np.float32) / scale
            valid = raw > 0
        elif tag == 'la':
            if image.mode != 'LA':
                raise ValueError(f'{path}: la tag requires LA PNG, got {image.mode}; use --depth-tag uint16 for ScanNet++')
            value = raw[..., 0].astype(np.float32) / 255.0
            depth = np.where(value < 0.5, value * 20.0,
                             5.0 / np.maximum(1.0 - value, 1e-6))
            valid = (raw[..., 1] > 127) & (raw[..., 0] > 0) & (raw[..., 0] < 255)
        else:
            raise ValueError(f'Unknown depth tag: {tag}')
    valid &= np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0).astype(np.float32)
    if size is not None and tuple(size) != (depth.shape[1], depth.shape[0]):
        depth = np.array(Image.fromarray(depth).resize(size, Image.Resampling.NEAREST))
        valid = np.array(Image.fromarray(valid).resize(size, Image.Resampling.NEAREST))
    return depth, valid


def read_training_depth(path, settings, size):
    """Called by dataset/normal loaders; return CPU tensors in meters."""
    import torch
    depth, valid = read_depth(path, settings['tag'], settings['scale'], size)
    return torch.from_numpy(depth.copy()), torch.from_numpy(valid.copy())
