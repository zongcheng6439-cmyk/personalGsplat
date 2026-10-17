"""Build edge-aware weight maps and texture-aware image sampling weights.

The core implementation intentionally depends only on NumPy and Pillow.  A
semantic model (OCR, vehicle/lane/plate segmentation, etc.) can be connected by
passing its binary masks to :func:`compute_weight_map`; the colour-change map
still remains zero in flat, same-colour regions.
"""

from __future__ import annotations
import os 
import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _as_rgb_float(image: np.ndarray) -> np.ndarray:
    """Convert an HxW[xC] image to finite float32 RGB in [0, 1]."""
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxW or HxWxC image, got {image.shape}")
    rgb = image[..., :3].astype(np.float32)
    if rgb.size and float(np.nanmax(rgb)) > 1.5:
        rgb /= 255.0
    return np.clip(np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _binary_mask(mask: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    """Validate and convert one HxW mask to bool."""
    result = np.squeeze(np.asarray(mask)).astype(bool)
    if result.shape != shape:
        raise ValueError(f"{name} shape {result.shape} != image shape {shape}")
    return result


def _maximum_neighbour_distance(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Largest normalized RGB distance to any 8-connected neighbour."""
    height, width = rgb.shape[:2]
    result = np.zeros((height, width), dtype=np.float32)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        if height <= dy or width <= abs(dx):
            continue
        if dx >= 0:
            y0, x0 = slice(0, height - dy), slice(0, width - dx)
            y1, x1 = slice(dy, height), slice(dx, width)
        else:
            y0, x0 = slice(0, height - dy), slice(-dx, width)
            y1, x1 = slice(dy, height), slice(0, width + dx)
        distance = np.sqrt(np.mean((rgb[y0, x0] - rgb[y1, x1]) ** 2, axis=-1))
        distance = np.where(valid[y0, x0] & valid[y1, x1], distance, 0.0)
        np.maximum(result[y0, x0], distance, out=result[y0, x0])
        np.maximum(result[y1, x1], distance, out=result[y1, x1])
    return result


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    """Remove a noise floor while preserving the [0, 1] endpoint."""
    if threshold == 0.0:
        return values
    return np.maximum(values - threshold, 0.0) / (1.0 - threshold)


def _box_mean(values: np.ndarray, radius: int) -> np.ndarray:
    """Fast reflected-boundary box mean for a 2-D float array."""
    if radius <= 0:
        return values.astype(np.float32, copy=True)
    height, width = values.shape
    padded = np.pad(values, ((radius, radius), (radius, radius)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = 2 * radius + 1
    sums = (
        integral[size : size + height, size : size + width]
        - integral[:height, size : size + width]
        - integral[size : size + height, :width]
        + integral[:height, :width]
    )
    return (sums / float(size * size)).astype(np.float32)


def _local_contrast(rgb: np.ndarray, radius: int = 3) -> np.ndarray:
    """Local luminance standard deviation, normalized approximately to [0, 1]."""
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    mean = _box_mean(gray, radius)
    variance = np.maximum(_box_mean(gray * gray, radius) - mean * mean, 0.0)
    # A bounded [0,1] signal; std=0.25 maps to 0.5 and std=0 maps exactly to 0.
    std = np.sqrt(variance)
    return (std / (std + 0.25)).astype(np.float32)


def _laplacian_response(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Normalized high-frequency response without counting mask boundaries."""
    result = np.zeros(valid.shape, dtype=np.float32)
    if min(valid.shape) < 3:
        return result
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    center = gray[1:-1, 1:-1]
    laplacian = np.abs(
        gray[1:-1, :-2]
        + gray[1:-1, 2:]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        - 4.0 * center
    ) / 4.0
    lap_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
    )
    result[1:-1, 1:-1] = np.where(lap_valid, laplacian, 0.0)
    return np.clip(result, 0.0, 1.0)


def _gaussian_blur(rgb: np.ndarray, radius: float) -> np.ndarray:
    """Blur an RGB float image using Pillow, keeping the same shape."""
    image = Image.fromarray(np.round(rgb * 255.0).astype(np.uint8), mode="RGB")
    blurred = np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32
    )
    return blurred / 255.0


def _robust_unit_map(values: np.ndarray) -> np.ndarray:
    """Robustly map a spatial signal to [0,1], preserving an all-zero map."""
    values = np.maximum(np.nan_to_num(np.asarray(values, dtype=np.float32)), 0.0)
    positive = values[values > 0.0]
    if positive.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    high = float(np.percentile(positive, 95.0))
    if high <= 1e-12:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip(values / high, 0.0, 1.0).astype(np.float32)


def compute_weight_map(
    image: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    *,
    color_threshold: float = 0.0,
    gamma: float = 1.0,
    semantic_masks: Optional[Mapping[str, np.ndarray]] = None,
    semantic_weights: Optional[Mapping[str, float]] = None,
    multiscale: bool = True,
    sharpness_weight: float = 0.20,
    local_contrast_weight: float = 0.15,
    neural_map: Optional[np.ndarray] = None,
    neural_weight: float = 0.25,
) -> np.ndarray:
    """Compute a per-pixel colour-change/semantic weight map.

    The map combines perceptual colour changes at several scales, Laplacian
    sharpness and local contrast. Consequently a same-colour region is exactly
    zero. Multi-scale agreement makes isolated sensor/JPEG noise less valuable
    than a stable contour, text stroke, lane marking or plate boundary.

    ``semantic_masks`` may contain masks from any OCR/detection/segmentation
    model, for example ``text``, ``vehicle``, ``lane`` and ``license_plate``.
    For label ``k`` the final value is ``base * (1 + semantic_weights[k])``.
    Thus semantic masks emphasize useful edges without inventing weight in a
    flat area.  Unspecified labels use a boost of 1.0.

    Args:
        image: HxW, HxWx3 or HxWx4 array, uint8 [0,255] or float [0,1].
        valid_mask: Optional HxW mask. Edges crossing its boundary are ignored.
        color_threshold: Ignore colour distances up to this value, in [0,1).
        gamma: Power applied after thresholding. Values below 1 emphasize weak
            edges; values above 1 emphasize strong edges.
        semantic_masks: Optional mapping from semantic label to HxW mask.
        semantic_weights: Optional non-negative boost for each label.
        multiscale: Blend gradients from the original and two blurred images.
        sharpness_weight: Fraction assigned to Laplacian sharpness.
        local_contrast_weight: Fraction assigned to local luminance contrast.
        neural_map: Optional HxW saliency/feature map from a neural network.
        neural_weight: Multiplicative neural-map boost. Flat RGB regions remain
            zero even when the neural map is non-zero.

    Returns:
        Float32 HxW weight map. Values are non-negative and may exceed one when
        semantic boosts overlap.
    """
    if not 0.0 <= color_threshold < 1.0:
        raise ValueError("color_threshold must be in [0, 1)")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if sharpness_weight < 0.0 or local_contrast_weight < 0.0:
        raise ValueError("component weights must be non-negative")
    if sharpness_weight + local_contrast_weight > 1.0:
        raise ValueError("sharpness_weight + local_contrast_weight must be <= 1")
    if neural_weight < 0.0:
        raise ValueError("neural_weight must be non-negative")

    rgb = _as_rgb_float(image)
    height, width = rgb.shape[:2]
    valid = (
        np.ones((height, width), dtype=bool)
        if valid_mask is None
        else _binary_mask(valid_mask, (height, width), "valid_mask")
    )
    raw_edge = _maximum_neighbour_distance(rgb, valid)
    edge = _soft_threshold(raw_edge, color_threshold)
    if multiscale and min(height, width) >= 5:
        edge_1 = _soft_threshold(
            _maximum_neighbour_distance(_gaussian_blur(rgb, 1.0), valid),
            color_threshold,
        )
        edge_2 = _soft_threshold(
            _maximum_neighbour_distance(_gaussian_blur(rgb, 2.0), valid),
            color_threshold,
        )
        # Fine detail matters, but structure that survives blur is more likely
        # to be a real boundary rather than one-pixel noise.
        edge = 0.50 * edge + 0.30 * edge_1 + 0.20 * edge_2

    edge_weight = 1.0 - sharpness_weight - local_contrast_weight
    weight_map = (
        edge_weight * edge
        + sharpness_weight * _laplacian_response(rgb, valid)
        + local_contrast_weight * _local_contrast(rgb)
    )
    # Contrast windows can extend into flat neighbours. Preserve the important
    # invariant requested by the sampler: no local colour change means zero.
    weight_map[raw_edge <= color_threshold] = 0.0
    if gamma != 1.0:
        np.power(weight_map, gamma, out=weight_map)
    weight_map[~valid] = 0.0

    if semantic_masks:
        boosts = semantic_weights or {}
        multiplier = np.ones((height, width), dtype=np.float32)
        for label, mask in semantic_masks.items():
            boost = float(boosts.get(label, 1.0))
            if boost < 0.0:
                raise ValueError(f"semantic weight for {label!r} must be non-negative")
            multiplier += boost * _binary_mask(
                mask, (height, width), f"semantic mask {label!r}"
            )
        weight_map *= multiplier

    if neural_map is not None and neural_weight > 0.0:
        neural = np.squeeze(np.asarray(neural_map))
        if neural.shape != (height, width):
            raise ValueError(
                f"neural_map shape {neural.shape} != image shape {(height, width)}"
            )
        weight_map *= 1.0 + neural_weight * _robust_unit_map(neural)

    return weight_map.astype(np.float32, copy=False)


def score_weight_map(weight_map: np.ndarray, top_fraction: float = 0.1) -> float:
    """Reduce a weight map to one image score.

    The full-map mean rewards useful-edge coverage while the top-fraction mean
    rewards a small number of crisp structures such as text and plate edges.
    """
    values = np.asarray(weight_map, dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("weight_map must be a non-empty HxW array")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    values = np.maximum(np.nan_to_num(values, nan=0.0), 0.0)
    flattened = values.ravel()
    count = max(1, int(np.ceil(flattened.size * top_fraction)))
    top_mean = float(np.partition(flattened, -count)[-count:].mean())
    return 0.65 * float(values.mean()) + 0.35 * top_mean


def semantic_region_statistics(
    weight_map: np.ndarray, semantic_masks: Mapping[str, np.ndarray]
) -> dict[str, dict[str, float]]:
    """Return pixel coverage and edge-weight share for every semantic mask."""
    weight_map = np.asarray(weight_map, dtype=np.float32)
    if weight_map.ndim != 2:
        raise ValueError("weight_map must be HxW")
    total_weight = float(weight_map.sum())
    result: dict[str, dict[str, float]] = {}
    for label, mask in semantic_masks.items():
        binary = _binary_mask(mask, weight_map.shape, f"semantic mask {label!r}")
        result[label] = {
            "pixel_ratio": float(binary.mean()),
            "weight_ratio": (
                float(weight_map[binary].sum()) / total_weight
                if total_weight > 0.0
                else 0.0
            ),
        }
    return result


class TorchvisionFeatureExtractor:
    """Optional pretrained ResNet feature-map and semantic-richness extractor.

    This class is deliberately lazy: importing this module does not require
    torchvision and does not download model weights. Constructing the class
    loads ImageNet weights (and may download them into PyTorch's normal cache).
    Exact text/lane/plate recognition still requires a task-specific detector;
    pass those detector masks through ``semantic_masks``.

    The returned spatial map is the L2 energy of the last convolutional feature
    grid, resized to the input image. The scalar score combines spatial feature
    variation with channel entropy and is intended to be normalized across the
    whole image collection before it is mixed with the classical score.
    """

    _SUPPORTED_MODELS = {"resnet18", "resnet50"}

    def __init__(
        self,
        model_name: str = "resnet18",
        device: str = "auto",
        max_input_size: int = 512,
    ) -> None:
        if model_name not in self._SUPPORTED_MODELS:
            raise ValueError(
                f"model_name must be one of {sorted(self._SUPPORTED_MODELS)}, "
                f"got {model_name!r}"
            )
        if max_input_size <= 0:
            raise ValueError("max_input_size must be positive")
        try:
            import torch
            from torchvision import models
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Neural scoring needs a torchvision build compatible with the "
                "installed PyTorch. Install/repair torchvision, or omit "
                "--neural-model to use the NumPy/Pillow scorer."
            ) from exc

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        try:
            if model_name == "resnet18":
                model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            else:
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load pretrained {model_name} weights. Ensure the "
                "weights are cached or network access is available."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.max_input_size = max_input_size
        self.backbone = torch.nn.Sequential(*list(model.children())[:-2])
        self.backbone.eval().to(self.device)
        self.mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(
            1, 3, 1, 1
        )
        self.std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(
            1, 3, 1, 1
        )

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """Return ``(HxW feature map, raw neural richness score)``."""
        torch = self.torch
        rgb = _as_rgb_float(image)
        height, width = rgb.shape[:2]
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        scale = min(1.0, float(self.max_input_size) / max(height, width))
        if scale < 1.0:
            resized = (max(1, round(height * scale)), max(1, round(width * scale)))
            tensor = torch.nn.functional.interpolate(
                tensor, size=resized, mode="bilinear", align_corners=False
            )
        tensor = (tensor - self.mean) / self.std

        with torch.inference_mode():
            features = self.backbone(tensor).float()
            energy = torch.sqrt(torch.mean(features * features, dim=1, keepdim=True))
            feature_map = torch.nn.functional.interpolate(
                energy, size=(height, width), mode="bilinear", align_corners=False
            )[0, 0]

            spatial_variation = features.flatten(2).std(dim=2, unbiased=False).mean()
            channels = features.relu().mean(dim=(2, 3))[0]
            probabilities = channels / channels.sum().clamp_min(1e-12)
            channel_entropy = -(
                probabilities * probabilities.clamp_min(1e-12).log()
            ).sum() / np.log(max(2, channels.numel()))
            score = spatial_variation * (0.5 + 0.5 * channel_entropy)

        return feature_map.cpu().numpy().astype(np.float32), float(score.item())


def _top_mean(values: np.ndarray, fraction: float = 0.1) -> float:
    """Mean of the largest values, or zero when no positive values exist."""
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    count = max(1, int(np.ceil(values.size * fraction)))
    return float(np.partition(values, -count)[-count:].mean())


def texture_energy(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strong_threshold: float = 0.08,
) -> float:
    """Measure sharp texture using only pixels inside ``mask``.

    Image values may be uint8 [0, 255] or float [0, 1]. Mask values are valid
    when non-zero. A gradient is counted only when both neighboring pixels are
    valid, so the mask boundary itself never becomes a texture edge.
    """
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., :3].astype(np.float32)
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    elif image.ndim == 2:
        gray = image.astype(np.float32)
    else:
        raise ValueError(f"Expected HxW or HxWxC image, got {image.shape}")

    if gray.size and float(np.nanmax(gray)) > 1.5:
        gray /= 255.0
    gray = np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0)

    valid = (
        np.ones(gray.shape, dtype=bool)
        if mask is None
        else np.squeeze(mask).astype(bool)
    )
    if valid.shape != gray.shape:
        raise ValueError(f"Mask shape {valid.shape} != image shape {gray.shape}")
    if min(gray.shape) < 3:
        return 0.0

    # Forward gradients on a common (H-1)x(W-1) grid.
    gx = np.diff(gray, axis=1)[:-1]
    gy = np.diff(gray, axis=0)[:, :-1]
    gradient = np.sqrt(gx * gx + gy * gy)
    gradient_valid = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1]
    gradients = gradient[gradient_valid]
    if gradients.size == 0:
        return 0.0

    # Only changes above 8% intensity per pixel contribute. This rejects broad
    # weak illumination changes and most low-amplitude noise.
    strong_gradient = np.maximum(gradients - strong_threshold, 0.0)
    edge_strength = _top_mean(strong_gradient)
    edge_density = float(np.mean(strong_gradient > 0))

    center = gray[1:-1, 1:-1]
    laplacian = np.abs(
        gray[1:-1, :-2]
        + gray[1:-1, 2:]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        - 4.0 * center
    )
    lap_valid = (
        valid[1:-1, 1:-1]
        & valid[1:-1, :-2]
        & valid[1:-1, 2:]
        & valid[:-2, 1:-1]
        & valid[2:, 1:-1]
    )
    strong_laplacian = np.maximum(laplacian[lap_valid] - strong_threshold, 0.0)
    sharpness = _top_mean(strong_laplacian)

    # Strength captures clear text/contours, density rewards multiple useful
    # structures, and Laplacian response penalizes blurred edges.
    return 0.55 * edge_strength + 0.25 * np.sqrt(edge_density) + 0.20 * sharpness


def score_image(
    image_path: str | Path,
    mask_path: Optional[str | Path] = None,
    max_size: int = 512,
) -> float:
    """Read one image/mask pair and return its raw texture score."""
    with Image.open(image_path) as image_file:
        image_file = image_file.convert("RGB")
        original_size = image_file.size
        scale = min(1.0, float(max_size) / max(original_size))
        if scale < 1.0:
            size = tuple(max(1, round(value * scale)) for value in original_size)
            image_file = image_file.resize(size, Image.Resampling.BOX)
        image = np.asarray(image_file)

    mask = None
    if mask_path is not None:
        with Image.open(mask_path) as mask_file:
            mask_file = mask_file.convert("L")
            if mask_file.size != original_size:
                raise ValueError(
                    f"Mask {mask_path} size {mask_file.size} does not match "
                    f"image {image_path} size {original_size}"
                )
            if scale < 1.0:
                mask_file = mask_file.resize(image_file.size, Image.Resampling.NEAREST)
            mask = np.asarray(mask_file) > 0
    return texture_energy(image, mask)


def _load_resized_image(image_path: str | Path, max_size: int) -> np.ndarray:
    """Load an RGB image and optionally shrink its longest side."""
    with Image.open(image_path) as image_file:
        image_file = image_file.convert("RGB")
        scale = min(1.0, float(max_size) / max(image_file.size))
        if scale < 1.0:
            size = tuple(max(1, round(value * scale)) for value in image_file.size)
            image_file = image_file.resize(size, Image.Resampling.BOX)
        return np.asarray(image_file)


def _load_resized_mask(mask_path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    """Load a binary mask and resize it to ``shape`` with nearest sampling."""
    height, width = shape
    with Image.open(mask_path) as mask_file:
        mask_file = mask_file.convert("L")
        if mask_file.size != (width, height):
            mask_file = mask_file.resize((width, height), Image.Resampling.NEAREST)
        return np.asarray(mask_file) > 0


def rank_images_by_weight_map(
    image_paths: Sequence[str | Path],
    *,
    valid_mask_paths: Optional[Sequence[str | Path]] = None,
    top_k: int = 20,
    max_size: int = 1024,
    color_threshold: float = 0.02,
    gamma: float = 1.0,
    weight_map_dir: Optional[str | Path] = None,    
    image_dir: Optional[str | Path] = None,
    neural_extractor: Optional[TorchvisionFeatureExtractor] = None,
    neural_weight: float = 0.25,
    skip_errors: bool = False,
) -> list[dict[str, object]]:
    """Rank images and return the top-k paths and normalized scores.

    If ``weight_map_dir`` is given, every exact float map is saved as ``.npy``
    and a display-only grayscale preview is saved as ``.png``.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("image_paths must not be empty")
    if valid_mask_paths is not None and len(valid_mask_paths) != len(paths):
        raise ValueError("valid_mask_paths and image_paths must have the same length")
    masks = (
        None
        if valid_mask_paths is None
        else [Path(path) for path in valid_mask_paths]
    )

    output_dir = None if weight_map_dir is None else Path(weight_map_dir)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 <= neural_weight <= 1.0:
        raise ValueError("neural_weight must be in [0, 1]")

    raw_scores: list[float] = []
    neural_scores: list[float] = []
    map_paths: list[Optional[str]] = []
    processed_paths: list[Path] = []
    for index, path in enumerate(paths):
        try:
            image = _load_resized_image(path, max_size=max_size)
        except OSError as exc:
            if not skip_errors:
                raise
            print(f"[WARN] skipping unreadable image {path}: {exc}", file=sys.stderr)
            continue
        valid_mask = (
            None
            if masks is None
            else _load_resized_mask(masks[index], image.shape[:2])
        )
        neural_map = None
        neural_score = 0.0
        if neural_extractor is not None:
            neural_map, neural_score = neural_extractor(image)
        weight_map = compute_weight_map(
            image,
            valid_mask=valid_mask,
            color_threshold=color_threshold,
            gamma=gamma,
            neural_map=neural_map,
            neural_weight=neural_weight,
        )
        raw_scores.append(score_weight_map(weight_map))
        neural_scores.append(neural_score)
        processed_paths.append(path)

        saved_map: Optional[str] = None
        if output_dir is not None:
            base = Path(str(path).replace(image_dir, str(output_dir)))
            base.parent.mkdir(parents=True, exist_ok=True)
            np.save(base.with_suffix(".npy"), weight_map)
            peak = float(weight_map.max())
            preview = (
                np.zeros(weight_map.shape, dtype=np.uint8)
                if peak == 0.0
                else np.round(np.clip(weight_map / peak, 0.0, 1.0) * 255).astype(
                    np.uint8
                )
            )
            Image.fromarray(preview).save(base.with_suffix(".png"))
            saved_map = str(base.with_suffix(".npy").resolve())
        map_paths.append(saved_map)

    if not processed_paths:
        raise ValueError("no readable images were scored")
    classical = normalize_scores(raw_scores)
    if neural_extractor is None or neural_weight == 0.0:
        combined = classical
    else:
        neural = normalize_scores(neural_scores)
        combined = (1.0 - neural_weight) * classical + neural_weight * neural
    
    # order = np.argsort(-combined, kind="stable")[: min(top_k, len(processed_paths))]
    order = np.argsort(-combined, kind="stable")
    results = []
    for rank, index in enumerate(order, start=1):
        path = str(processed_paths[index])
        name = path.split(image_dir)[-1]
        results.append(
            {
                "name": str(name)[1:],
                "score": float(combined[index]),
            }
        )
    return results


def find_images(image_dir: str | Path, recursive: bool = True) -> list[Path]:
    """Return supported image files below ``image_dir`` in stable order."""
    image_dir = Path(image_dir)
    iterator = image_dir.rglob("*") if recursive else image_dir.glob("*")
    return sorted(path for path in iterator if path.suffix.lower() in IMAGE_SUFFIXES)


def match_masks_by_stem(
    image_paths: Sequence[str | Path], mask_dir: str | Path, image_dir: str | Path
) -> list[Path]:
    """Match every image to exactly one mask by filename stem.

    Unlike independently sorting two directories, this cannot silently pair an
    image with the wrong mask when one side has a missing file.
    """
    result: list[Path] = []
    for image_path in image_paths:        
        mask_path = str(image_path).replace(image_dir, mask_dir)        
        if not os.path.exists(mask_path):
            raise ValueError(
                f"Expected mask for {image_path}, but {mask_path} do not exist. "
            )
        result.append(mask_path)
    return result


def normalize_scores(raw_scores: Sequence[float]) -> np.ndarray:
    """Robustly normalize scores to [0,1] without destroying their ordering.

    The central 5th--95th percentile range occupies [0.05,0.95], while both
    tails are linearly mapped into the remaining intervals. This retains
    outlier robustness without clipping every top-5% image to the same score.
    """
    scores = np.asarray(raw_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("raw_scores must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(scores)):
        raise ValueError("raw_scores must contain only finite values")
    minimum, maximum = float(scores.min()), float(scores.max())
    if maximum <= minimum + 1e-12:
        return np.zeros(scores.shape, dtype=np.float32)

    low, high = np.percentile(scores, [5.0, 95.0])
    if high <= low + 1e-12:
        return ((scores - minimum) / (maximum - minimum)).astype(np.float32)

    normalized = 0.05 + 0.90 * (scores - low) / (high - low)
    below = scores < low
    above = scores > high
    if low > minimum + 1e-12:
        normalized[below] = 0.05 * (scores[below] - minimum) / (low - minimum)
    else:
        normalized[below] = 0.0
    if maximum > high + 1e-12:
        normalized[above] = 0.95 + 0.05 * (scores[above] - high) / (
            maximum - high
        )
    else:
        normalized[above] = 1.0
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def score_images(
    image_paths: Sequence[str | Path],
    mask_paths: Optional[Sequence[str | Path]] = None,
    max_size: int = 512,
) -> np.ndarray:
    """Score image/mask pairs and return scores normalized to [0, 1]."""
    if mask_paths is not None and len(mask_paths) != len(image_paths):
        raise ValueError("mask_paths and image_paths must have the same length")
    raw = [
        score_image(path, None if mask_paths is None else mask_paths[index], max_size)
        for index, path in enumerate(image_paths)
    ]
    return normalize_scores(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_dir", type=Path,default='/media/zc/SSD/ar/AIBEE_16/L16/showroom/')
    parser.add_argument("--image_dir", type=str,default='images')
    parser.add_argument("--mask_dir", type=str, default='masks')
    parser.add_argument("--output", type=Path, default='texture_score.json')
    parser.add_argument("--weight-map-dir", type=Path, default='weight_maps')
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--color_threshold", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument(
        "--neural-model",
        choices=("resnet18", "resnet50"),
        help="optionally mix pretrained ImageNet features into the score",
    )
    parser.add_argument("--neural-weight", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, ...")
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="warn and continue when an image is unreadable",
    )
    
    args = parser.parse_args()
    image_dir = os.path.join(args.project_dir, args.image_dir)
    images = find_images(image_dir) if args.image_dir else None
    if not images:
        parser.error("no supported images found")
    try:
        mask_paths = (
            match_masks_by_stem(images, args.mask_dir, args.image_dir)
            if args.mask_dir is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))
    neural_extractor = (
        None
        if args.neural_model is None
        else TorchvisionFeatureExtractor(args.neural_model, device=args.device)
    )

    result = rank_images_by_weight_map(
        images,
        valid_mask_paths=mask_paths,
        top_k=args.top_k,
        max_size=args.max_size,
        color_threshold=args.color_threshold,
        gamma=args.gamma,
        weight_map_dir=args.weight_map_dir,
        image_dir=args.image_dir,
        neural_extractor=neural_extractor,
        neural_weight=args.neural_weight,
        skip_errors=args.skip_errors,
    )
    output = Path(os.path.join(args.project_dir, args.output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    index = 1
    for row in result[:20]:
        print(f" index: {index}  score={row['score']:.6f}  {image_dir}/{row['name']}")
    print(f"Saved top {len(result)} results to {output.resolve()}")


if __name__ == "__main__":
    main()
