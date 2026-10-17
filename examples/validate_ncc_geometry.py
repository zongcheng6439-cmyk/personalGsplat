#!/usr/bin/env python3
"""Validate plane-warp NCC with a frozen trained 3DGS model.

This file does not train or modify the model. It:
  1. picks SfM-covisible reference/source image pairs;
  2. renders expected depth from the latest voxel-plane MCMC PLY;
  3. builds local planes from rendered depth and depth-derived normals;
  4. warps real image patches with a plane-induced homography;
  5. repeats the warp after perturbing plane depth and reports whether NCC is
     minimized near the unperturbed depth.

Default execution for parking01_section:

    python examples/validate_ncc_geometry.py

The default PLY is the newest ``gsplat_point_cloud_*.ply`` under
``gsplat_voxel_plane_mcmc/ply``. No training code is changed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import imageio.v2 as imageio
import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.colmap_adc import Dataset, Parser
from gsplat.exporter import load_ply_to_splats
from gsplat.rendering import rasterization


DEFAULT_DATA = Path("/lutu/parking01_section")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--ply", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-refs", type=int, default=20)
    parser.add_argument("--num-sources", type=int, default=4)
    parser.add_argument("--samples-per-pair", type=int, default=1024)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--min-shared-points", type=int, default=50)
    parser.add_argument("--min-baseline", type=float, default=0.05)
    parser.add_argument("--max-baseline", type=float, default=30.0)
    parser.add_argument("--max-view-angle", type=float, default=75.0)
    parser.add_argument("--min-patch-std", type=float, default=0.03)
    parser.add_argument("--min-alpha", type=float, default=0.8)
    parser.add_argument("--plane-window", type=int, default=11)
    parser.add_argument("--plane-max-rmse", type=float, default=0.02)
    parser.add_argument("--plane-max-eigen-ratio", type=float, default=0.05)
    parser.add_argument("--depth-edge-abs", type=float, default=0.20)
    parser.add_argument("--depth-edge-rel", type=float, default=0.02)
    parser.add_argument("--edge-dilation", type=int, default=5)
    parser.add_argument("--occlusion-abs", type=float, default=0.15)
    parser.add_argument("--occlusion-rel", type=float, default=0.02)
    parser.add_argument(
        "--depth-offsets-cm",
        type=float,
        nargs="+",
        default=[-20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0],
    )
    parser.add_argument("--visualize-patches", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def newest_training_ply(folder: Path) -> Path:
    files = list(folder.glob("gsplat_point_cloud_*.ply"))
    if not files:
        raise FileNotFoundError(f"No gsplat_point_cloud_*.ply under {folder}")

    def step(path: Path) -> int:
        try:
            return int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1

    return max(files, key=lambda path: (step(path), path.stat().st_mtime))


def colmap_directory(data_dir: Path) -> Path:
    folder = data_dir / "sparse" / "0"
    return folder if folder.is_dir() else data_dir / "sparse"


def valid_point3d_ids(image: object) -> Iterable[int]:
    for point in image.points2D:
        point_id = int(point.point3D_id)
        if point_id >= 0 and point_id < np.iinfo(np.uint64).max:
            yield point_id


def build_covisibility(parser: Parser, data_dir: Path) -> Dict[int, Counter]:
    """Return shared sparse-track counts in Parser image-index coordinates."""
    reconstruction = pycolmap.Reconstruction(str(colmap_directory(data_dir)))
    name_to_index = {name: i for i, name in enumerate(parser.image_names)}
    tracks: Dict[int, List[int]] = defaultdict(list)

    images = reconstruction.images
    for image_id in reconstruction.reg_image_ids():
        image = images[image_id]
        image_index = name_to_index.get(image.name)
        if image_index is None:
            continue
        for point_id in set(valid_point3d_ids(image)):
            tracks[point_id].append(image_index)

    covisibility: Dict[int, Counter] = defaultdict(Counter)
    for image_indices in tracks.values():
        for ref in image_indices:
            covisibility[ref].update(i for i in image_indices if i != ref)
    return covisibility


def choose_sources(
    ref: int,
    covisibility: Dict[int, Counter],
    camtoworlds: np.ndarray,
    args: argparse.Namespace,
) -> List[Tuple[int, int]]:
    ref_pose = camtoworlds[ref]
    ref_center = ref_pose[:3, 3]
    ref_forward = ref_pose[:3, 2]
    selected = []
    for source, shared in covisibility[ref].most_common():
        if shared < args.min_shared_points:
            break
        source_pose = camtoworlds[source]
        baseline = float(np.linalg.norm(source_pose[:3, 3] - ref_center))
        cosine = np.clip(
            np.dot(ref_forward, source_pose[:3, 2])
            / (
                np.linalg.norm(ref_forward)
                * np.linalg.norm(source_pose[:3, 2])
                + 1e-12
            ),
            -1.0,
            1.0,
        )
        angle = math.degrees(math.acos(float(cosine)))
        if (
            args.min_baseline <= baseline <= args.max_baseline
            and angle <= args.max_view_angle
        ):
            selected.append((source, shared))
            if len(selected) == args.num_sources:
                break
    return selected


class FrozenRenderer:
    def __init__(self, ply: Path, device: torch.device):
        splats = load_ply_to_splats(str(ply))
        self.means = splats["means"].to(device)
        self.quats = splats["quats"].to(device)
        self.scales = splats["scales"].to(device).exp()
        self.opacities = splats["opacities"].to(device).sigmoid()
        self.colors = torch.cat([splats["sh0"], splats["shN"]], dim=1).to(device)
        self.sh_degree = int(math.sqrt(self.colors.shape[1]) - 1)

    @torch.inference_mode()
    def __call__(
        self, data: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = data["image"].shape[:2]
        camtoworld = data["camtoworld"].to(device)[None]
        K = data["K"].to(device)[None]
        renders, alphas, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
            viewmats=torch.linalg.inv(camtoworld),
            Ks=K,
            width=width,
            height=height,
            packed=False,
            rasterize_mode="classic",
            camera_model="pinhole",
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB+ED",
            sh_degree=self.sh_degree,
        )
        return renders[0, ..., 3], alphas[0, ..., 0]


def grayscale(image: torch.Tensor) -> torch.Tensor:
    return (image * image.new_tensor([0.299, 0.587, 0.114])).sum(-1)


def depth_geometry(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    image_mask: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Return conservative candidates; normals are fitted later with 3D PCA."""
    valid = (
        torch.isfinite(depth)
        & (depth > 0)
        & (alpha >= args.min_alpha)
        & image_mask
    )
    jump = torch.zeros_like(depth, dtype=torch.bool)
    horizontal = (depth[:, 1:] - depth[:, :-1]).abs()
    horizontal_limit = torch.maximum(
        depth[:, 1:].minimum(depth[:, :-1]) * args.depth_edge_rel,
        depth.new_tensor(args.depth_edge_abs),
    )
    horizontal = horizontal > horizontal_limit
    jump[:, 1:] |= horizontal
    jump[:, :-1] |= horizontal
    vertical = (depth[1:] - depth[:-1]).abs()
    vertical_limit = torch.maximum(
        depth[1:].minimum(depth[:-1]) * args.depth_edge_rel,
        depth.new_tensor(args.depth_edge_abs),
    )
    vertical = vertical > vertical_limit
    jump[1:] |= vertical
    jump[:-1] |= vertical
    dilation = args.edge_dilation
    jump = F.max_pool2d(
        jump.float()[None, None], dilation, stride=1, padding=dilation // 2
    )[0, 0].bool()

    valid &= ~jump
    border = max(args.patch_size, args.plane_window) // 2 + 1
    valid[:border] = False
    valid[-border:] = False
    valid[:, :border] = False
    valid[:, -border:] = False
    return valid


def textured_centers(
    image: torch.Tensor,
    valid: torch.Tensor,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> torch.Tensor:
    gray = grayscale(image)
    kernel = args.patch_size
    mean = F.avg_pool2d(gray[None, None], kernel, 1, kernel // 2)[0, 0]
    square_mean = F.avg_pool2d(
        gray.square()[None, None], kernel, 1, kernel // 2
    )[0, 0]
    std = (square_mean - mean.square()).clamp_min(0).sqrt()
    ids = torch.nonzero(valid & (std >= args.min_patch_std), as_tuple=False)
    candidate_count = args.samples_per_pair * 4
    if len(ids) > candidate_count:
        choice = rng.choice(len(ids), candidate_count, replace=False)
        ids = ids[torch.from_numpy(choice).to(ids.device)]
    return ids[:, [1, 0]]  # [x, y]


def fit_local_planes(
    depth: torch.Tensor,
    alpha: torch.Tensor,
    image_mask: torch.Tensor,
    K: torch.Tensor,
    centers: torch.Tensor,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit one camera-space plane per center from an 11x11 3D neighborhood."""
    if centers.numel() == 0:
        empty = depth.new_empty((0, 3))
        return centers, empty, depth.new_empty(0), depth.new_empty(0), depth.new_empty(0)

    xy = patch_pixels(centers, args.plane_window).long()
    x, y = xy[..., 0], xy[..., 1]
    local_depth = depth[y, x]
    local_alpha = alpha[y, x]
    local_mask = image_mask[y, x]
    valid = (
        torch.isfinite(local_depth)
        & (local_depth > 0)
        & (local_alpha >= args.min_alpha)
        & local_mask
    )
    weights = valid.to(depth.dtype)
    valid_count = weights.sum(-1)

    rays = torch.stack(
        [
            (x - K[0, 2]) / K[0, 0],
            (y - K[1, 2]) / K[1, 1],
            torch.ones_like(local_depth),
        ],
        dim=-1,
    ).to(depth.dtype)
    points = rays * torch.where(valid, local_depth, 0.0)[..., None]
    denominator = valid_count.clamp_min(1.0)
    centroid = (points * weights[..., None]).sum(1) / denominator[:, None]
    centered = points - centroid[:, None]
    covariance = torch.einsum(
        "npi,npj,np->nij", centered, centered, weights
    ) / denominator[:, None, None]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    normals = eigenvectors[..., 0]
    normals = torch.where(
        (normals * centroid).sum(-1, keepdim=True) > 0, -normals, normals
    )
    rmse = eigenvalues[:, 0].clamp_min(0).sqrt()
    eigen_ratio = eigenvalues[:, 0] / eigenvalues[:, 1].clamp_min(1e-12)
    plane_distance = (normals * centroid).sum(-1)
    keep = (
        (valid_count >= args.plane_window**2 * 0.9)
        & (eigenvalues[:, 1] > 1e-10)
        & (rmse <= args.plane_max_rmse)
        & (eigen_ratio <= args.plane_max_eigen_ratio)
        & torch.isfinite(normals).all(-1)
        & (plane_distance.abs() > 1e-6)
    )
    centers = centers[keep]
    normals = normals[keep]
    plane_distance = plane_distance[keep]
    rmse = rmse[keep]
    eigen_ratio = eigen_ratio[keep]
    if len(centers) > args.samples_per_pair:
        choice = rng.choice(len(centers), args.samples_per_pair, replace=False)
        choice = torch.from_numpy(choice).to(centers.device)
        centers = centers[choice]
        normals = normals[choice]
        plane_distance = plane_distance[choice]
        rmse = rmse[choice]
        eigen_ratio = eigen_ratio[choice]
    return centers, normals, plane_distance, rmse, eigen_ratio


def patch_pixels(centers: torch.Tensor, patch_size: int) -> torch.Tensor:
    radius = patch_size // 2
    oy, ox = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=centers.device),
        torch.arange(-radius, radius + 1, device=centers.device),
        indexing="ij",
    )
    offsets = torch.stack([ox.flatten(), oy.flatten()], dim=-1)
    return centers[:, None, :] + offsets[None]


def plane_warp(
    ref_image: torch.Tensor,
    source_image: torch.Tensor,
    source_depth: torch.Tensor,
    source_alpha: torch.Tensor,
    source_mask: torch.Tensor,
    centers: torch.Tensor,
    ref_depth: torch.Tensor,
    plane_normals: torch.Tensor,
    base_plane_distance: torch.Tensor,
    K_ref: torch.Tensor,
    K_source: torch.Tensor,
    c2w_ref: torch.Tensor,
    c2w_source: torch.Tensor,
    depth_offset: float,
    check_visibility: bool,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Warp patches and return per-patch NCC loss, validity and source RGB."""
    count = len(centers)
    patch_area = args.patch_size**2
    if count == 0:
        empty = ref_depth.new_empty(0)
        return empty, empty.bool(), ref_depth.new_empty((0, patch_area, 3))

    xy = patch_pixels(centers, args.patch_size).to(ref_depth.dtype)
    homogeneous = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
    normals = plane_normals
    center_ray = torch.stack(
        [
            (centers[:, 0] - K_ref[0, 2]) / K_ref[0, 0],
            (centers[:, 1] - K_ref[1, 2]) / K_ref[1, 1],
            torch.ones_like(centers[:, 0]),
        ],
        dim=-1,
    )
    plane_distance = base_plane_distance + depth_offset * (
        normals * center_ray
    ).sum(-1)

    ref_to_source = torch.linalg.inv(c2w_source) @ c2w_ref
    rotation = ref_to_source[:3, :3]
    translation = ref_to_source[:3, 3]
    safe_distance = torch.where(
        plane_distance.abs() < 1e-6,
        plane_distance.sign().masked_fill(plane_distance == 0, 1.0) * 1e-6,
        plane_distance,
    )
    homography = K_source @ (
        rotation[None]
        + translation[None, :, None]
        * normals[:, None, :]
        / safe_distance[:, None, None]
    ) @ torch.linalg.inv(K_ref)
    projected = torch.einsum("nij,npj->npi", homography, homogeneous)
    source_xy = projected[..., :2] / projected[..., 2:].clamp_min(1e-6)

    source_height, source_width = source_depth.shape
    grid = torch.stack(
        [
            2.0 * source_xy[..., 0] / max(source_width - 1, 1) - 1.0,
            2.0 * source_xy[..., 1] / max(source_height - 1, 1) - 1.0,
        ],
        dim=-1,
    )

    def sample(value: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
        channels = 1 if value.ndim == 2 else value.shape[-1]
        source = value.reshape(source_height, source_width, channels)
        source = source.permute(2, 0, 1)[None]
        sampled = F.grid_sample(
            source,
            grid[None],
            mode=mode,
            padding_mode="zeros",
            align_corners=True,
        )[0].permute(1, 2, 0)
        return sampled[..., 0] if channels == 1 else sampled

    source_patch = sample(source_image)
    sampled_depth = sample(source_depth)
    sampled_alpha = sample(source_alpha)
    sampled_mask = sample(source_mask.float(), mode="nearest") > 0.5

    inv_K_ref = torch.linalg.inv(K_ref)
    rays = torch.einsum("ij,npj->npi", inv_K_ref, homogeneous)
    ray_denominator = torch.einsum("ni,npi->np", normals, rays)
    ray_denominator = torch.where(
        ray_denominator.abs() < 1e-6,
        ray_denominator.sign().masked_fill(ray_denominator == 0, 1.0) * 1e-6,
        ray_denominator,
    )
    ray_scale = safe_distance[:, None] / ray_denominator
    points_ref = rays * ray_scale[..., None]
    points_source = torch.einsum(
        "ij,npj->npi", rotation, points_ref
    ) + translation
    projected_depth = points_source[..., 2]
    depth_tolerance = torch.maximum(
        projected_depth.abs() * args.occlusion_rel,
        projected_depth.new_tensor(args.occlusion_abs),
    )
    inside = (
        (source_xy[..., 0] >= 0)
        & (source_xy[..., 0] <= source_width - 1)
        & (source_xy[..., 1] >= 0)
        & (source_xy[..., 1] <= source_height - 1)
        & (projected_depth > 0)
    )
    pixel_valid = inside
    if check_visibility:
        pixel_valid &= (
            sampled_mask
            & (sampled_alpha >= args.min_alpha)
            & ((sampled_depth - projected_depth).abs() <= depth_tolerance)
        )

    ref_x = xy[..., 0].long()
    ref_y = xy[..., 1].long()
    ref_patch = ref_image[ref_y, ref_x]
    ref_gray = grayscale(ref_patch)
    source_gray = grayscale(source_patch)
    ref_centered = ref_gray - ref_gray.mean(-1, keepdim=True)
    source_centered = source_gray - source_gray.mean(-1, keepdim=True)
    denominator = (
        ref_centered.square().sum(-1) * source_centered.square().sum(-1)
    ).clamp_min(1e-12).sqrt()
    correlation = (ref_centered * source_centered).sum(-1) / denominator
    loss = 1.0 - correlation.clamp(-1.0, 1.0)
    valid_patch = (
        pixel_valid.all(-1)
        & (ref_gray.std(-1) >= args.min_patch_std)
        & (source_gray.std(-1) >= args.min_patch_std)
        & torch.isfinite(loss)
    )
    return loss, valid_patch, source_patch


def save_reference_debug(
    output_dir: Path,
    name: str,
    image: torch.Tensor,
    normals: torch.Tensor,
    centers: torch.Tensor,
) -> None:
    folder = output_dir / "references"
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    rgb = (image.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    marked = rgb.copy()
    for x, y in centers.detach().cpu().numpy().astype(int):
        marked[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3] = [255, 0, 0]
    normal_rgb = np.zeros_like(rgb)
    plane_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    normal_colors = (
        (normals.detach().cpu().numpy().clip(-1, 1) * 0.5 + 0.5) * 255
    ).astype(np.uint8)
    for (x, y), color in zip(
        centers.detach().cpu().numpy().astype(int), normal_colors
    ):
        normal_rgb[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3] = color
        plane_mask[max(y - 2, 0) : y + 3, max(x - 2, 0) : x + 3] = 255
    imageio.imwrite(folder / f"{stem}_samples.png", marked)
    imageio.imwrite(folder / f"{stem}_normals.png", normal_rgb)
    imageio.imwrite(
        folder / f"{stem}_plane_mask.png",
        plane_mask,
    )


def save_patch_mosaic(
    path: Path,
    ref_image: torch.Tensor,
    centers: torch.Tensor,
    source_patches: torch.Tensor,
    valid: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    ids = torch.where(valid)[0][: args.visualize_patches]
    if ids.numel() == 0:
        return
    xy = patch_pixels(centers[ids], args.patch_size).long()
    ref_patch = ref_image[xy[..., 1], xy[..., 0]]
    source_patch = source_patches[ids]
    difference = (ref_patch - source_patch).abs()
    rows = []
    for ref, source, diff in zip(ref_patch, source_patch, difference):
        row = torch.cat([ref, source, diff], dim=1)
        row = (row.clamp(0, 1) * 255).byte().cpu().numpy()
        row = np.repeat(np.repeat(row, 12, axis=0), 12, axis=1)
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.concatenate(rows, axis=0))


def main() -> None:
    args = parse_args()
    if args.patch_size < 3 or args.patch_size % 2 == 0:
        raise ValueError("--patch-size must be an odd integer >= 3")
    if args.plane_window < args.patch_size or args.plane_window % 2 == 0:
        raise ValueError("--plane-window must be odd and >= --patch-size")
    if args.edge_dilation < 1 or args.edge_dilation % 2 == 0:
        raise ValueError("--edge-dilation must be an odd positive integer")
    if 0.0 not in args.depth_offsets_cm:
        raise ValueError("--depth-offsets-cm must contain 0")

    args.data_dir = args.data_dir.expanduser().resolve()
    args.ply = (
        args.ply.expanduser().resolve()
        if args.ply is not None
        else newest_training_ply(
            args.data_dir.expanduser().resolve()
            / "gsplat_voxel_plane_mcmc"
            / "ply"
        )
    )
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.data_dir / "ncc_geometry_validation_pca"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    parser = Parser(
        data_dir=str(args.data_dir),
        factor=args.data_factor,
        normalize=False,
        test_every=0,
        load_exposure=False,
    )
    dataset = Dataset(parser, split="train")
    covisibility = build_covisibility(parser, args.data_dir)
    renderer = FrozenRenderer(args.ply, device)

    source_map = {
        ref: choose_sources(ref, covisibility, parser.camtoworlds, args)
        for ref in range(len(dataset))
    }
    eligible_refs = [ref for ref, sources in source_map.items() if sources]
    if not eligible_refs:
        raise RuntimeError("No reference image has a valid SfM-covisible source")
    num_refs = min(args.num_refs, len(eligible_refs))
    positions = np.linspace(0, len(eligible_refs) - 1, num_refs, dtype=int)
    ref_indices = [eligible_refs[position] for position in positions]
    offsets_m = [value / 100.0 for value in args.depth_offsets_cm]
    rows = []
    all_losses: Dict[float, List[torch.Tensor]] = defaultdict(list)
    rendered_cache: OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]] = (
        OrderedDict()
    )

    def render(index: int, data: Dict[str, torch.Tensor]):
        if index not in rendered_cache:
            depth, alpha = renderer(data, device)
            rendered_cache[index] = (depth.cpu(), alpha.cpu())
            if len(rendered_cache) > 8:
                rendered_cache.popitem(last=False)
        else:
            rendered_cache.move_to_end(index)
        return tuple(value.to(device) for value in rendered_cache[index])

    for ref in tqdm(ref_indices, desc="Reference images"):
        ref = int(ref)
        source_rows = source_map[ref]
        ref_data = dataset[ref]
        ref_image = ref_data["image"].to(device) / 255.0
        ref_mask = ref_data["mask"].to(device)
        K_ref = ref_data["K"].to(device)
        c2w_ref = ref_data["camtoworld"].to(device)
        ref_depth, ref_alpha = render(ref, ref_data)
        valid_centers = depth_geometry(ref_depth, ref_alpha, ref_mask, args)
        centers = textured_centers(ref_image, valid_centers, args, rng)
        centers, normals, plane_distances, plane_rmse, plane_ratio = (
            fit_local_planes(
                ref_depth,
                ref_alpha,
                ref_mask,
                K_ref,
                centers,
                args,
                rng,
            )
        )
        save_reference_debug(
            args.output_dir,
            parser.image_names[ref],
            ref_image,
            normals,
            centers,
        )

        for source, shared in source_rows:
            source_data = dataset[source]
            source_image = source_data["image"].to(device) / 255.0
            source_mask = source_data["mask"].to(device)
            source_depth, source_alpha = render(source, source_data)
            losses_by_offset = {}
            valid_by_offset = {}
            zero_source_patches = None
            for offset_cm, offset_m in zip(args.depth_offsets_cm, offsets_m):
                loss, valid, source_patches = plane_warp(
                    ref_image,
                    source_image,
                    source_depth,
                    source_alpha,
                    source_mask,
                    centers,
                    ref_depth,
                    normals,
                    plane_distances,
                    K_ref,
                    source_data["K"].to(device),
                    c2w_ref,
                    source_data["camtoworld"].to(device),
                    offset_m,
                    offset_cm == 0,
                    args,
                )
                losses_by_offset[offset_cm] = loss
                valid_by_offset[offset_cm] = valid
                if offset_cm == 0:
                    zero_source_patches = source_patches

            common = torch.stack(list(valid_by_offset.values())).all(0)
            if common.sum() == 0:
                continue
            row = {
                "reference": parser.image_names[ref],
                "source": parser.image_names[source],
                "shared_sfm_points": shared,
                "valid_patches": int(common.sum().item()),
                "plane_rmse_median": float(plane_rmse[common].median().item()),
                "plane_eigen_ratio_median": float(
                    plane_ratio[common].median().item()
                ),
            }
            for offset_cm in args.depth_offsets_cm:
                values = losses_by_offset[offset_cm][common]
                row[f"loss_{offset_cm:+g}cm"] = float(values.mean().item())
                all_losses[offset_cm].append(values.cpu())
            rows.append(row)

            save_patch_mosaic(
                args.output_dir
                / "patches"
                / (
                    f"{Path(parser.image_names[ref]).stem}__"
                    f"{Path(parser.image_names[source]).stem}.png"
                ),
                ref_image,
                centers,
                zero_source_patches,
                common,
                args,
            )

    if not rows:
        raise RuntimeError(
            "No valid image pair/patch remained. Inspect source thresholds, masks "
            "and the saved reference diagnostics."
        )

    with (args.output_dir / "pairs.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    mean_curve = {}
    aligned_losses = []
    for offset_cm in args.depth_offsets_cm:
        values = torch.cat(all_losses[offset_cm])
        aligned_losses.append(values)
        mean_curve[str(offset_cm)] = {
            "mean": float(values.mean().item()),
            "median": float(values.median().item()),
            "count": int(values.numel()),
        }
    with (args.output_dir / "depth_curve.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["depth_offset_cm", "mean_ncc_loss", "median_ncc_loss"])
        for offset_cm in args.depth_offsets_cm:
            statistics = mean_curve[str(offset_cm)]
            writer.writerow([offset_cm, statistics["mean"], statistics["median"]])
    best_offset = min(
        args.depth_offsets_cm, key=lambda value: mean_curve[str(value)]["mean"]
    )
    loss_matrix = torch.stack(aligned_losses)
    zero_index = args.depth_offsets_cm.index(0.0)
    zero_best_fraction = float(
        (loss_matrix.argmin(dim=0) == zero_index).float().mean().item()
    )
    zero_loss = mean_curve["0.0"]["mean"]
    nonzero_loss = min(
        mean_curve[str(value)]["mean"]
        for value in args.depth_offsets_cm
        if value != 0
    )
    summary = {
        "method": "11x11_3d_pca_plane_homography_ncc",
        "data_dir": str(args.data_dir),
        "ply": str(args.ply),
        "num_images": len(dataset),
        "num_reference_source_pairs": len(rows),
        "patch_size": args.patch_size,
        "depth_offsets_cm": args.depth_offsets_cm,
        "loss_curve": mean_curve,
        "best_depth_offset_cm": best_offset,
        "zero_is_global_minimum": best_offset == 0,
        "zero_is_best_patch_fraction": zero_best_fraction,
        "zero_margin_to_best_perturbation": nonzero_loss - zero_loss,
        "interpretation": (
            "PASS: unperturbed depth has the lowest mean NCC loss"
            if best_offset == 0
            else "FAIL: NCC prefers a perturbed depth; inspect calibration, depth, "
            "normal, occlusion and warp visualizations before training with NCC"
        ),
    }
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
