"""Cheap camera/block prefiltering before full-resolution visibility renders."""

from __future__ import annotations

from typing import Tuple

import torch

from gsplat.rendering import rasterization


def _camera_tensors(dataset, camera_idx: int, device: torch.device):
    """Read camera metadata without loading its image, mask, or depth map."""
    data_idx = int(dataset.indices[camera_idx])
    parser = dataset.parser
    camera_id = parser.camera_ids[data_idx]
    camtoworld = torch.as_tensor(
        parser.camtoworlds[data_idx], dtype=torch.float32, device=device
    )
    K = torch.as_tensor(parser.Ks_dict[camera_id], dtype=torch.float32, device=device)
    width, height = parser.imsize_dict[camera_id]
    return camtoworld, K, int(width), int(height)


def _sphere_intersects_camera_frustum(
    center_world: torch.Tensor,
    radius: torch.Tensor,
    camtoworld: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    near_plane: float = 0.01,
) -> bool:
    """Conservative sphere/frustum test; boundary intersections are retained."""
    center_h = torch.cat([center_world, center_world.new_ones(1)])
    center_cam = (torch.linalg.inv(camtoworld) @ center_h)[:3]
    x, y, z = center_cam.unbind()
    if z + radius < near_plane:
        return False

    fx, fy = K[0, 0].abs(), K[1, 1].abs()
    cx, cy = K[0, 2], K[1, 2]
    left_slope = cx / fx.clamp_min(1e-8)
    right_slope = (width - cx) / fx.clamp_min(1e-8)
    top_slope = cy / fy.clamp_min(1e-8)
    bottom_slope = (height - cy) / fy.clamp_min(1e-8)

    planes = (
        x + left_slope * z,
        -x + right_slope * z,
        y + top_slope * z,
        -y + bottom_slope * z,
    )
    slopes = (left_slope, right_slope, top_slope, bottom_slope)
    return all(
        bool(value >= -radius * torch.sqrt(1.0 + slope.square()))
        for value, slope in zip(planes, slopes)
    )


@torch.no_grad()
def filter_camera_block_pairs(
    *,
    gaussian,
    cameras,
    block_masks: torch.Tensor,
    unassigned_mask: torch.Tensor,
    downscale: int = 8,
    alpha_threshold: float = 0.01,
    min_visible_ratio: float = 0.0005,
    min_alpha_sum: float = 0.25,
) -> Tuple[torch.Tensor, dict]:
    """Return block/camera pairs worth running through the exact SSIM test.

    The first stage performs a conservative 3D bounding-sphere/frustum test.
    Surviving pairs receive a low-resolution block-only alpha render.
    """
    if downscale < 1:
        raise ValueError("downscale must be at least 1")
    device = gaussian["means"].device
    candidates = torch.zeros_like(unassigned_mask, dtype=torch.bool)

    means = gaussian["means"]
    quats = gaussian["quats"]
    scales = torch.exp(gaussian["scales"])
    opacities = torch.sigmoid(gaussian["opacities"])

    sphere_centers = []
    sphere_radii = []
    valid_blocks = []
    for block_id, block_mask in enumerate(block_masks):
        block_means = means[block_mask]
        if len(block_means) == 0:
            sphere_centers.append(means.new_zeros(3))
            sphere_radii.append(means.new_zeros(()))
            valid_blocks.append(False)
            continue
        bbox_min = block_means.amin(dim=0)
        bbox_max = block_means.amax(dim=0)
        sphere_centers.append((bbox_min + bbox_max) * 0.5)
        # Include a conservative 3-sigma Gaussian extent so splats whose centers
        # are near a frustum edge are not incorrectly rejected.
        max_extent = scales[block_mask].amax() * 3.0
        sphere_radii.append(
            torch.linalg.vector_norm(bbox_max - bbox_min) * 0.5 + max_extent
        )
        valid_blocks.append(True)

    geometry_pairs = 0
    alpha_pairs = 0
    for camera_idx in range(unassigned_mask.shape[1]):
        block_ids = torch.where(unassigned_mask[:, camera_idx])[0]
        if len(block_ids) == 0:
            continue
        camtoworld, K, width, height = _camera_tensors(cameras, camera_idx, device)
        low_width = max(1, width // downscale)
        low_height = max(1, height // downscale)
        low_K = K.clone()
        low_K[0, :] *= low_width / width
        low_K[1, :] *= low_height / height
        viewmat = torch.linalg.inv(camtoworld)[None]

        for block_id_tensor in block_ids:
            block_id = int(block_id_tensor.item())
            if not valid_blocks[block_id]:
                continue
            if not _sphere_intersects_camera_frustum(
                sphere_centers[block_id],
                sphere_radii[block_id],
                camtoworld,
                K,
                width,
                height,
            ):
                continue
            geometry_pairs += 1

            block_mask = block_masks[block_id]
            count = int(block_mask.sum().item())
            dummy_colors = means.new_ones((count, 1))
            _, alpha, _ = rasterization(
                means=means[block_mask],
                quats=quats[block_mask],
                scales=scales[block_mask],
                opacities=opacities[block_mask],
                colors=dummy_colors,
                viewmats=viewmat,
                Ks=low_K[None],
                width=low_width,
                height=low_height,
                sh_degree=None,
                rasterize_mode="classic",
                camera_model="pinhole",
            )
            alpha = alpha[0, ..., 0]
            visible_ratio = (alpha > alpha_threshold).float().mean()
            if (
                alpha.max() > alpha_threshold
                and visible_ratio >= min_visible_ratio
                and alpha.sum() >= min_alpha_sum
            ):
                candidates[block_id, camera_idx] = True
                alpha_pairs += 1

    stats = {
        "input_pairs": int(unassigned_mask.sum().item()),
        "geometry_pairs": geometry_pairs,
        "alpha_pairs": alpha_pairs,
        "downscale": downscale,
    }
    return candidates, stats
