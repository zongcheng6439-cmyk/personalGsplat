"""Optional multi-view patch NCC loss for voxel-plane training.

The loss is isolated here; cached LiDAR view selection is in
``ncc_view_selection.py``. Removing NCC only requires deleting these two files
and the small, marked NCC block in ``train_voxel_plane_mcmc.py``.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from functools import lru_cache
from typing import Tuple

import torch
import torch.nn.functional as F

from datasets.colmap_adc import Dataset
from ncc_view_selection import choose_depth_sources


def _grayscale(image: torch.Tensor) -> torch.Tensor:
    return (image * image.new_tensor([0.299, 0.587, 0.114])).sum(-1)


@lru_cache(maxsize=8)
def _patch_offsets(size: int, device: torch.device) -> torch.Tensor:
    radius = size // 2
    oy, ox = torch.meshgrid(
        torch.arange(-radius, radius + 1, device=device),
        torch.arange(-radius, radius + 1, device=device),
        indexing="ij",
    )
    return torch.stack([ox.flatten(), oy.flatten()], dim=-1)


def _patch_pixels(centers: torch.Tensor, size: int) -> torch.Tensor:
    return centers[:, None] + _patch_offsets(size, centers.device)[None]


def _safe_denominator(value: torch.Tensor) -> torch.Tensor:
    return torch.where(
        value.abs() < 1e-6,
        value.sign().masked_fill(value == 0, 1.0) * 1e-6,
        value,
    )


def _project_plane_patch(homogeneous, normals, distance, K_reference, K_source, transform):
    """Evaluate the original plane homography and depth using shared rays.

    H*p = K_source * (R*ray + t*(n*ray)/d). Keep this homogeneous
    scaling (and its clamp) exactly as before, including for invalid rays.
    """
    rays = torch.einsum("ij,npj->npi", torch.linalg.inv(K_reference), homogeneous)
    rotated = torch.einsum("ij,npj->npi", transform[:3, :3], rays)
    normal_dot = torch.einsum("ni,npi->np", normals, rays)
    distance = _safe_denominator(distance)[:, None]
    projected = torch.einsum(
        "ij,npj->npi", K_source,
        rotated + transform[:3, 3] * (normal_dot / distance)[..., None],
    )
    xy = projected[..., :2] / projected[..., 2:].clamp_min(1e-6)
    depth = rotated[..., 2] * (distance / _safe_denominator(normal_dot)) + transform[2, 3]
    return xy, depth


class PatchNCCLoss:
    """Select a LiDAR-covisible source and compute plane-warp ZNCC."""

    patch_size = 9
    plane_window = 11
    min_alpha = 0.8
    min_patch_std = 0.03
    plane_max_rmse = 0.02
    plane_max_eigen_ratio = 0.05
    depth_edge_abs = 0.20
    depth_edge_rel = 0.02
    edge_dilation = 5
    occlusion_abs = 0.15
    occlusion_rel = 0.02
    # Rendered-plane mode: minimum blended-normal norm as a consistency gate.
    min_plane_normal_norm = 0.5

    def __init__(self, parser, train_indices, data_dir: str, samples: int):
        self.dataset = Dataset(parser, split="train", load_depths=True)
        self.samples = samples
        self.cache: OrderedDict[int, dict] = OrderedDict()
        if list(map(int, train_indices)) != list(map(int, self.dataset.indices)):
            raise ValueError("NCC and training must use the same image indexing")
        self.source_items = choose_depth_sources(self.dataset, data_dir)
        if not self.source_items:
            raise ValueError("NCC found no depth-consistent source views; check poses/depth/masks")
        print(
            f"NCC: found LiDAR-covisible sources for "
            f"{len(self.source_items)}/{len(train_indices)} training images",
            flush=True,
        )

    def _source(self, item: int) -> dict:
        if item not in self.cache:
            data = self.dataset[item]
            if "depth_gs" not in data:
                raise KeyError("NCC requires the dense depth image for each source view")
            self.cache[item] = {
                key: data[key] for key in
                ("mask", "depth_gs", "depth_gs_mask", "K", "camtoworld")
            }
            # Preserve the original uint8 conversion, but cache one gray channel.
            self.cache[item]["gray"] = _grayscale(data["image"].to(torch.uint8).float() / 255.0)
            if len(self.cache) > 8:
                self.cache.popitem(last=False)
        self.cache.move_to_end(item)
        return self.cache[item]

    def _candidate_centers(
        self,
        gray: torch.Tensor,
        depth: torch.Tensor,
        alpha: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            valid = (
                torch.isfinite(depth)
                & (depth > 0)
                & (alpha >= self.min_alpha)
                & mask
            )
            jump = torch.zeros_like(valid)
            horizontal = (depth[:, 1:] - depth[:, :-1]).abs()
            horizontal_limit = torch.maximum(
                depth[:, 1:].minimum(depth[:, :-1]) * self.depth_edge_rel,
                depth.new_tensor(self.depth_edge_abs),
            )
            horizontal = horizontal > horizontal_limit
            jump[:, 1:] |= horizontal
            jump[:, :-1] |= horizontal
            vertical = (depth[1:] - depth[:-1]).abs()
            vertical_limit = torch.maximum(
                depth[1:].minimum(depth[:-1]) * self.depth_edge_rel,
                depth.new_tensor(self.depth_edge_abs),
            )
            vertical = vertical > vertical_limit
            jump[1:] |= vertical
            jump[:-1] |= vertical
            jump = F.max_pool2d(
                jump.float()[None, None],
                self.edge_dilation,
                stride=1,
                padding=self.edge_dilation // 2,
            )[0, 0].bool()
            valid &= ~jump

            border = max(self.patch_size, self.plane_window) // 2 + 1
            valid[:border] = valid[-border:] = False
            valid[:, :border] = valid[:, -border:] = False
            mean = F.avg_pool2d(
                gray[None, None], self.patch_size, 1, self.patch_size // 2
            )[0, 0]
            square_mean = F.avg_pool2d(
                gray.square()[None, None],
                self.patch_size,
                1,
                self.patch_size // 2,
            )[0, 0]
            std = (square_mean - mean.square()).clamp_min(0).sqrt()
            centers = torch.nonzero(
                valid & (std >= self.min_patch_std), as_tuple=False
            )[:, [1, 0]]
            limit = self.samples * 4
            if len(centers) > limit:
                centers = centers[
                    torch.randperm(len(centers), device=centers.device)[:limit]
                ]
        return centers

    def _plane_points(self, depth, alpha, mask, K, centers):
        """Backproject a set of depth windows; the caller controls autograd."""
        xy = _patch_pixels(centers, self.plane_window).long()
        x, y = xy[..., 0], xy[..., 1]
        local_depth = depth[y, x]
        valid = (
            torch.isfinite(local_depth)
            & (local_depth > 0)
            & (alpha[y, x] >= self.min_alpha)
            & mask[y, x]
        )
        rays = torch.stack(
            [
                (x - K[0, 2]) / K[0, 0],
                (y - K[1, 2]) / K[1, 1],
                torch.ones_like(local_depth),
            ],
            dim=-1,
        ).to(depth.dtype)
        points = rays * torch.where(valid, local_depth, 0.0)[..., None]
        return points, valid.to(depth.dtype)

    @torch.no_grad()
    def _select_planes(
        self,
        depth: torch.Tensor,
        alpha: torch.Tensor,
        mask: torch.Tensor,
        K: torch.Tensor,
        centers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit/filter candidates without retaining their backward intermediates."""
        if not len(centers):
            return centers, depth.new_empty((0, 3))

        points, weights = self._plane_points(depth, alpha, mask, K, centers)
        count = weights.sum(-1)
        # Reject insufficient depth support before the batched eigensolver.
        enough = count >= self.plane_window**2 * 0.9
        centers, points, weights, count = (
            value[enough] for value in (centers, points, weights, count)
        )
        if not len(centers):
            return centers, depth.new_empty((0, 3))
        denominator = count.clamp_min(1.0)
        centroid = (points * weights[..., None]).sum(1) / denominator[:, None]
        # Keep the float64 PCA fix; casting after covariance overflow is too late.
        centered = points.double() - centroid.double()[:, None]
        covariance = torch.einsum(
            "npi,npj,np->nij", centered, centered, weights.double()
        ) / denominator[:, None, None]
        fit_ids = torch.where(torch.isfinite(covariance).all(dim=(-2, -1)))[0]
        if not len(fit_ids):
            return centers[:0], depth.new_empty((0, 3))
        # Bound the CUDA eigensolver workspace even for large candidate batches.
        values, vectors = zip(*(
            torch.linalg.eigh(batch) for batch in covariance[fit_ids].split(1024)
        ))
        eigenvalues, eigenvectors = torch.cat(values), torch.cat(vectors)
        normals = eigenvectors[..., 0].to(depth.dtype)
        normals = torch.where(
            (normals * centroid[fit_ids]).sum(-1, keepdim=True) > 0, -normals, normals
        )
        plane_distance = (normals * centroid[fit_ids]).sum(-1)
        keep = (
            (eigenvalues[:, 1] > 1e-10)
            & (eigenvalues[:, 0].clamp_min(0).sqrt() <= self.plane_max_rmse)
            & (
                eigenvalues[:, 0]
                / eigenvalues[:, 1].clamp_min(1e-12)
                <= self.plane_max_eigen_ratio
            )
            & torch.isfinite(normals).all(-1)
            & (plane_distance.abs() > 1e-6)
        )
        ids = torch.where(keep)[0]
        if len(ids) > self.samples:
            ids = ids[torch.randperm(len(ids), device=ids.device)[: self.samples]]
        return centers[fit_ids[ids]], normals[ids]

    def _fit_planes(self, depth, alpha, mask, K, centers):
        centers, normals = self._select_planes(depth, alpha, mask, K, centers)
        if not len(centers):
            return centers, normals, depth.new_empty(0)
        # Rebuild only selected centroids with gradients. Normal estimation stays
        # detached, but plane distance still optimizes the rendered depth (and K).
        points, weights = self._plane_points(depth, alpha, mask, K, centers)
        centroid = (points * weights[..., None]).sum(1) / weights.sum(-1)[:, None]
        return centers, normals, (normals * centroid).sum(-1)

    def _rendered_planes(self, plane_normals, plane_distances, centers):
        """Gather rendered per-pixel planes; both maps keep their gradients.

        Used by the PGSR path: planes come from the alpha-blended normal and
        distance maps instead of a detached PCA fit, so gradients reach the
        Gaussian rotations and positions through the homography warp.
        """
        if not len(centers):
            return (
                centers,
                plane_normals.new_empty((0, 3)),
                plane_distances.new_empty(0),
            )
        x, y = centers[:, 0].long(), centers[:, 1].long()
        normals = plane_normals[y, x]
        distances = plane_distances[y, x]
        with torch.no_grad():
            keep = (
                torch.isfinite(normals).all(-1)
                & torch.isfinite(distances)
                & (normals.norm(dim=-1) >= self.min_plane_normal_norm)
                & (distances.abs() > 1e-6)
            )
        centers, normals, distances = centers[keep], normals[keep], distances[keep]
        if len(centers) > self.samples:
            ids = torch.randperm(len(centers), device=centers.device)[: self.samples]
            centers, normals, distances = (
                centers[ids],
                normals[ids],
                distances[ids],
            )
        return centers, normals, distances

    def __call__(
        self,
        reference_item: int,
        reference_image: torch.Tensor,
        reference_depth: torch.Tensor,
        reference_alpha: torch.Tensor,
        reference_mask: torch.Tensor,
        K_reference: torch.Tensor,
        c2w_reference: torch.Tensor,
        pose_adjust=None,
        plane_normals: torch.Tensor = None,
        plane_distances: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        if (plane_normals is None) != (plane_distances is None):
            raise ValueError(
                "plane_normals and plane_distances must be provided together"
            )
        sources = self.source_items.get(reference_item)
        if not sources:
            return reference_depth.reshape(-1)[:0].sum(), {"patches": 0, "source": -1}
        source_item = random.choice(sources)

        device = reference_depth.device
        source = {
            key: value.to(device, non_blocking=True)
            for key, value in self._source(source_item).items()
        }
        source_depth = source["depth_gs"]
        source_mask = source["mask"] & source["depth_gs_mask"].bool()
        c2w_source = source["camtoworld"]
        if pose_adjust is not None:
            source_id = torch.tensor([source_item], device=device)
            c2w_source = pose_adjust(c2w_source[None], source_id)[0]

        reference_gray = _grayscale(reference_image)
        centers = self._candidate_centers(
            reference_gray,
            reference_depth.detach(),
            reference_alpha.detach(),
            reference_mask,
        )
        if plane_normals is None:
            centers, normals, plane_distance = self._fit_planes(
                reference_depth,
                reference_alpha,
                reference_mask,
                K_reference,
                centers,
            )
        else:
            centers, normals, plane_distance = self._rendered_planes(
                plane_normals, plane_distances, centers
            )
        if not len(centers):
            return reference_depth.reshape(-1)[:0].sum(), {
                "patches": 0,
                "source": source_item,
            }

        xy = _patch_pixels(centers, self.patch_size).to(reference_depth.dtype)
        homogeneous = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)
        transform = torch.linalg.inv(c2w_source) @ c2w_reference
        source_xy, projected_depth = _project_plane_patch(
            homogeneous, normals, plane_distance, K_reference, source["K"], transform
        )

        source_height, source_width = source_depth.shape
        grid = torch.stack(
            [
                2.0 * source_xy[..., 0] / max(source_width - 1, 1) - 1.0,
                2.0 * source_xy[..., 1] / max(source_height - 1, 1) - 1.0,
            ],
            dim=-1,
        )

        def sample(value: torch.Tensor) -> torch.Tensor:
            return F.grid_sample(
                value[None, None],
                grid[None],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[0, 0]

        # Gray conversion is linear, so it commutes with bilinear sampling.
        source_gray = sample(source["gray"])
        with torch.no_grad():
            sampled_depth = sample(source_depth)
            # Require the full bilinear footprint to have valid source depth.
            sampled_mask = sample(source_mask.float()) > 0.999

        depth_tolerance = torch.maximum(
            projected_depth.abs() * self.occlusion_rel,
            projected_depth.new_tensor(self.occlusion_abs),
        )
        inside = (
            (source_xy[..., 0] >= 0)
            & (source_xy[..., 0] <= source_width - 1)
            & (source_xy[..., 1] >= 0)
            & (source_xy[..., 1] <= source_height - 1)
            & (projected_depth > 0)
        )

        ref_x, ref_y = xy[..., 0].long(), xy[..., 1].long()
        ref_gray = reference_gray[ref_y, ref_x]
        ref_centered = ref_gray - ref_gray.mean(-1, keepdim=True)
        source_centered = source_gray - source_gray.mean(-1, keepdim=True)
        correlation = (ref_centered * source_centered).sum(-1) / (
            ref_centered.square().sum(-1)
            * source_centered.square().sum(-1)
        ).clamp_min(1e-12).sqrt()
        losses = 1.0 - correlation.clamp(-1.0, 1.0)
        with torch.no_grad():
            valid = (
                inside.all(-1)
                & sampled_mask.all(-1)
                & reference_mask[ref_y, ref_x].all(-1)
                & (reference_alpha[ref_y, ref_x] >= self.min_alpha).all(-1)
                & (
                    (sampled_depth - projected_depth).abs()
                    <= depth_tolerance
                ).all(-1)
                & (ref_gray.std(-1) >= self.min_patch_std)
                & (source_gray.std(-1) >= self.min_patch_std)
                & torch.isfinite(losses)
            )
        if not valid.any():
            return reference_depth.reshape(-1)[:0].sum(), {
                "patches": 0,
                "source": source_item,
            }
        return losses[valid].mean(), {
            "patches": int(valid.sum().item()),
            "source": source_item,
        }
