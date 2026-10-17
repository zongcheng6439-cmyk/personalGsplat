"""Optional PGSR-style plane-parameter rendering for voxel-plane training.

Following PGSR (Chen et al., 2024), each Gaussian is flattened along its
shortest-scale axis into a local plane. One rasterization pass alpha-blends
the per-Gaussian plane normals and camera-to-plane distances through gsplat's
``extra_signals`` channels; dividing the two maps yields the unbiased
ray-plane depth. The single-view loss compares depth-derived normals with the
rendered normal map; the multi-view homography NCC reuses ``ncc_loss.py``
with these rendered planes instead of PCA-fitted ones.

Isolated like ``ncc_loss.py``: deleting this file plus the marked PGSR block
in ``train_voxel_plane_mcmc.py`` removes the feature entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from gsplat.rendering import rasterization
from gsplat.utils import depth_to_normal, normalized_quat_to_rotmat


def rgb_to_gray(image: Tensor) -> Tensor:
    return (image * image.new_tensor([0.299, 0.587, 0.114])).sum(-1)


def _safe_denominator(value: Tensor) -> Tensor:
    return torch.where(
        value.abs() < 1e-6,
        value.sign().masked_fill(value == 0, 1.0) * 1e-6,
        value,
    )


def plane_param_signals(
    means: Tensor, quats: Tensor, scales_log: Tensor, viewmats: Tensor
) -> Tensor:
    """Per-Gaussian PGSR plane parameters in each camera frame.

    Normal = shortest-scale axis transformed to the camera frame and flipped
    to face the camera (viewing-direction sign disambiguation, PGSR IV-A);
    distance = n . mu_cam, the signed distance from the camera origin to the
    Gaussian's tangent plane (negative for planes in front of the camera,
    matching the PCA-plane convention in ``ncc_loss.py``).

    Args:
        means: [N, 3], quats: [N, 4], scales_log: [N, 3] (log space; argmin is
            unaffected by the exp activation), viewmats: [C, 4, 4] world-to-cam.

    Returns:
        [C, N, 4] extra signals: (nx, ny, nz, distance).
    """
    rotations = normalized_quat_to_rotmat(F.normalize(quats, dim=-1))
    shortest = scales_log.argmin(-1)
    normals_world = rotations.gather(
        2, shortest[:, None, None].expand(-1, 3, 1)
    ).squeeze(2)
    R = viewmats[..., :3, :3]
    t = viewmats[..., :3, 3]
    centers_cam = torch.einsum("cij,nj->cni", R, means) + t[:, None]
    normals_cam = torch.einsum("cij,nj->cni", R, normals_world)
    facing_away = (normals_cam * centers_cam).sum(-1, keepdim=True) > 0
    normals_cam = torch.where(facing_away, -normals_cam, normals_cam)
    distances = (normals_cam * centers_cam).sum(-1, keepdim=True)
    return torch.cat([normals_cam, distances], dim=-1)


def pixel_rays(Ks: Tensor, height: int, width: int) -> Tensor:
    """Unit-z camera rays [C, H, W, 3] at integer pixel coordinates.

    Uses the same (x - cx) / fx convention as ``ncc_loss._plane_points`` so
    depth values stay compatible with the LiDAR z-depth used elsewhere.
    """
    dtype, device = Ks.dtype, Ks.device
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(x)
    return torch.stack(
        (
            (x - Ks[:, 0, 2, None, None]) / Ks[:, 0, 0, None, None],
            (y - Ks[:, 1, 2, None, None]) / Ks[:, 1, 1, None, None],
            ones.expand(Ks.shape[0], -1, -1),
        ),
        dim=-1,
    )


def unbiased_depth(
    normal_map: Tensor, distance_map: Tensor, rays: Tensor
) -> tuple[Tensor, Tensor]:
    """PGSR Eq. (4): D(p) = Dblended / (N(p) . K^-1 p~).

    The alpha-accumulation weights cancel in the division, so the depth is the
    intersection of each pixel ray with the blended local plane (a camera
    z-depth, since rays have z = 1). Returns (depth, raw denominator); callers
    should gate pixels whose |denominator| is tiny (grazing angles or
    inconsistent blended normals).
    """
    denominator = (normal_map * rays).sum(-1)
    return distance_map / _safe_denominator(denominator), denominator


@dataclass
class PlaneRender:
    normal_map: Tensor  # [C, H, W, 3] blended plane normals (unnormalized)
    distance_map: Tensor  # [C, H, W] blended camera-to-plane distances
    depth: Tensor  # [C, H, W] unbiased ray-plane z-depth
    ray_dot: Tensor  # [C, H, W] raw N . ray denominator before clamping
    rays: Tensor  # [C, H, W, 3]
    alpha: Tensor  # [C, H, W]


def render_plane_params(
    means: Tensor,
    quats: Tensor,
    scales_log: Tensor,
    opacities_log: Tensor,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    *,
    rasterize_mode: str = "classic",
    camera_model: str = "pinhole",
) -> PlaneRender:
    """Render PGSR plane maps in one pass through ``extra_signals``.

    Gradients flow from the rendered maps back through the extra channels to
    quats (via normals) and to means/quats (via distances). Dummy zero colors
    keep the standard RGB kernel path; the CUDA kernel only instantiates a
    fixed set of channel counts, so the 4 geometry channels are padded to 5
    (3 colors + 5 extra = 8 channels, which is compiled; 7 is not).
    """
    viewmats = torch.linalg.inv_ex(camtoworlds).inverse
    signals = plane_param_signals(means, quats, scales_log, viewmats)
    C = signals.shape[0]
    extra = torch.cat([signals, signals.new_zeros(signals.shape[:-1] + (1,))], -1)
    colors = means.new_zeros((C, means.shape[0], 3))
    _, alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales_log.exp(),
        opacities=opacities_log.sigmoid(),
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        render_mode="RGB",
        packed=False,
        rasterize_mode=rasterize_mode,
        camera_model=camera_model,
        extra_signals=extra,
    )
    geometry = info["render_extra_signals"][..., :4]
    normal_map, distance_map = geometry[..., :3], geometry[..., 3]
    rays = pixel_rays(Ks, height, width)
    depth, ray_dot = unbiased_depth(normal_map, distance_map, rays)
    return PlaneRender(
        normal_map=normal_map,
        distance_map=distance_map,
        depth=depth,
        ray_dot=ray_dot,
        rays=rays,
        alpha=alphas[..., 0],
    )


def single_view_normal_loss(
    depth: Tensor,
    normal_map: Tensor,
    alpha: Tensor,
    gray: Tensor,
    mask: Tensor,
    K: Tensor,
    rays: Tensor,
    *,
    min_alpha: float = 0.8,
) -> Tensor:
    """PGSR Eq. (5)-(6): edge-aware L1 between depth and rendered normals.

    Args:
        depth: [H, W] unbiased z-depth (carries gradients).
        normal_map: [H, W, 3] rendered plane normals (carries gradients).
        alpha: [H, W] rendered alpha (used detached as a confidence gate).
        gray: [H, W] reference gray image for the edge-aware weight.
        mask: [H, W] bool valid-pixel mask.
        K: [3, 3] camera intrinsics.
        rays: [H, W, 3] unit-z camera rays.
    """
    eye = torch.eye(4, device=depth.device, dtype=depth.dtype)[None]
    from_depth = depth_to_normal(
        depth[..., None][None], eye, K[None], z_depth=True
    )[0]
    facing_away = (from_depth * rays).sum(-1, keepdim=True) > 0
    from_depth = torch.where(facing_away, -from_depth, from_depth)
    with torch.no_grad():
        grad_x = gray.new_zeros(gray.shape)
        grad_x[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]).abs() * 0.5
        grad_y = gray.new_zeros(gray.shape)
        grad_y[1:-1] = (gray[2:] - gray[:-2]).abs() * 0.5
        magnitude = grad_x + grad_y
        magnitude = magnitude / magnitude.max().clamp_min(1e-6)
        edge_weight = (1.0 - magnitude).square()
        valid = (
            mask
            & (alpha >= min_alpha)
            & torch.isfinite(depth)
            & (depth > 0)
            # depth_to_normal zero-pads a one-pixel border and degenerate
            # cross products; both show up as (near-)zero norms.
            & (from_depth.norm(dim=-1) >= 0.5)
        )
    difference = (from_depth - normal_map).abs().sum(-1)
    if not valid.any():
        return depth.reshape(-1)[:0].sum()
    return (edge_weight * difference)[valid].mean()
