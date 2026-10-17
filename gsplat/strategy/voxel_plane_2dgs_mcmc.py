from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from scipy.spatial import cKDTree
from torch import Tensor

from gsplat.utils import normalized_quat_to_rotmat

from .texture_mcmc import TextureMCMCStrategy


def read_plane_anchors(path: str):
    vertex = PlyData.read(path)["vertex"]
    names = vertex.data.dtype.names
    required = ("x", "y", "z", "nx", "ny", "nz", "is_plane")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"plane-anchor PLY is missing properties: {missing}")

    def columns(keys):
        return np.stack([np.asarray(vertex[key]) for key in keys], axis=1)

    xyz = columns(("x", "y", "z")).astype(np.float32)
    source_keys = ("source_x", "source_y", "source_z")
    match_xyz = (
        columns(source_keys).astype(np.float32)
        if all(key in names for key in source_keys)
        else xyz
    )
    normals = columns(("nx", "ny", "nz")).astype(np.float32)
    is_plane = np.asarray(vertex["is_plane"], dtype=np.uint8) != 0
    if len(xyz) == 0 or not np.isfinite(xyz).all() or not np.isfinite(match_xyz).all():
        raise ValueError("plane-anchor PLY contains empty or invalid XYZ data")
    lengths = np.linalg.norm(normals, axis=1)
    invalid = is_plane & (~np.isfinite(normals).all(axis=1) | (lengths < 1e-6))
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} planar anchors have invalid normals")
    normals[is_plane] /= lengths[is_plane, None]
    normals[~is_plane] = 0.0
    return xyz, match_xyz, normals, is_plane


@dataclass
class VoxelPlaneMCMC2DStrategy(TextureMCMCStrategy):
    """Minimal 2DGS MCMC with static voxel-plane supervision.

    Initial Gaussians are matched once to their nearest anchor. Added and
    relocated Gaussians inherit the donor anchor. Anchor position and normal
    constraints are differentiable losses; this first version deliberately has
    no dynamic anchor reassignment or free-GS state. Texture statistics use
    pure GS/pixel ownership counts, and one GS is retained per occupied anchor.
    """

    plane_anchor_ply: str | None = None
    plane_position_lambda: float = 0.1
    plane_normal_lambda: float = 0.01
    plane_loss_start_iter: int = 0
    plane_position_beta: float = 0.02
    initial_tangent_scale_multiplier: float = 1.5

    use_opacity_score: bool = True
    use_grad_score: bool = True
    use_texture_score: bool = False
    use_fastgs_score: bool = False
    opacity_weight: float = 1.0
    texture_weight: float = 0.0
    grad_weight: float = 1.0
    absgrad: bool = True
    texture_update_every: int = 10
    active_noise_steps: int = 500
    keep_one_per_anchor: bool = True

    def initialize_state(self) -> Dict[str, Any]:
        state = super().initialize_state()
        state.update(
            anchor_xyz=None,
            anchor_normals=None,
            anchor_is_plane=None,
            anchor_ids=None,
            anchor_keeper_ids=None,
            visible_gs_ids=None,
        )
        return state

    def check_sanity(self, params, optimizers):
        super().check_sanity(params, optimizers)
        if not self.plane_anchor_ply:
            raise ValueError("plane_anchor_ply is required")
        if min(self.plane_position_lambda, self.plane_normal_lambda) < 0:
            raise ValueError("plane loss weights must be non-negative")
        if self.plane_position_beta <= 0:
            raise ValueError("plane_position_beta must be positive")
        if self.initial_tangent_scale_multiplier <= 0:
            raise ValueError("initial_tangent_scale_multiplier must be positive")
        if self.texture_update_every <= 0:
            raise ValueError("texture_update_every must be positive")

    @torch.no_grad()
    def set_surface_anchors(self, state: Dict[str, Any], means: Tensor) -> None:
        super().set_surface_anchors(state, means)
        xyz, match_xyz, normals, is_plane = read_plane_anchors(self.plane_anchor_ply)
        distances, anchor_ids = cKDTree(match_xyz).query(
            means.detach().float().cpu().numpy(), k=1, workers=-1
        )
        if not np.isfinite(distances).all():
            raise ValueError("some initial Gaussians have no finite plane anchor")

        device = means.device
        used_anchor_ids, remapped_anchor_ids = np.unique(
            anchor_ids, return_inverse=True
        )
        removed_empty = len(xyz) - len(used_anchor_ids)
        xyz = xyz[used_anchor_ids]
        normals = normals[used_anchor_ids]
        is_plane = is_plane[used_anchor_ids]
        anchor_ids = remapped_anchor_ids

        state["anchor_xyz"] = torch.as_tensor(xyz, device=device, dtype=means.dtype)
        state["anchor_normals"] = torch.as_tensor(
            normals, device=device, dtype=means.dtype
        )
        state["anchor_is_plane"] = torch.as_tensor(
            is_plane, device=device, dtype=torch.bool
        )
        state["anchor_ids"] = torch.as_tensor(
            anchor_ids, device=device, dtype=torch.long
        )
        q = np.quantile(distances, [0.5, 0.9, 0.95, 0.99, 1.0])
        print(
            "2DGS anchor match distance [m]: "
            f"P50={q[0]:.4f}, P90={q[1]:.4f}, P95={q[2]:.4f}, "
            f"P99={q[3]:.4f}, max={q[4]:.4f}; "
            f"planar GS={int(is_plane[anchor_ids].sum()):,}/{len(anchor_ids):,}; "
            f"active anchors={len(xyz):,}, removed empty={removed_empty:,}."
        )

    @torch.no_grad()
    def _dead_mask(self, opacity: Tensor) -> Tensor:
        dead = super()._dead_mask(opacity)
        if not self.keep_one_per_anchor:
            return dead

        state = getattr(self, "_active_state", None)
        if state is None or state.get("anchor_ids") is None or opacity.numel() == 0:
            return dead
        anchor_ids = state["anchor_ids"]
        if anchor_ids.shape != opacity.shape:
            raise RuntimeError("anchor_ids and opacity must have identical shapes")

        ranked_opacity = torch.nan_to_num(opacity, nan=-torch.inf)
        num_anchors = len(state["anchor_xyz"])
        max_opacity = torch.full(
            (num_anchors,), -torch.inf, device=opacity.device, dtype=opacity.dtype
        )
        max_opacity.scatter_reduce_(
            0, anchor_ids, ranked_opacity, reduce="amax", include_self=True
        )

        indices = torch.arange(opacity.numel(), device=opacity.device)
        candidates = torch.where(
            ranked_opacity == max_opacity[anchor_ids], indices, opacity.numel()
        )
        keepers = torch.full(
            (num_anchors,), opacity.numel(), device=opacity.device, dtype=torch.long
        )
        keepers.scatter_reduce_(
            0, anchor_ids, candidates, reduce="amin", include_self=True
        )
        keepers = keepers[keepers < opacity.numel()]
        dead[keepers] = False
        state["anchor_keeper_ids"] = keepers
        return dead

    @torch.no_grad()
    def initialize_plane_gaussians(self, params, state: Dict[str, Any]) -> None:
        """Align each planar 2DGS local Z axis with its anchor normal."""
        anchor_ids = state["anchor_ids"]
        indices = torch.where(state["anchor_is_plane"][anchor_ids])[0]
        if indices.numel() == 0:
            return

        normals = state["anchor_normals"][anchor_ids[indices]]
        quats = torch.stack(
            (
                1.0 + normals[:, 2],
                -normals[:, 1],
                normals[:, 0],
                torch.zeros_like(normals[:, 0]),
            ),
            dim=-1,
        )
        antiparallel = quats.norm(dim=-1) < 1e-8
        if antiparallel.any():
            quats[antiparallel] = quats.new_tensor((0.0, 1.0, 0.0, 0.0))
        params["quats"][indices] = F.normalize(quats, dim=-1, eps=1e-8)
        params["scales"][indices, :2] = (
            params["scales"][indices, :2]
            + math.log(self.initial_tangent_scale_multiplier)
        )
        if self.verbose:
            print(
                f"Initialized {indices.numel():,} planar 2DGS normals and "
                f"tangent scales (x{self.initial_tangent_scale_multiplier:g})."
            )

    @staticmethod
    def _tracked_state(state: Dict[str, Any]) -> Dict[str, Tensor]:
        # Only per-GS score accumulators are extended by sample_add. Static
        # anchor tables and anchor IDs are handled explicitly below.
        return {
            key: state[key]
            for key in (
                "texture_info",
                "texture_pixel_num",
                "absgrad_sum",
                "grad_count",
            )
            if isinstance(state.get(key), Tensor)
        }

    @torch.no_grad()
    def _inherit_surface_anchors(
        self,
        state: Dict[str, Any],
        indices: Tensor,
        source_indices: Tensor,
        append: bool = False,
    ) -> None:
        super()._inherit_surface_anchors(state, indices, source_indices, append)
        anchor_ids = state.get("anchor_ids")
        if anchor_ids is None or indices.numel() == 0:
            return
        inherited = anchor_ids[source_indices].clone()
        if append:
            state["anchor_ids"] = torch.cat((anchor_ids, inherited))
        else:
            anchor_ids[indices] = inherited

    @torch.no_grad()
    def step_pre_backward(self, params, optimizers, state, step, info):
        del params, optimizers, step
        n = len(state["anchor_ids"])
        radii = info["radii"]
        if radii.ndim == 3:
            visible = (radii > 0).all(dim=-1).any(dim=0)
            state["visible_gs_ids"] = torch.where(visible)[0]
        else:
            selected = info["gaussian_ids"][(radii > 0).all(dim=-1)]
            state["visible_gs_ids"] = torch.unique(selected)
        if state["visible_gs_ids"].numel() > 0:
            if int(state["visible_gs_ids"].max()) >= n:
                raise RuntimeError("visible Gaussian ID is outside anchor_ids")

    @torch.no_grad()
    def _update_statistics(self, params, state, info):
        n = len(params["means"])
        device = params["means"].device
        self._visible_noise_mask = torch.zeros(n, dtype=torch.bool, device=device)

        if self.use_texture_score and "texture_info" in info:
            texture = info["texture_info"].detach()
            count = info["texture_pixel_num"].detach()
            if texture.shape != (n,) or count.shape != (n,):
                raise ValueError("texture statistics must have shape [num_gaussians]")
            if state["texture_info"] is None:
                state["texture_info"] = torch.zeros_like(texture)
                state["texture_pixel_num"] = torch.zeros_like(count)
            state["texture_info"].add_(texture)
            state["texture_pixel_num"].add_(count)

        gradient = info.get("gradient_2dgs")
        if gradient is None:
            raise KeyError("2DGS MCMC requires info['gradient_2dgs']")
        grads = getattr(gradient, "absgrad", None)
        if grads is None:
            grads = gradient.grad
            if grads is not None:
                grads = grads.abs()
        if grads is None:
            raise RuntimeError("gradient_2dgs has no gradient; enable absgrad")

        grads = grads.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]
        if grads.ndim == 3:
            selected = (info["radii"] > 0).all(dim=-1)
            ids = torch.where(selected)[1]
            grads = grads[selected]
        else:
            ids = info["gaussian_ids"]

        norms = grads.norm(dim=-1)
        positive = norms > 0
        ids = ids[positive]
        norms = norms[positive]
        self._visible_noise_mask[ids] = True
        if state["absgrad_sum"] is None:
            state["absgrad_sum"] = torch.zeros(n, device=device)
            state["grad_count"] = torch.zeros(n, device=device)
        state["absgrad_sum"].index_add_(0, ids, norms)
        state["grad_count"].index_add_(0, ids, torch.ones_like(norms))

    @torch.no_grad()
    def _inject_active_noise(self, params, indices: Tensor, lr: float) -> None:
        del lr
        if indices.numel() == 0 or self.active_noise_steps == 0:
            return
        indices = indices[indices < self._visible_noise_mask.numel()]
        indices = indices[self._visible_noise_mask[indices]]
        if indices.numel() == 0:
            return

        quats = params["quats"][indices]
        log_scales = params["scales"][indices, :2]
        valid = (
            torch.isfinite(quats).all(dim=-1)
            & torch.isfinite(log_scales).all(dim=-1)
            & torch.isfinite(params["means"][indices]).all(dim=-1)
            & torch.isfinite(params["opacities"][indices].flatten())
            & (quats.norm(dim=-1) > 1e-8)
        )
        indices = indices[valid]
        if indices.numel() == 0:
            return
        quats = F.normalize(quats[valid], dim=-1, eps=1e-8)
        tangent_scales = torch.exp(log_scales[valid]).square()
        finite_scale = torch.isfinite(tangent_scales).all(dim=-1)
        indices = indices[finite_scale]
        if indices.numel() == 0:
            return
        rotations = normalized_quat_to_rotmat(quats[finite_scale])
        tangent_scales = tangent_scales[finite_scale]
        eps = torch.randn_like(tangent_scales) * tangent_scales
        delta = (
            rotations[:, :, 0] * eps[:, :1]
            + rotations[:, :, 1] * eps[:, 1:2]
        )
        opacity = torch.sigmoid(params["opacities"][indices].flatten())
        gate = torch.sigmoid(
            -self.noise_opacity_k * (opacity - self.noise_opacity_t)
        ).unsqueeze(-1)
        delta = delta * gate * self.noise_scale_at_step(self._active_step)
        finite_delta = torch.isfinite(delta).all(dim=-1)
        params["means"][indices[finite_delta]].add_(delta[finite_delta])

    def compute_losses(self, params, state, step: int) -> Dict[str, Tensor]:
        # Do not derive zero from all means: a single unrelated invalid GS
        # would otherwise turn every empty plane loss into NaN.
        zero = params["means"].new_zeros(())
        if step < self.plane_loss_start_iter:
            return {"loss": zero, "plane_position": zero, "plane_normal": zero}
        visible = state.get("visible_gs_ids")
        if visible is None or visible.numel() == 0:
            return {"loss": zero, "plane_position": zero, "plane_normal": zero}

        anchor_ids = state["anchor_ids"][visible]
        planar = state["anchor_is_plane"][anchor_ids]
        ids = visible[planar]
        anchor_ids = anchor_ids[planar]
        if ids.numel() == 0:
            return {"loss": zero, "plane_position": zero, "plane_normal": zero}

        means = params["means"][ids]
        quats = params["quats"][ids]
        anchor_xyz = state["anchor_xyz"][anchor_ids]
        anchor_normals = state["anchor_normals"][anchor_ids]

        position_valid = (
            torch.isfinite(means).all(dim=-1)
            & torch.isfinite(anchor_xyz).all(dim=-1)
            & torch.isfinite(anchor_normals).all(dim=-1)
        )
        signed_distance = (
            (means[position_valid] - anchor_xyz[position_valid])
            * anchor_normals[position_valid]
        ).sum(-1)
        signed_distance = signed_distance[torch.isfinite(signed_distance)]
        position_loss = (
            F.smooth_l1_loss(
                signed_distance,
                torch.zeros_like(signed_distance),
                beta=self.plane_position_beta,
            )
            if signed_distance.numel() > 0
            else zero
        )

        quat_norm = quats.norm(dim=-1)
        normal_valid = (
            torch.isfinite(quats).all(dim=-1)
            & torch.isfinite(quat_norm)
            & (quat_norm > 1e-8)
            & torch.isfinite(anchor_normals).all(dim=-1)
        )
        if normal_valid.any():
            normalized_quats = F.normalize(
                quats[normal_valid], dim=-1, eps=1e-8
            )
            gs_normals = normalized_quat_to_rotmat(normalized_quats)[:, :, 2]
            cosine = (
                (gs_normals * anchor_normals[normal_valid])
                .sum(-1)
                .abs()
                .clamp(max=1.0)
            )
            normal_loss = (1.0 - cosine).mean()
        else:
            normal_loss = zero
        total = (
            self.plane_position_lambda * position_loss
            + self.plane_normal_lambda * normal_loss
        )
        return {
            "loss": total,
            "plane_position": position_loss.detach(),
            "plane_normal": normal_loss.detach(),
        }

    def step_post_backward(
        self,
        params,
        optimizers,
        state,
        step,
        info,
        packed=False,
        scene=None,
    ):
        del packed
        super().step_post_backward(
            params=params,
            optimizers=optimizers,
            state=state,
            step=step,
            info=info,
            lr=0.0,
            scene=scene,
        )
