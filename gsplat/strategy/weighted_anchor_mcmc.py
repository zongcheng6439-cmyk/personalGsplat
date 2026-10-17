from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from .anchor_io import read_anchor_capacity_ply
from .ops import inject_noise_to_position, relocate
from .texture_mcmc import TextureMCMCStrategy


@dataclass
class WeightedAnchorMCMCStrategy(TextureMCMCStrategy):
    """Keep uniform coverage while moving redundant GSs toward detail regions.

    The PLY ``capacity`` value is treated as a relative anchor weight. It is
    used only to select low-weight source anchors and does not impose a hard
    per-anchor Gaussian capacity. Spatially overlapping GSs are relocated to
    donors sampled by persistent per-GS history scores.
    """

    anchor_ply: str | None = None
    anchor_voxel_size: float = 0.1
    initial_jitter_ratio: float = 0.5

    capacity_move_every: int = 3000
    redundant_move: bool = False
    redundant_source_anchor_fraction: float = 0.5
    redundant_min_gs_per_anchor: int = 3
    redundant_overlap_ratio: float = 0.25

    # Independent of the means optimizer learning rate. This value decays
    # exponentially to 10% over the configured training duration.
    # The inherited active_noise_steps option is ignored: copied GSs stay
    # active and move only when they receive a current-view gradient.
    noise_scale: float = 10
    noise_opacity_t: float = 0.054
    noise_opacity_k: float = 100.0
    _noise_decay_steps: int | None = field(
        default=None, init=False, repr=False
    )

    surface_anchor_constraint: bool = False
    surface_anchor_max_distance: float = 0.3

    def initialize_state(self) -> Dict[str, Any]:
        state = super().initialize_state()
        state.update(
            anchor_xyz=None,
            anchor_low_mask=None,
            anchor_ids=None,
            history_score=None,
        )
        return state

    def check_sanity(self, params, optimizers):
        super().check_sanity(params, optimizers)
        if self.anchor_voxel_size <= 0:
            raise ValueError("anchor_voxel_size must be positive")
        if not 0 <= self.initial_jitter_ratio <= 1:
            raise ValueError("initial_jitter_ratio must be in [0, 1]")
        if self.capacity_move_every < 0:
            raise ValueError("capacity_move_every must be non-negative")
        if not 0 < self.redundant_source_anchor_fraction < 1:
            raise ValueError("redundant_source_anchor_fraction must be in (0, 1)")
        if self.redundant_min_gs_per_anchor < 1:
            raise ValueError("redundant_min_gs_per_anchor must be at least 1")
        if not 0 < self.redundant_overlap_ratio <= 1:
            raise ValueError("redundant_overlap_ratio must be in (0, 1]")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")
        if self.surface_anchor_constraint and self.surface_anchor_max_distance <= 0:
            raise ValueError("surface_anchor_max_distance must be positive")

    def set_training_steps(self, max_steps: int) -> None:
        """Bind the total step count used by the noise decay schedule."""
        super().set_training_steps(max_steps)

    def noise_scale_at_step(self, step: int) -> float:
        if self._noise_decay_steps is None:
            raise RuntimeError(
                "call set_training_steps(max_steps) before training"
            )
        progress = min(max(float(step) / self._noise_decay_steps, 0.0), 1.0)
        return self.noise_scale * 0.1 ** progress

    @torch.no_grad()
    def _inject_active_noise(self, params, indices: Tensor, lr: float) -> None:
        """Inject noise only into active GSs supervised by this image."""
        del lr
        if indices.numel() == 0:
            return
        visible = self._visible_noise_mask
        indices = indices[indices < len(visible)]
        indices = indices[visible[indices]]
        if indices.numel() == 0:
            return
        active_params = {
            key: params[key][indices].detach().clone()
            for key in ("means", "quats", "scales", "opacities")
        }
        inject_noise_to_position(
            params=active_params,
            optimizers={},
            state={},
            noise_scale=self.noise_scale_at_step(self._active_step),
            t=self.noise_opacity_t,
            k=self.noise_opacity_k,
        )
        params["means"][indices] = active_params["means"]

    @torch.no_grad()
    def _update_statistics(self, params, state, info) -> None:
        super()._update_statistics(params, state, info)
        self._visible_noise_mask = torch.zeros(
            len(params["means"]), dtype=torch.bool, device=params["means"].device
        )
        ids = info.get("current_grad_ids")
        if ids is not None:
            self._visible_noise_mask[ids] = True

    @staticmethod
    def _mark_noise_active(state: Dict[str, Any], indices: Tensor) -> None:
        """Keep relocated and added GSs eligible for view-gated noise."""
        if indices.numel() > 0:
            state["noise_active_until"][indices] = torch.iinfo(torch.long).max

    @torch.no_grad()
    def _jitter_gs(self, params, indices: Tensor) -> None:
        """Spread copied GSs uniformly inside a voxel-sized 3D ball."""
        if indices.numel() == 0 or self.initial_jitter_ratio == 0:
            return
        offset = torch.randn_like(params["means"][indices])
        offset /= offset.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        radius = torch.rand_like(offset[:, :1]).pow_(1.0 / 3.0)
        radius *= self.anchor_voxel_size * self.initial_jitter_ratio
        params["means"][indices] = params["means"][indices] + offset * radius

    @torch.no_grad()
    def set_surface_anchors(self, state: Dict[str, Any], means: Tensor) -> None:
        if not self.anchor_ply:
            raise ValueError("WeightedAnchorMCMCStrategy requires anchor_ply")
        xyz, voxel_weight = read_anchor_capacity_ply(
            self.anchor_ply, expected_voxel_size=self.anchor_voxel_size
        )
        self.set_anchors(state, means, xyz, voxel_weight)

    @torch.no_grad()
    def set_anchors(
        self,
        state: Dict[str, Any],
        means: Tensor,
        anchor_xyz,
        voxel_weight,
    ) -> None:
        anchors = torch.as_tensor(
            anchor_xyz, dtype=means.dtype, device=means.device
        ).detach()
        weights = torch.as_tensor(
            voxel_weight, dtype=torch.float32, device=means.device
        ).clamp_min(0)
        if anchors.ndim != 2 or anchors.shape[1] != 3 or len(anchors) == 0:
            raise ValueError("anchor_xyz must have shape [V, 3] with V > 0")
        if weights.ndim != 1 or len(weights) != len(anchors):
            raise ValueError("voxel_weight must have shape [V]")
        if not torch.isfinite(anchors).all() or not torch.isfinite(weights).all():
            raise ValueError("anchor_xyz and voxel_weight must be finite")

        ids = self._voxel_anchor_ids(means, anchors)
        occupied = torch.bincount(ids, minlength=len(anchors)) > 0
        removed = int((~occupied).sum())
        if removed:
            ids = (occupied.long().cumsum(0) - 1)[ids]
            anchors = anchors[occupied]
            weights = weights[occupied]
        low_mask, low_threshold = self._low_mask(
            weights, self.redundant_source_anchor_fraction
        )

        state["anchor_xyz"] = anchors
        state["anchor_low_mask"] = low_mask
        state["anchor_ids"] = ids
        state["noise_active_until"] = torch.full(
            (len(means),), -1, dtype=torch.long, device=means.device
        )
        state["history_score"] = torch.zeros(
            len(means), dtype=torch.float32, device=means.device
        )

        if self.verbose:
            print(
                f"Weighted anchors: stored_weight={float(weights.sum()):,.1f}, "
                f"anchors={len(anchors):,}, removed_empty={removed:,}, "
                f"low_source_voxels={int(low_mask.sum()):,} "
                f"({low_mask.float().mean():.1%}, weight<={low_threshold:g}), "
                f"global_target_voxels={len(anchors):,}, "
                f"min_source_gs={self.redundant_min_gs_per_anchor}, "
                f"overlap_ratio={self.redundant_overlap_ratio:g}."
            )

    @staticmethod
    def _low_mask(weights: Tensor, fraction: float) -> tuple[Tensor, float]:
        """Select exactly the requested fraction of lowest-weight anchors."""
        low_count = math.ceil(len(weights) * fraction)
        low_indices = weights.topk(low_count, largest=False).indices
        low_mask = torch.zeros_like(weights, dtype=torch.bool)
        low_mask[low_indices] = True
        return low_mask, float(weights[low_indices].max().item())

    @torch.no_grad()
    def _voxel_anchor_ids(self, means: Tensor, anchors: Tensor) -> Tensor:
        started = time.perf_counter()
        anchor_np = np.ascontiguousarray(
            anchors.detach().float().cpu().numpy(), dtype=np.float64
        )
        means_np = np.ascontiguousarray(
            means.detach().float().cpu().numpy(), dtype=np.float64
        )
        distances, ids = cKDTree(anchor_np, leafsize=32).query(
            means_np, k=1, workers=-1
        )

        max_distance = 5
        invalid = ~np.isfinite(distances) | (distances > max_distance + 1e-6)
        if np.any(invalid):
            raise ValueError(
                f"{int(invalid.sum())} initial GSs are too far from every "
                f"anchor; max distance={float(distances.max()):.6f}, allowed={max_distance:.6f}"
            )
        if self.verbose:
            print(
                f"Matched {len(means):,} initial GSs to {len(anchors):,} "
                f"anchors in {time.perf_counter() - started:.2f}s (KDTree)."
            )
        return torch.from_numpy(np.asarray(ids, dtype=np.int64)).to(means.device)

    def _require_anchor_state(self, params, state: Dict[str, Any]) -> None:
        ids = state.get("anchor_ids")
        if ids is None:
            raise RuntimeError("call set_surface_anchors() before training")
        if len(ids) != len(params["means"]):
            raise RuntimeError("anchor_ids must contain one ID per Gaussian")

    def _tracked_state(self, state: Dict[str, Any]) -> Dict[str, Tensor]:
        tracked = super()._tracked_state(state)
        for key in ("anchor_xyz", "anchor_low_mask", "anchor_ids"):
            tracked.pop(key, None)
        return tracked

    @torch.no_grad()
    def _inherit_surface_anchors(
        self,
        state: Dict[str, Any],
        indices: Tensor,
        source_indices: Tensor,
        append: bool = False,
    ) -> None:
        ids = state["anchor_ids"]
        if append:
            state["anchor_ids"] = torch.cat((ids, ids[source_indices]))
        else:
            ids[indices] = ids[source_indices]

    @torch.no_grad()
    def _constrain_surface_anchored_gs(
        self, params, state: Dict[str, Any], indices: Tensor
    ) -> None:
        if not self.surface_anchor_constraint or indices.numel() == 0:
            return
        centers = state["anchor_xyz"][state["anchor_ids"][indices]]
        offset = params["means"][indices] - centers
        distance = offset.norm(dim=-1, keepdim=True)
        period = 2.0 * self.surface_anchor_max_distance
        reflected_distance = distance.remainder(period)
        reflected_distance = torch.minimum(
            reflected_distance, period - reflected_distance
        )
        ratio = reflected_distance / distance.clamp_min(1e-12)
        params["means"][indices] = centers + offset * ratio

    @torch.no_grad()
    def _reset_state(
        self, params, optimizers, state: Dict[str, Any], indices: Tensor
    ) -> None:
        count = len(params["means"])
        for name, optimizer in optimizers.items():
            parameter_state = optimizer.state.get(params[name], {})
            for key, value in parameter_state.items():
                if (
                    key != "step"
                    and isinstance(value, Tensor)
                    and value.ndim > 0
                    and value.shape[0] == count
                ):
                    value[indices] = 0
        for value in self._tracked_state(state).values():
            if value.shape[0] == count:
                value[indices] = 0

    @torch.no_grad()
    def _clustered_redundant_ids(
        self,
        means: Tensor,
        opacity: Tensor,
        ids: Tensor,
        low_anchor_mask: Tensor,
    ) -> Tensor:
        """Pick one lower-opacity GS from each tightly clustered anchor."""
        counts = torch.bincount(ids, minlength=len(low_anchor_mask))
        source = low_anchor_mask[ids] & (
            counts[ids] > self.redundant_min_gs_per_anchor
        )
        indices = torch.where(source)[0]
        if indices.numel() == 0:
            return indices

        indices = indices[torch.argsort(ids[indices], stable=True)]
        _, group_counts = torch.unique_consecutive(
            ids[indices], return_counts=True
        )
        selected = []
        closest_distance2 = []

        for group in indices.split(group_counts.tolist()):
            points = means[group]
            distance2 = (points[:, None] - points[None]).square().sum(dim=-1)
            distance2.fill_diagonal_(float("inf"))
            closest = distance2.argmin()
            size = len(group)
            pair = group[torch.stack((closest // size, closest % size))]
            selected.append(pair[opacity[pair].argmin()])
            closest_distance2.append(distance2.flatten()[closest])

        threshold = self.redundant_overlap_ratio * self.anchor_voxel_size
        overlaps = torch.stack(closest_distance2) < threshold * threshold
        return torch.stack(selected)[overlaps]

    def _dead_mask(self, opacity: Tensor) -> Tensor:
        dead = super()._dead_mask(opacity)
        ids = self._active_state["anchor_ids"]
        anchor_count = len(self._active_state["anchor_xyz"])
        indices = torch.arange(len(ids), device=ids.device)
        keepers = torch.full(
            (anchor_count,), len(ids), dtype=torch.long, device=ids.device
        ).scatter_reduce_(
            0, ids, indices, reduce="amin", include_self=True
        )
        dead[keepers] = False
        return dead

    @torch.no_grad()
    def _relocate_gs(self, params, optimizers, binoms, scene=None) -> int:
        opacity = torch.sigmoid(params["opacities"].flatten())
        relocated = torch.where(self._dead_mask(opacity))[0]
        count = super()._relocate_gs(
            params, optimizers, binoms, scene=scene
        )
        if count:
            self._active_state["history_score"][relocated] = 0
            self._visible_noise_mask[relocated] = False
            # self._jitter_gs(params, relocated)
        return count

    @torch.no_grad()
    def _add_new_gs(self, params, optimizers, binoms, scene=None) -> int:
        current = len(params["means"])
        count = super()._add_new_gs(params, optimizers, binoms, scene=scene)
        if count:
            added = torch.arange(
                current, current + count, device=params["means"].device
            )
            self._mark_noise_active(self._active_state, added)
            # self._jitter_gs(params, added)
        return count

    @torch.no_grad()
    def move_redundant_gs(
        self, params, optimizers, state, step: int, scene=None
    ) -> int:
        """Move overlapping GSs from low-weight anchors to detail regions."""
        self._require_anchor_state(params, state)
        ids = state["anchor_ids"]
        opacity = torch.sigmoid(params["opacities"].flatten())

        indices = self._clustered_redundant_ids(
            params["means"],
            opacity,
            ids,
            state["anchor_low_mask"],
        )
        if indices.numel() == 0:
            return 0
        redundant = torch.zeros_like(opacity, dtype=torch.bool)
        redundant[indices] = True

        valid = (opacity > self.min_opacity) & ~redundant
        weights = torch.nan_to_num(
            state["history_score"], nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0)
        weights = weights.masked_fill(~valid, 0)

        if weights.sum() <= 0:
            return 0

        state["binoms"] = state["binoms"].to(params["means"].device)
        sampled_idxs = relocate(
            params=params,
            optimizers=optimizers,
            state={},
            mask=redundant,
            binoms=state["binoms"],
            min_opacity=self.min_opacity,
            scene=scene,
            sampling_weights=weights,
        )
        self._inherit_surface_anchors(state, indices, sampled_idxs)
        self._reset_state(params, optimizers, state, indices)
        # self._jitter_gs(params, indices)
        self._mark_noise_active(state, indices)
        if self.verbose:
            destination_anchor_count = ids[sampled_idxs].unique().numel()
            print(
                f"Step {step}: bottom-overlap->history relocated {len(indices):,} GSs "
                f"across {destination_anchor_count:,} target anchors."
            )
        return int(indices.numel())

    def step_post_backward(self, params, optimizers, state, step, info, lr, scene=None):
        self._require_anchor_state(params, state)
        if self.redundant_move:
            if 0 < step < self.refine_stop_iter and step % self.refine_every == 0:
                state["history_score"].lerp_(
                    self._donor_weights(params, state), 0.9
                )

            if (
                self.capacity_move_every > 0
                and len(params["means"]) >= self.cap_max
                and self.refine_start_iter < step < self.refine_stop_iter
                and step % self.capacity_move_every == 0
            ):
                moved = self.move_redundant_gs(
                    params, optimizers, state, step, scene=scene
                )
                if self.verbose:
                    print(
                        f"Step {step}: Moved {moved} redundant voxel GSs "
                        "using history_score."
                    )
        super().step_post_backward(params, optimizers, state, step, info, lr, scene)
        del self._visible_noise_mask
