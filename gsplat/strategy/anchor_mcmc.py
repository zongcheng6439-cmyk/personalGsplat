from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from torch import Tensor

from .anchor_io import read_anchor_capacity_ply
from .mcmc import MCMCStrategy
from .ops import relocate, sample_add
from .texture_mcmc import TextureMCMCStrategy


@dataclass
class AnchorMCMCStrategy(TextureMCMCStrategy):
    """Texture-guided MCMC with a deferred per-anchor Gaussian budget.

    The static anchor table contains one 3D center and one maximum count per
    anchor.  Each Gaussian stores only an ``anchor_id``.  During densification,
    dead relocation and score-based sampling ignore the per-anchor capacities;
    relocated and added Gaussians inherit their donor's anchor.  By default,
    one final transaction moves the lowest-opacity excess Gaussians into
    anchors that still have free capacity when MCMC relocation/addition stops.  Set
    ``capacity_rebalance_every`` to also repair voxel capacities periodically.

    Call :meth:`set_anchors` after :meth:`initialize_state` and before the
    first training step.  The default assignment uses the world-aligned voxel
    keys from the stage-one budget.  Explicit ``anchor_ids`` are also accepted;
    nearest-anchor assignment is a bounded-memory fallback for small inputs.
    """

    # Two score modes are supported.  The normal mode is the requested
    # opacity + grad + weight-map score.  FastGS mode adds a normalized
    # multi-view high-loss count without replacing any of those signals.
    texture_weight: float = 1.0
    grad_weight: float = 1.0    
    loss_count_weight: float = 1.0
    fastgs_min_loss_count: int = 5    

    # Add at most this fraction of the current population on a refine step.
    add_growth_rate: float = 0.05
    # Optional in-training capacity repair.  Zero keeps capacity balancing
    # deferred until refine_stop_iter; a positive value repairs overfull
    # voxels every N steps and still performs one final repair at the stop.
    capacity_rebalance_every: int = 0

    # Optional output of build_voxel_anchor_budget.py.  Existing texture
    # trainers call set_surface_anchors(), which loads and binds this file.
    anchor_ply: str | None = None
    use_anchor_capacity_as_cap_max: bool = True

    # Reuse the surface-anchor options as the on/off switch and radius for the
    # fixed anchor centers.  A value of 0.1 means 10 cm in a metric scene.
    surface_anchor_constraint: bool = True
    surface_anchor_max_distance: float = 0.1

    # Used by existing texture trainer code when selecting top contributors.
    top_k: int = 3

    # Bounded-memory nearest-anchor fallback, used only when
    assignment_chunk_size: int = 2048
    anchor_chunk_size: int = 4096

    def initialize_state(self) -> Dict[str, Any]:
        state = super().initialize_state()
        state.update(
            anchor_xyz=None,
            anchor_capacity=None,
            anchor_ids=None,
            final_capacity_rebalance_done=False,
        )
        return state

    def check_sanity(self, params, optimizers):
        super().check_sanity(params, optimizers)
        if self.fastgs_min_loss_count < 1:
            raise ValueError("fastgs_min_loss_count must be positive")
        if self.add_growth_rate < 0:
            raise ValueError("add_growth_rate must be non-negative")
        if self.capacity_rebalance_every < 0:
            raise ValueError("capacity_rebalance_every must be non-negative")
        if not 0 < self.min_opacity < 1:
            raise ValueError("min_opacity must be in (0, 1)")
        if self.grad_weight > 0 and not self.absgrad:
            raise ValueError("grad_weight > 0 requires absgrad=True")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.assignment_chunk_size <= 0 or self.anchor_chunk_size <= 0:
            raise ValueError("anchor assignment chunk sizes must be positive")


    @torch.no_grad()
    def set_surface_anchors(self, state: Dict[str, Any], means: Tensor) -> None:
        """Load ``anchor_ply`` for compatibility with existing trainers.

        Existing trainers call this method for every TextureMCMC subclass.  The
        anchor-aware strategy never creates TextureMCMC's legacy per-GS anchor.
        When no PLY is configured, callers may still bind tensors explicitly
        with :meth:`set_anchors` before the first training step.
        """
        if not self.anchor_ply:
            raise ValueError(
                "AnchorMCMCStrategy requires anchor_ply in the example trainers; "
                "for custom loops call set_anchors(...) directly"
            )
        xyz, capacity = read_anchor_capacity_ply(self.anchor_ply)
        self.set_anchors(
            state,
            means,
            torch.from_numpy(xyz),
            torch.from_numpy(capacity),
        )

    @torch.no_grad()
    def set_anchors(
        self,
        state: Dict[str, Any],
        means: Tensor,
        anchor_xyz: Tensor,
        anchor_capacity: Tensor,        
    ) -> None:
        """Bind fixed anchor centers, per-anchor maxima, and GS assignments."""
        anchors = torch.as_tensor(
            anchor_xyz, device=means.device, dtype=means.dtype
        ).detach()
        capacity = torch.as_tensor(anchor_capacity, device=means.device).to(torch.long).detach()    

        if torch.any(capacity < 0):
            raise ValueError("anchor_capacity cannot be negative")

        ids = self._nearest_anchor_ids(means.detach(), anchors)

        if torch.any(ids < 0) or torch.any(ids >= len(anchors)):
            raise ValueError("anchor_ids contains an out-of-range anchor index")

        state["anchor_xyz"] = anchors.clone()
        state["anchor_capacity"] = capacity.clone()
        state["anchor_ids"] = ids.clone()
        state["noise_active_until"] = torch.full(
            (len(means),), -1, dtype=torch.long, device=means.device
        )
        state["final_capacity_rebalance_done"] = False
        state.pop("surface_anchor", None)
        state.pop("surface_anchor_active", None)

    @torch.no_grad()
    def _nearest_anchor_ids(self, means: Tensor, anchors: Tensor) -> Tensor:
        result = torch.empty(len(means), dtype=torch.long, device=means.device)
        for point_start in range(0, len(means), self.assignment_chunk_size):
            points = means[point_start : point_start + self.assignment_chunk_size]
            best_distance = torch.full(
                (len(points),), float("inf"), device=means.device
            )
            best_id = torch.zeros(len(points), dtype=torch.long, device=means.device)
            for anchor_start in range(0, len(anchors), self.anchor_chunk_size):
                anchor_block = anchors[
                    anchor_start : anchor_start + self.anchor_chunk_size
                ]
                distances = torch.cdist(points.float(), anchor_block.float())
                block_distance, block_id = distances.min(dim=1)
                better = block_distance < best_distance
                best_distance[better] = block_distance[better]
                best_id[better] = block_id[better] + anchor_start
            result[point_start : point_start + len(points)] = best_id
        return result

    def _require_anchor_state(self, params, state: Dict[str, Any]) -> None:
        anchors = state.get("anchor_xyz")
        capacity = state.get("anchor_capacity")
        ids = state.get("anchor_ids")
        if anchors is None or capacity is None or ids is None:
            raise RuntimeError(
                "AnchorMCMCStrategy is not bound; call set_anchors(state, means, "
                "anchor_xyz, anchor_capacity, anchor_ids) before training"
            )
        if anchors.shape != (len(capacity), 3):
            raise RuntimeError("anchor_xyz and anchor_capacity shapes are inconsistent")
        if ids.shape != (len(params["means"]),):
            raise RuntimeError("anchor_ids must have one entry per Gaussian")
        if (
            anchors.device != params["means"].device
            or capacity.device != params["means"].device
            or ids.device != params["means"].device
        ):
            raise RuntimeError("anchor state and Gaussian parameters must share a device")

    @staticmethod
    def _tracked_state(state: Dict[str, Any]) -> Dict[str, Tensor]:
        static_or_managed = {
            "binoms",
            "surface_anchor",
            "surface_anchor_active",
            "anchor_xyz",
            "anchor_capacity",
            "anchor_ids",
            "noise_active_until",
        }
        return {
            key: value
            for key, value in state.items()
            if key not in static_or_managed
            and isinstance(value, Tensor)
            and value.ndim > 0
        }

    def _base_donor_weights(self, params, state: Dict[str, Any]) -> Tensor:
        """Return the normal ``opacity + grad + weight-map`` score."""
        opacity = torch.sigmoid(params["opacities"].flatten())
        texture = self._texture_signal(state, len(opacity), opacity)
        grad = self._grad_signal(state, len(opacity), opacity)
        importance = (
            self.texture_floor
            + self.texture_weight * texture
            + self.grad_weight * grad
        )
        return importance * opacity
        

    def _fastgs_donor_weights(self, params, state: Dict[str, Any]) -> Tensor:
        """Add the normalized FastGS loss count to the normal score."""
        score = self._base_donor_weights(params, state)
        loss_count = state.get("loss_count")
        if loss_count is None:
            return score
        raw_loss_count = torch.where(
            loss_count >= self.fastgs_min_loss_count,
            loss_count,
            torch.zeros_like(loss_count),
        )
        loss_signal = self._quantile_normalize(
            raw_loss_count, self.grad_quantile
        )
        return score + self.loss_count_weight * loss_signal

    def _donor_weights(self, params, state: Dict[str, Any]) -> Tensor:
        """Select the normal or FastGS-augmented score calculation."""
        score = (
            self._fastgs_donor_weights(params, state)
            if self.use_fastgs_score
            else self._base_donor_weights(params, state)
        )
        return torch.nan_to_num(
            score, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(1e-8)

    @torch.no_grad()
    def _update_statistics(self, params, state: Dict[str, Any], info: Dict[str, Any]):
        """Collect loss counts and absgrad together in FastGS mode."""
        super()._update_statistics(params, state, info)

    @staticmethod
    def anchor_counts(state: Dict[str, Any]) -> Tensor:
        capacity = state["anchor_capacity"]
        return torch.bincount(state["anchor_ids"], minlength=len(capacity))

    @torch.no_grad()
    def _allocate_targets(self, free_slots: Tensor, n: int) -> Tensor:
        """Allocate exactly ``n`` targets without exceeding any free-slot count."""
        if n < 0 or n > int(free_slots.sum().item()):
            raise ValueError("requested targets do not fit in the anchor free slots")
        if n == 0:
            return torch.empty(0, dtype=torch.long, device=free_slots.device)

        allocation = torch.zeros_like(free_slots, dtype=torch.long)
        remaining_n = n
        while remaining_n > 0:
            remaining_slots = free_slots - allocation
            weights = remaining_slots.float()
            shares = remaining_n * weights / weights.sum().clamp_min(1e-12)
            base = torch.minimum(shares.floor().long(), remaining_slots)
            taken = int(base.sum().item())
            if taken:
                allocation.add_(base)
                remaining_n -= taken
                if remaining_n == 0:
                    break

            remaining_slots = free_slots - allocation
            candidates = torch.where(remaining_slots > 0)[0]
            if candidates.numel() == 0:
                raise RuntimeError("anchor target allocation exhausted its free slots")
            residual = shares - shares.floor()
            order = torch.argsort(residual[candidates], descending=True)
            take = min(remaining_n, len(candidates))
            chosen = candidates[order[:take]]
            allocation[chosen] += 1
            remaining_n -= take

        anchor_ids = torch.arange(len(free_slots), device=free_slots.device)
        return torch.repeat_interleave(anchor_ids, allocation)

    @torch.no_grad()
    def _constrain_surface_anchored_gs(
        self, params, state: Dict[str, Any], indices: Tensor
    ) -> None:
        if not self.surface_anchor_constraint or indices.numel() == 0:
            return
        self._require_anchor_state(params, state)
        centers = state["anchor_xyz"][state["anchor_ids"][indices]]
        means = params["means"]
        offset = torch.nan_to_num(
            means[indices] - centers, nan=0.0, posinf=0.0, neginf=0.0
        )
        distance = offset.norm(dim=-1, keepdim=True)
        ratio = (
            self.surface_anchor_max_distance / distance.clamp_min(1e-12)
        ).clamp(max=1.0)
        means[indices] = centers + offset * ratio

    @staticmethod
    @torch.no_grad()
    def _reset_gaussian_state(
        params, optimizers, state: Dict[str, Any], index: Tensor
    ) -> None:
        """Clear optimizer moments and accumulated scores for one GS slot."""
        count = len(params["means"])
        for name, optimizer in optimizers.items():
            parameter = params[name]
            parameter_state = optimizer.state.get(parameter, {})
            for key, value in parameter_state.items():
                if (
                    key != "step"
                    and isinstance(value, Tensor)
                    and value.ndim > 0
                    and value.shape[0] == count
                ):
                    value[index] = 0
        for value in AnchorMCMCStrategy._tracked_state(state).values():
            if value.shape[0] == count:
                value[index] = 0

    @torch.no_grad()
    def _rebalance_anchor_capacity(
        self,
        params,
        optimizers,
        state: Dict[str, Any],
        *,
        mark_final: bool,
    ) -> int:
        """Move lowest-opacity excess GSs into anchors with unused capacity."""
        self._require_anchor_state(params, state)
        if mark_final and state["final_capacity_rebalance_done"]:
            return 0

        ids = state["anchor_ids"]
        capacity = state["anchor_capacity"]
        opacity = torch.sigmoid(params["opacities"].flatten())

        counts = self.anchor_counts(state)
        excess = (counts - capacity).clamp_min(0)

        redundant = torch.zeros_like(ids, dtype=torch.bool)
        point_ids = torch.arange(len(ids), device=ids.device)
        if point_ids.numel() and torch.any(excess > 0):
            # Stable opacity sort followed by stable anchor sort gives each
            # anchor's points in ascending-opacity order without scanning N
            # once for every overfull anchor.
            opacity_order = torch.argsort(
                opacity[point_ids], descending=False, stable=True
            )
            grouped_points = point_ids[opacity_order]
            anchor_order = torch.argsort(ids[grouped_points], stable=True)
            grouped_points = grouped_points[anchor_order]
            grouped_anchor_ids = ids[grouped_points]
            grouped_counts = torch.bincount(
                grouped_anchor_ids, minlength=len(capacity)
            )
            grouped_offsets = grouped_counts.cumsum(0) - grouped_counts
            rank_in_anchor = torch.arange(
                len(grouped_points), device=grouped_points.device
            ) - torch.repeat_interleave(grouped_offsets, grouped_counts)
            selected = rank_in_anchor < excess[grouped_anchor_ids]
            redundant[grouped_points[selected]] = True

        source_ids = torch.where(redundant)[0]
        n_move = len(source_ids)
        self._last_dead_relocated = 0
        self._last_capacity_relocated = n_move
        if n_move == 0:
            if mark_final:
                state["final_capacity_rebalance_done"] = True
            return 0

        free_slots = (capacity - counts).clamp_min(0)
        if int(free_slots.sum().item()) < n_move:
            raise RuntimeError(
                "total anchor capacity cannot absorb all excess Gaussians"
            )
        target_ids = self._allocate_targets(free_slots, n_move)

        # Capacity repair is a spatial move, not another MCMC split.  Keep each
        # selected GS's appearance/scale/opacity, place it at the new anchor
        # center, and clear optimizer/statistics tied to the old voxel.
        params["means"][source_ids] = state["anchor_xyz"][target_ids]
        ids[source_ids] = target_ids
        self._reset_gaussian_state(params, optimizers, state, source_ids)
        self._mark_noise_active(state, source_ids)

        if torch.any(self.anchor_counts(state) > capacity):
            raise RuntimeError("anchor-capacity move did not converge")
        if mark_final:
            state["final_capacity_rebalance_done"] = True
        return n_move

    @torch.no_grad()
    def rebalance_anchor_capacity(
        self, params, optimizers, state: Dict[str, Any]
    ) -> int:
        """Repair current voxel over-capacity without ending training."""
        return self._rebalance_anchor_capacity(
            params, optimizers, state, mark_final=False
        )

    @torch.no_grad()
    def finalize_anchor_capacity(
        self, params, optimizers, state: Dict[str, Any]
    ) -> int:
        """Perform the final capacity repair after relocate/add has stopped."""
        return self._rebalance_anchor_capacity(
            params, optimizers, state, mark_final=True
        )

    @torch.no_grad()
    def _relocate_gs(self, params, optimizers, binoms, scene=None) -> int:
        """Relocate dead GSs globally by score and inherit the donor anchor."""

        state = self._active_state
        self._require_anchor_state(params, state)
        scores = self._donor_weights(params, state)
        opacity = torch.sigmoid(params["opacities"].flatten())
        if opacity.numel() == 0:
            self._last_dead_relocated = 0
            self._last_capacity_relocated = 0
            return 0
        # The relocation kernel can output exactly min_opacity.  A tiny margin
        # prevents numerical round-off from immediately recycling that GS.
        raw_dead = opacity < self.min_opacity * (1.0 - 1e-6)
        if not torch.any(raw_dead):
            self._last_dead_relocated = 0
            self._last_capacity_relocated = 0
            return 0

        if bool(torch.all(raw_dead)):
            raise RuntimeError(
                "all Gaussians are dead; scene optimization has failed"
            )

        # Process every dead GS in one call.  The shared MCMC op samples live
        # donors from this score and returns the actual donor indices, which are
        # then used to inherit anchor IDs.
        source_ids = torch.where(raw_dead)[0]
        n_relocate = len(source_ids)
        self._last_dead_relocated = n_relocate
        self._last_capacity_relocated = 0

        tracked = self._tracked_state(state)
        sampled = relocate(
            params=params,
            optimizers=optimizers,
            state=tracked,
            mask=raw_dead,
            binoms=binoms,
            min_opacity=self.min_opacity,
            scene=scene,
            sampling_weights=scores,
        )
        state.update(tracked)
        state["anchor_ids"][source_ids] = state["anchor_ids"][sampled]
        self._mark_noise_active(state, source_ids)
        self._reset_gaussian_state(
            params,
            optimizers,
            state,
            torch.unique(torch.cat((source_ids, sampled))),
        )

        state["final_capacity_rebalance_done"] = False
        return n_relocate

    @torch.no_grad()
    def _add_new_gs(self, params, optimizers, binoms, scene=None) -> int:
        state = self._active_state
        self._require_anchor_state(params, state)
        current = len(params["means"])
        if current == 0 or self.add_growth_rate == 0:
            self._last_added = 0
            return 0

        growth = max(1, math.ceil(current * self.add_growth_rate))
        n_gs = min(
            growth,
            max(0, self.cap_max - current),
        )
        if n_gs == 0:
            self._last_added = 0
            return 0

        scores = self._donor_weights(params, state)
        old_ids = state["anchor_ids"]

        tracked = self._tracked_state(state)
        sampled = sample_add(
            params=params,
            optimizers=optimizers,
            state=tracked,
            n=n_gs,
            binoms=binoms,
            min_opacity=self.min_opacity,
            scene=scene,
            sampling_weights=scores,
        )
        state.update(tracked)
        state["anchor_ids"] = torch.cat((old_ids, old_ids[sampled]))
        added = torch.arange(
            current, current + n_gs, device=state["anchor_ids"].device
        )
        active_until = state["noise_active_until"]
        step = getattr(self, "_active_step", 0)
        new_until = torch.full(
            (n_gs,),
            step + max(1, self.active_noise_steps) - 1,
            dtype=torch.long,
            device=active_until.device,
        )
        state["noise_active_until"] = torch.cat((active_until, new_until))
        self._reset_gaussian_state(
            params,
            optimizers,
            state,
            torch.unique(torch.cat((sampled, added))),
        )
        state["final_capacity_rebalance_done"] = False
        self._last_added = n_gs
        return n_gs

    def step_post_backward(self, params, optimizers, state, step, info, lr, scene=None):
        self._require_anchor_state(params, state)
        if state["noise_active_until"] is None:
            state["noise_active_until"] = torch.full(
                (len(params["means"]),),
                -1,
                dtype=torch.long,
                device=params["means"].device,
            )
        if not state["final_capacity_rebalance_done"]:
            self._update_statistics(params, state, info)
        self._active_state = state
        self._active_step = step
        self._last_dead_relocated = 0
        self._last_capacity_relocated = 0
        self._last_added = 0
        capacity_finalized = False
        configured_noise_stop = self.noise_injection_stop_iter
        try:
            # Disable the parent's global noise. Only recently relocated,
            # added, or capacity-moved GSs are perturbed below.
            self.noise_injection_stop_iter = 0
            MCMCStrategy.step_post_backward(
                self, params, optimizers, state, step, info, lr, scene
            )

            # Periodic MCMC relocation/addition uses only the global score and
            # global cap.  Capacity repair is optional during this phase.
            if (
                self.capacity_rebalance_every > 0
                and self.refine_start_iter < step < self.refine_stop_iter
                and step % self.capacity_rebalance_every == 0
            ):
                n_moved = self.rebalance_anchor_capacity(
                    params, optimizers, state
                )
                if self.verbose:
                    print(
                        f"Step {step}: Moved {n_moved} excess GSs into "
                        "under-capacity anchors."
                    )

            # Once relocate/add stops, settle the per-anchor budget exactly
            # once and leave the remaining steps for local refinement.
            if (
                step >= self.refine_stop_iter
                and not state["final_capacity_rebalance_done"]
            ):
                n_moved = self.finalize_anchor_capacity(params, optimizers, state)
                capacity_finalized = True
                if self.verbose:
                    print(
                        f"Step {step}: Moved {n_moved} excess GSs to finalize "
                        "anchor capacities."
                    )
            active = torch.where(state["noise_active_until"] >= step)[0]
            if configured_noise_stop < 0 or step < configured_noise_stop:
                self._inject_active_noise(params, active, lr)
            self._constrain_surface_anchored_gs(params, state, active)
        finally:
            self.noise_injection_stop_iter = configured_noise_stop
            del self._active_state
            del self._active_step

        if capacity_finalized or (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            for key in (
                "texture_info",
                "texture_pixel_num",
                "absgrad_sum",
                "grad_count",
                "loss_count",
            ):
                if state[key] is not None:
                    state[key].zero_()
