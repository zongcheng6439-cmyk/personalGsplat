from __future__ import annotations

from dataclasses import dataclass, field
import math
import os 
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from torch import Tensor

from .mcmc import MCMCStrategy
from .ops import inject_noise_to_position, relocate, sample_add


@dataclass
class TextureMCMCStrategy(MCMCStrategy):
    """MCMC densification guided jointly by opacity and a texture weight map.

    Texture and optional absolute screen-space gradients determine donor
    importance. Opacity is only a soft reliability gate, so low-opacity GSs in
    important regions are downweighted rather than completely excluded.
    """

    # Kept only for compatibility with MCMCStrategy; this implementation
    # relocates every GS below min_opacity and does not apply a fraction cap.    
    use_opacity_score: bool = False
    use_texture_score: bool = False
    use_grad_score: bool = False
    use_fastgs_score: bool = False

    opacity_weight: float = 0.25
    texture_weight: float = 0.5
    grad_weight: float = 0.25
    fastgs_weight: float = 0.25

    texture_quantile: float = 0.9
    grad_quantile: float = 0.9
    absgrad: bool = True        
    texture_update_every: int = 1

    # Save the highest-scoring 30% GS candidates as white binary PLY files.
    score_debug_dir: str | None = None
    add_hot_fraction: float = 0.3
    add_hot_probability: float = 0.7
    active_noise_steps: int = 1000
    # Independent of the means optimizer learning rate. It decays
    # exponentially to 10% over the configured training duration.
    noise_scale: float = 10
    noise_opacity_t: float = 0.054
    noise_opacity_k: float = 100.0
    _noise_decay_steps: int | None = field(default=None, init=False, repr=False)
    # Optional surface-anchor constraint for newly added/relocated GSs only.
    surface_anchor_constraint: bool = False
    surface_anchor_max_distance: float = 0.1

    def initialize_state(self) -> Dict[str, Any]:
        state = super().initialize_state()
        state.update(
            texture_info=None,
            texture_pixel_num=None,
            absgrad_sum=None,
            grad_count=None,
            loss_count=None,
            noise_active_until=None,
        )
        return state

    def check_sanity(self, params, optimizers):
        super().check_sanity(params, optimizers)
        if min(self.opacity_weight, self.texture_weight, self.grad_weight) < 0:
            raise ValueError("score weights must be non-negative")
        if not 0 < self.texture_quantile <= 1 or not 0 < self.grad_quantile <= 1:
            raise ValueError("texture_quantile and grad_quantile must be in (0, 1]")
        if not any(
            (
                self.use_opacity_score,
                self.use_texture_score,
                self.use_grad_score,
                self.use_fastgs_score,
            )
        ):
            raise ValueError("at least one donor score must be enabled")
        if not 0 < self.add_hot_fraction < 1:
            raise ValueError("add_hot_fraction must be in (0, 1)")
        if not 0 < self.add_hot_probability < 1:
            raise ValueError("add_hot_probability must be in (0, 1)")
        if self.active_noise_steps < -1:
            raise ValueError("active_noise_steps must be -1 or non-negative")
        if self.noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")
        if self.surface_anchor_constraint and self.surface_anchor_max_distance <= 0:
            raise ValueError("surface_anchor_max_distance must be positive")

    def set_training_steps(self, max_steps: int) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._noise_decay_steps = int(max_steps)

    def noise_scale_at_step(self, step: int) -> float:
        if self._noise_decay_steps is None:
            return self.noise_scale
        progress = min(max(float(step) / self._noise_decay_steps, 0.0), 1.0)
        return self.noise_scale * 0.1**progress


    # Surface-anchor constraint block. Remove this method and the marked calls
    # below to completely remove the feature.
    @torch.no_grad()
    def set_surface_anchors(self, state: Dict[str, Any], means: Tensor) -> None:
        state["noise_active_until"] = torch.full(
            (len(means),), -1, dtype=torch.long, device=means.device
        )
        if self.surface_anchor_constraint:
            state["surface_anchor"] = means.detach().clone()

    @torch.no_grad()
    def _inherit_surface_anchors(
        self,
        state: Dict[str, Any],
        indices: Tensor,
        source_indices: Tensor,
        append: bool = False,
    ) -> None:
        anchors = state.get("surface_anchor")
        if not self.surface_anchor_constraint or anchors is None or indices.numel() == 0:
            return
        inherited = anchors[source_indices].clone()
        if append:
            state["surface_anchor"] = torch.cat((anchors, inherited))
        else:
            anchors[indices] = inherited

    @torch.no_grad()
    def _constrain_surface_anchored_gs(
        self, params, state: Dict[str, Any], indices: Tensor
    ) -> None:
        anchors = state.get("surface_anchor")
        if (
            not self.surface_anchor_constraint
            or anchors is None
            or indices.numel() == 0
        ):
            return
        means = params["means"]
        if anchors.shape != means.shape:
            raise RuntimeError(
                "surface_anchor and means must have identical shapes, got "
                f"{tuple(anchors.shape)} and {tuple(means.shape)}"
            )
        offset = means[indices] - anchors[indices]
        distance = offset.norm(dim=-1, keepdim=True)
        period = 2.0 * self.surface_anchor_max_distance
        reflected_distance = distance.remainder(period)
        reflected_distance = torch.minimum(
            reflected_distance, period - reflected_distance
        )
        ratio = reflected_distance / distance.clamp_min(1e-12)
        means[indices] = anchors[indices] + offset * ratio

    def _mark_noise_active(self, state: Dict[str, Any], indices: Tensor) -> None:
        if indices.numel() > 0:
            step = getattr(self, "_active_step", 0)
            state["noise_active_until"][indices] = self._active_until(step)

    def _active_until(self, step: int) -> int:
        return (
            torch.iinfo(torch.long).max
            if self.active_noise_steps < 0
            else step + max(1, self.active_noise_steps) - 1
        )

    @torch.no_grad()
    def _inject_active_noise(self, params, indices: Tensor, lr: float) -> Tensor:
        """Move active, view-supervised GSs and return the IDs actually changed."""
        del lr
        if indices.numel() == 0 or self.active_noise_steps == 0 or self.noise_scale == 0:
            return indices[:0]
        visible = self._visible_noise_mask
        indices = indices[indices < visible.numel()]
        indices = indices[visible[indices]]
        if indices.numel() == 0:
            return indices
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
        changed = (active_params["means"] != params["means"][indices]).any(-1)
        params["means"][indices] = active_params["means"]
        return indices[changed]

    def _apply_active_noise(self, params, state, step: int, lr: float, noise_stop: int):
        """Default constraint schedule; plane strategies specialize this hook."""
        active = torch.where(state["noise_active_until"] >= step)[0]
        if noise_stop < 0 or step < noise_stop:
            self._inject_active_noise(params, active, lr)
        self._constrain_surface_anchored_gs(params, state, active)

    @staticmethod
    def _quantile_normalize(values: Tensor, quantile: float) -> Tensor:
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
        positive = values[values > 0]
        if positive.numel() == 0:
            return torch.zeros_like(values)
        scale = torch.quantile(positive.float(), quantile).to(values)
        return (values / scale.clamp_min(1e-12)).clamp(0.0, 2.0)

    def _texture_signal(self, state: Dict[str, Any], n: int, ref: Tensor) -> Tensor:
        if state["texture_info"] is None:
            return torch.zeros(n, device=ref.device, dtype=ref.dtype)
        # count = state["texture_pixel_num"].clamp_min(1.0)
        # values = state["texture_info"] / count * torch.log1p(count)
        # i need more gs in detail area, so use average texture_info 
        values = state["texture_info"] / state["texture_pixel_num"].clamp_min(1e-4)
        return self._quantile_normalize(values, self.texture_quantile)

    def _grad_signal(self, state: Dict[str, Any], n: int, ref: Tensor) -> Tensor:
        if state["absgrad_sum"] is None:
            return torch.zeros(n, device=ref.device, dtype=ref.dtype)
        values = state["absgrad_sum"] / state["grad_count"].clamp_min(1.0)
        return self._quantile_normalize(values, self.grad_quantile)

    def _donor_weights(self, params, state: Dict[str, Any]) -> Tensor:
        opacity = torch.sigmoid(params["opacities"].flatten())
        importance = torch.zeros_like(opacity)
        if self.use_opacity_score:
            importance.add_(opacity, alpha=self.opacity_weight)

        if self.use_texture_score:
            importance.add_(
                self._texture_signal(state, len(opacity), opacity),
                alpha=self.texture_weight,
            )
        if self.use_grad_score:
            importance.add_(
                self._grad_signal(state, len(opacity), opacity),
                alpha=self.grad_weight,
            )
        if self.use_fastgs_score and state["loss_count"] is not None:
            raw_loss_count = torch.where(state["loss_count"] >= 5,
                                         state["loss_count"],
                                         torch.zeros_like(state["loss_count"]))
            importance.add_(
                self._quantile_normalize(raw_loss_count, self.grad_quantile),
                alpha=self.fastgs_weight,
            )
        if importance.sum() <= 0:
            importance.fill_(1.0)
        return importance

    def _focused_weights(
        self, scores: Tensor, candidate_mask: Tensor | None = None
    ):
        """Focus sampling on the top-scoring candidates."""
        scores = torch.nan_to_num(
            scores, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0.0)
        n = len(scores)
        if candidate_mask is None:
            candidate_mask = torch.ones(n, dtype=torch.bool, device=scores.device)
        candidates = torch.where(candidate_mask)[0]

        hot_count = max(1, math.ceil(len(candidates) * self.add_hot_fraction))
        hot_ids = candidates[torch.topk(scores[candidates], hot_count).indices]

        hot_mask = torch.zeros(n, dtype=torch.bool, device=scores.device)
        hot_mask[hot_ids] = True

        probabilities = torch.zeros_like(scores)
        candidate_scores = scores[candidates]

        total = candidate_scores.sum()
        if total > 0:
            probabilities[candidates] = candidate_scores / total
        else:
            probabilities[candidates] = 1.0 / len(candidates)

        raw_hot_probability = probabilities[hot_mask].sum()
        cold_mask = candidate_mask & ~hot_mask

        if raw_hot_probability < self.add_hot_probability and cold_mask.any():
            # Preserve relative probabilities within each group and only
            # increase the top group's total probability when necessary.
            probabilities = probabilities.clone()
            probabilities[hot_mask] *= (
                self.add_hot_probability / raw_hot_probability.clamp_min(1e-12)
            )
            cold_probability = probabilities[cold_mask].sum()
            probabilities[cold_mask] *= (
                (1.0 - self.add_hot_probability)
                / cold_probability.clamp_min(1e-12)
            )
        return probabilities, hot_mask, raw_hot_probability

    @torch.no_grad()
    def _update_statistics(self, params, state: Dict[str, Any], info: Dict[str, Any]):
        n = len(params["means"])
        self._visible_noise_mask = torch.zeros(
            n, dtype=torch.bool, device=params["means"].device
        )
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

        if self.use_fastgs_score and "loss_count" in info:
            loss_count = info["loss_count"].detach()
            if loss_count.shape != (n,):
                raise ValueError("FastGS loss statistics must have shape [num_gaussians]")
            if state["loss_count"] is None:
                state["loss_count"] = torch.zeros_like(loss_count)
            state["loss_count"].add_(loss_count)

        means2d = info.get("means2d")
        grads = getattr(means2d, "absgrad", None) if means2d is not None else None
        if grads is None:
            return
        if grads.ndim == 3:
            visible = (info["radii"] > 0).all(dim=-1)
            ids = torch.where(visible)[1]
            grads = grads[visible]
        else:
            ids = info["gaussian_ids"]
        grad_norm = grads.norm(dim=-1)
        info["current_grad_ids"] = ids[grad_norm > 0]
        self._visible_noise_mask[info["current_grad_ids"]] = True
        if not self.use_grad_score:
            return
        if state["absgrad_sum"] is None:
            state["absgrad_sum"] = torch.zeros(n, device=grads.device)
            state["grad_count"] = torch.zeros(n, device=grads.device)
        state["absgrad_sum"].index_add_(0, ids, grad_norm)
        state["grad_count"].index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))

    @staticmethod
    def _tracked_state(state: Dict[str, Any]) -> Dict[str, Tensor]:
        return {
            key: value
            for key, value in state.items()
            if key not in {"binoms", "surface_anchor", "noise_active_until"}
            and isinstance(value, Tensor)
            and value.ndim > 0
        }

    @torch.no_grad()
    def _dead_mask(self, opacity: Tensor) -> Tensor:
        return opacity <= self.min_opacity

    @torch.no_grad()
    def _relocate_gs(self, params, optimizers, binoms, scene=None) -> int:
        opacity = torch.sigmoid(params["opacities"].flatten())
        dead_mask = self._dead_mask(opacity)
        n_gs = int(dead_mask.sum().item())

        if n_gs > 0:
            self._weight = self._donor_weights(
                params, self._active_state
            )     
            self._save_top_score_gs(params, self._weight, "relocate")   
            sampled_idxs = relocate(
                params=params,
                optimizers=optimizers,
                state={},
                mask=dead_mask,
                binoms=binoms,
                min_opacity=self.min_opacity,
                scene=scene,
                sampling_weights=self._weight,
            )                    
            # Mark only the GSs relocated in this call as surface-constrained.
            self._inherit_surface_anchors(
                self._active_state, torch.where(dead_mask)[0], sampled_idxs
            )
            self._mark_noise_active(
                self._active_state, torch.where(dead_mask)[0]
            )
            return n_gs        
        else:
            return 0

    @torch.no_grad()
    def _add_new_gs(self, params, optimizers, binoms, scene=None) -> int:
        current = len(params["means"])
        n_gs = max(0, min(self.cap_max, int(1.05 * current)) - current)
        if n_gs > 0:
            if self._weight is None:
                self._weight = self._donor_weights(params, self._active_state)

            if self._active_step % (self.refine_every * 10) == 0:
                self._save_top_score_gs(params, self._weight, "add")            

            add_weights, hot_mask, raw_hot_probability = self._focused_weights(self._weight)
            tracked = self._tracked_state(self._active_state)

            sampled_idxs = sample_add(
                params=params,
                optimizers=optimizers,
                state=tracked,
                n=n_gs,
                binoms=binoms,
                min_opacity=self.min_opacity,
                scene=scene,
                sampling_weights=add_weights,
            )

            if self.verbose:
                hot_sampled = int(hot_mask[sampled_idxs].sum().item())
                print(
                    f"Add top {self.add_hot_fraction:.0%}: "
                    f"raw probability={raw_hot_probability.item():.1%}, "
                    f"used probability={add_weights[hot_mask].sum().item():.1%}, "
                    f"sampled={hot_sampled / n_gs:.1%} ({hot_sampled}/{n_gs})."
                )

            self._active_state.update(tracked)
            # New GSs inherit the donor's original surface anchor.
            added = torch.arange(current, current + n_gs, device=params["means"].device)
            self._inherit_surface_anchors(
                self._active_state, added, sampled_idxs, append=True
            )
            active_until = self._active_state["noise_active_until"]
            step = getattr(self, "_active_step", 0)
            new_until = torch.full(
                (n_gs,),
                self._active_until(step),
                dtype=torch.long,
                device=active_until.device,
            )
            self._active_state["noise_active_until"] = torch.cat(
                (active_until, new_until)
            )
        return n_gs

    def step_post_backward(self, params, optimizers, state, step, info, lr, scene=None):
        self._update_statistics(params, state, info)
        self._active_state = state
        self._active_step = step
        self._weight = None
        configured_noise_stop = self.noise_injection_stop_iter
        try:
            # The parent noise affects every GS. Disable it and perturb only
            # recently relocated or added GSs below.
            self.noise_injection_stop_iter = 0
            super().step_post_backward(params, optimizers, state, step, info, lr, scene)

            self._apply_active_noise(params, state, step, lr, configured_noise_stop)

        finally:
            self.noise_injection_stop_iter = configured_noise_stop
            del self._active_state
            del self._active_step
            del self._weight

        if (
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

    @torch.no_grad()
    def _save_top_score_gs(
        self,
        params,
        scores: Tensor,
        name: str,
        candidate_mask: Tensor | None = None,
    ) -> None:
        if self.score_debug_dir is None:
            return
        candidates = torch.arange(len(scores), device=scores.device)
        if candidate_mask is not None:
            candidates = candidates[candidate_mask]
        if candidates.numel() == 0:
            return
        candidate_scores = torch.nan_to_num(
            scores[candidates], nan=-float("inf")
        )
        count = max(1, math.ceil(candidates.numel() * self.add_hot_fraction))
        selected = candidates[torch.topk(candidate_scores, count).indices]
        xyz = params["means"][selected].detach().float().cpu().numpy()
        vertices = np.empty(
            len(xyz),
            dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ],
        )
        vertices["x"], vertices["y"], vertices["z"] = xyz.T
        vertices["red"] = vertices["green"] = vertices["blue"] = 255
        output_dir = Path(self.score_debug_dir)
        output_dir.mkdir(parents=True, exist_ok=True)        
        path = output_dir / f"step_{self._active_step:06d}_{name}_top10.ply"
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        with path.open("wb") as file:
            file.write(header.encode("ascii"))
            vertices.tofile(file)
