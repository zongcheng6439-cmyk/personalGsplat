from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .default import DefaultStrategy
from .ops import duplicate, split

if TYPE_CHECKING:
    from gsplat.scene import Scene


@dataclass
class TextureDefaultStrategy(DefaultStrategy):
    """Default densification augmented with per-pixel texture importance.

    ``texture_info`` is the top-contributor render weight multiplied by the
    corresponding weight-map pixel value. The ordinary screen-space gradient
    remains the base signal; texture importance only promotes additional GSs.
    """

    absgrad: bool = True
    grow_grad2d: float = 0.0008
    texture_weight: float = 1
    texture_quantile: float = 0.9
    grad_weight: float = 0.5
    grad_quantile: float = 0.9
    grow_score: float = 1.0
    top_k: int = 3
    texture_update_every: int = 1
    score_debug_dir: str | None = None
    surface_anchor_constraint: bool = False
    surface_anchor_max_distance: float = 0.1

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        state = super().initialize_state(scene_scale)
        state["texture_info"] = None
        state["texture_pixel_num"] = None
        state["absgrad_sum"] = None
        state["grad_count"] = None
        state["surface_anchor"] = None
        return state

    def check_sanity(self, params, optimizers):
        super().check_sanity(params, optimizers)
        if min(self.texture_weight, self.grad_weight) < 0:
            raise ValueError("score weights must be non-negative")
        if not 0.0 < self.texture_quantile <= 1.0 or not 0.0 < self.grad_quantile <= 1.0:
            raise ValueError("score quantiles must be in (0, 1]")
        if self.top_k <= 0 or self.texture_update_every <= 0:
            raise ValueError("top_k and texture_update_every must be positive")
        if self.surface_anchor_constraint and self.surface_anchor_max_distance <= 0:
            raise ValueError("surface_anchor_max_distance must be positive")

    @torch.no_grad()
    def set_surface_anchors(self, state: Dict[str, Any], means: Tensor) -> None:
        if self.surface_anchor_constraint:
            state["surface_anchor"] = means.detach().clone()

    @torch.no_grad()
    def _constrain_surface_anchored_gs(self, params, state: Dict[str, Any]) -> None:
        anchors = state.get("surface_anchor")
        if not self.surface_anchor_constraint:
            return
        if anchors is None:
            raise RuntimeError("call set_surface_anchors() before training")
        means = params["means"]
        if anchors.shape != means.shape:
            raise RuntimeError(
                "surface_anchor and means must have identical shapes, got "
                f"{tuple(anchors.shape)} and {tuple(means.shape)}"
            )
        offset = means - anchors
        distance = offset.norm(dim=-1, keepdim=True)
        ratio = torch.clamp(
            self.surface_anchor_max_distance / distance.clamp_min(1e-12),
            max=1.0,
        )
        means.copy_(anchors + offset * ratio)

    def _update_state(self, params, state, info, packed: bool = False):
        super()._update_state(params, state, info, packed=packed)
        n = len(params["means"])
        if "texture_info" in info and "texture_pixel_num" in info:
            texture_info = info["texture_info"].detach()
            texture_pixel_num = info["texture_pixel_num"].detach()
            if texture_info.shape != (n,) or texture_pixel_num.shape != (n,):
                raise ValueError("texture statistics must have shape [num_gaussians]")
            if state["texture_info"] is None:
                state["texture_info"] = torch.zeros_like(texture_info)
                state["texture_pixel_num"] = torch.zeros_like(texture_pixel_num)
            state["texture_info"].add_(texture_info)
            state["texture_pixel_num"].add_(texture_pixel_num)

        means2d = info.get("means2d")
        grads = getattr(means2d, "absgrad", None) if means2d is not None else None
        if grads is None:
            return
        if state["absgrad_sum"] is None:
            state["absgrad_sum"] = torch.zeros(n, device=grads.device)
            state["grad_count"] = torch.zeros(n, device=grads.device)
        if grads.ndim == 3:
            visible = (info["radii"] > 0).all(dim=-1)
            ids = torch.where(visible)[1]
            grads = grads[visible]
        else:
            ids = info["gaussian_ids"]
        state["absgrad_sum"].index_add_(0, ids, grads.norm(dim=-1))
        state["grad_count"].index_add_(
            0, ids, torch.ones_like(ids, dtype=torch.float32)
        )

    def _growth_score(self, params, state: Dict[str, Any]) -> torch.Tensor:
        opacity = torch.sigmoid(params["opacities"].flatten())
        if state["texture_info"] is None:
            return opacity

        texture = state["texture_info"] / state["texture_pixel_num"].clamp_min(1e-4)
        texture = self._quantile_normalize(texture, self.texture_quantile)

        if state["absgrad_sum"] is None:
            grad = torch.zeros_like(opacity)
        else:
            grad = state["absgrad_sum"] / state["grad_count"].clamp_min(1.0)
            grad = self._quantile_normalize(grad, self.grad_quantile)
        return self.texture_weight * texture + self.grad_weight * grad

    @staticmethod
    def _quantile_normalize(values: Tensor, quantile: float) -> Tensor:
        values = torch.nan_to_num(
            values, nan=0.0, posinf=0.0, neginf=0.0
        ).clamp_min(0)
        positive = values[values > 0]
        if positive.numel() == 0:
            return torch.zeros_like(values)
        scale = torch.quantile(positive.float(), quantile).to(values)
        return (values / scale.clamp_min(1e-12)).clamp(0.0, 2.0)

    @torch.no_grad()
    def _save_top_score_gs(
        self, params, score: torch.Tensor, step: int, candidate_mask: torch.Tensor
    ) -> None:
        if self.score_debug_dir is None:
            return
        candidates = torch.where(candidate_mask)[0]
        if candidates.numel() == 0:
            return
        count = max(1, (candidates.numel() + 9) // 10)
        selected = candidates[torch.topk(score[candidates], count).indices]
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
        path = output_dir / f"step_{step:06d}_grow_top10.ply"
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

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        scene: Scene | None = None,
    ) -> Tuple[int, int]:
        score = self._growth_score(params, state)
        candidates = (state["count"] > 0) & (score >= self.grow_score)
        candidate_count = int(candidates.sum().item())
        self._save_top_score_gs(
            params, score, step, state["count"] > 0
        )
        if self.verbose:
            print(f"Texture growth candidates: {candidate_count}.")

        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = candidates & is_small
        is_split = candidates & ~is_small
        if step < self.refine_scale2d_stop_iter:
            is_split |= state["radii"] > self.grow_scale2d

        n_dupli = int(is_dupli.sum().item())
        n_split = int(is_split.sum().item())
        if n_dupli:
            duplicate(params, optimizers, state, is_dupli, scene=scene)
        is_split = torch.cat(
            [is_split, torch.zeros(n_dupli, dtype=torch.bool, device=score.device)]
        )
        if n_split:
            split(
                params,
                optimizers,
                state,
                is_split,
                revised_opacity=self.revised_opacity,
                scene=scene,
            )
        return n_dupli, n_split

    def step_post_backward(self, *args, **kwargs):
        state = kwargs.get("state")
        if state is None and len(args) >= 3:
            state = args[2]
        step = kwargs.get("step")
        if step is None and len(args) >= 4:
            step = args[3]
        did_refine = (
            step is not None
            and step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        )
        super().step_post_backward(*args, **kwargs)
        params = kwargs.get("params")
        if params is None and args:
            params = args[0]
        if state is not None and params is not None:
            self._constrain_surface_anchored_gs(params, state)
        if did_refine and state is not None:
            for key in (
                "texture_info",
                "texture_pixel_num",
                "absgrad_sum",
                "grad_count",
            ):
                if state.get(key) is not None:
                    state[key].zero_()
