from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from torch import Tensor

from .texture_mcmc import TextureMCMCStrategy
from gsplat.utils import normalized_quat_to_rotmat


def read_voxel_plane_ply(path: str | Path):
    """Read plane anchors produced by ``build_voxel_plane_anchors.py``."""
    vertex = PlyData.read(str(path))["vertex"]
    required = ("x", "y", "z", "nx", "ny", "nz", "is_plane")
    missing = [name for name in required if name not in vertex.data.dtype.names]
    if missing:
        raise ValueError(f"plane-anchor PLY is missing properties: {missing}")

    def columns(names):
        return np.stack([np.asarray(vertex[name]) for name in names], axis=1)

    xyz = np.asarray(columns(("x", "y", "z")), dtype=np.float32)
    match_names = ("source_x", "source_y", "source_z")
    match_xyz = (
        np.asarray(columns(match_names), dtype=np.float32)
        if all(name in vertex.data.dtype.names for name in match_names)
        else xyz
    )
    normals = np.asarray(columns(("nx", "ny", "nz")), dtype=np.float32)
    is_plane = np.asarray(vertex["is_plane"], dtype=np.uint8) != 0
    if (
        len(xyz) == 0
        or not np.isfinite(xyz).all()
        or not np.isfinite(match_xyz).all()
    ):
        raise ValueError("plane-anchor PLY contains empty or invalid XYZ data")
    normal_norm = np.linalg.norm(normals, axis=1)
    invalid = is_plane & (~np.isfinite(normals).all(1) | (normal_norm < 1e-6))
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} planar anchors have invalid normals")
    normals[is_plane] /= normal_norm[is_plane, None]
    normals[~is_plane] = 0
    return xyz, match_xyz, normals, is_plane


@dataclass
class VoxelPlaneMCMCStrategy(TextureMCMCStrategy):
    """Texture MCMC with GeoGaussian-style planar voxel anchors.

    Initial GSs are matched once to voxel anchors. New and relocated GSs inherit
    the donor anchor. Planar initialization, add/relocate and active noise stay
    inside a finite slab around the plane; ordinary optimizer updates stay free.
    Radial reflection keeps every constrained GS near its inherited anchor.
    """

    plane_anchor_ply: str | None = None
    anchor_voxel_size: float = 0.1
    limit_anchor_distance: bool = True
    anchor_max_distance: float = 0.2
    keep_one_per_anchor: bool = True
    align_initial_plane_gs: bool = True
    initial_plane_thickness: float = 1e-4
    plane_normal_tolerance: float = 0.03
    plane_knn: int = 8
    plane_debug_dir: str | None = None
    plane_debug_anchor_count: int = 100
    plane_s3_max: float = field(default=0.001, init=False, repr=False)

    def initialize_state(self) -> Dict[str, Any]:
        state = super().initialize_state()
        state.update(
            anchor_xyz=None,
            anchor_normals=None,
            anchor_is_plane=None,
            anchor_ids=None,
            anchor_keeper_ids=None,
            gs_knn_ids=None,
            initial_tangent_scale=None,
            plane_gs_ids=None,
            visible_gs_ids=None,
            plane_loss_step=-1,
        )
        return state

    def check_sanity(self, params, optimizers):
        super().check_sanity(params, optimizers)
        if not self.plane_anchor_ply:
            raise ValueError("VoxelPlaneMCMCStrategy requires plane_anchor_ply")
        if self.anchor_voxel_size <= 0:
            raise ValueError("anchor_voxel_size must be positive")
        if self.limit_anchor_distance and self.anchor_max_distance <= 0:
            raise ValueError("anchor_max_distance must be positive")
        if min(self.initial_plane_thickness, self.plane_normal_tolerance) <= 0:
            raise ValueError("plane thickness and normal tolerance must be positive")
        if self.plane_s3_max <= 0:
            raise ValueError("plane_s3_max must be positive")
        if self.plane_knn < 2:
            raise ValueError("plane_knn must be at least 2")
        if self.plane_debug_anchor_count <= 0:
            raise ValueError("plane_debug_anchor_count must be positive")

    @torch.no_grad()
    def set_surface_anchors(self, state: Dict[str, Any], means: Tensor) -> None:
        xyz, match_xyz, normals, is_plane = read_voxel_plane_ply(
            self.plane_anchor_ply
        )
        self.set_plane_anchors(
            state, means, xyz, match_xyz, normals, is_plane
        )

    @torch.no_grad()
    def set_plane_anchors(
        self,
        state: Dict[str, Any],
        means: Tensor,
        xyz: np.ndarray,
        match_xyz: np.ndarray,
        normals: np.ndarray,
        is_plane: np.ndarray,
    ) -> None:
        """Bind preloaded anchors, allowing block trainers to crop them first."""
        xyz = np.asarray(xyz, dtype=np.float32)
        match_xyz = np.asarray(match_xyz, dtype=np.float32)
        normals = np.asarray(normals, dtype=np.float32)
        is_plane = np.asarray(is_plane, dtype=bool)
        count = len(xyz)
        if count == 0:
            raise ValueError("no voxel-plane anchors were selected")
        if match_xyz.shape != (count, 3) or normals.shape != (count, 3):
            raise ValueError("voxel-plane XYZ and normals must have shape [N, 3]")
        if is_plane.shape != (count,):
            raise ValueError("voxel-plane is_plane must have shape [N]")
        if not (
            np.isfinite(xyz).all()
            and np.isfinite(match_xyz).all()
            and np.isfinite(normals).all()
        ):
            raise ValueError("voxel-plane anchors contain non-finite values")
        means_cpu = means.detach().float().cpu().numpy()
        started = time.perf_counter()
        distances, ids = cKDTree(match_xyz).query(means_cpu, k=1, workers=-1)
        invalid = ~np.isfinite(distances)
        if invalid.any():
            raise ValueError(
                f"{int(invalid.sum())} initial GSs have no finite nearest anchor"
            )
        # This is a diagnostic only: report the farthest GS from the actual
        # anchor position it was assigned to, without rejecting any distance.
        anchor_distances = np.linalg.norm(means_cpu - xyz[ids], axis=1)
        max_anchor_distance = float(anchor_distances.max())
        print(
            "Voxel-plane anchor check: "
            f"max GS-to-anchor distance={max_anchor_distance:.6f} m."
        )

        counts = np.bincount(ids, minlength=len(xyz))
        occupied = counts > 0
        removed_anchor_count = int((~occupied).sum())
        if removed_anchor_count:
            # Block cropping can retain an anchor that is not the nearest one
            # for any initial GS. Such anchors cannot own, protect or receive a
            # GS, so discard them and compact the GS-to-anchor indices.
            old_to_new = np.full(len(xyz), -1, dtype=np.int64)
            old_to_new[occupied] = np.arange(occupied.sum(), dtype=np.int64)
            ids = old_to_new[ids]
            xyz = xyz[occupied]
            normals = normals[occupied]
            is_plane = is_plane[occupied]
            counts = counts[occupied]
            if self.verbose:
                print(
                    f"Removed {removed_anchor_count:,} voxel-plane anchors "
                    "with no initial GS."
                )

        order = np.argsort(ids, kind="stable")
        first = np.r_[True, ids[order][1:] != ids[order][:-1]]
        keeper_ids = np.full(len(xyz), -1, dtype=np.int64)
        keeper_ids[ids[order[first]]] = order[first]

        gs_knn = self._find_gs_knn(means_cpu)

        # Estimate the planar sampling interval from eight neighbors. Exact and
        # near-exact duplicates are ignored so one repeated LiDAR point cannot
        # collapse both tangent axes. Query in chunks to bound peak CPU memory.
        neighbor_distance = np.full(len(means_cpu), np.nan, dtype=np.float32)
        if len(means_cpu) > 1:
            tree = cKDTree(means_cpu)
            k = min(9, len(means_cpu))  # self + up to eight neighbors
            chunk_size = 500_000
            for start in range(0, len(means_cpu), chunk_size):
                end = min(start + chunk_size, len(means_cpu))
                distances = tree.query(
                    means_cpu[start:end], k=k, workers=-1
                )[0][:, 1:]
                valid = np.isfinite(distances) & (distances >= 1e-5)
                has_valid = valid.any(axis=1)
                if has_valid.any():
                    values = np.where(
                        valid[has_valid], distances[has_valid], np.nan
                    )
                    chunk_distance = neighbor_distance[start:end]
                    chunk_distance[has_valid] = np.nanmedian(
                        values, axis=1
                    ).astype(np.float32)
        tangent_scale = np.sqrt(
            np.clip(neighbor_distance**2, 1e-7, 1e-2) / 2.0
        ).astype(np.float32)

        device, dtype = means.device, means.dtype
        state.update(
            anchor_xyz=torch.as_tensor(xyz, dtype=dtype, device=device),
            anchor_normals=torch.as_tensor(normals, dtype=dtype, device=device),
            anchor_is_plane=torch.as_tensor(is_plane, device=device),
            anchor_ids=torch.as_tensor(ids, dtype=torch.long, device=device),
            anchor_keeper_ids=torch.as_tensor(
                keeper_ids, dtype=torch.long, device=device
            ),
            gs_knn_ids=torch.as_tensor(
                gs_knn, dtype=torch.int32, device=device
            ),
            initial_tangent_scale=torch.as_tensor(
                tangent_scale, dtype=dtype, device=device
            ),
            noise_active_until=torch.full(
                (len(means),), -1, dtype=torch.long, device=device
            ),
        )
        if self.verbose:
            print(
                f"Plane anchors: {len(xyz):,}, planar={int(is_plane.sum()):,} "
                f"({is_plane.mean():.1%}); matched {len(means):,} GSs in "
                f"{time.perf_counter() - started:.2f}s."
            )

    @staticmethod
    def _tracked_state(state: Dict[str, Any]) -> Dict[str, Tensor]:
        excluded = {
            "binoms",
            "surface_anchor",
            "noise_active_until",
            "anchor_xyz",
            "anchor_normals",
            "anchor_is_plane",
            "anchor_ids",
            "anchor_keeper_ids",
            "gs_knn_ids",
            "initial_tangent_scale",
            "plane_gs_ids",
            "visible_gs_ids",
        }
        return {
            key: value
            for key, value in state.items()
            if key not in excluded and isinstance(value, Tensor) and value.ndim > 0
        }

    def _find_gs_knn(self, means: np.ndarray) -> np.ndarray:
        """Return the nearest GS IDs, excluding each query GS itself."""
        n = len(means)
        neighbors = np.repeat(
            np.arange(n, dtype=np.int32)[:, None], self.plane_knn, axis=1
        )
        if n <= 1:
            return neighbors
        k = min(self.plane_knn + 1, n)
        found = cKDTree(means).query(means, k=k, workers=-1)[1]
        if found.ndim == 1:
            found = found[:, None]
        found = found.astype(np.int32, copy=False)
        rows = np.arange(n, dtype=np.int32)[:, None]
        valid = found != rows
        slots = np.cumsum(valid, axis=1) - 1
        take = valid & (slots < self.plane_knn)
        row_ids = np.broadcast_to(rows, found.shape)[take]
        neighbors[row_ids, slots[take]] = found[take]
        return neighbors

    @torch.no_grad()
    def _update_gs_knn(self, state: Dict[str, Any], means: Tensor) -> None:
        started = time.perf_counter()
        points = means.detach().float().cpu().numpy()
        state["gs_knn_ids"] = torch.as_tensor(
            self._find_gs_knn(points), dtype=torch.int32, device=means.device
        )
        if self.verbose:
            print(
                f"Updated {self.plane_knn}-NN for {len(points):,} GSs in "
                f"{time.perf_counter() - started:.2f}s."
            )

    @torch.no_grad()
    def _dead_mask(self, opacity: Tensor) -> Tensor:
        dead = super()._dead_mask(opacity)
        if (
            self.keep_one_per_anchor
            and getattr(self, "_active_state", None) is not None
        ):
            keepers = self._active_state["anchor_keeper_ids"]
            dead[keepers[keepers >= 0]] = False
        return dead

    @torch.no_grad()
    def _inherit_surface_anchors(
        self,
        state: Dict[str, Any],
        indices: Tensor,
        source_indices: Tensor,
        append: bool = False,
    ) -> None:
        ids = state["anchor_ids"]
        inherited = ids[source_indices].clone()
        if append:
            state["anchor_ids"] = torch.cat((ids, inherited))
        else:
            ids[indices] = inherited

    @torch.no_grad()
    def _constrain_surface_anchored_gs(
        self, params, state: Dict[str, Any], indices: Tensor
    ) -> None:
        if indices.numel() == 0:
            return
        anchor_ids = state["anchor_ids"][indices]
        centers = state["anchor_xyz"][anchor_ids]
        normals = state["anchor_normals"][anchor_ids]
        planar = state["anchor_is_plane"][anchor_ids]
        offset = params["means"][indices] - centers

        # Allow a finite slab around the LiDAR plane. Only an active GS that
        # crosses the slab boundary is reflected; in-range offsets are untouched.
        plane_offset = offset[planar]
        plane_normal = normals[planar]
        signed = (plane_offset * plane_normal).sum(-1, keepdim=True)
        half_width = self.plane_normal_tolerance
        phase = (signed + half_width).remainder(4.0 * half_width)
        bounded = torch.where(
            phase <= 2.0 * half_width,
            phase - half_width,
            3.0 * half_width - phase,
        )
        offset[planar] = plane_offset + (bounded - signed) * plane_normal

        if self.limit_anchor_distance:
            distance = offset.norm(dim=-1, keepdim=True)
            period = 2.0 * self.anchor_max_distance
            reflected = distance.remainder(period)
            reflected = torch.minimum(reflected, period - reflected)
            offset *= reflected / distance.clamp_min(1e-12)
        params["means"][indices] = centers + offset

    @torch.no_grad()
    def _constrain_planar_shape(
        self, params, state: Dict[str, Any], indices: Tensor
    ) -> None:
        if indices.numel() == 0:
            return
        params["scales"][indices] = params["scales"][indices].clamp_max(
            math.log(self.anchor_voxel_size)
        )
        anchor_ids = state["anchor_ids"][indices]
        indices = indices[state["anchor_is_plane"][anchor_ids]]
        if indices.numel() == 0:
            return

        target = state["anchor_normals"][state["anchor_ids"][indices]]
        quats = F.normalize(params["quats"][indices], dim=-1)
        current = normalized_quat_to_rotmat(quats)[:, :, 2]
        target = torch.where(
            (current * target).sum(-1, keepdim=True) < 0, -target, target
        )
        dot = (current * target).sum(-1, keepdim=True).clamp(-1.0, 1.0)
        correction = F.normalize(
            torch.cat((1.0 + dot, torch.cross(current, target, dim=-1)), dim=-1),
            dim=-1,
        )
        cw, cx, cy, cz = correction.unbind(-1)
        qw, qx, qy, qz = quats.unbind(-1)
        params["quats"][indices] = F.normalize(
            torch.stack(
                (
                    cw * qw - cx * qx - cy * qy - cz * qz,
                    cw * qx + cx * qw + cy * qz - cz * qy,
                    cw * qy - cx * qz + cy * qw + cz * qx,
                    cw * qz + cx * qy - cy * qx + cz * qw,
                ),
                dim=-1,
            ),
            dim=-1,
        )
        params["scales"][indices, 2] = params["scales"][indices, 2].clamp_max(
            math.log(self.plane_s3_max)
        )

    @torch.no_grad()
    def _save_plane_gs(self, params, state: Dict[str, Any], phase: str) -> None:
        if not self.plane_debug_dir:
            return
        step = getattr(self, "_active_step", 0)
        if step % 3000 != 0:
            return 
        anchor_ids_tensor = state["anchor_ids"]
        counts = torch.bincount(
            anchor_ids_tensor, minlength=len(state["anchor_is_plane"])
        )
        plane_anchors = torch.where(state["anchor_is_plane"] & (counts > 0))[0]
        if plane_anchors.numel() == 0:
            return
        count = min(self.plane_debug_anchor_count, plane_anchors.numel())
        selected = plane_anchors[
            torch.topk(counts[plane_anchors], k=count).indices
        ]
        selected_mask = torch.zeros_like(state["anchor_is_plane"])
        selected_mask[selected] = True
        indices = torch.where(selected_mask[anchor_ids_tensor])[0]
        if indices.numel() == 0:
            return

        means = params["means"][indices].detach().float().cpu().numpy()
        quats = F.normalize(params["quats"][indices], dim=-1)
        normals = normalized_quat_to_rotmat(
            quats
        )[:, :, 2].detach().float().cpu().numpy()
        quats = quats.detach().float().cpu().numpy()
        scales = params["scales"][indices].detach().float().cpu().numpy()
        opacity = params["opacities"][indices].flatten()
        opacity = opacity.detach().float().cpu().numpy()
        if "sh0" in params:
            sh0 = params["sh0"][indices].transpose(1, 2).flatten(1)
            shN = params["shN"][indices].transpose(1, 2).flatten(1)
        else:
            sh0 = (torch.sigmoid(params["colors"][indices]) - 0.5) / 0.28209479
            shN = sh0.new_empty((len(indices), 0))
        sh0 = sh0.detach().float().cpu().numpy()
        shN = shN.detach().float().cpu().numpy()
        selected_anchor_ids = anchor_ids_tensor[indices]
        anchor_ids = selected_anchor_ids.detach().cpu().numpy()

        properties = [
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ]
        properties += [(f"f_dc_{i}", "f4") for i in range(sh0.shape[1])]
        properties += [(f"f_rest_{i}", "f4") for i in range(shN.shape[1])]
        properties += [("opacity", "f4")]
        properties += [(f"scale_{i}", "f4") for i in range(3)]
        properties += [(f"rot_{i}", "f4") for i in range(4)]
        vertices = np.empty(
            len(indices),
            dtype=properties,
        )
        for axis, name in enumerate(("x", "y", "z")):
            vertices[name] = means[:, axis]
            vertices[f"n{name}"] = normals[:, axis]
            vertices[f"scale_{axis}"] = scales[:, axis]
        for axis in range(sh0.shape[1]):
            vertices[f"f_dc_{axis}"] = sh0[:, axis]
        for axis in range(shN.shape[1]):
            vertices[f"f_rest_{axis}"] = shN[:, axis]
        for axis in range(4):
            vertices[f"rot_{axis}"] = quats[:, axis]
        vertices["opacity"] = opacity

        output_dir = Path(self.plane_debug_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        path = output_dir / f"plane_gs_{step:06d}_{phase}.ply"
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)
        if self.verbose:
            n_anchors = np.unique(anchor_ids).size
            print(
                f"Saved {len(vertices):,} GSs from {n_anchors} plane anchors "
                f"after {phase}: {path}"
            )

    @torch.no_grad()
    def _relocate_gs(self, params, optimizers, binoms, scene=None) -> int:
        opacity = torch.sigmoid(params["opacities"].flatten())
        indices = torch.where(self._dead_mask(opacity))[0]
        count = super()._relocate_gs(params, optimizers, binoms, scene)
        if count:
            self._constrain_surface_anchored_gs(
                params, self._active_state, indices
            )
            self._constrain_planar_shape(params, self._active_state, indices)
            self._save_plane_gs(params, self._active_state, "relocate")
        return count

    @torch.no_grad()
    def _add_new_gs(self, params, optimizers, binoms, scene=None) -> int:
        start = len(params["means"])
        count = super()._add_new_gs(params, optimizers, binoms, scene)
        if count:
            indices = torch.arange(
                start, start + count, device=params["means"].device
            )
            self._constrain_surface_anchored_gs(
                params, self._active_state, indices
            )
            self._constrain_planar_shape(params, self._active_state, indices)
            self._save_plane_gs(params, self._active_state, "add")
        return count

    @torch.no_grad()
    def initialize_plane_gaussians(self, params, state: Dict[str, Any]) -> None:
        """Apply GeoGaussian initialization to reliable planar LiDAR GSs."""
        if not self.align_initial_plane_gs:
            state["initial_tangent_scale"] = None
            return
        anchor_ids = state["anchor_ids"]
        indices = torch.where(state["anchor_is_plane"][anchor_ids])[0]
        if indices.numel() == 0:
            state["initial_tangent_scale"] = None
            return

        normals = state["anchor_normals"][anchor_ids[indices]]
        # Preprocessing orients the dominant component positively, so this
        # shortest-arc quaternion is stable even for walls and the ground.
        quats = torch.stack(
            (
                1.0 + normals[:, 2],
                -normals[:, 1],
                normals[:, 0],
                torch.zeros_like(normals[:, 0]),
            ),
            dim=-1,
        )
        params["quats"][indices] = F.normalize(quats, dim=-1)
        tangent = state["initial_tangent_scale"][indices]
        # A point with no valid non-duplicate neighbor keeps the robust scale
        # produced by the ordinary GS initializer instead of using a fake
        # voxel-sized or near-zero tangent scale.
        fallback = params["scales"][indices].exp().mean(dim=-1)
        tangent = torch.where(torch.isfinite(tangent), tangent, fallback)
        params["scales"][indices] = torch.stack(
            (
                tangent.log(),
                tangent.log(),
                torch.full_like(tangent, self.initial_plane_thickness).log(),
            ),
            dim=-1,
        )
        state["initial_tangent_scale"] = None
        self._constrain_surface_anchored_gs(params, state, indices)
        if self.verbose:
            print(
                f"GeoGaussian-initialized {len(indices):,} planar GSs "
                f"(thickness={self.initial_plane_thickness:g} m)."
            )

    @torch.no_grad()
    def step_pre_backward(self, params, optimizers, state, step, info):
        del params, optimizers
        radii = info["radii"]
        visible = (
            (radii > 0).all(-1).any(0)
            if radii.ndim == 3
            else (radii > 0).any(0)
        )
        state["visible_gs_ids"] = torch.where(visible)[0]
        state["plane_gs_ids"] = torch.where(
            state["anchor_is_plane"][state["anchor_ids"]]
        )[0]
        state["plane_loss_step"] = step

    def step_post_backward(
        self, params, optimizers, state, step, info, lr, scene=None
    ):
        super().step_post_backward(
            params, optimizers, state, step, info, lr, scene
        )
        if (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            self._update_gs_knn(state, params["means"])
