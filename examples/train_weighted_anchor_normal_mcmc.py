#!/usr/bin/env python3
"""Weighted Anchor MCMC with a simple LiDAR-anchor shape loss."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from plyfile import PlyData
from scipy.spatial import cKDTree

from gsplat.distributed import cli
from gsplat.utils import normalized_quat_to_rotmat
import simple_trainer_texture as base
import train_weighted_anchor_mcmc as weighted


@dataclass
class Config(weighted.Config):
    anchor_normal_ply: str = ""
    anchor_shape_weight: float = 1e-3
    anchor_shape_ratio: float = 0.2


class Runner(weighted.Runner):
    def bind_strategy_state(self) -> None:
        super().bind_strategy_state()
        path = Path(self.cfg.anchor_normal_ply).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"anchor normal PLY does not exist: {path}")

        vertex = PlyData.read(str(path))["vertex"].data
        xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")])
        normal = np.column_stack([vertex[name] for name in ("nx", "ny", "nz")])
        anchor_xyz = self.strategy_state["anchor_xyz"]
        distance, anchor_id = cKDTree(anchor_xyz.cpu().numpy()).query(xyz, k=1)
        if np.any(distance > 1e-4) or len(np.unique(anchor_id)) != len(anchor_id):
            raise ValueError("normal PLY does not match the anchor weight PLY")

        self.anchor_normals = torch.zeros_like(anchor_xyz)
        normal = torch.as_tensor(normal, dtype=anchor_xyz.dtype, device=self.device)
        self.anchor_normals[torch.as_tensor(anchor_id, device=self.device)] = F.normalize(
            normal, dim=-1
        )

    def extra_training_loss(self, info, **kwargs):
        del kwargs
        if self.cfg.anchor_shape_weight == 0:
            return None
        visible = torch.where((info["radii"] > 0).any(dim=0))[0]
        normal = self.anchor_normals[self.strategy_state["anchor_ids"][visible]]
        valid = normal.norm(dim=-1) > 0
        if not valid.any():
            return None

        visible, normal = visible[valid], normal[valid]
        rotation = normalized_quat_to_rotmat(
            F.normalize(self.splats["quats"][visible], dim=-1)
        )
        local_normal = torch.einsum(
            "nij,nj->ni", rotation.transpose(1, 2), normal
        )
        scale2 = self.splats["scales"][visible].mul(2).exp()
        normal_var = (local_normal.square() * scale2).sum(dim=-1)
        tangent_var = (scale2.sum(dim=-1) - normal_var).mul(0.5).clamp_min(1e-12)
        log_ratio = 0.5 * torch.log(normal_var.clamp_min(1e-12) / tangent_var)
        excess = F.relu(log_ratio - np.log(self.cfg.anchor_shape_ratio))
        return self.cfg.anchor_shape_weight * excess.square().mean()


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    if cfg.anchor_shape_weight < 0 or not 0 < cfg.anchor_shape_ratio < 1:
        raise ValueError("anchor shape weight/ratio must be >= 0 and in (0, 1)")
    original_runner = base.Runner
    base.Runner = Runner
    try:
        return base.main(local_rank, world_rank, world_size, cfg)
    finally:
        base.Runner = original_runner


if __name__ == "__main__":
    config = tyro.cli(Config)
    if config.backend == "dgx":
        import torch_dgx  # noqa: F401
    config.adjust_steps(config.steps_scaler)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one GPU")
    cli(main, config, verbose=True)
