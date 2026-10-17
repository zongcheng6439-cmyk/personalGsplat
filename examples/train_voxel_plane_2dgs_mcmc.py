"""Minimal parking 2DGS trainer with MCMC and static plane anchors."""

from dataclasses import dataclass, field

import torch
import tyro

import train_2dgs_parking as parking
from render_utils import compute_texture_statistics_2dgs
from gsplat.strategy.voxel_plane_2dgs_mcmc import VoxelPlaneMCMC2DStrategy


@dataclass
class Config(parking.Config):
    absgrad: bool = True
    strategy: VoxelPlaneMCMC2DStrategy = field(
        default_factory=lambda: VoxelPlaneMCMC2DStrategy(
            verbose=True,
            use_opacity_score=True,
            use_grad_score=True,
            use_texture_score=False,
            opacity_weight=1.0,
            texture_weight=0.0,
            grad_weight=1.0,
            keep_one_per_anchor=True,
        )
    )

    def adjust_steps(self, factor: float):
        super().adjust_steps(factor)
        self.strategy.refine_start_iter = int(
            self.strategy.refine_start_iter * factor
        )
        self.strategy.refine_stop_iter = int(
            self.strategy.refine_stop_iter * factor
        )
        self.strategy.refine_every = max(
            1, int(self.strategy.refine_every * factor)
        )
        self.strategy.active_noise_steps = int(
            self.strategy.active_noise_steps * factor
        )
        self.strategy.plane_loss_start_iter = int(
            self.strategy.plane_loss_start_iter * factor
        )


class Runner(parking.Runner):
    def __init__(self, cfg: Config):
        class TextureParkingDataset(parking.ParkingDataset):
            def __init__(self, *args, **kwargs):
                kwargs["load_weight_maps"] = (
                    cfg.strategy.use_texture_score
                    and kwargs.get("split") == "train"
                )
                super().__init__(*args, **kwargs)

        original_dataset = parking.ParkingDataset
        parking.ParkingDataset = TextureParkingDataset
        try:
            super().__init__(cfg)
        finally:
            parking.ParkingDataset = original_dataset
        self.strategy = cfg.strategy
        self.strategy.set_training_steps(cfg.max_steps)
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state()
        self.strategy.set_surface_anchors(
            self.strategy_state,
            self.splats["means"],
        )
        self.strategy.initialize_plane_gaussians(
            self.splats,
            self.strategy_state,
        )

    @torch.no_grad()
    def collect_strategy_statistics(self, info, data, masks, step):
        strategy = self.strategy
        if (
            not strategy.use_texture_score
            or step >= strategy.refine_stop_iter
            or step % strategy.texture_update_every != 0
        ):
            return
        weight_map = data["weight_map"].to(self.device)
        if masks is not None:
            weight_map = weight_map * masks
        texture_sum, texture_count = compute_texture_statistics_2dgs(
            info=info,
            pixel_importance=weight_map,
            num_gaussians=len(self.splats["means"]),
        )
        info["texture_info"] = texture_sum
        info["texture_pixel_num"] = texture_count


def main(cfg: Config):
    runner = Runner(cfg)
    if cfg.ckpt is None:
        runner.train()
        return
    checkpoint = torch.load(cfg.ckpt, map_location=runner.device)
    for key in runner.splats:
        runner.splats[key].data = checkpoint["splats"][key]
    runner.eval(step=checkpoint["step"])
    runner.render_traj(step=checkpoint["step"])


if __name__ == "__main__":
    config = tyro.cli(Config)
    config.adjust_steps(config.steps_scaler)
    main(config)
