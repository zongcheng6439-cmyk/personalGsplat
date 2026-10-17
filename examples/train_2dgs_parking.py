"""Parking-scene 2D Gaussian Splatting baseline.

This keeps the existing 2DGS trainer unchanged and only adapts it to the
current parking COLMAP data loader (masks, LiDAR initialization and dense
rendered depth).  It intentionally uses the standard DefaultStrategy so the
first experiment measures 2DGS itself, without texture MCMC or anchors.

Run inside the gsplat Docker container::

    cd /code/gsplat/examples
    python train_2dgs_parking.py \
        --data-dir /lutu/parking01 \
        --result-dir /lutu/parking01/gsplat_2dgs_edge_depth \
        --data-factor 1 \
        --max-steps 30000 \
        --sh-degree 0 \
        --depth-lambda 0.05 \
        --dist-lambda 0.01 \
        --dist-start-iter 2000 \
        --normal-lambda 0.01 \
        --normal-start-iter 4000
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import torch
import tyro

import simple_trainer_2dgs as base
from datasets.colmap_adc import Dataset as ADCDataset
from datasets.colmap_adc import Parser as ADCParser
from gsplat.exporter import export_splats


@dataclass
class Config(base.Config):
    # Paths as seen inside the gsplat Docker container.
    data_dir: str = "/lutu/parking01"
    result_dir: str = "/lutu/parking01/gsplat_2dgs_edge_depth"
    data_factor: int = 1

    normalize_world_space: bool = False
    # Use the native 2DGS DefaultStrategy gradient path, not AbsGS or MCMC.
    absgrad: bool = False

    # Use every registered image for training. Evaluation is skipped when the
    # validation split is empty.
    test_every: int = 0
    eval_steps: List[int] = field(default_factory=list)
    save_steps: List[int] = field(default_factory=lambda: [30_000])
    disable_viewer: bool = True
    disable_video: bool = True
    # Save the initialization at step 0, then one PLY every N training steps.
    ply_every: int = 3_000

    # colmap_adc automatically prefers sparse/0/extracted.ply when it exists.
    init_type: str = "sfm"
    near_plane: float = 0.01
    far_plane: float = 10_000.0

    depth_loss: bool = True
    depth_lambda: float = 0.05
    depth_refine_lambda: float = 0.01
    depth_final_ramp_steps: int = 3_000

    # Keep the native DefaultStrategy threshold. 0.05 removed too many sparse
    # parking-scene initialization points, especially after opacity resets.
    prune_opa: float = 0.005

    # Native 2DGS regularizers. They start after appearance and geometry have
    # received a short warm-up.
    dist_loss: bool = True
    dist_lambda: float = 0.01
    dist_start_iter: int = 2_000
    normal_loss: bool = True
    normal_lambda: float = 0.01
    normal_start_iter: int = 4_000

    # The third scale is unused by the 2DGS rasterizer. Give exported splats a
    # small physical thickness so common 3DGS viewers display them as disks.
    export_thickness: float = 1e-4


class ParkingParser(ADCParser):
    """Use the parking loader and require edge-filtered rendered depth."""

    def __init__(self, data_dir, factor=1, normalize=False, test_every=0):
        super().__init__(
            data_dir=data_dir,
            factor=factor,
            normalize=normalize,
            test_every=test_every,
            load_exposure=False,
        )
        edge_depth_dir = (Path(data_dir) / "depths_render_color_edge").resolve()
        if not edge_depth_dir.is_dir():
            raise FileNotFoundError(
                f"Missing edge-filtered depth directory: {edge_depth_dir}"
            )

        missing = [path for path in self.depth_paths if not Path(path).is_file()]
        if missing:
            preview = "\n".join(missing[:5])
            raise FileNotFoundError(
                f"Missing {len(missing)} edge-filtered depth maps; first paths:\n"
                f"{preview}"
            )

        outside_edge_dir = [
            path
            for path in self.depth_paths
            if not Path(path).resolve().is_relative_to(edge_depth_dir)
        ]
        if outside_edge_dir:
            raise RuntimeError(
                "Parking 2DGS must use depths_render_color_edge; the dataset "
                f"resolved another depth source: {outside_edge_dir[0]}"
            )


class ParkingDataset(ADCDataset):
    """Expose every valid pixel from the edge-filtered rendered depth map."""

    def __getitem__(self, item):
        data = super().__getitem__(item)
        if self.load_depths:
            depth_mask = data.pop("depth_gs_mask").bool()
            depth = data.pop("depth_gs")
            if not depth_mask.any():
                raise ValueError(f"Image {self.image_names[item]} has no valid depth")
            data["depths"] = depth.float()
            data["depth_mask"] = depth_mask
        return data


class Runner(base.Runner):
    def __init__(self, cfg: Config):
        if cfg.export_thickness <= 0:
            raise ValueError("export_thickness must be positive")
        if cfg.ply_every < 0:
            raise ValueError("ply_every must be non-negative; use 0 to disable")
        if cfg.app_opt:
            raise ValueError("This baseline does not support app_opt")

        old_parser, old_dataset = base.Parser, base.Dataset
        base.Parser, base.Dataset = ParkingParser, ParkingDataset
        try:
            super().__init__(cfg)
        finally:
            base.Parser, base.Dataset = old_parser, old_dataset

    @torch.no_grad()
    def eval(self, step: int):
        if len(self.valset) == 0:
            print("Evaluation skipped: all images are used for training.")
            return
        super().eval(step)

    @torch.no_grad()
    def render_traj(self, step: int):
        if self.cfg.disable_video:
            return
        super().render_traj(step)

    @torch.no_grad()
    def save_training_ply(self, step: int):
        ply_dir = os.path.join(self.cfg.result_dir, "ply")
        os.makedirs(ply_dir, exist_ok=True)

        invalid_counts = {}
        for name, value in self.splats.items():
            count = (~torch.isfinite(value)).sum().item()
            if count:
                invalid_counts[name] = count
        if invalid_counts:
            raise FloatingPointError(
                f"Refusing to export non-finite 2DGS parameters at step {step}: "
                f"{invalid_counts}"
            )

        scales = self.splats["scales"].detach().clone()
        scales[:, 2] = math.log(self.cfg.export_thickness)
        output = os.path.join(ply_dir, f"gsplat_2dgs_{step:06d}.ply")
        export_splats(
            means=self.splats["means"],
            scales=scales,
            quats=self.splats["quats"],
            opacities=self.splats["opacities"],
            sh0=self.splats["sh0"],
            shN=self.splats["shN"],
            format="ply",
            save_to=output,
        )
        print(f"2DGS PLY saved to {output}")

    def train(self):
        super().train()
        self.save_training_ply(self.cfg.max_steps - 1)


def main(cfg: Config):
    runner = Runner(cfg)
    if cfg.ckpt is None:
        runner.train()
    else:
        ckpt = torch.load(cfg.ckpt, map_location=runner.device)
        for key in runner.splats:
            runner.splats[key].data = ckpt["splats"][key]
        runner.eval(step=ckpt["step"])
        runner.render_traj(step=ckpt["step"])


if __name__ == "__main__":
    config = tyro.cli(Config)
    config.adjust_steps(config.steps_scaler)
    main(config)
