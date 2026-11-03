r"""Standalone texture-MCMC experiment with LiDAR voxel-plane constraints.

Run inside the gsplat training environment after building the plane anchors

Inputs: COLMAP images/cameras, initialization point cloud, and plane-anchor PLY.
Outputs: trained Gaussian model, checkpoints, and optional geometry previews.
From the repository root (replace the example paths):

    python examples/train_voxel_plane_mcmc.py original \
        --data-dir /path/to/scene --result-dir /path/to/output \
        --strategy.plane-anchor-ply /path/to/plane_anchors.ply --pgsr-loss

Use ``original`` for the fixed-anchor method or ``dynamic`` for dynamic voxel
ownership. Both subcommands expose the same trainer and geometry-loss options.

    python examples/train_voxel_plane_mcmc.py original --help
    python examples/train_voxel_plane_mcmc.py dynamic --help

ScanNet++ 068ba2946c depth supervision (both original and dynamic):

    python examples/train_voxel_plane_mcmc.py original \
        --data-dir /media/zc/SSD/GS/scannetTpp/068ba2946c \
        --result-dir results/068ba2946c_voxel_plane \
        --strategy.plane-anchor-ply /media/zc/SSD/GS/scannetTpp/068ba2946c/cgal_voxel_plane_anchors.ply \
        --strategy.anchor-voxel-size 0.05 \
        --depth-loss --depth-tag uint16 --depth-dir depth --depth-scale 1000

Depth arguments are inherited from simple_trainer_texture.Config and passed
through its Runner to colmap_adc.Parser. uint16 uses millimeter camera-z PNGs
(zero invalid); auto reads prepared format metadata; la uses legacy encoding.
For filtered uint16 PNGs use --depth-dir depths_render_color_edge instead.
The anchor voxel size must match the value used to generate the anchors.
The depth tag does not enable supervision: also pass --depth-loss.

Optional opacity binarization (Bernoulli entropy on visible Gaussian logits):
append --opacity-entropy-lambda 0.01 --opacity-entropy-start-iter 7000
--opacity-entropy-warmup-steps 1000 to either training command. Weight defaults
to 0 (disabled); --opacity-entropy-stop-iter -1 keeps it on after warmup.
This is independent of --opacity-reg, which penalizes mean opacity towards 0.
Logs at --tb-every include raw/weighted entropy and visible opacity fractions.

Enable the isolated PGSR geometry losses with ``--pgsr-loss``. Plane normals
and camera-to-plane distances are rendered per pixel through gsplat
``extra_signals`` (one additional rasterization on PGSR steps), divided into
the unbiased ray-plane depth, then supervised by PGSR's single-view normal
loss and a multi-view homography NCC (rendered planes replace the PCA fit).
The default schedule starts at step 5000, warms up for 5000 steps, and
evaluates every two steps. Set ``--pgsr-sv-lambda 0`` for NCC alone.

Add ``--pgsr-vis`` (usable with or without ``--pgsr-loss``) to save colorized
unbiased-depth / normal / alpha maps of the first validation views into
``result_dir/pgsr_vis/step_XXXXXX/`` on every evaluation.

All visible GSs use KNN local smoothing, including planar-anchor GSs.
Both normal and position losses require neighbors to agree on a local plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from nerfview import apply_float_colormap

from gsplat.distributed import cli
from opacity_regularization import opacity_entropy_loss
from gsplat.strategy.plane_mcmc import PlaneMCMCStrategy
from gsplat.strategy.voxel_plane_mcmc import VoxelPlaneMCMCStrategy
from gsplat.utils import normalized_quat_to_rotmat
import simple_trainer_texture as base


def _dynamic_strategy() -> PlaneMCMCStrategy:
    return PlaneMCMCStrategy(
        absgrad=True,
        use_texture_score=True,
        use_grad_score=True,
        texture_weight=2,
        grad_weight=1,
        active_noise_steps=-1,
        noise_injection_stop_iter=-1,
        noise_scale=30,
        noise_opacity_t=0.054,
        noise_opacity_k=100.0,
        limit_anchor_distance=False,
        anchor_max_distance=0.3,
        initial_plane_thickness=1e-3,
        plane_normal_tolerance=0.02,
        verbose=True,
    )


def _original_strategy() -> VoxelPlaneMCMCStrategy:
    return VoxelPlaneMCMCStrategy(
        absgrad=True,
        use_texture_score=True,
        use_grad_score=True,
        texture_weight=2,
        grad_weight=1,
        active_noise_steps=-1,
        noise_injection_stop_iter=-1,
        anchor_voxel_size=0.1,
        noise_scale=30,
        noise_opacity_t=0.054,
        noise_opacity_k=100.0,
        limit_anchor_distance=True,
        anchor_max_distance=0.3,
        initial_plane_thickness=1e-3,
        plane_normal_tolerance=0.02,
        verbose=True,
    )


@dataclass
class Config(base.Config):
    normalize_world_space: bool = False
    means_lr: float = 1.6e-5
    init_opa: float = 0.5
    init_scale: float = 0.1
    opacity_reg: float = 0.01
    # Bernoulli entropy on currently visible Gaussian opacities; 0 disables it.
    opacity_entropy_lambda: float = 0.0
    opacity_entropy_start_iter: int = 7_000
    opacity_entropy_warmup_steps: int = 1_000
    # Exclusive stop iteration; -1 means no scheduled stop.
    opacity_entropy_stop_iter: int = -1
    scale_reg: float = 0.01
    anchor_normal_lambda: float = 0.1
    knn_plane_lambda: float = 0.5
    knn_normal_lambda: float = 5
    plane_s3_max: float = 0.005
    plane_scale_lambda: float = 0.01
    anchor_scale_max_ratio: float = 2.0
    anchor_scale_lambda: float = 0.02
    save_plane_gs_after_refine: bool = False

    # Optional PGSR block. Renders per-pixel plane parameters (shortest-axis
    # normal + camera-to-plane distance) through gsplat extra_signals, derives
    # the unbiased ray-plane depth, and applies PGSR's single-view normal loss
    # plus a multi-view homography NCC that reuses the NCC module with the
    # rendered planes instead of PCA-fitted ones. To remove this block,
    # delete these fields, the pgsr_geom import, and the marked Runner methods.
    pgsr_loss: bool = True
    pgsr_sv_lambda: float = 0.015
    pgsr_mv_lambda: float = 0.15
    pgsr_start_iter: int = 5_000
    pgsr_warmup_steps: int = 5_000
    pgsr_every: int = 2
    pgsr_samples: int = 4_096

    # PGSR visualization: on each val eval, save rendered unbiased-depth /
    # normal / alpha maps for the first pgsr_vis_count validation views into
    # result_dir/pgsr_vis/step_XXXXXX/. Independent of --pgsr-loss so the
    # geometry can be inspected without enabling the losses.
    pgsr_vis: bool = False
    pgsr_vis_count: int = 10

    strategy: VoxelPlaneMCMCStrategy = field(default_factory=_original_strategy)

    def adjust_steps(self, factor: float):
        super().adjust_steps(factor)
        self.opacity_entropy_start_iter = int(self.opacity_entropy_start_iter * factor)
        self.opacity_entropy_warmup_steps = int(self.opacity_entropy_warmup_steps * factor)
        if self.opacity_entropy_stop_iter >= 0:
            self.opacity_entropy_stop_iter = int(self.opacity_entropy_stop_iter * factor)
        self.pgsr_start_iter = int(self.pgsr_start_iter * factor)
        self.pgsr_warmup_steps = int(self.pgsr_warmup_steps * factor)


@dataclass
class DynamicConfig(Config):
    strategy: PlaneMCMCStrategy = field(default_factory=_dynamic_strategy)


CONFIGS = {
    "original": ("Original fixed-anchor voxel-plane MCMC.", Config()),
    "dynamic": ("Dynamic voxel ownership with free holes.", DynamicConfig()),
}


class Runner(base.Runner):
    def __init__(self, local_rank, world_rank, world_size, cfg):
        if (
            not math.isfinite(cfg.opacity_entropy_lambda)
            or cfg.opacity_entropy_lambda < 0
            or cfg.opacity_entropy_start_iter < 0
            or cfg.opacity_entropy_warmup_steps < 0
            or cfg.opacity_entropy_stop_iter < -1
            or (cfg.opacity_entropy_stop_iter >= 0
                and cfg.opacity_entropy_stop_iter <= cfg.opacity_entropy_start_iter)
        ):
            raise ValueError("Opacity entropy requires non-negative weight/start/warmup and stop=-1 or stop>start")
        if world_size != 1:
            raise ValueError("voxel-plane MCMC currently supports one GPU")
        if cfg.normalize_world_space:
            raise ValueError("voxel anchors require normalize_world_space=False")
        if cfg.packed:
            raise ValueError("texture statistics require packed=False")
        if cfg.sparse_grad:
            raise ValueError("anchor normal loss requires sparse_grad=False")
        if cfg.visible_adam:
            raise ValueError("plane losses require visible_adam=False")
        if min(
            cfg.anchor_normal_lambda,
            cfg.knn_plane_lambda,
            cfg.knn_normal_lambda,
            cfg.plane_scale_lambda,
            cfg.anchor_scale_lambda,
        ) < 0:
            raise ValueError("plane loss weights must be non-negative")
        if min(cfg.plane_s3_max, cfg.anchor_scale_max_ratio) <= 0:
            raise ValueError("scale limits must be positive")
        if cfg.pgsr_loss and (
            cfg.pgsr_sv_lambda < 0
            or cfg.pgsr_mv_lambda < 0
            or cfg.pgsr_start_iter < 0
            or cfg.pgsr_warmup_steps < 0
            or cfg.pgsr_every < 1
            or cfg.pgsr_samples < 1
            or cfg.batch_size != 1
        ):
            raise ValueError("PGSR requires valid parameters and batch_size=1")
        if cfg.pgsr_loss and cfg.pgsr_sv_lambda + cfg.pgsr_mv_lambda == 0:
            raise ValueError("PGSR is enabled but both loss weights are zero")
        if cfg.pgsr_loss and (
            cfg.data_type != "colmap" or cfg.camera_model != "pinhole" or cfg.pose_noise
        ):
            raise ValueError("PGSR requires COLMAP pinhole cameras without pose_noise")
        if cfg.pgsr_loss and (cfg.with_ut or cfg.with_eval3d):
            raise ValueError(
                "PGSR plane rendering requires with_ut=False and with_eval3d=False"
            )
        if cfg.pgsr_vis and (
            cfg.pgsr_vis_count < 1
            or cfg.camera_model != "pinhole"
            or cfg.with_ut
            or cfg.with_eval3d
        ):
            raise ValueError(
                "PGSR visualization requires pinhole cameras, a positive "
                "pgsr_vis_count, and with_ut=False / with_eval3d=False"
            )
        cfg.strategy.plane_s3_max = cfg.plane_s3_max
        cfg.strategy.plane_debug_dir = (
            str(Path(cfg.result_dir) / "plane_gs_after_refine")
            if cfg.save_plane_gs_after_refine
            else None
        )
        anchor_path = Path(cfg.strategy.plane_anchor_ply or "").expanduser()
        if not anchor_path.is_file():
            raise FileNotFoundError(
                f"pass a valid --strategy.plane-anchor-ply: {anchor_path}"
            )
        cfg.strategy.plane_anchor_ply = str(anchor_path.resolve())
        super().__init__(local_rank, world_rank, world_size, cfg)
        self._ncc = None
        if cfg.pgsr_loss and cfg.pgsr_mv_lambda > 0:
            from ncc_loss import PatchNCCLoss

            self._ncc = PatchNCCLoss(
                self.parser,
                self.trainset.indices,
                cfg.data_dir,
                cfg.pgsr_samples,
            )
        self._pgsr = None
        if cfg.pgsr_loss or cfg.pgsr_vis:
            import pgsr_geom

            self._pgsr = pgsr_geom

    def extra_training_loss(self, info, **context):
        pgsr_loss = self._pgsr_extra_loss(**context)
        entropy_loss = self._opacity_entropy_loss(info, context["step"])
        if entropy_loss is None:
            return pgsr_loss
        return entropy_loss if pgsr_loss is None else pgsr_loss + entropy_loss

    def _opacity_entropy_loss(self, info, step):
        cfg = self.cfg
        if (
            cfg.opacity_entropy_lambda == 0
            or step < cfg.opacity_entropy_start_iter
            or (cfg.opacity_entropy_stop_iter >= 0 and step >= cfg.opacity_entropy_stop_iter)
        ):
            return None
        progress = (min(1.0, (step - cfg.opacity_entropy_start_iter) / cfg.opacity_entropy_warmup_steps)
                    if cfg.opacity_entropy_warmup_steps else 1.0)
        weight = cfg.opacity_entropy_lambda * progress
        radii = info["radii"]
        # Same unpacked visibility rule as VoxelPlaneMCMCStrategy; combine views.
        visible = (radii > 0).all(-1).any(0) if radii.ndim == 3 else (radii > 0).any(0)
        logits = self.splats["opacities"][visible]
        raw_loss = opacity_entropy_loss(logits)
        weighted_loss = weight * raw_loss
        if self.world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
            with torch.no_grad():
                alpha = logits.sigmoid()
                low = (alpha < 0.01).float().mean().item() if alpha.numel() else 0.0
                high = (alpha > 0.99).float().mean().item() if alpha.numel() else 0.0
            values = {
                "opacity_entropy": raw_loss.item(),
                "opacity_entropy_weighted": weighted_loss.item(),
                "opacity_entropy_weight": weight,
                "opacity_entropy_visible_count": logits.numel(),
                "opacity_visible_below_001": low,
                "opacity_visible_above_099": high,
            }
            for name, value in values.items():
                self.writer.add_scalar(f"train/{name}", value, step)
            print(f"Opacity step={step} | entropy={values['opacity_entropy']:.6f} | "
                  f"weighted={values['opacity_entropy_weighted']:.6f} | weight={weight:.5f} | "
                  f"visible={logits.numel()} | alpha<0.01={low:.1%} | alpha>0.99={high:.1%}")
        return weighted_loss

    # PGSR block: renders the
    # per-pixel plane parameters itself, so it does not need the main pass's
    # depth channel; the multi-view term reuses the NCC module with rendered
    # planes (gradients flow through both plane maps back to the Gaussians).
    def _pgsr_extra_loss(self, **context):
        step = context["step"]
        cfg = self.cfg
        if (
            not cfg.pgsr_loss
            or self._pgsr is None
            or step < cfg.pgsr_start_iter
            or step % cfg.pgsr_every != 0
        ):
            return None
        if cfg.pgsr_warmup_steps == 0:
            progress = 1.0
        else:
            progress = min(
                1.0,
                max(0.0, (step - cfg.pgsr_start_iter) / cfg.pgsr_warmup_steps),
            )
        if progress == 0:
            return None

        pixels = context["pixels"]
        height, width = pixels.shape[1:3]
        render = self._pgsr.render_plane_params(
            self.splats["means"],
            self.splats["quats"],
            self.splats["scales"],
            self.splats["opacities"],
            context["camtoworlds"],
            context["Ks"],
            width,
            height,
            rasterize_mode="antialiased" if cfg.antialiased else "classic",
            camera_model=cfg.camera_model,
        )
        masks = context["masks"]
        reference_mask = (
            masks[0]
            if masks is not None
            else torch.ones_like(render.alpha[0], dtype=torch.bool)
        )
        terms = []
        sv_value = float("nan")
        if cfg.pgsr_sv_lambda > 0:
            sv_loss = self._pgsr.single_view_normal_loss(
                render.depth[0],
                render.normal_map[0],
                render.alpha[0].detach(),
                self._pgsr.rgb_to_gray(pixels[0]),
                reference_mask,
                context["Ks"][0],
                render.rays[0],
            )
            terms.append(cfg.pgsr_sv_lambda * progress * sv_loss)
            sv_value = sv_loss.item()
        mv_value = float("nan")
        mv_patches = 0
        mv_source = -1
        if cfg.pgsr_mv_lambda > 0:
            mv_loss, statistics = self._ncc(
                reference_item=int(context["data"]["image_id"].item()),
                reference_image=pixels[0],
                reference_depth=render.depth[0],
                reference_alpha=render.alpha[0],
                reference_mask=reference_mask,
                K_reference=context["Ks"][0],
                c2w_reference=context["camtoworlds"][0],
                pose_adjust=self.pose_adjust if cfg.pose_opt else None,
                plane_normals=render.normal_map[0],
                plane_distances=render.distance_map[0],
            )
            terms.append(cfg.pgsr_mv_lambda * progress * mv_loss)
            mv_value = mv_loss.item()
            mv_patches = statistics["patches"]
            mv_source = statistics["source"]
        if not terms:
            return None
        if (
            self.world_rank == 0
            and cfg.tb_every > 0
            and step % cfg.tb_every == 0
        ):
            print(
                f"PGSR step={step} | sv_loss={sv_value:.4f} | "
                f"mv_loss={mv_value:.4f} | patches={mv_patches} | "
                f"source={mv_source} | weight_scale={progress:.4f}"
            )
        return sum(terms)

    # PGSR visualization block: on each val eval, render the plane parameters
    # for the first pgsr_vis_count validation views and save colorized
    # unbiased-depth / normal / alpha maps into result_dir/pgsr_vis.
    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        super().eval(step, stage)
        if (
            self._pgsr is not None
            and self.cfg.pgsr_vis
            and self.world_rank == 0
            and stage == "val"
        ):
            self._save_pgsr_visualizations(step)

    @torch.no_grad()
    def _save_pgsr_visualizations(self, step: int) -> None:
        cfg = self.cfg
        vis_dir = Path(cfg.result_dir) / "pgsr_vis" / f"step_{step:06d}"
        vis_dir.mkdir(parents=True, exist_ok=True)
        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        saved = 0
        for i, data in enumerate(valloader):
            if saved >= cfg.pgsr_vis_count:
                break
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            height, width = data["image"].shape[1:3]
            render = self._pgsr.render_plane_params(
                self.splats["means"],
                self.splats["quats"],
                self.splats["scales"],
                self.splats["opacities"],
                camtoworlds,
                Ks,
                width,
                height,
                rasterize_mode="antialiased" if cfg.antialiased else "classic",
                camera_model=cfg.camera_model,
            )
            depth, alpha = render.depth[0], render.alpha[0]
            normals = F.normalize(render.normal_map[0], dim=-1)
            valid = (alpha > 0.5) & torch.isfinite(depth) & (depth > 0)
            if "mask" in data:
                valid &= data["mask"].to(self.device)[0]

            depth_vis = torch.zeros_like(depth)
            if valid.any():
                # Percentile stretch on a subsample (torch.quantile rejects
                # tensors above 2**24 elements).
                flat = depth[valid].flatten()
                stride = max(1, flat.numel() // 1_000_000)
                sample = flat[::stride]
                lo = sample.quantile(0.02)
                hi = sample.quantile(0.98)
                depth_vis[valid] = (depth[valid] - lo) / (hi - lo).clamp_min(1e-6)
            depth_rgb = apply_float_colormap(depth_vis[..., None]).clamp(0, 1)
            depth_rgb[~valid] = 0
            normal_rgb = (normals * 0.5 + 0.5).clamp(0, 1)
            normal_rgb[~valid] = 0
            alpha_rgb = alpha[..., None].expand(-1, -1, 3).clamp(0, 1)

            for suffix, image in (
                ("depth", depth_rgb),
                ("normal", normal_rgb),
                ("alpha", alpha_rgb),
            ):
                path = vis_dir / f"val_{i:04d}_{suffix}.png"
                imageio.imwrite(
                    path, (image.cpu().numpy() * 255).astype(np.uint8)
                )
            saved += 1
        if saved:
            print(f"Saved {saved} PGSR validation visualizations to {vis_dir}")

    def bind_strategy_state(self) -> None:
        super().bind_strategy_state()
        self._geometry_loss_values = {}
        self.cfg.strategy.initialize_plane_gaussians(
            self.splats, self.strategy_state
        )
        self.optimizers["quats"].register_step_pre_hook(
            self._add_anchor_normal_gradient
        )
        self.optimizers["means"].register_step_pre_hook(
            self._add_knn_plane_gradients
        )
        self.optimizers["scales"].register_step_pre_hook(
            self._add_anchor_scale_gradient
        )

    def _add_anchor_scale_gradient(self, optimizer, args, kwargs):
        del optimizer, args, kwargs
        self._geometry_loss_values["plane_scale"] = float("nan")
        self._geometry_loss_values["anchor_scale"] = float("nan")
        self._geometry_loss_values["large_scale_gs"] = -1
        anchor_ids = self.strategy_state["anchor_ids"]
        scales = self.splats["scales"]
        if scales.grad is None:
            return
        if scales.numel() == 0:
            self._geometry_loss_values["plane_scale"] = 0.0
            self._geometry_loss_values["anchor_scale"] = 0.0
            self._geometry_loss_values["large_scale_gs"] = 0
            return

        max_scale = (
            self.cfg.anchor_scale_max_ratio
            * self.cfg.strategy.anchor_voxel_size
        )
        # The voxel size is a global size reference, including GSs without anchors.
        anchor_excess = F.relu(scales.detach() - math.log(max_scale))
        oversized_axes = anchor_excess.gt(0)
        large_mask = oversized_axes.any(dim=-1)
        large_count = int(large_mask.sum().item())
        self._geometry_loss_values["large_scale_gs"] = large_count
        if large_count:
            large_excess = anchor_excess[oversized_axes]
            self._geometry_loss_values["anchor_scale"] = (
                large_excess.square().mean().item()
            )
            scales.grad[oversized_axes] += (
                2.0
                * self.cfg.anchor_scale_lambda
                * large_excess
                / large_excess.numel()
            )
        else:
            self._geometry_loss_values["anchor_scale"] = 0.0

        # All GSs retain the per-axis size limit above. Only planar
        # anchors define an oriented third axis that should be kept thin.
        plane_ids = torch.where(self.strategy_state["anchor_is_plane"][anchor_ids])[0]
        self._geometry_loss_values["plane_scale"] = 0.0
        if self.cfg.plane_scale_lambda > 0 and plane_ids.numel():
            plane_excess = F.relu(
                scales.detach()[plane_ids, 2] - math.log(self.cfg.plane_s3_max)
            )
            self._geometry_loss_values["plane_scale"] = (
                plane_excess.square().mean().item()
            )
            scales.grad[plane_ids, 2] += (
                2.0
                * self.cfg.plane_scale_lambda
                * plane_excess
                / plane_ids.numel()
            )

    def _add_anchor_normal_gradient(self, optimizer, args, kwargs):
        del optimizer, args, kwargs
        weight = self.cfg.anchor_normal_lambda
        ids = self.strategy_state["plane_gs_ids"]
        quats = self.splats["quats"]
        if weight == 0 or ids is None or ids.numel() == 0 or quats.grad is None:
            return

        anchor_ids = self.strategy_state["anchor_ids"][ids]
        target = self.strategy_state["anchor_normals"][anchor_ids]
        with torch.enable_grad():
            local_quats = quats.detach()[ids].requires_grad_(True)
            rotation = normalized_quat_to_rotmat(F.normalize(local_quats, dim=-1))
            normal = rotation[:, :, 2]
            dot = (normal * target).sum(-1, keepdim=True)
            target = torch.where(dot.detach() < 0, -target, target)
            normal_loss = (normal - target).abs().mean()
            gradient = torch.autograd.grad(weight * normal_loss, local_quats)[0]
        quats.grad.index_add_(0, ids, gradient)

        step = self.strategy_state["plane_loss_step"]
        if (
            self.world_rank == 0
            and self.cfg.tb_every > 0
            and step % self.cfg.tb_every == 0
        ):
            values = self._geometry_loss_values
            print(
                f"Step {step}: anchor_scale_loss={values.get('anchor_scale', float('nan')):.4f} | "
                f"large_scale_gs={values.get('large_scale_gs', -1)} | "
                f"plane_scale_loss={values.get('plane_scale', float('nan')):.4f} | "
                f"anchor_normal_loss={normal_loss.item():.4f} | "
                f"knn_plane_loss={values.get('knn_plane', float('nan')):.4f} | "
                f"knn_normal_loss={values.get('knn_normal', float('nan')):.4f}"
            )

    def _add_knn_plane_gradients(self, optimizer, args, kwargs):
        del optimizer, args, kwargs
        self._geometry_loss_values["knn_plane"] = float("nan")
        self._geometry_loss_values["knn_normal"] = float("nan")
        position_weight = self.cfg.knn_plane_lambda
        normal_weight = self.cfg.knn_normal_lambda
        ids = self.strategy_state["visible_gs_ids"]
        means, quats = self.splats["means"], self.splats["quats"]
        scales = self.splats["scales"].detach()
        if (
            position_weight + normal_weight == 0
            or ids is None
            or ids.numel() == 0
            or means.grad is None
            or quats.grad is None
        ):
            return

        # All visible centers participate, regardless of anchor membership.
        neighbor_ids = self.strategy_state["gs_knn_ids"][ids].long()
        with torch.no_grad():
            center_xyz = means[ids]
            neighbor_xyz = means[neighbor_ids]
            neighbor_rotation = normalized_quat_to_rotmat(
                F.normalize(quats[neighbor_ids], dim=-1)
            )
            neighbor_axis = scales[neighbor_ids].argmin(-1)
            neighbor_normal = neighbor_rotation.gather(
                3, neighbor_axis[..., None, None].expand(-1, -1, 3, 1)
            ).squeeze(3)
            near = (
                (neighbor_xyz - center_xyz[:, None]).square().sum(-1)
                <= (2.0 * self.cfg.strategy.anchor_voxel_size) ** 2
            ) & (neighbor_ids != ids[:, None])

            # Let all eight neighbors vote for the dominant direction at once.
            # The same six-neighbor consensus drives orientation and position.
            agreement = torch.bmm(
                neighbor_normal, neighbor_normal.transpose(1, 2)
            ).abs().gt(math.cos(math.radians(10.0)))
            agreement &= near[:, :, None] & near[:, None, :]
            votes = agreement.sum(-1)
            best = votes.argmax(-1)
            rows = torch.arange(len(best), device=best.device)
            agree = agreement[rows, best]
            valid = votes[rows, best] >= 6
            if not valid.any():
                self._geometry_loss_values["knn_plane"] = 0.0
                self._geometry_loss_values["knn_normal"] = 0.0
                return

            reference = neighbor_normal[rows, best, None]
            reference_dot = (neighbor_normal * reference).sum(-1)
            aligned = torch.where(
                (reference_dot < 0)[..., None],
                -neighbor_normal,
                neighbor_normal,
            )
            target = F.normalize(
                (aligned * agree[..., None]).sum(1), dim=-1
            )
            selected_f = agree.to(center_xyz.dtype)
            count = selected_f.sum(-1, keepdim=True).clamp_min(1.0)
            neighbor_mean = (
                neighbor_xyz * selected_f[..., None]
            ).sum(1) / count

            # Both losses require a local plane, not merely parallel normals.
            heights = (neighbor_xyz * target[:, None]).sum(-1)
            thickness = heights.masked_fill(~agree, -torch.inf).amax(-1) - (
                heights.masked_fill(~agree, torch.inf).amin(-1)
            )
            valid &= thickness <= self.cfg.strategy.plane_normal_tolerance
            if not valid.any():
                self._geometry_loss_values["knn_plane"] = 0.0
                self._geometry_loss_values["knn_normal"] = 0.0
                return

            ids = ids[valid]
            target = target[valid]
            neighbor_mean = neighbor_mean[valid]

        # The consensus is a fixed local target. Only visible center GSs receive
        # gradients; their neighbors will be optimized when they become visible.
        with torch.enable_grad():
            center_xyz = means.detach()[ids].requires_grad_(True)
            center_quats = quats.detach()[ids].requires_grad_(True)
            center_rotation = normalized_quat_to_rotmat(
                F.normalize(center_quats, dim=-1)
            )
            center_axis = scales[ids].argmin(-1)
            center_normal = center_rotation.gather(
                2, center_axis[:, None, None].expand(-1, 3, 1)
            ).squeeze(2)
            plane_error = (
                (center_xyz - neighbor_mean) * target
            ).sum(-1).abs()
            plane_loss = plane_error.mean()
            dot = (center_normal.detach() * target).sum(-1, keepdim=True)
            oriented_target = torch.where(dot < 0, -target, target)
            normal_loss = (center_normal - oriented_target).abs().mean()

            gradients = torch.autograd.grad(
                position_weight * plane_loss + normal_weight * normal_loss,
                (center_xyz, center_quats),
            )
        self._geometry_loss_values["knn_plane"] = plane_loss.item()
        self._geometry_loss_values["knn_normal"] = normal_loss.item()

        means.grad.index_add_(0, ids, gradients[0])
        quats.grad.index_add_(0, ids, gradients[1])



def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    original_runner = base.Runner
    base.Runner = Runner
    try:
        return base.main(local_rank, world_rank, world_size, cfg)
    finally:
        base.Runner = original_runner


if __name__ == "__main__":
    config = tyro.extras.overridable_config_cli(CONFIGS)
    if config.backend == "dgx":
        import torch_dgx  # noqa: F401
    config.adjust_steps(config.steps_scaler)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("set CUDA_VISIBLE_DEVICES to exactly one GPU")
    cli(main, config, verbose=True)
