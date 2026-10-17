#!/usr/bin/env python3
"""Render every training view and independently validate per-GS texture scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from tqdm import tqdm

import gsplat  # noqa: E402
from datasets.colmap_adc import Dataset, Parser  # noqa: E402
from gsplat.exporter import load_ply_to_splats  # noqa: E402
from gsplat.losses import l1_loss, ssim_loss  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402
from render_utils import compute_top_gs_for_each_pixel  # noqa: E402


def save_white_ply(
    path: Path,
    means: np.ndarray,
    indices: np.ndarray,
    scores: np.ndarray,
    render_weights: np.ndarray,
) -> None:
    """Save selected GS centers as a white point cloud with diagnostic fields."""
    vertices = np.empty(
        len(indices),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("source_index", "i4"),
            ("score", "f4"),
            ("render_weight", "f4"),
        ],
    )
    vertices["x"] = means[indices, 0]
    vertices["y"] = means[indices, 1]
    vertices["z"] = means[indices, 2]
    vertices["red"] = 255
    vertices["green"] = 255
    vertices["blue"] = 255
    vertices["source_index"] = indices.astype(np.int32)
    vertices["score"] = scores[indices]
    vertices["render_weight"] = render_weights[indices]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render all dataset images and compute texture_score[g] = "
            "sum(weight_map * alpha*T) / sum(alpha*T)."
        )
    )
    parser.add_argument("--data-dir", required=True, help="COLMAP dataset root")
    parser.add_argument("--ply", required=True, help="trained 3DGS PLY")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-factor", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--texture-weight", type=float, default=0.5)
    parser.add_argument("--grad-weight", type=float, default=0.25)
    parser.add_argument("--loss-count-weight", type=float, default=0.25)
    parser.add_argument(
        "--opacity-weight",
        type=float,
        default=0.0,
        help="Current TextureMCMC does not add opacity to the combined score",
    )
    parser.add_argument("--texture-quantile", type=float, default=0.9)
    parser.add_argument("--grad-quantile", type=float, default=0.9)
    parser.add_argument("--ssim-lambda", type=float, default=0.2)
    parser.add_argument("--loss-threshold", type=float, default=0.1)
    parser.add_argument("--loss-min-count", type=int, default=5)
    parser.add_argument("--near-plane", type=float, default=0.01)
    parser.add_argument("--far-plane", type=float, default=1e10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="0 uses every image; a positive value is useful for a smoke test",
    )
    return parser.parse_args()


def quantile_normalize(values: torch.Tensor, quantile: float) -> torch.Tensor:
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
    positive = values[values > 0]
    if positive.numel() == 0:
        return torch.zeros_like(values)
    scale = torch.quantile(positive.float(), quantile).to(values)
    return (values / scale.clamp_min(1e-12)).clamp(0.0, 2.0)


def top_indices(scores: torch.Tensor, count: int) -> np.ndarray:
    return torch.topk(scores, k=count, sorted=True).indices.cpu().numpy()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("--top-fraction must be in (0, 1]")
    if not 0.0 < args.texture_quantile <= 1.0:
        raise ValueError("--texture-quantile must be in (0, 1]")
    if not 0.0 < args.grad_quantile <= 1.0:
        raise ValueError("--grad-quantile must be in (0, 1]")
    if min(
        args.texture_weight,
        args.grad_weight,
        args.opacity_weight,
        args.loss_count_weight,
    ) < 0:
        raise ValueError("score weights must be non-negative")
    if not 0.0 <= args.loss_threshold <= 1.0:
        raise ValueError("--loss-threshold must be in [0, 1]")
    if args.loss_min_count < 1:
        raise ValueError("--loss-min-count must be at least 1")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    splats_cpu = load_ply_to_splats(args.ply)
    means = splats_cpu["means"].to(device).requires_grad_(True)
    quats = splats_cpu["quats"].to(device)
    scales = splats_cpu["scales"].to(device).exp()
    opacities = splats_cpu["opacities"].to(device).sigmoid()
    colors = torch.cat(
        [splats_cpu["sh0"], splats_cpu["shN"]], dim=1
    ).to(device)
    sh_degree = int(math.sqrt(colors.shape[1]) - 1)
    num_gaussians = len(means)

    parser = Parser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        normalize=False,
        test_every=0,
        load_exposure=False,
    )
    dataset = Dataset(parser, split="train", load_weight_maps=True)
    num_images = len(dataset)
    if args.max_images > 0:
        num_images = min(num_images, args.max_images)
    if num_images == 0:
        raise ValueError("No images were selected")

    texture_sum = torch.zeros(num_gaussians, device=device)
    render_weight_sum = torch.zeros_like(texture_sum)
    absgrad_sum = torch.zeros_like(texture_sum)
    grad_count = torch.zeros_like(texture_sum)
    loss_count = torch.zeros_like(texture_sum)

    for image_idx in tqdm(range(num_images), desc="Rendering views"):
        data = dataset[image_idx]
        height, width = data["image"].shape[:2]
        camtoworld = data["camtoworld"].to(device)[None]
        K = data["K"].to(device)[None]

        renders, _, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv_ex(camtoworld).inverse,
            Ks=K,
            width=width,
            height=height,
            packed=False,
            absgrad=True,
            sparse_grad=False,
            rasterize_mode="classic",
            camera_model="pinhole",
            near_plane=args.near_plane,
            far_plane=args.far_plane,
            render_mode="RGB",
            sh_degree=sh_degree,
        )
        pixels = data["image"].to(device) / 255.0
        mask = data["mask"].to(device)
        pixel_l1 = (renders.detach() - pixels[None]).abs().mean(dim=-1)
        loss_min = pixel_l1.amin(dim=(-2, -1), keepdim=True)
        loss_max = pixel_l1.amax(dim=(-2, -1), keepdim=True)
        normalized_loss = (pixel_l1 - loss_min) / (
            loss_max - loss_min
        ).clamp_min(1e-8)
        high_loss_mask = normalized_loss > args.loss_threshold
        high_loss_mask &= mask[None]

        # Use the exact helper used by training for texture and FastGS counts.
        pixel_weight = data["weight_map"].to(device)
        pixel_weight *= mask.to(dtype=pixel_weight.dtype)
        statistics = compute_top_gs_for_each_pixel(
            info=info,
            pixel_importance=pixel_weight[None],
            num_gaussians=num_gaussians,
            top_k=args.top_k,
            high_loss_mask=high_loss_mask,
        )
        image_texture_sum, image_render_weight, image_loss_count = statistics
        texture_sum += image_texture_sum
        render_weight_sum += image_render_weight
        loss_count += image_loss_count

        l1 = l1_loss(renders[0][mask], pixels[mask]).mean()
        rendered_ssim = renders * mask[None, ..., None]
        pixels_ssim = pixels[None] * mask[None, ..., None]
        ssim = ssim_loss(
            rendered_ssim.permute(0, 3, 1, 2),
            pixels_ssim.permute(0, 3, 1, 2),
        )
        torch.lerp(l1, ssim, args.ssim_lambda).backward()

        grads = getattr(info["means2d"], "absgrad", None)
        if grads is None:
            raise RuntimeError("Rasterizer did not produce means2d.absgrad")
        visible = (info["radii"] > 0).all(dim=-1)
        ids = torch.where(visible)[1]
        absgrad_sum.index_add_(0, ids, grads[visible].norm(dim=-1))
        grad_count.index_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
        means.grad = None

    texture_score = texture_sum / render_weight_sum.clamp_min(1e-12)
    observed = render_weight_sum > 0
    texture_score[~observed] = 0
    grad_score = absgrad_sum / grad_count.clamp_min(1.0)
    opacity_score = opacities.detach()
    texture_normalized = quantile_normalize(
        texture_score, args.texture_quantile
    )
    grad_normalized = quantile_normalize(grad_score, args.grad_quantile)
    combined_score = (
        args.texture_weight * texture_normalized
        + args.grad_weight * grad_normalized
        + args.opacity_weight * opacity_score
    )
    filtered_loss_count = torch.where(
        loss_count >= args.loss_min_count,
        loss_count,
        torch.zeros_like(loss_count),
    )
    fastgs_score = quantile_normalize(filtered_loss_count, args.grad_quantile)
    combined_fastgs_score = (
        combined_score + args.loss_count_weight * fastgs_score
    )

    texture_score_np = texture_score.detach().cpu().numpy()
    grad_score_np = grad_score.detach().cpu().numpy()
    opacity_np = opacity_score.cpu().numpy()
    combined_np = combined_score.detach().cpu().numpy()
    loss_count_np = loss_count.cpu().numpy()
    fastgs_np = fastgs_score.cpu().numpy()
    combined_fastgs_np = combined_fastgs_score.detach().cpu().numpy()

    texture_sum_np = texture_sum.cpu().numpy()
    render_weight_np = render_weight_sum.cpu().numpy()
    absgrad_sum_np = absgrad_sum.cpu().numpy()
    grad_count_np = grad_count.cpu().numpy()

    means_np = splats_cpu["means"].numpy()

    top_count = max(1, math.ceil(num_gaussians * args.top_fraction))
    texture_top = top_indices(texture_score, top_count)
    grad_top = top_indices(grad_score, top_count)
    opacity_top = top_indices(opacity_score, top_count)
    combined_top = top_indices(combined_score, top_count)
    fastgs_top = top_indices(fastgs_score, top_count)
    combined_fastgs_top = top_indices(combined_fastgs_score, top_count)

    np.savez_compressed(
        output_dir / "texture_score_all_gs.npz",
        texture_score=texture_score_np,
        texture_sum=texture_sum_np,
        render_weight_sum=render_weight_np,
        texture_normalized=texture_normalized.detach().cpu().numpy(),
        grad_score=grad_score_np,
        absgrad_sum=absgrad_sum_np,
        grad_count=grad_count_np,
        grad_normalized=grad_normalized.detach().cpu().numpy(),
        opacity=opacity_np,
        combined_score=combined_np,
        loss_count=loss_count_np,
        fastgs_score=fastgs_np,
        combined_fastgs_score=combined_fastgs_np,
        texture_top_indices=texture_top,
        grad_top_indices=grad_top,
        opacity_top_indices=opacity_top,
        combined_top_indices=combined_top,
        fastgs_top_indices=fastgs_top,
        combined_fastgs_top_indices=combined_fastgs_top,
    )
    
    save_white_ply(
        output_dir / "texture_score_top10_white.ply",
        means_np,
        texture_top,
        texture_score_np,
        render_weight_np,
    )
    save_white_ply(
        output_dir / "grad_score_top10_white.ply",
        means_np,
        grad_top,
        grad_score_np,
        render_weight_np,
    )
    save_white_ply(
        output_dir / "opacity_top10_white.ply",
        means_np,
        opacity_top,
        opacity_np,
        render_weight_np,
    )
    save_white_ply(
        output_dir / "combined_score_top10_white.ply",
        means_np,
        combined_top,
        combined_np,
        render_weight_np,
    )
    save_white_ply(
        output_dir / "fastgs_score_top10_white.ply",
        means_np,
        fastgs_top,
        fastgs_np,
        render_weight_np,
    )
    save_white_ply(
        output_dir / "combined_fastgs_score_top10_white.ply",
        means_np,
        combined_fastgs_top,
        combined_fastgs_np,
        render_weight_np,
    )

    positive_texture = texture_score_np[render_weight_np > 0]
    positive_grad = grad_score_np[grad_count_np > 0]
    summary = {
        "texture_formula": "sum(weight_map * alpha*T) / sum(alpha*T)",
        "grad_formula": "sum(norm(means2d.absgrad)) / visible_count",
        "fastgs_formula": (
            "quantile_normalize(high-loss top-K occurrence count where "
            f"count >= {args.loss_min_count})"
        ),
        "combined_formula": (
            f"{args.texture_weight}*texture_normalized + "
            f"{args.grad_weight}*grad_normalized + "
            f"{args.opacity_weight}*sigmoid(opacity)"
        ),
        "combined_fastgs_formula": (
            f"combined_score + {args.loss_count_weight}*fastgs_score"
        ),
        "gsplat_import": str(Path(gsplat.__file__).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "ply": str(Path(args.ply).resolve()),
        "num_images": num_images,
        "num_gaussians": num_gaussians,
        "num_observed_gaussians": int(observed.sum().item()),
        "num_grad_observed_gaussians": int((grad_count > 0).sum().item()),
        "num_fastgs_gaussians": int((filtered_loss_count > 0).sum().item()),
        "top_k": args.top_k,
        "top_fraction": args.top_fraction,
        "top_count": top_count,
        "loss_threshold": args.loss_threshold,
        "loss_min_count": args.loss_min_count,
        "texture_score_min": float(positive_texture.min()) if positive_texture.size else 0.0,
        "texture_score_median": float(np.median(positive_texture)) if positive_texture.size else 0.0,
        "texture_score_max": float(positive_texture.max()) if positive_texture.size else 0.0,
        "grad_score_min": float(positive_grad.min()) if positive_grad.size else 0.0,
        "grad_score_median": float(np.median(positive_grad)) if positive_grad.size else 0.0,
        "grad_score_max": float(positive_grad.max()) if positive_grad.size else 0.0,
        "opacity_min": float(opacity_np.min()),
        "opacity_median": float(np.median(opacity_np)),
        "opacity_max": float(opacity_np.max()),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved: {output_dir / 'texture_score_top10_white.ply'}")
    print(f"Saved: {output_dir / 'grad_score_top10_white.ply'}")
    print(f"Saved: {output_dir / 'opacity_top10_white.ply'}")
    print(f"Saved: {output_dir / 'combined_score_top10_white.ply'}")
    print(f"Saved: {output_dir / 'fastgs_score_top10_white.ply'}")
    print(f"Saved: {output_dir / 'combined_fastgs_score_top10_white.ply'}")
    print(f"Saved: {output_dir / 'texture_score_all_gs.npz'}")


if __name__ == "__main__":
    main()
