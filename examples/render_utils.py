"""Helpers for projecting pixel texture importance back to Gaussians."""

import torch
import gsplat


@torch.no_grad()
def compute_texture_statistics_2dgs(
    info,
    pixel_importance: torch.Tensor,
    num_gaussians: int,
):
    """Average the weight-map values of pixels covered by each 2D Gaussian.

    This deliberately uses pure GS/pixel ownership counts.  It does not
    multiply the weight map by alpha or rendering-contribution weights.
    """
    if info["means2d"].ndim != 3:
        raise ValueError("2DGS texture statistics currently require packed=False")

    height, width = info["height"], info["width"]
    pixel_importance = pixel_importance.to(
        device=info["means2d"].device, dtype=info["means2d"].dtype
    )
    expected_shape = (info["n_cameras"], height, width)
    if tuple(pixel_importance.shape) != expected_shape:
        raise ValueError(
            f"weight_map shape {tuple(pixel_importance.shape)} does not match "
            f"render shape {expected_shape}"
        )

    transmittances = torch.ones(
        expected_shape,
        device=info["means2d"].device,
        dtype=info["means2d"].dtype,
    )
    gs_ids, pixel_ids, image_ids = gsplat.rasterize_to_indices_in_range_2dgs(
        range_start=0,
        range_end=2**31 - 1,
        transmittances=transmittances,
        means2d=info["means2d"],
        ray_transforms=info["ray_transforms"],
        opacities=info["opacities"],
        image_width=width,
        image_height=height,
        tile_size=info["tile_size"],
        isect_offsets=info["isect_offsets"],
        flatten_ids=info["flatten_ids"],
    )

    texture_sum = torch.zeros(
        num_gaussians, device=pixel_importance.device, dtype=pixel_importance.dtype
    )
    texture_count = torch.zeros_like(texture_sum)
    if gs_ids.numel() == 0:
        return texture_sum, texture_count

    values = pixel_importance.reshape(info["n_cameras"], -1)[
        image_ids.long(), pixel_ids.long()
    ]
    valid = (
        (gs_ids >= 0)
        & (gs_ids < num_gaussians)
        & torch.isfinite(values)
    )
    ids = gs_ids[valid].long()
    values = values[valid]
    texture_sum.index_add_(0, ids, values)
    texture_count.index_add_(0, ids, torch.ones_like(values))
    return texture_sum, texture_count


def accumulate_texture_weights(
    top_ids: torch.Tensor,
    top_weights: torch.Tensor,
    pixel_importance: torch.Tensor,
    num_gaussians: int,
):
    """Compute ``sum(render_weight * pixel_weight)`` for every Gaussian."""
    if top_ids.shape != top_weights.shape:
        raise ValueError("top_ids and top_weights must have identical shapes")
    if pixel_importance.shape != top_ids.shape[:-1]:
        raise ValueError(
            f"pixel map shape {tuple(pixel_importance.shape)} does not match "
            f"top-K shape {tuple(top_ids.shape)}"
        )

    ids = top_ids.reshape(-1).long()
    render_weight = top_weights.reshape(-1)
    products = (top_weights * pixel_importance[..., None]).reshape(-1)
    valid = (
        (ids >= 0)
        & (ids < num_gaussians)
        & torch.isfinite(products)
    )
    ids = ids[valid]
    products = products[valid]
    render_weight = render_weight[valid]
    
    gs_weights = torch.zeros(
        num_gaussians, device=products.device, dtype=products.dtype
    )
    pixel_num = torch.zeros_like(gs_weights)

    gs_weights.scatter_add_(0, ids, products)
    # pixel_num.scatter_add_(0, ids, torch.ones_like(products))
    pixel_num.scatter_add_(0, ids, render_weight)
    return gs_weights, pixel_num


def accumulate_fastgs_loss_counts(
    top_ids: torch.Tensor,
    high_loss_mask: torch.Tensor,
    num_gaussians: int,
):
    """Count high-error pixel/GS associations across the rendered views."""
    if high_loss_mask.shape != top_ids.shape[:-1]:
        raise ValueError(
            f"loss mask shape {tuple(high_loss_mask.shape)} does not match "
            f"top-K shape {tuple(top_ids.shape)}"
        )

    ids = top_ids[high_loss_mask.bool()].reshape(-1).long()
    valid = (ids >= 0) & (ids < num_gaussians)
    ids = ids[valid]
    return torch.bincount(ids, minlength=num_gaussians).to(torch.float32)


@torch.no_grad()
def compute_top_gs_for_each_pixel(
    info,
    pixel_importance: torch.Tensor,
    num_gaussians: int,
    top_k: int = 3,
    high_loss_mask: torch.Tensor | None = None,    
):
    """Accumulate per-pixel texture importance onto the top contributing GSs.

    ``texture_info[g]`` is the sum of ``pixel_grad * render_weight`` over
    every top-K occurrence of Gaussian ``g``. ``pixel_num[g]`` records the
    corresponding number of occurrences.
    """
    top_ids, top_weights = gsplat.rasterize_top_contributing_gaussian_ids(
        means2d=info["means2d"],
        conics=info["conics"],
        opacities=info["opacities"],
        tile_offsets=info["isect_offsets"],
        flatten_ids=info["flatten_ids"],
        image_width=info["width"],
        image_height=info["height"],
        tile_size=info["tile_size"],
        num_depth_samples=top_k,
    )
    pixel_importance = pixel_importance.to(
        device=top_weights.device, dtype=top_weights.dtype
    )
    if pixel_importance.shape != top_ids.shape[:-1]:
        raise ValueError(
            f"weight_map shape {tuple(pixel_importance.shape)} does not match "
            f"top-K shape {tuple(top_ids.shape)}"
        )

    if high_loss_mask is None:
        return accumulate_texture_weights(
            top_ids, top_weights, pixel_importance, num_gaussians
        )

    loss_mask = high_loss_mask.to(device=top_ids.device)
    loss_count = accumulate_fastgs_loss_counts(top_ids, loss_mask, num_gaussians)

    texture_info, texture_pixel_num = accumulate_texture_weights(
        top_ids, top_weights, pixel_importance, num_gaussians
    )
    return texture_info, texture_pixel_num, loss_count
