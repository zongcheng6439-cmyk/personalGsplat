"""CPU tests for texture and FastGS statistic accumulation."""

import pytest
import torch

from examples import render_utils


def test_accumulate_texture_weights_matches_hand_computation():
    top_ids = torch.tensor(
        [[[[0, 1], [0, -1]], [[1, 0], [2, 1]]]], dtype=torch.int64
    )
    top_weights = torch.tensor(
        [[[[0.50, 0.25], [0.20, 0.00]], [[0.40, 0.10], [0.50, 0.25]]]]
    )
    pixel_importance = torch.tensor([[[2.0, 1.0], [3.0, 4.0]]])

    weights, pixel_count = render_utils.accumulate_texture_weights(
        top_ids, top_weights, pixel_importance, num_gaussians=3
    )

    torch.testing.assert_close(weights, torch.tensor([1.5, 2.7, 2.0]))
    torch.testing.assert_close(pixel_count, torch.tensor([3.0, 3.0, 1.0]))
    torch.testing.assert_close(
        weights.sum(), (top_weights * pixel_importance[..., None]).sum()
    )


def test_accumulate_fastgs_loss_counts_counts_multiview_hits():
    top_ids = torch.tensor(
        [
            [[[0, 1], [0, 2]]],
            [[[0, 2], [1, -1]]],
        ]
    )
    high_loss_mask = torch.tensor(
        [
            [[True, False]],
            [[True, True]],
        ]
    )

    loss_count = render_utils.accumulate_fastgs_loss_counts(
        top_ids, high_loss_mask, num_gaussians=3
    )

    torch.testing.assert_close(loss_count, torch.tensor([2.0, 2.0, 1.0]))


def test_compute_top_gs_requires_batched_pixel_map(monkeypatch):
    top_ids = torch.tensor([[[[0, 1], [1, 0]]]])
    top_weights = torch.ones_like(top_ids, dtype=torch.float32)
    monkeypatch.setattr(
        render_utils.gsplat,
        "rasterize_top_contributing_gaussian_ids",
        lambda **_kwargs: (top_ids, top_weights),
    )
    info = {
        "means2d": None,
        "conics": None,
        "opacities": None,
        "isect_offsets": None,
        "flatten_ids": None,
        "width": 2,
        "height": 1,
        "tile_size": 16,
    }

    with pytest.raises(ValueError, match="weight_map shape"):
        render_utils.compute_top_gs_for_each_pixel(
            info=info,
            pixel_importance=torch.ones(1, 2),
            num_gaussians=2,
        )
