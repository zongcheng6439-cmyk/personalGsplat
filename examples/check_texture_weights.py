"""Verify GS texture weights against a hand-computed example."""

import torch

from render_utils import accumulate_texture_weights


def main() -> None:
    top_ids = torch.tensor(
        [[[[0, 1], [0, -1]], [[1, 0], [2, 1]]]], dtype=torch.int64
    )
    render_weights = torch.tensor(
        [[[[0.50, 0.25], [0.20, 0.00]], [[0.40, 0.10], [0.50, 0.25]]]]
    )
    pixel_weights = torch.tensor([[[2.0, 1.0], [3.0, 4.0]]])

    actual, pixel_num = accumulate_texture_weights(
        top_ids,
        render_weights,
        pixel_weights,
        num_gaussians=3,
    )
    expected = torch.tensor([1.5, 2.7, 2.0])
    expected_pixel_num = torch.tensor([3.0, 3.0, 1.0])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(pixel_num, expected_pixel_num)
    torch.testing.assert_close(
        actual.sum(), (render_weights * pixel_weights[..., None]).sum()
    )

    print("Formula: GS_weight[g] = sum(render_weight[p,k] * pixel_weight[p])")
    print("Expected:", expected.tolist())
    print("Actual:  ", actual.tolist())
    print("Pixel counts:", pixel_num.tolist())


if __name__ == "__main__":
    main()
