"""PGSR plane-parameter rendering and loss tests.

Render tests need CUDA; the pure-tensor loss/NCC tests also run on CPU.
"""

import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from pgsr_geom import (
    plane_param_signals,
    pixel_rays,
    render_plane_params,
    rgb_to_gray,
    single_view_normal_loss,
    unbiased_depth,
)
from ncc_loss import PatchNCCLoss, _grayscale


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)


def test_plane_signals_face_camera_and_distance():
    # Identity rotation: shortest axis is z. Second Gaussian: z axis rotated
    # onto +x; placed at +x so the flipped normal must point to -x.
    means = torch.tensor([[0.0, 0.0, 5.0], [5.0, 0.0, 5.0]])
    quats = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [2**0.5 / 2, 0.0, 2**0.5 / 2, 0.0]]
    )
    scales_log = torch.log(torch.tensor([[0.2, 0.2, 0.02], [0.2, 0.2, 0.02]]))
    viewmats = torch.eye(4)[None]
    signals = plane_param_signals(means, quats, scales_log, viewmats)
    assert signals.shape == (1, 2, 4)
    torch.testing.assert_close(
        signals[0, 0], torch.tensor([0.0, 0.0, -1.0, -5.0]), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        signals[0, 1], torch.tensor([-1.0, 0.0, 0.0, -5.0]), atol=1e-5, rtol=1e-5
    )


def test_unbiased_depth_matches_ray_plane_intersection(device):
    K = torch.tensor([[[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]]],
                     device=device)
    rays = pixel_rays(K, 8, 8)
    normal_map = torch.tensor([0.0, 0.0, -1.0], device=device).expand(1, 8, 8, 3)
    distance_map = torch.full((1, 8, 8), -5.0, device=device)
    depth, ray_dot = unbiased_depth(normal_map, distance_map, rays)
    torch.testing.assert_close(
        depth, torch.full_like(depth, 5.0), atol=1e-5, rtol=1e-5
    )
    points = rays * depth[..., None]
    # z-depth property (rays have z = 1) and plane-equation consistency.
    torch.testing.assert_close(points[..., 2], depth, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        (points * normal_map).sum(-1), distance_map, atol=1e-4, rtol=1e-4
    )
    assert (ray_dot < 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA rasterizer")
def test_render_plane_params_flat_plane():
    device = torch.device("cuda")
    torch.manual_seed(0)
    # A 21x21 grid of flat, camera-facing Gaussians on the z = 5 plane.
    grid = torch.linspace(-3.2, 3.2, 21, device=device)
    gy, gx = torch.meshgrid(grid, grid, indexing="ij")
    means = torch.stack([gx.flatten(), gy.flatten()], -1)
    means = torch.cat([means, torch.full_like(means[:, :1], 5.0)], -1)
    means.requires_grad_()
    quats = torch.zeros(len(means), 4, device=device)
    quats[:, 0] = 1.0
    quats.requires_grad_()
    scales_log = torch.log(torch.tensor([0.2, 0.2, 0.02], device=device)).expand(
        len(means), 3
    )
    opacities_log = torch.full((len(means),), 4.6, device=device)  # sigmoid ~ 0.99
    camtoworlds = torch.eye(4, device=device)[None]
    Ks = torch.tensor(
        [[[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]]], device=device
    )
    render = render_plane_params(
        means, quats, scales_log, opacities_log, camtoworlds, Ks, 64, 64
    )
    covered = render.alpha[0] > 0.8
    assert covered.any()
    # Accumulation weights cancel in the division: even partially covered
    # pixels land exactly on the blended plane.
    torch.testing.assert_close(
        render.depth[0][covered],
        torch.full_like(render.depth[0][covered], 5.0),
        atol=2e-2,
        rtol=2e-3,
    )
    normals = render.normal_map[0][covered]
    normals = normals / normals.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(
        normals,
        torch.tensor([0.0, 0.0, -1.0], device=device).expand_as(normals),
        atol=2e-2,
        rtol=2e-2,
    )
    loss = render.depth[0][covered].sum() + render.distance_map[0][covered].sum()
    loss.backward()
    assert means.grad is not None and float(means.grad.abs().sum()) > 0
    assert quats.grad is not None and float(quats.grad.abs().sum()) > 0


def test_single_view_loss_zero_on_consistent_plane(device):
    H = W = 32
    K = torch.tensor([[50.0, 0.0, 16.0], [0.0, 50.0, 16.0], [0.0, 0.0, 1.0]],
                     device=device)
    rays = pixel_rays(K[None], H, W)[0]
    # Slightly tilted true plane; rendered normals match its real normal.
    y, x = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    depth = 5.0 + 1e-3 * (x + y)
    normal_map = torch.zeros(H, W, 3, device=device)
    normal_map[..., 2] = -1.0
    alpha = torch.ones(H, W, device=device)
    mask = torch.ones(H, W, device=device, dtype=torch.bool)
    gray = torch.zeros(H, W, device=device)
    loss = single_view_normal_loss(depth, normal_map, alpha, gray, mask, K, rays)
    # Finite-difference normals of a perspective back-projection carry a small
    # tilt of the same order as the plane slope plus ray divergence.
    assert float(loss) < 5e-2
    # Exact consistency: rendered normals equal the depth-derived ones.
    from gsplat.utils import depth_to_normal

    exact = depth_to_normal(
        depth[..., None][None], torch.eye(4, device=device)[None], K[None]
    )[0]
    flip = (exact * rays).sum(-1, keepdim=True) > 0
    exact = torch.where(flip, -exact, exact)
    loss_exact = single_view_normal_loss(
        depth, exact, alpha, gray, mask, K, rays
    )
    assert float(loss_exact) < 1e-6
    # Positive control: a 90-degree-wrong rendered normal must score high.
    wrong = torch.zeros(H, W, 3, device=device)
    wrong[..., 0] = 1.0
    bad = single_view_normal_loss(depth, wrong, alpha, gray, mask, K, rays)
    assert float(bad) > 0.5


def test_single_view_loss_gradients(device):
    H = W = 16
    K = torch.tensor([[50.0, 0.0, 8.0], [0.0, 50.0, 8.0], [0.0, 0.0, 1.0]],
                     device=device)
    rays = pixel_rays(K[None], H, W)[0]
    depth = torch.full((H, W), 5.0, device=device, requires_grad=True)
    normal_map = torch.full((H, W, 3), 0.5, device=device, requires_grad=True)
    alpha = torch.ones(H, W, device=device)
    mask = torch.ones(H, W, device=device, dtype=torch.bool)
    gray = torch.rand(H, W, device=device)
    loss = single_view_normal_loss(depth, normal_map, alpha, gray, mask, K, rays)
    loss.backward()
    assert depth.grad is not None and float(depth.grad.abs().sum()) > 0
    assert normal_map.grad is not None and float(normal_map.grad.abs().sum()) > 0


def _textured_reference(size, device):
    y, x = torch.meshgrid(
        torch.arange(size, device=device, dtype=torch.float32),
        torch.arange(size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    gray = 0.5 + 0.22 * torch.sin(x * 0.83) + 0.22 * torch.sin(y * 0.63 + x * 0.3)
    return gray[..., None].repeat(1, 1, 3)


def test_ncc_with_rendered_planes_flows_to_plane_maps(device):
    size = 96
    rgb = _textured_reference(size, device)
    gray = _grayscale(rgb)
    K = rgb.new_tensor([[100.0, 0.0, 48.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]])
    pose = torch.eye(4, device=device)
    src_pose = pose.clone()
    src_pose[0, 3] = 0.2
    source = dict(
        image=rgb * 255,
        gray=gray,
        depth_gs=torch.full_like(gray, 5.0),
        depth_gs_mask=torch.ones_like(gray, dtype=torch.bool),
        mask=torch.ones_like(gray, dtype=torch.bool),
        K=K,
        camtoworld=src_pose,
    )
    ncc = object.__new__(PatchNCCLoss)
    ncc.samples = 128
    ncc.source_items = {0: [1]}
    ncc._source = lambda _: source

    depth = torch.full_like(gray, 5.04, requires_grad=True)
    plane_normals = torch.zeros(size, size, 3, device=device)
    plane_normals[..., 2] = -1.0
    plane_normals.requires_grad_()
    # Signed plane distance: n . point = -5.04 for camera-facing normals.
    plane_distances = torch.full_like(gray, -5.04, requires_grad=True)
    random.seed(8)
    torch.manual_seed(8)
    loss, stats = ncc(
        0,
        rgb,
        depth,
        torch.ones_like(gray),
        source["mask"],
        K,
        pose,
        plane_normals=plane_normals,
        plane_distances=plane_distances,
    )
    assert stats["patches"] > 0
    assert torch.isfinite(loss)
    loss.backward()
    assert plane_normals.grad is not None
    assert float(plane_normals.grad.abs().sum()) > 0
    assert plane_distances.grad is not None
    assert float(plane_distances.grad.abs().sum()) > 0
    # Rendered-plane mode supervises the two plane maps, not the depth input:
    # depth only gates candidate selection (detached) by design.
    assert depth.grad is None or float(depth.grad.abs().sum()) == 0


def test_ncc_rejects_partial_plane_input(device):
    ncc = object.__new__(PatchNCCLoss)
    ncc.source_items = {}
    with pytest.raises(ValueError):
        ncc(
            0,
            torch.zeros(8, 8, 3, device=device),
            torch.ones(8, 8, device=device),
            torch.ones(8, 8, device=device),
            torch.ones(8, 8, device=device, dtype=torch.bool),
            torch.eye(3, device=device),
            torch.eye(4, device=device),
            plane_normals=torch.zeros(8, 8, 3, device=device),
        )
