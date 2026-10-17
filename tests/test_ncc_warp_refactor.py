"""Numerical equivalence tests. NCC_BEFORE_REFACTOR enables full old/new comparison."""

import importlib.util
import os
from pathlib import Path
import random
import sys
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from ncc_loss import (
    PatchNCCLoss, _grayscale, _patch_offsets, _patch_pixels, _project_plane_patch,
)


@pytest.fixture(params=["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def device(request):
    return torch.device(request.param)


def original_projection(homogeneous, normals, distance, Kr, Ks, transform):
    R, t = transform[:3, :3], transform[:3, 3]
    d = torch.where(distance.abs() < 1e-6,
                    distance.sign().masked_fill(distance == 0, 1.) * 1e-6, distance)
    H = Ks @ (R[None] + t[None, :, None] * normals[:, None] / d[:, None, None]) @ torch.linalg.inv(Kr)
    projected = torch.einsum("nij,npj->npi", H, homogeneous)
    xy = projected[..., :2] / projected[..., 2:].clamp_min(1e-6)
    rays = torch.einsum("ij,npj->npi", torch.linalg.inv(Kr), homogeneous)
    denominator = torch.einsum("ni,npi->np", normals, rays)
    denominator = torch.where(denominator.abs() < 1e-6,
        denominator.sign().masked_fill(denominator == 0, 1.) * 1e-6, denominator)
    points = rays * (d[:, None] / denominator)[..., None]
    z = (torch.einsum("ij,npj->npi", R, points) + t)[..., 2]
    return xy, z


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_projection_and_gradients(device, dtype):
    torch.manual_seed(10)
    Kr = torch.tensor([[400., 0., 160.], [0., 410., 120.], [0., 0., 1.]], device=device, dtype=dtype)
    Ks = Kr.clone(); Ks[0, 0] = 420.
    pixels = torch.rand(32, 25, 2, device=device, dtype=dtype) * 240
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
    normals = F.normalize(torch.randn(32, 3, device=device, dtype=dtype) * .15
                          + torch.tensor([0., 0., -1.], device=device, dtype=dtype), dim=-1)
    distance = torch.linspace(-2., -15., 32, device=device, dtype=dtype)
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[:3, :3] = torch.tensor([[.98, 0., .1989975], [0., 1., 0.], [-.1989975, 0., .98]], device=device, dtype=dtype)
    transform[:3, 3] = transform.new_tensor([.3, -.1, .2])
    results = []
    for fn in (original_projection, _project_plane_patch):
        d, n, pose = [v.detach().clone().requires_grad_() for v in (distance, normals, transform)]
        xy, z = fn(homogeneous, n, d, Kr, Ks, pose)
        grads = torch.autograd.grad(xy.square().mean() / 10000 + z.square().mean(), (d, n, pose))
        results.append((xy, z, *grads))
    tol = dict(atol=1e-4, rtol=3e-5) if dtype == torch.float32 else dict(atol=1e-10, rtol=1e-10)
    for old, new in zip(*results):
        torch.testing.assert_close(old, new, **tol)


def test_projection_preserves_signed_and_zero_denominator_clamps(device):
    K = torch.eye(3, device=device, dtype=torch.float64)
    pixels = torch.tensor([[[0., 0., 1.], [1e-8, 0., 1.]]], device=device, dtype=torch.float64).repeat(4, 1, 1)
    normals = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 0., -1.], [0., 0., 1.]], device=device, dtype=torch.float64)
    distances = torch.tensor([0., -1e-8, -5., 5.], device=device, dtype=torch.float64)
    transform = torch.eye(4, device=device, dtype=torch.float64)
    transform[:3, 3] = transform.new_tensor([.1, .2, -.3])
    old = original_projection(pixels, normals, distances, K, K, transform)
    new = _project_plane_patch(pixels, normals, distances, K, K, transform)
    for a, b in zip(old, new):
        torch.testing.assert_close(a, b, atol=1e-8, rtol=1e-10)


def test_gray_sampling_and_gradients(device):
    torch.manual_seed(4)
    image = torch.rand(32, 40, 3, device=device)
    initial_grid = torch.rand(1, 12, 25, 2, device=device) * 1.6 - .8
    results = []
    for cache_gray in (False, True):
        grid = initial_grid.clone().requires_grad_()
        if cache_gray:
            sampled = F.grid_sample(_grayscale(image)[None, None], grid, align_corners=True)[0, 0]
        else:
            rgb = F.grid_sample(image.permute(2, 0, 1)[None], grid, align_corners=True)[0].permute(1, 2, 0)
            sampled = _grayscale(rgb)
        gradient, = torch.autograd.grad(sampled.square().mean(), grid)
        results.append((sampled, gradient))
    for old, new in zip(*results):
        torch.testing.assert_close(old, new, atol=2e-6, rtol=3e-5)


def test_cached_offsets_and_source_gray(device):
    centers = torch.tensor([[16, 20], [30, 40]], device=device)
    offsets = _patch_offsets(5, device)
    assert _patch_offsets(5, device).data_ptr() == offsets.data_ptr()
    torch.testing.assert_close(_patch_pixels(centers, 5), centers[:, None] + offsets)
    ncc = object.__new__(PatchNCCLoss)
    from collections import OrderedDict
    ncc.cache = OrderedDict()
    image = torch.rand(8, 8, 3) * 255
    data = dict(image=image, mask=torch.ones(8, 8, dtype=torch.bool), depth_gs=torch.ones(8, 8),
                depth_gs_mask=torch.ones(8, 8, dtype=torch.bool), K=torch.eye(3), camtoworld=torch.eye(4))
    ncc.dataset = [data]
    first = ncc._source(0)
    torch.testing.assert_close(first["gray"], _grayscale(image.to(torch.uint8).float()/255))
    assert "image" not in first and ncc._source(0) is first


@pytest.mark.parametrize("tilted", [False, True])
def test_complete_loss_patch_count_and_depth_gradient(device, tilted):
    baseline = os.environ.get("NCC_BEFORE_REFACTOR")
    if not baseline:
        pytest.skip("set NCC_BEFORE_REFACTOR to the exact pre-refactor ncc_loss.py")
    spec = importlib.util.spec_from_file_location("ncc_before_refactor", baseline)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    size = 96
    y, x = torch.meshgrid(torch.arange(size, device=device), torch.arange(size, device=device), indexing="ij")
    gray = .5 + .22 * torch.sin(x*.83) + .22 * torch.sin(y*.63+x*.3)
    rgb = gray[..., None].repeat(1, 1, 3)
    K = rgb.new_tensor([[100.,0.,48.],[0.,100.,48.],[0.,0.,1.]])
    pose = torch.eye(4, device=device); src_pose = pose.clone(); src_pose[0, 3] = .2
    source = dict(image=rgb*255, gray=_grayscale(rgb), depth_gs=torch.full_like(gray, 5.),
                  depth_gs_mask=torch.ones_like(gray, dtype=torch.bool), mask=torch.ones_like(gray, dtype=torch.bool),
                  K=K, camtoworld=src_pose)
    initial_depth = 5.04 / (1. + (.1 if tilted else 0.) * (x - 48) / 100.)
    results = []
    for cls in (old.PatchNCCLoss, PatchNCCLoss):
        ncc = object.__new__(cls); ncc.samples = 128; ncc.source_items = {0:[1]}
        ncc._source = lambda _: source
        depth = initial_depth.detach().clone().requires_grad_()
        random.seed(8); torch.manual_seed(8)
        loss, stats = ncc(0, rgb, depth, torch.ones_like(gray), source["mask"], K, pose)
        gradient, = torch.autograd.grad(loss, depth)
        results.append((loss.detach(), stats, gradient))
    assert results[0][1] == results[1][1] and results[1][1]["patches"] > 0
    torch.testing.assert_close(results[0][0], results[1][0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(results[0][2], results[1][2], atol=2e-6, rtol=3e-4)
