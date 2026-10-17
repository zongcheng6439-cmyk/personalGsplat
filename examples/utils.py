# SPDX-FileCopyrightText: Copyright 2024-2025 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps
from plyfile import PlyData, PlyElement


def depth_to_normal(
    depth: Tensor,
    K: Tensor,
    valid: Optional[Tensor] = None,
    window_size: int = 10,
    find_planes: bool = False,
):
    """Convert Z-depth to camera-space normals and an optional planar mask."""
    single = depth.ndim == 2
    if single:
        depth, K = depth[None], K[None]
        valid = None if valid is None else valid[None]

    valid = torch.isfinite(depth) & (depth > 0) if valid is None else valid.bool()
    valid &= torch.isfinite(depth) & (depth > 0)
    depth = torch.nan_to_num(depth)
    batch, height, width = depth.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    z = depth
    x = (xs[None] - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None] * z
    y = (ys[None] - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None] * z
    points = torch.stack((x, y, z), dim=1)
    weights = valid[:, None].to(depth.dtype)

    left = (window_size - 1) // 2
    right = window_size - 1 - left

    def box_mean(value: Tensor) -> Tensor:
        value = F.pad(value, (left, right, left, right))
        return F.avg_pool2d(value, window_size, stride=1)

    coverage = box_mean(weights)
    mean_points = box_mean(points * weights) / coverage.clamp_min(1e-6)
    tangent_x = F.pad(
        mean_points[:, :, :, 2:] - mean_points[:, :, :, :-2], (1, 1, 0, 0)
    )
    tangent_y = F.pad(
        mean_points[:, :, 2:, :] - mean_points[:, :, :-2, :], (0, 0, 1, 1)
    )
    raw_normal = torch.cross(tangent_x, tangent_y, dim=1)
    normal_length = raw_normal.norm(dim=1)
    normal = F.normalize(raw_normal, dim=1, eps=1e-8)
    normal = torch.where(
        (normal * points).sum(dim=1, keepdim=True) > 0, -normal, normal
    )
    normal_valid = valid & (coverage[:, 0] >= 0.9) & (normal_length > 1e-8)

    if find_planes:
        px, py, pz = points.unbind(dim=1)
        moments = torch.stack(
            (px * px, py * py, pz * pz, px * py, px * pz, py * pz),
            dim=1,
        )
        moments = box_mean(moments * weights) / coverage.clamp_min(1e-6)
        mx, my, mz = mean_points.unbind(dim=1)
        exx, eyy, ezz, exy, exz, eyz = moments.unbind(dim=1)
        cxx = (exx - mx * mx).clamp_min(0)
        cyy = (eyy - my * my).clamp_min(0)
        czz = (ezz - mz * mz).clamp_min(0)
        cxy, cxz, cyz = exy - mx * my, exz - mx * mz, eyz - my * mz
        nx, ny, nz = normal.unbind(dim=1)
        plane_variance = (
            nx * nx * cxx
            + ny * ny * cyy
            + nz * nz * czz
            + 2 * (nx * ny * cxy + nx * nz * cxz + ny * nz * cyz)
        ).clamp_min(0)
        total_variance = (cxx + cyy + czz).clamp_min(1e-8)
        tolerance = torch.maximum(depth * 0.005, torch.full_like(depth, 0.05))
        normal_valid &= (
            plane_variance.sqrt() < tolerance
        ) & (plane_variance / total_variance < 0.02)

    normal = normal.movedim(1, -1)
    if single:
        return normal[0], normal_valid[0]
    return normal, normal_valid


def save_depth_normals(
    path: Union[str, Path], normal: Tensor, plane_mask: Tensor
) -> None:
    """Save int8 normals and a bit-packed planar mask in one NPZ file."""
    path = Path(path)
    normal = normal * plane_mask[..., None]
    normal_q = (normal.clamp(-1, 1) * 127).round().to(torch.int8).cpu().numpy()
    mask_q = np.packbits(plane_mask.cpu().numpy().reshape(-1))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, normal=normal_q, plane_mask=mask_q)


def load_depth_normals(path: Union[str, Path]):
    """Load normals and planar mask saved by ``save_depth_normals``."""
    with np.load(path) as cache:
        normal = torch.from_numpy(cache["normal"].astype(np.float32)) / 127.0
        height, width = normal.shape[:2]
        plane_mask = np.unpackbits(
            cache["plane_mask"], count=height * width
        ).reshape(height, width)
    return normal, torch.from_numpy(plane_mask.astype(bool))

class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


@torch.no_grad()
def gs_neighbor_distance(x: Tensor) -> Tensor:
    """RMS distance to three nearest GSs, using CUDA when available."""
    try:
        from simple_knn._C import distCUDA2

        if x.is_cuda:
            return distCUDA2(x.contiguous()).clamp_min(0).sqrt()
    except ImportError:
        pass

    from scipy.spatial import cKDTree

    x_np = x.detach().float().cpu().numpy()
    distances, _ = cKDTree(x_np).query(x_np, k=4, workers=-1)
    rms = np.sqrt(np.mean(np.square(distances[:, 1:]), axis=1))
    return torch.from_numpy(rms).to(device=x.device, dtype=x.dtype)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RepetitionSampler(torch.utils.data.Sampler[int]):
    """Visit every image an exact number of times in randomized epoch order."""

    def __init__(self, counts: Sequence[int], seed: int = 42):
        counts = torch.as_tensor(counts, dtype=torch.int64)
        if counts.ndim != 1 or counts.numel() == 0:
            raise ValueError("counts must be a non-empty one-dimensional sequence")
        if torch.any(counts <= 0):
            raise ValueError("all repetition counts must be positive")
        self.indices = torch.repeat_interleave(
            torch.arange(len(counts), dtype=torch.int64), counts
        )
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        order = torch.randperm(len(self.indices), generator=generator)
        return iter(self.indices[order].tolist())

    def __len__(self) -> int:
        return len(self.indices)


def make_repetition_sampler(counts: Sequence[int], seed: int = 42):
    return RepetitionSampler(counts, seed=seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img

def export_splats_sibr(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    format: Literal["ply", "splat", "ply_compressed"] = "ply",
    save_to: Optional[str] = None,
) -> bytes:
    """Export a Gaussian Splats model to bytes.
    The three supported formats are:
    - ply: A standard PLY file format. Supported by most viewers.
    - splat: A custom Splat file format. Supported by antimatter15 viewer.
    - ply_compressed: A compressed PLY file format. Used by Supersplat viewer.

    Args:
        means (torch.Tensor): Splat means. Shape (N, 3)
        scales (torch.Tensor): Splat scales. Shape (N, 3)
        quats (torch.Tensor): Splat quaternions. Shape (N, 4)
        opacities (torch.Tensor): Splat opacities. Shape (N,)
        sh0 (torch.Tensor): Spherical harmonics. Shape (N, 1, 3)
        shN (torch.Tensor): Spherical harmonics. Shape (N, K, 3)
        format (str): Export format. Options: "ply", "splat", "ply_compressed". Default: "ply"
        save_to (str): Output file path. If provided, the bytes will be written to file.
    """
    total_splats = means.shape[0]
    assert means.shape == (total_splats, 3), "Means must be of shape (N, 3)"
    assert scales.shape == (total_splats, 3), "Scales must be of shape (N, 3)"
    assert quats.shape == (total_splats, 4), "Quaternions must be of shape (N, 4)"
    assert opacities.shape == (total_splats,), "Opacities must be of shape (N,)"
    assert sh0.shape == (total_splats, 1, 3), "sh0 must be of shape (N, 1, 3)"
    assert (
        shN.ndim == 3 and shN.shape[0] == total_splats and shN.shape[2] == 3
    ), f"shN must be of shape (N, K, 3), got {shN.shape}"

    # Reshape spherical harmonics
    sh0 = sh0.squeeze(1)  # Shape (N, 3)
    shN = shN.permute(0, 2, 1).reshape(means.shape[0], -1)  # Shape (N, K * 3)

    # Check for NaN or Inf values
    invalid_mask = (
        torch.isnan(means).any(dim=1)
        | torch.isinf(means).any(dim=1)
        | torch.isnan(scales).any(dim=1)
        | torch.isinf(scales).any(dim=1)
        | torch.isnan(quats).any(dim=1)
        | torch.isinf(quats).any(dim=1)
        | torch.isnan(opacities)
        | torch.isinf(opacities)
        | torch.isnan(sh0).any(dim=1)
        | torch.isinf(sh0).any(dim=1)
        | torch.isnan(shN).any(dim=1)
        | torch.isinf(shN).any(dim=1)
    )

    # Filter out invalid entries
    valid_mask = ~invalid_mask
    means = means[valid_mask]
    scales = scales[valid_mask]
    quats = quats[valid_mask]
    opacities = opacities[valid_mask]
    sh0 = sh0[valid_mask]
    shN = shN[valid_mask]
    
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(sh0.shape[1]):
        l.append('f_dc_{}'.format(i))
    for i in range(shN.shape[1]):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scales.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(quats.shape[1]):
        l.append('rot_{}'.format(i))

    means = means.detach().cpu().numpy() # [size, 3]
    normals = np.zeros_like(means) # [size, 3]
    scales = scales.detach().cpu().numpy()   # [size, 3]
    quats = quats.detach().cpu().numpy()  # [size, 4]
    opacities = opacities.detach().cpu().numpy().reshape(-1, 1) # [size, 1]
    sh0 = sh0.detach().cpu().numpy() # [size, 3]
    shN = shN.detach().cpu().numpy() # [size, 45]
     
    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = np.concatenate((means, normals, sh0, shN, opacities, scales, quats), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(save_to)



def lidar_depth_loss(
    pred: Tensor,
    target: Tensor,
    beta: float,
    back_weight: float,
    front_focused: bool,
) -> Tensor:
    """LetsGo-style compressed depth loss in metric camera depth."""

    def compress(depth: Tensor) -> Tensor:
        return torch.where(
            depth < beta,
            depth / (2.0 * beta),
            1.0 - beta / (2.0 * depth.clamp_min(1e-6)),
        )

    error = compress(pred) - compress(target)
    if front_focused:
        # pred < target means a rendered surface is floating in front of LiDAR.
        return (F.relu(-error) + back_weight * F.relu(error)).mean()
    return error.abs().mean()

