import torch
from torch import Tensor
from typing import Dict, Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from gsplat.rendering import rasterization
from gsplat.cuda._wrapper import CameraModel

from sh_utils import *
from datasets.colmap_adc import Parser

def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)

def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def rasterize_splats(
    splats,
    camtoworlds: Tensor,
    Ks: Tensor,
    width: int,
    height: int,
    masks: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    
    means = splats["means"]  # [N, 3]
    quats = splats["quats"]  # [N, 4]
    scales = torch.exp(splats["scales"])  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N,]
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

    # if rasterize_mode is None:
    #     rasterize_mode = "antialiased" 
    rasterize_mode = "classic" 
    camera_model: CameraModel = "pinhole"


    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
        Ks=Ks,  # [C, 3, 3]
        width=width,
        height=height,  
        rasterize_mode=rasterize_mode,  
        camera_model=camera_model,
    )
    if masks is not None:
        render_colors[~masks] = 0

    return render_colors, render_alphas, info

def create_splats(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm" or init_type == "lidar":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm, random, or lidar")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points, requires_grad=False), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales, requires_grad=False), scales_lr),
        ("quats", torch.nn.Parameter(quats, requires_grad=False), quats_lr),
        ("opacities", torch.nn.Parameter(opacities, requires_grad=False), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients. 标准颜色方案  视角相关的颜色变化
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)

        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :], requires_grad=False), sh0_lr)) # [size, 1, 3]
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :], requires_grad=False), shN_lr)) # [size, 15, 3]
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features, requires_grad=False), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors, requires_grad=False), sh0_lr))

    # splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    splats = torch.nn.ParameterDict({n: v for n, v, _ in params})
    
    return splats

def SH2RGB(sh):
    return sh * C0 + 0.5

def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b, dim=-1)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3, dtype=a.dtype, device=a.device) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=a.dtype,
        device=a.device,
    )
    return torch.eye(3, dtype=a.dtype, device=a.device) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s ** 2 + 1e-8))
