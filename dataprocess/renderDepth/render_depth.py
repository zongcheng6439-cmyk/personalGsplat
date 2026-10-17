"""用途：固定 LiDAR 点云转 GS(opacity=0.999)，使用 LetsGo 官方光栅器渲染深度。
输入：cutpano images/、sparse/0/{cameras,images}.txt 和 extracted.ply（ASCII/二进制均可）。
输出：depths_render_color/imageN/时间戳.png，8 位 RGBA，与 section_0907 一致；
      RGB=round(255*R(Z))，R(Z)=Z/20(Z<10m)，否则 1-5/Z；Alpha=有效 mask。
      输入点云单位须为米；无效像素 RGBA=0；render_config.json 记录参数。
运行：bash run_depth.sh /实际数据目录 [--limit 3]；容器内 python render_depth.py --data-dir /data。
注意：官方固定版本实际采用中心 Z 的 alpha 加权和，不是论文逐像素协方差深度修正。
默认保留该合成公式；--normalize-alpha 才除以累计 alpha。低覆盖像素始终置 0。
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree
import torch
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# 复用官方 COLMAP 读取器，不引入训练 Scene、simple-knn、pycolmap。
sys.path.insert(0, str(Path(__file__).parent / "LetsGo" / "scene"))
from colmap_loader import read_intrinsics_text, read_extrinsics_text


def load_points(path, multiplier):
    vertices = PlyData.read(str(path))["vertex"].data
    xyz = np.column_stack([vertices[k] for k in ("x", "y", "z")]).astype(np.float32)
    if len(xyz) < 4 or not np.isfinite(xyz).all():
        raise ValueError("点云至少需要 4 个有限的 XYZ 点")
    # 与普通 GS 初始化一致：三个近邻距离平方均值开根号；不移动点云，不拟合额外平面。
    tree = cKDTree(xyz)
    scales = np.empty(len(xyz), np.float32)
    for start in tqdm(range(0, len(xyz), 100000), desc="初始化 GS scale"):
        distances = tree.query(xyz[start:start + 100000], k=4, workers=4)[0][:, 1:]
        scales[start:start + len(distances)] = np.sqrt(np.maximum((distances ** 2).mean(1), 1e-7))
    print(f"GS={len(xyz):,}; scale P50/P95/max={np.quantile(scales * multiplier, [.5, .95, 1])}", flush=True)
    xyz = torch.tensor(xyz, device="cuda")
    scale = torch.tensor(scales * multiplier, device="cuda")[:, None].repeat(1, 3)
    rotations = torch.zeros((len(xyz), 4), device="cuda")
    rotations[:, 0] = 1
    return xyz, scale, rotations


def camera_settings(camera, image):
    fx, fy, cx, cy = camera.params
    width, height = camera.width, camera.height
    if not np.isfinite(camera.params).all() or min(fx, fy, width, height) <= 0:
        raise ValueError("相机参数无效")
    # COLMAP 像素中心为 0.5；光栅器 ndc2Pix=(ndc+1)*size/2-0.5。
    # 官方协方差投影假设中心主点；cutpano 正好满足此条件，避免默默支持错误相机。
    if not np.allclose([cx, cy], [width / 2, height / 2], atol=1e-5):
        raise ValueError("仅支持 cutpano 的中心主点 PINHOLE 相机")
    if not np.isfinite(np.r_[image.qvec, image.tvec]).all() or not np.isclose(np.linalg.norm(image.qvec), 1):
        raise ValueError("相机位姿无效")
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = image.qvec2rotmat()
    world_to_camera[:3, 3] = image.tvec
    projection = np.zeros((4, 4), np.float32)
    near, far = .01, 10000.
    projection[0, 0], projection[1, 1] = 2 * fx / width, 2 * fy / height
    projection[2, 2], projection[2, 3] = far / (far - near), -far * near / (far - near)
    projection[3, 2] = 1
    view = torch.tensor(world_to_camera.T.copy(), device="cuda")
    proj = torch.tensor(projection.T.copy(), device="cuda")
    center = -world_to_camera[:3, :3].T @ world_to_camera[:3, 3]
    return GaussianRasterizationSettings(
        image_height=height, image_width=width, tanfovx=width / (2 * fx), tanfovy=height / (2 * fy),
        bg=torch.zeros(3, device="cuda"), scale_modifier=1., viewmatrix=view,
        projmatrix=view @ proj, sh_degree=0, campos=torch.tensor(center, device="cuda"),
        prefiltered=False, debug=False,
    )


@torch.no_grad()
def render(settings, xyz, scales, rotations, min_alpha, normalize):
    # 白色 + 黑背景的 RGB 输出就是累计 alpha，不需要修改 CUDA 或多渲染一次。
    rgb, depth, _ = GaussianRasterizer(settings)(
        means3D=xyz, means2D=torch.zeros_like(xyz),
        colors_precomp=torch.ones_like(xyz), opacities=torch.full((len(xyz), 1), .999, device="cuda"),
        scales=scales, rotations=rotations,
    )
    alpha = rgb[0]
    depth = depth.squeeze(0)
    if normalize:
        depth = depth / alpha.clamp_min(1e-8)
    valid = (alpha >= min_alpha) & torch.isfinite(depth) & (depth > 0)
    return torch.where(valid, depth, 0).cpu().numpy().astype(np.float32)


def encode_depth_png(depth):
    """由 main 调用：匹配 colmap_depth.py 的 warpped_depth_inv，避免 255 解码为无穷。"""
    valid = np.isfinite(depth) & (depth > 0)
    warped = np.zeros_like(depth, dtype=np.float32)
    near = valid & (depth < 10)
    far = valid & ~near
    warped[near] = depth[near] / 20
    warped[far] = 1 - 5 / depth[far]
    quantized = np.rint(warped * 255)
    # 不饱和截断远处深度；无法用有限非零码值表示时标为无效。
    valid &= (quantized >= 1) & (quantized <= 254)
    gray = np.where(valid, quantized, 0).astype(np.uint8)
    return np.dstack((gray, gray, gray, valid.astype(np.uint8) * 255))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--point-cloud", type=Path, help="默认 data-dir/sparse/0/extracted.ply")
    parser.add_argument("--output", type=Path, help="默认 data-dir/depths_render_color")
    parser.add_argument("--scale-multiplier", type=float, default=1.)
    parser.add_argument("--min-alpha", type=float, default=.95)
    parser.add_argument("--normalize-alpha", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="0=全部图像；正数=只渲染前 N 张")
    args = parser.parse_args()
    if not np.isfinite(args.scale_multiplier) or args.scale_multiplier <= 0 or not 0 < args.min_alpha <= 1 or args.limit < 0:
        parser.error("scale-multiplier 必须正数，min-alpha 在 (0,1]，limit 非负")
    model = args.data_dir / "sparse/0"
    cameras = read_intrinsics_text(str(model / "cameras.txt"))
    images = sorted(read_extrinsics_text(str(model / "images.txt")).values(), key=lambda im: im.id)
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise ValueError("没有相机记录")
    output = args.output or args.data_dir / "depths_render_color"
    names = [Path(im.name).with_suffix(".png") for im in images]
    if len(set(names)) != len(names):
        raise ValueError("输出文件名冲突")
    for im, name in zip(images, names):
        if name.is_absolute() or ".." in name.parts or not (args.data_dir / "images" / im.name).is_file():
            raise ValueError(f"图像路径无效或文件缺失：{im.name}")
        if (output / name).exists():
            raise FileExistsError(f"不覆盖已有深度：{output / name}；请用 --output 指定新目录")
        camera_settings(cameras[im.camera_id], im)
    cloud = args.point_cloud or model / "extracted.ply"
    xyz, scales, rotations = load_points(cloud, args.scale_multiplier)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(opacity=.999, min_alpha=args.min_alpha, scale_multiplier=args.scale_multiplier,
                  normalize_alpha=args.normalize_alpha, point_cloud=str(cloud), images=len(images),
                  depth_type="camera_z", encoding="RGBA8: RGB=round(255*R(Z)), R=Z/20 if Z<10 else 1-5/Z; alpha=valid",
                  invalid_value=0, rasterizer_commit="48c21361559d37f379fa140dc89ef40f51a48391")
    (output / "render_config.json").write_text(json.dumps(config, indent=2) + "\n")
    progress = tqdm(zip(images, names), total=len(images), desc="渲染深度")
    for im, name in progress:
        depth = render(camera_settings(cameras[im.camera_id], im), xyz, scales, rotations,
                       args.min_alpha, args.normalize_alpha)
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        rgba = encode_depth_png(depth)
        with target.open("xb") as stream:
            Image.fromarray(rgba).save(stream, format="PNG")
        progress.set_postfix(image=im.name, valid=f"{(rgba[..., 3] > 0).mean():.1%}")
    print(f"完成：{output}")


if __name__ == "__main__":
    main()
