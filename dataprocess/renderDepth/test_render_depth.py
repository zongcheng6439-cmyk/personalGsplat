"""用途：用合成点云测试投影、深度、遮挡、PLY 和完整 CLI；不读取实际数据。
输入：程序生成的小型平面；输出：断言结果，临时文件自动清理。
运行（本目录）：docker run --rm --gpus all -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD:/code" gsplat-letsgo-depth:local test_render_depth.py
"""
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
import torch

from render_depth import camera_settings, load_points, render, encode_depth_png


def main():
    sample = np.array([[0, 5, 10, 20, np.nan, np.inf, 10000]], dtype=np.float32)
    encoded = encode_depth_png(sample)
    assert encoded.dtype == np.uint8 and encoded.shape == (1, 7, 4)
    assert encoded[0, :, 0].tolist() == [0, 64, 128, 191, 0, 0, 0]
    assert encoded[0, :, 3].tolist() == [0, 255, 255, 255, 0, 0, 0]
    camera = SimpleNamespace(params=np.array([32., 32., 32., 32.]), width=64, height=64)
    image = SimpleNamespace(qvec=np.array([1., 0., 0., 0.]), tvec=np.zeros(3), qvec2rotmat=lambda: np.eye(3))
    settings = camera_settings(camera, image)
    xyz = torch.tensor([[0., 0., 5.]], device="cuda")
    scales = torch.full_like(xyz, .3)
    rotations = torch.tensor([[1., 0., 0., 0.]], device="cuda")
    depth = render(settings, xyz, scales, rotations, .01, True)
    assert np.allclose(depth[depth > 0], 5, atol=2e-5)
    assert depth[31, 31] > 0 and depth[0, 0] == 0
    raw = render(settings, xyz, scales, rotations, .01, False)
    assert 0 < raw[31, 31] < 5
    # 前景 5m 与背景 10m：中心应由前景主导，而不是平均为 7.5m。
    layered = render(settings, xyz.repeat(2, 1) * torch.tensor([[1.], [2.]], device="cuda"),
                     scales.repeat(2, 1), rotations.repeat(2, 1), .01, True)
    assert 5 <= layered[31, 31] < 5.5
    # 相机位置与朝向改变后，用同样 camera-space 点构造世界点，应得到同样图像。
    rotation = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    center = np.array([2., 3., 4.])
    moved = SimpleNamespace(qvec=np.array([2**-.5, 0., 2**-.5, 0.]),
                            tvec=-rotation @ center, qvec2rotmat=lambda: rotation)
    world = torch.tensor((np.array([[0., 0., 5.]]) @ rotation + center).astype(np.float32), device="cuda")
    assert np.allclose(render(camera_settings(camera, moved), world, scales, rotations, .01, True), depth, atol=1e-4)
    with tempfile.TemporaryDirectory(prefix="depth-test-", dir=Path(__file__).parent) as tmp:
        root = Path(tmp)
        model = root / "sparse/0"
        model.mkdir(parents=True)
        (root / "images/image0").mkdir(parents=True)
        # 渲染仅需相机参数，不读取图像像素。
        (root / "images/image0/test.png").touch()
        (model / "cameras.txt").write_text("# camera\n1 PINHOLE 64 64 32 32 32 32\n")
        (model / "images.txt").write_text("# image\n1 1 0 0 0 0 0 0 1 image0/test.png\n\n")
        x, y = np.meshgrid(np.linspace(-3, 3, 25), np.linspace(-3, 3, 25))
        vertices = np.zeros(x.size, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        vertices["x"], vertices["y"], vertices["z"] = x.ravel(), y.ravel(), 5
        for ascii_format in (True, False):
            PlyData([PlyElement.describe(vertices, "vertex")], text=ascii_format).write(str(model / "extracted.ply"))
            points, scale, _ = load_points(model / "extracted.ply", 1.)
            assert points.shape == (625, 3) and torch.isfinite(scale).all()
        command = [sys.executable, str(Path(__file__).parent / "render_depth.py"),
                   "--data-dir", str(root), "--normalize-alpha"]
        subprocess.run(command, check=True)
        with Image.open(root / "depths_render_color/image0/test.png") as png:
            assert png.mode == "RGBA"
            saved = np.asarray(png)
            la = np.asarray(png.convert("LA")) / 255.
        assert saved.dtype == np.uint8 and saved.shape == (64, 64, 4)
        gray, mask = la[..., 0], la[..., 1] > 0
        decoded = np.where(gray < .5, gray * 20, 5 / (1 - gray))
        assert np.allclose(decoded[mask], 5, atol=.04) and mask.mean() > .4
        assert (saved[~mask] == 0).all()
        refusal = subprocess.run(command, capture_output=True, text=True)
        assert refusal.returncode != 0 and "FileExistsError" in refusal.stderr
    print("PASS: camera transform, metric Z, alpha, occlusion, ASCII/binary PLY, CLI, overwrite guard")


if __name__ == "__main__":
    main()
