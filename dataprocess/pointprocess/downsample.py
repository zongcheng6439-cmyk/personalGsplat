#!/usr/bin/env python3
"""GTLR 几何/纹理加权点云下采样：输入彩色 PLY，输出原始点子集 PLY 和 JSON。

工作目录：/Users/zongcheng/code/3dgs/personalGsplat
执行（将 /path/to/cloud.ply 换成实际路径）：
python dataprocess/pointprocess/downsample.py --input_ply /path/to/cloud.ply \
    --output_ply outputs/init.ply --num_samples 400000
依赖：numpy scipy open3d；CPU 执行，不需要编译 gsplat，不改变世界坐标。
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def sampling_weights(xyz, rgb, knn=64, chunk_size=4096, workers=4):
    """由 main 调用；返回每点的几何/纹理混合权重，邻居按索引排除自身。"""
    n = len(xyz)
    if n < 3:
        return np.ones(n, dtype=np.float64)
    k = min(knn, n - 1)
    tree = cKDTree(xyz)
    curvature = np.empty(n)
    texture = np.empty(n)
    started = time.perf_counter()
    last_log = started
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        _, indices = tree.query(xyz[start:stop], k=k + 1, workers=workers)
        # 重合点可能排在自身前面；不能简单丢弃第一个邻居。
        is_self = indices == np.arange(start, stop)[:, None]
        order = np.argsort(is_self, axis=1, kind="stable")
        indices = np.take_along_axis(indices, order, axis=1)[:, :k]
        neighbors = xyz[indices]
        centered = neighbors - neighbors.mean(axis=1, keepdims=True)
        cov = np.einsum("bki,bkj->bij", centered, centered) / (k - 1)
        eigenvalues = np.maximum(np.linalg.eigvalsh(cov), 0)
        curvature[start:stop] = eigenvalues[:, 0] / (eigenvalues.sum(axis=1) + 1e-12)
        texture[start:stop] = rgb[indices].var(axis=1).mean(axis=1)
        now = time.perf_counter()
        if now - last_log >= 10 or stop == n:
            print(f"[score] {stop:,}/{n:,} points, {now - started:.1f}s", flush=True)
            last_log = now
    for values in (curvature, texture):
        values -= values.min()
        span = values.max()
        if span > 0:
            values /= span
    return 0.5 * curvature + 0.5 * texture


def sample_indices(weights, count, seed=42):
    """由 main 调用；无放回采样，退化权重使用均匀概率兜底。"""
    n = len(weights)
    if not 1 <= count <= n:
        raise ValueError(f"num_samples 必须在 1 到输入点数 {n} 之间")
    if count == n:
        return np.arange(n)
    p = np.asarray(weights, dtype=np.float64).copy()
    if p.sum() == 0:
        p.fill(1.0)
    p /= p.sum()
    if np.count_nonzero(p) < count:
        p = 0.99 * p + 0.01 / n
    return np.random.default_rng(seed).choice(n, count, replace=False, p=p)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_ply", "--input-ply", type=Path, required=True)
    parser.add_argument("--output_ply", "--output-ply", type=Path, required=True)
    parser.add_argument("--num_samples", "--num-samples", type=int, required=True)
    parser.add_argument("--knn", type=int, default=64)
    parser.add_argument("--chunk_size", "--chunk-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.knn < 2 or args.chunk_size < 1 or args.workers == 0 or args.workers < -1:
        parser.error("knn >= 2，chunk_size >= 1，workers 为正整数或 -1")
    if args.input_ply.resolve() == args.output_ply.resolve():
        parser.error("输入和输出路径不能相同")
    started = time.perf_counter()
    cloud = o3d.io.read_point_cloud(str(args.input_ply))
    xyz = np.asarray(cloud.points)
    rgb = np.asarray(cloud.colors)
    if not cloud.has_colors() or not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        parser.error("输入必须是包含有限 XYZ 和 RGB 的非空点云")
    if not 1 <= args.num_samples <= len(xyz):
        parser.error(f"num_samples 必须在 1 到 {len(xyz)} 之间；此脚本只做下采样")
    print(f"[load] {len(xyz):,} points, {time.perf_counter() - started:.2f}s", flush=True)
    score_start = time.perf_counter()
    weights = (sampling_weights(xyz, rgb, args.knn, args.chunk_size, args.workers)
               if args.num_samples < len(xyz) else np.ones(len(xyz)))
    print(f"[neighbors + score] {time.perf_counter() - score_start:.2f}s", flush=True)
    select_start = time.perf_counter()
    selected = sample_indices(weights, args.num_samples, args.seed)
    print(f"[select] {time.perf_counter() - select_start:.2f}s", flush=True)
    output = o3d.geometry.PointCloud()
    output.points = o3d.utility.Vector3dVector(xyz[selected])
    output.colors = o3d.utility.Vector3dVector(rgb[selected])
    args.output_ply.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(args.output_ply), output):
        raise OSError(f"无法保存 {args.output_ply}")
    metadata = dict(coordinate_frame="raw", source_ply=str(args.input_ply.resolve()),
                    input_points=len(xyz), num_samples=len(selected), knn=args.knn,
                    effective_knn=min(args.knn, len(xyz) - 1), seed=args.seed,
                    backend="scipy.cKDTree", reference="full_input", chunk_size=args.chunk_size,
                    workers=args.workers, method="geometry_texture_weighted_without_replacement")
    args.output_ply.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[saved] {args.output_ply}, {len(selected):,} points; total {time.perf_counter() - started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
