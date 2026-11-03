"""Fuse ScanNet++ iPhone RGB-D frames into a metric colored surface point cloud.

Run from /Users/zongcheng/code/3dgs/personalGsplat:
    python dataprocess/pointprocess/tsdf_fusion.py --scene /path/to/068ba2946c \
        --output outputs/068ba2946c/tsdf
Replace /path/to/068ba2946c with the actual scene directory.
Dependencies: pip install -r dataprocess/pointprocess/requirements-tsdf.txt
Inputs: scene_iphone_metadata.npz, images/, depth/ under --scene.
Outputs: point_cloud.ply and fusion_report.json under --output
(default: <scene>/tsdf_output).
Assumes OpenCV camera-to-world poses and millimeter camera-z depth.
No world normalization, camera-axis flips, or pose optimization is performed.
"""
import argparse
import json
import time
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.cache' / 'matplotlib'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scene', type=Path, required=True,
                    help='Scene directory containing metadata NPZ, images/ and depth/')
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--voxel-size', type=float, default=0.01)
    ap.add_argument('--sdf-trunc', type=float, default=0.04)
    ap.add_argument('--depth-scale', type=float, default=1000.0)
    ap.add_argument('--depth-max', type=float, default=10.0)
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()
    import numpy as np
    import open3d as o3d

    args.scene = args.scene.expanduser().resolve()
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
    if min(args.voxel_size, args.sdf_trunc, args.depth_scale, args.depth_max, args.stride) <= 0:
        ap.error('All numerical parameters must be positive')
    out = args.output or args.scene / 'tsdf_output'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'point_cloud.ply').exists():
        raise FileExistsError('Output already exists; select a new --output directory')
    meta = np.load(args.scene/'scene_iphone_metadata.npz', allow_pickle=False)
    names, poses, intrinsics = meta['images'], meta['trajectories'], meta['intrinsics']
    assert poses.shape == (len(names), 4, 4)
    assert intrinsics.shape == (len(names), 3, 3)
    assert np.isfinite(poses).all() and np.isfinite(intrinsics).all()
    assert np.allclose(poses[:,3,:], [0,0,0,1])
    rotations = poses[:,:3,:3]
    assert np.allclose(rotations.transpose(0,2,1) @ rotations, np.eye(3), atol=1e-6)
    assert np.allclose(np.linalg.det(rotations), 1, atol=1e-6)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel_size, sdf_trunc=args.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    start = time.monotonic()
    selected = list(range(0, len(names), args.stride))
    for count, i in enumerate(selected, 1):
        name = str(names[i])
        color = o3d.io.read_image(str(args.scene/'images'/name))
        depth = o3d.io.read_image(str(args.scene/'depth'/Path(name).with_suffix('.png')))
        da, ca = np.asarray(depth), np.asarray(color)
        assert da.ndim == 2 and da.dtype == np.uint16, name
        h, w = da.shape
        assert ca.shape == (h,w,3), name
        assert np.any((da > 0) & (da < args.depth_max*args.depth_scale)), name
        k = intrinsics[i]
        assert k[0,0] > 0 and k[1,1] > 0 and np.allclose(k[2], [0,0,1])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=args.depth_scale,
            depth_trunc=args.depth_max, convert_rgb_to_intensity=False)
        camera = o3d.camera.PinholeCameraIntrinsic(w,h,k[0,0],k[1,1],k[0,2],k[1,2])
        volume.integrate(rgbd, camera, np.linalg.inv(poses[i]))
        if count == 1 or count % 25 == 0 or count == len(selected):
            print(f'Integrated {count}/{len(selected)} frames; {time.monotonic()-start:.1f}s', flush=True)
    cloud = volume.extract_point_cloud()
    points = np.asarray(cloud.points)
    assert len(points) > 0 and np.isfinite(points).all()
    assert cloud.has_colors() and cloud.has_normals()
    target = out/'point_cloud.ply'
    if not o3d.io.write_point_cloud(str(target), cloud):
        raise OSError(f'Failed to write point cloud: {target}')
    check = o3d.io.read_point_cloud(str(target))
    assert len(check.points) == len(points) and check.has_colors() and check.has_normals()
    report = dict(library='Open3D', version=o3d.__version__, scene=str(args.scene.resolve()),
        frames=len(selected), points=len(points), voxel_size_m=args.voxel_size,
        sdf_trunc_m=args.sdf_trunc, depth_scale=args.depth_scale, depth_max_m=args.depth_max,
        pose_convention='OpenCV camera_to_world; inverted for integration',
        normalize_world=False, depth_provenance='Third-party supplied depth; original sensor/render source unverified',
        bounds_min=points.min(0).tolist(), bounds_max=points.max(0).tolist(),
        elapsed_seconds=time.monotonic()-start, output=str(target.resolve()))
    (out/'fusion_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
