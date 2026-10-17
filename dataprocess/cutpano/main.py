"""将 2:1 全景图切成指定朝向的透视图，正俯仰角表示向上看。

输入：--input 指定单张全景图或图像目录。
输出：默认在输入路径的父目录下生成 perspective_2048，按
images/image0/<时间戳>.png 至 images/image17/<时间戳>.png 保存图像，
每个全景的元数据保存在 cameras/<时间戳>.json。使用三次插值生成 2048×2048 PNG，
同时保存相机内参、相对旋转 cameras.json。传入 --pose-file 后还保存世界位姿
以及 sparse/0 下的 COLMAP PINHOLE 文本模型（不含三维点）。

运行（任意工作目录，服务器示例）：
    python3 /home/zc/code/aibee/gsplat/dataprocess/cutpano/main.py --input /media/zc/SSD/ar/ABXBJQ_BJ/B3/B3_Full/pano/
可选参数：--output 输出目录 --size 2048 --fov 90 --pitch -30 0 30 --roll 0
          --pose-file section0907_poses.json --workers 6。
"""

import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import cv2
import numpy as np


def rotation(yaw, pitch, roll):
    """Camera-to-panorama rotation; both frames use x right, y down, z forward."""
    y, p, r = np.deg2rad([yaw, pitch, roll])
    cy, sy, cp, sp, cr, sr = np.cos(y), np.sin(y), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return ry @ rx @ rz


def sampling_map(width, height, size, fov, orientation):
    focal = size / (2 * np.tan(np.deg2rad(fov) / 2))
    center = (size - 1) / 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    rays = np.stack(((x - center) / focal, (y - center) / focal, np.ones_like(x)), axis=-1)
    rays = rays @ orientation.T
    longitude = np.arctan2(rays[..., 0], rays[..., 2])
    latitude = np.arctan2(-rays[..., 1], np.hypot(rays[..., 0], rays[..., 2]))
    # Cubic interpolation needs two wrapped columns on either side of the seam.
    u = ((longitude / (2 * np.pi) + 0.5) * width - 0.5) % width + 2
    v = np.clip((0.5 - latitude / np.pi) * height - 0.5, 0, height - 1)
    intrinsic = [[focal, 0, center], [0, focal, center], [0, 0, 1]]
    return u.astype(np.float32), v.astype(np.float32), intrinsic


@lru_cache(maxsize=18)
def cached_map(width, height, size, fov, yaw, pitch, roll):
    return sampling_map(width, height, size, fov, rotation(yaw, pitch, roll))


def cut_image(path, output, args, views, poses):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'cannot read {path}')
    height, width = image.shape[:2]
    if width != height * 2:
        raise ValueError(f'{path}: expected 2:1 panorama, got {width}x{height}')
    padded = np.concatenate((image[:, -2:], image, image[:, :2]), axis=1)
    destination = output / 'images'
    metadata = {'source': str(path), 'width': args.size, 'height': args.size,
                'pixel_convention': 'OpenCV integer pixel centers; COLMAP export adds 0.5 to cx,cy',
                'coordinate_system': 'x right, y down, z forward; yaw 0 is panorama center; positive pitch up',
                'views': []}
    for view_index, (yaw, pitch, orientation) in enumerate(views):
        u, v, intrinsic = cached_map(width, height, args.size, args.fov, yaw, pitch, args.roll)
        view = cv2.remap(padded, u, v, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        name = f'image{view_index}/{path.stem}.png'
        if not cv2.imwrite(str(destination / name), view):
            raise RuntimeError(f'cannot save {destination / name}')
        record = {'image': name, 'yaw': yaw, 'pitch': pitch, 'roll': args.roll,
                  'K': intrinsic, 'R_camera_to_panorama': orientation.tolist()}
        if poses is not None:
            world = poses[path.stem].copy()
            world[:3, :3] = world[:3, :3] @ orientation
            record['T_camera_to_world'] = world.tolist()
            record['T_world_to_camera'] = np.linalg.inv(world).tolist()
        metadata['views'].append(record)
    (output / 'cameras' / f'{path.stem}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return path.name


def export_colmap(output, files, size):
    from scipy.spatial.transform import Rotation
    sparse = output / 'sparse' / '0'
    sparse.mkdir(parents=True, exist_ok=True)
    rows = ['# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME; world-to-camera\n']
    for path in files:
        metadata = json.loads((output / 'cameras' / f'{path.stem}.json').read_text())
        for view in metadata['views']:
            cw = np.array(view['T_world_to_camera'])
            q = Rotation.from_matrix(cw[:3, :3]).as_quat()[[3, 0, 1, 2]]
            values = ' '.join(f'{x:.17g}' for x in np.r_[q, cw[:3, 3]])
            rows.append(f'{len(rows)} {values} 1 {view["image"]}\n\n')
    k = metadata['views'][0]['K']
    (sparse / 'cameras.txt').write_text(f'1 PINHOLE {size} {size} {k[0][0]:.17g} {k[1][1]:.17g} {k[0][2]+0.5:.17g} {k[1][2]+0.5:.17g}\n')
    (sparse / 'images.txt').write_text(''.join(rows))
    (sparse / 'points3D.txt').write_text('# No triangulated points; image root is output/images.\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Panorama file or directory')
    parser.add_argument('--output', type=Path, help='Defaults to perspective_<size> in the input parent directory')
    parser.add_argument('--size', type=int, default=2048)
    parser.add_argument('--fov', type=float, default=90, help='Horizontal field of view in degrees')
    parser.add_argument('--yaw', type=float, nargs='+', default=[0, 60, 120, 180, 240, 300])
    parser.add_argument('--pitch', type=float, nargs='+', default=[-30, 0, 30], help='Positive looks up')
    parser.add_argument('--roll', type=float, default=0, help='Camera tilt around its optical axis')
    parser.add_argument('--pose-file', type=Path, help='JSON from prepare_section_poses.py')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.size <= 0 or not 0 < args.fov < 180:
        parser.error('size must be positive and 0 < fov < 180')
    if not np.isfinite(args.yaw + args.pitch + [args.roll]).all():
        parser.error('angles must be finite')
    if args.workers < 1:
        parser.error('workers must be positive')
    source = args.input.expanduser().resolve()
    if not source.exists():
        parser.error(f'input does not exist: {source}')
    files = [source] if source.is_file() else sorted(
        p for p in source.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
    if not files:
        parser.error(f'no images found: {source}')
    if len({p.stem for p in files}) != len(files):
        parser.error('input image stems must be unique')
    poses = None
    if args.pose_file:
        from scipy.spatial.transform import Rotation  # Verify dependency before cutting.
        poses = {n: np.array(p, float) for n, p in json.loads(args.pose_file.read_text())['poses'].items()}
        for path in files:
            p = poses.get(path.stem)
            if p is None or p.shape != (4, 4) or not np.isfinite(p).all():
                parser.error(f'missing or invalid pose: {path.stem}')
            if not (np.allclose(p[3], [0, 0, 0, 1]) and np.allclose(p[:3,:3].T @ p[:3,:3], np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(p[:3,:3]), 1)):
                parser.error(f'non-rigid pose: {path.stem}')
    output = args.output or source.parent / f'perspective_{args.size}'
    output.mkdir(parents=True, exist_ok=True)
    views = [(yaw, pitch, rotation(yaw, pitch, args.roll)) for yaw in args.yaw for pitch in args.pitch]
    (output / 'cameras').mkdir(exist_ok=True)
    for index in range(len(views)):
        (output / 'images' / f'image{index}').mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    first = cv2.imread(str(files[0]))
    if first is None:
        raise RuntimeError(f'cannot read {files[0]}')
    height, width = first.shape[:2]
    del first
    for yaw, pitch, _ in views:
        cached_map(width, height, args.size, args.fov, yaw, pitch, args.roll)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(cut_image, path, output, args, views, poses) for path in files]
        for index, job in enumerate(as_completed(jobs), 1):
            print(f'[{index}/{len(files)}] {job.result()}: {len(views)} views -> {output}', flush=True)
    if poses is not None:
        export_colmap(output, files, args.size)


if __name__ == '__main__':
    main()
