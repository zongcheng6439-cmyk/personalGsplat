"""验证切图尺寸、世界位姿、COLMAP 导出及实际图像特征重投影。

输入：--output 为 main.py 输出目录（images/imageN/时间戳.png、cameras/时间戳.json），--poses 为 prepare_section_poses.py 输出，
      --legacy-images 为 section0907/images，用旧 image0 独立验证全景零方向。
输出：切图目录 validation.json，任一关键校验失败则抛出异常。
运行（任意目录）：
    python3 validate_section_views.py --output /path/to/section0907_perspective_2048 --poses /path/to/section0907_poses.json --legacy-images /path/to/section_0907/images
"""
import argparse
import json
from pathlib import Path
import struct

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def matches(source, target, homography):
    a, b = cv2.imread(str(source), 0), cv2.imread(str(target), 0)
    if a is None or b is None:
        raise ValueError(f'Cannot read {source} or {target}')
    sift = cv2.SIFT_create(nfeatures=2500)
    ka, da = sift.detectAndCompute(a, None)
    kb, db = sift.detectAndCompute(b, None)
    if da is None or db is None:
        raise ValueError('No SIFT descriptors')
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.65 * n.distance]
    if len(good) < 30:
        raise ValueError(f'Insufficient feature matches: {len(good)}')
    pa = np.array([ka[m.queryIdx].pt for m in good])
    pb = np.array([kb[m.trainIdx].pt for m in good])
    projected = np.c_[pa, np.ones(len(pa))] @ homography.T
    projected = projected[:, :2] / projected[:, 2:]
    errors = np.linalg.norm(projected - pb, axis=1)
    result = {'matches': len(good), 'median_pixels': float(np.median(errors)),
              'p95_pixels': float(np.percentile(errors, 95)), 'fraction_under_3px': float(np.mean(errors < 3))}
    # Fit geometry from image matches independently of the exported camera poses.
    fitted, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if fitted is None or mask is None:
        raise ValueError('Independent image homography estimation failed')
    inliers = mask.ravel().astype(bool)
    result['ransac_inliers'] = int(inliers.sum())
    result['ransac_inlier_expected_projection_p95_pixels'] = float(np.percentile(errors[inliers], 95))
    if inliers.sum() < 30 or inliers.mean() < 0.6 or result['ransac_inlier_expected_projection_p95_pixels'] > 3:
        raise ValueError(f'Independent image geometry disagrees with camera poses: {result}')
    if result['median_pixels'] > 2 or result['fraction_under_3px'] < 0.8:
        raise ValueError(f'Feature projection failed: {source}, {target}: {result}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'poses', 'legacy-images'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    poses = json.loads(args.poses.read_text())['poses']
    metadata_files = sorted((args.output/'cameras').glob('*.json'))
    assert len(metadata_files) == len(poses), (len(metadata_files), len(poses))
    records = {}
    max_center_error = max_inverse_error = 0.0
    for path in metadata_files:
        data = json.loads(path.read_text())
        original = np.array(poses[Path(data['source']).stem])
        assert len(data['views']) == 18
        for view in data['views']:
            file = args.output / 'images' / view['image']
            with file.open('rb') as f:
                header = f.read(24)
            assert header[:8] == b'\x89PNG\r\n\x1a\n'
            assert struct.unpack('>II', header[16:24]) == (data['width'], data['height'])
            wc, cw = np.array(view['T_camera_to_world']), np.array(view['T_world_to_camera'])
            assert np.isfinite(wc).all() and np.isfinite(cw).all()
            assert np.allclose(wc[:3,:3].T @ wc[:3,:3], np.eye(3), atol=1e-8)
            assert np.isclose(np.linalg.det(wc[:3,:3]), 1)
            max_center_error = max(max_center_error, float(np.linalg.norm(wc[:3,3] - original[:3,3])))
            max_inverse_error = max(max_inverse_error, float(np.max(np.abs(wc @ cw - np.eye(4)))))
            # Verify the center and corner rays project back through world coordinates.
            k = np.array(view['K'])
            pixels = np.array([[0,0,1],[data['width']-1,data['height']-1,1],[k[0,2],k[1,2],1]])
            rays = pixels @ np.linalg.inv(k).T
            # 先从像素构造测试点 → 转到世界坐标 → 再转回相机并投影，检查是否回到原像素
            world = (rays * 3) @ wc[:3,:3].T + wc[:3,3]
            recovered = (world @ cw[:3,:3].T + cw[:3,3]) @ k.T
            recovered = recovered / recovered[:,2:]
            assert np.allclose(recovered, pixels, atol=1e-6)
            records[view['image']] = view
    assert max_center_error < 1e-8 and max_inverse_error < 1e-8
    camera = (args.output/'sparse/0/cameras.txt').read_text().split()
    assert camera[1] == 'PINHOLE'
    sample_k = np.array(next(iter(records.values()))['K'])
    assert np.allclose(np.array(camera[4:],float), [sample_k[0,0], sample_k[1,1], sample_k[0,2]+0.5, sample_k[1,2]+0.5])
    exported = set()
    for line in (args.output/'sparse/0/images.txt').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        s = line.split(); assert len(s) == 10
        v = np.array(s[1:8],float)
        cw = np.array(records[s[9]]['T_world_to_camera'])
        assert np.allclose(Rotation.from_quat(v[[1,2,3,0]]).as_matrix(),cw[:3,:3],atol=1e-8)
        assert np.allclose(v[4:],cw[:3,3],atol=1e-8)
        assert s[9] not in exported
        exported.add(s[9])
    assert exported == set(records)
    features = []
    for index in (0, len(metadata_files)//2, len(metadata_files)-1):
        path = metadata_files[index]
        data = json.loads(path.read_text())
        source = next(v for v in data['views'] if v['yaw'] == 0 and v['pitch'] == 0)
        k = np.array(source['K']); wc = np.array(source['T_camera_to_world'])
        for yaw, pitch in ((0,-30), (60,0)):
            target = next(v for v in data['views'] if v['yaw']==yaw and v['pitch']==pitch)
            twc = np.array(target['T_camera_to_world'])
            h = np.array(target['K']) @ twc[:3,:3].T @ wc[:3,:3] @ np.linalg.inv(k)
            result = matches(args.output/'images'/source['image'],args.output/'images'/target['image'],h)
            result.update({'panorama':path.stem,'pair':f'yaw0_pitch0 -> yaw{yaw}_pitch{pitch}'})
            features.append(result)
        legacy = args.legacy_images/'image0'/f'{Path(data["source"]).stem}.png'
        old = cv2.imread(str(legacy),0)
        assert old is not None
        # Original Equriectangular.py uses (size-1)/2 focal length at 90 degrees.
        cx,cy = (old.shape[1]-1)/2, (old.shape[0]-1)/2
        old_k = np.array([[cx,0,cx],[0,cy,cy],[0,0,1]])
        result = matches(legacy,args.output/'images'/source['image'],k @ np.linalg.inv(old_k))
        result.update({'panorama':path.stem,'pair':'legacy image0 -> new yaw0_pitch0'})
        features.append(result)
    report = {'panoramas':len(metadata_files),'images':len(records),'max_center_error':max_center_error,
              'max_inverse_error':max_inverse_error,'colmap_export_valid':True,'feature_checks':features}
    (args.output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
