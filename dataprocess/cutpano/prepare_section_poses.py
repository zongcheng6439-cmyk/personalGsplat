"""核对 section0907 时间戳，从 IMU 与原始切图 pose 恢复并验证相机标定。

输入：--imu、--reference（原流程 pano_pose.txt）、--section-images（COLMAP images.txt）。
输出：--output JSON，含每张全景图的 T_camera_to_world、外参和验证统计。
运行（脚本所在目录）：
    python3 prepare_section_poses.py --imu imu_pose.txt --reference pano_pose_reference.txt --section-images images.txt --output section0907_poses.json
原流程约定：IMU 为 id,timestamp,qx,qy,qz,qw,x,y,z；reference 为 name,qw,qx,qy,qz,tx,ty,tz（world-to-camera）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def transform(qxyzw, translation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(qxyzw).as_matrix()
    result[:3, 3] = translation
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('imu', 'reference', 'section-images', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    rows = [s for l in args.section_images.read_text().splitlines()
            if len(s := l.split()) == 10 and not s[0].startswith('#')]
    names = sorted({Path(s[9]).stem for s in rows})
    imu = {}
    for line in args.imu.read_text().splitlines():
        s = line.split(',')
        if len(s) != 9 or s[1] not in names:
            continue
        v = np.array(s[2:], float)
        if not np.isfinite(v).all() or abs(np.linalg.norm(v[:4]) - 1) > 1e-3:
            raise ValueError(f'Invalid IMU pose: {s[1]}')
        if s[1] in imu:
            raise ValueError(f'Duplicate IMU timestamp: {s[1]}')
        imu[s[1]] = transform(v[:4], v[4:])
    reference = {}
    for line in args.reference.read_text().splitlines():
        s = line.split(',')
        if len(s) != 8 or Path(s[0]).stem not in imu:
            continue
        v = np.array(s[1:], float)
        reference[s[0]] = np.linalg.inv(transform(v[[1, 2, 3, 0]], v[4:]))
    if len(imu) != len(names) or any(f'image0/{n}.png' not in reference for n in names):
        raise ValueError('Missing IMU/reference timestamp')
    extrinsics = np.array([np.linalg.inv(imu[n]) @ reference[f'image0/{n}.png'] for n in names])
    calibration = np.eye(4)
    calibration[:3, :3] = Rotation.from_matrix(extrinsics[:, :3, :3]).mean().as_matrix()
    calibration[:3, 3] = np.median(extrinsics[:, :3, 3], axis=0)
    translations, angles = [], []
    poses = {n: imu[n] @ calibration for n in names}
    for n in names:
        for i, yaw in enumerate([0, 270, 180, 90, 315, 225, 135, 45]):
            expected = reference[f'image{i}/{n}.png']
            predicted = poses[n].copy()
            predicted[:3, :3] = poses[n][:3, :3] @ Rotation.from_euler('y', yaw, degrees=True).as_matrix()
            translations.append(np.linalg.norm(predicted[:3, 3] - expected[:3, 3]))
            angles.append(np.rad2deg(Rotation.from_matrix(predicted[:3, :3].T @ expected[:3, :3]).magnitude()))
    report = {'section_image_count': len(rows), 'panorama_count': len(names),
              'reference_pose_count_checked': len(angles),
              'reference_position_error_max': float(max(translations)),
              'reference_rotation_error_max_degrees': float(max(angles)),
              'reference_position_error_median': float(np.median(translations))}
    # Truncated IMU decimal values cause small residuals; reject a wrong frame/convention.
    if max(translations) > 0.002 or max(angles) > 0.01:
        raise ValueError(f'Inconsistent IMU/camera calibration: {report}')
    # Refined SfM poses are a separate diagnostic, not a calibration target.
    refined_errors = []
    for s in rows:
        v = np.array(s[1:8], float)
        cw = transform(v[[1, 2, 3, 0]], v[4:])
        refined_errors.append(np.linalg.norm(np.linalg.inv(cw)[:3, 3] - poses[Path(s[9]).stem][:3, 3]))
    report['refined_sfm_center_difference_p50_p95_max'] = np.percentile(refined_errors, [50, 95, 100]).tolist()
    data = {'convention': 'T_camera_to_world; panorama center frame: x right, y down, z forward',
            'calibration_source': str(args.reference), 'T_imu_camera': calibration.tolist(),
            'validation': report, 'poses': {n: pose.tolist() for n, pose in poses.items()}}
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
