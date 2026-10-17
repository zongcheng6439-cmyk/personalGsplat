"""导出新旧 image0 位姿，比较相机中心距离和 SO(3) 旋转角差。

输入：--dataset 为新切图目录，IMU 位姿读取其中 sparse/0/images.txt，
可用 --imu-images 显式覆盖；--sfm-images 为旧 section_0907/sparse/0/images.txt。
两组均使用 COLMAP 的 world-to-camera 外参求逆，不读取 cameras/*.json 位姿。
输出：--output 下三组 T_camera_to_world 位姿文本、两组逐帧差异和 summary.json。
新 image0 是 yaw0/pitch-30；另导出新平视 image1，对应旧 image0，消除视角差。
运行（任意目录）：
    python3 compare_image0_poses.py --dataset /media/zc/SSD/ar/ABXBJQ_BJ/B3/B3_Full/section0907_perspective_2048 --sfm-images /media/zc/SSD/ar/ABXBJQ_BJ/B3/section_0907/sparse/0/images.txt --output /media/zc/SSD/ar/ABXBJQ_BJ/B3/B3_Full/section0907_perspective_2048/image0_pose_comparison
位姿行：timestamp Cx Cy Cz qx qy qz qw（相机到世界）；距离单位沿用输入坐标，当前数据为米。
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def stats(values):
    a=np.asarray(values)
    return dict(zip(('min','median','p95','max'),map(float,np.percentile(a,[0,50,95,100]))),mean=float(a.mean()))


def pose_row(stamp,wc):
    q=Rotation.from_matrix(wc[:3,:3]).as_quat()
    return stamp+' '+' '.join(f'{v:.17g}' for v in np.r_[wc[:3,3],q])+'\n'


def read_image_poses(path):
    """Read COLMAP two-line image records and convert both inputs identically."""
    poses={}
    expect_pose=True
    for line in path.read_text().splitlines():
        if line.lstrip().startswith('#'):continue
        if not expect_pose:
            expect_pose=True  # Skip the POINTS2D line, including an empty one.
            continue
        if not line.strip():continue
        s=line.split(maxsplit=9)
        if len(s)!=10:raise ValueError(f'Invalid image record in {path}: {line}')
        values=np.array(s[1:8],float)
        if not np.isfinite(values).all() or np.linalg.norm(values[:4])<1e-12:
            raise ValueError(f'Invalid pose in {path}: {s[9]}')
        cw=np.eye(4)
        cw[:3,:3]=Rotation.from_quat(values[[1,2,3,0]]).as_matrix()
        cw[:3,3]=values[4:]
        if s[9] in poses:raise ValueError(f'Duplicate image in {path}: {s[9]}')
        # Camera to world: R_wc = R_cw.T, C = -R_cw.T @ t_cw.
        poses[s[9]]=np.linalg.inv(cw)
        expect_pose=False
    return poses


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','sfm-images','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--imu-images',type=Path,help='Defaults to <dataset>/sparse/0/images.txt')
    args=p.parse_args()
    imu_path=args.imu_images or args.dataset/'sparse/0/images.txt'
    imu=read_image_poses(imu_path)
    sfm=read_image_poses(args.sfm_images)
    poses={name:['# timestamp Cx Cy Cz qx qy qz qw; T_camera_to_world\n'] for name in ('sfm_image0','imu_directory_image0','imu_aligned_image0')}
    differences={mode:['# timestamp dx_m dy_m dz_m distance_m angle_deg; delta_center=IMU-SfM\n'] for mode in ('directory_image0','same_direction')}
    records=[]
    for literal_name in sorted(n for n in imu if n.startswith('image0/')):
        stamp=Path(literal_name).stem
        aligned_name='image1/'+Path(literal_name).name
        if literal_name not in sfm:raise ValueError(f'Missing SfM image0: {literal_name}')
        if aligned_name not in imu:raise ValueError(f'Missing IMU horizontal view: {aligned_name}')
        ref=sfm[literal_name];poses['sfm_image0'].append(pose_row(stamp,ref))
        entry={'timestamp':stamp,'sfm_image':literal_name,'T_camera_to_world_sfm':ref.tolist()}
        for mode,name,pitch,pose_name in [('directory_image0',literal_name,-30,'imu_directory_image0'),('same_direction',aligned_name,0,'imu_aligned_image0')]:
            wc=imu[name]
            if not np.isfinite(wc).all() or not np.allclose(wc[:3,:3].T@wc[:3,:3],np.eye(3),atol=1e-8):
                raise ValueError(f'Invalid rotation {stamp}')
            delta=wc[:3,3]-ref[:3,3]
            distance=float(np.linalg.norm(delta))
            relative=ref[:3,:3].T@wc[:3,:3]
            angle=float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))
            trace_angle=float(np.rad2deg(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
            if not np.isclose(angle,trace_angle,atol=1e-7):raise ValueError('Angle verification failed')
            poses[pose_name].append(pose_row(stamp,wc))
            differences[mode].append(stamp+' '+' '.join(f'{v:.12g}' for v in np.r_[delta,distance,angle])+'\n')
            entry[mode]={'imu_image':name,'yaw':0,'pitch':pitch,
                         'delta_center_m':delta.tolist(),'distance_m':distance,'angle_deg':angle,'T_camera_to_world_imu':wc.tolist()}
        records.append(entry)
    if not records:raise ValueError('No matching poses')
    result={'count':len(records),'pose_convention':'T_camera_to_world; C is world camera center; quaternion xyzw',
            'imu_pose_source':str(imu_path.resolve()),'sfm_pose_source':str(args.sfm_images.resolve()),
            'angle_definition':'geodesic rotation angle of R_sfm.T @ R_imu, degrees',
            'summary':{mode:{key:stats([r[mode][key] for r in records]) for key in ('distance_m','angle_deg')} for mode in differences},
            'records':records}
    args.output.mkdir(parents=True,exist_ok=True)
    for name,lines in poses.items():(args.output/f'{name}_poses.txt').write_text(''.join(lines))
    for name,lines in differences.items():(args.output/f'{name}_differences.txt').write_text(''.join(lines))
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'count':len(records),'summary':result['summary']},indent=2))


if __name__=='__main__':main()
