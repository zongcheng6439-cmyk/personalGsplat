"""用 extracted.ply 检查新切图的 IMU 位姿与 SfM 对照位姿的点云对齐。

输入：--cloud 彩色 ASCII PLY、--dataset 切图根目录、--sfm-images 原 images.txt。
IMU 外参及内参分别从 dataset/sparse/0/images.txt、cameras.txt 读取；
SfM 外参从 --sfm-images 读取。两组对同一张新切图使用相同内参。
JSON 仅用于图像名称和切图角度选择，不作为位姿或内参来源。
输出：--output 内的图像/点云/叠加对照图与 report.json。仅诊断，不优化位姿。
随机抽取三张：在下面命令末尾追加 --random-count 3 --seed 20261012。
同角度直接比较：追加 --matched-timestamps 3 --seed 20261012，随机选三个时间戳，
仅比较 yaw=0/180、pitch=roll=0，SfM 分别直接读取旧 image0/image2 外参。
额外输出 comparison.jpg（三图对照）与 *_pose_difference.jpg（同一3D点的投影差异箭头）。
运行（任意目录）：
    python3 validate_cloud_projection.py --cloud /media/zc/SSD/ar/ABXBJQ_BJ/B3/section_0907/sparse/0/extracted.ply --dataset /media/zc/SSD/ar/ABXBJQ_BJ/B3/B3_Full/section0907_perspective_2048 --sfm-images /media/zc/SSD/ar/ABXBJQ_BJ/B3/section_0907/sparse/0/images.txt --output /media/zc/SSD/ar/ABXBJQ_BJ/B3/B3_Full/section0907_perspective_2048/cloud_validation
每像素保留最近点，再以 3x3 点覆盖近似表面遮挡；稀疏孔洞不参与评分。
边缘距离是诊断量（在 --size 分辨率下），不是真实 3D-2D 对应点重投影误差。
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from compare_image0_poses import read_image_poses


def read_new_colmap(model):
    poses=read_image_poses(model/'images.txt')
    image_cameras={}
    expect_pose=True
    for line in (model/'images.txt').read_text().splitlines():
        if line.lstrip().startswith('#'):continue
        if not expect_pose:
            expect_pose=True
            continue
        if not line.strip():continue
        s=line.split(maxsplit=9)
        image_cameras[s[9]]=int(s[8])
        expect_pose=False
    cameras={}
    for line in (model/'cameras.txt').read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        s=line.split()
        if len(s)!=8 or s[1]!='PINHOLE':
            raise ValueError('New perspective images require PINHOLE cameras')
        width,height=map(int,s[2:4]);fx,fy,cx,cy=map(float,s[4:])
        # Convert COLMAP half-integer pixel centers to OpenCV integer centers.
        k=np.array([[fx,0,cx-0.5],[0,fy,cy-0.5],[0,0,1]],float)
        cameras[int(s[0])]=(width,height,k)
    return poses,image_cameras,cameras


def read_cloud(path):
    with path.open() as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Missing PLY header')
            header.append(line.strip())
            if line.strip() == 'end_header':
                break
        if 'format ascii 1.0' not in header:
            raise ValueError('This diagnostic reader expects the supplied ASCII PLY')
        start = next(i for i, s in enumerate(header) if s.startswith('element vertex '))
        count = int(header[start].split()[-1])
        properties = []
        for line in header[start+1:]:
            if not line.startswith('property '):
                break
            properties.append(line.split()[-1])
        columns = [properties.index(n) for n in ('x','y','z','red','green','blue')]
        values = np.loadtxt(f, usecols=columns, max_rows=count, dtype=np.float32)
    if len(values) != count or not np.isfinite(values).all():
        raise ValueError('Invalid point cloud')
    return values[:, :3].copy(), values[:, [5,4,3]].astype(np.uint8)


def render(xyz, colors, wc, k, size):
    # For row vectors, world-to-camera is (X-C) @ R_camera_to_world.
    camera = (xyz-wc[:3,3]) @ wc[:3,:3]
    ids = np.flatnonzero(camera[:,2] > 0.1)
    points = camera[ids]
    uv = points[:,:2] / points[:,2:] * [k[0,0], k[1,1]] + [k[0,2], k[1,2]]
    inside = (uv[:,0]>=0)&(uv[:,0]<size-1)&(uv[:,1]>=0)&(uv[:,1]<size-1)
    ids, points, uv = ids[inside], points[inside], np.rint(uv[inside]).astype(int)

    flat = uv[:,1]*size+uv[:,0]
    depth = np.full(size*size,np.inf,np.float32)

    np.minimum.at(depth,flat,points[:,2])
    # Depth was stored as float32; compare at the same precision.
    # 只有深度等于该像素最小深度的点才算"最近点"
    nearest = points[:,2].astype(np.float32) == depth[flat]
    rgb = np.zeros((size*size,3),np.uint8)
    rgb[flat[nearest]] = colors[ids[nearest]]
    depth, rgb = depth.reshape(size,size), rgb.reshape(size,size,3)
    padded_depth = np.pad(depth,1,constant_values=np.inf)
    padded_rgb = np.pad(rgb,((1,1),(1,1),(0,0)))
    result_depth = np.full_like(depth,np.inf)
    result_rgb = np.zeros_like(rgb)
    for dy in range(3):
        for dx in range(3):
            z = padded_depth[dy:dy+size,dx:dx+size]
            update = z < result_depth
            result_depth[update] = z[update]
            result_rgb[update] = padded_rgb[dy:dy+size,dx:dx+size][update]
    return result_rgb,result_depth


def score(image, cloud, depth, common):
    reliable = cv2.erode(common.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    original_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cloud_gray = cv2.cvtColor(cloud,cv2.COLOR_BGR2GRAY)
    original_edges = cv2.Canny(cv2.GaussianBlur(original_gray,(3,3),0),50,120)
    cloud_edges = cv2.Canny(cv2.GaussianBlur(cloud_gray,(3,3),0),50,120)>0
    cloud_edges &= reliable
    distance = cv2.distanceTransform((original_edges==0).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    edges = distance[cloud_edges]
    valid = np.flatnonzero(reliable.ravel())[::8]
    if len(edges)<100 or len(valid)<100:
        return {'valid':False,'edge_count':len(edges)},cloud_edges
    a = cloud.reshape(-1,3)[valid].astype(float)
    b = image.reshape(-1,3)[valid].astype(float)
    design = np.c_[a,np.ones(len(a))]
    corrected = design @ np.linalg.lstsq(design,b,rcond=None)[0]
    correlation = np.corrcoef(cloud_gray.ravel()[valid],original_gray.ravel()[valid])[0,1]
    shifts = {}
    for shift in (-16,-8,0,8,16):
        shifted = np.zeros_like(cloud_edges)
        if shift>0: shifted[:,shift:]=cloud_edges[:,:-shift]
        elif shift<0: shifted[:,:shift]=cloud_edges[:,-shift:]
        else: shifted=cloud_edges
        shifts[str(shift)] = float(np.mean(np.minimum(distance[shifted],20)))
    return {'valid':True,'edge_count':int(len(edges)),
            'edge_distance_median_px':float(np.median(edges)),
            'edge_distance_p90_px':float(np.percentile(edges,90)),
            'edge_distance_capped_mean_px':float(np.minimum(edges,20).mean()),
            'edge_fraction_within_3px':float(np.mean(edges<=3)),
            'gray_correlation':float(correlation),'color_affine_mae_0_255':float(np.abs(corrected-b).mean()),
            'horizontal_shift_capped_mean_px':shifts},cloud_edges


def panel(image,title):
    resized = cv2.resize(image,(512,512),interpolation=cv2.INTER_AREA)
    result = cv2.copyMakeBorder(resized,28,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
    cv2.putText(result,title,(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    return result


def projection_difference(xyz, imu_wc, sfm_wc, k, depths, image):
    """Compare identical cloud points visible under both poses, not photo matches."""
    size=image.shape[0]
    pixels=[]; valid=np.ones(len(xyz),bool)
    for wc,depth in zip((imu_wc,sfm_wc),depths):
        camera=(xyz-wc[:3,3]) @ wc[:3,:3]
        z=camera[:,2]
        safe=np.where(z>0.1,z,1)
        uv=camera[:,:2]/safe[:,None]*[k[0,0],k[1,1]]+[k[0,2],k[1,2]]
        inside=(z>0.1)&(uv[:,0]>=0)&(uv[:,0]<size-1)&(uv[:,1]>=0)&(uv[:,1]<size-1)
        indices=np.flatnonzero(inside)
        rounded=np.rint(uv[indices]).astype(int)
        visible=np.zeros(len(xyz),bool)
        visible[indices]=np.abs(z[indices]-depth[rounded[:,1],rounded[:,0]]) <= np.maximum(0.05,0.01*z[indices])
        valid &= visible;pixels.append(uv)
    a,b=pixels[0][valid],pixels[1][valid]
    preview=image.copy()
    if not len(a):
        return {'common_visible_points':0},preview
    delta=np.linalg.norm(a-b,axis=1)
    # One arrow per 64-pixel cell so dense cloud areas do not dominate the display.
    cells=(a[:,1]//64).astype(int)*(size//64+1)+(a[:,0]//64).astype(int)
    _,chosen=np.unique(cells,return_index=True)
    for i in chosen:
        start,end=tuple(np.rint(a[i]).astype(int)),tuple(np.rint(b[i]).astype(int))
        cv2.arrowedLine(preview,start,end,(255,255,0),1,cv2.LINE_AA,tipLength=0.25)
        cv2.circle(preview,end,2,(255,0,255),-1)
    return {'common_visible_points':len(a),'median_px':float(np.median(delta)),
            'p95_px':float(np.percentile(delta,95))},preview


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cloud','dataset','sfm-images','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--size',type=int,default=1024)
    parser.add_argument('--random-count',type=int,default=0,help='Uniform random sample from all perspective images; 0 uses the original 18-view check')
    parser.add_argument('--seed',type=int,default=20261012)
    parser.add_argument('--matched-timestamps',type=int,default=0,help='Sample timestamps and compare yaw 0/180 against direct original image0/image2 SfM poses')
    args=parser.parse_args()
    if args.matched_timestamps < 0 or (args.matched_timestamps and args.random_count):
        parser.error('matched-timestamps must be nonnegative and cannot combine with random-count')
    cv2.setNumThreads(1)
    xyz,colors=read_cloud(args.cloud)
    model=args.dataset/'sparse/0'
    imu_poses,image_cameras,cameras=read_new_colmap(model)
    metadata=sorted((args.dataset/'cameras').glob('*.json'))
    if args.matched_timestamps:
        eligible=[]
        for path in metadata:
            views=json.loads(path.read_text())['views']
            indices=[next((i for i,v in enumerate(views) if v['yaw']==yaw and v['pitch']==0 and v['roll']==0),None) for yaw in (0,180)]
            if all(i is not None for i in indices): eligible.append((path,indices))
        if args.matched_timestamps > len(eligible): parser.error('not enough timestamps with both directions')
        chosen=np.random.default_rng(args.seed).choice(len(eligible),args.matched_timestamps,replace=False)
        selected=[eligible[i] for i in sorted(chosen)]
    elif args.random_count:
        population=[(p,i) for p in metadata for i in range(len(json.loads(p.read_text())['views']))]
        if not 0 < args.random_count <= len(population):
            parser.error('invalid random-count')
        indices=np.random.default_rng(args.seed).choice(len(population),args.random_count,replace=False)
        selected=[(population[i][0],[population[i][1]]) for i in indices]
    else:
        selected=[]
        for i in np.linspace(0,len(metadata)-1,6).astype(int):
            p=metadata[i];views=json.loads(p.read_text())['views']
            selected.append((p,[j for j,v in enumerate(views) if (v['yaw'],v['pitch']) in ((0,0),(0,-30),(120,0))]))
    total=sum(len(indices) for _,indices in selected)
    sfm=read_image_poses(args.sfm_images)
    for path,indices in selected:
        required=('image0','image2') if args.matched_timestamps else ('image0',)
        for folder in required:
            if f'{folder}/{path.stem}.png' not in sfm:
                raise ValueError(f'Missing direct SfM pose: {folder}/{path.stem}.png')
    args.output.mkdir(parents=True,exist_ok=True)
    reports=[];comparison=[]
    for path,view_indices in selected:
        data=json.loads(path.read_text())
        timestamp_comparison=[]
        for view_index in view_indices:
            view=data['views'][view_index];yaw,pitch=view['yaw'],view['pitch']
            image=cv2.imread(str(args.dataset/'images'/view['image']))
            if image is None: raise ValueError(view['image'])
            width,height,source_k=cameras[image_cameras[view['image']]]
            if image.shape[:2]!=(height,width):raise ValueError(f'Image size disagrees with cameras.txt: {view["image"]}')
            size=args.size
            image=cv2.resize(image,(size,size),interpolation=cv2.INTER_AREA)
            k=source_k.copy()
            for axis,scale in enumerate((size/width,size/height)):
                k[axis]*=scale;k[axis,2]+=(scale-1)/2
            imu_wc=imu_poses[view['image']]
            sfm_name=f'image{2 if args.matched_timestamps and yaw==180 else 0}/{path.stem}.png'
            refined_wc=sfm[sfm_name].copy()
            if not args.matched_timestamps:
                base_name=next(v['image'] for v in data['views'] if v['yaw']==0 and v['pitch']==0 and v['roll']==0)
                relative=imu_poses[base_name][:3,:3].T @ imu_wc[:3,:3]
                refined_wc[:3,:3]=refined_wc[:3,:3] @ relative
            imu_rgb,imu_depth=render(xyz,colors,imu_wc,k,size)
            sfm_rgb,sfm_depth=render(xyz,colors,refined_wc,k,size)
            common=np.isfinite(imu_depth)&np.isfinite(sfm_depth)
            entry={'timestamp':path.stem,'image':view['image'],'yaw':yaw,'pitch':pitch,'score_image_size':size,
                   'sfm_source_image':sfm_name,'direct_sfm_pose':bool(args.matched_timestamps),
                   'T_camera_to_world_imu':imu_wc.tolist(),'T_camera_to_world_sfm':refined_wc.tolist(),
                   'camera_id':image_cameras[view['image']],'K_at_diagnostic_resolution':k.tolist(),
                   'imu_coverage':float(np.isfinite(imu_depth).mean()),'sfm_coverage':float(np.isfinite(sfm_depth).mean()),
                   'common_coverage':float(common.mean()),
                   'center_difference':float(np.linalg.norm(imu_wc[:3,3]-refined_wc[:3,3])),
                   'rotation_difference_degrees':float(np.rad2deg(Rotation.from_matrix(imu_wc[:3,:3].T@refined_wc[:3,:3]).magnitude()))}
            difference,arrows=projection_difference(xyz,imu_wc,refined_wc,k,(imu_depth,sfm_depth),image)
            entry['same_point_projection_difference']=difference
            arrow_name=f'{path.stem}_yaw{yaw}_pitch{pitch}_pose_difference.jpg'
            if not cv2.imwrite(str(args.output/arrow_name),arrows): raise RuntimeError(arrow_name)
            entry['pose_difference_image']=arrow_name
            rows=[];compact=[panel(image,f'{path.stem} yaw={yaw} pitch={pitch}')]
            for name,rgb,depth in [('IMU',imu_rgb,imu_depth),('SfM',sfm_rgb,sfm_depth)]:
                metrics,edges=score(image,rgb,depth,common);entry[name]=metrics
                overlay=image.copy();visible=np.isfinite(depth)
                overlay[visible]=cv2.addWeighted(image,0.5,rgb,0.5,0)[visible]
                if metrics['valid']:
                    title=f'{name}: edge={metrics["edge_distance_capped_mean_px"]:.2f}px; <=3px={metrics["edge_fraction_within_3px"]:.1%}'
                else: title=name+': insufficient supported edges'
                compact.append(panel(overlay,title))
                edge_overlay=image.copy();edge_overlay[edges]=(0,255,0)
                rows.append(np.concatenate([panel(image,'Photo'),panel(rgb,name+' cloud (nearest depth)'),panel(overlay,name+' 50% overlay'),panel(edge_overlay,name+' projected color edges')],axis=1))
            preview=args.output/f'{path.stem}_yaw{yaw}_pitch{pitch}.jpg'
            if not cv2.imwrite(str(preview),np.concatenate(rows,axis=0),[cv2.IMWRITE_JPEG_QUALITY,92]):
                raise RuntimeError(preview)
            entry['preview']=preview.name;reports.append(entry)
            if len(comparison)<6: comparison.append(np.concatenate(compact,axis=1))
            timestamp_comparison.append(np.concatenate(compact,axis=1))
            print(f'[{len(reports)}/{total}] {preview.name}: IMU={entry["IMU"].get("edge_distance_capped_mean_px")}, SfM={entry["SfM"].get("edge_distance_capped_mean_px")}',flush=True)
        if args.matched_timestamps and not cv2.imwrite(str(args.output/f'{path.stem}_comparison.jpg'),np.concatenate(timestamp_comparison,axis=0),[cv2.IMWRITE_JPEG_QUALITY,94]):
            raise RuntimeError('Cannot write timestamp comparison')
    valid=[r for r in reports if r['IMU']['valid'] and r['SfM']['valid']]
    summary={name:{key:float(np.median([r[name][key] for r in valid])) for key in
                  ('edge_distance_capped_mean_px','edge_fraction_within_3px','gray_correlation','color_affine_mae_0_255')}
             for name in ('IMU','SfM')}
    report={'point_count':len(xyz),'cloud':str(args.cloud),'sampled_timestamps':[p.stem for p,_ in selected],
            'parameter_sources':{'imu_pose':str(model/'images.txt'),'sfm_pose':str(args.sfm_images),
                                 'intrinsics_for_both':str(model/'cameras.txt'),'json_use':'image names and angle selection only'},
            'random_seed':args.seed if (args.random_count or args.matched_timestamps) else None,
            'sfm_comparison_mode':'direct original image0/image2, same yaw/pitch' if args.matched_timestamps else 'transported image0',
            'sampled_view_count':len(reports),'diagnostic_resolution':args.size,'summary_medians':summary,
            'limitations':['Depth-buffered 3x3 point splats approximate visibility; holes and foreground undersampling remain.',
                          'Color/edge metrics are diagnostic, not known-correspondence reprojection error or absolute accuracy.',
                          'Same-point projection displacement measures differences between two poses, not error relative to a true image correspondence.',
                          'SfM poses are not ground truth. See sfm_comparison_mode for direct vs transported poses.'],
            'views':reports}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if not cv2.imwrite(str(args.output/'comparison.jpg'),np.concatenate(comparison,axis=0),[cv2.IMWRITE_JPEG_QUALITY,94]):
        raise RuntimeError('Cannot write comparison.jpg')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
