import os
import argparse
import numpy as np
import cv2
import json
from joblib import delayed, Parallel
from colmap import read_model, qvec2rotmat
from utils import eval_all_methods, compute_scale_offset_v1, compute_scale_offset_v3

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default='/media/zc/SSD/ar/ABXBJQ_BJ/B3/smallest/')
parser.add_argument("--depth_dir", type=str, default=None)
parser.add_argument("--output", "-o", type=str, default=None)
parser.add_argument("--point-max-error", type=float, default=1.5)
parser.add_argument("--depth_type", type=str, default='da3')
args = parser.parse_args()

if args.depth_dir is None:
    if args.depth_type == 'da2':
        args.depth_dir = os.path.join(args.dataset_dir, "estimated_depths")
    else:
        args.depth_dir = os.path.join(args.dataset_dir, "estimated_depths_v3_0706")
        
if args.output is None:
    if args.depth_type == 'da2':
        args.output = os.path.join(args.dataset_dir, "depth_scales_da2.json")
    else:
        args.output = os.path.join(args.dataset_dir, "depth_scales_da3.json")

sparse_model_dir = os.path.join(args.dataset_dir, "sparse")
if not os.path.exists(os.path.join(sparse_model_dir, "images.bin")):
    sparse_model_dir = os.path.join(sparse_model_dir, "0")

cameras, images, points3d = read_model(sparse_model_dir)

pts_indices = np.array([points3d[key].id for key in points3d])
pts_xyzs = np.array([points3d[key].xyz for key in points3d])
pts_errors = np.array([points3d[key].error for key in points3d])

points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
points3d_error_ordered = np.zeros([pts_indices.max() + 1, ])
points3d_ordered[pts_indices] = pts_xyzs
points3d_error_ordered[pts_indices] = pts_errors

# 主线程汇总所有图片各算法指标
all_image_metrics = []

def get_scales_da2(key, cameras, images, points3d_ordered, points3d_error_ordered, args):
    image_meta = images[key]
    image_name = image_meta.name
    directory_name = image_name.split('/')[0]
    if directory_name != 'image3':
        return {"image_name": image_name, "scale": None, "offset": None, "metric_dict": None}
    
    cam_intrinsic = cameras[image_meta.camera_id]
    depth_name = image_name+'.npy'
    depth_path = os.path.join(args.depth_dir, depth_name)
    if not os.path.exists(depth_path):
        return {"image_name": image_name, "scale": None, "offset": None, "metric_dict": None}

    pts_idx = images[key].point3D_ids
    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    pts_errors = points3d_error_ordered[pts_idx]
    valid_errors = pts_errors < args.point_max_error
    pts_idx = pts_idx[valid_errors]
    valid_xys = valid_xys[valid_errors]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec
    invcolmapdepth = 1. / pts[..., 2]
    invmonodepthmap = np.load(depth_path)

    if invmonodepthmap is None:
        return {"image_name": image_name, "scale": None, "offset": None, "metric_dict": None}

    s = invmonodepthmap.shape[0] / cam_intrinsic.height
    maps = (valid_xys * s).astype(np.float32)
    valid = (
            (maps[..., 0] >= 0) &
            (maps[..., 1] >= 0) &
            (maps[..., 0] < cam_intrinsic.width * s - 1e-6) &
            (maps[..., 1] < cam_intrinsic.height * s - 1e-6) & (invcolmapdepth > 1e-6))

    scale = 0
    offset = 0
    invmonodepth = None
    invcolmap_valid = None
    metric_dict = None

    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmap_valid = invcolmapdepth[valid]
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]

        p5_r, p95_r = np.percentile(invcolmap_valid, [5, 95])
        p5_m, p95_m = np.percentile(invmonodepth, [5, 95])
        robust_mask = (invcolmap_valid > p5_r) & (invcolmap_valid < p95_r) & (invmonodepth > p5_m) & (invmonodepth < p95_m)
        invcolmap_valid = invcolmap_valid[robust_mask]
        invmonodepth = invmonodepth[robust_mask]

        if len(invcolmap_valid) > 10:
            metric_dict = eval_all_methods(invcolmap_valid, invmonodepth)
            # 默认用v1 MAD输出json
            scale, offset = compute_scale_offset_v3(invcolmap_valid, invmonodepth)
            
            from test import evaluate_sparse_reprojection, build_world_map, select
            world_pred_v1, gt_world = evaluate_sparse_reprojection(
                image_meta = image_meta,
                camera = cam_intrinsic,
                points3d_ordered = points3d_ordered,
                point3d_error_ordered = points3d_error_ordered,
                depth_map = invmonodepthmap,
                scale = scale,
                offset = offset,
                inverse_depth = True,
            )
            
          
            world_points = build_world_map(
                image_meta = image_meta,
                camera = cam_intrinsic,
                depth_map = invmonodepthmap,
                scale = scale,
                offset = offset,
                save_dir='/media/zc/SSD/ar/ABXBJQ_BJ/B3/smallest/da2_all_depth/',
                inverse_depth = True,
            )

            world_pred_v2 = select(image_meta,
                    points3d_ordered,
                    points3d_error_ordered,
                    world_points,
                    point_max_error = 1.5,           
            )

            err = np.linalg.norm(world_pred_v1-world_pred_v2,axis=1)
            rmse = np.sqrt(np.mean(err**2))
            print("RMSE      :",rmse)

    return {"image_name": image_meta.name, "scale": scale, "offset": offset, "metric_dict": metric_dict}

def get_scales_da3(key, cameras, images, points3d_ordered, points3d_error_ordered, args):
    image_meta = images[key]
    image_name = image_meta.name
    directory_name = image_name.split('/')[0]
    if directory_name != 'image3':
        return {"image_name": image_name, "scale": None, "offset": None, "metric_dict": None}
    
    cam_intrinsic = cameras[image_meta.camera_id]
    depth_name = image_name+'.npy'
    depth_path = os.path.join(args.depth_dir, depth_name)
    if not os.path.exists(depth_path):
        return {"image_name": image_name, "scale": None, "offset": None, "metric_dict": None}

    pts_idx = images[key].point3D_ids
    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    pts_errors = points3d_error_ordered[pts_idx]
    valid_errors = pts_errors < args.point_max_error
    pts_idx = pts_idx[valid_errors]
    valid_xys = valid_xys[valid_errors]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    pts_cam = np.dot(pts, R.T) + image_meta.tvec

    colmapdepth = pts_cam[..., 2]
    monodepthmap = np.load(depth_path)
    
    # DA3转逆深度对齐
    # colmapdepth = 1 / (colmapdepth + 1e-8)
    # monodepthmap = 1 / (monodepthmap + 1e-8) 
    
    if monodepthmap is None:
        return {"image_name": image_name, "scale": None, "offset": None, "metric_dict": None}

    s = monodepthmap.shape[0] / cam_intrinsic.height
    maps = (valid_xys * s).astype(np.float32)
    valid = (
            (maps[..., 0] >= 0) &
            (maps[..., 1] >= 0) &
            (maps[..., 0] < cam_intrinsic.width * s - 1e-6) &
            (maps[..., 1] < cam_intrinsic.height * s - 1e-6) & (colmapdepth > 1e-6))

    # convert monodepthmap to world 

    scale = 0
    offset = 0
    mono_samp = None
    colmap_samp = None
    metric_dict = None

    if valid.sum() > 10 and (colmapdepth.max() - colmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        colmap_samp = colmapdepth[valid]
        mono_samp = cv2.remap(monodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        p5_r, p95_r = np.percentile(colmap_samp, [5, 95])
        p5_m, p95_m = np.percentile(mono_samp, [5, 95])
        robust_mask = (colmap_samp > p5_r) & (colmap_samp < p95_r) & (mono_samp > p5_m) & (mono_samp < p95_m)
        colmap_samp = colmap_samp[robust_mask]
        mono_samp = mono_samp[robust_mask]

        if len(colmap_samp) > 10:
            metric_dict = eval_all_methods(colmap_samp, mono_samp)
            # json默认输出v1 MAD结果
            scale, offset = compute_scale_offset_v3(colmap_samp, mono_samp)

            # valid scale and offset, compute feature point points and transform to world coordinate 
            from test import evaluate_sparse_reprojection, build_world_map, select
            world_pred_v1, gt_world = evaluate_sparse_reprojection(
                image_meta = image_meta,
                camera = cam_intrinsic,
                points3d_ordered = points3d_ordered,
                point3d_error_ordered = points3d_error_ordered,
                depth_map = monodepthmap,
                scale = scale,
                offset = offset,
            )

            world_points = build_world_map(
                image_meta = image_meta,
                camera = cam_intrinsic,
                depth_map = monodepthmap,
                scale = scale,
                offset = offset,
                save_dir='/media/zc/SSD/ar/ABXBJQ_BJ/B3/smallest/da3_all_depth/'
            )

            world_pred_v2 = select(image_meta,
                    points3d_ordered,
                    points3d_error_ordered,
                    world_points,
                    point_max_error = 1.5,           
            )

            err = np.linalg.norm(world_pred_v1-world_pred_v2,axis=1)
            rmse = np.sqrt(np.mean(err**2))
            print("RMSE      :",rmse)


    return {"image_name": image_meta.name, "scale": scale, "offset": offset, "metric_dict": metric_dict}

# Parallel run
if args.depth_type =='da2':
    # depth_param_list = Parallel(n_jobs=-1, backend="threading")(
    #     delayed(get_scales_da2)(key, cameras, images, points3d_ordered, points3d_error_ordered, args) for key in images
    # )
    for key in images:
        get_scales_da2(key, cameras, images, points3d_ordered, points3d_error_ordered, args)

else:
    # depth_param_list = Parallel(n_jobs=-1, backend="threading")(
    #     delayed(get_scales_da3)(key, cameras, images, points3d_ordered, points3d_error_ordered, args) for key in images
    # )
    for key in images:
        get_scales_da3(key, cameras, images, points3d_ordered, points3d_error_ordered, args)
# Save scale offset json
depth_param_list = None
depth_params = {}
for item in depth_param_list:
    img_name = item["image_name"]
    depth_params[img_name] = {"scale": item["scale"], "offset": item["offset"]}
    # 收集有效指标
    if item["metric_dict"] is not None:
        all_image_metrics.append({
            "img": img_name,
            "metrics": item["metric_dict"]
        })
        # 单张图片打印全部6种算法
        print("="*70)
        print(f"Image: {img_name}")
        for method_name, data in item["metric_dict"].items():
            print(f"{method_name:10s} | scale={data['scale']:6.3f} offset={data['offset']:6.3f} | corr={data['corr']:.3f} rmse={data['rmse']:.3f} cos={data['cos']:.3f}")

with open(args.output, "w") as f:
    json.dump(depth_params, f, indent=4, ensure_ascii=False)
print(f"\nScale & Offset Saved to `{args.output}`")

# -------------------------- Compute global average for every method --------------------------
method_names = ["v1 MAD","v2 MeanStd","v3 LS","v4 Huber-LS","v5 L1-Reg","v6 RANSAC-LS"]
storage = {name:{"corr":[], "rmse":[], "cos":[]} for name in method_names}

# 聚合每张图各算法指标
for img_info in all_image_metrics:
    met = img_info["metrics"]
    for m_name in method_names:
        d = met[m_name]
        storage[m_name]["corr"].append(d["corr"])
        storage[m_name]["rmse"].append(d["rmse"])
        storage[m_name]["cos"].append(d["cos"])

# 打印每种算法全局均值、有效图片数量
print("\n" + "="*80)
print(f"Global Average Metrics | Total valid images: {len(all_image_metrics)}")
print("="*80)
for m_name in method_names:
    corr_list = storage[m_name]["corr"]
    rmse_list = storage[m_name]["rmse"]
    cos_list = storage[m_name]["cos"]
    cnt = len(corr_list)
    if cnt == 0:
        print(f"{m_name:10s} | No valid data")
        continue
    avg_corr = np.mean(corr_list)
    avg_rmse = np.mean(rmse_list)
    avg_cos = np.mean(cos_list)
    print(f"{m_name:10s} | AvgCorr={avg_corr:.4f} AvgRMSE={avg_rmse:.4f} AvgCos={avg_cos:.4f} | ValidImgNum={cnt}")