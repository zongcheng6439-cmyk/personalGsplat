import os
import argparse
import numpy as np
import cv2
import json
from colmap import read_model, qvec2rotmat
from utils import eval_all_methods, compute_scale_offset_v1, compute_scale_offset_v3
# Move external import to top (avoid repeated import inside loop)
from test import evaluate_sparse_reprojection, build_world_map, select

# -------------------------- Global Constants --------------------------
PERCENTILE_LOW = 5
PERCENTILE_HIGH = 95
MIN_VALID_SAMPLES = 10
DEPTH_EPS = 1e-6
DIFF_EPS = 1e-3
BOUND_EPS = 1e-6

# -------------------------- Helper Functions --------------------------
def collect_image_metric(metrics_dict, img_name, all_metric_storage):
    """Collect single image all fitting method metrics"""
    all_metric_storage.append({
        "img": img_name,
        "metrics": metrics_dict
    })
    print("=" * 70)
    print(f"Image: {img_name}")
    for method_name, data in metrics_dict.items():
        line = (f"{method_name:10s} | scale={data['scale']:6.3f} offset={data['offset']:6.3f} "
                f"| corr={data['corr']:.3f} rmse={data['rmse']:.3f} cos={data['cos']:.3f}")
        print(line)

def calc_global_avg_metrics(all_image_metrics):
    """Aggregate all images & compute global average for each fitting method"""
    method_names = ["v1 MAD", "v2 MeanStd", "v3 LS", "v4 Huber-LS", "v5 L1-Reg", "v6 RANSAC-LS"]
    storage = {name: {"corr": [], "rmse": [], "cos": []} for name in method_names}

    for img_info in all_image_metrics:
        met = img_info["metrics"]
        for m_name in method_names:
            d = met[m_name]
            storage[m_name]["corr"].append(d["corr"])
            storage[m_name]["rmse"].append(d["rmse"])
            storage[m_name]["cos"].append(d["cos"])
    return method_names, storage

def print_global_summary(method_names, storage, total_valid_img):
    """Pretty print global average metric table"""
    print("\n" + "=" * 80)
    print(f"Global Average Metrics | Total valid images: {total_valid_img}")
    print("=" * 80)
    for m_name in method_names:
        corr_list = storage[m_name]["corr"]
        rmse_list = storage[m_name]["rmse"]
        cos_list = storage[m_name]["cos"]
        count = len(corr_list)
        if count == 0:
            print(f"{m_name:10s} | No valid data")
            continue
        avg_corr = np.mean(corr_list)
        avg_rmse = np.mean(rmse_list)
        avg_cos = np.mean(cos_list)
        print(f"{m_name:10s} | AvgCorr={avg_corr:.4f} AvgRMSE={avg_rmse:.4f} AvgCos={avg_cos:.4f} | ValidImgNum={count}")

def filter_valid_sparse_samples(colmap_vals, mono_vals):
    """5%-95 percentile robust filter for sparse matching samples"""
    p5_r, p95_r = np.percentile(colmap_vals, [PERCENTILE_LOW, PERCENTILE_HIGH])
    p5_m, p95_m = np.percentile(mono_vals, [PERCENTILE_LOW, PERCENTILE_HIGH])
    robust_mask = (colmap_vals > p5_r) & (colmap_vals < p95_r) & (mono_vals > p5_m) & (mono_vals < p95_m)
    return colmap_vals[robust_mask], mono_vals[robust_mask]

def run_depth_evaluation(image_meta, cam_intrinsic, points3d_ordered, points3d_error_ordered,
                         depth_map, scale, offset, inverse_depth, save_dir):
    """Unified sparse & dense depth evaluation + RMSE print"""
    # Sparse reprojection error
    world_pred_v1, gt_world = evaluate_sparse_reprojection(
        image_meta=image_meta,
        camera=cam_intrinsic,
        points3d_ordered=points3d_ordered,
        point3d_error_ordered=points3d_error_ordered,
        depth_map=depth_map,
        scale=scale,
        offset=offset,
        inverse_depth=inverse_depth,
    )
    # Full dense depth to world point cloud
    world_points = build_world_map(
        image_meta=image_meta,
        camera=cam_intrinsic,
        depth_map=depth_map,
        scale=scale,
        offset=offset,
        save_dir=save_dir,
        inverse_depth=inverse_depth,
    )
    # Sample dense points at sparse keypoint positions
    world_pred_v2 = select(
        image_meta,
        points3d_ordered,
        points3d_error_ordered,
        world_points,
        point_max_error=1.5,
    )
    # Compute point-wise RMSE between two projections
    err = np.linalg.norm(world_pred_v1 - world_pred_v2, axis=1)
    rmse = np.sqrt(np.mean(err ** 2))
    print(f"Sparse-Dense Matching RMSE: {rmse:.6f}")
    return rmse

# -------------------------- Core DA Processing Wrapper --------------------------
def process_single_image(image_meta, cameras, points3d_ordered, points3d_error_ordered,
                         args, inverse_depth: bool, output_save_dir: str):
    img_name = image_meta.name
    dir_name = img_name.split('/')[0]
    # Skip non-image3 folder
    if dir_name != "image3":
        return {"image_name": img_name, "scale": None, "offset": None, "metric_dict": None}

    cam_intrinsic = cameras[image_meta.camera_id]
    depth_file = f"{img_name}.npy"
    depth_full_path = os.path.join(args.depth_dir, depth_file)
    if not os.path.exists(depth_full_path):
        return {"image_name": img_name, "scale": None, "offset": None, "metric_dict": None}

    # Filter valid COLMAP sparse 3D points
    point_ids = image_meta.point3D_ids
    mask_valid_id = (point_ids >= 0) & (point_ids < len(points3d_ordered))
    point_ids = point_ids[mask_valid_id]
    pixel_xys = image_meta.xys[mask_valid_id]

    # Filter points under max reprojection error threshold
    point_errs = points3d_error_ordered[point_ids]
    mask_good_point = point_errs < args.point_max_error
    point_ids = point_ids[mask_good_point]
    pixel_xys = pixel_xys[mask_good_point]

    # Load depth map
    mono_depth_raw = np.load(depth_full_path).astype(np.float32)
    h_depth, w_depth = mono_depth_raw.shape[:2]
    img_h, img_w = cam_intrinsic.height, cam_intrinsic.width
    scale_uv = h_depth / img_h

    # Sample mono depth at sparse keypoint pixels
    sample_uv = (pixel_xys * scale_uv).astype(np.float32)
    uv_mask = (sample_uv[:, 0] >= 0) & (sample_uv[:, 1] >= 0) \
              & (sample_uv[:, 0] < img_w * scale_uv - BOUND_EPS) \
              & (sample_uv[:, 1] < img_h * scale_uv - BOUND_EPS)

    sample_uv = sample_uv[uv_mask, :]
    # Get colmap camera Z value
    world_pts = points3d_ordered[point_ids][uv_mask]
    R = qvec2rotmat(image_meta.qvec)
    cam_space_pts = np.dot(world_pts, R.T) + image_meta.tvec

    if inverse_depth:
        colmap_vals = 1.0 / (cam_space_pts[..., 2] + DEPTH_EPS)
    else:
        colmap_vals = cam_space_pts[..., 2]

    mono_sampled = cv2.remap(
        mono_depth_raw,
        sample_uv[:, 0],
        sample_uv[:, 1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )[:, 0]

    scale_out = 0.0
    offset_out = 0.0
    metric_result = None

    # Only run fitting when enough valid samples & depth dynamic range exists
    valid_sample_count = np.sum(uv_mask)
    depth_range = colmap_vals.max() - colmap_vals.min()
    if valid_sample_count > MIN_VALID_SAMPLES and depth_range > DIFF_EPS:
        colmap_clean, mono_clean = filter_valid_sparse_samples(colmap_vals, mono_sampled)
        if len(colmap_clean) > MIN_VALID_SAMPLES:
            metric_result = eval_all_methods(colmap_clean, mono_clean)
            # Main fitting method: v3 LS
            scale_out, offset_out = compute_scale_offset_v3(colmap_clean, mono_clean)
            # Evaluate sparse vs dense depth projection error
            run_depth_evaluation(
                image_meta=image_meta,
                cam_intrinsic=cam_intrinsic,
                points3d_ordered=points3d_ordered,
                points3d_error_ordered=points3d_error_ordered,
                depth_map=mono_depth_raw,
                scale=scale_out,
                offset=offset_out,
                inverse_depth=inverse_depth,
                save_dir=output_save_dir
            )
    return {
        "image_name": img_name,
        "scale": scale_out,
        "offset": offset_out,
        "metric_dict": metric_result
    }

# -------------------------- Main Program Entry --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default='/media/zc/SSD/ar/ABXBJQ_BJ/B3/smallest/')
    parser.add_argument("--depth_dir", type=str, default=None)
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--point-max-error", type=float, default=1.5)
    parser.add_argument("--depth_type", type=str, default='da3', choices=["da2", "da3"])
    args = parser.parse_args()

    # Auto fill depth folder path
    if args.depth_dir is None:
        if args.depth_type == "da2":
            args.depth_dir = os.path.join(args.dataset_dir, "estimated_depths")
            save_dense_dir = os.path.join(args.dataset_dir, "da2_all_depth")
            use_inverse_depth = True
        else:
            args.depth_dir = os.path.join(args.dataset_dir, "estimated_depths_v3_0706")
            save_dense_dir = os.path.join(args.dataset_dir, "da3_all_depth")
            use_inverse_depth = False
    os.makedirs(save_dense_dir, exist_ok=True)

    # Auto fill output json path
    if args.output is None:
        if args.depth_type == "da2":
            args.output = os.path.join(args.dataset_dir, "depth_scales_da2.json")
        else:
            args.output = os.path.join(args.dataset_dir, "depth_scales_da3.json")

    # Load COLMAP sparse reconstruction
    sparse_root = os.path.join(args.dataset_dir, "sparse")
    if not os.path.exists(os.path.join(sparse_root, "images.bin")):
        sparse_root = os.path.join(sparse_root, "0")
    cameras, images, points3d = read_model(sparse_root)

    # Preprocess ordered 3D points array
    pts_ids = np.array([points3d[key].id for key in points3d])
    pts_xyz = np.array([points3d[key].xyz for key in points3d])
    pts_err = np.array([points3d[key].error for key in points3d])
    max_pt_id = pts_ids.max()
    points3d_ordered = np.zeros((max_pt_id + 1, 3), dtype=np.float32)
    points3d_error_ordered = np.zeros(max_pt_id + 1, dtype=np.float32)
    points3d_ordered[pts_ids] = pts_xyz
    points3d_error_ordered[pts_ids] = pts_err

    # Global storage
    all_image_metrics = []
    depth_param_output_list = []

    # Process all images sequentially
    for img_key in images:
        img_meta = images[img_key]
        res = process_single_image(
            image_meta=img_meta,
            cameras=cameras,
            points3d_ordered=points3d_ordered,
            points3d_error_ordered=points3d_error_ordered,
            args=args,
            inverse_depth=use_inverse_depth,
            output_save_dir=save_dense_dir
        )
        depth_param_output_list.append(res)
        # Collect fitting metrics if available
        if res["metric_dict"] is not None:
            collect_image_metric(res["metric_dict"], res["image_name"], all_image_metrics)

    # Save scale & offset json file
    json_export_dict = {}
    for item in depth_param_output_list:
        img_n = item["image_name"]
        json_export_dict[img_n] = {
            "scale": item["scale"],
            "offset": item["offset"]
        }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(json_export_dict, f, indent=4, ensure_ascii=False)
    print(f"\nSuccessfully saved scale & offset file: {args.output}")

    # Calculate & print global average metrics across all valid images
    method_name_list, metric_storage = calc_global_avg_metrics(all_image_metrics)
    print_global_summary(method_name_list, metric_storage, len(all_image_metrics))