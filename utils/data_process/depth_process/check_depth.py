import os
import argparse
import numpy as np
import cv2
import json
from colmap import *

# -------------------------- PLY Save Function (Add RGB Color) --------------------------
def save_ply(points, colors, save_path):
    """
    points: Nx3 float array, world coordinate
    colors: Nx3 uint8 array, RGB 0~255
    """
    N = points.shape[0]
    with open(save_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")

# -------------------------- Unproject Depth + Get Valid RGB --------------------------
def depth2camera_with_color(cam, depth_map, rgb_img, scale, offset):
    """
    cam: Camera namedtuple
    depth_map: H,W normalized DA3 depth
    rgb_img: H,W,3 BGR image from cv2.imread
    scale, offset: v3 LS fit, real_z = scale * norm_z + offset
    return cam_pts(N,3), valid_rgb(N,3), valid_mask(H,W)
    """
    h, w = depth_map.shape
    fx, fy, cx, cy = cam.params[:4]
    # Pixel grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    # Real metric depth
    z_real = scale * depth_map + offset
    
    # z_real = scale * depth_map + offset
    # z_real = 1 / z_real

    mask = z_real > 0.01  # filter invalid small depth

    # Camera space XYZ
    X = (u - cx) / fx * z_real
    Y = (v - cy) / fy * z_real
    Z = z_real
    
    cam_pts = np.stack([X,Y,Z], axis=-1)
    cam_pts = cam_pts[mask]

    # Extract corresponding RGB, convert BGR to RGB
    rgb_img_rgb = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    valid_rgb = rgb_img_rgb[mask]
    return cam_pts, valid_rgb, mask

def cam2world(points_cam, qvec, tvec):
    R = qvec2rotmat(qvec) # camera to world 
    pts_world = (R.T @ (points_cam - tvec[None, :]).T).T

    # T_cw = np.eye(4)
    # T_cw[:3, :3] = R
    # T_cw[:3, 3] = tvec
    # T_wc = np.linalg.inv(T_cw)
    # pts_world = (T_wc[:3, :3] @ points_cam.T ).T + T_wc[:3, 3]

    return pts_world

# -------------------------- Main Entry --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/media/zc/SSD/ar/AIBEE_16/L16/showroom/")
    parser.add_argument("--image_dir", type=str, default="images") 
    parser.add_argument("--depth_dir", type=str, default="estimated_depths_v3_0706")
    parser.add_argument("--scale_json", type=str, default="depth_scales_da3.json")
    parser.add_argument("--ply_out_dir", type=str, default="da3_ply_output")
    args = parser.parse_args()


    # 1. Load COLMAP bin
    sparse_folder = os.path.join(args.dataset_dir, "sparse", "0")
    cam_bin = os.path.join(sparse_folder, "cameras.bin")
    img_bin = os.path.join(sparse_folder, "images.bin")
    cameras = read_cameras_binary(cam_bin)
    images_dict = read_images_binary(img_bin)


    ply_out_dir = os.path.join(args.dataset_dir, args.ply_out_dir)
    raw_img_root = os.path.join(args.dataset_dir, args.image_dir)
    depth_dir = os.path.join(args.dataset_dir, args.depth_dir)
    scale_json = os.path.join(args.dataset_dir, args.scale_json)

    # 2. Load precomputed scale offset (v3 LS result)
    with open(scale_json, "r") as f:
        scale_offset_dict = json.load(f)

    # 3. Output ply folder
    os.makedirs(ply_out_dir, exist_ok=True)

    # 4. Iterate every image
    count = 0
    for img_id, img_meta in images_dict.items():
        img_name = img_meta.name
        if not img_name.startswith("image3/"):
            continue
        # Load scale offset
        count = count + 1
        if count > 10:
            break
        if img_name not in scale_offset_dict:
            print(f"Skip {img_name}: no scale/offset record")
            continue
        scale = scale_offset_dict[img_name]["scale"]
        offset = scale_offset_dict[img_name]["offset"]
        if scale is None or scale <= 0:
            print(f"Skip {img_name}: invalid scale")
            continue

        # Path prepare
        stem = img_name.replace("/", "_").split(".")[0]
        img_filename = img_name.split("/")[-1]
        raw_img_path = os.path.join(raw_img_root, img_name)
        depth_npy_path = os.path.join(depth_dir, f"{img_name}.npy")

        # Check image & depth exist
        if not os.path.exists(raw_img_path):
            print(f"Skip {img_name}: missing raw rgb image {raw_img_path}")
            continue
        if not os.path.exists(depth_npy_path):
            print(f"Skip {img_name}: missing depth file")
            continue

        # Load depth
        depth_norm = np.load(depth_npy_path).astype(np.float32)
        if depth_norm.ndim == 3:
            depth_norm = depth_norm[...,0]

        # Load RGB image
        rgb_img = cv2.imread(raw_img_path)
        if rgb_img is None:
            print(f"Skip {img_name}: fail to read rgb image")
            continue
        h_depth, w_depth = depth_norm.shape
        h_img, w_img = rgb_img.shape[:2]
        if (h_depth != h_img) or (w_depth != w_img):
            rgb_img = cv2.resize(rgb_img, (w_depth, h_depth), interpolation=cv2.INTER_LINEAR)

        # Camera intrinsic
        cam = cameras[img_meta.camera_id]
        # Unproject to camera coordinate + get valid color
        points, valid_rgb, valid_mask = depth2camera_with_color(cam, depth_norm, rgb_img, scale, offset)
        if len(points) < 100:
            print(f"Skip {img_name}: too few valid points")
            continue
        # Transform camera points to world using COLMAP pose
        points = cam2world(points, img_meta.qvec, img_meta.tvec) # world points 

        # Save colored ply
        ply_save_path = os.path.join(ply_out_dir, f"{stem}.ply")
        save_ply(points, valid_rgb, ply_save_path)
        print(f"Saved {ply_save_path}, point num: {len(points)}")


