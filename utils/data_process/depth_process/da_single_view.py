import os
import cv2
import json
import numpy as np
from PIL import Image
import open3d as o3d
from tqdm import tqdm
from da_op3d_depth2pc import load_colmap_data
# ===================== Config =====================
PROJECT_DIR = "/media/zc/SSD/ar/ABXBJQ_BJ/B3/smallest"
FOCAL_X = 1023
FOCAL_Y = 1023
MAX_DEPTH_V2 = 20
MAX_PROCESS = 1  # 最多处理图片数量
IMG_EXTS = {"jpg", "JPG", "jpeg", "JPEG", "png", "PNG"}


def get_ply_filename(rel_img_path: str) :
    """subdir/xxx.png -> subdir_xxx.ply，兼容所有图片后缀"""
    sanitized = rel_img_path.replace("/", "_")
    stem, _ = os.path.splitext(sanitized)
    return f"{stem}.ply"

def collect_all_images(base_img_dir: str) :
    """遍历images下所有子文件夹，返回图片相对路径列表 [subdir/xxx.png]"""
    rel_img_paths: List[str] = []
    if not os.path.isdir(base_img_dir):
        raise FileNotFoundError(f"Image folder not found: {base_img_dir}")

    for sub_dir_name in os.listdir(base_img_dir):
        sub_dir_full = os.path.join(base_img_dir, sub_dir_name)
        if not os.path.isdir(sub_dir_full):
            continue
        for img_file in os.listdir(sub_dir_full):
            ext = img_file.split(".")[-1]
            if ext in IMG_EXTS:
                rel_img_paths.append(os.path.join(sub_dir_name, img_file))

    if not rel_img_paths:
        raise RuntimeError("No valid images found in images directory")
    return sorted(rel_img_paths)


# def generate_ply(rgb_pil: Image.Image, raw_depth: np.ndarray, save_path: str, fx: float, fy: float, reverse_depth: bool = True):
#     """从原图+深度图生成彩色点云PLY文件"""
#     w, h = rgb_pil.size
#     depth_resized = np.array(Image.fromarray(raw_depth).resize((w, h), Image.NEAREST))

#     # 像素网格转相机空间三维点
#     y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
#     x = (x_grid - w / 2) / fx
#     y = (y_grid - h / 2) / fy
#     z = depth_resized

#     pts = np.stack([x * z, y * z, z], axis=-1).reshape(-1, 3)
#     rgb = np.array(rgb_pil).reshape(-1, 3) / 255.0

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pts)
#     pcd.colors = o3d.utility.Vector3dVector(rgb)
#     o3d.io.write_point_cloud(save_path, pcd)

def generate_ply(rgb_pil, raw_depth, save_path, fx, fy, T_wc):
    w, h = rgb_pil.size
    depth_resized = np.array(Image.fromarray(raw_depth).resize((w, h), Image.NEAREST))

    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    x = (x_grid - w / 2) / fx
    y = (y_grid - h / 2) / fy
    z = depth_resized
    pts = np.stack([x*z, y*z, z], axis=-1).reshape(-1, 3)

    if T_wc is not None:
        N = pts.shape[0]
        # 齐次坐标：(N,4)
        pts_cam_homo = np.hstack([pts, np.ones((N, 1))])
        # T_cw @ P_cam.T -> (4,N)，转置回 (N,3)
        pts_world_homo = (T_wc @ pts_cam_homo.T).T
        pts_world = pts_world_homo[:, :3]
        pts = pts_world
    else:
        pts = pts
    
    rgb = np.array(rgb_pil).reshape(-1, 3) / 255.0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(save_path, pcd)


# ===================== Main Process =====================
def process_da2():
    IMAGE_DIR = os.path.join(PROJECT_DIR, "images")
    DEPTH_NPY_ROOT = os.path.join(PROJECT_DIR, "estimated_depths")
    PLY_SAVE_ROOT = os.path.join(PROJECT_DIR, "da2_ply")
    SCALE_JSON_PATH = os.path.join(PROJECT_DIR, "estimated_depth_scales.json")
    os.makedirs(PLY_SAVE_ROOT, exist_ok=True)

    with open(SCALE_JSON_PATH, "r", encoding="utf-8") as f:
        depth_scales: dict = json.load(f)

    img_rel_list = collect_all_images(IMAGE_DIR)

    processed_cnt = 0
    for img_rel in tqdm(img_rel_list, desc="Generating point cloud"):
        if processed_cnt >= MAX_PROCESS:
            break

        print(f"\nProcessing: {img_rel}")
        full_img_path = os.path.join(IMAGE_DIR, img_rel)
        full_npy_path = os.path.join(DEPTH_NPY_ROOT, f"{img_rel}.npy")
        ply_filename = get_ply_filename(img_rel)
        full_ply_path = os.path.join(PLY_SAVE_ROOT, ply_filename)

        img_bgr = cv2.imread(full_img_path)
        img_rgb_pil = Image.open(full_img_path).convert("RGB")

        scale_info = depth_scales.get(img_rel, {"scale": 1.0, "offset": 0.0})
        raw_norm_depth = np.load(full_npy_path)
        real_depth = raw_norm_depth * scale_info["scale"] + scale_info["offset"]

        generate_ply(img_rgb_pil, real_depth, full_ply_path, FOCAL_X, FOCAL_Y)
        processed_cnt += 1

    print(f"\nDone! Total processed images: {processed_cnt}")
    print(f"PLY point clouds saved at: {PLY_SAVE_ROOT}")

def process_da3():
    IMAGE_DIR = os.path.join(PROJECT_DIR, "images")
    DEPTH_NPY_ROOT = os.path.join(PROJECT_DIR, "depth_method_compare/da2_vs_da3_output/da3_depth_npy")
    PLY_SAVE_ROOT = os.path.join(PROJECT_DIR, "da3_ply")
    SCALE_JSON_PATH = os.path.join(PROJECT_DIR, "depth_scales_da3.json")
    
    os.makedirs(PLY_SAVE_ROOT, exist_ok=True)
    depth_scales = None
    if os.path.exists(SCALE_JSON_PATH):
        with open(SCALE_JSON_PATH, "r", encoding="utf-8") as f:
            depth_scales: dict = json.load(f)

    img_rel_list = collect_all_images(IMAGE_DIR)
    colmap_path = os.path.join(PROJECT_DIR, 'sparse/0') 
    cameras, images = load_colmap_data(colmap_path)

    processed_cnt = 0
    for img_rel in tqdm(img_rel_list, desc="Generating world-space point cloud"):
        if processed_cnt >= MAX_PROCESS:
            break
        
        print(f"\nProcessing: {img_rel}")
        image_data = images[img_rel]
        dir_name = img_rel.split('/')[0]
        if dir_name != 'image3':
            continue 

        extrinsic = image_data['extrinsic']  # 4x4 T_wc: world -> camera
        
        # 修复拼写错误：np.linalg.inv
        T_wc = np.linalg.inv(extrinsic)      # 4x4 T_cw: camera -> world

        full_img_path = os.path.join(IMAGE_DIR, img_rel)
        pure_name = img_rel.split('/')[-1].split('.')[0]
        full_npy_path = os.path.join(DEPTH_NPY_ROOT, f"{pure_name}.npy")
        ply_filename = get_ply_filename(img_rel)
        full_ply_path = os.path.join(PLY_SAVE_ROOT, ply_filename)

        img_bgr = cv2.imread(full_img_path)
        img_rgb_pil = Image.open(full_img_path).convert("RGB")
        
        if depth_scales is not None:
            scale_info = depth_scales.get(img_rel, {"scale": 1.0, "offset": 0.0})
        else:
            scale_info = {"scale": 1.0, "offset": 0.0}
        
        raw_norm_depth = np.load(full_npy_path)
        raw_norm_depth = raw_norm_depth * scale_info["scale"] + scale_info["offset"]

        # 传入相机到世界矩阵 T_cw，函数内部自动转世界坐标
        generate_ply(img_rgb_pil, raw_norm_depth, full_ply_path, FOCAL_X, FOCAL_Y, T_wc=None)
        processed_cnt += 1


    print(f"\nDone! Total processed images: {processed_cnt}")
    print(f"PLY point clouds saved at: {PLY_SAVE_ROOT}")


if __name__ == "__main__":

    # process_da2()
    process_da3()