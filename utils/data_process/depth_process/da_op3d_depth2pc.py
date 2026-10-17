import os
import json 
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from tqdm import tqdm
import struct

def warpped_depth(depth):
    return torch.where(depth < 10.0, depth / 10.0, 2.0 - 10 / depth) / 2.0

def warpped_depth_inv(depth):
    return torch.where(depth < 0.5, 2 * depth * 10.0, 10 / (1.0 - depth) / 2)

def read_colmap_cameras_text(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(list(map(float, elems[4:])))
            
            model_id = 1 if model == "PINHOLE" else 0
            cameras[camera_id] = {
                "model": model_id, "width": width, "height": height, "params": params
            }
    return cameras

def read_colmap_images_text(path):
    """读取 images.txt"""
    images = {}
    with open(path, "r") as f:
        lines = f.readlines()
        # images.txt 中每两行代表一个图像，第一行是外参，第二行是 2D 点
        for i in range(0, len(lines)):
            if i % 2 !=0 :
                continue
            line = lines[i].strip()
            if not line or line.startswith("#"):
                continue
            
            elems = line.split()
            image_id = int(elems[0])
            qvec = np.array(list(map(float, elems[1:5]))) # w, x, y, z
            tvec = np.array(list(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]
            
            #  (World-to-Camera)
            R = o3d.geometry.get_rotation_matrix_from_quaternion(qvec)
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = tvec
            
            images[image_name] = {
                "extrinsic": extrinsic, "camera_id": camera_id
            }            
            next(iter(range(1)), None) 
    return images

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):

    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_colmap_cameras(path):
    """读取 cameras.bin"""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = 4 # 假设是 PINHOLE 或 OPENCV 模型
            if model_id == 0: # SIMPLE_PINHOLE
                num_params = 3
            elif model_id == 1: # PINHOLE
                num_params = 4
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = {
                "model": model_id, "width": width, "height": height, "params": params
            }
    return cameras

def read_colmap_images(path):
    """读取 images.bin 获取外参"""
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5]) # w, x, y, z
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(24 * num_points2D) # 跳过 2D 点数据
            
            # 将四元数转为旋转矩阵
            R = o3d.geometry.get_rotation_matrix_from_quaternion(qvec)
            # 构建 World-to-Camera 矩阵 (Extrinsic)
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = tvec
            
            images[image_name] = {
                "extrinsic": extrinsic, "camera_id": camera_id
            }
    return images

def load_colmap_data(colmap_path):      
    if os.path.exists(os.path.join(colmap_path, "cameras.bin")):
        print("Found .bin files, loading binary format...")
        cameras = read_colmap_cameras(os.path.join(colmap_path, "cameras.bin"))
        images = read_colmap_images(os.path.join(colmap_path, "images.bin"))
    elif os.path.exists(os.path.join(colmap_path, "cameras.txt")):
        print("Found .txt files, loading text format...")
        cameras = read_colmap_cameras_text(os.path.join(colmap_path, "cameras.txt"))
        images = read_colmap_images_text(os.path.join(colmap_path, "images.txt"))
    else:
        raise FileNotFoundError(f"Could not find COLMAP data in {colmap_path}")
    return cameras, images

def get_o3d_intrinsics(cam_data, r = 1):
    params = cam_data["params"]
    if cam_data["model"] == 1: 
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else: # SIMPLE_PINHOLE
        fx, fy, cx, cy = params[0], params[0], params[1], params[2]
        
    return o3d.camera.PinholeCameraIntrinsic(
        cam_data["width"] // r, cam_data["height"] // r, fx / r,  fy / r, cx / r, cy / r
    )

def post_process_mesh(mesh, cluster_to_keep=1):
    print("Post-processing mesh...")
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    if len(cluster_n_triangles) == 0:
        return mesh
        
    largest_cluster_indices = np.argsort(cluster_n_triangles)[-cluster_to_keep:]
    triangles_to_remove = [i for i, c_id in enumerate(triangle_clusters) if c_id not in largest_cluster_indices]
    
    mesh.remove_triangles_by_mask(np.isin(range(len(triangle_clusters)), triangles_to_remove))
    mesh.remove_unreferenced_vertices()
    return mesh

def read_lod_depth(depth_path):
    depth_img = Image.open(depth_path).convert("LA") # 读取你保存的 LA 格式 (L是深度, A是Alpha)
    depth_np = np.array(depth_img)
    
    depth_norm = torch.from_numpy(depth_np[:, :, 0]).float() / 255.0
    
    depth_metric = warpped_depth_inv(depth_norm)
    
    alpha = torch.from_numpy(depth_np[:, :, 1]).float() / 255.0
    depth_metric[alpha < 0.5] = 0
    # np.load
    # depth_path = '/media/zc/SSD/ar/ABXBJQ_BJ/B3/smallest/o3d_depth/image0/25184908_depth.npy'
    # depth_np = np.load(depth_path)
    return depth_metric.numpy()    

def process_frame(rgb_path, depth_path, intrinsic, extrinsic, volume, depth_scale, depth_trunc = 5 ):
    color_raw = o3d.io.read_image(rgb_path)
    # TSDF 期望 float32 格式，单位通常是米
    if depth_scale is not None:        
        depth_metric = np.load(depth_path) *  depth_scale["scale"] + depth_scale["offset"]
        #np.load(depth_file_path) * depth_scale["scale"] + depth_scale["offset"]
    else:
        depth_metric = read_lod_depth(depth_path)
    
    
    depth_o3d = o3d.geometry.Image(depth_metric)
    
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, 
        depth_o3d, 
        depth_scale=1.0,  # 已经转换好了，缩放设为 1
        depth_trunc=depth_trunc, 
        convert_rgb_to_intensity=False
    )
    
    volume.integrate(rgbd, intrinsic, extrinsic)

def main(project_dir, depth_dir_name='depths_render_color'):
    colmap_path = os.path.join(project_dir, 'sparse/0') 

    colmap_cams, colmap_imgs = load_colmap_data(colmap_path)
    # 2. 配置 TSDF 卷
    # 根据场景大小调整 voxel_size，例如 0.005 对应 5mm 精度
    voxel_size = 0.03
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=0.09, # 通常设为 3x voxel_size
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    # depth_dir = os.path.join(project_dir, depth_dir_name)
    depth_dir = os.path.join(project_dir, 'estimated_depths_v3_0706')
    image_dir = os.path.join(project_dir, 'images')
    print("Integrating frames...")
    with open(os.path.join(project_dir, "depth_scales_da3.json"), "r") as f:
        depth_scales = json.load(f)
    # depth_scales = None
    for img_name, data in tqdm(colmap_imgs.items()):
        rgb_file = os.path.join(image_dir, img_name)
        # depth_file = os.path.join(depth_dir, os.path.splitext(img_name)[0] + ".png")
        depth_file = os.path.join(depth_dir, img_name + ".npy")
        
        if not os.path.exists(rgb_file) or not os.path.exists(depth_file):
            continue

        if depth_scales is not None and img_name in depth_scales:
            depth_scale = depth_scales[img_name]
        else:
            depth_scale = None

        intrinsic = get_o3d_intrinsics(colmap_cams[data['camera_id']], r = 1)
        process_frame(
            rgb_file, 
            depth_file, 
            intrinsic, 
            data['extrinsic'], 
            volume,
            depth_scale=depth_scale
        )
    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    mesh = post_process_mesh(mesh, cluster_to_keep=1)

    output_path = os.path.join(project_dir, "fused_mesh_latest.ply")
    # o3d.io.write_triangle_mesh(output_path, mesh)
    pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud(output_path.replace(".ply", "_pcd.ply"), pcd)
    print(f"Finished! Mesh saved to {output_path}")


if __name__ == "__main__":
    # 使用示例
    project_dir = '/media/zc/SSD/ar/ABXBJQ_BJ/B3/smallest/'
    # project_dir = '/media/zc/SSD/ar/AIBEE_16/L16/showroom/'
    main(project_dir)
    pass