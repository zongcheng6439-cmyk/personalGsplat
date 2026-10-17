import os 
import numpy as np 
from plyfile import PlyData, PlyElement
import sys
import argparse
# /training/zcli/code/gaussians_camera//utils
sys.path.append("/home/zcli/code/gaussians_camera/")
sys.path.append('/home/zc/code/aibee/gaussians_camera/')
from api.utils.read_write_model import read_points3D_binary, read_points3D_text # type: ignore 

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz) #xyz (n, 3)

    elements = np.empty(xyz.shape[0], dtype=dtype) # 
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', default="/home/zc/", type=str)
    parser.add_argument('--label', default="0", type=int)    
    args = parser.parse_args()
    args.project_dir = '/media/zc/SSD/ar/WSMSD_TCC_WH/B2F/757ef6909dca4449adca9c5a595691fa/sfm_superpoint+superglue/model_reg/'
    xyz = None
    rgb = None
    if args.label == 0:
        sparse_dir = args.project_dir
        bin_path = os.path.join(sparse_dir, "points3D.bin")
        txt_path = os.path.join(sparse_dir, "points3D.txt")
        ply_path = os.path.join(sparse_dir, 'points3D.ply')
        if os.path.exists(bin_path): 
            xyz, rgb, _ = read_points3D_binary(bin_path)
        elif os.path.exists(txt_path):
            xyz, rgb, _ = read_points3D_text(txt_path)
        else:
            print(f'no such file!!!!')
            sys.exit(1)

        storePly(ply_path, xyz, rgb)   
    else:
        
        for chunk_dir in os.listdir(args.project_dir):
            sparse_dir = os.path.join(args.project_dir, chunk_dir, "sparse/0")
            bin_path = os.path.join(sparse_dir, "points3D.bin")
            txt_path = os.path.join(sparse_dir, "points3D.txt")
            ply_path = os.path.join(sparse_dir, 'point3D.ply')
            if os.path.exists(bin_path):
                xyz, rgb, _ = read_points3D_binary(bin_path)
            elif os.path.exists(txt_path):
                xyz, rgb, _ = read_points3D_text(txt_path)
            else:
                print(f'no such file!!!!')

            storePly(ply_path, xyz, rgb)    
