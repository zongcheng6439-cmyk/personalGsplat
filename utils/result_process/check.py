from plyfile import PlyData
import numpy as np

def inspect_gs_ply(ply_path: str):
    """
    Read Gaussian Splatting ply and print all vertex attributes, shape, dtype.
    """
    plydata = PlyData.read(ply_path)
    vertex = plydata["vertex"]

    print(f"===== GS PLY INFO: {ply_path} =====")
    print(f"Total Gaussians (points): {vertex.count}\n")

    print("All vertex attributes in this ply:")
    for prop_name in vertex.data.dtype.names:
        arr = vertex[prop_name]
        print(f"[{prop_name}] dtype={arr.dtype}, shape={arr.shape}, {arr[:5]}")

    print("\n==== Sample value preview (first 5 entries) ====")

    return vertex


if __name__ == "__main__":
    ply_file = "/media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/gsplat_latest_texture_mcmc_depth_anchor/ply/point_cloud_4999.ply"   # CHANGE YOUR PATH HERE
    vert_data = inspect_gs_ply(ply_file)
    gsplat_result = '/media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section/gsplat_latest_texture_mcmc_depth_anchor/ply/gsplat_point_cloud_4999.ply'
    vert_data = inspect_gs_ply(gsplat_result)

# x           
# y           
# z           
# f_dc_0      
# f_dc_1      
# f_dc_2      
# opacity     
# scale_0     
# scale_1     
# scale_2     
# rot_0       
# rot_1       
# rot_2       
# rot_3