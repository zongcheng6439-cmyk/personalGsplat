import os
import argparse
import torch
import glob 
import re
import numpy as np
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from citygs_gsplat import CityGSPartitioning, PartitionCoordinates

def load_ply(path, sh=3):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(sh + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (sh + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    return xyz, features_dc, features_extra, opacities, scales, rots

def mkdir_p(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def save_merged_ply(
    xyz, features_dc, features_rest, opacity, scaling, rotation, path,
    save_f_rest: bool = False  
):
    mkdir_p(os.path.dirname(path))
    xyz = xyz
    normals = np.zeros_like(xyz)

    f_dc = features_dc.transpose(0, 2, 1).reshape(features_dc.shape[0], -1)
    
    if save_f_rest:
        f_rest = features_rest.transpose(0, 2, 1).reshape(features_rest.shape[0], -1)
        f_rest_dim = f_rest.shape[1]
    else:
        f_rest = None
        f_rest_dim = 0

    opacities = opacity
    scale = scaling
    rot = rotation

    attribute_names = [
        'x', 'y', 'z',
        'nx', 'ny', 'nz',
        'f_dc_0', 'f_dc_1', 'f_dc_2',
    ]

    if save_f_rest:
        attribute_names.extend([f'f_rest_{i}' for i in range(f_rest_dim)])
    
    attribute_names += [
        'opacity',
        'scale_0', 'scale_1', 'scale_2',
        'rot_0', 'rot_1', 'rot_2', 'rot_3'
    ]
    dtype_full = [(attr, 'f4') for attr in attribute_names]

    concat_list = [xyz, normals, f_dc]
    if save_f_rest:
        concat_list.append(f_rest)
    concat_list.extend([opacities, scale, rot])
    attributes = np.concatenate(concat_list, axis=1)

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, attributes))
    
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def search_load_file(model_path: str) -> str:
    # if a directory path is provided, auto search checkpoint or ply
    if os.path.isdir(model_path) is False:
        return model_path
    
    # not a checkpoint can be found, search point cloud
    load_from = None
    latest_step = -1
    for path in glob.glob(os.path.join(model_path, "ply", "gsplat_point_cloud_*.ply")):
        match = re.search(r"(\d+)(?=\.ply$)", os.path.basename(path))
        if match and int(match.group(1)) > latest_step:
            latest_step = int(match.group(1))
            load_from = path
    if load_from is None:
        previous_point_cloud_iteration = -1
        for i in glob.glob(os.path.join(model_path, "point_cloud", "iteration_*")):
            try:
                point_cloud_iteration = int(os.path.basename(i).replace("iteration_", ""))
            except Exception as err:
                print("error occurred when parsing iteration from {}: {}".format(i, err))
                continue

            if point_cloud_iteration > previous_point_cloud_iteration:
                previous_point_cloud_iteration = point_cloud_iteration
                load_from = os.path.join(i, "point_cloud.ply")

    assert load_from is not None, f"not a checkpoint or point cloud can be found {model_path}"

    return load_from

def mkdir_p(path):
    if path != '' and not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser()
parser.add_argument("--path",default='/media/zc/SSD/GS/lutu/parking01/coarse',
                     help="Path to the model output directory")
parser.add_argument("--block_dir",default='/media/zc/SSD/GS/lutu/parking01/coarse',
                     help="Path to the model output directory")
parser.add_argument("--sh",default='3', type=int)
parser.add_argument("--block_dim", type=int, nargs="+", default=[3, 3],
                    help="Block dimension list, example: --block_dim 2 3")                     
args = parser.parse_args()


checkpoint_dir = os.path.join(args.path, "block")
dim_x = args.block_dim[0]
dim_y = args.block_dim[1]
print(f"model_dir: {checkpoint_dir}, dim_x : {dim_x}, dim_y: {dim_y}")
print("Searching point cloud files...")
max_iteration = -1
checkpoint_files = []

for i in os.listdir(checkpoint_dir):
    block_path = os.path.join(checkpoint_dir, i)
    loadable_file = search_load_file(block_path)    
   
    if not loadable_file.endswith(".ply"):
        continue
    checkpoint_files.append(loadable_file)

checkpoint_files = sorted(checkpoint_files)
assert len(checkpoint_files) > 0, "No point cloud found!"
print(f"Found {len(checkpoint_files)} block point cloud ")


partitions_pt_path = os.path.join(args.block_dir, 'partitions.pt')
print(f"partitions_pt_path: {partitions_pt_path}")
try:
    partitions = torch.load(partitions_pt_path, map_location="cpu", weights_only=False)
except TypeError:
    partitions = torch.load(partitions_pt_path, map_location="cpu")
partition_coordinates = PartitionCoordinates(
    id=partitions['partition_coordinates']['id'],
    xy=partitions['partition_coordinates']['xy'],
)
partition_bounding_boxes = partition_coordinates.get_bounding_boxes(
    partitions['scene_config']['partition_size'], enlarge=0.
)
rotation_transform = partitions['extra_data']['rotation_transform'][:3, :3].T
scene_contract = partitions['scene_config']['contract']
scene_aabb = partitions['scene_config']['aabb']

merged_xyz = []
merged_features_dc = []
merged_features_rest = []
merged_opacity = []
merged_scaling = []
merged_rotation = []

print("Loading and merging all block Gaussians...")

partition_ids = torch.as_tensor(partitions['partition_coordinates']['id'], dtype=torch.long)

for pc_path in tqdm(checkpoint_files, desc="Loading blocks"):
    # print(f"pc path: {pc_path}")
    result_name = pc_path.split('/')[-3] # 000_001  2 
    index_x, index_y = result_name.split("_") 
    target_id = torch.tensor([int(index_x), int(index_y)], dtype=torch.long)
    matches = torch.nonzero(torch.all(partition_ids == target_id, dim=-1), as_tuple=False).flatten()
    if len(matches) != 1:
        raise ValueError(f"Partition {result_name} occurs {len(matches)} times in partitions.pt")
    block_id = int(matches.item())
    print(f"pc path: {pc_path}, result_name: {result_name}, block id: {block_id}")
    # block_id = eval(pc_path.split('/')[-4].split('_')[-1])
    xyz, f_dc, f_rest, opacity, scaling, rotation = load_ply(pc_path, sh=args.sh)
    selection_xyz = torch.from_numpy(xyz).to(rotation_transform.dtype) @ rotation_transform

    if scene_contract:
        xyz_contract = CityGSPartitioning.contract_to_unisphere(
            selection_xyz[:, :2], scene_aabb, ord=torch.inf
        )
        selection_xyz = torch.cat([xyz_contract, selection_xyz[:, 2:]], dim=1)


    mask = CityGSPartitioning.is_in_bounding_boxes(
        bounding_boxes=partition_bounding_boxes,
        coordinates=selection_xyz[:, :2],
    )[block_id]
    mask = mask.detach().cpu().numpy()

    # The transformed coordinates are used only for block selection. Keep the
    # trained Gaussian parameters in their original world coordinate system.
    merged_xyz.append(xyz[mask])
    merged_features_dc.append(f_dc[mask])
    merged_features_rest.append(f_rest[mask])
    merged_opacity.append(opacity[mask])
    merged_scaling.append(scaling[mask])
    merged_rotation.append(rotation[mask])
    print(f"Block id : {block_id}, merged gs nums: {xyz[mask].shape[0]}")


print("Concatenating all Gaussian parameters...")
merged_xyz = np.concatenate(merged_xyz, axis=0)
merged_features_dc = np.concatenate(merged_features_dc, axis=0)
merged_features_rest = np.concatenate(merged_features_rest, axis=0)
merged_opacity = np.concatenate(merged_opacity, axis=0)
merged_scaling = np.concatenate(merged_scaling, axis=0)
merged_rotation = np.concatenate(merged_rotation, axis=0)

print(f"Total Gaussians: {merged_xyz.shape[0]}")


output_ply_path = os.path.join(args.path, "final_merged_model.ply")
print(f"Saving merged PLY to: {output_ply_path}")
save_merged_ply(
    merged_xyz, merged_features_dc, merged_features_rest,
    merged_opacity, merged_scaling, merged_rotation,
    output_ply_path
)

print("All done! Merged model saved successfully.")
