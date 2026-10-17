# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import json 
import argparse
from tqdm import tqdm
from utils import SH2RGB
from datasets.colmap_adc import Dataset, Parser
from citygs_gsplat import (
    CityGSPartitioning,
    CityGSSceneConfig,
    CityGSPartitionableScene,
)
from gsplat.exporter import export_splats

from utils import create_splats


@torch.no_grad()
def visible_gs_centers(
    means,
    camera,
    occlusion_cell=4,
    depth_tolerance=0.1,
    chunk_size=1_000_000,
):
    """Return GS centers visible in one camera using a coarse z-buffer."""
    device = means.device
    c2w = camera["camtoworld"].to(device)
    world_to_camera = torch.linalg.inv(c2w)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    K = camera["K"].to(device)
    height, width = camera["image"].shape[:2]
    cells_x = (width + occlusion_cell - 1) // occlusion_cell
    cells_y = (height + occlusion_cell - 1) // occlusion_cell
    nearest = torch.full(
        (cells_x * cells_y,), float("inf"), device=device, dtype=means.dtype
    )

    def project_chunks():
        for start in range(0, len(means), chunk_size):
            end = min(start + chunk_size, len(means))
            camera_xyz = means[start:end] @ rotation.T + translation
            depth = camera_xyz[:, 2]
            projected = camera_xyz @ K.T
            u = projected[:, 0] / projected[:, 2].clamp_min(1e-12)
            v = projected[:, 1] / projected[:, 2].clamp_min(1e-12)
            valid = (
                torch.isfinite(camera_xyz).all(dim=-1)
                & (depth > 1e-6)
                & (u >= 0)
                & (u < width)
                & (v >= 0)
                & (v < height)
            )
            if not valid.any():
                continue
            ids = torch.arange(start, end, device=device)[valid]
            cell_x = u[valid].long() // occlusion_cell
            cell_y = v[valid].long() // occlusion_cell
            yield ids, depth[valid], cell_y * cells_x + cell_x

    for _, depth, cell in project_chunks():
        nearest.scatter_reduce_(0, cell, depth, reduce="amin", include_self=True)

    visible = torch.zeros(len(means), dtype=torch.bool, device=device)
    for ids, depth, cell in project_chunks():
        visible[ids] = depth <= nearest[cell] + depth_tolerance
    return visible


@torch.no_grad()
def save_block_splats(splats, mask, output):
    """Save a preselected block initialization PLY with all GS attributes."""
    output = os.fspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".part"
    export_splats(
        means=splats["means"][mask],
        scales=splats["scales"][mask],
        quats=splats["quats"][mask],
        opacities=splats["opacities"][mask],
        sh0=splats["sh0"][mask],
        shN=splats["shN"][mask],
        format="ply",
        save_to=temporary,
    )
    os.replace(temporary, output)

def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.cross(a, b, dim=-1)
    c = torch.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3, dtype=a.dtype, device=a.device) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = torch.linalg.norm(v)
    skew_sym_mat = torch.tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=a.dtype,
        device=a.device,
    )
    return torch.eye(3, dtype=a.dtype, device=a.device) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s ** 2 + 1e-8))


class Runner:
    """Engine for split"""

    def __init__(self,
        data_dir, 
        data_factor,
    ) -> None:
        
        self.data_dir = data_dir
        self.data_factor = data_factor         
        normalize_world_space: bool = False        
        self.device = f"cuda:{0}"

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=self.data_dir,
            factor=self.data_factor,
            normalize=normalize_world_space,            
            test_every=0,
            load_exposure=None,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=None,
            load_depths=None,
        )
        
        self.splats = create_splats(
            self.parser,
            feature_dim=None,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))
        # Track if Gaussians are frozen (for controller distillation)
        self._gaussians_frozen = False

    def split(
        self,
        block_dim,
        reorient=False,
        origin="auto",
        build_block_init=True,
        occlusion_cell=4,
        depth_tolerance=0.1,
        visibility_chunk_size=1_000_000,
    ):
        
        # Partition only the cameras exposed by the training Dataset.  Using
        # every parser camera here makes assignment masks longer than trainset
        # when the evaluation split removes one or more images.
        c2w = torch.from_numpy(
            self.parser.camtoworlds[self.trainset.indices]
        ).to(self.device)
        
        # reoriented_camera_centers = c2w[:, :3, 3]
        # reoriented_point_cloud_xyz = self.splats["means"]
        point_rgbs = SH2RGB(self.splats["sh0"]).clamp(0, 1).to(c2w.device) * 255.0

        if reorient:
            # reorient the scene, calculate the up direction of the scene
            # NOTE: 
            #   the calculated direction may not be perfect or even incorrect sometimes, 
            #   in such a situation, you need to provide a correct up vector
            up = -torch.mean(c2w[:, :3, 1], dim=0)
            up = up / torch.linalg.norm(up)

            rotation = rotation_matrix(up, torch.tensor([0, 0, 1], dtype=up.dtype))
            rotation_transform = torch.eye(4, dtype=up.dtype)
            rotation_transform[:3, :3] = rotation

            reoriented_camera_centers = c2w[:, :3, 3] @ rotation_transform[:3, :3].T
            reoriented_point_cloud_xyz = self.splats['means'] @ rotation_transform[:3, :3].T
        else:
            up = torch.tensor([0., 0., 1.], dtype=c2w.dtype)
            rotation_transform = torch.eye(4, dtype=c2w.dtype)
            reoriented_camera_centers = c2w[:, :3, 3]
            reoriented_point_cloud_xyz = self.splats['means']


        scene_config = CityGSSceneConfig(
            origin=torch.tensor([0., 0.]),
            block_dim=block_dim,
        )
        scene = CityGSPartitionableScene(scene_config, reoriented_camera_centers[..., :2], reoriented_points=reoriented_point_cloud_xyz[..., :2])
        
        scene.get_bounding_box_by_points()

        if origin:
            if origin == 'auto':
                scene_config.origin = (scene.point_based_bounding_box.min + scene.point_based_bounding_box.max) / 2
            else:
                scene_config.origin = torch.tensor([0,0,0])

        scene.get_scene_bounding_box()
        scene.build_partition_coordinates()

        print(f"Camera center based partition assignment: {scene.camera_center_based_partition_assignment().sum(-1)}")
        # how to use rendering to select image
        projection_counts = scene.projection_based_partition_assignment(
            self.splats, self.trainset
        ).sum(-1)
        print(f"Projection based partition assignment: {projection_counts}")

        output_path = os.path.join(self.data_dir, scene.build_output_dirname())
        os.makedirs(output_path, exist_ok=True)
        print(f"Output path: {output_path}")
  
        scene.save(
            output_path,
            extra_data={
                "up": up,
                "rotation_transform": rotation_transform,
            }
        )
        scene.save_plot(scene.plot_partitions, os.path.join(output_path, "partitions.png"), notebook=False)

        is_images_assigned_to_partitions = torch.logical_or(scene.is_camera_in_partition, scene.is_partitions_visible_to_cameras)
        # is_images_assigned_to_partitions = scene.is_camera_in_partition
        print(f"Overall images assigned to partitions: {is_images_assigned_to_partitions.sum(-1)}")

        max_plot_points = 51_200
        plot_point_sparsify = max(reoriented_point_cloud_xyz.shape[0] // max_plot_points, 1)

        block_gs_masks = None
        core_gs_masks = None
        if build_block_init:
            boxes = scene.partition_coordinates.get_bounding_boxes(
                scene.scene_config.partition_size, enlarge=0.0
            ).to(device=self.splats["means"].device)
            core_gs_masks = CityGSPartitioning.is_in_bounding_boxes(
                bounding_boxes=boxes,
                coordinates=reoriented_point_cloud_xyz[:, :2],
            )
            block_gs_masks = core_gs_masks.clone()
            used_images = torch.where(
                is_images_assigned_to_partitions.any(dim=0)
            )[0].tolist()
            for image_index in tqdm(used_images, desc="Projecting block GS centers"):
                visible = visible_gs_centers(
                    self.splats["means"],
                    self.trainset[image_index],
                    occlusion_cell=occlusion_cell,
                    depth_tolerance=depth_tolerance,
                    chunk_size=visibility_chunk_size,
                )
                visible_ids = torch.where(visible)[0]
                assigned_blocks = torch.where(
                    is_images_assigned_to_partitions[:, image_index]
                )[0]
                for partition_idx in assigned_blocks.tolist():
                    block_gs_masks[partition_idx, visible_ids] = True

        written_idx_list = []
        for partition_idx in tqdm(list(range(is_images_assigned_to_partitions.shape[0]))):
            partition_image_indices = is_images_assigned_to_partitions[partition_idx].nonzero().squeeze(-1).tolist()
            if len(partition_image_indices) == 0:
                continue
                
            written_idx_list.append(partition_idx)
                
            camera_list = []
            
            with open(os.path.join(output_path, "{}.txt".format(scene.partition_coordinates.get_str_id(partition_idx))), "w") as f:
                for image_index in partition_image_indices:
                    f.write(self.trainset.image_names[image_index])
                    f.write("\n")
                    
                    # below camera list is just for visualization, not for training, so its camera intrinsics are fixed values
                    color = [0, 0, 255]
                    if scene.is_partitions_visible_to_cameras[partition_idx][image_index]:
                        color = [255, 0, 0]
                    camera_list.append({
                        "id": image_index,
                        "img_name": self.trainset.image_names[image_index],
                        "position": c2w[image_index][:3, 3].cpu().numpy().tolist(),
                        "rotation": c2w[image_index][:3, :3].cpu().numpy().tolist(),
                        "color": color,
                    })
                    
            with open(os.path.join(
                    output_path, 
                    f"cameras-{scene.partition_coordinates.get_str_id(partition_idx)}.json",
            ), "w") as f:
                json.dump(camera_list, f, indent=4, ensure_ascii=False)

            if block_gs_masks is not None:
                block_name = scene.partition_coordinates.get_str_id(partition_idx)
                output_ply = os.path.join(
                    output_path, "block_init", f"point_cloud_{block_name}.ply"
                )
                selected_mask = block_gs_masks[partition_idx]
                core_count = int(core_gs_masks[partition_idx].sum().item())
                total_count = int(selected_mask.sum().item())
                save_block_splats(self.splats, selected_mask, output_ply)
                with open(os.path.splitext(output_ply)[0] + ".json", "w") as f:
                    json.dump(
                        {
                            "block": block_name,
                            "images": len(partition_image_indices),
                            "core_gs": core_count,
                            "visible_added_gs": total_count - core_count,
                            "total_gs": total_count,
                            "occlusion_cell": occlusion_cell,
                            "depth_tolerance": depth_tolerance,
                        },
                        f,
                        indent=2,
                    )
                print(
                    f"Block {block_name} init GS: core={core_count:,}, "
                    f"visible_added={total_count - core_count:,}, "
                    f"total={total_count:,}, ply={output_ply}"
                )
            
            scene.save_plot(
                scene.plot_partition_assigned_cameras,
                os.path.join(output_path, "{}.png".format(scene.partition_coordinates.get_str_id(partition_idx))),
                False,
                partition_idx,
                reoriented_point_cloud_xyz.detach().cpu().numpy(),
                point_rgbs.detach().cpu().numpy(),
                point_sparsify=plot_point_sparsify,
            )


def get_latest_ckpt(ckpt_dir: str):
    """Auto find latest ckpt file by number suffix like ckpt_29999_rank0.pt"""
    ckpt_files = []
    for fname in os.listdir(ckpt_dir):
        if fname.endswith(".pt") and fname.startswith("ckpt_"):
            full_path = os.path.join(ckpt_dir, fname)
            ckpt_files.append(full_path)
    if not ckpt_files:
        raise FileNotFoundError(f"No ckpt *.pt found in {ckpt_dir}")
    
    # Sort by numeric step ckpt_<num>_rank0.pt
    def extract_step(path):
        basename = os.path.basename(path)
        num_str = basename.split("_")[1]
        return int(num_str)
    
    ckpt_files.sort(key=extract_step)
    latest = ckpt_files[-1]
    print(f"Found latest checkpoint: {latest}")
    return [latest]

def main(args):    
    runner = Runner(data_factor=args.data_factor,
                    data_dir=args.data_dir)
    
    ckpt_path = get_latest_ckpt(args.ckpt_dir)

    ckpts = [
        torch.load(file, map_location=runner.device, weights_only=True)
        for file in ckpt_path
    ]

    for k in runner.splats.keys():
        runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])

    runner.split(
        args.block_dim,
        build_block_init=not args.skip_block_init,
        occlusion_cell=args.gs_occlusion_cell,
        depth_tolerance=args.gs_depth_tolerance,
        visibility_chunk_size=args.gs_visibility_chunk_size,
    )
    


if __name__ == "__main__":
    # Build argument parser
    """
    python partition.py \
        --data_dir /media/zc/SSD/GS/lutu/parking01 \
        --data_factor 1.0 \
        --ckpt_paths /media/zc/SSD/GS/lutu/parking01/3dgs_depth_localv1/ckpts/ckpt_29999_rank0.pt
    """
    parser = argparse.ArgumentParser(description="Merge multiple 3DGS checkpoint splats")

    # Required arguments for Runner
    parser.add_argument("--data_dir", type=str, default='/media/zc/SSD/ar/XX/B3/B3_seg/', 
                        help="Dataset root directory path")
    parser.add_argument("--data_factor", type=int, default=2, 
                        help="Data downsample factor")
    
    parser.add_argument("--block_dim", type=int, nargs="+", default=[3, 3],
                    help="Block dimension list, example: --block_dim 2 3")
    # Optional: add ckpt path as cli param if you want dynamic input
    parser.add_argument("--ckpt_dir", type=str, default='/media/zc/SSD/ar/XX/B3/B3_seg/gsaplt_large/ckpts/', 
                        help="List of checkpoint files")
    parser.add_argument("--skip_block_init", action="store_true")
    parser.add_argument("--gs_occlusion_cell", type=int, default=4)
    parser.add_argument("--gs_depth_tolerance", type=float, default=0.1)
    parser.add_argument("--gs_visibility_chunk_size", type=int, default=1_000_000)
    args = parser.parse_args()
    if args.gs_occlusion_cell <= 0 or args.gs_visibility_chunk_size <= 0:
        parser.error("GS visibility cell and chunk size must be positive")
    if args.gs_depth_tolerance < 0:
        parser.error("GS depth tolerance must be non-negative")
    main(args)
