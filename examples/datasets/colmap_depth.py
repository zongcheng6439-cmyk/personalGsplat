# SPDX-FileCopyrightText: Copyright 2023-2026 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import pycolmap
import torch
from PIL import Image
from tqdm import tqdm
from typing_extensions import assert_never

from exif import compute_exposure_from_exif
from datasets.colmap import _as_dict, _camera_distortion, _image_w2c
from datasets.colmap_custom_utils import (
    _get_rel_paths,
    _resize_image_folder,
    count_files,
    fetchPly,
    reconstruction_points,
)
from datasets.normalize import (
    align_principal_axes,
    similarity_from_cameras,
    transform_cameras,
    transform_points,
)

class Parser:
    """COLMAP parser."""

    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        load_exposure: bool = False,
        texture_score_file: str = "texture_score.json",
        load_texture_scores: bool = False,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.load_exposure = load_exposure

        colmap_dir = os.path.join(data_dir, "sparse/0/")
        if not os.path.exists(colmap_dir):
            colmap_dir = os.path.join(data_dir, "sparse")
        assert os.path.exists(
            colmap_dir
        ), f"COLMAP directory {colmap_dir} does not exist."
    
        colmap_image_dir = os.path.join(data_dir, "images")
        
        reconstruction = pycolmap.Reconstruction(colmap_dir)

        # Extract extrinsic matrices in world-to-camera format.
        cameras = _as_dict(reconstruction.cameras)
        images = _as_dict(reconstruction.images)
        registered_ids = [int(i) for i in reconstruction.reg_image_ids()]
        imdata = {image_id: images[image_id] for image_id in registered_ids}
        w2c_mats = []
        camera_ids = []
        image_names = []
        Ks_dict = dict()
        params_dict = dict()
        camtype_dict = dict()
        imsize_dict = dict()  # width, height
        mask_dict = dict()
        for image_id, im in imdata.items():

            image_path = os.path.join(colmap_image_dir, im.name)
            if not os.path.exists(image_path):
                continue
            
            image_names.append(im.name)
            w2c_mats.append(_image_w2c(im))

            # support different camera intrinsics
            camera_id = int(im.camera_id)
            camera_ids.append(camera_id)

            # camera intrinsics
            cam = cameras[camera_id]
            K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
            K[:2, :] /= factor
            Ks_dict[camera_id] = K

            # Get distortion parameters.
            params, camtype = _camera_distortion(cam)
            params_dict[camera_id] = params
            camtype_dict[camera_id] = camtype
            imsize_dict[camera_id] = (cam.width // factor, cam.height // factor)
            mask_dict[camera_id] = None
        print(
            f"[Parser] {len(imdata)} images, taken by {len(set(camera_ids))} cameras.\
                valid image num: {len(image_names)}"
        )

        if len(imdata) == 0:
            raise ValueError("No images found in COLMAP.")
        if any(len(params) > 0 for params in params_dict.values()):
            print("Warning: COLMAP Camera is not PINHOLE. Images have distortion.")

        w2c_mats = np.stack(w2c_mats, axis=0)

        # Convert extrinsics to camera-to-world.
        camtoworlds = np.linalg.inv(w2c_mats)

        # Image names from COLMAP. No need for permuting the poses according to
        # image names anymore.
        # image_names = [imdata[k].name for k in imdata]
        
        # Previous Nerf results were generated with images sorted by filename,
        # ensure metrics are reported on the same test set.
        inds = np.argsort(image_names)
        image_names = [image_names[i] for i in inds]
        camtoworlds = camtoworlds[inds]
        camera_ids = [camera_ids[i] for i in inds]

        # Load extended metadata. Used by Bilarf dataset.
        self.extconf = {
            "spiral_radius_scale": 1.0,
            "no_factor_suffix": False,
        }
        extconf_file = os.path.join(data_dir, "ext_metadata.json")
        if os.path.exists(extconf_file):
            with open(extconf_file) as f:
                self.extconf.update(json.load(f))
                
        # Load images.
        if factor > 1 and not self.extconf["no_factor_suffix"]:
            image_dir_suffix = f"_{factor}"
        else:
            image_dir_suffix = ""
        image_dir = os.path.join(data_dir, "images" + image_dir_suffix)                
        # for d in [image_dir, colmap_image_dir]:
        #     if not os.path.exists(d):
        #         raise ValueError(f"Image folder {d} does not exist.")
        # Downsampled images may have different names vs images used for COLMAP,
        # so we need to map between the two sorted lists of files.
        colmap_files = sorted(_get_rel_paths(colmap_image_dir))
        image_files = sorted(_get_rel_paths(image_dir))
        resize_dir = image_dir + "_png"
        print(f"image_dir: {image_dir}, resize dir: {resize_dir}")
        # if factor > 1 and os.path.splitext(image_files[0])[1].lower() == ".jpg"
        if factor > 1:
            if not os.path.exists(resize_dir) or \
               count_files(colmap_image_dir) != count_files(resize_dir):
               print(f"colmap_image_dir size: {count_files(colmap_image_dir)} \
                       resize image dir : {count_files(resize_dir)}")
               image_dir = _resize_image_folder(
                    colmap_image_dir, resize_dir, factor=factor
                )
            else:
                image_dir = resize_dir
            image_files = sorted(_get_rel_paths(image_dir))

        colmap_to_image = dict(zip(colmap_files, image_files))
        image_paths = [os.path.join(image_dir, colmap_to_image[f]) for f in image_names]
        
        # read image mask remove person 
        mask_dir = os.path.join(data_dir, 'masks')
        if factor > 1:
            mask_dir = os.path.join(data_dir, f'masks_{factor}_png')            
            if not os.path.exists(mask_dir) or count_files(mask_dir) != count_files(image_dir):
                mask_dir = _resize_image_folder(
                    os.path.join(data_dir, 'masks') , mask_dir, factor=factor,
                    resize_mask=True
                )                
        mask_paths = [os.path.join(mask_dir, colmap_to_image[f]) for f in image_names]

        # read depth map from gs render 
        depths_dir = os.path.join(data_dir, 'depths_render_color_edge')
        if not os.path.isdir(depths_dir):
            depths_dir = os.path.join(data_dir, 'depths_render_color')
        depths_paths = [
            os.path.join(depths_dir, Path(colmap_to_image[f]).stem + ".png")
            for f in image_names
        ]

        # 3D points and {image_name -> [point_idx]}
        points, points_err, points_rgb = reconstruction_points(
            reconstruction, _as_dict
        )

        # read lidar pc 
        if os.path.exists(os.path.join(colmap_dir, 'extracted.ply')):
            points, points_rgb, normals = fetchPly(os.path.join(colmap_dir, 'extracted.ply'))

        point_indices = dict()

        # Normalize the world space.
        if normalize:
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points = transform_points(T1, points)

            T2 = align_principal_axes(points)
            camtoworlds = transform_cameras(T2, camtoworlds)
            points = transform_points(T2, points)

            transform = T2 @ T1

            # Fix for up side down. We assume more points towards
            # the bottom of the scene which is true when ground floor is
            # present in the images.
            if np.median(points[:, 2]) > np.mean(points[:, 2]):
                # rotate 180 degrees around x axis such that z is flipped
                T3 = np.array(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                camtoworlds = transform_cameras(T3, camtoworlds)
                points = transform_points(T3, points)
                transform = T3 @ transform
        else:
            transform = np.eye(4)

        self.image_names = image_names  # List[str], (num_images,)
        self.image_paths = image_paths  # List[str], (num_images,)
        self.mask_paths  = mask_paths   # List[str], (num_images,)
        self.depth_paths = depths_paths # List[str], (num_images,)

        self.texture_scores = None
        score_path = Path(data_dir) / texture_score_file
        if load_texture_scores:
            if not score_path.is_file():
                raise FileNotFoundError(f"Texture score file not found: {score_path}")
            rows = json.loads(score_path.read_text())
            scores = {
                Path(str(row["name"])).with_suffix("").as_posix(): row["score"]
                for row in rows
            }
            basename_scores = {
                Path(name).name: score for name, score in scores.items()
            }
            aligned = []
            for name in image_names:
                key = Path(name).with_suffix("").as_posix()
                if key in scores:
                    aligned.append(scores[key])
                elif Path(key).name in basename_scores:
                    aligned.append(basename_scores[Path(key).name])
                else:
                    raise KeyError(f"No texture score in {score_path} for image {name}")
            self.texture_scores = np.asarray(aligned, dtype=np.float32)
        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_err = points_err  # np.ndarray, (num_points,)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
        self.point_indices = point_indices  # Dict[str, np.ndarray], image_name -> [M,]
        self.transform = transform  # np.ndarray, (4, 4)

        # Create 0-based contiguous camera indices from COLMAP camera_ids.
        # This is useful for camera-based embeddings/modules.
        unique_camera_ids = sorted(set(camera_ids))
        self.camera_id_to_idx = {cid: idx for idx, cid in enumerate(unique_camera_ids)}
        self.camera_indices = [self.camera_id_to_idx[cid] for cid in camera_ids]
        self.num_cameras = len(unique_camera_ids)

        # Load EXIF exposure data if requested.
        # Always read from original (non-downscaled) images since PNG doesn't support EXIF.
        if load_exposure:
            exposure_values: List[Optional[float]] = []
            for image_name in tqdm(image_names, desc="Loading EXIF exposure"):
                original_path = Path(colmap_image_dir) / image_name
                exposure_values.append(compute_exposure_from_exif(original_path))

            # Compute mean across all valid exposures and subtract
            valid_exposures = [e for e in exposure_values if e is not None]
            if valid_exposures:
                exposure_mean = sum(valid_exposures) / len(valid_exposures)
                self.exposure_values: List[Optional[float]] = [
                    (e - exposure_mean) if e is not None else None
                    for e in exposure_values
                ]
                print(
                    f"[Parser] Loaded exposure for {len(valid_exposures)}/{len(exposure_values)} images "
                    f"(mean={exposure_mean:.3f} EV)"
                )
            else:
                self.exposure_values = [None] * len(exposure_values)
                print("[Parser] No valid EXIF exposure data found in any image.")
        else:
            self.exposure_values = [None] * len(image_paths)

        # load one image to check the size. In the case of tanksandtemples dataset, the
        # intrinsics stored in COLMAP corresponds to 2x upsampled images.
        actual_image = imageio.imread(self.image_paths[0])[..., :3]
        self.actual_height, self.actual_width = actual_image.shape[:2]
        colmap_width, colmap_height = self.imsize_dict[self.camera_ids[0]]
        s_height, s_width = self.actual_height / colmap_height, self.actual_width / colmap_width

        for camera_id, K in self.Ks_dict.items():
            K[0, :] *= s_width
            K[1, :] *= s_height
            self.Ks_dict[camera_id] = K
            width, height = self.imsize_dict[camera_id]
            self.imsize_dict[camera_id] = (int(width * s_width), int(height * s_height))

        # undistortion
        self.mapx_dict = dict()
        self.mapy_dict = dict()
        self.roi_undist_dict = dict()
        for camera_id in self.params_dict.keys():
            params = self.params_dict[camera_id]
            if len(params) == 0:
                continue  # no distortion
            camtype = camtype_dict[camera_id]
            assert camera_id in self.Ks_dict, f"Missing K for camera {camera_id}"
            assert (
                camera_id in self.params_dict
            ), f"Missing params for camera {camera_id}"
            K = self.Ks_dict[camera_id]
            width, height = self.imsize_dict[camera_id]

            if camtype == "perspective":
                K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
                    K, params, (width, height), 0
                )
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K, params, None, K_undist, (width, height), cv2.CV_32FC1
                )
                mask = None
            elif camtype == "fisheye":
                fx = K[0, 0]
                fy = K[1, 1]
                cx = K[0, 2]
                cy = K[1, 2]
                grid_x, grid_y = np.meshgrid(
                    np.arange(width, dtype=np.float32),
                    np.arange(height, dtype=np.float32),
                    indexing="xy",
                )
                x1 = (grid_x - cx) / fx
                y1 = (grid_y - cy) / fy
                theta = np.sqrt(x1**2 + y1**2)
                r = (
                    1.0
                    + params[0] * theta**2
                    + params[1] * theta**4
                    + params[2] * theta**6
                    + params[3] * theta**8
                )
                mapx = (fx * x1 * r + width // 2).astype(np.float32)
                mapy = (fy * y1 * r + height // 2).astype(np.float32)

                # Use mask to define ROI
                mask = np.logical_and(
                    np.logical_and(mapx > 0, mapy > 0),
                    np.logical_and(mapx < width - 1, mapy < height - 1),
                )
                y_indices, x_indices = np.nonzero(mask)
                y_min, y_max = y_indices.min(), y_indices.max() + 1
                x_min, x_max = x_indices.min(), x_indices.max() + 1
                mask = mask[y_min:y_max, x_min:x_max]
                K_undist = K.copy()
                K_undist[0, 2] -= x_min
                K_undist[1, 2] -= y_min
                roi_undist = [x_min, y_min, x_max - x_min, y_max - y_min]
            else:
                assert_never(camtype)

            self.mapx_dict[camera_id] = mapx
            self.mapy_dict[camera_id] = mapy
            self.Ks_dict[camera_id] = K_undist
            self.roi_undist_dict[camera_id] = roi_undist
            self.imsize_dict[camera_id] = (roi_undist[2], roi_undist[3])
            self.mask_dict[camera_id] = mask

        # size of the scene measured by cameras
        camera_locations = camtoworlds[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)
        self.scene_scale = np.max(dists)



class Dataset:
    """A simple dataset class."""

    def __init__(
        self,
        parser: Parser,
        split: str = "train",
        patch_size: Optional[int] = None,
        load_depths: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        indices = np.arange(len(self.parser.image_names))        
        if split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

        self.image_names = [self.parser.image_names[i] for i in self.indices]
        self.texture_scores = (
            None
            if self.parser.texture_scores is None
            else self.parser.texture_scores[self.indices]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:     
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        H, W  = image.shape[0], image.shape[1]
        mask_image = imageio.imread(self.parser.mask_paths[index])
        if mask_image.shape[-1] == 3:
            mask_image = mask_image[..., 0]
        else:
            mask_image = mask_image
        mask_image = mask_image > 128

        if self.load_depths:
            depth_map = Image.open(self.parser.depth_paths[index])
            resized_depth_map = PILtoTorch(depth_map.convert("LA"), (W, H))
            depth_gs = warpped_depth_inv(resized_depth_map[0:1])[0, :, :]
            has_inf = torch.isinf(depth_gs)
            has_nan = torch.isnan(depth_gs)
            invalid_depth = has_inf | has_nan  
            # print(invalid_depth.sum())
            depth_gs_mask = resized_depth_map[1:2][0, :, :].bool() & (~invalid_depth)
            depth_gs = torch.nan_to_num(depth_gs, nan=0.0, posinf=0.0, neginf=0.0)

        camera_id = self.parser.camera_ids[index]
        K = self.parser.Ks_dict[camera_id].copy()  # undistorted K
        params = self.parser.params_dict[camera_id]
        camtoworlds = self.parser.camtoworlds[index]
        mask = self.parser.mask_dict[camera_id]

        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                self.parser.mapx_dict[camera_id],
                self.parser.mapy_dict[camera_id],
            )
           
            image = cv2.remap(image, mapx, mapy, cv2.INTER_CUBIC)
            x, y, w, h = self.parser.roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]
            mask_image = mask_image[y : y + h, x : x + w]
            if self.load_depths:
                depth_gs = depth_gs[y : y + h, x : x + w]
                depth_gs_mask = depth_gs_mask[y : y + h, x : x + w]

        if mask is not None:
            mask_image = np.logical_and(mask, mask_image)

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            mask_image = mask_image[y : y + self.patch_size, x : x + self.patch_size]
            if self.load_depths:
                depth_gs = depth_gs[y : y + self.patch_size, x : x + self.patch_size]
                depth_gs_mask = depth_gs_mask[
                    y : y + self.patch_size, x : x + self.patch_size
                ]
            K[0, 2] -= x
            K[1, 2] -= y

        data = {
            "K": torch.from_numpy(K).float(),
            "camtoworld": torch.from_numpy(camtoworlds).float(),
            "image": torch.from_numpy(image).float(),
            "image_id": item,  # the index of the image in the dataset
            "camera_idx": self.parser.camera_indices[
                index
            ],  # 0-based contiguous camera index
        }
        data["mask"] = torch.from_numpy(mask_image).bool()
        if self.load_depths:
            data['depth_gs'] = depth_gs
            data['depth_gs_mask'] = depth_gs_mask

        # Add exposure if available for this image
        exposure = self.parser.exposure_values[index]
        if exposure is not None:
            data["exposure"] = torch.tensor(exposure, dtype=torch.float32)

        return data

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def warpped_depth(depth):
    return torch.where(depth < 10.0, depth / 10.0, 2.0 - 10 / depth) / 2.0

def warpped_depth_inv(depth):
    return torch.where(depth < 0.5, 2 * depth * 10.0, 10 / (1.0 - depth) / 2)

def verify_mask_filter(dataset, target_image_name):
    """
    Verify that depth points are correctly filtered inside the mask region
    Checks only the target image: 28377822.png
    Saves comparison plot to disk for debugging
    """
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = (16, 10)

    # Find target image index
    parser = dataset.parser
    target_idx = None
    for i, name in enumerate(parser.image_names):
        if target_image_name in name:
            target_idx = i
            break
    assert target_idx is not None, f"Cannot find image {target_image_name}"

    # Run exactly your depth filtering logic
    camtoworlds = parser.camtoworlds[target_idx]
    K = parser.Ks_dict[parser.camera_ids[target_idx]]
    image = imageio.imread(parser.image_paths[target_idx])[..., :3]
    mask_image = imageio.imread(parser.mask_paths[target_idx]) > 128

    # Projection logic
    worldtocams = np.linalg.inv(camtoworlds)
    points_world = parser.points
    points_cam = (worldtocams[:3, :3] @ points_world.T + worldtocams[:3, 3:4]).T
    points_proj = (K @ points_cam.T).T
    points = points_proj[:, :2] / points_proj[:, 2:3]
    depths = points_cam[:, 2]

    # Basic filter
    selector = (
        (points[:, 0] >= 0) &
        (points[:, 0] < image.shape[1]) &
        (points[:, 1] >= 0) &
        (points[:, 1] < image.shape[0]) &
        (depths > 0)
    )

    # Remove NaN / inf
    valid_mask = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    selector = selector & valid_mask

    # Safe integer conversion without warnings
    u = np.zeros_like(points[:, 0], dtype=np.int32)
    v = np.zeros_like(points[:, 1], dtype=np.int32)
    u[selector] = np.floor(points[selector, 0]).astype(np.int32)
    v[selector] = np.floor(points[selector, 1]).astype(np.int32)

    # Mask filtering
    mask_valid = np.zeros_like(selector, dtype=bool)
    mask_valid[selector] = mask_image[v[selector], u[selector]]

    # Final selection (invalid points are automatically False)
    final_selector = selector & mask_valid

    # # SAVE VALID WORLD POINTS (WHITE)
    # def save_ply_white(path, points):
    #     with open(path, 'w') as f:
    #         f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n" % len(points))
    #         for x, y, z in points:
    #             f.write(f"{x} {y} {z} 255 255 255\n")

    # save_ply_white("/home/zc/valid_world_points.ply", select_world_points)

    fliter_image = image.copy()
    points_in_img = points[final_selector]
    for (x, y) in points_in_img:
        xx = int(x)
        yy = int(y)
        fliter_image[yy, xx] = [255, 0, 0]
    
    vis_mask = np.stack([mask_image * 255] * 3, axis=-1).astype(np.uint8)


    all_valid_depth = image.copy()
    points_in_img = points[selector]
    for (x, y) in points_in_img:
        xx = int(x)
        yy = int(y)
        all_valid_depth[yy, xx] = [255, 0, 0]
    
    vis_mask = np.stack([mask_image * 255] * 3, axis=-1).astype(np.uint8)

    plt.subplot(1,4,1)
    plt.title(f"Original: {target_image_name}")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1,4,2)
    plt.title("Mask (white=valid)")
    plt.imshow(vis_mask)
    plt.axis('off')

    plt.subplot(1,4,3)
    plt.title(f"all valid depth")
    plt.imshow(all_valid_depth)
    plt.axis('off')

    plt.subplot(1,4,4)
    plt.title(f"fliter depth")
    plt.imshow(fliter_image)
    plt.axis('off')

    plt.tight_layout()
    save_path = "/home/zc/verify_mask_28377822.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import argparse

    import imageio.v2 as imageio

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/media/zc/SSD/ar/ABXBJQ_BJ/B3/small_section")
    parser.add_argument("--factor", type=int, default=4)
    args = parser.parse_args()

    # Parse COLMAP data.
    parser = Parser(
        data_dir=args.data_dir, factor=args.factor, normalize=False, test_every=8
    )
    dataset = Dataset(parser, split="train", load_depths=True)
    # print(f"Dataset: {len(dataset)} images.")
    # verify_mask_filter(dataset, "image5/20648796.png")
    # exit()
    writer = imageio.get_writer(f"{args.data_dir}/points.mp4", fps=30)
    for data in tqdm(dataset, desc="Plotting points"):
        image = data["image"].numpy().astype(np.uint8)
        points = data["points"].numpy()
        depths = data["depths"].numpy()
        for x, y in points:
            cv2.circle(image, (int(x), int(y)), 2, (255, 0, 0), -1)
        writer.append_data(image)
    writer.close()
