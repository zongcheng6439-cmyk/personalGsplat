"""COLMAP dataset called by training scripts; input scene, output camera/RGB/depth tensors.
Depth format is read from depths_render_color_edge/depth_format.json (legacy: LA).
Example caller: examples/simple_trainer.py via Dataset and Parser.
"""
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

from datasets.depth_io import resolve_depth_source, read_training_depth

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
        block_cameras_json: Optional[str] = None,
        depth_tag: str = "auto",
        depth_dir: Optional[str] = None,
        depth_scale: Optional[float] = None,
    ):
        self.data_dir = data_dir
        self.factor = factor
        self.normalize = normalize
        self.test_every = test_every
        self.load_exposure = load_exposure

        selected_images = None
        if block_cameras_json:
            with open(block_cameras_json, "r", encoding="utf-8") as file:
                selected_images = {row["img_name"] for row in json.load(file)}

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
            if selected_images is not None and im.name not in selected_images:
                continue

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

        if not image_names:
            raise ValueError("No existing COLMAP images matched the dataset selection.")
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
        
        def mask_path(mask_dir: Path, name: str) -> Path:
            image_relative = Path(colmap_to_image[name])
            candidates = (
                mask_dir / image_relative,
                mask_dir / image_relative.with_suffix(".png"),
                mask_dir / Path(name).with_suffix(".png"),
            )
            return next((path for path in candidates if path.is_file()), candidates[0])

        mask_dir = Path(data_dir) / "masks"
        if factor > 1:
            resized_mask_dir = Path(data_dir) / f"masks_{factor}_png"
            if not all(
                mask_path(resized_mask_dir, name).is_file() for name in image_names
            ):
                _resize_image_folder(
                    str(mask_dir),
                    str(resized_mask_dir),
                    factor=factor,
                    resize_mask=True,
                    skip_existing=True,
                )
            mask_dir = resized_mask_dir

        mask_paths = [str(mask_path(mask_dir, name)) for name in image_names]

        depths_dir, self.depth_settings = resolve_depth_source(
            data_dir, depth_dir, depth_tag, depth_scale
        )
        print(f"[Depth] directory={depths_dir}, tag={self.depth_settings['tag']}, "
              f"scale={self.depth_settings['scale']}")
        depths_paths = [
            os.path.join(depths_dir, os.path.splitext(colmap_to_image[f])[0] + ".png")
            for f in image_names
        ]
        # Per-pixel texture/gradient maps. Preserve relative image subfolders and
        # replace the image extension with .npy.
        weight_map_dir = next(
            (
                path
                for path in (
                    Path(data_dir) / "weight_map",
                    Path(data_dir) / "weight_maps",
                )
                if path.is_dir()
            ),
            Path(data_dir) / "weight_map",
        )
        weight_paths = []
        for name, image_path in zip(image_names, image_paths):
            actual_name = Path(os.path.relpath(image_path, image_dir))
            candidates = (
                weight_map_dir / Path(name).with_suffix(".npy"),
                weight_map_dir / actual_name.with_suffix(".npy"),
                weight_map_dir / f"{Path(name).stem}.npy",
                weight_map_dir / f"{actual_name.stem}.npy",
            )
            weight_paths.append(
                str(next((path for path in candidates if path.is_file()), candidates[0]))
            )

        # Initial 3D points.
        points, _, points_rgb = reconstruction_points(
            reconstruction, _as_dict
        )

        # read lidar pc 
        if os.path.exists(os.path.join(colmap_dir, 'extracted.ply')):
            points, points_rgb, _ = fetchPly(os.path.join(colmap_dir, 'extracted.ply'))

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
        self.weight_paths = weight_paths

        self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
        self.camera_ids = camera_ids  # List[int], (num_images,)
        self.Ks_dict = Ks_dict  # Dict of camera_id -> K
        self.params_dict = params_dict  # Dict of camera_id -> params
        self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
        self.mask_dict = mask_dict  # Dict of camera_id -> mask
        self.points = points  # np.ndarray, (num_points, 3)
        self.points_rgb = points_rgb  # np.ndarray, (num_points, 3)
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
            from exif import compute_exposure_from_exif

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
        load_weight_maps: bool = False,
    ):
        self.parser = parser
        self.split = split
        self.patch_size = patch_size
        self.load_depths = load_depths
        self.load_weight_maps = load_weight_maps
        indices = np.arange(len(self.parser.image_names))
        if self.parser.test_every <= 0:
            self.indices = indices if split == "train" else indices[:0]
        elif split == "train":
            self.indices = indices[indices % self.parser.test_every != 0]
        else:
            self.indices = indices[indices % self.parser.test_every == 0]

        self.image_names = [self.parser.image_names[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item: int) -> Dict[str, Any]:     
        index = self.indices[item]
        image = imageio.imread(self.parser.image_paths[index])[..., :3]
        H, W  = image.shape[0], image.shape[1]
        weight_map = None
        if self.load_weight_maps:
            weight_path = self.parser.weight_paths[index]
            if not os.path.isfile(weight_path):
                raise FileNotFoundError(
                    f"Missing weight map for {self.parser.image_names[index]}: "
                    f"{weight_path}"
                )
            weight_map = np.asarray(np.load(weight_path), dtype=np.float32)
            weight_map = np.squeeze(weight_map)
            if weight_map.ndim == 3:
                weight_map = weight_map.mean(axis=-1)
            if weight_map.ndim != 2:
                raise ValueError(
                    f"Weight map must be [H,W] (optionally singleton/RGB), "
                    f"got {weight_map.shape} from {weight_path}"
                )
        mask_image = imageio.imread(self.parser.mask_paths[index])
        if mask_image.ndim == 3:
            mask_image = mask_image[..., 0]
        mask_image = mask_image > 128

        if self.load_depths:
            depth_gs, depth_gs_mask = read_training_depth(
                self.parser.depth_paths[index], self.parser.depth_settings, (W, H)
            )

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
            mask_image = cv2.remap(
                mask_image.astype(np.uint8), mapx, mapy, cv2.INTER_NEAREST
            )[y : y + h, x : x + w].astype(bool)
            if weight_map is not None:
                # A map matching the pre-undistorted image follows the same remap.
                # A map already matching the final training image is used as-is.
                if weight_map.shape == (H, W):
                    weight_map = cv2.remap(
                        weight_map, mapx, mapy, cv2.INTER_LINEAR
                    )[y : y + h, x : x + w]
                elif weight_map.shape != image.shape[:2]:
                    weight_map = cv2.resize(
                        weight_map,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
            if self.load_depths:
                depth_gs = depth_gs[y : y + h, x : x + w]
                depth_gs_mask = depth_gs_mask[y : y + h, x : x + w]
            # mask_image = cv2.resize(mask_image, (H, W), interpolation=cv2.INTER_CUBIC)

        if mask is not None:
            if mask.shape != mask_image.shape:
                raise ValueError(
                    f"camera-valid mask shape {mask.shape} does not match "
                    f"person mask shape {mask_image.shape}"
                )
            mask_image = np.asarray(mask, dtype=bool) & mask_image

        if self.patch_size is not None:
            # Random crop.
            h, w = image.shape[:2]
            x = np.random.randint(0, max(w - self.patch_size, 1))
            y = np.random.randint(0, max(h - self.patch_size, 1))
            image = image[y : y + self.patch_size, x : x + self.patch_size]
            if weight_map is not None:
                weight_map = weight_map[
                    y : y + self.patch_size, x : x + self.patch_size
                ]
            mask_image = mask_image[
                y : y + self.patch_size, x : x + self.patch_size
            ]
            if self.load_depths:
                depth_gs = depth_gs[
                    y : y + self.patch_size, x : x + self.patch_size
                ]
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
        if weight_map is not None:
            if weight_map.shape != image.shape[:2]:
                weight_map = cv2.resize(
                    weight_map,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            data["weight_map"] = torch.from_numpy(
                np.ascontiguousarray(weight_map)
            ).float()
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

def warpped_depth_inv(depth):
    return torch.where(depth < 0.5, 2 * depth * 10.0, 10 / (1.0 - depth) / 2)
