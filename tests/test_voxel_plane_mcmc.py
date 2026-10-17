import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from gsplat.strategy.voxel_plane_mcmc import VoxelPlaneMCMCStrategy
from gsplat.utils import normalized_quat_to_rotmat


class VoxelPlaneMCMCTest(unittest.TestCase):
    def test_gs_knn_has_eight_nonself_neighbors(self):
        strategy = VoxelPlaneMCMCStrategy(plane_knn=8)
        points = np.arange(30, dtype=np.float32).reshape(10, 3)
        neighbors = strategy._find_gs_knn(points)
        self.assertEqual(neighbors.shape, (10, 8))
        for index, row in enumerate(neighbors):
            self.assertNotIn(index, row)

    def test_relocated_scale_and_planar_shape_are_clamped(self):
        strategy = VoxelPlaneMCMCStrategy()
        strategy.plane_s3_max = 0.01
        half = 2.0**-0.5
        params = {
            "quats": torch.tensor(
                [[half, 0.0, half, 0.0], [half, 0.0, half, 0.0]]
            ),
            "scales": torch.full((2, 3), 0.2).log(),
        }
        state = {
            "anchor_ids": torch.tensor([0, 1]),
            "anchor_is_plane": torch.tensor([True, False]),
            "anchor_normals": torch.tensor(
                [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
            ),
        }

        strategy._constrain_planar_shape(params, state, torch.tensor([0, 1]))

        normal = normalized_quat_to_rotmat(params["quats"][:1])[0, :, 2]
        torch.testing.assert_close(normal, torch.tensor([0.0, 0.0, 1.0]))
        self.assertTrue((params["scales"].exp() <= 0.1).all())
        self.assertLessEqual(params["scales"][0, 2].exp().item(), 0.01)
        torch.testing.assert_close(
            params["quats"][1], torch.tensor([half, 0.0, half, 0.0])
        )

    def test_plane_slab_allows_small_normal_offset(self):
        strategy = VoxelPlaneMCMCStrategy(
            limit_anchor_distance=False,
            anchor_max_distance=1.0,
            plane_normal_tolerance=0.03,
        )
        params = {"means": torch.tensor([[0.3, 0.0, 0.02], [2.5, 0.0, 0.0]])}
        state = {
            "anchor_xyz": torch.zeros(2, 3),
            "anchor_normals": torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1),
            "anchor_is_plane": torch.tensor([True, False]),
            "anchor_ids": torch.tensor([0, 1]),
        }
        strategy._constrain_surface_anchored_gs(
            params, state, torch.tensor([0, 1])
        )
        torch.testing.assert_close(params["means"][0], torch.tensor([0.3, 0.0, 0.02]))
        torch.testing.assert_close(params["means"][1], torch.tensor([2.5, 0.0, 0.0]))

    def test_plane_slab_reflects_only_excess_normal_offset(self):
        strategy = VoxelPlaneMCMCStrategy(
            limit_anchor_distance=False, plane_normal_tolerance=0.03
        )
        params = {"means": torch.tensor([[0.3, 0.0, 0.04]])}
        state = {
            "anchor_xyz": torch.zeros(1, 3),
            "anchor_normals": torch.tensor([[0.0, 0.0, 1.0]]),
            "anchor_is_plane": torch.tensor([True]),
            "anchor_ids": torch.tensor([0]),
        }
        strategy._constrain_surface_anchored_gs(params, state, torch.tensor([0]))
        torch.testing.assert_close(params["means"][0], torch.tensor([0.3, 0.0, 0.02]))

    def test_optional_anchor_distance_uses_reflection(self):
        strategy = VoxelPlaneMCMCStrategy(
            limit_anchor_distance=True, anchor_max_distance=1.0
        )
        params = {"means": torch.tensor([[2.5, 0.0, 0.0]])}
        state = {
            "anchor_xyz": torch.zeros(1, 3),
            "anchor_normals": torch.zeros(1, 3),
            "anchor_is_plane": torch.tensor([False]),
            "anchor_ids": torch.tensor([0]),
        }
        strategy._constrain_surface_anchored_gs(params, state, torch.tensor([0]))
        torch.testing.assert_close(params["means"][0], torch.tensor([0.5, 0.0, 0.0]))

    def test_initial_short_axis_aligns_with_plane_normal(self):
        strategy = VoxelPlaneMCMCStrategy(
            initial_plane_thickness=1e-4, verbose=False
        )
        params = {
            "quats": torch.zeros(1, 4),
            "scales": torch.zeros(1, 3),
            "means": torch.zeros(1, 3),
        }
        state = {
            "anchor_ids": torch.tensor([0]),
            "anchor_is_plane": torch.tensor([True]),
            "anchor_normals": torch.tensor([[0.0, 1.0, 0.0]]),
            "anchor_xyz": torch.zeros(1, 3),
            "initial_tangent_scale": torch.tensor([0.02]),
        }
        strategy.initialize_plane_gaussians(params, state)
        rotation = normalized_quat_to_rotmat(params["quats"])
        torch.testing.assert_close(rotation[0, :, 2], state["anchor_normals"][0])
        torch.testing.assert_close(
            params["scales"].exp()[0], torch.tensor([0.02, 0.02, 1e-4])
        )

    def test_added_and_relocated_gs_inherit_anchor_id(self):
        strategy = VoxelPlaneMCMCStrategy()
        state = {"anchor_ids": torch.tensor([0, 1])}
        strategy._inherit_surface_anchors(
            state, torch.tensor([0]), torch.tensor([1])
        )
        strategy._inherit_surface_anchors(
            state, torch.tensor([2]), torch.tensor([0]), append=True
        )
        self.assertEqual(state["anchor_ids"].tolist(), [1, 1, 1])

    def test_dead_mask_keeps_one_initial_gs_per_anchor(self):
        strategy = VoxelPlaneMCMCStrategy(min_opacity=0.1)
        strategy._active_state = {"anchor_keeper_ids": torch.tensor([0, 2])}
        dead = strategy._dead_mask(torch.tensor([0.01, 0.01, 0.01, 0.2]))
        self.assertEqual(dead.tolist(), [False, True, False, False])

    def test_plane_debug_ply_contains_only_planar_gs(self):
        with TemporaryDirectory() as directory:
            strategy = VoxelPlaneMCMCStrategy(
                plane_debug_dir=directory, verbose=False
            )
            params = {
                "means": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
                "scales": torch.zeros(2, 3),
                "opacities": torch.zeros(2),
                "sh0": torch.zeros(2, 1, 3),
                "shN": torch.zeros(2, 15, 3),
            }
            state = {
                "anchor_ids": torch.tensor([0, 1]),
                "anchor_is_plane": torch.tensor([True, False]),
                "anchor_xyz": torch.zeros(2, 3),
                "anchor_normals": torch.tensor(
                    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
                ),
            }
            strategy._active_step = 10
            strategy._save_plane_gs(params, state, "add")
            path = Path(directory) / "plane_gs_000010_add.ply"
            self.assertTrue(path.is_file())
            from plyfile import PlyData

            vertices = PlyData.read(path)["vertex"]
            self.assertEqual(len(vertices), 1)
            self.assertEqual(float(vertices["rot_0"][0]), 1.0)
            self.assertIn("f_dc_0", vertices.data.dtype.names)
            self.assertIn("f_rest_44", vertices.data.dtype.names)

            from gsplat.exporter import load_ply_to_splats

            splats = load_ply_to_splats(str(path))
            self.assertEqual(tuple(splats["means"].shape), (1, 3))
            self.assertEqual(tuple(splats["shN"].shape), (1, 15, 3))


if __name__ == "__main__":
    unittest.main()
