import sys
import tempfile
import unittest
from pathlib import Path

import torch

TRAIN_UTIL_DIR = Path(__file__).resolve().parents[1] / "utils" / "train"
sys.path.insert(0, str(TRAIN_UTIL_DIR))

from block_utils import _filter_splats_for_block, find_latest_coarse_ply


class BlockUtilsTest(unittest.TestCase):
    def test_find_latest_coarse_ply_uses_largest_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step in (100, 300):
                path = root / "point_cloud" / f"iteration_{step}" / "point_cloud.ply"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            selected = find_latest_coarse_ply(str(root))

            self.assertIn("iteration_300", selected)

    def test_filter_splats_keeps_only_requested_block(self):
        with tempfile.TemporaryDirectory() as directory:
            partitions_path = Path(directory) / "partitions.pt"
            torch.save(
                {
                    "partition_coordinates": {
                        "id": torch.tensor([[0, 0]]),
                        "xy": torch.tensor([[0.0, 0.0]]),
                    },
                    "scene_config": {
                        "partition_size": 1.0,
                        "contract": False,
                    },
                    "extra_data": {"rotation_transform": torch.eye(4)},
                },
                partitions_path,
            )
            splats = {
                "means": torch.tensor([[0.5, 0.5, 0.0], [2.0, 2.0, 0.0]]),
                "opacities": torch.tensor([0.1, 0.2]),
            }

            selected = _filter_splats_for_block(
                splats,
                str(partitions_path),
                block_id=(0, 0),
                bbox_enlarge=0.0,
                bbox_margin=None,
            )

            torch.testing.assert_close(
                selected["means"], torch.tensor([[0.5, 0.5, 0.0]])
            )
            torch.testing.assert_close(selected["opacities"], torch.tensor([0.1]))


if __name__ == "__main__":
    unittest.main()
