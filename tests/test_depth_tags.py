"""Regression tests: depth tags, metric units, filtering and dataset tensor reads.

From repository root: python tests/test_depth_tags.py
Requires numpy, Pillow, OpenCV and torch; uses temporary fixtures under .cache/.
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'examples'))
from datasets.depth_io import read_depth, read_training_depth, depth_settings

EDGE = ROOT / 'utils/data_process/depth_process/generate_depth_edge_masks.py'
spec = importlib.util.spec_from_file_location('edge_masks', EDGE)
edge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(edge)


class DepthTests(unittest.TestCase):
    def setUp(self):
        (ROOT / '.cache').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / '.cache')
        self.root = Path(self.tmp.name)
        self.raw = self.root / 'depth'
        self.raw.mkdir()
        self.output = self.root / 'depths_render_color_edge'

    def tearDown(self):
        self.tmp.cleanup()

    def test_uint16_precision_filter_and_training(self):
        values = np.array([[0, 1234, 1234, 5000, 5000]], dtype=np.uint16)
        path = self.raw / 'frame.png'
        Image.fromarray(values).save(path)
        depth, valid = read_depth(path, 'uint16', 1000)
        np.testing.assert_allclose(depth, values / 1000, atol=1e-6)
        self.assertFalse(valid[0, 0])
        edge.process_depth_map(path, self.raw, self.output, .2, .02,
                               np.ones((1, 1), np.uint8), False, 'uint16', 1000)
        np.testing.assert_array_equal(np.asarray(Image.open(self.output / path.name)),
                                      [[0, 1234, 0, 0, 5000]])
        settings = {'tag': 'uint16', 'scale': 1000}
        (self.output / 'depth_format.json').write_text(json.dumps(settings))
        tensor, mask = read_training_depth(self.output / path.name, depth_settings(self.output), (10, 2))
        self.assertEqual(tuple(tensor.shape), (2, 10))
        np.testing.assert_allclose(tensor[0, 2:4], 1.234)
        self.assertEqual(mask.sum().item(), 8)
        with self.assertRaises(ValueError):
            read_depth(path, 'la')

    def test_la_and_wrong_tag(self):
        raw = np.array([[[0, 255], [64, 255], [192, 255], [255, 255], [64, 0]]], np.uint8)
        path = self.raw / 'frame.png'
        Image.fromarray(raw).save(path)
        depth, valid = read_depth(path, 'la')
        np.testing.assert_allclose(depth[0, 1:3], [64 / 255 * 20, 5 / (1 - 192 / 255)], rtol=1e-6)
        np.testing.assert_array_equal(valid, [[False, True, True, False, False]])
        edge.process_depth_map(path, self.raw, self.output, 100, 100,
                               np.ones((1, 1), np.uint8), False)
        np.testing.assert_array_equal(np.asarray(Image.open(self.output / path.name))[..., 0], raw[..., 0])
        with self.assertRaises(ValueError):
            read_depth(path, 'uint16')

    def test_cli_metadata_and_stale_output(self):
        Image.fromarray(np.full((8, 8), 1234, np.uint16)).save(self.raw / 'frame.png')
        command = [sys.executable, str(EDGE), '--data-dir', str(self.root), '--input-dir', 'depth', '--depth-tag', 'uint16']
        subprocess.run(command, check=True, capture_output=True)
        self.assertEqual(depth_settings(self.output)['tag'], 'uint16')
        subprocess.run(command, check=True, capture_output=True)
        failed = subprocess.run(command + ['--depth-scale', '500'], capture_output=True, text=True)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn('settings changed', failed.stderr)
        subprocess.run(command + ['--depth-scale', '500', '--overwrite'], check=True, capture_output=True)
        self.assertEqual(depth_settings(self.output)['scale'], 500)


if __name__ == '__main__':
    unittest.main()
