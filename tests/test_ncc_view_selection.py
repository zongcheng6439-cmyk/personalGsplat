"""CPU tests for LiDAR view selection, independent of CUDA / SfM tracks."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from ncc_view_selection import (
    _depth_overlap, _pose_candidates, _sample_world_points, _small_depth,
    choose_depth_sources,
)


def depth_view(x=0.0, depth=5.0):
    pose = torch.eye(4)
    pose[0, 3] = x
    return dict(depth_gs=torch.full((64, 80), depth),
                depth_gs_mask=torch.ones(64, 80, dtype=torch.bool),
                mask=torch.ones(64, 80, dtype=torch.bool),
                K=torch.tensor([[100., 0., 40.], [0., 100., 32.], [0., 0., 1.]]),
                camtoworld=pose)


class DepthDataset:
    def __init__(self, folder):
        self.data = [depth_view(), depth_view(0.2), depth_view(0.1, depth=2.0)]
        self.indices = np.array([0, 2, 3])  # non-contiguous parser IDs
        self.calls = 0
        paths = []
        for i in range(4):
            path = folder / f"input_{i}.png"
            path.write_bytes(b"input")
            paths.append(str(path))
        poses = np.repeat(np.eye(4)[None], 4, axis=0)
        for index, data in zip(self.indices, self.data):
            poses[index] = data["camtoworld"].numpy()
        self.parser = SimpleNamespace(
            image_names=[f"image_{i}.jpg" for i in range(4)],
            image_paths=paths, depth_paths=paths, mask_paths=paths,
            camtoworlds=poses, camera_ids=[1] * 4,
            Ks_dict={1: self.data[0]["K"].numpy()}, params_dict={1: np.empty(0)}, factor=1,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        self.calls += 1
        return self.data[item]


def test_depth_overlap_rejects_occluded_and_invalid_sources():
    reference = _small_depth(depth_view(), 320)
    points = _sample_world_points(reference, 2048)
    assert _depth_overlap(points, reference) == 1.0
    assert _depth_overlap(points, _small_depth(depth_view(0.2), 320)) > 0.9
    assert _depth_overlap(points, _small_depth(depth_view(0.2, 2.0), 320)) == 0.0
    invalid = _small_depth(depth_view(0.2), 320)
    invalid["mask"][:] = False
    assert _depth_overlap(points, invalid) == 0.0
    assert _depth_overlap(np.empty((0, 3)), reference) == 0.0


def test_striding_scales_intrinsics_without_mixing_depth():
    data = depth_view()
    full = _small_depth(data, 320)
    small = _small_depth(data, 20)
    np.testing.assert_allclose(small["K"][:2], full["K"][:2] / 4)
    points = _sample_world_points(small, 2048)
    assert _depth_overlap(points, full) == 1.0
    data["depth_gs"][0, 0] = float("nan")
    data["depth_gs"][4, 4] = 0.0
    small = _small_depth(data, 20)
    assert not small["mask"][0, 0] and not small["mask"][1, 1]
    assert np.isfinite(_sample_world_points(small, 2048)).all()


def test_pose_candidates_exclude_self_coincident_distant_and_opposite():
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[1, 0, 3] = 0.2
    poses[2, 0, 3] = 0.001
    poses[3, 0, 3] = 31.0
    poses[4, 0, 3] = 0.1
    poses[4, :3, :3] = np.diag([-1., 1., -1.])
    ids = _pose_candidates(poses, 20, 0.05, 30.0, 75.0)
    np.testing.assert_array_equal(ids[0], [1])


def test_choose_sources_without_sfm_and_reuse_cache(tmp_path):
    dataset = DepthDataset(tmp_path)
    sources = choose_depth_sources(dataset, tmp_path)
    assert sources == {0: [1], 1: [0]}
    assert dataset.calls == 3
    saved = json.loads((tmp_path / "ncc_source_views.json").read_text())
    assert saved["sources"]["image_0.jpg"][0]["name"] == "image_2.jpg"
    assert saved["sources"]["image_3.jpg"] == []
    assert choose_depth_sources(dataset, tmp_path) == sources
    assert dataset.calls == 3  # no depth/image reads on a cache hit


def test_cache_invalidated_by_data_settings_pose_and_split(tmp_path):
    dataset = DepthDataset(tmp_path)
    choose_depth_sources(dataset, tmp_path)
    Path(dataset.parser.depth_paths[0]).write_bytes(b"changed input")
    choose_depth_sources(dataset, tmp_path)
    assert dataset.calls == 6
    choose_depth_sources(dataset, tmp_path, samples=1024)
    assert dataset.calls == 9
    dataset.parser.camtoworlds[0, 0, 3] += 0.01
    choose_depth_sources(dataset, tmp_path, samples=1024)
    assert dataset.calls == 12
    dataset.indices = np.array([2, 0, 3])
    choose_depth_sources(dataset, tmp_path, samples=1024)
    assert dataset.calls == 15


def test_empty_depth_has_no_fabricated_sources(tmp_path):
    dataset = DepthDataset(tmp_path)
    for data in dataset.data:
        data["depth_gs_mask"].zero_()
    assert choose_depth_sources(dataset, tmp_path) == {}


def test_choose_top_five_and_minimum_overlap(tmp_path):
    dataset = DepthDataset(tmp_path)
    dataset.data[2] = depth_view(0.1)
    sources = choose_depth_sources(dataset, tmp_path, keep=1)
    assert all(len(ids) == 1 for ids in sources.values())
    assert sources[0] == [2]  # closer translated view has more visible overlap
    assert choose_depth_sources(dataset, tmp_path, min_overlap=1.0) == {}
