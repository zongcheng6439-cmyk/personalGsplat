import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from gsplat.strategy import AnchorMCMCStrategy, read_anchor_capacity_ply


def _params(means, opacity):
    means = torch.tensor(means, dtype=torch.float32)
    opacity = torch.tensor(opacity, dtype=torch.float32)
    count = len(means)
    return {
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.zeros(count, 3)),
        "quats": torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1)
        ),
        "opacities": torch.nn.Parameter(torch.logit(opacity)),
    }


def _optimizers(params):
    return {name: torch.optim.Adam([value], lr=1e-3) for name, value in params.items()}


def _identity_relocation(opacities, scales, ratios, binoms, min_opacity):
    del ratios, binoms, min_opacity
    return opacities, scales


class AnchorMCMCStrategyTest(unittest.TestCase):
    def test_stage_one_ply_loads_and_binds_exact_capacities(self):
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        vertices = np.zeros(2, dtype=dtype)
        vertices["x"] = [0.0, 10.0]
        vertices["red"] = vertices["green"] = vertices["blue"] = [0, 255]
        header = "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "comment target_count_total 6",
                "comment target_count_lut 0 2",
                "comment target_count_lut 255 4",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "",
            ]
        ).encode("ascii")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.ply"
            with path.open("wb") as handle:
                handle.write(header)
                vertices.tofile(handle)

            xyz, capacity = read_anchor_capacity_ply(path)
            np.testing.assert_array_equal(capacity, [2, 4])
            strategy = AnchorMCMCStrategy(anchor_ply=str(path), cap_max=1)
            params = _params([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [0.5, 0.5])
            state = strategy.initialize_state()
            strategy.set_surface_anchors(state, params["means"])

        np.testing.assert_array_equal(xyz[:, 0], [0.0, 10.0])
        torch.testing.assert_close(state["anchor_capacity"], torch.tensor([2, 4]))
        torch.testing.assert_close(state["anchor_ids"], torch.tensor([0, 1]))
        self.assertEqual(strategy.cap_max, 6)

    def test_score_has_normal_and_fastgs_modes(self):
        strategy = AnchorMCMCStrategy(
            texture_weight=1.0,
            grad_weight=1.0,
            opacity_weight=1.0,
            loss_count_weight=1.0,
            texture_quantile=1.0,
            grad_quantile=1.0,
        )
        state = strategy.initialize_state()
        state["texture_info"] = torch.tensor([1.0, 4.0])
        state["texture_pixel_num"] = torch.tensor([1.0, 2.0])
        state["absgrad_sum"] = torch.tensor([4.0, 1.0])
        state["grad_count"] = torch.tensor([2.0, 1.0])
        params = {"opacities": torch.nn.Parameter(torch.logit(torch.tensor([0.2, 0.8])))}

        # texture=[0.5, 1], grad=[1, 0.5], opacity=[0.2, 0.8]
        torch.testing.assert_close(
            strategy._donor_weights(params, state), torch.tensor([1.7, 2.3])
        )

        # Normal mode ignores loss_count.  FastGS mode adds its normalized
        # signal while keeping opacity, gradient and texture unchanged.
        state["loss_count"] = torch.tensor([4.0, 10.0])
        torch.testing.assert_close(
            strategy._donor_weights(params, state), torch.tensor([1.7, 2.3])
        )
        strategy.fastgs_loss = True
        torch.testing.assert_close(
            strategy._donor_weights(params, state), torch.tensor([1.7, 3.3])
        )

        # Missing texture must not erase a valid gradient signal.
        state["texture_info"] = None
        state["texture_pixel_num"] = None
        torch.testing.assert_close(
            strategy._donor_weights(params, state), torch.tensor([1.2, 2.3])
        )

    def test_fastgs_mode_collects_loss_count_and_absgrad(self):
        strategy = AnchorMCMCStrategy(fastgs_loss=True, absgrad=True)
        params = _params(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.5, 0.5]
        )
        state = strategy.initialize_state()
        means2d = SimpleNamespace(
            absgrad=torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
        )

        strategy._update_statistics(
            params,
            state,
            {
                "texture_info": torch.tensor([1.0, 2.0]),
                "texture_pixel_num": torch.ones(2),
                "loss_count": torch.tensor([4.0, 7.0]),
                "means2d": means2d,
                "radii": torch.ones(1, 2, 2),
            },
        )

        torch.testing.assert_close(state["loss_count"], torch.tensor([4.0, 7.0]))
        torch.testing.assert_close(state["absgrad_sum"], torch.tensor([5.0, 2.0]))
        torch.testing.assert_close(state["grad_count"], torch.ones(2))

    def test_anchor_radius_projects_every_gaussian(self):
        strategy = AnchorMCMCStrategy(surface_anchor_max_distance=0.25)
        params = _params(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [0.5, 0.5]
        )
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            torch.tensor([1, 1]),
        )
        original_ids = state["anchor_ids"].clone()
        with torch.no_grad():
            params["means"][0] = torch.tensor([2.0, 0.0, 0.0])
            params["means"][1] = torch.tensor([float("nan"), 0.0, 0.0])
        strategy._constrain_surface_anchored_gs(params, state)

        centers = state["anchor_xyz"][state["anchor_ids"]]
        distance = (params["means"] - centers).norm(dim=-1)
        self.assertTrue(torch.all(distance <= 0.25 + 1e-6))
        self.assertTrue(torch.isfinite(params["means"]).all())
        torch.testing.assert_close(params["means"][0], torch.tensor([0.25, 0.0, 0.0]))
        torch.testing.assert_close(state["anchor_ids"], original_ids)

    @mock.patch("gsplat.strategy.ops.compute_relocation", side_effect=_identity_relocation)
    def test_active_relocate_ignores_excess_and_add_inherits_donor_anchor(
        self, _mock_relocation
    ):
        torch.manual_seed(4)
        strategy = AnchorMCMCStrategy(
            cap_max=6,
            min_opacity=0.05,
            add_growth_rate=1.0,
            surface_anchor_max_distance=0.25,
        )
        params = _params(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            [0.9, 0.8, 0.2, 0.9],
        )
        optimizers = _optimizers(params)
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            torch.tensor([2, 4]),
            torch.tensor([0, 0, 0, 1]),
        )
        strategy._active_state = state
        try:
            relocated = strategy._relocate_gs(
                params, optimizers, state["binoms"], scene=None
            )
            self.assertEqual(relocated, 0)
            # Capacity excess is intentionally left untouched during ACTIVE.
            torch.testing.assert_close(
                strategy.anchor_counts(state), torch.tensor([3, 1])
            )

            with mock.patch(
                "torch.multinomial", return_value=torch.tensor([0, 1])
            ) as sample_mock:
                added = strategy._add_new_gs(
                    params, optimizers, state["binoms"], scene=None
                )
            self.assertEqual(added, 2)
            sample_mock.assert_called_once()
            # New points inherit the score-sampled donor's anchor even though
            # that voxel is already above its final capacity.
            torch.testing.assert_close(
                state["anchor_ids"], torch.tensor([0, 0, 0, 1, 0, 0])
            )
            torch.testing.assert_close(
                strategy.anchor_counts(state), torch.tensor([5, 1])
            )
            self.assertEqual(len(params["means"]), 6)
            self.assertEqual(len(state["anchor_ids"]), 6)
            self.assertEqual(
                strategy._add_new_gs(
                    params, optimizers, state["binoms"], scene=None
                ),
                0,
            )
        finally:
            del strategy._active_state

        strategy._constrain_surface_anchored_gs(params, state)
        centers = state["anchor_xyz"][state["anchor_ids"]]
        self.assertTrue(
            torch.all((params["means"] - centers).norm(dim=-1) <= 0.25 + 1e-6)
        )

    def test_capacity_is_deferred_until_stop_and_moves_lowest_opacity(self):
        torch.manual_seed(7)
        strategy = AnchorMCMCStrategy(
            cap_max=6,
            min_opacity=0.05,
            refine_start_iter=0,
            refine_stop_iter=2,
            refine_every=1,
            noise_injection_stop_iter=0,
            add_growth_rate=0.0,
        )
        params = _params(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [0.9, 0.8, 0.2, 0.9],
        )
        optimizers = _optimizers(params)
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            torch.tensor([2, 2]),
            torch.tensor([0, 0, 0, 1]),
        )
        self.assertFalse(state["final_capacity_rebalance_done"])

        loss = sum(value.square().sum() for value in params.values())
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        strategy.step_post_backward(
            params, optimizers, state, step=1, info={}, lr=1e-3
        )
        torch.testing.assert_close(strategy.anchor_counts(state), torch.tensor([3, 1]))
        self.assertFalse(state["final_capacity_rebalance_done"])

        # GS 2 has the lowest opacity but deliberately receives the highest
        # texture score. Capacity repair must still select it as redundant.
        state["texture_info"] = torch.tensor([0.0, 0.0, 100.0, 0.0])
        state["texture_pixel_num"] = torch.ones(4)
        state["absgrad_sum"] = torch.zeros(4)
        state["grad_count"] = torch.ones(4)
        state["loss_count"] = torch.zeros(4)

        strategy.step_post_backward(
            params, optimizers, state, step=2, info={}, lr=1e-3
        )
        torch.testing.assert_close(strategy.anchor_counts(state), torch.tensor([2, 2]))
        self.assertTrue(state["final_capacity_rebalance_done"])
        self.assertEqual(int(state["anchor_ids"][2]), 1)
        torch.testing.assert_close(params["means"][2], torch.tensor([10.0, 0.0, 0.0]))
        for key in (
            "texture_info",
            "texture_pixel_num",
            "absgrad_sum",
            "grad_count",
            "loss_count",
        ):
            torch.testing.assert_close(state[key], torch.zeros(4))
        for name, optimizer in optimizers.items():
            optimizer_state = optimizer.state[params[name]]
            torch.testing.assert_close(
                optimizer_state["exp_avg"][2],
                torch.zeros_like(optimizer_state["exp_avg"][2]),
            )
            torch.testing.assert_close(
                optimizer_state["exp_avg_sq"][2],
                torch.zeros_like(optimizer_state["exp_avg_sq"][2]),
            )

        ids_after_final = state["anchor_ids"].clone()
        strategy.step_post_backward(
            params,
            optimizers,
            state,
            step=3,
            info={
                "texture_info": torch.ones(4),
                "texture_pixel_num": torch.ones(4),
            },
            lr=1e-3,
        )
        torch.testing.assert_close(state["anchor_ids"], ids_after_final)
        self.assertEqual(strategy._last_capacity_relocated, 0)
        torch.testing.assert_close(state["texture_info"], torch.zeros(4))

    def test_periodic_capacity_rebalance_is_configurable_and_not_final(self):
        strategy = AnchorMCMCStrategy(
            min_opacity=0.05,
            refine_start_iter=0,
            refine_stop_iter=10,
            refine_every=100,
            noise_injection_stop_iter=0,
            add_growth_rate=0.0,
            capacity_rebalance_every=2,
        )
        params = _params(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [10.0, 0.0, 0.0],
            ],
            [0.9, 0.8, 0.2, 0.9],
        )
        optimizers = _optimizers(params)
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            torch.tensor([2, 2]),
            torch.tensor([0, 0, 0, 1]),
        )

        strategy.step_post_backward(
            params, optimizers, state, step=1, info={}, lr=1e-3
        )
        torch.testing.assert_close(strategy.anchor_counts(state), torch.tensor([3, 1]))

        strategy.step_post_backward(
            params, optimizers, state, step=2, info={}, lr=1e-3
        )
        torch.testing.assert_close(strategy.anchor_counts(state), torch.tensor([2, 2]))
        self.assertEqual(int(state["anchor_ids"][2]), 1)
        self.assertEqual(strategy._last_capacity_relocated, 1)
        self.assertFalse(state["final_capacity_rebalance_done"])

        # A periodic repair must not freeze later score accumulation.
        strategy.step_post_backward(
            params,
            optimizers,
            state,
            step=3,
            info={
                "texture_info": torch.ones(4),
                "texture_pixel_num": torch.ones(4),
            },
            lr=1e-3,
        )
        torch.testing.assert_close(state["texture_info"], torch.ones(4))

    def test_final_capacity_move_handles_zero_capacity_source_anchor(self):
        strategy = AnchorMCMCStrategy(min_opacity=0.05)
        params = _params(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
            [0.001, 0.001],
        )
        optimizers = _optimizers(params)
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            torch.tensor([0, 2]),
            torch.tensor([0, 0]),
        )

        moved = strategy.finalize_anchor_capacity(params, optimizers, state)

        self.assertEqual(moved, 2)
        self.assertTrue(state["final_capacity_rebalance_done"])
        torch.testing.assert_close(strategy.anchor_counts(state), torch.tensor([0, 2]))
        torch.testing.assert_close(
            params["means"], torch.tensor([[10.0, 0.0, 0.0]]).repeat(2, 1)
        )
        self.assertEqual(
            strategy.finalize_anchor_capacity(params, optimizers, state), 0
        )

    @mock.patch("gsplat.strategy.ops.compute_relocation", side_effect=_identity_relocation)
    def test_scheduled_add_uses_score_donor_and_ignores_voxel_capacity(
        self, _mock_relocation
    ):
        strategy = AnchorMCMCStrategy(
            refine_start_iter=0,
            refine_stop_iter=10,
            refine_every=1,
            noise_injection_stop_iter=0,
            add_growth_rate=1.0,
        )
        params = _params(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            [0.9, 0.9],
        )
        optimizers = _optimizers(params)
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            torch.tensor([1, 3]),
            torch.tensor([0, 1]),
        )
        scene = mock.Mock()
        loss = sum(value.square().sum() for value in params.values())
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        with mock.patch(
            "torch.multinomial", return_value=torch.tensor([0, 1])
        ) as sample_mock:
            strategy.step_post_backward(
                params,
                optimizers,
                state,
                step=1,
                info={},
                lr=1e-3,
                scene=scene,
            )

        sample_mock.assert_called_once()
        torch.testing.assert_close(strategy.anchor_counts(state), torch.tensor([2, 2]))
        torch.testing.assert_close(state["anchor_ids"], torch.tensor([0, 1, 0, 1]))
        self.assertFalse(state["final_capacity_rebalance_done"])
        self.assertEqual(len(params["means"]), 4)
        scene.on_sample_add.assert_called_once()
        for name, optimizer in optimizers.items():
            optimizer_state = optimizer.state[params[name]]
            reset_indices = torch.tensor([0, 1, 2, 3])
            torch.testing.assert_close(
                optimizer_state["exp_avg"][reset_indices],
                torch.zeros_like(optimizer_state["exp_avg"][reset_indices]),
            )
            torch.testing.assert_close(
                optimizer_state["exp_avg_sq"][reset_indices],
                torch.zeros_like(optimizer_state["exp_avg_sq"][reset_indices]),
            )

    @mock.patch("gsplat.strategy.ops.compute_relocation", side_effect=_identity_relocation)
    def test_dead_point_inherits_global_donor_anchor(self, _mock_relocation):
        strategy = AnchorMCMCStrategy(
            cap_max=2,
            min_opacity=0.05,
            add_growth_rate=0.0,
        )
        params = _params([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], [0.001, 0.9])
        optimizers = _optimizers(params)
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            torch.tensor([1, 1]),
            torch.tensor([0, 1]),
        )
        strategy._active_state = state
        try:
            self.assertEqual(
                strategy._relocate_gs(params, optimizers, state["binoms"]), 1
            )
        finally:
            del strategy._active_state
        torch.testing.assert_close(strategy.anchor_counts(state), torch.tensor([0, 2]))
        torch.testing.assert_close(state["anchor_ids"], torch.tensor([1, 1]))
        torch.testing.assert_close(params["means"][0], torch.tensor([10.0, 0.0, 0.0]))

    @mock.patch("gsplat.strategy.ops.compute_relocation", side_effect=_identity_relocation)
    def test_all_dead_reports_scene_optimization_failure(
        self, _mock_relocation
    ):
        strategy = AnchorMCMCStrategy(min_opacity=0.05, add_growth_rate=0.0)
        params = _params([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.001, 0.001])
        optimizers = _optimizers(params)
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.tensor([1, 1]),
            torch.tensor([0, 1]),
        )
        strategy._active_state = state
        try:
            with self.assertRaisesRegex(RuntimeError, "optimization has failed"):
                strategy._relocate_gs(params, optimizers, state["binoms"])
        finally:
            del strategy._active_state

        _mock_relocation.assert_not_called()
        torch.testing.assert_close(state["anchor_ids"], torch.tensor([0, 1]))

    @mock.patch("gsplat.strategy.ops.compute_relocation", side_effect=_identity_relocation)
    def test_all_dead_candidates_are_processed_without_using_dead_donors(
        self, _mock_relocation
    ):
        strategy = AnchorMCMCStrategy(
            min_opacity=0.05,
            add_growth_rate=0.0,
        )
        params = _params(
            [[float(i), 0.0, 0.0] for i in range(10)],
            [0.001] * 8 + [0.9, 0.8],
        )
        optimizers = _optimizers(params)
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]),
            torch.tensor([8, 2]),
            torch.tensor([0] * 8 + [1, 1]),
        )
        strategy._active_state = state
        try:
            relocated = strategy._relocate_gs(
                params, optimizers, state["binoms"]
            )
        finally:
            del strategy._active_state

        self.assertEqual(relocated, 8)
        self.assertEqual(strategy._last_dead_relocated, 8)
        self.assertEqual(
            int((torch.sigmoid(params["opacities"]) < 0.05).sum().item()), 0
        )
        torch.testing.assert_close(strategy.anchor_counts(state), torch.tensor([0, 10]))

    @mock.patch("gsplat.strategy.ops.compute_relocation", side_effect=_identity_relocation)
    def test_relocation_keeps_parameter_and_anchor_donor_mapping_aligned(
        self, _mock_relocation
    ):
        strategy = AnchorMCMCStrategy(min_opacity=0.05, add_growth_rate=0.0)
        params = _params(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0],
             [2.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            [0.002, 0.8, 0.001, 0.9],
        )
        optimizers = _optimizers(params)
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor(
                [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [30.0, 0.0, 0.0]]
            ),
            torch.tensor([2, 1, 1]),
            torch.tensor([0, 1, 0, 2]),
        )
        scene = mock.Mock()
        strategy._active_state = state
        try:
            with mock.patch(
                "torch.multinomial",
                return_value=torch.tensor([0, 1]),
            ):
                relocated = strategy._relocate_gs(
                    params, optimizers, state["binoms"], scene=scene
                )
        finally:
            del strategy._active_state

        self.assertEqual(relocated, 2)
        torch.testing.assert_close(
            params["means"][[0, 2]],
            torch.tensor([[10.0, 0.0, 0.0], [30.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(state["anchor_ids"], torch.tensor([1, 1, 2, 2]))
        callback_sources, callback_donors = scene.on_relocate.call_args.args
        torch.testing.assert_close(callback_sources, torch.tensor([0, 2]))
        torch.testing.assert_close(callback_donors, torch.tensor([1, 3]))

    def test_relocation_floor_is_not_immediately_classified_as_dead(self):
        strategy = AnchorMCMCStrategy(min_opacity=0.005, add_growth_rate=0.0)
        params = _params([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.005, 0.9])
        optimizers = _optimizers(params)
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.tensor([1, 1]),
            torch.tensor([0, 1]),
        )
        strategy._active_state = state
        try:
            relocated = strategy._relocate_gs(
                params, optimizers, state["binoms"]
            )
        finally:
            del strategy._active_state

        self.assertEqual(relocated, 0)
        torch.testing.assert_close(state["anchor_ids"], torch.tensor([0, 1]))

    def test_noise_is_clamped_and_does_not_change_anchor_ids(self):
        strategy = AnchorMCMCStrategy(
            refine_start_iter=500,
            surface_anchor_max_distance=0.1,
        )
        params = _params([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.5, 0.5])
        state = strategy.initialize_state()
        strategy.set_anchors(
            state,
            params["means"],
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.tensor([1, 1]),
            torch.tensor([0, 1]),
        )
        original_ids = state["anchor_ids"].clone()

        def move_far(**kwargs):
            with torch.no_grad():
                kwargs["params"]["means"].add_(100.0)

        with mock.patch(
            "gsplat.strategy.mcmc.inject_noise_to_position", side_effect=move_far
        ):
            strategy.step_post_backward(
                params,
                _optimizers(params),
                state,
                step=0,
                info={},
                lr=1e-3,
            )

        centers = state["anchor_xyz"][state["anchor_ids"]]
        self.assertTrue(
            torch.all((params["means"] - centers).norm(dim=-1) <= 0.1 + 1e-6)
        )
        torch.testing.assert_close(state["anchor_ids"], original_ids)

    def test_invalid_capacity_and_missing_setup_fail_early(self):
        params = _params([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.5, 0.5])
        strategy = AnchorMCMCStrategy(cap_max=10)
        state = strategy.initialize_state()
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            strategy.set_anchors(
                state,
                params["means"],
                torch.tensor([[0.0, 0.0, 0.0]]),
                torch.tensor([1]),
                torch.tensor([0, 0]),
            )
        with self.assertRaisesRegex(RuntimeError, "call set_anchors"):
            strategy.step_post_backward(
                params, _optimizers(params), strategy.initialize_state(), 0, {}, 1e-3
            )
        with self.assertRaisesRegex(ValueError, "requires absgrad=True"):
            AnchorMCMCStrategy(absgrad=False).check_sanity(
                params, _optimizers(params)
            )


if __name__ == "__main__":
    unittest.main()
