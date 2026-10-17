import unittest

import torch

from gsplat.strategy.weighted_anchor_mcmc import WeightedAnchorMCMCStrategy


class WeightedAnchorMCMCTest(unittest.TestCase):
    def test_duplicate_voxel_keys_choose_nearest_anchor(self):
        strategy = WeightedAnchorMCMCStrategy(
            anchor_voxel_size=1.0, verbose=False
        )
        anchors = torch.tensor(
            [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [1.1, 0.0, 0.0]]
        )
        means = torch.tensor(
            [[0.11, 0.0, 0.0], [0.19, 0.0, 0.0], [1.05, 0.0, 0.0]]
        )

        ids = strategy._voxel_anchor_ids(means, anchors)

        self.assertEqual(ids.tolist(), [0, 1, 2])

    def test_weight_is_preserved_and_used_for_region_ranking(self):
        strategy = WeightedAnchorMCMCStrategy(
            anchor_voxel_size=1.0,
            redundant_source_voxel_fraction=0.4,
            redundant_donor_voxel_fraction=0.2,
            verbose=False,
        )
        state = strategy.initialize_state()
        anchors = torch.tensor(
            [[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0],
             [3.1, 0.0, 0.0], [4.1, 0.0, 0.0]]
        )
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        strategy.set_anchors(state, anchors.clone(), anchors, weights)

        self.assertEqual(state["anchor_weight"].tolist(), weights.tolist())
        self.assertEqual(state["anchor_low_mask"].tolist(), [True, True, False, False, False])
        self.assertEqual(state["anchor_high_mask"].tolist(), [False, False, False, False, True])

    def test_equal_weight_class_is_not_split_between_regions(self):
        low, high, _, _ = WeightedAnchorMCMCStrategy._region_masks(
            torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0]), 0.5, 0.2
        )
        self.assertEqual(low.tolist(), [True, True, True, False, False])
        self.assertEqual(high.tolist(), [False, False, False, False, True])

    def test_minimum_gs_per_source_voxel(self):
        ids = torch.tensor([0, 0, 0, 1, 1])
        score = torch.tensor([0.1, 0.9, 0.5, 0.4, 0.8])
        keepers = WeightedAnchorMCMCStrategy._keeper_ids(
            ids, score, torch.ones(5, dtype=torch.bool), min_per_voxel=2
        )
        self.assertEqual(set(keepers.tolist()), {1, 2, 3, 4})

    def test_top_anchor_mass_is_independent_of_current_gs_count(self):
        weights = WeightedAnchorMCMCStrategy._top_region_sampling_weights(
            ids=torch.tensor([0, 0, 1]),
            donor_score=torch.tensor([1.0, 3.0, 2.0]),
            valid=torch.ones(3, dtype=torch.bool),
            anchor_weight=torch.tensor([2.0, 4.0]),
            high_mask=torch.tensor([True, True]),
        )

        torch.testing.assert_close(weights, torch.tensor([0.5, 1.5, 4.0]))
        torch.testing.assert_close(weights[:2].sum(), torch.tensor(2.0))
        torch.testing.assert_close(weights[2:].sum(), torch.tensor(4.0))

    def test_move_keeps_configured_minimum_in_bottom_voxel(self):
        if not torch.cuda.is_available():
            self.skipTest("gsplat relocation requires CUDA")
        torch.manual_seed(1)
        device = torch.device("cuda")
        strategy = WeightedAnchorMCMCStrategy(
            redundant_min_gs_per_source_voxel=2,
            redundant_opacity_threshold=0.01,
            verbose=False,
        )
        params = {
            "means": torch.zeros((7, 3), device=device),
            "scales": torch.zeros((7, 3), device=device),
            "opacities": torch.logit(
                torch.tensor(
                    [0.9, 0.005, 0.004, 0.003, 0.5, 0.5, 0.5],
                    device=device,
                )
            ),
        }
        state = strategy.initialize_state()
        state.update(
            anchor_xyz=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                device=device,
            ),
            anchor_weight=torch.tensor([1.0, 5.0, 10.0], device=device),
            anchor_capacity=torch.tensor([1, 5, 10], device=device),
            anchor_low_mask=torch.tensor([True, False, False], device=device),
            anchor_high_mask=torch.tensor([False, True, True], device=device),
            anchor_ids=torch.tensor([0, 0, 0, 0, 1, 1, 2], device=device),
            noise_active_until=torch.full(
                (7,), -1, dtype=torch.long, device=device
            ),
            history_score=torch.zeros(7, device=device),
        )

        moved = strategy.move_redundant_gs(params, {}, state, step=100)

        self.assertEqual(moved, 2)
        self.assertEqual(int((state["anchor_ids"] == 0).sum()), 2)
        self.assertTrue((state["anchor_ids"][2:4] > 0).all())

    def test_relocate_and_add_inherit_donor_anchor(self):
        strategy = WeightedAnchorMCMCStrategy(verbose=False)
        state = {"anchor_ids": torch.tensor([0, 1])}
        strategy._inherit_surface_anchors(
            state, torch.tensor([0]), torch.tensor([1])
        )
        self.assertEqual(state["anchor_ids"].tolist(), [1, 1])
        strategy._inherit_surface_anchors(
            state, torch.tensor([2]), torch.tensor([0]), append=True
        )
        self.assertEqual(state["anchor_ids"].tolist(), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
