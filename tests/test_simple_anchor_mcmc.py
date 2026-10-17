import unittest

import torch

from gsplat.strategy.simple_anchor_mcmc import SimpleAnchorMCMCStrategy


class SimpleAnchorMCMCTest(unittest.TestCase):
    def test_precomputed_counts_become_fixed_capacity(self):
        strategy = SimpleAnchorMCMCStrategy(verbose=False)
        state = strategy.initialize_state()
        means = torch.zeros((4, 3))
        strategy.set_precomputed_anchors(
            state,
            means,
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            torch.tensor([0, 0, 1, 1]),
        )

        self.assertEqual(state["anchor_capacity"].tolist(), [2, 2])
        self.assertEqual(strategy.cap_max, 4)
        self.assertEqual(
            state["noise_active_until"].tolist(), [-1, -1, -1, -1]
        )

    def test_add_is_disabled(self):
        strategy = SimpleAnchorMCMCStrategy(verbose=False)
        self.assertEqual(strategy._add_new_gs({}, {}, torch.empty(0)), 0)

    def test_capacity_move_stops_with_refinement(self):
        class SpyStrategy(SimpleAnchorMCMCStrategy):
            calls = 0

            def move_redundant_gs(self, params, optimizers, state, step):
                self.calls += 1
                return 0

        strategy = SpyStrategy(
            refine_start_iter=0,
            refine_stop_iter=10,
            refine_every=100,
            capacity_move_every=5,
            active_noise_steps=0,
            verbose=False,
        )
        params = {
            "means": torch.nn.Parameter(torch.zeros((1, 3))),
            "scales": torch.nn.Parameter(torch.zeros((1, 3))),
            "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]])),
            "opacities": torch.nn.Parameter(torch.zeros(1)),
        }
        state = strategy.initialize_state()
        strategy.set_precomputed_anchors(
            state, params["means"], torch.zeros((1, 3)), torch.zeros(1)
        )
        info = {"means2d": None}

        strategy.step_post_backward(params, {}, state, 5, info, 0.0)
        strategy.step_post_backward(params, {}, state, 10, info, 0.0)

        self.assertEqual(strategy.calls, 1)


if __name__ == "__main__":
    unittest.main()
