import unittest
from unittest import mock

import torch

from gsplat.strategy.weighted_anchor_mcmc import WeightedAnchorMCMCStrategy


class WeightedAnchorNoiseScheduleTest(unittest.TestCase):
    def test_noise_scale_decays_to_ten_percent(self):
        strategy = WeightedAnchorMCMCStrategy(noise_scale=2.0)
        strategy.set_training_steps(1000)

        self.assertAlmostEqual(strategy.noise_scale_at_step(0), 2.0)
        self.assertAlmostEqual(
            strategy.noise_scale_at_step(500), 2.0 * 0.1**0.5
        )
        self.assertAlmostEqual(strategy.noise_scale_at_step(1000), 0.2)
        self.assertAlmostEqual(strategy.noise_scale_at_step(2000), 0.2)

    def test_noise_injection_does_not_use_means_lr(self):
        strategy = WeightedAnchorMCMCStrategy(noise_scale=2.0)
        strategy.set_training_steps(1000)
        strategy._active_step = 500
        params = {
            "means": torch.zeros(2, 3),
            "quats": torch.zeros(2, 4),
            "scales": torch.zeros(2, 3),
            "opacities": torch.zeros(2),
        }

        with mock.patch(
            "gsplat.strategy.weighted_anchor_mcmc.inject_noise_to_position"
        ) as inject:
            strategy._inject_active_noise(
                params, torch.tensor([1]), lr=12345.0
            )

        self.assertAlmostEqual(
            inject.call_args.kwargs["noise_scale"], 2.0 * 0.1**0.5
        )


if __name__ == "__main__":
    unittest.main()
