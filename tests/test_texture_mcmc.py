import unittest
from unittest import mock

import torch

from gsplat.strategy import TextureMCMCStrategy


def _params():
    return {
        "means": torch.nn.Parameter(
            torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        ),
        "scales": torch.nn.Parameter(torch.zeros(2, 3)),
        "quats": torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        ),
        "opacities": torch.nn.Parameter(torch.logit(torch.tensor([0.001, 0.9]))),
    }


def _identity_relocation(opacities, scales, ratios, binoms, min_opacity):
    del ratios, binoms, min_opacity
    return opacities, scales


class TextureMCMCStrategyTest(unittest.TestCase):
    @mock.patch(
        "gsplat.strategy.ops.compute_relocation", side_effect=_identity_relocation
    )
    def test_dead_relocation_inherits_anchor_below_and_at_cap(self, _relocation):
        for cap_max in (2, 10):
            with self.subTest(cap_max=cap_max):
                strategy = TextureMCMCStrategy(
                    cap_max=cap_max,
                    min_opacity=0.05,
                    surface_anchor_constraint=True,
                )
                params = _params()
                optimizers = {
                    name: torch.optim.Adam([value], lr=1e-3)
                    for name, value in params.items()
                }
                state = strategy.initialize_state()
                strategy.set_surface_anchors(state, params["means"])
                strategy._active_state = state
                strategy._active_step = 0
                try:
                    with mock.patch(
                        "torch.multinomial", return_value=torch.tensor([0])
                    ):
                        relocated = strategy._relocate_gs(
                            params, optimizers, state["binoms"]
                        )
                finally:
                    del strategy._active_state
                    del strategy._active_step

                self.assertEqual(relocated, 1)
                torch.testing.assert_close(
                    state["surface_anchor"][0],
                    torch.tensor([10.0, 0.0, 0.0]),
                )

if __name__ == "__main__":
    unittest.main()
