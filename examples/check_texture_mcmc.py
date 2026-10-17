"""Small deterministic check for TextureMCMCStrategy sampling scores."""

import torch

from gsplat.strategy import TextureMCMCStrategy


def main():
    strategy = TextureMCMCStrategy(texture_weight=1.0, grad_weight=0.0)
    state = strategy.initialize_state()
    state["texture_info"] = torch.tensor([0.1, 1.0, 4.0])
    state["texture_pixel_num"] = torch.tensor([1.0, 2.0, 4.0])
    state["absgrad_sum"] = torch.tensor([100.0, 1.0, 1.0])
    state["grad_count"] = torch.ones(3)
    params = {
        "opacities": torch.nn.Parameter(
            torch.logit(torch.tensor([0.2, 0.8, 0.8]))
        )
    }

    texture = strategy._texture_signal(state, 3, params["opacities"])
    donor = strategy._donor_weights(params, state)
    assert donor[2] > donor[1] > donor[0]

    state_without_grad = dict(state, absgrad_sum=torch.zeros(3))
    donor_without_grad = strategy._donor_weights(params, state_without_grad)
    torch.testing.assert_close(donor, donor_without_grad)

    strategy.grad_weight = 1.0
    donor_with_grad = strategy._donor_weights(params, state)
    assert donor_with_grad[0] > donor[0]
    print("texture_signal:", texture.tolist())
    print("donor_weight (grad disabled):", donor.tolist())
    print("donor_weight (grad enabled):", donor_with_grad.tolist())
    print("Texture MCMC score check passed.")


if __name__ == "__main__":
    main()
