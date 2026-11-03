"""Opacity binarization loss used by train_voxel_plane_mcmc.py (由其他模块调用).

Input: Gaussian opacity logits; output: scalar Bernoulli entropy, in nats.
Example: loss = opacity_entropy_loss(splats["opacities"][visible])
Pure PyTorch; no CUDA extension compilation or gsplat reinstall is required.
"""
import torch
import torch.nn.functional as F
from torch import Tensor


def opacity_entropy_loss(opacities: Tensor) -> Tensor:
    """Mean Bernoulli entropy of sigmoid opacity logits (natural logarithm).

    Minimization encourages opacity towards 0 or 1. Unlike opacity_reg_loss,
    this does not prefer transparency. Pass visible logits to restrict scope.
    Evaluate H(sigmoid(z)) = softplus(-abs(z)) + abs(z)*sigmoid(-abs(z))
    to avoid log(0), hard probability clamps and large-logit cancellation.
    Empty inputs return a differentiable zero. Half inputs accumulate in fp32.
    """
    logits = opacities.float() if opacities.dtype in (torch.float16, torch.bfloat16) else opacities
    if logits.numel() == 0:
        return logits.sum() * 0.0
    magnitude = logits.abs()
    return (F.softplus(-magnitude) + magnitude * torch.sigmoid(-magnitude)).mean()

