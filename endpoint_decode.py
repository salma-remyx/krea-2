"""Truncated Jump Sampling (TJS) endpoint decoder for the K2 flow sampler.

On the affine flow-matching path used by ``sampling.sample`` an intermediate
latent and the model's path velocity together pin down a minimum-MSE estimate
of the clean sample ``x_0``. Truncated Jump Sampling exploits this: stop the
ODE at an early-exit time ``t*`` and return the decoded endpoint instead of
running every remaining Euler step. This cuts neural function evaluations
(NFEs) with no retraining, distillation, or change to the model — a training-
free acceleration that applies equally to the multi-step RAW and few-step
Turbo paths.

Path convention (matches ``sampling.sample``):

    x_t = (1 - t) * x_0 + t * x_1      # x_0 = clean image at t=0, x_1 = noise at t=1
    v   = d x_t / d t = x_1 - x_0       # the velocity the MMDiT predicts

Solving for the t=0 (clean) endpoint from an observed (x_t, v):

    x_0 = x_t - t * v

This is the endpoint-decodability identity — the model's velocity prediction
*is* an x-prediction once placed on the affine path.

Adapted from "x-Prediction Is All You Need: Training-Free Accelerated
Generation via Endpoint Decodability" (arXiv:2607.06114). Only the core
algebraic decode and the early-exit decision are implemented here; the decode
reuses the existing CFG-combined velocity from ``sampling.sample``.
"""

from __future__ import annotations

import torch


def decode_endpoint(x_t: torch.Tensor, velocity: torch.Tensor, t: float) -> torch.Tensor:
    """Estimate the clean (t=0) latent from an intermediate state and velocity.

    On the K2 affine path ``x_t = (1 - t) x_0 + t x_1`` with velocity
    ``v = x_1 - x_0``, the minimum-MSE endpoint estimate is ``x_0 = x_t - t * v``.
    ``velocity`` should be the same CFG-combined velocity the Euler step uses, so
    the decode is consistent with the trajectory being integrated.

    Args:
        x_t: Patchified latent at time ``t``, shape ``(b, n_tokens, dim)``.
        velocity: Path velocity at ``(x_t, t)``, same shape as ``x_t``.
        t: Current timestep (decreasing 1 -> 0 along the schedule).

    Returns:
        The decoded clean-sample estimate ``x_0`` (same shape as ``x_t``).
    """
    return x_t - t * velocity


def early_exit_timestep(ts: list[float], t_star: float | None) -> int | None:
    """Index of the first scheduled timestep at or below the early-exit time.

    The sampler integrates ``t: 1 -> 0``; once ``tcurr`` drops to ``<= t_star``
    we stop and decode the endpoint instead of taking another Euler step.

    Args:
        ts: Decreasing timestep schedule ``[1.0, ..., 0.0]``.
        t_star: Early-exit time in ``(0, 1]`` (``None`` disables early exit).

    Returns:
        The smallest ``i`` with ``ts[i] <= t_star``, or ``None`` when early exit
        is disabled or no scheduled timestep clears ``t_star``.
    """
    if t_star is None:
        return None
    for i, t in enumerate(ts):
        if t <= t_star:
            return i
    return None
