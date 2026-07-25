"""Truncated Jump Sampling (TJS): training-free ODE early-exit for flow matching.

Adapted from "x-Prediction Is All You Need: Training-Free Accelerated Generation
via Endpoint Decodability" (arxiv:2607.06114).

The K2 sampler follows an affine probability path ``x_t = t * noise + (1 - t) * x_0``
with ``t`` running 1 -> 0. Its velocity ``v = dx/dt = noise - x_0`` therefore pins
the data endpoint::

    x_t = x_0 + t * v      =>      x_0 = x_t - t * v.

``decode_endpoint`` recovers ``x_0`` from an intermediate state and its velocity,
so the Euler loop can stop early at an exit time ``t*`` (a fraction of the full
schedule) and jump straight to the decoded clean latent -- cutting NFEs by 20-70%
with no retraining, distillation, or architecture change. As ``t* -> 0`` the
decode reduces to the ordinary final Euler step, so a full rollout is unchanged.
"""

import math


def truncation_steps(steps, tjs):
    """Number of Euler steps to run under Truncated Jump Sampling.

    ``tjs`` in ``(0, 1]`` is the fraction of the full schedule kept; the rest of
    the trajectory is replaced by decoding ``x_0`` from the last computed
    velocity. At least one step is always kept so a single model call seeds the
    jump.
    """
    return max(1, math.ceil(tjs * steps))


def decode_endpoint(x_t, v, t):
    """Decode the clean endpoint ``x_0`` from a flow state and its velocity.

    The minimum-MSE estimate ``E[x_0 | x_t]`` under the affine path used by the
    K2 sampler. Operates on any object supporting ``-`` and scalar ``*`` (torch
    tensors, numpy arrays, plain numbers, ...).
    """
    return x_t - t * v
