"""Endpoint decodability for the affine flow-matching path used by K2.

During flow-matching sampling the probability path is affine in the flow
time ``t``::

    x_t = (1 - t) * x0 + t * eps ,   with   v_t = d/dt x_t = eps - x0

so the clean endpoint ``x0`` is recoverable from any intermediate state and
the velocity predicted at it::

    x0 = x_t - t * v_t

This identity is exact on the linear path; it is also the minimum-MSE
estimator ``E[x0 | x_t]`` under the usual ``l2`` velocity objective. It is
the "endpoint decodability" property formalized by Truncated Jump Sampling
(TJS), which stops the sampling ODE at an early-exit time ``t*`` and returns
this decoded ``x0`` instead of stepping the rest of the trajectory -- a
training-free, architecture-free NFE reduction.

Reference: "x-Prediction Is All You Need: Training-Free Accelerated
Generation via Endpoint Decodability" (arXiv:2607.06114).
"""

import torch


def decode_endpoint(x_t, velocity, t):
    """Decode the clean endpoint ``x0`` from a flow state and its velocity.

    Args:
        x_t: Current flow state ``x_t`` (any shape).
        velocity: Velocity ``v_t = eps - x0`` predicted by the model at
            ``(x_t, t)``. Must broadcast against ``x_t``.
        t: Flow time of ``x_t``. A Python/PyTorch scalar (shared across the
            batch, as in the K2 sampler) or a tensor that broadcasts against
            ``x_t``.

    Returns:
        The endpoint estimate ``x0 = x_t - t * velocity``.
    """
    return x_t - t * velocity


def truncated_jump(state, velocity, t):
    """Single-step "jump" of the ODE straight to ``t = 0`` (the endpoint).

    Equivalent to one Euler step whose target time is 0, and therefore equal
    in value to :func:`decode_endpoint` -- kept as a named alias so the
    sampling loop reads as "stop early, jump to x0".
    """
    return decode_endpoint(state, velocity, t)


# Public surface: the decoder plus a named alias that mirrors the paper's
# framing for callers writing sampling loops by hand.
__all__ = ["decode_endpoint", "truncated_jump"]
