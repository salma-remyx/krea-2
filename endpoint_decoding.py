"""Endpoint decoding for flow-matching samplers (Truncated Jump Sampling).

Krea 2 integrates an affine flow-matching path ``x_t = (1 - t) x_0 + t x_1``
with the flow time ``t`` running from 1 (pure noise) down to 0 (clean sample).
The path velocity is therefore ``v = x_1 - x_0``, and rearranging recovers the
clean sample from a single intermediate state::

    x_0 = x_t - t * v(x_t)

This is the *endpoint decoder*: the minimum-MSE estimator ``E[x_0 | x_t]`` of
the clean sample given one state and its velocity. It adds no parameters and
needs no retraining, distillation, or change to the network.

**Truncated Jump Sampling (TJS)** turns this into an inference speedup: stop
integrating the sampling ODE at an early-exit time ``t*`` and return the
decoded ``x_0`` instead of running the full schedule down to ``t = 0``. On
flow-matching MM-DiTs this trims a large fraction of the neural function
evaluations (NFEs) at near-matched quality. See ``sampling.sample`` for the
``tjs_early_exit`` knob that enables it.

Adapted from "x-Prediction Is All You Need: Training-Free Accelerated
Generation via Endpoint Decodability" (arXiv:2607.06114). Only the paper's
core endpoint-decoding result is implemented here; its benchmark / FID
evaluation suite is intentionally out of scope (no eval harness in this repo).
"""

from __future__ import annotations

from torch import Tensor


def decode_endpoint(x_t: Tensor, velocity: Tensor, t: float) -> Tensor:
    """Decode the clean-sample estimate ``x_0`` from a flow-matching state.

    Given an intermediate state ``x_t`` and its path velocity ``velocity`` at
    flow time ``t`` (1 = noise, 0 = clean), returns ``x_t - t * velocity``.
    On the exact affine path this recovers ``x_0`` exactly; under a learned
    velocity field trained with the usual ``l2`` objective it is the
    minimum-MSE estimate ``E[x_0 | x_t]``.

    ``t`` may be a Python scalar or any value broadcastable against ``x_t``.
    """
    return x_t - t * velocity
