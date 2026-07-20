"""Stability-guided step cache for the Euler flow-matching sampler.

Adapted from SADA (Stability-guided Adaptive Diffusion Acceleration, Jiang et
al., ICML 2025, https://arxiv.org/abs/2507.17135). SADA accelerates ODE-based
diffusion / flow-matching samplers by reusing the model's per-step output (the
velocity prediction y_t = dx/dt) whenever the denoising trajectory is locally
*stable* -- the velocity field's curvature is small -- instead of paying for a
fresh model forward pass on every step.

This ports SADA's *step-wise cache-assisted pruning* (Section 3.4) onto Krea 2's
hand-written Euler sampler. Two halves of SADA are intentionally left out
because this bare sampler has no surface for them:

  * Token-wise cache-assisted pruning (Section 3.5) is shipped in the reference
    repo by monkey-patching a HuggingFace Diffusers attention module; Krea 2
    invokes ``model(...)`` as an opaque forward with no per-layer hook.
  * The multistep Lagrange / third-order Adams-Moulton state extrapolation
    (Theorems 3.5 / 3.7) targets second-order DPM-Solver++ style schedulers that
    integrate via x0 reconstruction. Krea 2's sampler is a first-order Euler
    solver over the velocity, so direct velocity reuse with the Euler step is the
    matching-fidelity approximation.

The core signal kept at full fidelity is the *second-order difference of the
velocity trajectory*, d2(y)_t = y_t - 2*y_{t-1} + y_{t-2}, which SADA's stability
criterion (Criterion 3.4) is built on to "measure the local dynamics of the
denoising trajectory". SADA states the criterion as a sign test on the inner
product of the extrapolation error with d2(y); since that extrapolation error
needs a future-step estimate an online Euler loop does not have, we use the
directly-computable magnitude form ``||d2(y)_t|| / ||y_t|| <= threshold`` as the
parameter-free proxy for the same curvature signal.
"""

from __future__ import annotations

import torch
from torch import Tensor


class StabilityCache:
    """Reuse the model output on locally-linear steps of the flow trajectory.

    After each fresh model evaluation the combined velocity ``v`` is appended to
    a length-3 history. The next step is deemed *stable* (eligible to skip its
    forward pass and reuse the cached output) when the relative second-order
    difference of that history falls below ``threshold`` -- the trajectory is
    locally linear and the velocity is barely changing -- and we have not
    already reused ``max_skip`` times in a row. The streak cap bounds error
    accumulation, mirroring SADA recomputing once its online criterion flips.
    """

    def __init__(self, threshold: float = 0.05, max_skip: int = 1):
        self.threshold = float(threshold)
        self.max_skip = int(max_skip)
        # Rolling length-3 velocity history driving the second-order difference.
        self._vels: list[Tensor] = []
        # Most recent freshly-computed model outputs, reused on stable steps.
        self._cond: Tensor | None = None
        self._uncond: Tensor | None = None
        self._streak = 0  # consecutive reuses of the current cached output
        self.computed = 0
        self.reused = 0

    def record(
        self, cond: Tensor, uncond: Tensor | None, velocity: Tensor
    ) -> None:
        """Store a freshly-computed step's outputs and refresh the curvature history."""
        self._cond = cond
        self._uncond = uncond
        self._vels = [*self._vels[-2:], velocity.detach()]
        self._streak = 0
        self.computed += 1

    def should_reuse(self) -> bool:
        """True iff the trajectory is stable enough to skip the next forward pass."""
        if len(self._vels) < 3 or self._streak >= self.max_skip:
            return False
        prev, mid, curr = self._vels[-3:]
        d2 = curr - 2.0 * mid + prev
        denom = curr.float().norm() + 1e-8
        rel = (d2.float().norm() / denom).item()
        return rel <= self.threshold

    def cached(self) -> tuple[Tensor, Tensor | None]:
        """Return the most recently computed ``(cond, uncond)`` to reuse."""
        assert self._cond is not None
        return self._cond, self._uncond

    def mark_reused(self) -> None:
        self._streak += 1
        self.reused += 1


def relative_curvature(vels: list[Tensor]) -> Tensor:
    """Relative second-order difference ||d2(y)|| / ||y|| of a velocity history.

    Exposed for tests / introspection: 0 means the trajectory is perfectly
    linear (velocities identical), larger means sharper local curvature.
    """
    if len(vels) < 3:
        return torch.tensor(float("inf"))
    prev, mid, curr = vels[-3:]
    d2 = curr - 2.0 * mid + prev
    return d2.float().norm() / (curr.float().norm() + 1e-8)
