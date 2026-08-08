"""Speculative velocity caching for the K2 sampler.

A training-free, parameter-free *forecast-then-verify* accelerator that
wraps the per-step MMDiT forward used by :func:`sampling.sample`.

Adapted from **SpeCa: Accelerating Diffusion Transformers with Speculative
Feature Caching** (Chen et al., arXiv:2509.11628). SpeCa forecasts the
expensive per-step DiT *features* from a Taylor-series extrapolation of
recently cached features, then accepts or rejects the forecast with a
parameter-free verifier -- *skipping the full forward* when the forecast is
trusted. This is the speculative-decoding idea ("draft cheaply, verify,
accept or fall back") applied to the diffusion timestep axis.

This module is a **Mode 2 adapted port**: it keeps SpeCa's core mechanism
(forecast from a Taylor-series extrapolation of recent steps, parameter-free
accept/reject verification, skip-on-accept, bounded consecutive skips to
limit drift) but ports it to the only surface the K2 sampler exposes -- the
*velocity* returned by ``model(img, context, t, pos, mask)``. The K2 sampler
treats the MMDiT as a black-box velocity function, so the paper's
feature-level caching and its shallow-network verifier would require model
surgery on ``mmdit.SingleStreamDiT``. To stay out of the model internals:

  * **feature-level caching -> velocity-level (output) caching.** We cache the
    returned velocity ``v`` at its timestep ``t`` rather than hidden features.
  * **shallow-network verifier -> curvature proxy from the cache.** The
    forecast is accepted only when the velocity field is locally smooth,
    measured as the spacing-normalised change in slope between the last three
    *real* evaluations. No learned verifier, no extra parameters.

The payoff is at the granularity the repo can host: when the verifier says
"smooth", the full DiT forward is *skipped for that step* and the free Taylor
forecast advances the ODE instead. In locally smooth regions of the schedule
the number of MMDiT forwards drops; in rapidly changing regions (and during
the cold-start warmup) the real model runs every step. The Euler + CFG
integration loop in :func:`sampling.sample` is unchanged -- this wrapper is a
drop-in around ``model``.

Out of scope (intentional): the paper's shallow/deep model split (needs
``mmdit`` surgery), its separate benchmark / FID evaluation (a downstream
PR), and reported wall-clock numbers on real K2 weights (no weights or GPU
here). The deliverable is the opt-in accelerator and its accept/reject logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# A small absolute floor used when normalising tensor norms, so a genuinely
# zero velocity (e.g. a mock model) does not divide by zero.
_EPS = 1e-8


class _Trajectory:
    """Per-context cache of recent (timestep, velocity) evaluations."""

    __slots__ = ("_history", "skips", "t", "v")

    def __init__(self, history: int) -> None:
        self.t: list[float] = []
        self.v: list[torch.Tensor] = []
        self.skips: int = 0  # consecutive accepted forecasts since last real eval
        self._history = history

    def push(self, t: float, v: torch.Tensor) -> None:
        """Record a *real* evaluation, evicting the oldest beyond ``history``."""
        self.t.append(t)
        self.v.append(v)
        excess = len(self.t) - self._history
        if excess > 0:
            del self.t[:excess]
            del self.v[:excess]

    def __len__(self) -> int:
        return len(self.t)


def _scalar_t(t: torch.Tensor) -> float:
    """Pull the per-step timestep as a Python float.

    :func:`sampling.sample` builds ``t`` as ``torch.full((batch,), tcurr, ...)``,
    so every element of the batch shares one timestep for a given step.
    """
    return float(t.reshape(-1)[0].to(torch.float32).item())


def _slope(t0: float, v0: torch.Tensor, t1: float, v1: torch.Tensor) -> torch.Tensor:
    """Discrete derivative ``(v1 - v0) / (t1 - t0)`` in float32."""
    return (v1.float() - v0.float()) / (t1 - t0)


def _extrapolate(traj: _Trajectory, tnow: float) -> torch.Tensor:
    """First-order Taylor (linear) forecast of the velocity at ``tnow``.

    Uses the two most recent *real* evaluations. Linear extrapolation is the
    1-term Taylor forecast; we deliberately stop at first order because the
    verifier only ever trusts it across a handful of steps (see ``max_skip``),
    so a higher-order predictor is not worth its noise sensitivity here.
    """
    t1, v1 = traj.t[-1], traj.v[-1]
    t0, v0 = traj.t[-2], traj.v[-2]
    slope = _slope(t0, v0, t1, v1)
    forecast = v1.float() + slope * (tnow - t1)
    return forecast.to(v1.dtype)


def _curvature(traj: _Trajectory) -> float:
    """Parameter-free smoothness signal in ``[0, 2]`` from the last 3 evals.

    Relative change in slope, normalised so it is scale-free:

        curvature = ||s2 - s1|| / (||s1|| + ||s2||)

    where ``s1``, ``s2`` are consecutive discrete derivatives (timestep spacing
    already folded in). A perfectly linear-in-t velocity field gives ``0``; a
    field that reverses direction approaches ``2``. The forecast is accepted
    when this falls below ``tol``.
    """
    t2, v2 = traj.t[-1], traj.v[-1]
    t1, v1 = traj.t[-2], traj.v[-2]
    t0, v0 = traj.t[-3], traj.v[-3]
    s1 = _slope(t0, v0, t1, v1)
    s2 = _slope(t1, v1, t2, v2)
    denom = s1.norm() + s2.norm() + _EPS
    return float((s2 - s1).norm().item() / denom.item())


class SpeculativeVelocityCache(nn.Module):
    """Forecast-then-verify wrapper around a K2 velocity model.

    Drop it in where ``model`` is called in the sampler loop::

        model = SpeculativeVelocityCache(model, tol=0.25)
        v = model(img=img, context=txt, t=t, pos=pos, mask=mask)

    It preserves the exact ``(img, context, t, pos, mask) -> velocity``
    contract, maintains an independent cache per ``context`` tensor (so the
    conditional and unconditional CFG branches never contaminate each other),
    and exposes simple counters (``forward_calls``, ``skipped``, ``total``)
    for diagnosing the skip rate.

    Parameters
    ----------
    model:
        The MMDiT (or any callable with the velocity contract).
    tol:
        Accept-the-forecast threshold on the curvature signal. Smaller is more
        conservative (fewer skips, higher fidelity); larger skips more
        aggressively. Defaults to ``0.25``.
    max_skip:
        Maximum number of consecutive steps whose forwards may be skipped
        before a real evaluation is forced to refresh the cache. Bounds how far
        any single forecast can drift from the true velocity. Defaults to ``2``.
    history:
        How many recent real evaluations to retain per context (``>= 3`` is
        required for the curvature verifier). Defaults to ``3``.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        tol: float = 0.25,
        max_skip: int = 2,
        history: int = 3,
    ) -> None:
        super().__init__()
        if history < 3:
            raise ValueError("history must be >= 3 to estimate the curvature verifier")
        if max_skip < 1:
            raise ValueError("max_skip must be >= 1")
        self.model = model
        self.tol = float(tol)
        self.max_skip = int(max_skip)
        self._history = int(history)
        self._caches: dict[int, _Trajectory] = {}
        # Diagnostics -- introspect after sampling to report the skip rate.
        self.forward_calls = 0  # real model evaluations actually run
        self.skipped = 0  # accepted forecasts (forward was skipped)
        self.total = 0  # total invocations of this wrapper

    def _trajectory(self, context: torch.Tensor) -> _Trajectory:
        # Key on the identity of the context tensor. Within one sample() call
        # the cond/uncond context tensors are stable objects, so this cleanly
        # separates the two CFG trajectories without hashing tensor contents.
        key = id(context)
        traj = self._caches.get(key)
        if traj is None:
            traj = _Trajectory(self._history)
            self._caches[key] = traj
        return traj

    def forward(
        self,
        img: torch.Tensor,
        context: torch.Tensor,
        t: torch.Tensor,
        pos: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the velocity, skipping the model forward when the forecast verifies."""
        self.total += 1
        tnow = _scalar_t(t)
        traj = self._trajectory(context)

        # Need >=3 real evals: >=2 to extrapolate at all, and the 3rd to feed
        # the curvature verifier. Also refuse to skip once we have run up to
        # ``max_skip`` consecutive forecasts -- force a refresh to bound drift.
        if len(traj) >= 3 and traj.skips < self.max_skip:
            curvature = _curvature(traj)
            if curvature < self.tol:
                traj.skips += 1
                self.skipped += 1
                return _extrapolate(traj, tnow)

        # Cold start, rough region, or drift bound hit: run the real model and
        # refresh this trajectory's cache with a ground-truth evaluation.
        v = self.model(img=img, context=context, t=t, pos=pos, mask=mask)
        self.forward_calls += 1
        traj.push(tnow, v.detach())
        traj.skips = 0
        return v

    @property
    def skip_rate(self) -> float:
        """Fraction of invocations whose forward was skipped (0 if never called)."""
        return self.skipped / self.total if self.total else 0.0
