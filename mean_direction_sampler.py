"""Mean-direction flow-matching sampler.

The default sampler in ``sampling.py`` integrates the flow-matching ODE
``dx/dt = v(x, t)`` (t: 1 -> 0) with a first-order forward-Euler step. Euler
accumulates truncation error, which is what forces the RAW checkpoint to run
~52 steps for clean images.

Insight (adapted from "Fast ODE-based Sampling for Diffusion Models in Around
5 Steps", AMED-Solver, Geng et al. 2023, arXiv:2312.00094): a sampling
trajectory is nearly straight, so stepping along the trajectory's *mean
direction* -- rather than the direction at a single endpoint -- suppresses the
per-step truncation error and lets each step cover more of the trajectory.
AMED-Solver evaluates the model at two points along the trajectory, then
*learns* how to combine the two directions into the mean direction:

    x_pred = x + dt * v(x, t)                         # predictor direction
    x_next = x + dt * w(v(x, t), v(x_pred, t + dt))   # learned mean direction

This codebase is inference-only with no trainer, so the learned combiner ``w``
cannot be trained or shipped. We instead take its parameter-free special case
``w(a, b) = (a + b) / 2`` -- the classical trapezoidal (Heun) rule, i.e. the
equal-weight mean direction. That preserves AMED's two-evaluation,
mean-direction structure and trades one extra model evaluation per step for
roughly 2x larger, higher-quality steps -- a step-count reduction for the RAW
checkpoint, squarely the repo's #1 latency interest.

Intentionally out of scope: AMED's learned mean-direction predictor (the part
that reaches its ~5-step regime), which needs training and trained weights this
repo does not host; and AMED's score-based / probability-flow-ODE formulation,
since K2 uses flow matching.
"""

import torch


def cfg_velocity(model, txt, pos, mask, guidance, untxt=None, unpos=None, unmask=None):
    """Return a closure ``v(x, t)`` for the flow velocity, CFG folded in.

    ``untxt`` / ``unpos`` / ``unmask`` are only required when ``guidance > 0``.
    Folding CFG into the velocity lets the caller treat the sampler as a plain
    ODE integrator, independent of how the velocity is produced.
    """
    cfg = guidance > 0

    def velocity(x, t):
        t_tensor = torch.full((len(x),), t, dtype=x.dtype, device=x.device)
        cond = model(img=x, context=txt, t=t_tensor, pos=pos, mask=mask)
        if cfg:
            uncond = model(img=x, context=untxt, t=t_tensor, pos=unpos, mask=unmask)
            return cond + guidance * (cond - uncond)
        return cond

    return velocity


def mean_direction_step(velocity, x, tcurr, tprev):
    """One mean-direction (trapezoidal / Heun) step of the flow ODE.

    Predict with the start-direction velocity, evaluate again at the predicted
    endpoint, and step along the mean of the two directions -- the
    parameter-free realization of AMED-Solver's mean-direction step.
    """
    dt = tprev - tcurr
    v_start = velocity(x, tcurr)
    x_pred = x + dt * v_start
    v_end = velocity(x_pred, tprev)
    return x + dt * 0.5 * (v_start + v_end)
