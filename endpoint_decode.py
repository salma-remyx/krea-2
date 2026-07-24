"""Endpoint decodability + Truncated Jump Sampling (TJS) early exit.

Adapted from "x-Prediction Is All You Need: Training-Free Accelerated
Generation via Endpoint Decodability" (arXiv:2607.06114).

Krea 2 samples a flow-matching ODE on the affine probability path
x_t = t * noise + (1 - t) * data (flow time t goes 1 -> 0), so the velocity
field v the model already predicts is v = noise - data. Solving for the clean
sample at any visited point gives the one-line *endpoint decodability* decoder:

    x0 = x_t - t * v

``sampling.sample`` wires this into its Euler loop as a training-free early
exit (Truncated Jump Sampling, TJS): stop the ODE at an early-exit time t* and
return the decoded endpoint instead of integrating all the way to t = 0,
cutting neural function evaluations by 20-70% at near-matched quality. No
retraining, distillation, or architecture change is required.

These helpers carry no torch import so the decode math stays trivially unit
testable and works on any array-like that supports arithmetic.
"""


def decode_endpoint(img, t, velocity):
    """Decode the clean-sample estimate from an intermediate state and velocity.

    For the affine flow path x_t = t * noise + (1 - t) * data the model velocity
    is v = noise - data, so the minimum-MSE estimate of the endpoint is
    ``x0 = x_t - t * v``. ``img`` is the latent at flow time ``t`` (``x_t``) and
    ``velocity`` is the velocity the model predicts there. Accepts any
    array-like supporting ``-`` and scalar ``*`` (torch tensors, numpy arrays).
    """
    return img - t * velocity


def early_exit_index(timesteps, t_star):
    """Index into ``timesteps[:-1]`` of the first leading time <= ``t_star``.

    The sampler visits (tcurr, tprev) pairs from
    ``zip(timesteps[:-1], timesteps[1:])`` with flow time decreasing 1 -> 0.
    Truncated Jump Sampling stops the ODE the first time the leading time
    ``tcurr`` drops to ``t_star`` or below and decodes the endpoint there.

    Returns ``None`` when ``t_star`` is ``None`` (no early exit: integrate
    fully) or when no visited leading time satisfies the condition (the schedule
    never reaches ``t_star``, so there is nothing to truncate).
    """
    if t_star is None:
        return None
    for i, t in enumerate(timesteps[:-1]):
        if t <= t_star:
            return i
    return None
