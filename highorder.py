"""Higher-order flow-matching ODE solvers for the K2 sampler.

Adapted from the AMED-Solver ("Approximate MEan-Direction Solver") of
Zhou et al., "Fast ODE-based Sampling for Diffusion Models in Around 5
Steps" (arXiv:2312.00094, CVPR 2024).

The paper's core mechanism -- the single-step mean-direction update of
its Eq. 9, which evaluates the velocity at an intermediate timestep
reached by an Euler half-step and extrapolates to the step endpoint
along that direction -- is kept at full fidelity. Its learned auxiliary,
a tiny (~9k-param) alignment network g_phi that predicts the per-step
intermediate timestep ``s_n`` and scale ``c_n`` from the model's
bottleneck feature, is substituted with a parameter-free proxy:
``c_n = 1`` and ``s_n = tcurr**r * tprev**(1-r)`` (a power mean of the
step endpoints; ``r = 0.5`` -- the geometric mean -- is the DPM-Solver-2
baseline the paper names explicitly). The alignment knob ``r`` is the
parameter-free stand-in for the learned predictor.

Each step costs two velocity evaluations (~2 NFE/step versus Euler's 1),
but the update is second-order accurate: at iso-NFE the Raw model can be
run at roughly half its 52-step schedule for comparable quality.
"""

import torch


def denoise_amed(velocity, img, ts, r=0.5):
    """Integrate the flow ODE (``ts``: 1 -> 0) with the AMED mean-direction update.

    ``velocity(x, t)`` returns the CFG-combined flow velocity at latent
    ``x`` and scalar timestep ``t``. ``img`` is the patchified latent; the
    returned tensor has the same shape. See the module docstring for the
    AMED attribution and the parameter-free substitution of the learned
    alignment predictor.
    """
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        v0 = velocity(img, tcurr)
        # Intermediate timestep s_n (AMED Eq. 9 prep). Power mean of the
        # endpoints; fall back to the arithmetic midpoint when an endpoint
        # is 0 (the final schedule node) so s stays strictly positive.
        if tcurr > 0 and tprev > 0:
            s = tcurr**r * tprev ** (1.0 - r)
        else:
            s = 0.5 * (tprev + tcurr)
        xs = img + (s - tcurr) * v0
        vs = velocity(xs, s)
        img = img + (tprev - tcurr) * vs
    return img
