"""Truncated Jump Sampling (TJS) for the K2 flow-matching sampler.

Adapted from "x-Prediction Is All You Need: Training-Free Accelerated
Generation via Endpoint Decodability" (arXiv:2607.06114). The paper observes
that on the standard affine probability path an intermediate state ``x_t`` and
its path velocity ``v`` already determine a principled estimate of the clean
sample: the decoder ``x_0 = x_t - t * v`` is the minimum-MSE estimator
``E[x_0 | x_t]`` under the usual ``l2`` objective. TJS exploits this to stop
the sampling ODE at an early-exit time ``t*`` and *jump* straight to the
decoded endpoint, cutting NFEs without retraining, distillation, or any
architecture change.

For K2 the flow path is ``x_t = (1 - t) * x_0 + t * eps`` (noise at ``t=1``,
data at ``t=0``), the model predicts the rectified-flow velocity
``v = eps - x_0``, and the Euler step in ``sampling.sample`` is
``img += (tprev - tcurr) * v``. Solving the path for ``x_0`` in terms of
``(x_t, v, t)`` gives the decoder below.

This module is a drop-in superset of ``sampling.sample``: with ``tjs_exit``
left at its default of ``1.0`` it reproduces the original sampler exactly;
values in ``(0, 1)`` enable TJS by running only the first ``tjs_exit`` fraction
of the Euler schedule and then decoding the endpoint in a single extra model
evaluation.
"""

import torch
from einops import rearrange
from PIL import Image

from sampling import prepare, roundup, timesteps


def endpoint_decode(x_t, velocity, t):
    """Decode the clean-sample estimate ``x_0`` from a flow-matching state.

    For an affine path ``x_t = (1 - t) * x_0 + t * eps`` with velocity
    ``v = eps - x_0`` we have ``x_t = x_0 + t * v``, hence ``x_0 = x_t - t * v``.
    This is the endpoint-decodability identity from the paper (the minimum-MSE
    estimator of ``x_0`` given ``x_t``).

    ``t`` may be a python float (shared across the batch, as at the jump) or a
    per-sample tensor broadcastable over ``x_t``'s trailing dimensions.
    """
    if torch.is_tensor(t):
        # Broadcast a [b] tensor over the trailing feature dims of x_t.
        t = t.reshape((-1,) + (1,) * (x_t.dim() - 1)).to(x_t.dtype)
    return x_t - t * velocity


@torch.no_grad()
def sample(
    model,
    ae,
    encoder,
    prompts,
    *,
    negative_prompts=None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    guidance=4.5,
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
    tjs_exit=1.0,
):
    """End-to-end text-to-image sampling with optional Truncated Jump exit.

    Identical to ``sampling.sample`` (encode -> Euler+CFG denoise -> decode)
    except for the ``tjs_exit`` early exit. When ``tjs_exit >= 1.0`` (default)
    the full ODE is integrated to ``t = 0`` and the result matches
    ``sampling.sample`` bit-for-bit. When ``0 < tjs_exit < 1``, only the first
    ``tjs_exit`` fraction of the scheduled Euler steps are taken; the loop then
    stops at ``t*`` and a single extra velocity evaluation decodes the endpoint
    via :func:`endpoint_decode`, skipping the remaining steps. This trades a
    small amount of quality for a proportional NFE reduction (the paper reports
    20-70% fewer NFEs at near-matched quality across SDXL/SD3.5M/Z-Image-Turbo).
    """
    patch = model.config.patch

    # The latent grid (dim // ae.compression) is patchified in `patch`-sized blocks,
    # so width/height must be multiples of ae.compression * patch. Pad up otherwise.
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n

    # Per-prompt seeded gaussian latent noise.
    noise = torch.cat(
        [
            torch.randn(
                1,
                ae.channels,
                height // ae.compression,
                width // ae.compression,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )

    # Positive (conditional) text conditioning.
    txt, txtmask = encoder(prompts)
    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)

    # The unconditional branch is only used for CFG; skip encoding/prep entirely
    # when guidance is disabled.
    if cfg:
        untxt, untxtmask = encoder(negative_prompts)
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    # min_res/max_res define the (x1,y1)-(x2,y2) interpolation endpoints for `mu`.
    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    # Truncated Jump: run only the first `exit_index` Euler sub-steps, leaving the
    # state at t* = ts[exit_index]; the remaining steps are replaced by a single
    # endpoint decode. exit_index == steps (tjs_exit >= 1.0) integrates the full
    # ODE to t = 0 and skips the jump, matching sampling.sample exactly.
    exit_index = min(max(round(tjs_exit * steps), 1), steps)

    def velocity(img_now, t_val):
        """CFG-combined velocity at state `img_now`, time `t_val`."""
        t = torch.full((len(img_now),), t_val, dtype=img_now.dtype, device=device)
        cond = model(img=img_now, context=txt, t=t, pos=pos, mask=mask)
        if cfg:
            uncond = model(img=img_now, context=untxt, t=t, pos=unpos, mask=unmask)
            return cond + guidance * (cond - uncond)
        return cond

    # Euler integration of the flow ODE with CFG, truncated to `exit_index` steps.
    img = x
    for tcurr, tprev in zip(ts[:exit_index], ts[1 : exit_index + 1]):
        v = velocity(img, tcurr)
        img = img + (tprev - tcurr) * v

    # Truncated Jump: stop at t* and decode the endpoint x_0 = x_t* - t* * v*.
    if exit_index < steps:
        t_star = ts[exit_index]
        img = endpoint_decode(img, velocity(img, t_star), t_star)

    # Unpatchify back to a latent and decode to pixels.
    img = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=height // (ae.compression * patch),
        w=width // (ae.compression * patch),
    )
    img = ae.decode(img.to(torch.bfloat16))
    img = img.clamp(-1, 1) * 0.5 + 0.5
    img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return [Image.fromarray(img[i]) for i in range(len(img))]
