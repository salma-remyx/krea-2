"""Progressive Seed Pruning (PSP) for the K2 flow-matching sampler.

Adapted from "Inference-Time Scaling of Diffusion Models via Progressive
Seed Pruning" (arXiv:2607.21591). PSP relaxes the usual "constant memory
footprint" constraint on inference-time scaling and instead *front-loads*
seed exploration: many initial-noise candidates are denoised together for
the first part of the schedule, scored on their intermediate denoised
estimate, and the unpromising trajectories are pruned so that only the
survivors receive the remaining (expensive) denoise steps. The total number
of model evaluations is held fixed relative to a best-of-N baseline, so the
extra early exploration is effectively free.

Implementation mode (Mode 2 — adapted port): the paper's *core mechanism* is
kept at full fidelity — front-loaded exploration of many seeds, scoring of the
predicted-clean estimate, progressive pruning, fixed total NFE. Two
*auxiliary* components are substituted with target-native equivalents:

  * The paper's learned / black-box image reward is replaced by a
    parameter-free ``detail_energy`` proxy (per-candidate variance of the
    predicted-clean latent) that approximates "how much resolved structure
    this trajectory already carries". It is exposed as the ``score_fn``
    argument so a real reward model can be dropped in without touching the
    pruning logic.
  * The paper's separate benchmark / GenEval harness is intentionally cut —
    evaluation belongs in a downstream PR. This module only produces images.
"""

import math

import torch
from einops import rearrange
from PIL import Image

from sampling import prepare, roundup, timesteps


def detail_energy(x0_hat):
    """Parameter-free reward proxy: per-candidate variance of the clean estimate.

    ``x0_hat`` is the predicted-clean latent for each active candidate
    (any shape ``[b, ...]``). Trajectories whose early estimate already
    carries more resolved structure rank higher, which is the signal PSP
    exploits to prune *before* spending the expensive late steps. This stands
    in for the paper's learned reward (Mode 2 substitution); pass a real
    reward via ``score_fn`` to override it.
    """
    flat = x0_hat.reshape(x0_hat.shape[0], -1)
    return flat.float().var(dim=-1, unbiased=False)


def seeds_for_budget(baseline_n, steps, prune_at, keep):
    """Candidate-seed count that matches best-of-``baseline_n`` at fixed NFE.

    Total model evaluations under PSP equal ``baseline_n * steps`` (the cost of
    generating ``baseline_n`` full images, i.e. plain best-of-N), so the wider
    early exploration is compute-matched rather than extra. Returns how many
    seeds that budget affords given a prune point and survivor count.
    """
    prune_steps = max(1, min(steps - 1, int(round(prune_at * steps))))
    remaining = baseline_n * steps - keep * (steps - prune_steps)
    return max(keep, math.ceil(remaining / prune_steps))


@torch.no_grad()
def sample_psp(
    model,
    ae,
    encoder,
    prompts,
    *,
    num_seeds,
    keep=1,
    prune_at=0.25,
    score_fn=None,
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
):
    """PSP variant of :func:`sampling.sample`.

    For each prompt, ``num_seeds`` candidate noise latents are denoised
    together through the first ``prune_at`` fraction of the schedule, scored
    on their predicted-clean estimate, and pruned to ``keep`` survivors which
    are denoised to completion. Returns ``len(prompts) * keep`` images
    (prompt-major).
    """
    if score_fn is None:
        score_fn = detail_energy
    if keep > num_seeds:
        keep = num_seeds

    patch = model.config.patch
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    cfg = guidance > 0
    prune_steps = max(1, min(steps - 1, int(round(prune_at * steps))))

    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2

    images = []
    for pi, prompt in enumerate(prompts):
        # Front-loaded exploration: `num_seeds` candidates share one prompt,
        # differing only in their initial noise. Treat them as a batch exactly
        # like sampling.sample treats N prompts.
        noise = torch.cat(
            [
                torch.randn(
                    1,
                    ae.channels,
                    height // ae.compression,
                    width // ae.compression,
                    device=device,
                    dtype=dtype,
                    generator=torch.Generator(device=device).manual_seed(
                        seed + pi * num_seeds + i
                    ),
                )
                for i in range(num_seeds)
            ],
            dim=0,
        )

        txt, txtmask = encoder([prompt] * num_seeds)
        img, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
        if cfg:
            untxt, untxtmask = encoder([""] * num_seeds)
            _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

        ts = timesteps(img.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)
        active = num_seeds

        for idx, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
            t = torch.full((img.shape[0],), tcurr, dtype=img.dtype, device=img.device)
            cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
            if cfg:
                uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                v = cond + guidance * (cond - uncond)
            else:
                v = cond

            # Prune once, right after the exploration budget is spent: rank the
            # predicted-clean estimates and keep only the top `keep` trajectories.
            if active > keep and (idx + 1) >= prune_steps:
                scores = score_fn(img - tcurr * v)
                top = torch.topk(scores, keep).indices
                img, v, pos, mask = (x[top] for x in (img, v, pos, mask))
                txt = txt[top]
                if cfg:
                    untxt, unpos, unmask = (x[top] for x in (untxt, unpos, unmask))
                active = keep

            img = img + (tprev - tcurr) * v

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
        images.extend(Image.fromarray(img[i]) for i in range(len(img)))

    return images
