"""Progressive Seed Pruning (PSP) sampler for the K2 flow-matching pipeline.

Adapts "Inference-Time Scaling of Diffusion Models via Progressive Seed Pruning"
(arXiv:2607.21591) onto the existing K2 sampler without any model surgery.

PSP front-loads exploration: instead of spending a fixed compute budget on a
single noise seed (or fully denoising N seeds and keeping the best afterwards),
it denoises N candidate seeds together through the early steps, scores the
intermediate trajectories, and prunes the weakest candidates at fixed
checkpoints -- so only the most promising trajectory receives the full denoise
budget, while the total number of model evaluations stays well below N * steps.

Implementation mode: **adapted port (Mode 2)**. The *core mechanism* --
front-load -> score -> prune, targeting a constant total-evaluation budget -- is
kept at full fidelity. The paper's black-box / learned reward over the denoised
x0 estimate is substituted with a *parameter-free* proxy read off the in-loop
velocity (zero extra model evaluations). Pass a real reward via ``score_fn`` to
recover the paper's full reward-guided behaviour; the pruning schedule itself is
unchanged either way.

Paper attribution: arXiv:2607.21591v1 -- "Inference-Time Scaling of Diffusion
Models via Progressive Seed Pruning".
"""

import math

import torch
from einops import rearrange
from PIL import Image

from sampling import prepare, roundup, timesteps


def default_prune_schedule(num_seeds, steps):
    """Evenly spaced prune checkpoints that halve the candidates down to one.

    Returns a list of ``(step_index, survivors_per_group)`` pairs. ``num_seeds``
    candidates are reduced over ``ceil(log2(num_seeds))`` rounds placed at the
    interior fractions ``r / (rounds + 1)`` of the schedule, each round keeping
    ``ceil(num_seeds / 2**r)`` candidates per prompt-group, until one remains.
    """
    if num_seeds <= 1:
        return []
    rounds = math.ceil(math.log2(num_seeds))
    schedule = []
    seen = set()
    for r in range(1, rounds + 1):
        step = min(max(int((r / (rounds + 1)) * steps), 0), steps - 1)
        if step in seen:
            continue
        seen.add(step)
        keep = max(1, math.ceil(num_seeds / (2 ** r)))
        schedule.append((step, keep))
    return schedule


def latent_energy_score(img, x0_pred, cond, uncond, tcurr):
    """Default parameter-free trajectory score: mean |denoised x0 estimate|.

    Candidates whose mid-trajectory clean estimate carries more structured (less
    collapsed) latent energy score higher. This is a dependency-free stand-in for
    the paper's learned / black-box reward; it costs zero model evaluations
    because it reads the velocity already computed for the Euler step. Swap in a
    real reward (CLIP alignment, aesthetic score, ...) via ``score_fn`` to
    recover the paper's reward-guided selection.
    """
    return x0_pred.flatten(start_dim=1).abs().mean(dim=1)


@torch.no_grad()
def sample_psp(
    model,
    ae,
    encoder,
    prompts,
    *,
    num_seeds=4,
    prune_schedule=None,
    score_fn=None,
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
):
    """Best-of-``num_seeds`` sampling via Progressive Seed Pruning.

    For each prompt, ``num_seeds`` candidate noise seeds are denoised together;
    the intermediate trajectories are scored and pruned at fixed checkpoints so
    that a single surviving trajectory is fully denoised. Returns one image per
    prompt -- the pruned winner -- mirroring :func:`sampling.sample`'s
    encode -> euler(+CFG) -> decode shape so it is a drop-in at the same call
    site (and :func:`sampling.sample` remains the plain-seed fallback).
    """
    if num_seeds < 1:
        raise ValueError("num_seeds must be >= 1")
    if score_fn is None:
        score_fn = latent_energy_score
    if prune_schedule is None:
        prune_schedule = default_prune_schedule(num_seeds, steps)

    patch = model.config.patch
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n

    latent_h, latent_w = height // ae.compression, width // ae.compression
    # Per-prompt, per-seed seeded gaussian noise, ordered [p0 s0..sN, p1 s0..sN].
    noise = torch.cat(
        [
            torch.randn(
                1,
                ae.channels,
                latent_h,
                latent_w,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(
                    seed + p * num_seeds + s
                ),
            )
            for p in range(n)
            for s in range(num_seeds)
        ],
        dim=0,
    )

    # Positive (conditional) context, tiled across every candidate of its prompt.
    txt, txtmask = encoder(prompts)
    txt = txt.repeat_interleave(num_seeds, dim=0)
    txtmask = txtmask.repeat_interleave(num_seeds, dim=0)
    img, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)

    # The unconditional branch is only used for CFG; skip it entirely otherwise.
    if cfg:
        untxt, untxtmask = encoder(negative_prompts)
        untxt = untxt.repeat_interleave(num_seeds, dim=0)
        _, unpos, unmask = prepare(
            noise, untxt.shape[1], patch, untxtmask.repeat_interleave(num_seeds, dim=0)
        )

    # min_res/max_res define the (x1,y1)-(x2,y2) interpolation endpoints for mu.
    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2
    ts = timesteps(img.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    prune_at = {step: keep for step, keep in prune_schedule}
    survivors = num_seeds  # current number of candidates per prompt-group

    # Euler integration of the flow ODE (with optional CFG), pruning at checkpoints.
    for i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
        cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
        if cfg:
            uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
            v = cond + guidance * (cond - uncond)
        else:
            uncond = None
            v = cond

        keep = prune_at.get(i)
        if keep is not None and len(img) > n and keep < survivors:
            # Denoised x0 estimate from this step's velocity -- no extra eval.
            x0_pred = img - tcurr * v
            scores = score_fn(img, x0_pred, cond, uncond, tcurr).view(n, survivors)
            # Top-`keep` within each prompt-group, indices sorted to keep groups
            # contiguous so subsequent rounds can reshape by `survivors`.
            local = torch.topk(scores, keep, dim=1).indices.sort(dim=1).values
            flat = (
                torch.arange(n, device=img.device) * survivors
            ).unsqueeze(1) + local
            flat = flat.reshape(-1)
            img, pos, mask = img[flat], pos[flat], mask[flat]
            txt = txt[flat]
            v = v[flat]
            if cfg:
                untxt, unpos, unmask = untxt[flat], unpos[flat], unmask[flat]
            survivors = keep

        img = img + (tprev - tcurr) * v

    # Unpatchify back to a latent and decode to pixels (mirrors sampling.sample).
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
