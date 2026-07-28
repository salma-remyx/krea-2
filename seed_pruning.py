"""Progressive Seed Pruning (PSP) — inference-time scaling for the K2 sampler.

Capability
----------
Instead of drawing a single noise seed per prompt (or running ``N`` independent
full trajectories, i.e. best-of-N), PSP *front-loads* exploration: it samples
many candidate seeds, denoises them jointly for the early steps, scores each
intermediate estimate, and prunes the worst candidates at a few checkpoints.
Only the survivors pay for the expensive late denoising steps, so the same
total compute budget buys more exploration where it is cheap (noisy, early)
and concentration where it is expensive (clean, late).

Adapted from "Inference-Time Scaling of Diffusion Models via Progressive Seed
Pruning" (arXiv:2607.21591). The paper's *core pruning mechanism* — fan out
seeds, score intermediate denoised estimates, progressively narrow the
candidate set while holding the total number of model evaluations fixed — is
implemented at full fidelity. The paper scores those estimates with a black-box
reward model; this codebase ships no reward model, so that learned auxiliary is
substituted with a parameter-free *velocity-agreement* proxy
(:func:`cfg_velocity_agreement`), exposed as a pluggable ``reward`` callable.
Pass an HPS / ImageReward / CLIP-score callable to recover the paper's full
reward-guided selection.

The loop reuses :func:`sampling.prepare`, :func:`sampling.timesteps` and
:func:`sampling.roundup` and the same ``model()`` + CFG convention as
:func:`sampling.sample`, so the model forward is unaltered.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from einops import rearrange
from PIL import Image

from sampling import prepare, roundup, timesteps

# Scores intermediate denoised estimates. Higher == more promising trajectory.
# Receives the per-candidate denoised estimate ``x0_hat`` (tokens, dim), the
# conditional velocity ``cond``, the CFG-guided velocity ``v`` and the current
# timestep ``tcurr``; returns a 1-D score tensor with one entry per candidate.
RewardFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, float], torch.Tensor]


def cfg_velocity_agreement(x0_hat, cond, v, tcurr):
    """Default parameter-free reward proxy.

    Returns the per-candidate cosine alignment between the conditional velocity
    ``cond`` and the CFG-guided velocity ``v``. When the two agree, the model is
    already committing to the prompt and CFG is amplifying a confident
    direction rather than fighting an ambiguous one — a reasonable stand-in for
    "promising trajectory" that needs no learned reward model. With CFG
    disabled (``v == cond``) every candidate scores equally and pruning becomes
    a deterministic first-k tie-break; supply a real ``reward`` for reward-guided
    selection in that regime.
    """
    a = cond.flatten(1).float()
    b = v.flatten(1).float()
    num = (a * b).sum(-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-8
    return num / den


def _checkpoint_indices(num_checkpoints, steps, front_load_fraction):
    """Step indices (within ``range(steps)``) at which to prune.

    Checkpoints are spread evenly across the first ``front_load_fraction`` of
    the trajectory (the noisiest, cheapest-to-explore region) and clamped to
    ``[1, steps - 1]`` so each leaves room to denoise the survivors.
    """
    if num_checkpoints <= 0 or steps <= 1:
        return []
    raw = sorted(
        {round(j / num_checkpoints * front_load_fraction * steps)
         for j in range(1, num_checkpoints + 1)}
    )
    out = []
    for cp in raw:
        cp = max(1, min(cp, steps - 1))
        if not out or cp > out[-1]:
            out.append(cp)
    return out


def _topk_per_group(scores, group, k):
    """Row indices of the top-``k`` scorers within each prompt group."""
    keep = []
    for g in torch.unique(group):
        rows = (group == g).nonzero(as_tuple=False).reshape(-1)
        order = rows[torch.argsort(scores[rows].to(torch.float32), descending=True)]
        keep.append(order[:k])
    if not keep:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.cat(keep)


def estimated_nfe(num_candidates, keep_counts, steps, cfg, front_load_fraction=0.5):
    """Number of model forward passes PSP performs for one prompt batch.

    Counts both conditional and unconditional passes when CFG is on. Mirrors the
    schedule used by :func:`sample_with_progressive_pruning` so a caller can
    compare PSP against best-of-N at matched compute (see :func:`match_best_of_n`).
    """
    evals_per_step = 2 if cfg else 1
    boundaries = _checkpoint_indices(len(keep_counts), steps, front_load_fraction)
    alive = num_candidates
    nfe = 0
    bi = 0
    for step in range(steps):
        nfe += alive * evals_per_step
        if bi < len(boundaries) and step == boundaries[bi]:
            alive = keep_counts[bi]
            bi += 1
    return nfe


def _halving_keep_counts(num_candidates):
    """Default survivor schedule: halve the candidate set at each checkpoint."""
    keeps = []
    k = num_candidates // 2
    while k >= 1:
        keeps.append(k)
        k //= 2
    return tuple(keeps)


def match_best_of_n(baseline_candidates, steps, cfg, front_load_fraction=0.5,
                    max_candidates=64):
    """Pick a PSP schedule whose NFE matches a best-of-N baseline.

    Returns ``(num_candidates, keep_counts)`` for the largest initial fan-out
    whose :func:`estimated_nfe` does not exceed ``baseline_candidates`` full
    trajectories — i.e. the front-loaded exploration PSP can afford for the same
    compute budget. ``keep_counts`` uses the :func:`_halving_keep_counts` shape.
    """
    target = baseline_candidates * steps * (2 if cfg else 1)
    best = (baseline_candidates, _halving_keep_counts(baseline_candidates))
    for num_candidates in range(baseline_candidates + 1, max_candidates + 1):
        keep_counts = _halving_keep_counts(num_candidates)
        if not keep_counts:
            continue
        if estimated_nfe(num_candidates, keep_counts, steps, cfg,
                         front_load_fraction) <= target:
            best = (num_candidates, keep_counts)
        else:
            break
    return best


@torch.no_grad()
def sample_with_progressive_pruning(
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
    num_candidates=4,
    keep_counts=(2, 1),
    front_load_fraction=0.5,
    reward: RewardFn = cfg_velocity_agreement,
    return_all=False,
):
    """Text-to-image sampling with Progressive Seed Pruning.

    Behaves like :func:`sampling.sample` but fans each prompt out to
    ``num_candidates`` seeds, prunes to ``keep_counts[i]`` survivors per prompt
    at successive front-loaded checkpoints, and returns the best survivor per
    prompt (or every survivor when ``return_all=True``). Total model evaluations
    are reported by :func:`estimated_nfe` for the same arguments.
    """
    if num_candidates < 1:
        raise ValueError("num_candidates must be >= 1")
    if any(k < 1 or k > num_candidates for k in keep_counts):
        raise ValueError("each keep count must be in [1, num_candidates]")

    patch = model.config.patch
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n
    total = n * num_candidates

    # One distinct seed per (prompt, candidate): front-loaded exploration.
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
                    seed + p * num_candidates + c
                ),
            )
            for p in range(n)
            for c in range(num_candidates)
        ],
        dim=0,
    )

    # Encode each prompt once, then fan its conditioning out to all candidates.
    txt, txtmask = encoder(prompts)
    txt = txt.repeat_interleave(num_candidates, dim=0)
    txtmask = txtmask.repeat_interleave(num_candidates, dim=0)
    img, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)

    if cfg:
        untxt, untxtmask = encoder(negative_prompts)
        untxt = untxt.repeat_interleave(num_candidates, dim=0)
        untxtmask = untxtmask.repeat_interleave(num_candidates, dim=0)
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2
    ts = timesteps(img.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    # Per-row prompt id, for pruning *within* each prompt's candidate set.
    group = torch.arange(n, device=device).repeat_interleave(num_candidates)
    score_acc = torch.zeros(total, device=device, dtype=torch.float32)

    checkpoints = set(_checkpoint_indices(len(keep_counts), steps, front_load_fraction))
    keep_iter = iter(keep_counts)

    for step, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
        cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
        if cfg:
            uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
            v = cond + guidance * (cond - uncond)
        else:
            uncond = None
            v = cond

        # Score the intermediate denoised estimate and prune *before* stepping,
        # so only survivors pay for the Euler update (and every later step).
        if step in checkpoints:
            x0_hat = img - tcurr * v
            scores = reward(x0_hat, cond, v, float(tcurr)).reshape(-1).to(score_acc.dtype)
            score_acc = score_acc + scores
            idx = _topk_per_group(scores, group, next(keep_iter))
            img, txt, pos, mask = img[idx], txt[idx], pos[idx], mask[idx]
            cond, v = cond[idx], v[idx]
            if cfg:
                uncond, unpos, unmask = uncond[idx], unpos[idx], unmask[idx]
            group = group[idx]
            score_acc = score_acc[idx]

        img = img + (tprev - tcurr) * v

    # Best survivor per prompt by accumulated checkpoint reward.
    chosen = (
        torch.arange(len(img), device=device)
        if return_all
        else _topk_per_group(score_acc, group, 1)
    )
    img = img[chosen]

    # Unpatchify and decode, mirroring sampling.sample.
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
