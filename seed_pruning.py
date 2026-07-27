"""Progressive Seed Pruning (PSP) — inference-time scaling for the K2 sampler.

Adapted from "Inference-Time Scaling of Diffusion Models via Progressive Seed
Pruning" (arXiv:2607.21591). PSP front-loads seed exploration — denoising many
candidate seeds for the first steps — then progressively prunes the pool by
scoring each candidate's predicted-clean estimate, so only promising
trajectories are denoised to completion. Total backbone evaluations stay fixed:
the extra early breadth is paid for by fewer late-step candidates, which is the
axis the paper shows beats best-of-N / importance sampling / tree search at
matched compute.

Mode 2 (adapted port). The paper treats the per-candidate reward as a black
box. This repo is inference-only and ships no reward model, so the reward is a
pluggable callable with a parameter-free default (decoded-image sharpness) that
stands in for a learned prompt-alignment reward. The pruning schedule — PSP's
actual contribution — is implemented at full fidelity, reusing the existing
``model`` / ``ae.decode`` / ``encoder`` / ``prepare`` / ``timesteps`` /
``cfg_velocity`` untouched. Drop in any ``reward(x0_latents, decode) -> scores``
(e.g. a CLIP-score head) to recover the paper's reward-guided setup; the
benchmark / human-eval harness is intentionally out of scope (downstream PR).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from sampling import cfg_velocity, prepare, roundup, timesteps


def sharpness_reward(x0_latents, decode):
    """Parameter-free quality proxy: Laplacian variance of the decoded estimate.

    Higher = more local high-frequency detail. No learnable parameters — a
    fixed 3x3 Laplacian kernel scores the predicted-clean image as a stand-in
    for the learned prompt-alignment reward PSP assumes. PSP is reward-agnostic,
    so any ``reward(x0_latents, decode) -> (B,)`` scores drops in here.
    """
    imgs = decode(x0_latents).float()  # (B, C, H, W) in pixel space
    gray = imgs.mean(dim=1, keepdim=True)
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=gray.dtype,
        device=gray.device,
    ).view(1, 1, 3, 3)
    lap = F.conv2d(gray, kernel, padding=1)
    return lap.flatten(1).var(dim=1)


def _x0_estimate(img, v, t):
    """Predicted-clean (t=0) ranking signal from a flow state at time ``t``.

    Linear-flow extrapolation: under constant-velocity flow ``x(t) = (1-t) x0 +
    t x1`` the data estimate is ``x(t) - t * v``. Used only to *rank*
    candidates, so it need not be an exact reconstruction.
    """
    return img - t * v


def prune_steps(steps, prune_to):
    """Step indices (0-based, ascending) after which the pool is pruned.

    ``prune_to`` is the strictly-decreasing sequence of pool sizes after each
    prune (e.g. ``[4, 2, 1]``). Events are spread as evenly as the step count
    allows; entries are clamped to ``[0, steps - 2]`` and de-duplicated.
    """
    n_events = len(prune_to)
    idx = sorted(
        {((i + 1) * steps) // (n_events + 1) - 1 for i in range(n_events)}
    )
    return [j for j in idx if 0 <= j < steps - 1][:n_events]


def nfe(steps, num_candidates, prune_to):
    """Conditional-branch backbone calls for a PSP run (x2 when CFG is on)."""
    after = prune_steps(steps, prune_to)
    sizes = [num_candidates, *prune_to]
    pool, si, total = num_candidates, 0, 0
    for j in range(steps):
        total += pool
        if si < len(after) and j == after[si]:
            si += 1
            pool = sizes[si]
    return total


def _topk(scores, k):
    """Indices of the ``k`` highest scores, descending (ties keep order)."""
    return torch.argsort(scores, descending=True)[:k].tolist()


@torch.no_grad()
def sample_psp(
    model,
    ae,
    encoder,
    prompts,
    *,
    num_candidates,
    prune_to,
    reward=None,
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
    """Progressive Seed Pruning over the K2 flow-matching sampler.

    For each prompt, draws ``num_candidates`` noise seeds, denoises the whole
    pool with the shared Euler+CFG stepper, and at evenly spaced checkpoints
    scores each candidate's predicted-clean estimate, keeping the top
    ``prune_to[i]``. Survivors are denoised to completion.

    Returns ``len(prompts) * prune_to[-1]`` images (one set of survivors per
    prompt). To compare against best-of-N at matched compute, pick ``steps``,
    ``num_candidates`` and ``prune_to`` so ``nfe(steps, num_candidates, prune_to)``
    equals the best-of-N budget (``n_full_trajectories * steps``).
    """
    if reward is None:
        reward = sharpness_reward
    if not prune_to or prune_to[-1] < 1 or any(
        prune_to[i] <= prune_to[i + 1] for i in range(len(prune_to) - 1)
    ):
        raise ValueError("prune_to must be non-empty, strictly decreasing, end >= 1")
    if prune_to[0] > num_candidates:
        raise ValueError("prune_to[0] must not exceed num_candidates")

    patch = model.config.patch
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")
    cfg = guidance > 0
    n = len(prompts)
    if negative_prompts is None:
        negative_prompts = [""] * n

    h_ = height // (ae.compression * patch)
    w_ = width // (ae.compression * patch)
    after = prune_steps(steps, prune_to)
    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2

    def decode(x):
        x = rearrange(
            x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=h_, w=w_
        )
        return ae.decode(x.to(dtype))

    out = []
    for p in range(n):
        m = num_candidates
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
                        seed + p * num_candidates + i
                    ),
                )
                for i in range(m)
            ],
            dim=0,
        )

        # All candidates share the prompt conditioning; replicate across the pool.
        txt, txtmask = encoder([prompts[p]])
        txt = txt.expand(m, -1, -1)
        img, pos, mask = prepare(noise, txt.shape[1], patch, txtmask.expand(m, -1))

        untxt = unpos = unmask = None
        if cfg:
            untxt, untxtmask = encoder([negative_prompts[p]])
            untxt = untxt.expand(m, -1, -1)
            _, unpos, unmask = prepare(
                noise, untxt.shape[1], patch, untxtmask.expand(m, -1)
            )

        ts = timesteps(img.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

        si = 0
        for j, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
            v = cfg_velocity(
                model, img, tcurr, guidance, txt, pos, mask, untxt, unpos, unmask, cfg
            )
            img = img + (tprev - tcurr) * v
            if si < len(after) and j == after[si]:
                keep = _topk(reward(_x0_estimate(img, v, tprev), decode), prune_to[si])
                img, txt, pos, mask = img[keep], txt[keep], pos[keep], mask[keep]
                if cfg:
                    untxt, unpos, unmask = untxt[keep], unpos[keep], unmask[keep]
                si += 1

        final = decode(img)
        final = final.clamp(-1, 1) * 0.5 + 0.5
        final = rearrange(final * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
        out.extend(Image.fromarray(final[i]) for i in range(len(final)))
    return out
