"""Progressive Seed Pruning (PSP) — training-free inference-time scaling.

Adapted from "Inference-Time Scaling of Diffusion Models via Progressive Seed
Pruning" (arXiv:2607.21591). The paper's CORE mechanism — front-load seed
exploration, score intermediate denoised estimates, and progressively prune
unpromising trajectories so only the best seeds are fully denoised — is kept at
full fidelity. Its AUXILIARY black-box reward scorer is replaced by a
parameter-free latent-structure proxy (``latent_structure_score``); pass a real
reward via ``reward=`` to recover the paper's setup.

This wraps the existing flow-matching sampler primitives in ``sampling`` — the
model forward, checkpoints, and prompt->image contract are untouched.
"""

from itertools import pairwise

import torch
from einops import rearrange
from PIL import Image

from sampling import prepare, roundup, timesteps


def latent_structure_score(x0, *, patch, h, w):
    """Parameter-free reward proxy for a predicted-clean latent ``x0``.

    PSP scores intermediate denoised estimates with a (learned) reward model.
    As a target-native stand-in needing no extra weights, we unpatchify ``x0``
    back to the latent grid and take its mean squared spatial-gradient energy —
    a sharpness / detail prior. Candidates whose estimate already commits to
    structured detail score higher and are kept; blurry or uncommitted
    estimates are pruned. It is cheap (no autoencoder decode) and parameter
    free. Swap in a real reward via the ``reward`` argument of ``sample_psp``.

    Args:
        x0: predicted clean latent, patchified as ``[b, n_tok, c*patch*patch]``.
        patch / h / w: patch size and latent-grid dims used to unpatchify.

    Returns:
        A ``[b]`` tensor of per-candidate scores (higher == keep).
    """
    lat = rearrange(
        x0, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=h, w=w
    )
    dx = lat[..., :, 1:] - lat[..., :, :-1]
    dy = lat[..., 1:, :] - lat[..., :-1, :]
    return dx.pow(2).flatten(1).mean(1) + dy.pow(2).flatten(1).mean(1)


def _prune_schedule(num_candidates, keep, nsteps):
    """Candidate counts to prune down to, keyed by the step index after which
    the prune fires. Counts halve geometrically until reaching ``keep``;
    checkpoints spread across the trajectory so the bulk of the (larger)
    candidate set is evaluated up front — i.e. exploration is front-loaded."""
    levels = []
    c = num_candidates
    while c > keep:
        c = max(keep, c // 2)
        levels.append(c)
    if not levels or nsteps < 1:
        return {}
    k = len(levels)
    idxs = [min(max(1, round((i + 1) * nsteps / k)), nsteps - 1) for i in range(k)]
    # Later (smaller) targets win on index collisions; levels[-1] == keep lands
    # at the final step, guaranteeing we reach exactly `keep` survivors.
    return dict(zip(idxs, levels))


@torch.no_grad()
def sample_psp(
    model,
    ae,
    encoder,
    prompts,
    *,
    explore=4,
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
    """Sample ``len(prompts)`` images via Progressive Seed Pruning.

    Explores ``explore`` candidate seeds per requested image (``m = n*explore``
    trajectories), scores the predicted-clean latent at progressive checkpoints,
    and prunes to the highest-scoring survivors so only promising seeds are
    fully denoised. Returns ``len(prompts)`` images — same contract as
    ``sampling.sample``.

    ``reward`` (optional) maps a patchified predicted-clean latent
    ``[b, n_tok, c*patch*patch]`` to per-candidate scores ``[b]``; when omitted
    the parameter-free ``latent_structure_score`` proxy is used.

    The printed NFE (network forward count) lets you match a best-of-N budget:
    pruning keeps total evaluations close to a plain run over the survivors.
    """
    patch = model.config.patch
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = len(prompts)
    m = n * explore
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n
    negative_prompts = negative_prompts * explore  # tile per-prompt negatives to m seeds

    # Per-candidate seeded gaussian latent noise (seeds seed..seed+m-1).
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
            for i in range(m)
        ],
        dim=0,
    )

    txt, txtmask = encoder(prompts)
    # Broadcast conditioning across all candidate seeds of the same prompt.
    txt = txt.repeat(explore, 1, 1)
    txtmask = txtmask.repeat(explore, 1)
    img, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)

    if cfg:
        untxt, untxtmask = encoder(negative_prompts)
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2
    ts = timesteps(img.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    latent_h = height // (ae.compression * patch)
    latent_w = width // (ae.compression * patch)
    prune_after = _prune_schedule(m, n, len(ts) - 1)

    nfe = 0
    tcurr = tprev = v = None
    for step, (tcurr, tprev) in enumerate(pairwise(ts)):
        t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
        cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
        nfe += len(img)
        if cfg:
            uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
            nfe += len(img)
            v = cond + guidance * (cond - uncond)
        else:
            v = cond
        img = img + (tprev - tcurr) * v

        target = prune_after.get(step)
        if target is not None and len(img) > target:
            keep = _select(img - tcurr * v, target, reward, patch, latent_h, latent_w)
            img, txt, pos, mask = img[keep], txt[keep], pos[keep], mask[keep]
            if cfg:
                untxt, unpos, unmask = untxt[keep], unpos[keep], unmask[keep]

    # Safety net: guarantee exactly n survivors regardless of step/schedule edges.
    if len(img) > n:
        x0 = img - tcurr * v if tcurr is not None else img
        keep = _select(x0, n, reward, patch, latent_h, latent_w)
        img = img[keep]

    img = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=latent_h,
        w=latent_w,
    )
    img = ae.decode(img.to(torch.bfloat16))
    img = img.clamp(-1, 1) * 0.5 + 0.5
    img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    baseline = n * (len(ts) - 1) * (2 if cfg else 1)
    print(
        f"[psp] explore={explore} candidates={m}->{n} "
        f"NFE={nfe} (best-of-{n} baseline NFE={baseline})"
    )
    return [Image.fromarray(img[i]) for i in range(len(img))]


def _select(x0, k, reward, patch, h, w):
    """Indices of the ``k`` highest-scoring candidates in ``x0``."""
    scores = (
        reward(x0)
        if reward is not None
        else latent_structure_score(x0, patch=patch, h=h, w=w)
    )
    return torch.topk(scores, k).indices
