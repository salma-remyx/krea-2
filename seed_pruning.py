"""Progressive Seed Pruning (PSP) inference-time scaling for the K2 sampler.

Inference-time scaling for diffusion / flow-matching models that, for a FIXED
model-evaluation budget, front-loads exploration (many initial noise seeds),
scores intermediate denoised estimates, and progressively prunes the candidate
set so that only the most promising trajectories are denoised to completion.

Adapted from:
    "Inference-Time Scaling of Diffusion Models via Progressive Seed Pruning"
    (arXiv:2607.21591v1) -- https://www.vision.caltech.edu/psp

ADAPTED PORT (Mode 2). The core PSP mechanism -- a pool of seed trajectories
sharing one batched black-box denoiser forward, pruned at checkpoints under a
fixed evaluation budget -- is implemented at full fidelity against this repo's
``model(img, context, t, pos, mask)`` contract (unchanged). The paper's LEARNED
reward model is substituted with a pluggable ``reward(x0_hat)`` callable plus
parameter-free default scorers (CFG-alignment / latent-sharpness proxies). The
paper's GenEval / human-eval benchmark suite is intentionally out of scope --
evaluation belongs in a downstream PR.

The compute budget is conserved exactly: scoring reuses already-computed
velocities (no extra model forward), so the total number of model evaluations
equals ``sum(survivors_per_segment * steps_per_segment)``.
"""

import torch
from einops import rearrange, repeat
from PIL import Image

from sampling import prepare, roundup, timesteps


def _checkpoints(steps, prunes):
    """Evenly place ``prunes`` pruning boundaries strictly inside the schedule."""
    if prunes <= 0 or steps < 2:
        return []
    raw = (
        min(max(round((j + 1) * steps / (prunes + 1)), 1), steps - 1)
        for j in range(prunes)
    )
    return sorted(set(raw))


def _schedule(seeds, keep, n_ckpts):
    """Per-segment survivor counts (non-increasing), from ``seeds`` down to ``keep``.

    ``n_ckpts`` prune events split the run into ``n_ckpts + 1`` segments; segment
    ``i`` is denoised with ``schedule[i]`` survivors per prompt.
    """
    keep = max(int(keep), 1)
    seeds = max(int(seeds), keep)
    if n_ckpts <= 0:
        return [seeds]
    pts = [round(seeds + (keep - seeds) * j / n_ckpts) for j in range(n_ckpts + 1)]
    out = []
    prev = seeds + 1
    for p in pts:
        p = max(min(p, prev), 1)  # non-increasing, at least 1
        out.append(p)
        prev = p
    out[0], out[-1] = seeds, keep
    return out


def _cfg_alignment(cond, uncond):
    """Parameter-free prompt-alignment proxy: mean ``|cond - uncond|`` per row.

    Higher == the prompt steers the trajectory more strongly. A parameter-free
    stand-in for a learned reward model's alignment score; only defined under
    classifier-free guidance.
    """
    return (cond - uncond).abs().mean(dim=tuple(range(1, cond.dim())))


def _latent_sharpness(x0_hat):
    """Parameter-free quality proxy: spatial spread of the denoised estimate.

    Higher token-wise std == more structure / contrast in the intermediate x0
    estimate, used as a fallback scorer when CFG is disabled. A coarse proxy,
    not a calibrated quality model.
    """
    tokens = x0_hat.mean(dim=-1) if x0_hat.dim() >= 3 else x0_hat
    return tokens.std(dim=-1)


def _prune(tensors, scores, group, survivors, n_prompts):
    """Keep the top-``survivors`` scoring rows within each prompt group.

    ``tensors`` maps name -> Tensor whose batch dim is the candidate pool; all are
    indexed identically (None entries are passed through). ``group`` gives each
    row's prompt id. Returns a new dict of indexed tensors, best-first per group.
    """
    keep_idx = []
    for gi in range(n_prompts):
        rows = (group == gi).nonzero(as_tuple=False).squeeze(-1)
        if rows.numel() == 0:
            continue
        k = min(survivors, rows.numel())
        keep_idx.append(rows[torch.topk(scores[rows], k).indices])
    idx = torch.sort(torch.cat(keep_idx)).values
    return {
        name: (t[idx] if torch.is_tensor(t) else t) for name, t in tensors.items()
    }


@torch.no_grad()
def sample_psp(
    model,
    ae,
    encoder,
    prompts,
    *,
    seeds=8,
    keep=1,
    prunes=3,
    reward=None,
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
    """Progressive Seed Pruning sampling: encode -> pooled prune-denoise -> decode.

    Mirrors ``sampling.sample``'s contract (same model / ae / encoder, same
    prompt -> image return type) but replaces the single-trajectory Euler loop
    with PSP: ``seeds`` candidate trajectories per prompt are denoised together,
    scored at ``prunes`` checkpoints, and pruned down to ``keep`` survivors so the
    returned images come only from the most promising trajectories -- at a fixed
    evaluation budget (logged as ``[psp]`` on stderr/stdout).

    ``reward``, if given, is a black-box scorer ``reward(x0_hat) -> 1D scores``
    (one per candidate), matching the paper's reward interface. When omitted, a
    parameter-free default is used: CFG-alignment under guidance, else
    latent-sharpness.

    Returns a list of ``len(prompts) * keep`` PIL images (survivors). With
    ``prunes == 0`` nothing is pruned and all ``seeds`` candidates per prompt are
    returned (best-of-seeds at full compute).
    """
    patch = model.config.patch
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    seeds = max(int(seeds), 1)
    if keep > seeds:
        print(f"[psp] keep={keep} > seeds={seeds}; clamping keep to {seeds}")
        keep = seeds

    n = len(prompts)
    cfg = guidance > 0
    neg = [""] * n if cfg else None

    # Front-load exploration: `seeds` distinct initial noises per prompt.
    pool = n * seeds
    noise = torch.cat(
        [
            torch.randn(
                1,
                ae.channels,
                height // ae.compression,
                width // ae.compression,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + j),
            )
            for j in range(pool)
        ],
        dim=0,
    )

    # Conditioning is shared across a prompt's seeds -> tile to the pool batch.
    txt, txtmask = encoder(prompts)
    txt = repeat(txt, "n s l d -> (n r) s l d", r=seeds)
    txtmask = repeat(txtmask, "n s -> (n r) s", r=seeds)
    img, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)

    untxt = unpos = unmask = None
    if cfg:
        untxt, untxtmask = encoder(neg)
        untxt = repeat(untxt, "n s l d -> (n r) s l d", r=seeds)
        untxtmask = repeat(untxtmask, "n s -> (n r) s", r=seeds)
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2
    ts = timesteps(img.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    ckpts = _checkpoints(steps, prunes)
    sched = _schedule(seeds, keep, len(ckpts))

    # Budget accounting (matches the paper's "fixed model evaluations" framing).
    bounds = ckpts + [steps]
    seg_steps = []
    prev = 0
    for b in bounds:
        seg_steps.append(b - prev)
        prev = b
    evals = (2 if cfg else 1) * sum(s * ln for s, ln in zip(sched, seg_steps))
    print(
        f"[psp] seeds={seeds} keep={keep} prunes={len(ckpts)} "
        f"schedule={sched} seg_steps={seg_steps} "
        f"evals/prompt={evals} (~best-of-{evals / ((2 if cfg else 1) * steps):.2f})"
    )

    tensors = {
        "img": img,
        "pos": pos,
        "mask": mask,
        "txt": txt,
        "untxt": untxt,
        "unpos": unpos,
        "unmask": unmask,
    }
    group = torch.arange(n, device=device).repeat_interleave(seeds)
    v = cond = uncond = None
    seg_start = 0
    for ci, bound in enumerate(bounds):
        # Denoise every current survivor through this segment's intervals.
        for k in range(seg_start, bound):
            cur = tensors["img"]
            t = torch.full((cur.shape[0],), ts[k], dtype=cur.dtype, device=device)
            cond = model(img=cur, context=tensors["txt"], t=t, pos=tensors["pos"], mask=tensors["mask"])
            if cfg:
                uncond = model(img=cur, context=tensors["untxt"], t=t, pos=tensors["unpos"], mask=tensors["unmask"])
                v = cond + guidance * (cond - uncond)
            else:
                uncond = None
                v = cond
            tensors["img"] = cur + (ts[k + 1] - ts[k]) * v
        seg_start = bound
        # Prune at every checkpoint except the final boundary.
        if ci < len(ckpts):
            nxt = sched[ci + 1]
            if nxt < sched[ci] and v is not None:
                x0_hat = tensors["img"] - ts[bound] * v
                if reward is not None:
                    scores = reward(x0_hat)
                elif cfg:
                    scores = _cfg_alignment(cond, uncond)
                else:
                    scores = _latent_sharpness(x0_hat)
                tensors = _prune(tensors, scores, group, nxt, n)
                group = torch.arange(n, device=device).repeat_interleave(nxt)

    img = tensors["img"]
    img = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=height // (ae.compression * patch),
        w=width // (ae.compression * patch),
    )
    img = ae.decode(img.to(dtype))
    img = img.clamp(-1, 1) * 0.5 + 0.5
    img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    return [Image.fromarray(img[i]) for i in range(len(img))]
