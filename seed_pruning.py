"""Progressive seed pruning — inference-time scaling for the K2 flow sampler.

Fork K initial-noise candidates, partially denoise all of them, score the
intermediate latents, and prune to the most promising subset before finishing
the rest — all at a fixed number of model evaluations (NFEs). Front-loading
exploration (many candidates early, few late) picks better initial noise than
best-of-N at matched compute, an inference-time scaling axis that needs no
retraining and slots directly onto the seeded Euler loop in ``sampling.sample``.

Adapted from "Inference-Time Scaling of Diffusion Models via Progressive Seed
Pruning" (arXiv:2607.21591). The progressive prune schedule and the
fixed-NFE front-loading — the paper's core mechanism — are kept at full
fidelity. The reward is a swappable callable that defaults to a
parameter-free latent-energy proxy; plug in CLIP / aesthetics /
prompt-alignment to reproduce the paper's selection gains.
"""

import torch
from einops import rearrange
from PIL import Image

from sampling import prepare, roundup, timesteps


def latent_energy_reward(x, t, context):
    """Default parameter-free reward proxy: mean squared latent energy.

    A deterministic placeholder ranking over candidates (higher energy means a
    trajectory that has committed to more structure). Replace with a real
    reward — e.g. CLIP prompt-alignment on the decoded image — for the paper's
    reported gains; the pruning schedule, not this proxy, is the contribution.
    """
    return x.flatten(1).pow(2).mean(dim=1)


def nfe_budget(num_seeds, keep, steps, cfg=False):
    """Per-image model forwards PSP spends, for matched-compute comparisons.

    Phase ``i`` runs ``candidates_i * phase_steps_i`` forwards (x2 under CFG),
    where ``candidates_0 = num_seeds`` and ``candidates_i = keep[i-1]`` for
    ``i >= 1``, and ``steps`` are split as evenly as possible across
    ``len(keep)`` phases. Compare against best-of-N's ``N * steps`` when
    choosing ``num_seeds`` / ``keep`` so PSP explores more seeds at the same
    budget rather than spending extra compute.
    """
    keep = [keep] if isinstance(keep, int) else list(keep)
    nphases = len(keep)
    if steps < nphases:
        raise ValueError("steps must be >= number of prune phases (len(keep))")
    base, rem = divmod(steps, nphases)
    sizes = [base + (1 if i < rem else 0) for i in range(nphases)]
    candidates = [num_seeds, *keep[:-1]]
    total = sum(c * s for c, s in zip(candidates, sizes))
    return total * (2 if cfg else 1)


def _prune_per_group(x, scores, group, keep_n):
    """Keep the top-``keep_n`` scoring rows within each contiguous ``group``."""
    ngroups = x.shape[0] // group
    idx = []
    for g in range(ngroups):
        base = g * group
        idx.append(torch.topk(scores[base : base + group], keep_n).indices + base)
    idx = torch.cat(idx)
    return x[idx], idx


@torch.no_grad()
def sample_pruned(
    model,
    ae,
    encoder,
    prompts,
    *,
    num_seeds,
    keep,
    reward_fn=None,
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
    """Seed-pruned variant of ``sampling.sample`` at a fixed NFE budget.

    ``num_seeds`` candidates are forked per prompt and progressively narrowed
    to ``keep[i]`` survivors after each evenly-spaced checkpoint; the final
    ``keep[-1]`` per prompt are decoded. ``keep`` must be non-increasing and
    its first entry cannot exceed ``num_seeds``. Candidate ``j`` of prompt
    ``i`` starts from ``seed + i * num_seeds + j`` and occupies a contiguous
    row, so pruning is performed within each prompt's group. Returns
    ``len(prompts) * keep[-1]`` images.
    """
    if reward_fn is None:
        reward_fn = latent_energy_reward
    keep = [keep] if isinstance(keep, int) else list(keep)
    if len(keep) == 0 or any(k < 1 for k in keep):
        raise ValueError("keep must be a non-empty sequence of positive ints")
    if any(keep[i] > keep[i - 1] for i in range(1, len(keep))):
        raise ValueError("keep must be non-increasing")
    if keep[0] > num_seeds:
        raise ValueError("keep[0] cannot exceed num_seeds")
    nphases = len(keep)

    patch = model.config.patch
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n

    # K seeded latents per prompt; candidate j of prompt i is row i*K + j.
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
                    seed + i * num_seeds + j
                ),
            )
            for i in range(n)
            for j in range(num_seeds)
        ],
        dim=0,
    )

    txt, txtmask = encoder(prompts)
    txt = txt.repeat_interleave(num_seeds, dim=0)
    txtmask = txtmask.repeat_interleave(num_seeds, dim=0)
    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)

    ctx = {"txt": txt, "pos": pos, "mask": mask}
    if cfg:
        untxt, untxtmask = encoder(negative_prompts)
        untxt = untxt.repeat_interleave(num_seeds, dim=0)
        untxtmask = untxtmask.repeat_interleave(num_seeds, dim=0)
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)
        ctx.update(untxt=untxt, unpos=unpos, unmask=unmask)

    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)
    if len(ts) - 1 < nphases:
        raise ValueError("steps must be >= number of prune phases (len(keep))")

    # Evenly split the step intervals across the prune phases (each >= 1 step).
    base, rem = divmod(len(ts) - 1, nphases)
    sizes = [base + (1 if i < rem else 0) for i in range(nphases)]
    bounds = [0]
    for s in sizes:
        bounds.append(bounds[-1] + s)

    img = x
    per_prompt = num_seeds
    for phase in range(nphases):
        # Euler integration of the flow ODE (with CFG) over this phase's steps.
        for k in range(bounds[phase], bounds[phase + 1]):
            tcurr, tprev = ts[k], ts[k + 1]
            t = torch.full((img.shape[0],), tcurr, dtype=img.dtype, device=device)
            cond = model(
                img=img, context=ctx["txt"], t=t, pos=ctx["pos"], mask=ctx["mask"]
            )
            if cfg:
                uncond = model(
                    img=img,
                    context=ctx["untxt"],
                    t=t,
                    pos=ctx["unpos"],
                    mask=ctx["unmask"],
                )
                v = cond + guidance * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v

        # Score the partially-denoised latents and prune to keep[phase]/prompt.
        t_score = torch.full(
            (img.shape[0],), ts[bounds[phase + 1]], dtype=img.dtype, device=device
        )
        scores = reward_fn(img, t_score, ctx["txt"])
        img, idx = _prune_per_group(img, scores, per_prompt, keep[phase])
        ctx = {key: val[idx] for key, val in ctx.items()}
        per_prompt = keep[phase]

    # Unpatchify the final survivors back to a latent and decode to pixels.
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
