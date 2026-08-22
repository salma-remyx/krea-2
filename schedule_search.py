"""Black-box timestep-schedule search for the K2 sampler.

Adapted from "Optimize Your Sampling: Tuned Diffusion Sampling with Bayesian
Optimization" (OYS, arXiv:2608.18040). OYS treats timestep selection as a
black-box optimization problem on the quality metric you actually care about,
rather than on a theoretically derived surrogate for it. The search here runs
entirely at inference time -- no training, no gradients through the model --
so it applies to the distilled few-step checkpoints exactly as it does to the
raw model: a candidate schedule is just a monotone list of timesteps handed to
the existing Euler loop in `sampling.sample`.

Substitutions relative to the paper (Mode 2): the GP surrogate is a
squared-exponential GP with ARD length scales fitted by direct likelihood
ascent rather than a BO library's optimizer, acquisition is GP-UCB over a
random candidate pool mixed with a shrinking neighborhood of the incumbent
rather than continuous acquisition optimization, and the target metric is
supplied by the caller as a `scorer` callable (the paper's reward models and
human-eval protocol are out of scope).
"""

import json
import math
import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from sampling import sample, timesteps

# Weight-space box; softmax keeps every decode monotone for any value inside.
WEIGHT_BOUNDS = (-6.0, 6.0)


@dataclass
class SearchResult:
    """Best schedule found, plus every (schedule, score) pair evaluated."""

    timesteps: list[float]
    score: float
    steps: int
    history: list[tuple[list[float], float]] = field(default_factory=list)

    def save(self, path):
        """Write the winning schedule to `path` as a JSON list of timesteps."""
        save_schedule(self.timesteps, path)


class ScheduleSpace:
    """Search space over monotone t: 1 -> 0 schedules with a fixed step count.

    Endpoints are pinned (`t_0 = 1`, `t_N = 0`) and the interior timesteps are
    parameterized by softmax weights over the step gaps, so every candidate is
    strictly decreasing by construction -- no rejection sampling needed. The
    default `sampling.timesteps` schedule lives inside the space, which lets the
    tuner seed the search with it and guarantees the incumbent is always beaten
    on measured score rather than by assumption.
    """

    def __init__(self, steps, *, seq_len=1024, x1=256, x2=6400, y1=0.5, y2=1.15, mu=None):
        if steps < 2:
            raise ValueError("a schedule needs at least 2 steps (one interior timestep)")
        self.steps = steps
        # One weight per interval gap, so `steps` weights cover the `steps - 1`
        # interior timesteps between the two pinned endpoints.
        self.dim = steps
        # x1/x2 defaults mirror sampling.sample(minres=256, maxres=1280) at its
        # default latent alignment of 16.
        self.seed = timesteps(seq_len, steps, x1, x2, y1=y1, y2=y2, mu=mu)

    def decode(self, z):
        """Softmax weights -> interior timesteps -> full 1 -> 0 schedule."""
        gaps = torch.cumsum(torch.softmax(torch.as_tensor(z, dtype=torch.float64), 0), 0)
        interior = (1.0 - gaps[: self.steps - 1]).tolist()
        return [1.0, *interior, 0.0]

    def seed_weights(self):
        """Weights of the default `sampling.timesteps` schedule."""
        p = self.seed
        return [math.log(max(p[i] - p[i + 1], 1e-6)) for i in range(self.steps)]


@dataclass
class _Fit:
    x: Tensor
    y: Tensor
    scale: Tensor
    chol: Tensor


# Observation noise on the (normalized) scores. Rendered-image metrics are
# noisy in seed and prompt; even deterministic scorers benefit, because a
# noise-free GP interpolates exactly through every observation and leaves the
# acquisition nothing to explore with.
NOISE = 0.05


def _kernel(a, b, scale):
    """Squared-exponential kernel with one length scale per dimension (ARD)."""
    d2 = ((a[:, None, :] - b[None, :, :]) / scale) ** 2
    return torch.exp(-0.5 * d2.sum(-1))


def _chol(k):
    """Cholesky with escalating jitter, for near-duplicate observations."""
    eye = torch.eye(k.shape[0], dtype=k.dtype)
    for jitter in (1e-6, 1e-4, 1e-2):
        try:
            return torch.linalg.cholesky(k + jitter * eye)
        except torch.linalg.LinAlgError:
            continue
    raise torch.linalg.LinAlgError("kernel matrix is not positive definite")


def _log_marginal(x, y, log_scale):
    """GP log marginal likelihood, up to the constant dropped by argmax."""
    k = _kernel(x, x, log_scale.exp()) + NOISE**2 * torch.eye(len(y), dtype=torch.float64)
    chol = _chol(k)
    alpha = torch.cholesky_solve(y[:, None], chol)
    return -0.5 * (y @ alpha.squeeze(1)) - torch.log(torch.diagonal(chol)).sum()


def _fit_gp(z, y, rng, restarts=4, steps=40, lr=0.15):
    """Fit a GP surrogate: ARD length scales by random-restart local ascent.

    The paper fits its GP with a standard BO library. Here the marginal
    likelihood is maximized directly -- Adam over log length scales from a few
    random restarts, keeping the best likelihood seen along the way. Each
    evaluation is only a Cholesky of an n x n matrix, and the ascent rarely
    moves far from its start, so restarts cover the space more reliably than
    one long run.
    """
    x = torch.as_tensor(z, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    y = (y - y.mean()) / (y.std() + 1e-9)
    lo, hi = WEIGHT_BOUNDS
    span = hi - lo
    noise = NOISE**2 * torch.eye(len(y), dtype=torch.float64)

    best = (-torch.inf, None)
    for r in range(max(restarts, 1)):
        start = (
            torch.zeros(x.shape[1], dtype=torch.float64)
            if r == 0
            else torch.log(
                torch.tensor(
                    [span * 10 ** rng.uniform(-2.0, 0.5) for _ in range(x.shape[1])],
                    dtype=torch.float64,
                )
            )
        ).requires_grad_(True)
        opt = torch.optim.Adam([start], lr=lr)
        for _ in range(max(steps, 1)):
            opt.zero_grad()
            ll = _log_marginal(x, y, start)
            (-ll).backward()
            opt.step()
        ll = float(ll.detach())
        if ll > best[0]:
            with torch.no_grad():
                scale = start.exp()
                chol = _chol(_kernel(x, x, scale) + noise)
                best = (ll, _Fit(x, y, scale, chol))
    return best[1]


def _propose(fit, rng, candidates=512, kappa=2.0, incumbent=None):
    """Next point to evaluate: highest GP-UCB over a random candidate pool."""
    dim = fit.x.shape[1]
    lo, hi = WEIGHT_BOUNDS
    n_uniform = candidates if incumbent is None else candidates // 2
    pool = [
        [rng.uniform(lo, hi) for _ in range(dim)] for _ in range(n_uniform)
    ]
    if incumbent is not None:
        # Shrinking local neighborhood of the incumbent, so the search both
        # explores the box and localizes around the best schedule so far.
        radius = (hi - lo) / (2.0 + 3.0 * (len(fit.x) / 20.0))
        while len(pool) < candidates:
            pool.append(
                [min(hi, max(lo, v + rng.gauss(0, radius))) for v in incumbent]
            )
    pool = torch.tensor(pool, dtype=torch.float64)

    k = _kernel(pool, fit.x, fit.scale)
    solved = torch.cholesky_solve(k.T, fit.chol)
    var = (1.0 + NOISE**2 - torch.einsum("ij,ji->i", k, solved)).clamp_min(1e-9)
    mean = k @ fit.y
    ucb = mean + kappa * torch.sqrt(var)
    return [float(v) for v in pool[int(torch.argmax(ucb))]]


def tune(
    scorer,
    steps,
    *,
    init=5,
    iters=15,
    seed=0,
    space=None,
    kappa=2.0,
    verbose=True,
):
    """Search for a timestep schedule that maximizes `scorer(schedule)`.

    `scorer` receives a candidate schedule (a monotone list of timesteps,
    1 -> 0, of length `steps` + 1) and returns a float -- higher is better.
    Supply whatever metric you actually report, e.g. an image-reward model or
    CLIP score over a fixed prompt set. The default schedule is always
    evaluated first, so the returned schedule only loses to it through scorer
    noise, never through the search.
    """
    rng = random.Random(seed)
    space = space or ScheduleSpace(steps)
    seen, zs, ys = set(), [], []

    def evaluate(z):
        schedule = space.decode(z)
        key = tuple(round(t, 6) for t in schedule)
        if key in seen:
            return
        seen.add(key)
        score = float(scorer(schedule))
        zs.append(list(z))
        ys.append(score)
        if verbose:
            print(f"[tune] n={len(ys)} score={score:.4f} best={max(ys):.4f}")

    evaluate(space.seed_weights())
    for _ in range(max(init, 1)):
        evaluate([rng.uniform(*WEIGHT_BOUNDS) for _ in range(space.dim)])
    for _ in range(max(iters, 1)):
        incumbent = zs[max(range(len(ys)), key=ys.__getitem__)]
        evaluate(_propose(_fit_gp(zs, ys, rng), rng, kappa=kappa, incumbent=incumbent))

    best = max(range(len(ys)), key=ys.__getitem__)
    return SearchResult(
        timesteps=space.decode(zs[best]),
        score=ys[best],
        steps=space.steps,
        history=[(space.decode(z), s) for z, s in zip(zs, ys)],
    )


def sampler(model, ae, encoder, prompts, **kwargs):
    """Bind a scorer that renders one image per candidate schedule."""
    prompts = [prompts] if isinstance(prompts, str) else list(prompts)

    def render(schedule):
        return sample(model, ae, encoder, prompts, schedule=schedule, **kwargs)[0]

    return render


def mean_pairwise_distance(images):
    """Diversity reward over a batch of images -- a dependency-free smoke scorer.

    Mean L2 distance in pixel space between all pairs. Tuning to maximize this
    trades detail for coverage; it is a stand-in for the reward model you
    actually report, useful for checking the search loop end to end.
    """
    if len(images) < 2:
        return 0.0
    flat = torch.stack(
        [torch.as_tensor(np.asarray(img), dtype=torch.float32).flatten() for img in images]
    )
    d2 = ((flat[:, None, :] - flat[None, :, :]) ** 2).sum(-1)
    n = len(images)
    return float(d2.sum().item() / (n * (n - 1)))


def save_schedule(schedule, path):
    """Write a schedule to `path` as a JSON list of timesteps."""
    with open(path, "w") as f:
        json.dump([round(float(t), 8) for t in schedule], f, indent=2)


def load_schedule(path):
    """Read a schedule written by `save_schedule` / `SearchResult.save`."""
    with open(path) as f:
        schedule = json.load(f)
    schedule = [float(t) for t in schedule]
    if len(schedule) < 3:
        raise ValueError(f"{path}: a schedule needs at least 3 timesteps")
    if any(b >= a for a, b in zip(schedule, schedule[1:])):
        raise ValueError(f"{path}: timesteps must be strictly decreasing")
    return schedule
