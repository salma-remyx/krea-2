"""Stability-guided adaptive token merging for K2 sampling.

Adapted from SADA: Stability-guided Adaptive Diffusion Acceleration
(arXiv:2507.17135). The paper's central observation is that training-free
diffusion accelerators apply a *fixed* per-step compute-reduction schedule
regardless of prompt or timestep, and that this fixed schedule is the main
source of their fidelity gap: different prompts trace different denoising
trajectories, so the amount of per-step token compression that is safe varies
along the trajectory. SADA makes that schedule *adaptive* to trajectory
stability.

This module keeps that core idea -- a per-step merge ratio derived from how
stable the trajectory currently is -- and applies it as a training-free hook
around the existing ``model(...)`` call in :func:`sampling.sample`'s Euler
loop, without changing the model's forward I/O contract.

This is a Mode 2 (adapted) port; the following components are target-native
substitutions rather than the paper's exact method:

  * **Merge site.** Merging happens once per model call on the image-token
    sequence (merge before the call, scatter the returned velocity back after)
    instead of via per-layer hooks on every :class:`mmdit.Attention`. This
    yields a single speedup point that shortens the sequence *every* attention
    layer sees, without surgery across the 28 DiT blocks. It matches the
    integration shape called out for this repo ("wrap the existing ``model()``
    call ... without changing the forward I/O contract").
  * **Stability estimator.** Trajectory stability is a parameter-free proxy --
    the normalized latent delta between consecutive steps -- rather than a
    learned estimator.
  * **Token affinity.** Cluster assignment uses cosine similarity on the raw
    patch features (parameter-free), with a bipartite ToMe-style merge.

Per-prompt / benchmark fidelity evaluation against the paper's reported numbers
is intentionally out of scope here and belongs in a downstream PR.
"""

import torch
from torch import Tensor


def build_clusters(img: Tensor, ratio: float) -> tuple[Tensor, int]:
    """Assign each image token to a merge cluster via bipartite similarity.

    Returns ``(cluster_id, k)`` where ``cluster_id`` is a length-``n`` long
    tensor with values in ``[0, k)`` and ``k = n - r`` with ``r`` the number of
    tokens merged away at this step. Tokens at even positions anchor a cluster;
    the ``r`` odd-position tokens most similar (cosine) to their best anchor are
    folded into it, and the remaining odd tokens stay as their own clusters.

    The assignment is shared across the batch (every item shares one patch
    grid) and computed from batch-averaged affinity, so the reduced sequence
    stays stackable. With ``ratio == 0`` the result is a length-preserving
    permutation that the caller's scatter/gather round-trips back to identity.
    """
    n = img.shape[1]
    r = int(round(ratio * n))
    r = max(0, min(r, n - 1))

    a_idx = torch.arange(0, n, 2, device=img.device)
    b_idx = torch.arange(1, n, 2, device=img.device)
    na = a_idx.numel()
    nb = b_idx.numel()

    cluster_id = torch.empty(n, dtype=torch.long, device=img.device)
    cluster_id[a_idx] = torch.arange(na, device=img.device)

    r_eff = min(r, nb)
    if r_eff == 0:
        cluster_id[b_idx] = torch.arange(na, na + nb, device=img.device)
        return cluster_id, na + nb

    a = img[:, a_idx].mean(0).float()
    b = img[:, b_idx].mean(0).float()
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-6)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
    sim = b @ a.t()  # (nb, na): affinity of each b token to each a anchor

    best_a = sim.argmax(dim=-1)
    best_sim = sim.max(dim=-1).values
    merged_b = torch.topk(best_sim, r_eff, dim=-1).indices

    idx = torch.arange(nb, device=img.device)
    is_merged = torch.isin(idx, merged_b)
    lone_rank = (~is_merged).cumsum(0) - 1  # rank among the un-merged b tokens
    b_cluster = torch.where(is_merged, best_a, na + lone_rank)
    cluster_id[b_idx] = b_cluster

    return cluster_id, na + nb - r_eff


def scatter_mean(src: Tensor, cluster_id: Tensor, k: int) -> Tensor:
    """Average ``src`` (b, n, *) down to ``k`` cluster rows along dim 1."""
    counts = torch.bincount(cluster_id, minlength=k).clamp(min=1)
    out = torch.zeros(src.shape[0], k, src.shape[2], device=src.device, dtype=src.dtype)
    out.index_add_(1, cluster_id, src)
    return out / counts.view(1, -1, 1)


class StabilityMerge:
    """Per-step, stability-guided token-merging acceleration hook.

    Pass an instance to :func:`sampling.sample` via ``merge=StabilityMerge()``.
    Each Euler step the hook measures how far the latent moved since the
    previous step, turns that into a stability-weighted merge ratio, collapses
    redundant image tokens for the model call, then scatters the predicted
    velocity back onto the full token grid so the trajectory itself is never
    coarsened -- only the per-step forward pass is.

    Parameters
    ----------
    max_ratio:
        Upper bound on the fraction of image tokens merged in a single step.
    warmup:
        Fraction of the schedule (by progress) during which no merging is
        applied, letting coarse structure form before any compression.
    """

    def __init__(self, max_ratio: float = 0.5, warmup: float = 0.15):
        self.max_ratio = max_ratio
        self.warmup = warmup
        self._prev: Tensor | None = None

    def _stability(self, img: Tensor) -> float:
        """Stability in (0, 1]: 1 means the latent barely moved since last step."""
        prev = self._prev
        if prev is None or prev.shape != img.shape:
            return 0.0
        imgf = img.float()
        change = (imgf - prev.float()).norm() / (imgf.norm() + 1e-6)
        return float(1.0 / (1.0 + change))

    def _ratio(self, tcurr: float, stability: float) -> float:
        """Per-step merge ratio from schedule progress x trajectory stability."""
        progress = max(0.0, min(1.0, 1.0 - float(tcurr)))
        if self.warmup >= 1.0 or progress <= self.warmup:
            w = 0.0
        else:
            u = (progress - self.warmup) / (1.0 - self.warmup)
            w = u * u * (3 - 2 * u)  # smoothstep ramp
        ratio = self.max_ratio * w * stability
        return max(0.0, min(self.max_ratio, ratio))

    @torch.no_grad()
    def step(
        self,
        img: Tensor,
        model,
        txt: Tensor,
        untxt: Tensor | None,
        pos: Tensor,
        unpos: Tensor | None,
        mask: Tensor,
        unmask: Tensor | None,
        t: Tensor,
        tcurr: float,
        guidance: float,
        cfg: bool,
    ) -> Tensor:
        """Run one accelerated Euler step; returns the velocity on the full grid."""
        stability = self._stability(img)
        self._prev = img.detach()
        ratio = self._ratio(tcurr, stability)

        b = img.shape[0]
        txtlen = txt.shape[1]
        cluster_id, k = build_clusters(img, ratio)

        m_img = scatter_mean(img, cluster_id, k)
        m_imgpos = scatter_mean(pos[:, txtlen:], cluster_id, k)
        valid = torch.ones(b, k, dtype=torch.bool, device=img.device)

        cpos = torch.cat([pos[:, :txtlen], m_imgpos], dim=1)
        cmask = torch.cat([mask[:, :txtlen], valid], dim=1)
        cond = model(img=m_img, context=txt, t=t, pos=cpos, mask=cmask)

        if cfg:
            utxtlen = untxt.shape[1]
            upos = torch.cat([unpos[:, :utxtlen], m_imgpos], dim=1)
            umask = torch.cat([unmask[:, :utxtlen], valid], dim=1)
            uncond = model(img=m_img, context=untxt, t=t, pos=upos, mask=umask)
            v = cond + guidance * (cond - uncond)
        else:
            v = cond

        # Scatter the reduced velocity back onto the original token grid so the
        # Euler update advances every token (merged tokens share their
        # representative's velocity -- the acceleration approximation).
        return v[:, cluster_id]
