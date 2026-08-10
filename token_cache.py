"""Cluster-aware token caching for the flow-matching sampler.

Adapted from *CAT Pruning: Cluster-Aware Token Pruning For Text-to-Image
Diffusion Models* (Zhang et al., arXiv:2502.00433). The paper accelerates
diffusion sampling by scoring each image token with its *noise relative
magnitude* across denoising iterations, selecting a *spatially coherent*
cluster of significant tokens, and reusing the cached output for the rest.

This repository's MM-DiT attends globally over a fixed, 256-padded token
count (see :func:`mmdit.SingleStreamDiT.forward`), so pruning tokens *inside*
the transformer would require an architecture change. The caching signal
instead lifts to the **velocity** level -- the very quantity the paper gates
on -- which the sampler already produces. Two levers are exposed:

  * a **per-token velocity blend**: when a step is recomputed, tokens whose
    cluster-aggregated significance is below threshold reuse the previous
    step's velocity (the cluster-aware selection, applied to cached
    velocities rather than in-transformer features);
  * a **step-level skip**: when the fraction of significant tokens drops
    below ``skip_frac`` the model forward is skipped entirely and the cached
    velocity is reused -- this is where the wall-clock saving comes from
    (both the conditional and unconditional CFG passes are skipped).

Both are opt-in and off by default; the uncached Euler + CFG path is
unchanged when ``token_cache`` is not passed to :func:`sampling.sample`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange


@dataclass
class TokenCacheConfig:
    """Knobs for :class:`ClusterAwareTokenCache`. Caching is opt-in.

    The defaults are conservative starting points; tune for the speed/quality
    trade-off of a given checkpoint (e.g. the 8-step Turbo tolerates more
    aggressive caching than the 28-52 step RAW sampler).
    """

    enabled: bool = False
    # Side length (in image tokens) of the spatial pooling window. Per-token
    # significance is avg-pooled over ``cluster x cluster`` blocks so the
    # keep decision is spatially coherent (the cluster-aware selection).
    cluster: int = 4
    # A token is "significant" when its clustered significance exceeds
    # ``keep_frac`` times the peak significance seen so far. Smaller values
    # cache more aggressively.
    keep_frac: float = 0.2
    # Skip the model forward entirely when fewer than this fraction of tokens
    # are significant. This is the real speedup path.
    skip_frac: float = 0.05
    # Upper bound on consecutive skips before forcing a fresh forward, so a
    # stale cache cannot degrade quality unbounded.
    max_skips: int = 2


def noise_relative_magnitude(
    img: torch.Tensor, v: torch.Tensor, dt: float, eps: float = 1e-6
) -> torch.Tensor:
    """Per-token significance: norm of the predicted update vs. the latent.

    ``img`` and ``v`` are ``(B, N, D)``; returns ``(B, N)``. The predicted
    step update is ``dt * v``; dividing its per-token norm by the current
    latent norm gives the *noise relative magnitude* of each token's motion
    this step. Tokens that barely move across iterations are cache candidates.
    """
    update = abs(dt) * v.float()
    return update.norm(dim=-1) / (img.float().norm(dim=-1) + eps)


def cluster_keep_mask(
    sig: torch.Tensor,
    h: int,
    w: int,
    cluster: int,
    keep_frac: float,
    peak: torch.Tensor,
) -> torch.Tensor:
    """Spatially-coherent per-token significant-token mask.

    Pools per-token ``sig`` ``(B, N)`` over ``cluster``-sized spatial blocks,
    thresholds each block against ``keep_frac * peak`` (``peak`` is the
    largest significance seen so far), then upsamples back to the token grid.
    Neighbouring tokens therefore share a decision -- the cluster-aware
    enhancement from the paper.
    """
    b = sig.shape[0]
    grid = rearrange(sig, "b (h w) -> b 1 h w", h=h, w=w).float()
    pooled = torch.nn.functional.avg_pool2d(
        grid, kernel_size=cluster, stride=cluster, ceil_mode=True
    )
    thresh = keep_frac * peak.to(pooled)
    block = (pooled >= thresh).float()
    upsampled = torch.nn.functional.interpolate(block, size=(h, w), mode="nearest")
    return upsampled.squeeze(1).reshape(b, h * w).bool()


class ClusterAwareTokenCache:
    """Drives one cached denoise loop. See module docstring for the strategy."""

    def __init__(self, cfg: TokenCacheConfig, h: int, w: int):
        self.cfg = cfg
        self.h, self.w = h, w
        # Effective cluster window, clamped to the token grid so a tiny latent
        # (e.g. in tests) cannot ask for a kernel larger than the input.
        self.cluster_eff = max(1, min(cfg.cluster, h, w))
        self.v_prev: torch.Tensor | None = None
        self.last_sig: torch.Tensor | None = None
        self.peak_sig: torch.Tensor | None = None
        self._streak = 0
        # Diagnostics: number of fresh forwards run and steps served from cache.
        self.forwards = 0
        self.skips = 0

    def _keep(self, sig: torch.Tensor) -> torch.Tensor:
        return cluster_keep_mask(
            sig, self.h, self.w, self.cluster_eff, self.cfg.keep_frac, self.peak_sig
        )

    def will_skip(self) -> bool:
        """Decide whether the upcoming forward can be skipped (cache hit).

        Predicts from the most recent forward's significance: if almost no
        tokens were significant then the cached velocity is reused instead of
        re-running the model. ``max_skips`` bounds consecutive hits.
        """
        if self.v_prev is None or self.last_sig is None:
            return False
        if self._streak >= self.cfg.max_skips:
            return False
        frac = self._keep(self.last_sig).float().mean().item()
        return frac < self.cfg.skip_frac

    def reuse(self) -> torch.Tensor:
        """Return the cached velocity and book a skip."""
        self.skips += 1
        self._streak += 1
        return self.v_prev

    def update(
        self, img: torch.Tensor, v_fresh: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """Record a fresh forward: score it, blend with the cache, store.

        Returns the velocity to integrate with -- ``v_fresh`` for significant
        tokens, the previously cached velocity for the rest.
        """
        self.forwards += 1
        self._streak = 0
        sig = noise_relative_magnitude(img, v_fresh, dt)
        step_peak = sig.amax()
        self.peak_sig = (
            step_peak if self.peak_sig is None else torch.maximum(self.peak_sig, step_peak)
        )
        self.last_sig = sig
        if self.v_prev is None:
            self.v_prev = v_fresh
            return v_fresh
        keep = self._keep(sig)
        blended = torch.where(keep[..., None], v_fresh, self.v_prev.to(v_fresh))
        self.v_prev = blended
        return blended
