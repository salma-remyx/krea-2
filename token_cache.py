"""Token-wise attention feature caching for the K2 MMDiT.

Adapted from "Rethinking Token-wise Feature Caching: Accelerating Diffusion
Transformers with Dual Feature Caching" (DuCa), Zou et al., arXiv:2412.18911.

Two findings from the paper drive the design, both kept at full fidelity:

* Alternating strategies. Cached denoise steps alternate between an
  *aggressive* strategy (reuse the step's cached attention features
  wholesale) and a *conservative* one (recompute a random subset of tokens,
  reuse cached features for the rest). The strategies' caching errors
  partially offset each other across steps.
* Random selection. Which tokens get recomputed is drawn at random — the
  paper found importance-based selection unreliable, sometimes worse than
  random.

What is cached is each block's attention-branch output (post-``wo``, pre
``pregate``), so a reused feature is still rescaled by the *current* step's
timestep modulation. Entries are keyed by (CFG pass, block index) so the
conditional and unconditional branches never share features.

Substituted for the paper's auxiliary machinery: the per-layer reuse-ratio
schedules and the paper's benchmark suite are dropped, and selection is a
seeded per-step draw rather than a learned or attention-based scorer. On
conservative steps the k/v projections are still computed for the full
sequence (so recomputed queries attend to fresh keys) — the paper instead
reuses cached tokens' k/v outright.
"""

from contextlib import contextmanager

import torch


def gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather token rows `idx` (B, K) from the leading two dims of `x`.

    Works for (B, L, D) features, (B, H, L, D) q/k/v, and (B, 1, L, L)
    attention masks — anything whose token axis is dim 1 or 2.
    """
    if x.dim() == 3:
        return x.gather(1, idx[:, :, None].expand(-1, -1, x.shape[-1]))
    return x.gather(2, idx[:, None, :, None].expand(x.shape[0], -1, -1, x.shape[-1]))


def scatter_rows(x: torch.Tensor, idx: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Inverse of `gather_rows` along the token axis of a (B, L, D) tensor."""
    return x.scatter(1, idx[:, :, None].expand(-1, -1, x.shape[-1]), src)


class DualTokenCache:
    """Global cache state consulted by ``mmdit.Attention.forward``.

    A single instance (:data:`dual_cache`) is shared process-wide. The
    sampler opens it with :meth:`configured` and calls :meth:`begin_step`
    once per denoise step; the model calls :meth:`begin_pass` at the top of
    each forward so the cond / uncond CFG branches get separate entries.
    """

    def __init__(self, interval: int = 2):
        self.interval = interval  # compute fully every `interval` steps
        self.enabled = False
        self.reuse = 0.0
        self._step = -1
        self._cached_seen = 0
        self._pass = 0
        self._branch = 0
        self._mode = "compute"
        self._features: dict[tuple[int, int], torch.Tensor] = {}
        self._gen = torch.Generator().manual_seed(0x5EED)

    @contextmanager
    def configured(self, *, reuse: float | None = None):
        """Enable caching for a sampling run.

        `reuse` is the fraction of tokens kept cached on conservative steps;
        pass None (or <= 0) to run uncached, which the context guarantees on
        exit regardless of exceptions.
        """
        self.enabled = reuse is not None and reuse > 0
        self.reuse = min(float(reuse), 0.99) if self.enabled else 0.0
        self.reset()
        try:
            yield self
        finally:
            self.enabled = False
            self._features.clear()

    def reset(self):
        self._step = -1
        self._cached_seen = 0
        self._pass = 0
        self._branch = 0
        self._mode = "compute"
        self._features.clear()
        self._gen = torch.Generator().manual_seed(0x5EED)

    def begin_step(self):
        """Advance to the next denoise step and pick this step's strategy.

        Step 0 always computes fully (the cache needs a seed). After that,
        every other step caches; cached steps alternate aggressive ->
        conservative -> aggressive..., so the two strategies' caching errors
        offset each other across the trajectory (DuCa's core finding).
        """
        self._branch = 0
        self._pass = 0
        self._step += 1
        if not self.enabled or self._step % self.interval == 0:
            self._mode = "compute"
            return
        # Cached steps alternate aggressive -> conservative -> aggressive...
        # so the two strategies' caching errors offset each other.
        self._mode = "aggressive" if self._cached_seen % 2 == 0 else "conservative"
        self._cached_seen += 1

    def begin_pass(self):
        """Mark a new model forward (pass 0 = conditional, 1 = unconditional)."""
        self._branch = self._pass
        self._pass += 1

    @property
    def mode(self) -> str:
        return self._mode if self.enabled else "compute"

    def get(self, layer_idx: int) -> torch.Tensor | None:
        return self._features.get((self._branch, layer_idx))

    def store(self, layer_idx: int, features: torch.Tensor):
        self._features[(self._branch, layer_idx)] = features

    def fresh_indices(self, length: int, device) -> torch.Tensor:
        """Draw which token positions to recompute on a conservative step.

        Random selection, per the paper's finding that random matches or
        beats importance-based selection. Deterministic given the run seed.
        """
        keep = max(1, round(length * (1.0 - self.reuse)))
        scores = torch.rand((1, length), generator=self._gen).to(device)
        return scores.topk(keep, dim=-1).indices.expand(-1, -1)


dual_cache = DualTokenCache()
