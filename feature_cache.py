"""Training-free block-feature caching for the K2 diffusion transformer.

Adapted from "Accelerating Diffusion Transformer via Increment-Calibrated
Caching with Channel-Aware Singular Value Decomposition" (ICC,
arxiv:2505.05829). The core mechanism is ported at full fidelity:

  * **Increment-calibrated reuse** -- when a transformer block is skipped on a
    cached denoising step, its output is not reused verbatim. Instead we return
    the last *computed* output plus the step-to-step increment (the residual
    between the two most recent computed outputs), which ICC shows is a good
    predictor of the next step's features because increments are temporally
    smooth.

  * **Channel-aware SVD** -- the predicted increment is refined by a truncated
    SVD whose right singular vectors span the channel subspace, keeping the
    dominant channel-wise variation and discarding small / noisy directions.

Auxiliary components are substituted with target-native equivalents (Mode 2):
the paper's per-block adaptive cache schedule is replaced by a simple,
configurable uniform step-level schedule ("recompute every N steps"), and the
paper's standalone benchmark/eval framework is intentionally out of scope.
"""

from __future__ import annotations

import torch


def channel_aware_svd(increment: torch.Tensor, rank: int | None = None) -> torch.Tensor:
    """Low-rank regularization of a feature increment via truncated SVD.

    The increment has shape ``(..., tokens, channels)``; each leading batch is
    treated as an independent ``(tokens, channels)`` matrix and reconstructed
    from its leading ``rank`` singular components. The right singular vectors
    span the channel subspace, so truncation keeps the dominant channel-wise
    variation and drops small directions -- the "Channel-Aware SVD" refinement
    from ICC. Computed in float32 for SVD stability, cast back on return.
    """
    if rank is None:
        # Default to roughly half rank: a bias toward principal variation.
        rank = max(1, min(increment.shape[-2], increment.shape[-1]) // 2)

    orig_dtype = increment.dtype
    x = increment.float()
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    k = max(1, min(int(rank), min(x.shape[-2], x.shape[-1])))
    # Reconstruct from the leading-k components; zero the rest of the spectrum.
    s_trunc = s[..., :k]
    low = (u[..., :k] * s_trunc.unsqueeze(-2)) @ vh[..., :k, :]
    return low.to(orig_dtype)


class IncrementCalibratedCache:
    """Per-block feature cache with increment-calibrated reuse.

    Wrap a transformer's block list and call it from the block loop. On
    "compute" denoising steps each block runs for real and we record its output
    and (once two computed outputs exist) the step-to-step increment. On "cache"
    steps a block is not recomputed: we return ``last_output`` plus the refined
    increment. This preserves the model's velocity I/O contract while skipping
    block evaluations on cached steps.

    Multi-branch sampling (e.g. classifier-free guidance) keeps separate state
    per *slot* so the conditional and unconditional streams do not contaminate
    each other; advance the shared step counter once per denoising step.
    """

    def __init__(
        self,
        num_blocks: int,
        *,
        slots: int = 1,
        compute_every: int = 2,
        svd_enabled: bool = True,
        svd_rank: int | None = None,
        enabled: bool = True,
    ):
        self.num_blocks = int(num_blocks)
        self.slots = max(1, int(slots))
        self.compute_every = max(1, int(compute_every))
        self.svd_enabled = svd_enabled
        self.svd_rank = svd_rank
        self.enabled = enabled
        self.step = 0
        self.active_slot = 0
        self.outputs: list[list[torch.Tensor | None]] = [
            [None] * self.num_blocks for _ in range(self.slots)
        ]
        self.increments: list[list[torch.Tensor | None]] = [
            [None] * self.num_blocks for _ in range(self.slots)
        ]
        # Telemetry: lets callers/tests confirm caching actually engaged.
        self.block_calls = 0  # real block forward invocations
        self.cache_hits = 0  # cached (calibrated-reuse) outputs

    def reset(self) -> None:
        """Clear all cached state and telemetry (call at the start of a run)."""
        self.step = 0
        self.active_slot = 0
        self.outputs = [[None] * self.num_blocks for _ in range(self.slots)]
        self.increments = [[None] * self.num_blocks for _ in range(self.slots)]
        self.block_calls = 0
        self.cache_hits = 0

    def compute_step(self) -> bool:
        """Whether this denoising step recomputes all blocks.

        The first step of every ``compute_every`` window recomputes, which seeds
        the cache with a real output (and, on the second compute, a real
        increment) before any prediction is made.
        """
        return self.step % self.compute_every == 0

    def __call__(self, idx, block, x: torch.Tensor, *args) -> torch.Tensor:
        if not self.enabled:
            self.block_calls += 1
            return block(x, *args)

        outputs = self.outputs[self.active_slot]
        increments = self.increments[self.active_slot]

        if self.compute_step() or outputs[idx] is None:
            out = block(x, *args)
            self.block_calls += 1
            if outputs[idx] is not None:
                increments[idx] = (out.detach() - outputs[idx]).to(out.dtype)
            outputs[idx] = out.detach()
            return out

        # Cached path: calibrated reuse of the last computed output.
        self.cache_hits += 1
        base = outputs[idx]
        increment = increments[idx]
        if increment is None:
            # Only one computed sample so far -- reuse verbatim until an
            # increment is available on the next compute step.
            return base
        return base + self._refine(increment)

    def _refine(self, increment: torch.Tensor) -> torch.Tensor:
        if not self.svd_enabled:
            return increment
        return channel_aware_svd(increment, self.svd_rank)
