"""Training-free inference acceleration for the K2 ``SingleStreamDiT``.

Adapted from *Accelerating Diffusion Transformers with Dual Feature Caching*
(DuCa; Zou et al., arXiv:2412.18911). DuCa caches the post-block feature of the
DiT in a *fresh* denoising step and reuses it to skip block compute in later
steps, alternating two strategies inside a fixed-length cycle:

  * ``fresh``        - run every block; cache the resulting feature.
  * ``conservative`` - run every block, then correct a **randomly selected**
                       subset of tokens of the cache toward the fresh output
                       (DuCa's finding: random selection beats importance-based
                       selection because token *diversity* matters more than
                       token *importance*, and random selection keeps
                       FlashAttention compatibility).
  * ``aggressive``   - reuse the cached feature verbatim and skip the whole
                       block loop (the bulk of the speedup).

The two strategies alternate ``fresh -> conservative -> aggressive ->
conservative -> aggressive`` over a cycle of ``cycle_length`` steps, then
repeat. Conservative steps realign the cache so the aggressively-reused feature
does not drift, which is what lets aggressive skipping apply to more steps than
either strategy alone.

The cache is keyed per conditioning stream (by the identity of the ``context``
tensor) so the cond / uncond branches of classifier-free guidance keep
independent caches while staying phase-aligned at every denoising timestep.

This is opt-in and inference-only: enable it on a built, weight-loaded model
with :func:`apply_dual_feature_caching` and restore the original behaviour with
:func:`remove_dual_feature_caching`. The model's
``(img, context, t, pos, mask) -> velocity`` contract is unchanged.

Adaptation notes (Mode 2 port): the paper computes the selected token subset
*through* every layer in the conservative step (token-selective attention) to
save FLOPs. That selective attention is incompatible with the compiled fused
attention kernels used here, so conservative steps run the full block and only
*correct a random token subset of the cache* instead. Acceleration therefore
comes from the aggressive steps skipping the block loop; the conservative steps
buy quality by realigning the cache. DuCa's separate benchmark suite
(FLUX/PixArt/OpenSora/DiT quality metrics) is out of scope - evaluation belongs
in a downstream PR.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "apply_dual_feature_caching",
    "remove_dual_feature_caching",
    "is_dual_feature_caching_enabled",
]


# --- transient block stand-ins yielded by the cache proxy --------------------
# These are not nn.Modules (they are never registered): they live only for the
# duration of one forward's block loop and write the refreshed cache back to the
# owning proxy. The host loop calls every block as ``block(x, vec, freqs, mask)``.


class _ReplayBlock:
    """Aggressive step: skip block compute and return the cached feature."""

    def __init__(self, cached: Tensor):
        self._cached = cached

    def __call__(
        self, x: Tensor, vec: Tensor, freqs: Tensor, mask: Tensor | None = None
    ) -> Tensor:  # noqa: D401
        return self._cached


class _RecordBlock:
    """Fresh step's last block: compute normally and stash the output as cache."""

    def __init__(self, block: nn.Module, owner: "_DualCacheBlocks"):
        self._block = block
        self._owner = owner

    def __call__(
        self, x: Tensor, vec: Tensor, freqs: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        out = self._block(x, vec, freqs, mask)
        self._owner.new_cache = out
        return out


class _CorrectBlock:
    """Conservative step's last block: correct a random token subset of the cache.

    A fresh ``cache_ratio`` fraction of tokens is kept from the (stale) cache and
    the rest is taken from the freshly computed output, then the blended result is
    stored back as the cache. The random subset is resampled every conservative
    step, so different tokens are realigned over time (diversity > importance).
    """

    def __init__(
        self,
        block: nn.Module,
        cache: Tensor,
        cache_ratio: float,
        owner: "_DualCacheBlocks",
    ):
        self._block = block
        self._cache = cache
        self._cache_ratio = cache_ratio
        self._owner = owner

    def __call__(
        self, x: Tensor, vec: Tensor, freqs: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        out = self._block(x, vec, freqs, mask)
        if self._cache.shape == out.shape:
            length = out.shape[1]
            keep = (torch.rand(length, device=out.device) < self._cache_ratio).view(
                1, length, 1
            )
            out = torch.where(keep, self._cache, out)
        self._owner.new_cache = out
        return out


class _DualCacheBlocks(nn.Module):
    """Drop-in replacement for ``SingleStreamDiT.blocks`` that drives DuCa.

    The host forward loop is ``for block in self.blocks: combined = block(...)``.
    We intercept iteration to inject the cached strategy for the current step:

      * aggressive (cache present) -> yield a single replay block, skip compute;
      * fresh                      -> yield every real block, record the final
                                       output as the cache;
      * conservative (cache present)-> yield every real block, but the final one
                                       corrects a random token subset of the cache.

    All real blocks are kept under ``_real`` so the parameter tree, ``len()``,
    indexing, ``.to()`` and ``state_dict`` keep working.
    """

    def __init__(
        self,
        blocks: nn.ModuleList | list[nn.Module],
        *,
        cycle_length: int,
        cache_ratio: float,
    ):
        super().__init__()
        self._real = nn.ModuleList(blocks)
        self.cycle_length = cycle_length
        self.cache_ratio = cache_ratio
        self.phase = "fresh"
        self.cache: Tensor | None = None
        self.new_cache: Tensor | None = None

    def configure(self, phase: str, cache: Tensor | None) -> None:
        self.phase = phase
        self.cache = cache

    def take_new_cache(self) -> Tensor | None:
        cache = self.new_cache
        self.new_cache = None
        return cache

    def __iter__(self):
        self.new_cache = None
        phase = self.phase
        cache = self.cache
        if phase == "aggressive" and cache is not None:
            yield _ReplayBlock(cache)
            self.new_cache = cache
            return
        last = len(self._real) - 1
        for index, block in enumerate(self._real):
            if index != last:
                yield block
            elif phase == "conservative" and cache is not None:
                yield _CorrectBlock(block, cache, self.cache_ratio, self)
            else:  # fresh, or conservative before the first fresh step
                yield _RecordBlock(block, self)

    def __len__(self) -> int:
        return len(self._real)

    def __getitem__(self, index: int) -> nn.Module:
        return self._real[index]


# --- public API --------------------------------------------------------------


def _phase_at(step: int, cycle_length: int) -> str:
    """DuCa phase for ``step`` (1-indexed): fresh, then alternating cons/aggr.

    Within a cycle of ``cycle_length`` steps the layout is
    ``fresh, conservative, aggressive, conservative, aggressive, ...`` -
    conservative on odd positions, aggressive on even positions, matching the
    paper's arrangement.
    """
    index = (step - 1) % cycle_length
    if index == 0:
        return "fresh"
    return "conservative" if index % 2 == 1 else "aggressive"


def apply_dual_feature_caching(
    model: nn.Module,
    *,
    cycle_length: int = 5,
    cache_ratio: float = 0.9,
) -> nn.Module:
    """Enable DuCa dual feature caching on a built ``SingleStreamDiT``.

    Replaces ``model.blocks`` with a caching proxy and ``model.forward`` with a
    wrapper that drives the fresh/conservative/aggressive cycle. Apply this
    *after* loading weights (e.g. right after ``_pipeline`` builds the model in
    ``inference.py``); the original behaviour is restored with
    :func:`remove_dual_feature_caching`.

    Args:
        model: the ``SingleStreamDiT`` to accelerate.
        cycle_length: denoising steps per caching cycle (DuCa default 5).
        cache_ratio: fraction of tokens kept from the cache in each conservative
            step (DuCa default 0.9); the rest are recomputed.

    Returns:
        The same ``model`` (patched in place), for chaining.
    """
    if is_dual_feature_caching_enabled(model):
        return model
    if cycle_length < 1:
        raise ValueError("cycle_length must be >= 1")
    if not 0.0 <= cache_ratio <= 1.0:
        raise ValueError("cache_ratio must be in [0, 1]")

    original_forward: Callable = model.forward  # bound to the unpatched class forward
    proxy = _DualCacheBlocks(
        model.blocks, cycle_length=cycle_length, cache_ratio=cache_ratio
    )
    streams: dict[int, dict] = {}

    def duca_forward(img, context, t, pos, mask=None):  # type: ignore[no-untyped-def]
        key = id(context)
        state = streams.get(key)
        step = (state["step"] + 1) if state is not None else 1
        proxy.configure(
            _phase_at(step, cycle_length), state["cache"] if state else None
        )
        output = original_forward(img=img, context=context, t=t, pos=pos, mask=mask)
        streams[key] = {"step": step, "cache": proxy.take_new_cache()}
        return output

    model._duca_enabled = True
    model._duca_proxy = proxy
    model._duca_streams = streams
    model._duca_original_forward = original_forward
    model.blocks = proxy
    # Plain function on the instance: nn.Module.__call__ reads self.forward and
    # invokes it without binding, so the closure (which captures proxy/streams)
    # is called directly with the forward kwargs.
    model.forward = duca_forward
    return model


def remove_dual_feature_caching(model: nn.Module) -> nn.Module:
    """Undo :func:`apply_dual_feature_caching`, restoring the original blocks/forward."""
    if not is_dual_feature_caching_enabled(model):
        return model
    proxy: _DualCacheBlocks = model._duca_proxy
    model.blocks = proxy._real
    del model.forward  # fall back to the class-level (unpatched) forward
    for attr in (
        "_duca_enabled",
        "_duca_proxy",
        "_duca_streams",
        "_duca_original_forward",
    ):
        delattr(model, attr)
    return model


def is_dual_feature_caching_enabled(model: nn.Module) -> bool:
    """Whether DuCa caching is currently active on ``model``."""
    return bool(getattr(model, "_duca_enabled", False))
