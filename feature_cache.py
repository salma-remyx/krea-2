"""Dual feature caching for the Krea 2 MM-DiT.

Wraps a ``SingleStreamDiT`` so that selected transformer-block outputs are
cached across denoising timesteps and reused to skip computation on later
steps. Two tiers are cached:

* **static**  - shallow-block outputs frozen from the first step (these
  features drift slowly across adjacent timesteps);
* **dynamic** - deeper-block outputs refreshed on a rolling cadence.

With both tiers disabled the wrapper leaves the model untouched and is a
zero-overhead, bit-exact pass-through.

Adapted from "Accelerating Diffusion Transformers with Dual Feature Caching"
(DuCa). The dual static+dynamic caching mechanism is preserved; the
block-selection schedule and the dynamic refresh cadence are simplified
target-native heuristics (configurable block counts, every-other-step
refresh) rather than the paper's fixed profile, and no benchmark/quality
suite is reproduced here.
"""

import torch.nn as nn

__all__ = ["DualCacheController", "CachingBlock", "DualCacheDiT"]


class DualCacheController:
    """Per-branch cache state and per-step caching policy for the block loop.

    Blocks are partitioned into three contiguous tiers:

    * ``static``  - ``[0, n_static)``: output stored on step 0, reused on every
      later step.
    * ``dynamic`` - ``[n_static, n_static + n_dynamic)``: output stored on step
      0 and on every even step, reused on odd steps (rolling refresh).
    * ``compute`` - the remainder: always recomputed.

    State is kept per *branch* (keyed by an opaque id) so that the two
    classifier-free-guidance forwards (conditional / unconditional) maintain
    independent caches and step counters. With no branch active, ``policy``
    returns ``"compute"`` for every block, so an inactive controller is
    invisible to the wrapped model.
    """

    def __init__(self, n_blocks, n_static=0, n_dynamic=0):
        if n_static < 0 or n_dynamic < 0:
            raise ValueError("n_static and n_dynamic must be non-negative")
        if n_static + n_dynamic > n_blocks:
            raise ValueError(
                f"n_static + n_dynamic ({n_static + n_dynamic}) "
                f"exceeds n_blocks ({n_blocks})"
            )
        self.n_blocks = n_blocks
        self.n_static = n_static
        self.n_dynamic = n_dynamic
        self._branches = {}
        self._current = None
        self.skips = 0  # number of block evaluations skipped (diagnostic)

    def enter_branch(self, key):
        """Advance one step for ``key``, creating fresh state on first use."""
        state = self._branches.get(key)
        if state is None:
            state = {"step": 0, "cache": [None] * self.n_blocks}
            self._branches[key] = state
        else:
            state["step"] += 1
        self._current = state

    def deactivate(self):
        """Drop the active branch so cached features are never reused stale.

        Called after each wrapped forward so that, if the underlying model is
        ever invoked directly (bypassing ``DualCacheDiT``), every block simply
        recomputes instead of reading a stale cache.
        """
        self._current = None

    def reset(self):
        """Clear all branch state (e.g. between independent generations)."""
        self._branches.clear()
        self._current = None
        self.skips = 0

    def _tier(self, index):
        if index < self.n_static:
            return "static"
        if index < self.n_static + self.n_dynamic:
            return "dynamic"
        return "compute"

    def policy(self, index):
        """Return ``"store"``, ``"reuse"`` or ``"compute"`` for a block."""
        if self._current is None:
            return "compute"
        step = self._current["step"]
        tier = self._tier(index)
        if tier == "compute":
            return "compute"
        if tier == "static":
            return "store" if step == 0 else "reuse"
        # dynamic: refresh on step 0 and on even steps, reuse on odd steps
        return "store" if step % 2 == 0 else "reuse"

    def get(self, index):
        return self._current["cache"][index]

    def set(self, index, value):
        self._current["cache"][index] = value


class CachingBlock(nn.Module):
    """Wraps a transformer block, serving a cached output when policy says reuse."""

    def __init__(self, block, index, controller):
        super().__init__()
        self.block = block
        self.index = index
        self.controller = controller

    def forward(self, x, vec, freqs, mask=None):
        decision = self.controller.policy(self.index)
        if decision == "reuse":
            cached = self.controller.get(self.index)
            if cached is not None:
                self.controller.skips += 1
                return cached
        out = self.block(x, vec, freqs, mask)
        if decision == "store":
            self.controller.set(self.index, out)
        return out


class DualCacheDiT(nn.Module):
    """Drop-in wrapper applying dual feature caching to a ``SingleStreamDiT``.

    Exposes the same forward contract ``(img, context, t, pos, mask) -> velocity``
    as the wrapped model, so ``sampling.sample`` can call it unchanged. With
    ``n_static == 0`` and ``n_dynamic == 0`` the wrapped model is left
    untouched (exact, zero-overhead equivalence with the base model).
    """

    def __init__(self, dit, n_static=0, n_dynamic=0):
        super().__init__()
        self.dit = dit
        # sampling.sample reads ``model.config.patch``; mirror it for the drop-in.
        self.config = dit.config
        self.controller = DualCacheController(len(dit.blocks), n_static, n_dynamic)
        if n_static or n_dynamic:
            # Install caching hooks over the model's transformer blocks. While
            # no branch is active each CachingBlock forwards straight through to
            # the original block, so the wrapped model still behaves identically
            # if it is ever invoked outside this wrapper.
            dit.blocks = nn.ModuleList(
                [
                    CachingBlock(block, i, self.controller)
                    for i, block in enumerate(dit.blocks)
                ]
            )

    def forward(self, img, context, t, pos, mask=None):
        # Key the cache on the context tensor identity: sample() reuses the same
        # context across all steps of a branch but passes distinct tensors for
        # the conditional / unconditional CFG branches, so each gets its own
        # cache without any explicit branch plumbing.
        self.controller.enter_branch(id(context))
        try:
            return self.dit(img, context, t, pos, mask)
        finally:
            self.controller.deactivate()
