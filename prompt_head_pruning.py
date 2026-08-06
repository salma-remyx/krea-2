"""Training-free prompt-head pruning for the Krea 2 MM-DiT.

Adapted from "Text Template Tokens Are Implicit Semantic Registers in Diffusion
Transformers" (arXiv:2607.19139), which introduces a causal-interpretability
finding and a *training-free* pruning rule for text-to-image DiTs:

    In the joint (text+image) attention, the heads that attend *most strongly*
    to prompt tokens are causally dispensable. Pruning them removes a large
    fraction of attention FLOPs for a small quality drop.

This module delivers that rule as an opt-in, runtime patch over ``mmdit.Attention``
- it requires no edits to the model source, leaves every weight tensor untouched
(strict checkpoint loading stays valid), and is applied/removed by calling a
function. The existing ``SingleStreamBlock.forward`` invokes the patched attention
unchanged, so the capability is wired through the repo's real forward path.

Implementation notes
--------------------
* The Krea 2 DiT uses grouped-query attention (``heads=48, kvheads=12`` ->
  4 query heads per kv group). SDPA with ``enable_gqa`` requires the q:kv ratio
  to stay an integer, so we prune at the **kv-group** granularity: dropping a
  kv group also drops its 4 query heads, keeping the ratio at 4:1. A group is
  dispensable when its query heads carry the most prompt-attention mass.
* Pruning slices q/k/v to the kept groups, runs SDPA on the reduced set
  (genuine attention-FLOP reduction), then scatters the result back into a
  full ``(B, heads, L, D)`` tensor (zeros for pruned groups) so the downstream
  ``wo`` projection and weight shapes are unchanged.
* The pruned SDPA call is backend-agnostic (no CUDNN pin) so it runs on CPU and
  lets SDPA auto-select a fused kernel on GPU. Numerically this is equivalent to
  the repo's pinned path within fp tolerance; pruning changes the output by
  construction regardless of backend.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

import mmdit
from mmdit import ropeapply

_KEPT_ATTR = "_prune_kept_groups"  # bool (kvheads,) tensor set on patched modules
_ORIG_ATTR = "_prune_orig_forward"  # stash of the bound forward we replaced


def joint_attention_modules(model: torch.nn.Module) -> list[mmdit.Attention]:
    """Joint (text+image) attention modules - the single-stream DiT blocks.

    The text-fusion blocks attend over text only, so the paper's joint-attention
    rule does not apply to them and they are left untouched.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        return []
    return [
        b.attn for b in blocks if isinstance(getattr(b, "attn", None), mmdit.Attention)
    ]


def prompt_attention_mass(
    q: Tensor, k: Tensor, text_len: int, scale: float | None = None
) -> Tensor:
    """Per-head attention mass that image queries place on prompt (text) keys.

    Args:
        q: ``(B, H, L, D)`` query states (post qknorm/rope) over the joint seq.
        k: ``(B, Hkv, L, D)`` key states.
        text_len: number of leading text tokens in the joint sequence.
        scale: optional softmax scale; defaults to ``1/sqrt(D)``.

    Returns:
        ``(H,)`` tensor - higher means the head reads more from the prompt.
        Computed from image queries only (rows ``>= text_len``), averaged over
        batch and image positions, so it isolates image->text reading.
    """
    if text_len <= 0 or text_len >= q.shape[2]:
        raise ValueError("text_len must be within (0, seq_len)")
    group = q.shape[1] // k.shape[1]
    if q.shape[1] != k.shape[1]:
        # Expand kv heads to query heads using the same contiguous grouping SDPA
        # applies under enable_gqa, so each query head scores against its group.
        k = k.repeat_interleave(group, dim=1)
    sc = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
    logits = (q * sc) @ k.transpose(-1, -2)  # (B, H, L, L)
    img = logits[:, :, text_len:, :]  # image queries only
    attn = img.softmax(dim=-1)
    text_mass = attn[..., :text_len].sum(dim=-1)  # (B, H, L_img)
    return text_mass.mean(dim=(0, 2))


def select_pruned_groups(
    mass_per_head: Tensor, heads: int, kvheads: int, ratio: float
) -> Tensor:
    """Pick which kv groups to KEEP under the paper's dispensable-head rule.

    Groups whose query heads carry the most prompt-attention mass are the most
    dispensable and are pruned. ``ratio`` is the fraction of kv groups to drop.

    Returns:
        ``(kvheads,)`` bool tensor - ``True`` means keep the group. Always keeps
        at least one group.
    """
    if heads % kvheads != 0:
        raise ValueError(f"heads ({heads}) must be a multiple of kvheads ({kvheads})")
    group = heads // kvheads
    n_prune = round(ratio * kvheads)
    n_prune = min(max(n_prune, 0), kvheads - 1)
    kept = torch.ones(kvheads, dtype=torch.bool)
    if n_prune == 0:
        return kept
    group_mass = mass_per_head.reshape(kvheads, group).mean(dim=1)  # (kvheads,)
    dispensable = torch.topk(group_mass, n_prune).indices
    kept[dispensable] = False
    return kept


def _sdpa(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, gqa: bool) -> Tensor:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=gqa)


def _pruned_forward(
    self: mmdit.Attention,
    qkv: Tensor,
    freqs: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
    q, k, v = self.qknorm(q, k, v)
    if freqs is not None:
        q, k = ropeapply(q, k, freqs)

    kept = getattr(self, _KEPT_ATTR, None)  # bool (kvheads,) or None
    n_kept = int(kept.sum()) if kept is not None else self.kvheads
    if kept is not None and n_kept < self.kvheads:
        group = self.heads // self.kvheads
        kept_q = kept.repeat_interleave(group)  # (heads,) q-head mask
        q2, k2, v2 = q[:, kept_q], k[:, kept], v[:, kept]
        x = _sdpa(q2, k2, v2, mask, self.gqa)  # (B, n_kept*group, L, D)
        full = q.new_zeros(q.shape[0], self.heads, x.shape[2], x.shape[3])
        full[:, kept_q] = x  # zero contribution from pruned groups
        out = rearrange(full, "B H L D -> B L (H D)")
    else:
        out = mmdit.attention(q, k, v, mask=mask, gqa=self.gqa)
    return self.wo(out * F.sigmoid(gate))


def is_patched(module: mmdit.Attention) -> bool:
    return getattr(module, _KEPT_ATTR, None) is not None


def apply_kept_mask(module: mmdit.Attention, kept_groups: Tensor) -> None:
    """Patch a single joint ``Attention`` module to keep only ``kept_groups``.

    ``kept_groups`` is a ``(kvheads,)`` bool tensor. Re-calling replaces the
    previous mask. Weight tensors and the module's I/O shape are unchanged.
    """
    if kept_groups.shape != (module.kvheads,):
        raise ValueError(
            f"kept_groups must have shape ({module.kvheads},), got {tuple(kept_groups.shape)}"
        )
    kept_groups = kept_groups.to(torch.bool)
    if not is_patched(module):
        setattr(module, _ORIG_ATTR, module.forward)
        module.forward = _pruned_forward.__get__(module, type(module))  # type: ignore[method-assign]
    setattr(module, _KEPT_ATTR, kept_groups)


def apply_prompt_head_pruning(
    model: torch.nn.Module, kept_per_module: dict[mmdit.Attention, Tensor]
) -> None:
    """Apply a per-module kept-mask to every joint attention module in ``model``."""
    for module, kept in kept_per_module.items():
        apply_kept_mask(module, kept)


def remove_prompt_head_pruning(model: torch.nn.Module) -> None:
    """Restore original attention behavior on all patched modules of ``model``."""
    targets = joint_attention_modules(model)
    for module in targets:
        if is_patched(module):
            orig = getattr(module, _ORIG_ATTR)
            delattr(module, _KEPT_ATTR)
            module.forward = orig  # type: ignore[method-assign]
            delattr(module, _ORIG_ATTR)


def _make_capture_hook(store: list[tuple[Tensor, Tensor]]) -> Callable[..., None]:
    def hook(module: mmdit.Attention, args: tuple[Any, ...], output: Tensor) -> None:
        (qkv, freqs, _mask) = (args + (None, None, None))[:3]
        q = rearrange(module.wq(qkv), "B L (H D) -> B H L D", H=module.heads)
        k = rearrange(module.wk(qkv), "B L (H D) -> B H L D", H=module.kvheads)
        q, k, _ = module.qknorm(q, k, module.wv(qkv))
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        store.append((q.detach(), k.detach()))

    return hook


def calibrate_prompt_attention(
    model: torch.nn.Module, run_forward: Callable[[], Any], text_len: int
) -> dict[mmdit.Attention, Tensor]:
    """Measure per-head prompt-attention mass for each joint attention module.

    ``run_forward`` should run one or more representative denoising forwards on
    the model (e.g. a handful of prompts/timesteps) with the modules in eval
    mode and no grad. Returns the mean prompt-attention mass per query head.
    """
    modules = joint_attention_modules(model)
    stores: dict[mmdit.Attention, list[tuple[Tensor, Tensor]]] = {
        m: [] for m in modules
    }
    handles = [m.register_forward_hook(_make_capture_hook(stores[m])) for m in modules]
    try:
        with torch.no_grad():
            run_forward()
    finally:
        for h in handles:
            h.remove()
    masses: dict[mmdit.Attention, Tensor] = {}
    for module, samples in stores.items():
        per_pass = torch.stack(
            [prompt_attention_mass(q, k, text_len) for q, k in samples]
        )
        masses[module] = per_pass.mean(dim=0)
    return masses


def prune_prompt_heads(
    model: torch.nn.Module,
    ratio: float,
    run_forward: Callable[[], Any],
    text_len: int,
) -> dict[mmdit.Attention, Tensor]:
    """Calibrate prompt attention, then prune the top-``ratio`` dispensable groups.

    Convenience wrapper: ``calibrate_prompt_attention`` + ``select_pruned_groups``
    + ``apply_prompt_head_pruning``. Returns the applied kept-masks per module.
    """
    masses = calibrate_prompt_attention(model, run_forward, text_len)
    kept: dict[mmdit.Attention, Tensor] = {}
    for module, mass in masses.items():
        mask = select_pruned_groups(mass, module.heads, module.kvheads, ratio)
        kept[module] = mask
        apply_kept_mask(module, mask)
    return kept


@contextlib.contextmanager
def prompt_head_pruning(
    model: torch.nn.Module,
    ratio: float,
    run_forward: Callable[[], Any],
    text_len: int,
) -> Iterator[dict[mmdit.Attention, Tensor]]:
    """Apply prompt-head pruning for a block, then always restore the model."""
    kept = prune_prompt_heads(model, ratio, run_forward, text_len)
    try:
        yield kept
    finally:
        remove_prompt_head_pruning(model)
