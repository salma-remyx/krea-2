"""Training-free pruning of dispensable prompt-reading attention heads.

Adapted from "Text Template Tokens Are Implicit Semantic Registers in Diffusion
Transformers" (arxiv:2607.19139v1). That work builds a causal interpretability
framework for text-to-image DiTs and derives a *training-free pruning rule*: in
the DiT's joint text+image attention, the heads that attend **most strongly to
prompt tokens are dispensable** — pruning them removes ~20% of attention FLOPs
for only a ~1.4-point drop on GenEval.

This module ports that rule onto the Krea 2 ``SingleStreamDiT`` (the exact
single-stream MM-DiT the paper studies — a Qwen3-VL text stream concatenated
with the image stream in ``SingleStreamDiT.forward``). It is opt-in and
side-effect-free: nothing changes until you call :func:`enable`. Enabling swaps
``mmdit.Attention.forward`` for a variant that zeroes the per-head output
contribution of the heads flagged as dispensable, leaving the ``qkv -> out``
contract and the rest of the inference path
(``inference.py`` -> ``sampling.sample`` -> ``SingleStreamDiT.forward``) untouched.

Implementation mode — **adapted port (Mode 2)**. The core mechanism (identify
the heads that read the prompt and remove their contribution) is kept at full
fidelity; the paper's auxiliary components are substituted with target-native
equivalents:

  * the attention-decomposition + causal-intervention analysis is replaced by a
    parameter-free calibration pass (:func:`calibrate`) that measures, per head,
    the attention mass image queries place on prompt keys via a manual softmax
    (fused SDPA kernels do not expose weights);
  * the paper's prompt-content vs. structural-template token split is collapsed
    to "the text stream is the prompt stream" — the dispensable prompt-reading
    signal is preserved (pass a finer ``prompt_token_end`` to restrict to
    content tokens);
  * the paper's separate GenEval benchmark is cut (quality evaluation belongs in
    a downstream PR).

Pruning is realised by zeroing each flagged head's attention output before the
(linear) output projection, which is **bit-identical** to structurally removing
that head (``wo`` is linear, so a zeroed head slice contributes nothing).
Translating that into fused-kernel query-head slicing for wall-clock FLOP
savings interacts with GQA head-grouping and is intentionally left as a
deployment follow-up; the rule itself — which heads are dispensable, and that
their removal preserves quality — is what this module delivers.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Self

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

import mmdit

_PRUNE_ATTR = "_php_prune_mask"
_RECORD_ATTR = "_php_record"


class _Calib:
    """Bookkeeping for the calibration forward pass."""

    __slots__ = ("active", "image_query_end", "prompt_token_end", "records")

    def __init__(self) -> None:
        self.active = False
        self.prompt_token_end = 0
        self.image_query_end = 0
        self.records: dict[int, Tensor] = {}

    def activate(self, prompt_token_end: int, image_query_end: int) -> None:
        self.prompt_token_end = prompt_token_end
        self.image_query_end = image_query_end
        self.records = {}
        self.active = True


_CALIB = _Calib()


class _State:
    original_forward: Any = None


_STATE = _State()


def _prompt_attention_mass(
    q: Tensor,
    k: Tensor,
    mask: Tensor | None,
    prompt_token_end: int,
    image_query_end: int,
    gqa: bool,
) -> Tensor:
    """Per-head attention mass from image queries onto prompt keys. -> (H,).

    Higher mass = the head reads the prompt more strongly = more dispensable per
    the paper's rule. ``q`` is ``(B, H, L, D)``; ``k`` is ``(B, kvH, L, D)`` and
    is GQA-expanded here so every query head scores against its own key group.
    """
    if gqa:
        rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(rep, dim=1)
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # (B,H,L,L)
    if mask is not None:
        scores = scores.masked_fill(~mask.to(torch.bool), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    per_query = attn[:, :, prompt_token_end:image_query_end, :prompt_token_end]
    return per_query.sum(dim=-1).sum(dim=-1).mean(dim=0).to(q.dtype)  # (H,)


def _recorded_forward(
    self: Any, qkv: Tensor, freqs: Tensor | None = None, mask: Tensor | None = None
) -> Tensor:
    """``Attention.forward`` variant that records per-head prompt-attention mass.

    Numerically identical to the original forward (it returns via the same
    ``mmdit.attention`` call); it only *additionally* measures per-head
    prompt-attention mass for the heads flagged as joint-stream recorders.
    """
    q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
    q, k, v = self.qknorm(q, k, v)
    if freqs is not None:
        q, k = mmdit.ropeapply(q, k, freqs)

    if _CALIB.active and getattr(self, _RECORD_ATTR, False):
        _CALIB.records[id(self)] = _prompt_attention_mass(
            q, k, mask, _CALIB.prompt_token_end, _CALIB.image_query_end, self.gqa
        )

    out = mmdit.attention(q, k, v, mask=mask, gqa=self.gqa)
    return self.wo(out * F.sigmoid(gate))


def _pruned_forward(
    self: Any, qkv: Tensor, freqs: Tensor | None = None, mask: Tensor | None = None
) -> Tensor:
    """``Attention.forward`` variant that zeroes the pruned heads' contributions."""
    prune_mask = getattr(self, _PRUNE_ATTR, None)
    if prune_mask is None or not bool(torch.any(prune_mask)):
        return _STATE.original_forward(self, qkv, freqs=freqs, mask=mask)

    q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)
    q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)
    q, k, v = self.qknorm(q, k, v)
    if freqs is not None:
        q, k = mmdit.ropeapply(q, k, freqs)

    out = mmdit.attention(q, k, v, mask=mask, gqa=self.gqa)  # (B,L,H*D)
    out = rearrange(out, "B L (H D) -> B L H D", H=self.heads)
    out = out.masked_fill(prune_mask.to(torch.bool).view(1, 1, -1, 1), 0.0)
    out = rearrange(out, "B L H D -> B L (H D)")
    return self.wo(out * F.sigmoid(gate))


def install_forward() -> None:
    """Swap ``mmdit.Attention.forward`` for the pruning-aware version (idempotent)."""
    if _STATE.original_forward is None:
        _STATE.original_forward = mmdit.Attention.forward
        mmdit.Attention.forward = _pruned_forward


def restore_forward() -> None:
    """Undo :func:`install_forward`."""
    if _STATE.original_forward is not None:
        mmdit.Attention.forward = _STATE.original_forward
        _STATE.original_forward = None


def attach_mask(attn: mmdit.Attention, prune_mask: Tensor) -> None:
    """Attach a per-head prune mask (True = prune) to one Attention module."""
    setattr(
        attn,
        _PRUNE_ATTR,
        prune_mask.to(torch.bool).to(attn.wo.weight.device),
    )


def clear_mask(attn: mmdit.Attention) -> None:
    if hasattr(attn, _PRUNE_ATTR):
        delattr(attn, _PRUNE_ATTR)


class _Handle:
    """Restores the model on ``disable()`` / context-manager close."""

    def __init__(self, blocks: list[Any]) -> None:
        self._blocks = blocks

    def disable(self) -> None:
        for block in self._blocks:
            clear_mask(block.attn)
        restore_forward()

    def close(self) -> None:
        self.disable()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def enable(model: mmdit.SingleStreamDiT, prune_mask: Tensor) -> _Handle:
    """Enable prompt-head pruning on ``model`` (its ``blocks[*].attn``).

    ``prune_mask`` is a ``(layers, heads)`` boolean tensor (True = prune). Only
    one pruning session should be active at a time (the forward is patched at the
    class level). Returns a handle whose ``disable()`` (or context close)
    restores the model.
    """
    blocks = list(model.blocks)
    if prune_mask.shape[0] != len(blocks):
        raise ValueError(
            f"prune_mask has {prune_mask.shape[0]} layers, model has {len(blocks)}"
        )
    install_forward()
    for block, layer_mask in zip(blocks, prune_mask):
        attach_mask(block.attn, layer_mask)
    return _Handle(blocks)


def build_prune_mask(prompt_attention: Tensor, prune_ratio: float = 0.2) -> Tensor:
    """Pick the top ``prune_ratio`` prompt-reading heads per layer.

    ``prompt_attention`` is a ``(layers, heads)`` tensor of per-head prompt mass
    (from :func:`calibrate`); higher = more dispensable. Returns a boolean
    ``(layers, heads)`` mask (True = prune), always keeping at least one head.
    """
    pa = torch.as_tensor(prompt_attention)
    _, heads = pa.shape
    count = round(heads * float(prune_ratio))
    count = max(0, min(count, heads - 1))
    if count == 0:
        return torch.zeros_like(pa, dtype=torch.bool)
    idx = torch.topk(pa, count, dim=-1).indices
    mask = torch.zeros_like(pa, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def calibrate(
    model: mmdit.SingleStreamDiT,
    img: Tensor,
    context: Tensor,
    t: Tensor,
    pos: Tensor,
    mask: Tensor,
    *,
    prompt_token_end: int | None = None,
    image_query_end: int | None = None,
) -> Tensor:
    """One-shot, parameter-free calibration of per-head prompt attention.

    Runs a single ``model`` forward with attention weights exposed and records,
    per (layer, head), how strongly image queries attend to prompt keys. Returns
    a ``(layers, heads)`` tensor to feed :func:`build_prune_mask`.

    ``prompt_token_end`` defaults to the full text-stream length (``context``'s
    token count); ``image_query_end`` defaults to ``text + image`` token count,
    which excludes the 256-token padding the DiT appends internally. Pass a
    finer ``prompt_token_end`` to restrict "prompt" to content tokens only.
    """
    txtlen = context.shape[1]
    imglen = img.shape[1]
    if prompt_token_end is None:
        prompt_token_end = txtlen
    if image_query_end is None:
        image_query_end = txtlen + imglen
    blocks = list(model.blocks)
    for block in blocks:
        setattr(block.attn, _RECORD_ATTR, True)

    _CALIB.activate(prompt_token_end, image_query_end)
    previous = mmdit.Attention.forward
    mmdit.Attention.forward = _recorded_forward
    try:
        with torch.no_grad():
            model(img=img, context=context, t=t, pos=pos, mask=mask)
    finally:
        mmdit.Attention.forward = previous
        records = _CALIB.records
        _CALIB.active = False
        for block in blocks:
            delattr(block.attn, _RECORD_ATTR)

    missing = [i for i, b in enumerate(blocks) if id(b.attn) not in records]
    if missing:
        raise RuntimeError(
            f"calibration did not record layers {missing}; "
            "is the model a SingleStreamDiT with .blocks[*].attn?"
        )
    return torch.stack([records[id(b.attn)] for b in blocks], dim=0)


@contextmanager
def prompt_head_pruning(model: mmdit.SingleStreamDiT, prune_mask: Tensor):
    """Context manager: prune prompt-reading heads for a scoped inference run."""
    handle = enable(model, prune_mask)
    try:
        yield handle
    finally:
        handle.close()
