"""Training-free attention-head pruning for the K2 MM-DiT.

Implements the dispensable-head pruning rule from *Text Template Tokens Are
Implicit Semantic Registers in Diffusion Transformers* (arXiv:2607.19139):
attention heads that attend most strongly to the prompt are dispensable, so
pruning the top fraction of such heads removes a proportional share of
*attention* FLOPs at little quality cost.

Two pieces, both training-free:

* ``score_prompt_attention`` runs a single calibration forward and scores every
  (layer, head) by how much attention mass its image-query positions place on
  the prompt (text) key positions. No training, no labels.
* ``apply_head_pruning`` rewrites each ``SingleStreamBlock.attn`` forward at
  runtime so scaled-dot-product attention runs over the *kept* query heads only
  (the pruned heads' output slots are zeroed). Because attention compute scales
  with the number of query heads, pruning fraction ``f`` removes ~``f`` of the
  attention FLOPs while leaving every projection (wq/wk/wv/wo) intact.

The pruning is applied on the live ``mmdit.Attention`` instances, so no model
source has to change; ``remove()`` restores the original forwards.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from mmdit import ropeapply

__all__ = [
    "HeadPruner",
    "apply_head_pruning",
    "build_prune_mask",
    "score_prompt_attention",
]


def _image_to_text_mass(attn, qkv, freqs, mask, txtlen):
    """Return a ``(heads,)`` tensor of per-head prompt-attention mass.

    For each head: mean over batch and over valid image-query positions of the
    total attention weight that position places on the leading ``txtlen`` (text)
    key positions. Higher == the head "reads the prompt" harder == dispensable.
    """
    heads, kvheads, headdim = attn.heads, attn.kvheads, attn.headdim
    q = rearrange(attn.wq(qkv), "B L (H D) -> B H L D", H=heads)
    k = rearrange(attn.wk(qkv), "B L (H D) -> B H L D", H=kvheads)
    v = rearrange(attn.wv(qkv), "B L (H D) -> B H L D", H=kvheads)
    q, k, _ = attn.qknorm(q, k, v)
    if freqs is not None:
        q, k = ropeapply(q, k, freqs)
    if attn.gqa:  # expand kv heads to match query heads for plain QK^T
        k = k.repeat_interleave(heads // kvheads, dim=1)

    scale = headdim ** -0.5
    logits = (q.float() @ k.transpose(-1, -2).float()) * scale
    if mask is not None:
        logits = logits.masked_fill(~mask.bool(), float("-inf"))
    # Fused SDPA maps fully-masked (padding) rows to 0; emulate that so the
    # softmax over all-(-inf) rows does not contaminate the score with NaNs.
    weights = torch.nan_to_num(logits.softmax(dim=-1), nan=0.0)
    text_mass = weights[..., :txtlen].sum(dim=-1)  # (B, heads, Lq)

    if mask is not None:
        row_valid = mask[:, 0, :, 0]  # (B, Lq) validity of each query position
    else:
        row_valid = torch.ones(
            qkv.shape[0], weights.shape[-2], dtype=torch.bool, device=qkv.device
        )
    img_qw = row_valid.clone()
    img_qw[:, :txtlen] = False  # keep only image-query positions
    denom = img_qw.float().sum(dim=1).clamp(min=1.0)  # (B,)
    mass = (text_mass * img_qw[:, None, :].float()).sum(dim=-1) / denom[:, None]
    return mass.mean(dim=0)  # (heads,)


@torch.no_grad()
def score_prompt_attention(model, img, context, t, pos, mask):
    """Run one calibration forward; return ``(layers, heads)`` prompt scores.

    The text tokens are the leading ``context.shape[1]`` positions of the
    combined stream (see ``SingleStreamDiT.forward``), so the text/image split
    is known up front and does not need to be captured during the pass.
    """
    txtlen = context.shape[1]
    per_layer = [None] * len(model.blocks)
    originals = []

    for idx, block in enumerate(model.blocks):
        attn = block.attn
        orig = attn.forward
        originals.append((attn, orig))

        def _hook(qkv, freqs=None, mask=mask, _idx=idx, _attn=attn, _orig=orig):
            per_layer[_idx] = _image_to_text_mass(_attn, qkv, freqs, mask, txtlen)
            return _orig(qkv, freqs, mask)

        attn.forward = _hook

    try:
        model(img=img, context=context, t=t, pos=pos, mask=mask)
    finally:
        for attn, orig in originals:
            attn.forward = orig

    return torch.stack(per_layer, dim=0)


def build_prune_mask(scores, fraction, per_layer=False):
    """Boolean ``(layers, heads)`` mask; ``True`` == prune (dispensable) head.

    Heads with the *highest* prompt-attention score are selected (the paper's
    dispensable set). ``per_layer`` selects the top fraction within each layer;
    the default selects globally across all layers (matches the paper's single
    "remove 20% of attention FLOPs" rule). At least one head per layer is kept.
    """
    layers, heads = scores.shape
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if per_layer:
        # Top fraction within each layer, always sparing the least prompt-attending head.
        k = max(0, min(round(fraction * heads), heads - 1))
        if k:
            mask.scatter_(1, scores.topk(k, dim=1).indices, True)
    else:
        # Protect each layer's lowest-score head, then take the top fraction globally
        # so no layer is left with zero attention.
        total = layers * heads
        protected = scores.argmin(dim=1, keepdim=True)
        eligible = scores.masked_fill(
            torch.zeros_like(scores, dtype=torch.bool).scatter_(1, protected, True),
            float("-inf"),
        )
        k = max(0, min(round(fraction * total), total - layers))
        if k:
            mask.view(-1)[eligible.flatten().topk(k).indices] = True
    return mask


def _make_pruned_forward(attn, kept_idx):
    heads, kvheads = attn.heads, attn.kvheads
    rep = heads // kvheads
    kept_idx = kept_idx.to(attn.wq.weight.device)

    def forward(qkv, freqs=None, mask=None):
        q, k, v = attn.wq(qkv), attn.wk(qkv), attn.wv(qkv)
        gate = attn.gate(qkv)
        q = rearrange(q, "B L (H D) -> B H L D", H=heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=kvheads)
        q, k, v = attn.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        # Gather the kv head backing each kept query head, then run SDPA over
        # kept heads only (enable_gqa=False: q and k/v now share head count).
        qk = q[:, kept_idx]
        kv_idx = kept_idx // rep
        out = F.scaled_dot_product_attention(
            qk, k[:, kv_idx], v[:, kv_idx], attn_mask=mask, enable_gqa=False
        )
        full = torch.zeros_like(q)
        full[:, kept_idx] = out  # pruned head slots stay zero
        merged = rearrange(full, "B H L D -> B L (H D)")
        return attn.wo(merged * torch.sigmoid(gate))

    return forward


class _PruneState:
    def __init__(self, originals):
        self._originals = originals

    def remove(self):
        for attn, orig in self._originals:
            attn.forward = orig
        self._originals.clear()


def apply_head_pruning(model, prune_mask):
    """Rewrite each block's ``attn.forward`` to skip heads where mask is True.

    Returns a state whose ``remove()`` restores the original forwards.
    """
    prune_mask = prune_mask.to(torch.bool)
    originals = []
    for layer_mask, block in zip(prune_mask, model.blocks):
        if not layer_mask.any():
            continue
        kept_idx = torch.nonzero(~layer_mask, as_tuple=False).flatten()
        attn = block.attn
        originals.append((attn, attn.forward))
        attn.forward = _make_pruned_forward(attn, kept_idx)
    return _PruneState(originals)


class HeadPruner:
    """Calibrate dispensable heads on one prompt, then prune/restore at runtime."""

    def __init__(self, fraction=0.2, per_layer=False):
        self.fraction = fraction
        self.per_layer = per_layer
        self.scores = None
        self.prune_mask = None
        self._state = None

    def calibrate(self, model, img, context, t, pos, mask):
        self.scores = score_prompt_attention(model, img, context, t, pos, mask)
        self.prune_mask = build_prune_mask(self.scores, self.fraction, self.per_layer)
        return self.prune_mask

    def apply(self, model):
        if self.prune_mask is None:
            raise RuntimeError("Call calibrate(...) before apply(...).")
        self.remove()
        self._state = apply_head_pruning(model, self.prune_mask)
        return self

    def remove(self):
        if self._state is not None:
            self._state.remove()
            self._state = None

    @property
    def pruned_fraction(self):
        if self.prune_mask is None:
            return 0.0
        return float(self.prune_mask.float().mean().item())

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
