"""Training-free attention-head pruning for the Krea 2 MM-DiT.

Adapted from "Text Template Tokens Are Implicit Semantic Registers in Diffusion
Transformers" (arxiv:2607.19139). The paper's actionable result is a
training-free pruning rule: in a text-to-image DiT, the attention heads that
attend *most strongly* to the text/prompt tokens are dispensable, and disabling
them removes ~20% of attention FLOPs with only a ~1.4-point GenEval drop.

This module implements that rule for ``SingleStreamDiT``:

* :func:`calibrate_head_masks` runs a few calibration forwards, measures how much
  each joint-attention head attends to the text region of the concatenated
  ``[text, image]`` sequence, ranks heads, and writes a keep/prune mask.
* The mask is applied inside ``mmdit.Attention.forward`` via
  :func:`apply_head_mask`, which zeros the output of pruned heads.

Adaptations vs. the paper (Mode 2 — core mechanism kept, auxiliaries substituted):

* Token-span proxy. The paper separates prompt-content tokens from structural
  chat-template tokens. The inference repo exposes no per-token type map at the
  DiT boundary, so heads are scored on attention to the whole *text region* of
  the joint sequence — a parameter-free proxy for the same image-to-text
  attention mass the paper ranks heads by.
* No benchmark suite. GenEval/FID evaluation is out of scope here; this delivers
  the mask and calibration. The actual FLOP/latency win from *physically*
  removing heads requires restructuring the fused QKV projection + SDPA call
  (and re-tracing the compiled graph) and is left to a downstream kernel PR.
  Zeroing a head's output is mathematically identical to removing it, and
  :func:`pruned_head_fraction` reports the realized-FLOP target.
"""

import math
from collections.abc import Iterable, Mapping

import torch
from torch import Tensor


def iter_prunable_attentions(model) -> list:
    """Joint-attention ``Attention`` modules of a ``SingleStreamDiT``.

    These are the blocks where image tokens attend to text tokens; the paper's
    image-to-text pruning rule applies here, not to the text-only fusion blocks.
    """
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttributeError(
            "Expected a SingleStreamDiT (with a .blocks ModuleList); got an "
            "object without .blocks."
        )
    return [block.attn for block in blocks]


def set_head_mask(attn, keep_mask: Tensor) -> None:
    """Write a ``(heads,)`` keep-mask (1=keep, 0=prune) onto an Attention.

    Pruning is only marked active when at least one head is pruned, so the
    default all-keep path stays zero-overhead.
    """
    keep_mask = keep_mask.to(device=attn.head_mask.device, dtype=attn.head_mask.dtype)
    attn.head_mask.copy_(keep_mask)
    attn._head_pruning_active = bool((keep_mask == 0).any())


def clear_head_masks(model) -> None:
    """Reset every prunable Attention to all-keep (no pruning)."""
    for attn in iter_prunable_attentions(model):
        set_head_mask(attn, torch.ones_like(attn.head_mask))
        attn._head_pruning_active = False


def pruned_head_fraction(model) -> float:
    """Fraction of prunable attention heads currently disabled, in [0, 1]."""
    total = 0
    pruned = 0
    for attn in iter_prunable_attentions(model):
        total += attn.head_mask.numel()
        pruned += int((attn.head_mask == 0).sum().item())
    return pruned / total if total else 0.0


def apply_head_mask(attn_out: Tensor, head_mask: Tensor, heads: int) -> Tensor:
    """Zero the contribution of pruned heads in an attention output.

    ``attn_out`` is ``(B, L, heads * headdim)``; ``head_mask`` is ``(heads,)``.
    Returns the same shape with pruned head slices zeroed. Zeroing a head's
    output is exactly equivalent to removing the head: it contributes nothing to
    the residual stream downstream of ``wo``.
    """
    batch, length, hidden = attn_out.shape
    out = attn_out.reshape(batch, length, heads, hidden // heads)
    out = out * head_mask.to(out.dtype).view(1, 1, heads, 1)
    return out.reshape(batch, length, hidden)


def head_text_attention_score(
    q: Tensor, k: Tensor, mask: Tensor | None, txtlen: int
) -> Tensor:
    """Per-head average attention the image tokens pay to the text tokens.

    ``q`` is ``(B, Hq, L, dh)``, ``k`` is ``(B, Hkv, L, dh)`` (GQA supported via
    head-group replication). Returns a ``(Hq,)`` float32 vector; larger means the
    head attends more strongly to text and is therefore more dispensable per the
    paper's rule.
    """
    hq, hkv = q.shape[1], k.shape[1]
    dh = q.shape[-1]
    groups = hq // hkv
    kfull = k.repeat_interleave(groups, dim=1) if groups > 1 else k
    scores = torch.matmul(q.float(), kfull.float().transpose(-2, -1)) / math.sqrt(dh)
    if mask is not None:
        scores = scores.masked_fill(~mask.bool(), float("-inf"))
    probs = scores.softmax(dim=-1)
    # Padding query rows are fully masked -> softmax yields NaN; they carry no
    # signal and are excluded from the average below, so clean them to zero.
    probs = torch.nan_to_num(probs, nan=0.0)

    # Attention each query position places on the text key region (first txtlen
    # keys), then averaged over the (valid) image query positions.
    text_mass = probs[..., :txtlen].sum(dim=-1)  # (B, Hq, L)
    length = q.shape[2]
    img_query = torch.arange(length, device=q.device) >= txtlen  # text is [:txtlen]
    weight = img_query.view(1, 1, length).to(text_mass.dtype)
    if mask is not None:
        valid = mask.bool()[:, 0, :, 0]  # (B, L) key-padding validity
        weight = weight * valid[:, None, :].to(text_mass.dtype)
    return (text_mass * weight).sum(dim=(0, 2)) / weight.sum().clamp(min=1)


def _select_prune_masks(
    layer_scores: list[Tensor], prune_ratio: float, scope: str
) -> list[Tensor]:
    """Pick which heads to prune from per-layer scores; return per-layer keep masks."""
    masks: list[Tensor] = []
    if scope == "global":
        all_scores = torch.cat(layer_scores)
        total = all_scores.numel()
        n_prune = min(round(prune_ratio * total), total)
        top = torch.zeros(total)
        if n_prune > 0:
            top[all_scores.topk(n_prune).indices] = 1.0  # 1 at high-score (dispensable)
        offset = 0
        for scores in layer_scores:
            masks.append(1.0 - top[offset : offset + scores.numel()])  # 1=keep, 0=prune
            offset += scores.numel()
    else:  # per_layer
        for scores in layer_scores:
            keep = torch.ones(scores.numel())
            n_prune = min(round(prune_ratio * scores.numel()), scores.numel())
            if n_prune > 0:
                keep[scores.topk(n_prune).indices] = 0.0
            masks.append(keep)
    return masks


def calibrate_head_masks(
    model,
    calibration_inputs: Iterable[Mapping[str, Tensor]],
    prune_ratio: float = 0.2,
    *,
    txtlen: int | None = None,
    scope: str = "global",
) -> dict:
    """Calibrate training-free head-prune masks from calibration forwards.

    Runs one ``model(**inp)`` forward per item in ``calibration_inputs`` (each a
    kwargs dict for ``SingleStreamDiT.forward`` — ``img``/``context``/``t``/
    ``pos``/``mask``), records per-head image->text attention, ranks heads, and
    writes a keep/prune mask onto every prunable Attention.

    ``prune_ratio`` is the fraction of heads to disable. ``scope="global"`` ranks
    across all layers (paper-faithful — "heads that attend most strongly" form a
    single global set); ``scope="per_layer"`` ranks within each layer (safer, it
    cannot empty a layer). ``txtlen`` is auto-detected from the first input's
    ``context`` if omitted and is assumed constant across the calibration set
    (the repo's Qwen3-VL encoder pads to ``max_length``).

    Returns a dict with per-layer scores and the realized prune fraction.
    """
    if not 0.0 <= prune_ratio <= 1.0:
        raise ValueError(f"prune_ratio must be in [0, 1], got {prune_ratio}")
    if scope not in ("global", "per_layer"):
        raise ValueError(f"scope must be 'global' or 'per_layer', got {scope!r}")

    inputs_list = list(calibration_inputs)
    if not inputs_list:
        raise ValueError("calibration_inputs is empty; provide at least one forward")
    if txtlen is None:
        txtlen = int(inputs_list[0]["context"].shape[1])
    observed = {int(inp["context"].shape[1]) for inp in inputs_list}
    if observed != {txtlen}:
        raise ValueError(
            "calibration forwards have varying text lengths "
            f"({sorted(observed)}); head_text_attention_score expects a fixed "
            "text region. Pad your encoder to a constant length or calibrate one "
            "prompt at a time."
        )

    attns = iter_prunable_attentions(model)
    for attn in attns:
        attn._capture = []
        attn._head_pruning_active = False  # measure the un-pruned model
    try:
        with torch.no_grad():
            for inp in inputs_list:
                model(**inp)
    except Exception:
        for attn in attns:
            attn._capture = None
        raise

    layer_scores: list[Tensor] = []
    for attn in attns:
        per_step = [
            head_text_attention_score(qq, kk, mm, txtlen)
            for (qq, kk, mm) in attn._capture
        ]
        attn._capture = None  # free the captured tensors
        if not per_step:
            raise RuntimeError("an attention layer captured no forwards")
        layer_scores.append(torch.stack(per_step).mean(dim=0))

    masks = _select_prune_masks(layer_scores, prune_ratio, scope)
    for attn, keep in zip(attns, masks):
        set_head_mask(attn, keep)

    total = sum(scores.numel() for scores in layer_scores)
    pruned = int(sum(int((keep == 0).sum().item()) for keep in masks))
    return {
        "scope": scope,
        "requested_ratio": prune_ratio,
        "pruned_heads": pruned,
        "total_heads": total,
        "realized_ratio": pruned / total if total else 0.0,
        "per_layer_scores": layer_scores,
        "per_layer_masks": masks,
    }
