"""Opt-in block-sparse attention with a retained-mass threshold (LoSA-style).

Training-free sparse attention for the K2 MMDiT. After ``warmup`` dense
denoise steps (the paper constructs at t0=3, after warm-up steps 0-2, so
the measured masses reflect a settled trajectory rather than pure noise),
a single profiling step runs the dense path AND measures exact per-(head,
query-block) attention masses, then greedily keeps the smallest key/value
block set whose cumulative mass meets a fixed *retained-mass* threshold
(default 0.99) and freezes it. Every later step reuses the frozen block
indices. Near-lossless by construction: the threshold fixes fidelity
rather than a sparsity ratio, so the high-mass support is always retained.

Adapted from "LoSA: Near-Lossless Sparse Attention for Training-Free
Video Diffusion Acceleration" (arXiv:2608.12032). The paper validates the
measure-then-freeze retained-mass mechanism on *video* diffusion
transformers; image diffusion is never discussed there, so the
near-lossless claim for Krea 2's image tokens is the open empirical
question (the documented video->image transfer caveat). The dense cuDNN
path is intentionally retained as both the profiling step and the
correctness fallback.

Mode-2 substitutions vs. the paper:
  * The paper's fused block-sparse kernel is replaced by a gather-based
    path that calls ``scaled_dot_product_attention`` per query block, so no
    custom kernel is required. Selection is batched to the per-query-block
    max kept-set (a superset for heads needing fewer blocks), which trades
    a little efficiency for kernel friendliness while staying a strict
    superset of the paper's per-head set (so fidelity is preserved).
  * Frozen indices are kept separately for the conditional and
    unconditional CFG branches, as the paper specifies; the branch is
    identified by call parity within a denoise step (the sampler always
    runs the conditional branch first), so no sampler/model signature
    changes are needed.
  * Feature-caching composition (the paper's 3.2x result) and the video
    benchmark suite are intentionally out of scope for this slice;
    evaluation belongs in a downstream PR.

Wired in by ``mmdit.Attention`` and driven around the denoise loop by
``sampling.sample`` via the ``block_sparse_attention`` context manager.
Off by default; the original dense path is untouched when inactive.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel


class _SparseState:
    """Module-global measure-then-freeze state (mirrors ``sdpa_kernel``)."""

    def __init__(self) -> None:
        self.active = False
        self.threshold = 0.99
        self.block = 64
        self.warmup = 3
        # Denoise-step counter, advanced by ``advance_step`` once per Euler
        # step; steps below ``warmup`` stay fully dense.
        self.step = 0
        # id(Attention module) -> calls seen this step, used to tell the
        # conditional (first) and unconditional (second) CFG branches apart.
        self.branch_counts: dict[int, int] = {}
        # (id(Attention module), cfg branch) -> frozen selection
        # (None = keep this module dense).
        self.selections: dict[tuple[int, int], dict | None] = {}

    def clear(self) -> None:
        self.selections.clear()
        self.branch_counts.clear()
        self.step = 0


sparse_state = _SparseState()


@contextmanager
def block_sparse_attention(threshold: float = 0.99, block: int = 64, warmup: int = 3):
    """Activate measure-then-freeze block-sparse attention for a denoise loop.

    The first ``warmup`` denoise steps run fully dense (the paper profiles at
    t0=3 after warm-up steps 0-2). On the profiling step each attention
    module runs the dense path, measures exact block masses, and freezes the
    smallest key/value block set retaining ``threshold`` of the mass — one
    frozen set per CFG branch — which every later call reuses. Enter this
    around the sampler's denoise loop (see ``sampling.sample``).
    """
    prev = (
        sparse_state.active,
        sparse_state.threshold,
        sparse_state.block,
        sparse_state.warmup,
    )
    sparse_state.active = True
    sparse_state.threshold = threshold
    sparse_state.block = block
    sparse_state.warmup = warmup
    sparse_state.clear()
    try:
        yield
    finally:
        (
            sparse_state.active,
            sparse_state.threshold,
            sparse_state.block,
            sparse_state.warmup,
        ) = prev
        sparse_state.clear()


def advance_step() -> None:
    """Mark the end of one denoise step.

    Ages the warm-up counter and resets per-step CFG-branch parity. Called by
    ``sampling.sample`` once per Euler step; a no-op when inactive.
    """
    if sparse_state.active:
        sparse_state.step += 1
        sparse_state.branch_counts.clear()


def _expand_gqa(x: Tensor, heads: int) -> Tensor:
    """Replicate GQA key/value heads up to the query-head count."""
    if x.shape[1] == heads:
        return x
    return x.repeat_interleave(heads // x.shape[1], dim=1)


def _dense(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, scale: float | None, gqa: bool
) -> Tensor:
    # Prefer cuDNN (matches the repo's dense path on GPU); fall back to the
    # math/effective kernels so the feature is also exercisable on CPU.
    with sdpa_kernel(
        [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    ):
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    return rearrange(x, "B H L D -> B L (H D)")


def _block_masses(
    q: Tensor, k: Tensor, mask: Tensor | None, scale: float, block: int
) -> Tensor:
    """Per-(head, query-block) attention mass over key blocks, averaged over batch.

    ``q`` is (B, H, Lq, D) and ``k`` is (B, Hk, Lk, D); returns (H, nq, nk).
    Mass[h, i, j] sums the softmax weights from query block ``i`` onto key
    block ``j`` (then averaged over the batch). Computed one query block at
    a time to bound the O(Lq * Lk) memory of the profiling step.
    """
    b, h, lq, _ = q.shape
    lk = k.shape[2]
    k = _expand_gqa(k, h)
    nq, nk = lq // block, lk // block
    masses = q.new_zeros(h, nq, nk)
    for i in range(nq):
        lo, hi = i * block, (i + 1) * block
        qb = q[:, :, lo:hi, :]
        scores = torch.matmul(qb, k.transpose(-1, -2)) * scale  # (B, H, block, Lk)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, :, lo:hi, :], float("-inf"))
        # Padded query rows go all-(-inf) -> NaN; zero them so they never
        # bias the frozen selection.
        attn = scores.softmax(dim=-1).nan_to_num(0.0)
        block_mass = attn.view(b, h, block, nk, block).sum(dim=(2, 4))  # (B, H, nk)
        masses[:, i, :] = block_mass.mean(dim=0)
    return masses


def _select(masses: Tensor, threshold: float) -> dict:
    """Smallest top-mass key-block set per (head, query block) retaining ``threshold``."""
    sorted_mass, sorted_idx = masses.sort(dim=-1, descending=True)  # (H, nq, nk)
    cum = sorted_mass.cumsum(dim=-1)
    # Number of prefixes strictly below threshold; +1 crosses it. Rows whose
    # total mass falls short of the threshold (e.g. fully padded blocks)
    # clamp to keeping every block, i.e. fall back to dense for that block.
    reach = (cum < threshold).sum(dim=-1)
    keep = (reach + 1).clamp(max=masses.shape[-1])
    return {"sorted_idx": sorted_idx, "keep": keep}


def _sparse(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    sel: dict,
    mask: Tensor | None,
    scale: float,
    block: int,
) -> Tensor:
    """Block-sparse attention reusing a frozen retained-mass index set.

    For each query block, gather the top-mass key/value blocks (batched to
    the max kept-set over heads for that block) and run one
    ``scaled_dot_product_attention``. Output matches the dense layout.
    """
    b, h, lq, d = q.shape
    lk = k.shape[2]
    k = _expand_gqa(k, h)
    v = _expand_gqa(v, h)
    nq = lq // block
    sorted_idx: Tensor = sel["sorted_idx"]  # (H, nq, nk)
    keep: Tensor = sel["keep"]  # (H, nq)
    q_blk = q.view(b, h, nq, block, d)
    offsets = torch.arange(block, device=q.device)

    kvalid = qvalid = None
    if mask is not None:
        m2 = mask[:, 0]  # (B, Lq, Lk) outer-product key-padding mask
        kvalid = m2.any(dim=1)  # (B, Lk)
        qvalid = m2.any(dim=2)  # (B, Lq)

    outs = []
    for i in range(nq):
        ki = int(keep[:, i].amax().item())  # max blocks kept by any head here
        idx = sorted_idx[:, i, :ki]  # (H, ki)
        pos = (idx[:, :, None] * block + offsets).reshape(1, h, ki * block)
        pos = pos.expand(b, h, ki * block)  # (B, H, ki*block)
        gather_idx = pos.unsqueeze(-1).expand(-1, -1, -1, d)  # (B, H, ki*block, D)
        ki_sel = torch.gather(k, 2, gather_idx)
        vi_sel = torch.gather(v, 2, gather_idx)
        qi = q_blk[:, :, i, :, :]  # (B, H, block, D)
        attn_mask = None
        if mask is not None:
            key_valid = torch.gather(
                kvalid.unsqueeze(1).expand(b, h, lk), 2, pos
            )  # (B, H, ki*block)
            qry_valid = qvalid[:, i * block : (i + 1) * block]  # (B, block)
            attn_mask = qry_valid[:, None, :, None] & key_valid[:, :, None, :]
        outs.append(
            F.scaled_dot_product_attention(
                qi, ki_sel, vi_sel, attn_mask=attn_mask, scale=scale
            )
        )
    out = torch.cat(outs, dim=2)  # (B, H, Lq, D)
    return rearrange(out, "B H L D -> B L (H D)")


def sparse_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    key: object,
    mask: Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
) -> Tensor:
    """Drop-in sparse-or-dense attention for the K2 MMDiT.

    Inactive (the default): pure dense path. Active: denoise steps below
    ``warmup`` stay dense; on the profiling step each attention module
    returns the dense result, measures block masses, and freezes the
    smallest retained-mass block set, reused thereafter. ``key`` is the
    owning attention module; selections are frozen per (module, CFG branch),
    where the branch is the call parity within the current denoise step (the
    sampler runs the conditional branch first; without CFG every call is
    branch 0).
    """
    st = sparse_state
    if not st.active:
        return _dense(q, k, v, mask, scale, gqa)

    if st.step < st.warmup:
        # Dense warm-up: the paper constructs the block sets at t0=3, after
        # the trajectory has settled, not on the first (pure-noise) step.
        return _dense(q, k, v, mask, scale, gqa)

    block = st.block
    kid = id(key)
    branch = min(st.branch_counts.get(kid, 0), 1)
    st.branch_counts[kid] = branch + 1
    skey = (kid, branch)
    if skey not in st.selections:
        # Profiling step: return the dense result and freeze a selection.
        out = _dense(q, k, v, mask, scale, gqa)
        lq, lk = q.shape[2], k.shape[2]
        heads, kvheads = q.shape[1], k.shape[1]
        divisible = lq >= block and lk >= block and lq % block == 0 and lk % block == 0
        gqa_ok = kvheads == heads or heads % kvheads == 0
        if divisible and gqa_ok:
            sc = scale if scale is not None else q.shape[-1] ** -0.5
            masses = _block_masses(q, k, mask, sc, block)
            st.selections[skey] = _select(masses, st.threshold)
        else:
            st.selections[skey] = None  # keep this module on the dense path
        return out

    sel = st.selections[skey]
    if sel is None:
        return _dense(q, k, v, mask, scale, gqa)
    sc = scale if scale is not None else q.shape[-1] ** -0.5
    return _sparse(q, k, v, sel, mask, sc, block)
