"""Integration tests for the opt-in block-sparse attention hook.

These exercise the wiring in ``mmdit.Attention`` (the existing call site),
not just the new module in isolation.
"""

import torch

from mmdit import Attention
from sparse_attention import block_sparse_attention, sparse_state

BLOCK = 32
SEQ = 256  # SEQ // BLOCK == 8 key/query blocks


def _module():
    torch.manual_seed(0)
    return Attention(dim=64, heads=4)


def test_context_toggles_state_and_clears_selections():
    assert sparse_state.active is False
    with block_sparse_attention(threshold=0.99, block=BLOCK):
        assert sparse_state.active is True
    assert sparse_state.active is False
    assert sparse_state.selections == {}


def test_attention_profiles_then_reuses_near_lossless():
    attn = _module()
    x = torch.randn(2, SEQ, 64)

    with block_sparse_attention(threshold=0.99, block=BLOCK):
        profiled = attn(x)  # first call: dense output + measure + freeze
        assert id(attn) in sparse_state.selections
        sparse_out = attn(x)  # second call: reuses the frozen block set

    assert profiled.shape == sparse_out.shape == (2, SEQ, 64)
    assert torch.isfinite(sparse_out).all()

    # Higher retained mass must land strictly closer to the dense profile.
    def err_for(threshold):
        with block_sparse_attention(threshold=threshold, block=BLOCK):
            dense = attn(x)  # profile = dense reference
            sp = attn(x)  # sparse
        return (sp - dense).abs().mean().item()

    err_hi = err_for(0.99)
    err_lo = err_for(0.30)
    assert err_hi < err_lo, (err_hi, err_lo)

    # 99% retention should be a small fraction of the activation magnitude.
    with block_sparse_attention(threshold=0.99, block=BLOCK):
        dense = attn(x)
        sp = attn(x)
    rel = (sp - dense).abs().mean() / (dense.abs().mean() + 1e-6)
    assert rel < 0.5, rel


def test_lower_threshold_actually_drops_blocks():
    attn = _module()
    x = torch.randn(2, SEQ, 64)
    with block_sparse_attention(threshold=0.30, block=BLOCK):
        attn(x)  # profile
        keep = sparse_state.selections[id(attn)]["keep"]  # (H, nq)
    nk = SEQ // BLOCK
    # At 30% retention the frozen set must drop key blocks on average.
    assert keep.float().mean() < nk, (keep.float().mean().item(), nk)


def test_block_not_dividing_falls_back_to_dense():
    attn = _module()
    x = torch.randn(2, SEQ + 1, 64)  # 257 not divisible by BLOCK
    with block_sparse_attention(threshold=0.99, block=BLOCK):
        out = attn(x)  # profiles, but marks this module dense (None selection)
    assert out.shape == (2, SEQ + 1, 64)
    assert torch.isfinite(out).all()
