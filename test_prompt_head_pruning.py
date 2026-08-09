"""Tests for prompt_head_pruning.

These import the EXISTING ``mmdit`` module and exercise its real
``Attention`` / ``SingleStreamDiT`` call sites through the pruning patch.

Note: the repo's ``mmdit.attention`` forces the CUDNN SDPA backend, which is not
available on CPU. Each test swaps in a CPU-capable SDPA shim for ``mmdit.attention``
(the patch reads it at call time, so the swap is picked up by both the original
and patched forwards). This mirrors the production path on a CPU runner.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

import mmdit
import prompt_head_pruning as php

HEADS = 8
KVHEADS = 4
DIM = 128


def _cpu_attention(q, k, v, mask=None, scale=None, gqa=False):
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
    )
    return rearrange(out, "B H L D -> B L (H D)")


@pytest.fixture(autouse=True)
def _cpu_attention_shim(monkeypatch):
    """Make mmdit.attention runnable on CPU for the duration of each test."""
    monkeypatch.setattr(mmdit, "attention", _cpu_attention)
    yield
    # Belt-and-braces: never leave a class-level patch installed between tests.
    php.restore_forward()


def _attn():
    torch.manual_seed(0)
    return mmdit.Attention(DIM, heads=HEADS, kvheads=KVHEADS)


def _qkv():
    torch.manual_seed(1)
    return torch.randn(1, 16, DIM)


def test_no_prune_is_bit_identical_to_original():
    """An all-False mask must take the original-forward fast path verbatim."""
    attn = _attn()
    qkv = _qkv()
    baseline = attn(qkv)

    mask = torch.zeros(HEADS, dtype=torch.bool)
    handle = php.enable(_BlocksWrapper([_BlockWrapper(attn)]), mask.unsqueeze(0))
    try:
        patched = attn(qkv)
    finally:
        handle.disable()

    assert torch.equal(patched, baseline)
    # forward fully restored
    assert mmdit.Attention.forward is not php._pruned_forward
    assert not hasattr(attn, php._PRUNE_ATTR)


def test_pruning_all_heads_zeros_output():
    """Zeroing every head's contribution leaves only wo's (absent) bias -> zeros."""
    attn = _attn()  # bias=False by default
    qkv = _qkv()

    mask = torch.ones(HEADS, dtype=torch.bool)
    handle = php.enable(_BlocksWrapper([_BlockWrapper(attn)]), mask.unsqueeze(0))
    try:
        out = attn(qkv)
    finally:
        handle.disable()

    assert out.shape == (1, 16, DIM)
    assert torch.all(out == 0)


def test_pruned_head_contributes_zero():
    """Pruning exactly head h must equal the baseline with head h's output zeroed."""
    attn = _attn()
    qkv = _qkv()
    baseline = attn(qkv)

    h = 3
    mask = torch.zeros(HEADS, dtype=torch.bool)
    mask[h] = True
    handle = php.enable(_BlocksWrapper([_BlockWrapper(attn)]), mask.unsqueeze(0))
    try:
        pruned = attn(qkv)
    finally:
        handle.disable()

    # Manual reference: zero head h's attention output before wo, matching the patch.
    with torch.no_grad():
        q = rearrange(attn.wq(qkv), "B L (H D) -> B H L D", H=HEADS)
        k = rearrange(attn.wk(qkv), "B L (H D) -> B H L D", H=KVHEADS)
        v = rearrange(attn.wv(qkv), "B L (H D) -> B H L D", H=KVHEADS)
        q, k, v = attn.qknorm(q, k, v)
        out = rearrange(
            _cpu_attention(q, k, v, gqa=attn.gqa), "B L (H D) -> B L H D", H=HEADS
        )
        out[:, :, h, :] = 0.0
        ref = attn.wo(
            rearrange(out, "B L H D -> B L (H D)") * F.sigmoid(attn.gate(qkv))
        )

    assert torch.allclose(pruned, ref, atol=1e-6)
    # And it must differ from the unpruned baseline (head h did carry signal).
    assert not torch.allclose(pruned, baseline, atol=1e-6)


def test_disable_restores_baseline():
    attn = _attn()
    qkv = _qkv()
    baseline = attn(qkv)

    mask = torch.ones(HEADS, dtype=torch.bool)
    with php.prompt_head_pruning(
        _BlocksWrapper([_BlockWrapper(attn)]), mask.unsqueeze(0)
    ):
        assert not torch.allclose(attn(qkv), baseline, atol=1e-6)

    # After the context exits, behaviour is exactly the original.
    assert torch.equal(attn(qkv), baseline)


def test_build_prune_mask_picks_topk_per_layer():
    # Two layers, four heads. Layer 0: head 3 reads prompt most; layer 1: head 0.
    prompt_attn = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.9],
            [0.9, 0.1, 0.2, 0.3],
        ]
    )
    mask = php.build_prune_mask(prompt_attn, prune_ratio=0.5)
    assert mask.dtype == torch.bool
    assert mask.sum(dim=-1).tolist() == [2, 2]
    assert mask[0].tolist() == [False, False, True, True]  # top-2 of layer 0
    assert mask[1].tolist() == [True, False, False, True]  # top-2 of layer 1


def test_build_prune_mask_always_keeps_a_head():
    prompt_attn = torch.tensor([[0.1, 0.2, 0.3, 0.9]])
    mask = php.build_prune_mask(prompt_attn, prune_ratio=1.0)
    # Even at 100%, clamp keeps at least one head alive.
    assert mask.sum().item() <= 3


def test_prompt_attention_mass_ranks_prompt_readers():
    # head 0 attends prompt keys strongly, head 1 attends image keys.
    L, D = 6, 8
    txtlen = 2
    q = torch.zeros(2, 2, L, D)
    k = torch.zeros(2, 2, L, D)
    # head 0: queries point at prompt keys (cos ~1 with keys 0..txtlen)
    q[0, 0, txtlen:] = 1.0
    k[0, 0, :txtlen] = 1.0
    # head 1: queries point at image keys
    q[1, 1, txtlen:] = 1.0
    k[1, 1, txtlen:] = 1.0

    mass = php._prompt_attention_mass(
        q, k, mask=None, prompt_token_end=txtlen, image_query_end=L, gqa=False
    )
    assert mass.shape == (2,)
    assert mass[0] > mass[1]


def test_enable_attaches_masks_to_real_dit_blocks():
    cfg = mmdit.SingleMMDiTConfig(
        features=128,
        tdim=64,
        txtdim=64,
        heads=8,
        multiplier=2,
        layers=2,
        patch=2,
        channels=16,
        bias=False,
        theta=1e3,
        kvheads=4,
        txtlayers=2,
        txtheads=4,
        txtkvheads=4,
    )
    model = mmdit.SingleStreamDiT(cfg)
    prompt_attn = torch.rand(cfg.layers, cfg.heads)
    mask = php.build_prune_mask(prompt_attn, prune_ratio=0.25)

    handle = php.enable(model, mask)
    try:
        assert mmdit.Attention.forward is php._pruned_forward
        for block, layer_mask in zip(model.blocks, mask):
            assert torch.equal(block.attn._php_prune_mask, layer_mask)
        # Text-fusion attns are NOT joint-stream -> no mask attached.
        assert not hasattr(model.txtfusion.refiner_blocks[0].attn, php._PRUNE_ATTR)
    finally:
        handle.disable()

    assert mmdit.Attention.forward is not php._pruned_forward
    for block in model.blocks:
        assert not hasattr(block.attn, php._PRUNE_ATTR)


def test_calibrate_records_per_layer_head_mass():
    """calibrate() runs a real model forward and gathers (layers, heads) mass."""
    attn0 = _attn()
    attn1 = _attn()
    blocks = [_BlockWrapper(attn0), _BlockWrapper(attn1)]

    torch.manual_seed(7)
    txtlen, imglen, feat = 4, 6, DIM
    qkv = torch.randn(1, txtlen + imglen, feat)
    mask = torch.ones(1, txtlen + imglen, dtype=torch.bool)

    prompt_attn = php.calibrate(
        _CalibModel(blocks, qkv, mask),
        img=torch.randn(1, imglen, feat),
        context=torch.randn(1, txtlen, feat),
        t=torch.zeros(1),
        pos=torch.zeros(1, txtlen + imglen, 3),
        mask=mask,
    )
    assert prompt_attn.shape == (2, HEADS)
    assert torch.isfinite(prompt_attn).all()
    # Each image-query's prompt-key mass is <= 1; summed over imglen queries.
    assert torch.all(prompt_attn >= -1e-6)
    assert torch.all(prompt_attn <= imglen + 1e-5)


# ---- minimal stand-ins so we exercise mmdit.Attention without the compiled DiT ----


class _BlockWrapper:
    """Mimics SingleStreamBlock: exposes ``.attn`` and routes ``__call__`` to it."""

    def __init__(self, attn):
        self.attn = attn

    def __call__(self, qkv, freqs=None, mask=None):
        return self.attn(qkv, freqs=freqs, mask=mask)


class _BlocksWrapper:
    """Mimics SingleStreamDiT.blocks for enable()/disable()."""

    def __init__(self, blocks):
        self.blocks = blocks


class _CalibModel:
    """Stand-in model: runs each block's joint attention once (freqs=None).

    This lets calibrate() be tested without triggering the compiled
    LastLayer/PositionalEncoding/RMSNorm in the real SingleStreamDiT.
    """

    def __init__(self, blocks, qkv, mask):
        self.blocks = blocks
        self._qkv = qkv
        self._mask = mask

    def __call__(self, img, context, t, pos, mask):
        for block in self.blocks:
            block(self._qkv, freqs=None, mask=_bool_keypad_mask(self._mask))


def _bool_keypad_mask(mask):
    """Expand a (B, L) key-padding mask like mmdit._mask -> (B, 1, L, L)."""
    return mask.unsqueeze(1).unsqueeze(2) & mask.unsqueeze(1).unsqueeze(3)
