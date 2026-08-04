"""Integration tests for increment-calibrated feature caching.

These exercise the wiring in the *existing* modules (``mmdit.SingleStreamDiT``
block loop and ``sampling.sample`` Euler loop), not just the new cache in
isolation. ``mmdit.attention`` forces the cuDNN SDPA backend (GPU-only), so an
autouse fixture swaps in a math-backend stand-in to let the real model run on
CPU here; the caching logic under test is unchanged.
"""

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

import mmdit
import sampling
from feature_cache import IncrementCalibratedCache, channel_aware_svd


def _cpu_attention(q, k, v, mask=None, scale=None, gqa=False):
    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
    )
    return rearrange(x, "B H L D -> B L (H D)")


@pytest.fixture(autouse=True)
def patch_attention(monkeypatch):
    """``mmdit.attention`` hard-codes the cuDNN backend; use math on CPU."""
    monkeypatch.setattr(mmdit, "attention", _cpu_attention)


def _tiny_config():
    return mmdit.SingleMMDiTConfig(
        features=128,
        tdim=64,
        txtdim=128,
        heads=8,
        kvheads=4,
        multiplier=2,
        layers=4,
        patch=2,
        channels=4,
        txtlayers=2,
        txtheads=8,
        txtkvheads=4,
    )


def _model():
    torch.manual_seed(0)
    return mmdit.SingleStreamDiT(_tiny_config()).eval()


class StubEncoder:
    """Mimics encoder(prompts) -> (txt, txtmask) with the K2 context shape."""

    def __init__(self, txtlen=5, num_layers=2, txtdim=128):
        self.txtlen = txtlen
        self.num_layers = num_layers
        self.txtdim = txtdim

    def __call__(self, prompts):
        n = len(prompts)
        txt = torch.randn(n, self.txtlen, self.num_layers, self.txtdim)
        mask = torch.ones(n, self.txtlen, dtype=torch.bool)
        return txt, mask


class StubAE:
    compression = 8
    channels = 4

    def decode(self, x):
        b, _c, h, w = x.shape
        return torch.randn(b, 3, h * 8, w * 8)


def test_sample_without_cache_preserves_contract():
    images = sampling.sample(
        _model(),
        StubAE(),
        StubEncoder(),
        ["a fox in the snow"],
        width=32,
        height=32,
        steps=4,
        guidance=0.0,
        seed=0,
        cache_every=None,
        device="cpu",
        dtype=torch.float32,
    )
    assert len(images) == 1
    assert images[0].size == (32, 32)


def test_sample_with_cache_preserves_contract_and_engages():
    model = _model()
    images = sampling.sample(
        model,
        StubAE(),
        StubEncoder(),
        ["a fox in the snow"],
        width=32,
        height=32,
        steps=6,
        guidance=0.0,
        seed=0,
        cache_every=2,
        device="cpu",
        dtype=torch.float32,
    )
    assert len(images) == 1
    assert images[0].size == (32, 32)

    cache = model.cache
    assert cache is not None
    # compute_every=2 over 6 steps -> steps 0,2,4 compute (3x4 blocks),
    # steps 1,3,5 cached (3x4 blocks).
    assert cache.block_calls == 3 * len(model.blocks)
    assert cache.cache_hits == 3 * len(model.blocks)
    assert cache.block_calls + cache.cache_hits == 6 * len(model.blocks)


def test_mmdit_forward_routes_compute_then_cache():
    """The block-loop wiring in mmdit.forward: compute seeds, then cache skips."""
    model = _model()
    nblocks = len(model.blocks)
    cache = IncrementCalibratedCache(nblocks, slots=1, compute_every=2)
    model.cache = cache

    img = torch.randn(1, 16, 16)  # 16 tokens, in-channels = channels * patch**2
    context = torch.randn(1, 5, 2, 128)
    pos = torch.zeros(1, 21, 3)  # txtlen(5) + imglen(16)
    mask = torch.ones(1, 21, dtype=torch.bool)
    args = {
        "img": img,
        "context": context,
        "t": torch.zeros(1),
        "pos": pos,
        "mask": mask,
    }

    cache.step = 0  # compute step
    with torch.no_grad():
        out0 = model(**args)
    assert cache.block_calls == nblocks
    assert cache.cache_hits == 0
    assert torch.isfinite(out0).all()

    cache.step = 1  # cache step: no new real block calls, all blocks reused
    with torch.no_grad():
        out1 = model(**args)
    assert cache.block_calls == nblocks  # unchanged
    assert cache.cache_hits == nblocks
    assert torch.isfinite(out1).all()


def test_increment_calibration_applies_svd_refined_increment():
    """Core ICC mechanism on a real block: base + channel-aware-SVD increment."""
    model = _model()
    block = model.blocks[0]
    L = 7
    pos = torch.zeros(1, L, 3)
    freqs = model.posemb(pos)
    mask = mmdit._mask(torch.ones(1, L, dtype=torch.bool))
    args = (torch.randn(1, 768), freqs, mask)  # (vec, freqs, mask)
    xs = [torch.randn(1, L, 128) for _ in range(4)]

    cache = IncrementCalibratedCache(1, slots=1, compute_every=2, svd_enabled=True)

    cache.step = 0  # compute -> seeds output, no increment yet
    y0 = cache(0, block, xs[0], *args)
    assert cache.increments[0][0] is None

    cache.step = 1  # cache, but no increment -> verbatim reuse
    y1 = cache(0, block, xs[1], *args)
    assert torch.equal(y1, cache.outputs[0][0])

    cache.step = 2  # compute -> increment = y2 - y0 (distinct inputs)
    y2 = cache(0, block, xs[2], *args)
    inc = cache.increments[0][0]
    assert inc is not None
    assert not torch.equal(y2, y0)

    cache.step = 3  # cache -> base (y2) + svd-refined increment
    y3 = cache(0, block, xs[3], *args)
    base = cache.outputs[0][0]
    assert torch.allclose(y3, base + channel_aware_svd(inc))
    assert not torch.equal(y3, base)


def test_cache_disabled_runs_every_block():
    cache = IncrementCalibratedCache(3, slots=1, compute_every=2, enabled=False)
    calls = {"n": 0}

    class Block:
        def __call__(self, x, *_):
            calls["n"] += 1
            return x + 1

    cache.step = 1  # would normally be a cache step
    for i in range(3):
        cache(i, Block(), torch.zeros(1))
    assert calls["n"] == 3
    assert cache.cache_hits == 0
