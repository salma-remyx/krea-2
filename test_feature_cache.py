"""Tests for dual feature caching (``feature_cache``).

These exercise the existing MM-DiT (``mmdit``) together with the caching
wrapper. The repo's runtime deps (``torch``/``einops``) may be absent in a
lightweight CI image, so the whole module skips gracefully when they are
missing.
"""

import pytest

torch = pytest.importorskip("torch")

from feature_cache import CachingBlock, DualCacheController, DualCacheDiT
from mmdit import SingleMMDiTConfig, SingleStreamDiT  # existing module

# Tiny config sized so a real SingleStreamDiT can be constructed cheaply.
TINY = SingleMMDiTConfig(
    features=128,
    tdim=32,
    txtdim=128,
    heads=4,
    kvheads=2,
    multiplier=2,
    layers=4,
    patch=2,
    channels=4,
    txtheads=4,
    txtkvheads=2,
    txtlayers=2,
)


# --------------------------------------------------------------------------- #
# Controller policy (pure logic)
# --------------------------------------------------------------------------- #


def test_static_blocks_store_once_then_reuse():
    c = DualCacheController(n_blocks=6, n_static=3, n_dynamic=0)
    key = object()

    c.enter_branch(key)  # step 0
    assert [c.policy(i) for i in range(6)] == [
        "store",
        "store",
        "store",
        "compute",
        "compute",
        "compute",
    ]

    for _ in range(2):  # steps 1 and 2
        c.enter_branch(key)
        assert [c.policy(i) for i in range(6)] == [
            "reuse",
            "reuse",
            "reuse",
            "compute",
            "compute",
            "compute",
        ]


def test_dynamic_blocks_alternate_refresh_and_reuse():
    c = DualCacheController(n_blocks=6, n_static=0, n_dynamic=3)
    key = object()

    expected = ["store", "store", "store"]
    for step in range(4):
        c.enter_branch(key)
        assert [c.policy(i) for i in range(3)] == expected
        assert c.policy(5) == "compute"  # tier past n_dynamic always computes
        # step 0 store, step 1 reuse, step 2 store, step 3 reuse
        expected = ["reuse", "reuse", "reuse"] if expected[0] == "store" else [
            "store",
            "store",
            "store",
        ]


def test_branches_are_independent():
    c = DualCacheController(n_blocks=4, n_static=2, n_dynamic=0)
    a, b = object(), object()

    c.enter_branch(a)  # a step 0
    c.enter_branch(b)  # b step 0 (independent of a)
    assert c.policy(0) == "store"

    c.enter_branch(a)  # a advances to step 1
    assert c.policy(0) == "reuse"

    c.enter_branch(b)  # b still only its step 1, independent cache
    assert c.policy(0) == "reuse"


def test_inactive_controller_is_compute_only():
    c = DualCacheController(n_blocks=4, n_static=2, n_dynamic=1)
    # No branch entered -> every block recomputes (safe default).
    assert all(c.policy(i) == "compute" for i in range(4))


def test_rejects_oversubscribed_cache():
    with pytest.raises(ValueError):
        DualCacheController(n_blocks=4, n_static=3, n_dynamic=2)


# --------------------------------------------------------------------------- #
# CachingBlock skip behavior (CPU, no CUDA)
# --------------------------------------------------------------------------- #


def test_caching_block_skips_compute_on_reuse():
    calls = {"n": 0}

    class CountingBlock(torch.nn.Module):
        def forward(self, x, vec, freqs, mask=None):
            calls["n"] += 1
            return x + 1.0

    controller = DualCacheController(n_blocks=1, n_static=1, n_dynamic=0)
    block = CachingBlock(CountingBlock(), 0, controller)
    x = torch.zeros(2, 3)

    controller.enter_branch(0)  # step 0 -> store
    out0 = block(x, None, None, None)
    assert calls["n"] == 1
    assert controller.skips == 0
    assert torch.equal(out0, torch.ones(2, 3))

    controller.enter_branch(0)  # step 1 -> reuse (underlying block not called)
    out1 = block(x, None, None, None)
    assert calls["n"] == 1
    assert controller.skips == 1
    assert torch.equal(out1, out0)  # cached output served verbatim
    controller.deactivate()


# --------------------------------------------------------------------------- #
# Wrap a real SingleStreamDiT (construction only, CPU)
# --------------------------------------------------------------------------- #


def test_wrap_installs_caching_over_real_blocks():
    dit = SingleStreamDiT(TINY)
    n = len(dit.blocks)

    wrapped = DualCacheDiT(dit, n_static=2, n_dynamic=1)
    assert wrapped.config is dit.config  # drop-in exposes config.patch
    assert wrapped.controller.n_static == 2
    assert wrapped.controller.n_dynamic == 1
    assert len(dit.blocks) == n
    assert all(isinstance(b, CachingBlock) for b in dit.blocks)

    # Zero-cache leaves the model untouched.
    dit2 = SingleStreamDiT(TINY)
    DualCacheDiT(dit2, n_static=0, n_dynamic=0)
    assert not any(isinstance(b, CachingBlock) for b in dit2.blocks)


# --------------------------------------------------------------------------- #
# Full forward against the real MM-DiT (CUDA only: attention uses CUDNN)
# --------------------------------------------------------------------------- #


def _tiny_inputs(device, dtype):
    b, n_img, txtlen = 1, 4, 4
    img = torch.randn(b, n_img, TINY.channels * TINY.patch**2, device=device, dtype=dtype)
    context = torch.randn(b, txtlen, TINY.txtlayers, TINY.txtdim, device=device, dtype=dtype)
    t = torch.zeros(b, device=device, dtype=dtype)
    pos = torch.zeros(b, txtlen + n_img, 3, device=device, dtype=dtype)
    mask = torch.ones(b, txtlen + n_img, device=device, dtype=torch.bool)
    return img, context, t, pos, mask


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SingleStreamDiT attention uses the CUDNN backend (GPU only)",
)
def test_zero_cache_is_bit_exact_dropin():
    device, dtype = "cuda", torch.bfloat16
    dit_plain = SingleStreamDiT(TINY).to(device=device, dtype=dtype).eval()
    dit_wrapped = SingleStreamDiT(TINY).to(device=device, dtype=dtype).eval()
    dit_wrapped.load_state_dict(dit_plain.state_dict())
    wrapped = DualCacheDiT(dit_wrapped, n_static=0, n_dynamic=0)

    inputs = _tiny_inputs(device, dtype)
    with torch.no_grad():
        out_plain = dit_plain(*inputs)
        out_wrapped = wrapped(*inputs)
    assert torch.equal(out_plain, out_wrapped)  # zero overhead, no numerical drift


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SingleStreamDiT attention uses the CUDNN backend (GPU only)",
)
def test_caching_skips_blocks_across_steps():
    device, dtype = "cuda", torch.bfloat16
    dit = SingleStreamDiT(TINY).to(device=device, dtype=dtype).eval()
    wrapped = DualCacheDiT(dit, n_static=2, n_dynamic=1)
    inputs = _tiny_inputs(device, dtype)

    with torch.no_grad():
        out0 = wrapped(*inputs)  # step 0: stores
        out1 = wrapped(*inputs)  # step 1: static+dynamic reuse -> skips

    assert out0.shape == out1.shape
    # step 1 reuses 2 static + 1 dynamic block.
    assert wrapped.controller.skips == 3
