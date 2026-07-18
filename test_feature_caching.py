"""Integration tests for DuCa dual feature caching on the real K2 ``SingleStreamDiT``.

These exercise the wiring in ``feature_caching.py`` against an actual (tiny)
``SingleStreamDiT`` from ``mmdit.py`` - not a mock. The shipped model pins the
``CUDNN_ATTENTION`` SDPA backend and decorates leaf modules with
``torch.compile(fullgraph=True)``, neither of which runs on a CPU CI worker, so
we neutralise both for the duration of the tests. That does not touch the
caching code path under test, which only sees (eager) block outputs.
"""

import torch

# Neutralise torch.compile BEFORE importing mmdit: the decorators wrap the leaf
# forwards at class-definition time and hard-fail (fullgraph=True) once the
# caching cycle makes the call graph vary across steps. The imports below
# therefore intentionally follow this statement (E402 is load-bearing here).
torch.compile = lambda *a, **k: lambda f: f  # type: ignore[assignment]

import torch.nn.functional as F  # noqa: E402
from einops import rearrange  # noqa: E402

import mmdit  # noqa: E402
from mmdit import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402

import feature_caching  # noqa: E402


def _cpu_attention(q, k, v, mask=None, scale=None, gqa=False):
    """CUDNN-free SDPA so the real attention path runs on CPU."""
    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
    )
    return rearrange(x, "B H L D -> B L (H D)")


mmdit.attention = _cpu_attention


def _build_model(layers: int = 4) -> SingleStreamDiT:
    cfg = SingleMMDiTConfig(
        features=64,
        tdim=32,
        txtdim=64,
        heads=4,
        kvheads=4,
        multiplier=2,
        layers=layers,
        patch=2,
        channels=4,
        txtlayers=2,
        txtheads=4,
        txtkvheads=4,
    )
    torch.manual_seed(0)
    return SingleStreamDiT(cfg).eval()


def _inputs(seed: int = 1):
    torch.manual_seed(seed)
    batch, txtlen, image_tokens = 1, 3, 4
    return dict(
        img=torch.randn(batch, image_tokens, 4 * 2 * 2),
        context=torch.randn(batch, txtlen, 2, 64),
        t=torch.tensor([0.5]),
        pos=torch.zeros(batch, txtlen + image_tokens, 3),
        mask=torch.ones(batch, txtlen + image_tokens, dtype=torch.bool),
    )


def _run(model, steps: int = 5, **inputs):
    with torch.no_grad():
        for value in (0.9, 0.7, 0.5, 0.3, 0.1, 0.05, 0.02)[:steps]:
            inputs = dict(inputs)
            inputs["t"] = torch.tensor([value])
            out = model(**inputs)
    return out


def test_wiring_replaces_blocks_and_forward():
    model = _build_model()
    n_layers = len(model.blocks)
    assert not feature_caching.is_dual_feature_caching_enabled(model)

    feature_caching.apply_dual_feature_caching(model)

    assert feature_caching.is_dual_feature_caching_enabled(model)
    # The block list is swapped for the caching proxy, but still hosts every real
    # block (parameter tree, length and indexing preserved).
    assert isinstance(model.blocks, feature_caching._DualCacheBlocks)
    assert len(model.blocks) == n_layers
    assert model.blocks[0] is model.blocks._real[0]

    feature_caching.remove_dual_feature_caching(model)
    assert not feature_caching.is_dual_feature_caching_enabled(model)
    assert not isinstance(model.blocks, feature_caching._DualCacheBlocks)
    assert len(model.blocks) == n_layers


def test_aggressive_steps_skip_block_compute():
    """Over a 5-step cycle (fresh, cons, aggr, cons, aggr) each real block runs
    on the fresh + 2 conservative steps only -> 3 calls vs 5 without caching."""
    model = _build_model()
    inputs = _inputs()

    calls = {"n": 0}

    def _hook(_module, _inp, _out):
        calls["n"] += 1

    handle = model.blocks[0].register_forward_hook(_hook)
    try:
        _run(model, steps=5, **inputs)
        uncached = calls["n"]
        assert uncached == 5  # one call per step without caching

        calls["n"] = 0
        feature_caching.apply_dual_feature_caching(
            model, cycle_length=5, cache_ratio=0.9
        )
        out = _run(model, steps=5, **inputs)
        cached = calls["n"]
        # fresh(1) + conservative(1) + aggressive(0) + conservative(1) + aggressive(0)
        assert cached == 3, f"expected 3 block calls with caching, got {cached}"
        assert cached < uncached
        assert out.shape == (1, 4, 16)
        assert torch.isfinite(out).all()
    finally:
        handle.remove()
        feature_caching.remove_dual_feature_caching(model)


def test_cycle_phases_match_duca_schedule():
    """Phase index 0..4 maps to fresh, conservative, aggressive, conservative,
    aggressive - conservative on odd positions, aggressive on even positions."""
    ph = feature_caching._phase_at
    assert [ph(s, 5) for s in range(1, 11)] == [
        "fresh",
        "conservative",
        "aggressive",
        "conservative",
        "aggressive",
        "fresh",  # new cycle
        "conservative",
        "aggressive",
        "conservative",
        "aggressive",
    ]


def test_remove_restores_exact_output():
    """After removing the patch the model is bit-identical to a never-patched twin."""
    model = _build_model()
    twin = _build_model()  # same seed -> identical weights
    inputs = _inputs()

    feature_caching.apply_dual_feature_caching(model, cycle_length=5, cache_ratio=0.9)
    _run(model, steps=6, **inputs)  # exercise a full cached cycle + restart
    feature_caching.remove_dual_feature_caching(model)

    with torch.no_grad():
        restored = model(**inputs)
        expected = twin(**inputs)
    assert torch.equal(restored, expected)


def test_cache_is_stream_local_for_cfg():
    """cond / uncond branches (different ``context`` tensors) keep independent
    caches and each starts on its own fresh step."""
    model = _build_model()
    feature_caching.apply_dual_feature_caching(model, cycle_length=5, cache_ratio=0.9)
    cond = _inputs(seed=1)
    uncond = _inputs(seed=2)

    with torch.no_grad():
        # cond then uncond at the same timestep, twice - both must stay on fresh
        # for their first call (no cross-stream cache bleed).
        model(**cond)
        assert model.blocks.phase == "fresh"
        model(**uncond)
        assert model.blocks.phase == "fresh"
        # second timestep: each stream advances independently to conservative
        model(**cond)
        assert model.blocks.phase == "conservative"
        model(**uncond)
        assert model.blocks.phase == "conservative"
    feature_caching.remove_dual_feature_caching(model)
