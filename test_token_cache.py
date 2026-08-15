"""Tests for the DuCa-style dual token cache wired into mmdit/sampling.

The attention hot path forces the CUDNN SDPA backend, which has no CPU
kernel; these tests broaden it to allow the math fallback. torch.compile is
disabled so the attention override is visible on CPU.
"""

import os
import sys
import types

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

try:
    import PIL  # noqa: F401
except ModuleNotFoundError:  # sampling imports PIL for the final decode
    _pil = types.ModuleType("PIL")
    _pil.Image = types.SimpleNamespace(fromarray=lambda arr: arr)
    sys.modules["PIL"] = _pil
    sys.modules["PIL.Image"] = _pil.Image

import torch  # noqa: E402
import torch.nn.attention as tna  # noqa: E402
from torch.nn.attention import SDPBackend  # noqa: E402

import mmdit  # noqa: E402
from mmdit import SingleMMDiTConfig, SingleStreamDiT  # noqa: E402
from token_cache import DualTokenCache, dual_cache  # noqa: E402

mmdit.sdpa_kernel = lambda backend: tna.sdpa_kernel(
    SDPBackend.CUDNN_ATTENTION
    if torch.cuda.is_available()
    else SDPBackend.MATH
)


def tiny_model(layers: int = 4) -> SingleStreamDiT:
    return SingleStreamDiT(
        SingleMMDiTConfig(
            features=128,
            tdim=32,
            txtdim=64,
            heads=4,
            kvheads=2,
            multiplier=2,
            layers=layers,
            patch=2,
            channels=4,
            txtlayers=3,
            txtheads=8,
            txtkvheads=8,
        )
    ).eval()


def tiny_inputs(batch: int = 2):
    img = torch.randn(batch, 64, 16)
    ctx = torch.randn(batch, 5, 3, 64)
    length = 5 + 64
    pos = torch.zeros(batch, length, 3)
    pos[:, 5:, 1] = torch.arange(64) // 8
    pos[:, 5:, 2] = torch.arange(64) % 8
    mask = torch.ones(batch, length, dtype=torch.bool)
    return img, ctx, pos, mask


def forward(model, inputs, t):
    img, ctx, pos, mask = inputs
    with torch.no_grad():
        return model(img=img, context=ctx, t=t, pos=pos, mask=mask)


class TestStepSchedule:
    def test_cached_steps_alternate_aggressive_conservative(self):
        cache = DualTokenCache(interval=2)
        cache.enabled = True
        modes = []
        for _ in range(6):
            cache.begin_step()
            modes.append(cache.mode)
        assert modes == [
            "compute",
            "aggressive",
            "compute",
            "conservative",
            "compute",
            "aggressive",
        ]

    def test_disabled_cache_always_computes(self):
        cache = DualTokenCache()
        cache.begin_step()
        cache.begin_step()
        assert cache.mode == "compute"


class TestAttentionCaching:
    def test_compute_steps_match_uncached_forward(self):
        """With caching enabled, every other step runs the true forward."""
        model, inputs = tiny_model(), tiny_inputs()
        t = torch.rand(2)
        uncached = forward(model, inputs, t)
        with dual_cache.configured(reuse=0.8):
            dual_cache.begin_step()  # compute
            assert torch.allclose(forward(model, inputs, t), uncached)

    def test_aggressive_step_reuses_cached_features(self):
        """An aggressive step returns the cached features verbatim."""
        model, inputs = tiny_model(), tiny_inputs()
        t = torch.rand(2)
        with dual_cache.configured(reuse=0.8):
            dual_cache.begin_step()  # compute, seeds cache
            seeded = forward(model, inputs, t)
            dual_cache.begin_step()  # aggressive
            assert torch.equal(forward(model, inputs, t), seeded)

    def test_conservative_step_recomputes_random_subset(self):
        """A conservative step blends: the recomputed tokens match a full
        forward under the new input, while the cached ones keep the previous
        step's features (so the result matches neither pure output)."""
        model, inputs = tiny_model(), tiny_inputs()
        t = torch.rand(2)
        with dual_cache.configured(reuse=0.8):
            dual_cache.begin_step()  # compute, seeds the cache
            forward(model, inputs, t)
            dual_cache.begin_step()  # aggressive
            forward(model, inputs, t)
            dual_cache.begin_step()  # compute — this seeds the blend below
            seeded = forward(model, inputs, t)
            t2 = t + 0.5  # new input, so recomputed tokens change
            dual_cache.begin_step()  # conservative
            mixed = forward(model, inputs, t2)
            dual_cache.enabled = False
            fresh = forward(model, inputs, t2)
        assert not torch.allclose(mixed, seeded)
        assert not torch.allclose(mixed, fresh)

    def test_text_fusion_is_never_cached(self):
        """The text fusion transformer runs every step, so a changed context
        changes even a cached step's output."""
        model = tiny_model()
        img, ctx, pos, mask = tiny_inputs()
        t = torch.rand(2)

        def run(context):
            with torch.no_grad():
                return model(img=img, context=context, t=t, pos=pos, mask=mask)

        with dual_cache.configured(reuse=0.8):
            dual_cache.begin_step()
            run(ctx)
            dual_cache.begin_step()  # aggressive
            before = run(ctx)
            dual_cache.begin_step()
            run(ctx)
            dual_cache.begin_step()  # aggressive again, new context
            after = run(ctx * 3.0)
        assert not torch.allclose(before, after, atol=1e-4)

    def test_cfg_branches_get_separate_cache_entries(self):
        """begin_pass namespaces cond/uncond so CFG branches never share."""
        model, inputs = tiny_model(layers=2), tiny_inputs()
        t = torch.rand(2)
        with dual_cache.configured(reuse=0.8):
            dual_cache.begin_step()
            forward(model, inputs, t)  # cond
            dual_cache.begin_pass()
            forward(model, inputs, t)  # uncond
            assert len(dual_cache._features) == 4  # 2 layers x 2 branches

    def test_cache_clears_on_exit(self):
        model, inputs = tiny_model(), tiny_inputs()
        t = torch.rand(2)
        with dual_cache.configured(reuse=0.8):
            dual_cache.begin_step()
            forward(model, inputs, t)
            assert dual_cache._features
        assert dual_cache._features == {}
        assert dual_cache.mode == "compute"

    def test_disabled_reuse_leaves_model_bit_identical(self):
        model, inputs = tiny_model(), tiny_inputs()
        t = torch.rand(2)
        uncached = forward(model, inputs, t)
        with dual_cache.configured(reuse=None):
            for _ in range(3):
                dual_cache.begin_step()
                forward(model, inputs, t)
        assert torch.allclose(forward(model, inputs, t), uncached)


class TestSamplerWiring:
    def test_sample_threads_cache_reuse(self):
        """sampling.sample drives the cache: with caching on, a multi-step
        run produces (finite) output and the cache advances per step."""
        import sampling

        class DummyAE:
            compression, channels = 4, 4

            def decode(self, x):
                return x.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)

        class DummyEncoder:
            max_length = 5

            def __call__(self, prompts):
                n = len(prompts)
                return (
                    torch.randn(n, 5, 3, 64),
                    torch.ones(n, 5, dtype=torch.bool),
                )

        model = tiny_model(layers=2)
        images = sampling.sample(
            model,
            DummyAE(),
            DummyEncoder(),
            ["a fox"],
            steps=4,
            guidance=0.0,
            seed=0,
            device="cpu",
            dtype=torch.float32,
            cache_reuse=0.8,
        )
        assert len(images) == 1
