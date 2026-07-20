"""Integration tests for the stability-guided step cache wired into ``sample``.

These exercise the real ``sampling.sample`` Euler loop (a non-new module) with
lightweight stubs for the model / autoencoder / text encoder, and assert that
the SADA step cache actually reduces the number of ``model(...)`` forward calls
when the trajectory is stable -- and does not when it is not.
"""

from types import SimpleNamespace

import torch

from sampling import sample
from stability_cache import StabilityCache, relative_curvature


PATCH = 2
COMPRESSION = 8
CHANNELS = 16


class StubAE:
    compression = COMPRESSION
    channels = CHANNELS

    def decode(self, x):
        # Real VAE maps a (b, c, h, w) latent to (b, 3, h*8, w*8) pixels.
        b, _, h, w = x.shape
        return torch.zeros(b, 3, h * self.compression, w * self.compression)


class StubEncoder:
    """Returns a tiny constant text context; ignores the prompt strings."""

    def __call__(self, prompts):
        n = len(prompts)
        txtlen, txtdim = 5, 8
        txt = torch.zeros(n, txtlen, txtdim)
        mask = torch.ones(n, txtlen, dtype=torch.bool)
        return txt, mask


class CountingModel:
    """Model stub that counts forward calls and returns a velocity per call."""

    def __init__(self, velocity_fn):
        self.config = SimpleNamespace(patch=PATCH)
        self.calls = 0
        self.velocity_fn = velocity_fn

    def __call__(self, img, context, t, pos, mask):
        self.calls += 1
        return self.velocity_fn(img, self.calls)


def _constant_velocity(img, _calls):
    # Identical velocity every step => zero second-order difference => stable.
    return torch.full_like(img, 0.1)


def _wobbly_velocity(img, calls):
    # Sign-flipping, unit-magnitude velocity => large curvature => never stable.
    return torch.full_like(img, 1.0 if calls % 2 else -1.0)


def _run(model, **kwargs):
    return sample(
        model,
        StubAE(),
        StubEncoder(),
        ["a test prompt"],
        device="cpu",
        width=32,
        height=32,
        steps=12,
        guidance=0.0,  # no CFG: one model call per step, easy to reason about
        **kwargs,
    )


def test_sample_no_accel_computes_every_step():
    """With the cache disabled, every step pays for a forward (baseline path)."""
    model = CountingModel(_constant_velocity)
    _run(model, accel=None)
    assert model.calls == 12


def test_sample_accel_skips_stable_steps():
    """A stable (constant-velocity) trajectory reuses cached outputs."""
    model = CountingModel(_constant_velocity)
    images = _run(model, accel=0.05)
    # 3 warmup steps always compute; after that every other step is reused.
    assert model.calls < 12
    assert model.calls >= 3
    assert len(images) == 1
    assert images[0].size == (32, 32)


def test_sample_accel_unstable_skips_nothing():
    """A wobbly trajectory is never stable, so nothing is skipped."""
    model = CountingModel(_wobbly_velocity)
    _run(model, accel=0.05)
    assert model.calls == 12


def test_sample_accel_with_cfg_smoke():
    """CFG doubles the forwards per step; stable steps skip both branches."""
    model = CountingModel(_constant_velocity)
    images = sample(
        model,
        StubAE(),
        StubEncoder(),
        ["a fox in the snow"],
        device="cpu",
        width=32,
        height=32,
        steps=10,
        guidance=4.5,
        accel=0.05,
    )
    assert len(images) == 1
    # 10 steps * 2 forwards == 20 without the cache; the cache skips stable ones.
    assert model.calls < 20


def test_cache_requires_history_then_reuses():
    """Direct check of the cache's stability gate and reuse accounting."""
    cache = StabilityCache(threshold=0.05, max_skip=1)
    v = torch.full((1, 4, 8), 0.1)
    zero = torch.zeros((1, 4, 8))

    # Not enough history to judge curvature yet.
    cache.record(v, None, v)
    assert not cache.should_reuse()
    cache.record(v, None, v)
    assert not cache.should_reuse()
    cache.record(v, None, v)
    # Three identical velocities => zero curvature => stable.
    assert cache.should_reuse()
    cond, uncond = cache.cached()
    assert torch.equal(cond, v) and uncond is None
    cache.mark_reused()
    # max_skip=1 forces a fresh compute on the very next step.
    assert not cache.should_reuse()
    assert cache.computed == 3 and cache.reused == 1
    # relative_curvature of a linear ramp is ~0; of identical vels is exactly 0.
    assert float(relative_curvature([v, v, v])) == 0.0
    assert float(relative_curvature([zero, zero, zero])) == 0.0
