"""Tests for the SpeCa-style speculative velocity cache.

The integration test goes through the real :func:`sampling.sample` entry point
(the call site) with lightweight stand-ins for the autoencoder, text encoder
and MMDiT, so it exercises the wiring in ``sampling.py`` rather than just the
new module in isolation.
"""

import math
import types

import torch
import torch.nn as nn
from PIL import Image

from sampling import sample  # the existing call-site module under test
from speculative_cache import SpeculativeVelocityCache

PATCH = 2
CHANNELS = 4
COMPRESSION = 4
TXTLEN = 8
TXTDIM = 16


class _LinearVelocityModel(nn.Module):
    """MMDiT stand-in whose velocity is exactly linear in the timestep ``t``.

    A linear-in-t field has zero second derivative, so the cache's curvature
    verifier should accept the (exact) linear forecast and skip the forward.
    ``calls`` counts how many times the *real* forward ran.
    """

    def __init__(self, base: float = 0.1, slope: float = 0.2):
        super().__init__()
        self.config = types.SimpleNamespace(patch=PATCH, channels=CHANNELS)
        self.calls = 0
        self.base = base
        self.slope = slope

    def forward(self, img, context, t, pos, mask=None):
        self.calls += 1
        tn = float(t.reshape(-1)[0].to(torch.float32))
        return (self.base + self.slope * tn) * torch.ones_like(img)


class _TFuncModel(nn.Module):
    """MMDiT stand-in driven by an arbitrary scalar function of ``t``."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, img, context, t, pos, mask=None):
        tn = float(t.reshape(-1)[0].to(torch.float32))
        return float(self.fn(tn)) * torch.ones_like(img)


class _MockAE:
    compression = COMPRESSION
    channels = CHANNELS

    def decode(self, x):
        b, _c, h, w = x.shape
        return torch.zeros(b, 3, h, w)


class _MockEncoder:
    def __call__(self, prompts):
        n = len(prompts)
        txt = torch.zeros(n, TXTLEN, TXTDIM)
        txtmask = torch.ones(n, TXTLEN, dtype=torch.bool)
        return txt, txtmask


def _drive(cache, ts):
    """Feed a sequence of timesteps through a cache and return the outputs."""
    img = torch.randn(1, 4, CHANNELS * PATCH * PATCH)
    context = torch.zeros(1, TXTLEN, TXTDIM)
    pos = torch.zeros(1, TXTLEN + 4, 3)
    mask = torch.ones(1, TXTLEN + 4, dtype=torch.bool)
    outputs = []
    for tnow in ts:
        t = torch.full((1,), tnow)
        outputs.append(cache(img=img, context=context, t=t, pos=pos, mask=mask))
    return outputs


def test_sample_speca_skips_forwards_on_smooth_field():
    """End-to-end through sampling.sample: speca must cut real forwards."""
    ae, encoder = _MockAE(), _MockEncoder()

    baseline = _LinearVelocityModel()
    sample(
        baseline,
        ae,
        encoder,
        ["a fox in the snow"],
        width=16,
        height=16,
        steps=14,
        guidance=0,
        device="cpu",
        dtype=torch.float32,
        speca=False,
    )
    assert baseline.calls == 14  # no-skip path: one forward per step

    accelerated = _LinearVelocityModel()
    images = sample(
        accelerated,
        ae,
        encoder,
        ["a fox in the snow"],
        width=16,
        height=16,
        steps=14,
        guidance=0,
        device="cpu",
        dtype=torch.float32,
        speca=True,
    )

    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    # The forecast-then-verify path skipped real DiT forwards...
    assert accelerated.calls < baseline.calls
    # ...but still did real work during the cold-start warmup.
    assert accelerated.calls > 0


def test_cache_accepts_forecast_on_smooth_field():
    """Linear velocity -> curvature ~0 -> forecast accepted and exact."""
    model = _TFuncModel(lambda tn: 0.3 + 0.7 * tn)
    cache = SpeculativeVelocityCache(model, tol=0.25, max_skip=2)
    ts = [1.0 - 0.1 * i for i in range(10)]
    outputs = _drive(cache, ts)

    assert cache.total == len(ts)
    assert cache.skipped > 0  # the verifier accepted forecasts
    assert cache.forward_calls < cache.total
    # Linear extrapolation of a linear field is exact: every returned velocity
    # matches the true velocity at that timestep, skipped or not.
    for tnow, out in zip(ts, outputs):
        expected = (0.3 + 0.7 * tnow) * torch.ones_like(out)
        assert torch.allclose(out, expected, atol=1e-5)


def test_cache_skips_less_on_rough_field_than_smooth():
    """The verifier is curvature-sensitive: rougher fields accept fewer forecasts."""
    ts = [1.0 - 0.1 * i for i in range(10)]

    smooth = SpeculativeVelocityCache(_TFuncModel(lambda tn: tn), tol=0.25, max_skip=2)
    _drive(smooth, ts)

    rough = SpeculativeVelocityCache(
        _TFuncModel(lambda tn: math.sin(80.0 * tn)), tol=0.25, max_skip=2
    )
    _drive(rough, ts)

    assert smooth.skipped > 0
    # A high-frequency field has large curvature, so the verifier trusts far
    # fewer forecasts -- and therefore runs the real model more often.
    assert rough.skipped < smooth.skipped
    assert rough.forward_calls > smooth.forward_calls


def test_cache_bounds_consecutive_skips():
    """Even on a perfectly smooth field, max_skip forces periodic refreshes."""
    model = _TFuncModel(lambda tn: 0.3 + 0.7 * tn)
    cache = SpeculativeVelocityCache(model, tol=0.25, max_skip=1)
    _drive(cache, [1.0 - 0.1 * i for i in range(10)])

    # With max_skip=1 we never accept two forecasts back to back: across 10
    # steps there must be at least 5 real evaluations (one refresh every <=2
    # steps), yet still strictly fewer than the no-skip baseline of 10.
    assert cache.forward_calls >= 5
    assert cache.forward_calls < 10
