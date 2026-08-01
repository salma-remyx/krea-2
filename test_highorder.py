"""Tests for the opt-in AMED mean-direction sampler.

The integration test goes through the public ``sampling.sample`` entry
point (a non-new module) with lightweight fakes, so it exercises the
real wiring: prepare -> timesteps -> velocity closure -> solver dispatch
-> unpatchify -> decode. The unit test pins the exact AMED update math.
"""

import math

import torch
from PIL import Image

from highorder import denoise_amed
from sampling import sample


class _FakeConfig:
    patch = 2


class FakeModel:
    """Counts forward calls; returns a zero velocity so the ODE is trivial."""

    config = _FakeConfig()

    def __init__(self):
        self.calls = 0

    def __call__(self, img, context, t, pos, mask):
        self.calls += 1
        return torch.zeros_like(img)


class FakeAE:
    compression = 8
    channels = 16

    def decode(self, x):
        return torch.zeros((x.shape[0], 3, 4, 4), dtype=x.dtype)


class FakeEncoder:
    def __call__(self, prompts):
        n = len(prompts)
        txt = torch.zeros((n, 4, 8))
        txtmask = torch.ones((n, 4), dtype=torch.bool)
        return txt, txtmask


COMMON = dict(
    model=FakeModel(),
    ae=FakeAE(),
    encoder=FakeEncoder(),
    prompts=["a fox walking in the snow"],
    device="cpu",
    steps=4,
    width=32,
    height=32,
    seed=0,
)


def test_amed_runs_end_to_end_through_sample():
    """`sampler="amed"` produces images and is wired into the public path."""
    images = sample(guidance=4.5, sampler="amed", **COMMON)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


def test_amed_costs_two_evals_per_step_vs_euler():
    """AMED is the second-order path: 2 velocity evals/step vs Euler's 1.

    With CFG each eval is two model calls, so over N steps AMED makes 4N
    calls and Euler makes 2N -> AMED == 2 * Euler. Pins that the dispatch
    actually reaches the higher-order integrator rather than falling back.
    """
    model = COMMON["model"]
    model.calls = 0
    sample(guidance=4.5, sampler="euler", **COMMON)
    euler_calls = model.calls

    model.calls = 0
    sample(guidance=4.5, sampler="amed", **COMMON)
    amed_calls = model.calls

    assert euler_calls == 2 * COMMON["steps"]  # 2 calls/step (cfg)
    assert amed_calls == 2 * euler_calls


def test_denoise_amed_update_matches_eq9():
    """Pin the AMED Eq. 9 update for an x-independent velocity v(x, t) = t.

    One step tcurr=1 -> tprev=0.5, r=0.5: s = sqrt(1*0.5), and since v
    does not depend on x, vs = s, so x' = x + (tprev - tcurr) * s.
    """
    img = torch.zeros(1, 3)

    def velocity(x, t):
        return torch.full_like(x, float(t))

    out = denoise_amed(velocity, img, [1.0, 0.5], r=0.5)
    expected = (0.5 - 1.0) * math.sqrt(0.5)
    assert torch.allclose(out, torch.full_like(img, expected), atol=1e-6)


def test_denoise_amed_handles_zero_endpoint():
    """The last schedule node is t=0; the solver must not hit a 0**-power."""
    img = torch.zeros(2, 4)

    def velocity(x, t):
        return torch.full_like(x, 0.1)

    out = denoise_amed(velocity, img, [1.0, 0.5, 0.0])
    assert torch.isfinite(out).all()
    assert out.shape == img.shape
