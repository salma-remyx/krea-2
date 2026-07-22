"""Tests for Truncated Jump Sampling wired into sampling.sample().

These exercise the existing ``sampling`` module (the call site) with lightweight
fakes so no checkpoints or GPU are required. The key property: on an exact
affine flow path the endpoint decoder ``x0 = x_t - t * v`` recovers the clean
sample at *any* truncation point, so TJS must reproduce the full-schedule
output while spending far fewer neural function evaluations.
"""

from types import SimpleNamespace

import numpy as np
import torch
from einops import rearrange

from endpoint_decoding import decode_endpoint
from sampling import sample


class _FakeModel:
    """MMDiT stand-in: returns a constant flow velocity and counts NFEs."""

    def __init__(self, velocity, patch=2):
        self._v = velocity
        self.calls = 0
        self.config = SimpleNamespace(patch=patch)

    def __call__(self, img, context, t, pos, mask):
        self.calls += 1
        return self._v


class _FakeAE:
    """QwenAutoencoder stand-in: stashes the latent and emulates its output shape.

    The real AE maps the 16-channel latent to a 3-channel pixel image at 8x
    resolution; we reduce to a single luminance plane, expand to RGB and
    nearest-upsample so ``Image.fromarray`` gets a valid HxWx3 buffer. The exact
    pixels don't matter — only that identical latents yield identical images.
    """

    compression = 8
    channels = 16

    def __init__(self):
        self.last_latent = None

    def decode(self, x):
        self.last_latent = x
        x = x.float()
        gray = x.mean(dim=1, keepdim=True)
        rgb = gray.expand(-1, 3, -1, -1).contiguous()
        return torch.nn.functional.interpolate(
            rgb, scale_factor=self.compression, mode="nearest"
        )


class _FakeEncoder:
    """Text-encoder stand-in returning a tiny fixed conditioning tensor."""

    def __call__(self, prompts):
        n = len(prompts)
        txt = torch.zeros(n, 4, 8)
        txtmask = torch.ones(n, 4, dtype=torch.bool)
        return txt, txtmask


def test_decode_endpoint_recovers_clean_sample():
    # Pure check of the core formula on the affine path x_t = (1-t) x0 + t x1.
    x0 = torch.randn(2, 3, 4, 4)
    x1 = torch.randn(2, 3, 4, 4)
    t = 0.37
    xt = (1 - t) * x0 + t * x1
    velocity = x1 - x0  # path velocity
    assert torch.allclose(decode_endpoint(xt, velocity, t), x0, atol=1e-6)


def test_tjs_matches_full_output_with_fewer_nfe():
    # Reproduce the per-seed latent noise sample() starts from, then build the
    # exact affine-path velocity (v = noise - target) in patchified-token space.
    noise = torch.randn(
        1, 16, 32, 32, device="cpu", dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(0),
    )
    x1 = rearrange(noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    x0 = torch.full_like(x1, 0.3)
    velocity = x1 - x0

    kwargs = dict(
        device="cpu", dtype=torch.float32, width=256, height=256,
        steps=10, guidance=0.0, seed=0,
    )

    # Full schedule: 10 Euler segments -> 10 model evaluations.
    full_model, full_ae = _FakeModel(velocity), _FakeAE()
    full_imgs = sample(full_model, full_ae, _FakeEncoder(),
                       ["p"], tjs_early_exit=None, **kwargs)
    assert full_model.calls == 10
    assert len(full_imgs) == 1

    # Truncated Jump Sampling: 3 segments + a final decoded jump -> 3 evals.
    tjs_model, tjs_ae = _FakeModel(velocity), _FakeAE()
    tjs_imgs = sample(tjs_model, tjs_ae, _FakeEncoder(), ["p"],
                      tjs_early_exit=3, **kwargs)
    assert tjs_model.calls == 3

    # NFE is cut from 10 to 3 (~70% reduction) ...
    assert tjs_model.calls < full_model.calls
    # ... yet the decoded clean endpoint matches the full-schedule result.
    assert torch.allclose(
        tjs_ae.last_latent.float(), full_ae.last_latent.float(), atol=1e-2
    )
    full_px = np.asarray(full_imgs[0]).astype(int)
    tjs_px = np.asarray(tjs_imgs[0]).astype(int)
    assert np.abs(full_px - tjs_px).max() <= 1


def test_tjs_disabled_by_default_matches_full():
    # tjs_early_exit=None must leave the sampler byte-identical to the original.
    noise = torch.randn(
        1, 16, 32, 32, device="cpu", dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(7),
    )
    x1 = rearrange(noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    x0 = torch.full_like(x1, -0.2)
    velocity = x1 - x0

    base = _FakeModel(velocity)
    imgs_none = sample(base, _FakeAE(), _FakeEncoder(), ["p"], tjs_early_exit=None,
                       device="cpu", dtype=torch.float32, width=256, height=256,
                       steps=8, guidance=0.0, seed=7)
    assert base.calls == 8
    assert imgs_none[0].size == (256, 256)
