"""Tests for the sampler, including the mean-direction (trapezoidal) path.

These go through the public ``sampling.sample`` entry point with lightweight
stand-ins for the MMDiT / autoencoder / text encoder, so they exercise the
wiring in the existing call-site module (``sampling``) rather than the new
file in isolation.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from sampling import sample


class _Config:
    patch = 2


class _Model:
    config = _Config()

    def __call__(self, img, context, t, pos, mask):
        # Velocity depends on x and t, so the mean-direction predictor point
        # yields a different velocity than the start point (the two integrators
        # must therefore diverge).
        return -img + t[:1, None] * 0.01


class _AE:
    compression = 4
    channels = 4

    def decode(self, x):
        # Upsample the latent channels back to pixel space (input-dependent so
        # different latents produce different pixels).
        up = x[:, :3]
        return up.repeat_interleave(self.compression, dim=2).repeat_interleave(
            self.compression, dim=3
        )


class _Encoder:
    def __call__(self, prompts):
        n = len(prompts)
        return torch.zeros(n, 8, 16), torch.ones(n, 8, dtype=torch.bool)


@pytest.fixture()
def pipeline():
    return _Model(), _AE(), _Encoder()


def _sample(pipeline, method="euler", prompts=("a cat", "a dog"), steps=4):
    model, ae, encoder = pipeline
    return sample(
        model,
        ae,
        encoder,
        list(prompts),
        steps=steps,
        guidance=3.5,
        width=64,
        height=64,
        seed=0,
        device="cpu",
        dtype=torch.float32,
        method=method,
    )


def test_euler_returns_images(pipeline):
    out = _sample(pipeline, "euler")
    assert len(out) == 2
    assert isinstance(out[0], Image.Image)
    assert out[0].size == (64, 64)


def test_mean_direction_returns_images(pipeline):
    out = _sample(pipeline, "mean_direction")
    assert len(out) == 2
    assert isinstance(out[0], Image.Image)
    assert out[0].size == (64, 64)


def test_mean_direction_differs_from_euler(pipeline):
    """The mean-direction path must actually run a different integrator."""
    euler = np.asarray(_sample(pipeline, "euler")[0])
    md = np.asarray(_sample(pipeline, "mean_direction")[0])
    assert euler.shape == md.shape == (64, 64, 3)
    assert not np.array_equal(euler, md)


def test_default_matches_euler(pipeline):
    """Omitting method defaults to the existing first-order behavior."""
    model, ae, encoder = pipeline
    kwargs = dict(
        width=64, height=64, seed=0, device="cpu", dtype=torch.float32,
        steps=3, guidance=3.5,
    )
    default = sample(model, ae, encoder, ["p"], **kwargs)
    euler = sample(model, ae, encoder, ["p"], method="euler", **kwargs)
    assert np.array_equal(np.asarray(default[0]), np.asarray(euler[0]))


def test_mean_direction_step_is_trapezoidal():
    """With a constant velocity, mean-direction == euler (predictor==corrector)."""
    from mean_direction_sampler import mean_direction_step

    x = torch.randn(2, 5, 4)
    const = torch.randn(2, 5, 4)
    velocity = lambda val, t: const  # noqa: E731
    out = mean_direction_step(velocity, x, 1.0, 0.5)
    assert torch.allclose(out, x + (0.5 - 1.0) * const)
