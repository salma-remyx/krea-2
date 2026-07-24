"""Tests for endpoint decodability + Truncated Jump Sampling early exit.

The integration test exercises the wiring in ``sampling.sample`` with lightweight
fakes (no real MMDiT / autoencoder), proving the early-exit path is reachable
and reduces NFEs. Torch-dependent tests skip when a dependency is unavailable.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("PIL")

import torch  # noqa: E402

from endpoint_decode import decode_endpoint, early_exit_index  # noqa: E402
from sampling import sample  # noqa: E402


class _Config:
    patch = 2


class _FakeModel:
    """Counts neural function evaluations and returns a fixed velocity."""

    def __init__(self):
        self.config = _Config()
        self.calls = 0

    def __call__(self, img, context, t, pos, mask):
        self.calls += 1
        return torch.full_like(img, 0.25)


class _FakeAE:
    compression = 8
    channels = 4

    def decode(self, img):
        b, _, h, w = img.shape
        return torch.zeros(b, 3, h, w)


class _FakeEncoder:
    def __call__(self, prompts):
        n = len(prompts)
        txtlen = 5
        txt = torch.zeros(n, txtlen, 16)
        txtmask = torch.ones(n, txtlen, dtype=torch.bool)
        return txt, txtmask


def _run(**kwargs):
    """Run ``sample`` with fakes and CFG disabled (one model call per step)."""
    model = _FakeModel()
    images = sample(
        model,
        _FakeAE(),
        _FakeEncoder(),
        ["a test prompt"],
        device="cpu",
        dtype=torch.float32,
        width=32,
        height=32,
        steps=12,
        guidance=0.0,
        seed=0,
        **kwargs,
    )
    return model, images


def test_decode_endpoint_math():
    img = torch.tensor([[1.0, 2.0]])
    v = torch.tensor([[0.5, -1.0]])
    # x0 = x_t - t * v at t = 0.4
    assert torch.allclose(decode_endpoint(img, 0.4, v), img - 0.4 * v)


def test_early_exit_index_disabled():
    ts = [1.0, 0.75, 0.5, 0.25, 0.0]
    assert early_exit_index(ts, None) is None


def test_early_exit_index_picks_first_at_or_below_threshold():
    ts = [1.0, 0.75, 0.5, 0.25, 0.0]
    # leading times are ts[:-1] = [1.0, 0.75, 0.5, 0.25]; first <= 0.5 is index 2
    assert early_exit_index(ts, 0.5) == 2
    # nothing <= 0.1 in the leading times -> no truncation possible
    assert early_exit_index(ts, 0.1) is None


def test_sample_full_integration_nfe():
    model, images = _run()
    assert model.calls == 12  # one model call per step, CFG disabled
    assert len(images) == 1 and images[0].size == (4, 4)


def test_sample_tjs_reduces_nfe_and_returns_image():
    full_model, _ = _run()
    tjs_model, tjs_images = _run(early_exit=0.5)
    # Truncated Jump Sampling calls the model strictly fewer times...
    assert tjs_model.calls < full_model.calls
    # ...while still decoding a valid image.
    assert len(tjs_images) == 1 and tjs_images[0].size == (4, 4)
