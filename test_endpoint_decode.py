"""Tests for Truncated Jump Sampling wiring in ``sampling.sample`` and the
endpoint-decode algebra in ``endpoint_decode``.

The integration tests import the *existing* ``sampling`` module and drive its
public ``sample`` entry point with lightweight fakes, exercising the
``early_exit`` branch added at the call site (NFE reduction + still-decoded
output). The unit tests pin the paper's core identity directly on tensors.
"""

import torch
from PIL import Image

import endpoint_decode
import sampling


# --- lightweight fakes so sample() runs on CPU without real weights ---------


class FakeModel:
    """Counts forward calls and returns a zero velocity matching ``img``."""

    class config:
        patch = 2

    def __init__(self):
        self.calls = 0

    def __call__(self, **kw):
        self.calls += 1
        return torch.zeros_like(kw["img"])


class FakeAE:
    compression = 8
    channels = 4

    def decode(self, x):
        return x


class FakeEncoder:
    def __call__(self, prompts):
        n = len(prompts)
        txt = torch.zeros(n, 4, 8)
        txtmask = torch.ones(n, 4, dtype=torch.bool)
        return txt, txtmask


def _run(early_exit, steps=8, guidance=4.5):
    model = FakeModel()
    images = sampling.sample(
        model,
        FakeAE(),
        FakeEncoder(),
        ["a cat"],
        device="cpu",
        dtype=torch.float32,
        width=16,
        height=16,
        steps=steps,
        guidance=guidance,
        seed=0,
        early_exit=early_exit,
    )
    return model, images


# --- integration: the call-site edit in sampling.sample --------------------


def test_full_path_unchanged_without_early_exit():
    """early_exit=None keeps the default Euler path: steps * (CFG branches)."""
    model, images = _run(None, steps=8, guidance=4.5)
    assert model.calls == 8 * 2  # 8 steps, cond + uncond per step
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


def test_early_exit_reduces_nfe_and_still_decodes():
    """early_exit truncates the ODE: fewer model calls, still a decoded image."""
    full_model, _ = _run(None, steps=8, guidance=4.5)
    exit_model, images = _run(0.5, steps=8, guidance=4.5)

    # Core TJS result: the trajectory is truncated, so NFEs drop.
    assert exit_model.calls < full_model.calls
    # The exit step itself still evaluates the velocity before decoding.
    assert exit_model.calls >= 2
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)


def test_early_exit_exits_early_without_cfg():
    """Truncation also holds on the no-CFG (Turbo-style) path."""
    full_model, _ = _run(None, steps=8, guidance=0.0)
    exit_model, images = _run(0.5, steps=8, guidance=0.0)
    assert exit_model.calls < full_model.calls
    assert isinstance(images[0], Image.Image)


# --- unit: the endpoint-decodability algebra (the paper's core identity) ----


def test_decode_endpoint_recovers_clean_sample_at_any_t():
    """On the affine path x_t = x_0 + t*v, the decode x_0 = x_t - t*v is exact.

    This is why early exit works without straightening the trajectory: the
    velocity prediction is an x-prediction at every point along the path.
    """
    x0 = torch.randn(2, 5, 8)
    v = torch.randn(2, 5, 8)
    for t in (1.0, 0.5, 0.1):
        x_t = x0 + t * v
        decoded = endpoint_decode.decode_endpoint(x_t, v, t)
        assert torch.allclose(decoded, x0, atol=1e-6)


def test_early_exit_timestep_picks_first_clearing_index():
    ts = [1.0, 0.8, 0.5, 0.2, 0.0]
    assert endpoint_decode.early_exit_timestep(ts, None) is None
    assert endpoint_decode.early_exit_timestep(ts, 0.5) == 2
    assert endpoint_decode.early_exit_timestep(ts, 0.9) == 1
    assert endpoint_decode.early_exit_timestep(ts, 2.0) == 0
    assert endpoint_decode.early_exit_timestep(ts, -0.1) is None
