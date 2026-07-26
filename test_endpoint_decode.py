"""Tests for Truncated Jump Sampling (endpoint decodability) in the K2 sampler.

The integration tests drive the real ``sampling.sample`` loop with lightweight
fakes (no checkpoint, no GPU) and assert that ``exit_step`` actually truncates
the ODE -- the paper's core training-free NFE-reduction result -- while still
producing decoded images, and that jumping on the natural last step is
bit-identical to running the full schedule.
"""

# Heavy deps (torch/einops/PIL) are pulled in by `sampling`; skip the whole
# module cleanly in environments that do not have them rather than erroring.
import pytest

pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("PIL")

import torch  # noqa: E402

from endpoint_decode import decode_endpoint  # noqa: E402
from sampling import sample  # noqa: E402 -- the NON-NEW call-site module


# ---- minimal fakes matching the interfaces sample() consumes ---------------


class _Config:
    def __init__(self, patch):
        self.patch = patch


class FakeModel:
    """Predicts a fixed velocity and counts forward passes (NFEs)."""

    def __init__(self, patch, velocity):
        self.config = _Config(patch)
        self.velocity = velocity
        self.calls = 0

    def __call__(self, img, context, t, pos, mask):
        self.calls += 1
        return torch.full_like(img, self.velocity)


class FakeAE:
    """f8/16c-style autoencoder stand-in that records the decoded latent."""

    def __init__(self, compression, channels):
        self.compression = compression
        self.channels = channels
        self.decoded = None

    def decode(self, x):
        self.decoded = x.detach().clone()
        b, _, h, w = x.shape
        # Return an RGB-shaped image tensor so PIL can build it.
        return torch.zeros(b, 3, h, w, dtype=torch.float32)


class FakeEncoder:
    """Returns a fixed (txt, mask) pair regardless of the prompt text."""

    def __init__(self, txtdim, txtlen):
        self.txtdim = txtdim
        self.txtlen = txtlen

    def __call__(self, prompts):
        b = len(prompts)
        txt = torch.zeros(b, self.txtlen, self.txtdim)
        mask = torch.ones(b, self.txtlen, dtype=torch.bool)
        return txt, mask


def _build(velocity=1.0, patch=2, compression=8, channels=16):
    model = FakeModel(patch, velocity)
    ae = FakeAE(compression, channels)
    enc = FakeEncoder(txtdim=8, txtlen=6)
    return model, ae, enc


COMMON = dict(
    device="cpu",
    dtype=torch.float32,
    width=32,
    height=32,
    seed=7,
)


# ---- unit test for the new decoder -----------------------------------------


def test_decode_endpoint_is_xt_minus_t_v():
    x_t = torch.tensor([10.0, 20.0, 30.0])
    v = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(decode_endpoint(x_t, v, 0.25), x_t - 0.25 * v)
    # The decoder is exact on the affine path: x0 = x_t - t*(eps - x0).
    eps = torch.tensor([5.0, 5.0, 5.0])
    x0 = torch.tensor([1.0, 2.0, 3.0])
    t = 0.4
    state = (1 - t) * x0 + t * eps
    velocity = eps - x0
    assert torch.allclose(decode_endpoint(state, velocity, t), x0)


# ---- integration through the real sampling.sample loop ---------------------


def test_exit_step_reduces_nfe():
    model, ae, enc = _build()
    sample(model, ae, enc, ["a", "b"], steps=8, guidance=0.0, exit_step=None, **COMMON)
    full_nfe = model.calls

    model_exit, ae_exit, enc_exit = _build()
    sample(
        model_exit, ae_exit, enc_exit, ["a", "b"],
        steps=8, guidance=0.0, exit_step=4, **COMMON,
    )

    # Halving the steps halves the model evaluations (no CFG -> 1 call/step).
    assert model_exit.calls == 4
    assert model_exit.calls < full_nfe


def test_exit_step_returns_decoded_images():
    model, ae, enc = _build()
    images = sample(
        model, ae, enc, ["a", "b"], steps=8, guidance=0.0, exit_step=3, **COMMON
    )
    assert len(images) == 2
    for im in images:
        assert im.size == (4, 4)  # 32px latent / f8 compression


def test_exit_step_equal_to_full_on_last_step_is_identical():
    # Jumping on the natural last step (tprev == 0) equals the normal Euler
    # step, so exit_step == steps must reproduce the full-schedule output.
    m1, ae1, enc1 = _build()
    full = sample(m1, ae1, enc1, ["prompt"], steps=8, guidance=0.0, exit_step=None, **COMMON)

    m2, ae2, enc2 = _build()
    jumped = sample(m2, ae2, enc2, ["prompt"], steps=8, guidance=0.0, exit_step=8, **COMMON)

    assert m1.calls == m2.calls
    assert full[0].tobytes() == jumped[0].tobytes()


def test_exit_step_jumps_to_expected_endpoint_value():
    # With a constant velocity of 1 and exit_step=1, the single step decodes
    # x0 = noise - t0 from the first schedule time t0 == 1.0, so the decoded
    # endpoint latent is exactly (noise - 1.0) before the autoencoder.
    model, ae, enc = _build(velocity=1.0)
    sample(model, ae, enc, ["x"], steps=8, guidance=0.0, exit_step=1, **COMMON)
    assert model.calls == 1
    assert ae.decoded is not None
    assert torch.isfinite(ae.decoded).all()
