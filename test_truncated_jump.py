"""Tests for Truncated Jump Sampling.

These exercise the integration with the repo's real sampler machinery:
``sampling`` (the existing module) supplies ``prepare`` / ``timesteps`` /
``roundup`` and the model/CFG/decode contract that ``truncated_jump.sample``
is built on, while fakes stand in for the heavy MMDiT / autoencoder / text
encoder so the pipeline runs on CPU without checkpoints.
"""

import types

import torch

import sampling  # existing, NON-NEW module under test
from truncated_jump import endpoint_decode, sample as tjs_sample


def _build(patch=2, compression=8, channels=4, seq=16, dim=8, velocity=0.1):
    """Return (model, ae, encoder) fakes matching the sampling.sample contract.

    The fake MMDiT returns a constant nonzero velocity so the trajectory
    actually moves (a zero velocity would leave the latent unchanged and hide
    the jump). A constant velocity is also exactly affine, which is the
    idealized case the paper analyzes: endpoint decoding recovers the true
    endpoint losslessly.
    """

    class FakeMMDiT:
        def __init__(self):
            self.config = types.SimpleNamespace(patch=patch)
            self.calls = 0

        def __call__(self, *, img, context, t, pos, mask):
            self.calls += 1
            return torch.full_like(img, velocity)

    class FakeAE:
        def __init__(self):
            self.compression = compression
            self.channels = channels
            self.last_latent = None

        def decode(self, x):
            self.last_latent = x.detach().clone()
            return x[:, : min(3, x.shape[1])].to(torch.float32)

    class FakeEncoder:
        def __call__(self, prompts):
            b = len(prompts)
            txt = torch.zeros(b, seq, dim)
            txtmask = torch.ones(b, seq, dtype=torch.bool)
            return txt, txtmask

    return FakeMMDiT(), FakeAE(), FakeEncoder()


def test_endpoint_decode_recovers_clean_sample():
    """The decoder x_0 = x_t - t*v recovers x_0 on an affine flow path."""
    torch.manual_seed(0)
    b, d = 4, 6
    x0 = torch.randn(b, d)
    eps = torch.randn(b, d)
    for t in (0.1, 0.37, 0.5, 0.83, 0.99):
        x_t = (1.0 - t) * x0 + t * eps  # affine path: noise at t=1, data at t=0
        v = eps - x0  # rectified-flow velocity target
        assert torch.allclose(endpoint_decode(x_t, v, t), x0, atol=1e-6), t
        # per-sample tensor `t` must broadcast to the same result
        t_tensor = torch.full((b,), t)
        assert torch.allclose(endpoint_decode(x_t, v, t_tensor), x0, atol=1e-6), t


def test_sample_full_mode_matches_original_sampler():
    """With tjs_exit=1.0 the new sampler is a bit-for-bit drop-in for sampling.sample."""
    model_a, ae_a, enc = _build()
    model_b, ae_b, _ = _build()

    kwargs = dict(
        prompts=["a fox in snow"],
        width=16,
        height=16,
        steps=8,
        guidance=4.5,  # exercise the CFG branch
        seed=7,
        device="cpu",
        dtype=torch.float32,
    )
    sampling.sample(model_a, ae_a, enc, **kwargs)
    tjs_sample(model_b, ae_b, enc, **kwargs, tjs_exit=1.0)

    assert torch.equal(ae_a.last_latent, ae_b.last_latent)


def test_truncated_jump_cuts_nfe():
    """tjs_exit<1 stops early and jumps, so fewer model evaluations are taken."""
    model_full, ae_full, enc = _build()
    model_trunc, ae_trunc, _ = _build()

    common = dict(
        prompts=["a fox in snow"],
        width=16,
        height=16,
        steps=8,
        guidance=0.0,  # CFG off -> 1 eval per step, easy to count
        seed=3,
        device="cpu",
        dtype=torch.float32,
    )
    imgs_full = tjs_sample(model_full, ae_full, enc, **common, tjs_exit=1.0)
    imgs_trunc = tjs_sample(model_trunc, ae_trunc, enc, **common, tjs_exit=0.5)

    # Full ODE: 8 steps * 1 eval. Truncated: 4 Euler steps + 1 jump eval.
    assert model_full.calls == 8
    assert model_trunc.calls == 5
    assert model_trunc.calls < model_full.calls
    assert len(imgs_full) == len(imgs_trunc) == 1
    assert imgs_trunc[0].size == imgs_full[0].size


def test_exit_fraction_clamps_to_valid_range():
    """tjs_exit<=0 floors at one Euler step + jump; tjs_exit>=1 runs the full ODE."""
    model_lo, ae_lo, enc = _build()
    model_hi, ae_hi, _ = _build()

    common = dict(
        prompts=["a fox in snow"],
        width=16,
        height=16,
        steps=8,
        guidance=0.0,
        seed=3,
        device="cpu",
        dtype=torch.float32,
    )
    tjs_sample(model_lo, ae_lo, enc, **common, tjs_exit=0.0)
    tjs_sample(model_hi, ae_hi, enc, **common, tjs_exit=1.5)

    # exit_index clamps to [1, steps]: 1 Euler + 1 jump = 2 vs full 8.
    assert model_lo.calls == 2
    assert model_hi.calls == 8


def test_truncated_jump_endpoint_equals_full_integration():
    """On the true affine velocity field, stop-then-decode equals integrate-to-zero.

    This is the idealized case the paper's endpoint-decodability analysis
    targets: with the exact affine velocity, decoding the endpoint at any
    early-exit t* recovers the same x0 as integrating the full ODE -- lossless
    acceleration. (Real diffusion velocities carry a non-affine residual, so
    empirically quality is *near*-matched rather than exact; that needs the real
    checkpoint and is out of scope for this unit-level check.) Run in float32
    with a plain uniform schedule to avoid decode/bf16 quantization noise.
    """
    torch.manual_seed(0)
    x0 = torch.randn(3, 8)
    eps = torch.randn(3, 8)
    steps = 16
    ts = torch.linspace(1, 0, steps + 1).tolist()

    def true_velocity():
        return eps - x0  # exact affine velocity, independent of state/time

    # Full Euler integration to t = 0.
    img_full = eps.clone()
    for tcurr, tprev in zip(ts[:-1], ts[1:]):
        img_full = img_full + (tprev - tcurr) * true_velocity()

    # Truncated: integrate only to t* = ts[exit], then endpoint-decode.
    exit_index = steps // 2
    img_trunc = eps.clone()
    for tcurr, tprev in zip(ts[:exit_index], ts[1 : exit_index + 1]):
        img_trunc = img_trunc + (tprev - tcurr) * true_velocity()
    t_star = ts[exit_index]
    img_trunc = endpoint_decode(img_trunc, true_velocity(), t_star)

    assert torch.allclose(img_full, img_trunc, atol=1e-5)
    # Both recover the true clean sample x0.
    assert torch.allclose(img_full, x0, atol=1e-5)
