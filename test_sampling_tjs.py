"""Integration test for Truncated Jump Sampling wired into ``sampling.sample``.

Skipped entirely when torch is not installed (it is a heavy GPU dependency and
not needed for the pure unit tests in ``test_jump_sampling``). Imports the
NON-NEW ``sampling`` module and exercises the ``tjs`` hook end-to-end with
lightweight fakes for the MMDiT, autoencoder, and text encoder.
"""

import pytest

torch = pytest.importorskip("torch")  # skip this module without torch

import sampling  # noqa: E402


class _FakeConfig:
    patch = 2


class _CountingModel:
    """Stand-in MMDiT: records NFE and returns a zero velocity field."""

    def __init__(self):
        self.calls = 0
        self.config = _FakeConfig()

    def __call__(self, img, context, t, pos, mask):
        self.calls += 1
        return torch.zeros_like(img)


class _FakeAE:
    compression = 8
    channels = 16

    def decode(self, x):
        return torch.zeros(x.shape[0], 3, 8, 8)


def _fake_encoder(prompts):
    n = len(prompts)
    txt = torch.zeros(n, 4, 16)
    txtmask = torch.ones(n, 4, dtype=torch.bool)
    return txt, txtmask


def _run(tjs):
    model = _CountingModel()
    sampling.sample(
        model,
        _FakeAE(),
        _fake_encoder,
        ["a fox"],
        steps=28,
        guidance=4.5,
        width=64,
        height=64,
        seed=0,
        tjs=tjs,
    )
    return model


def test_sample_tjs_halves_nfe():
    # CFG doubles every step (cond + uncond), so 28 steps == 56 calls.
    full = _run(tjs=None)
    assert full.calls == 56

    # tjs=0.5 keeps 14 steps -> 28 calls, exactly half the NFE budget.
    half = _run(tjs=0.5)
    assert half.calls == 28


def test_sample_default_is_full_rollout():
    # With tjs disabled, sample must behave exactly as before (no truncation).
    assert _run(tjs=None).calls == 56
