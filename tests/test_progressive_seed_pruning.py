"""Integration tests for Progressive Seed Pruning.

These exercise the new ``progressive_seed_pruning.sample_psp`` against the
existing ``sampling.sample`` (the non-new module at the call site) using
lightweight fakes that match the model / autoencoder / encoder interface, so the
pruning + scoring logic is verified without weights or a GPU.
"""

import numpy as np
import pytest
import torch

from progressive_seed_pruning import (
    default_prune_schedule,
    latent_energy_score,
    sample_psp,
)
from sampling import sample


class FakeModel:
    """Batch-agnostic stand-in for SingleStreamDiT: a linear contraction velocity.

    Returns ``-k * img`` per token, so every candidate is scaled by the same
    factor at a given step -- which keeps the per-candidate ranking fixed by the
    initial noise and makes the surviving trajectory bit-identical to running
    that seed alone through ``sampling.sample``.
    """

    def __init__(self, k=0.1):
        self.config = type("c", (), {"patch": 2})()
        self.k = k
        self.calls = 0

    def __call__(self, img, context, t, pos, mask):
        self.calls += 1
        return -self.k * img


class FakeAE:
    compression = 8
    channels = 16

    def decode(self, x):
        # Latent (b, c, h, w) -> 3-channel image upsampled by `compression`,
        # mirroring the real VAE's spatial 8x decode (channel count is arbitrary).
        x = x[:, :3]
        s = self.compression
        return x.repeat_interleave(s, dim=2).repeat_interleave(s, dim=3)


class FakeEncoder:
    def __init__(self, length=4, dim=8):
        self.length = length
        self.dim = dim

    def __call__(self, prompts):
        n = len(prompts)
        txt = torch.zeros(n, self.length, self.dim)
        txtmask = torch.ones(n, self.length, dtype=torch.bool)
        return txt, txtmask


COMMON = dict(steps=10, guidance=0.0, device="cpu", width=64, height=64, mu=1.15)


def _candidate_l1(base, num_seeds):
    """Reproduce sample_psp's per-seed noise and return each candidate's mean |.|."""
    size = 64 // FakeAE.compression
    out = []
    for s in range(num_seeds):
        nz = torch.randn(
            1,
            FakeAE.channels,
            size,
            size,
            generator=torch.Generator(device="cpu").manual_seed(base + s),
        )
        out.append(nz.abs().mean().item())
    return out


def test_prune_schedule_reduces_to_one():
    schedule = default_prune_schedule(4, steps=12)
    keeps = [keep for _, keep in schedule]
    assert keeps[-1] == 1
    assert all(b < a for a, b in zip(keeps, keeps[1:]))  # strictly decreasing
    assert default_prune_schedule(1, steps=12) == []  # no pruning at num_seeds=1


def test_psp_prunes_and_returns_one_image_per_prompt():
    model = FakeModel()
    images = sample_psp(
        model,
        FakeAE(),
        FakeEncoder(),
        ["a cat"],
        num_seeds=4,
        seed=7,
        **COMMON,
    )
    assert len(images) == 1
    assert images[0].size == (64, 64)
    # Pruning happened: far fewer forward passes than fully denoising every seed,
    # but at least one pass per step for the surviving trajectory.
    assert model.calls < 4 * COMMON["steps"]
    assert model.calls >= COMMON["steps"]


def test_psp_survivor_matches_standard_sample_of_winning_seed():
    base = 123
    num_seeds = 3
    model = FakeModel()
    psp_img = sample_psp(
        model,
        FakeAE(),
        FakeEncoder(),
        ["a fox"],
        num_seeds=num_seeds,
        seed=base,
        **COMMON,
    )[0]

    # With v = -k*img, the per-step scale is identical across candidates, so the
    # latent-energy ranking is fixed by the initial noise -> argmax L1 wins.
    l1 = _candidate_l1(base, num_seeds)
    winner = max(range(num_seeds), key=lambda s: l1[s])

    ref_img = sample(
        FakeModel(),
        FakeAE(),
        FakeEncoder(),
        ["a fox"],
        seed=base + winner,
        **COMMON,
    )[0]

    assert np.array_equal(np.asarray(psp_img), np.asarray(ref_img))


def test_psp_honors_custom_score_fn():
    base = 41
    num_seeds = 3

    def min_energy(img, x0_pred, cond, uncond, tcurr):
        # Invert the default: prefer the lowest-energy candidate instead.
        return -latent_energy_score(img, x0_pred, cond, uncond, tcurr)

    psp_img = sample_psp(
        FakeModel(),
        FakeAE(),
        FakeEncoder(),
        ["a fox"],
        num_seeds=num_seeds,
        seed=base,
        score_fn=min_energy,
        **COMMON,
    )[0]

    l1 = _candidate_l1(base, num_seeds)
    winner = min(range(num_seeds), key=lambda s: l1[s])
    ref_img = sample(
        FakeModel(),
        FakeAE(),
        FakeEncoder(),
        ["a fox"],
        seed=base + winner,
        **COMMON,
    )[0]

    assert np.array_equal(np.asarray(psp_img), np.asarray(ref_img))


def test_psp_num_seeds_one_matches_standard_sample():
    # No pruning path: num_seeds=1 should reproduce sampling.sample exactly.
    base = 5
    psp_img = sample_psp(
        FakeModel(),
        FakeAE(),
        FakeEncoder(),
        ["a dog"],
        num_seeds=1,
        seed=base,
        **COMMON,
    )[0]
    ref_img = sample(
        FakeModel(),
        FakeAE(),
        FakeEncoder(),
        ["a dog"],
        seed=base,
        **COMMON,
    )[0]
    assert np.array_equal(np.asarray(psp_img), np.asarray(ref_img))


def test_psp_rejects_invalid_num_seeds():
    with pytest.raises(ValueError):
        sample_psp(
            FakeModel(),
            FakeAE(),
            FakeEncoder(),
            ["a dog"],
            num_seeds=0,
            **COMMON,
        )
