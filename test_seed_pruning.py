"""Tests for progressive seed pruning.

These exercise the integrated path: ``sample_pruned`` drives the real
``sampling`` helpers (``prepare`` / ``timesteps``) and is compared against
``sampling.sample`` for matched-compute equivalence. Fakes satisfy the same
``model(img=..., context=..., t=..., pos=..., mask=...) -> v`` /
``encoder(prompts) -> (txt, txtmask)`` / ``ae.decode`` contract the real
sampler uses, so no checkpoint is loaded.
"""

import torch

from sampling import sample
from seed_pruning import nfe_budget, sample_pruned


class _FakeConfig:
    patch = 2


class _CountingModel:
    """Identity-ish flow that also tallies per-image NFEs (batch size per call)."""

    def __init__(self):
        self.config = _FakeConfig()
        self.nfe = 0

    def __call__(self, img, context, t, pos, mask):
        self.nfe += img.shape[0]
        return 0.1 * img


class _FakeAE:
    compression = 4
    channels = 4

    def decode(self, x):
        b, _, h, w = x.shape
        return torch.zeros(b, 3, h, w, dtype=torch.float32)


class _FakeEncoder:
    def __call__(self, prompts):
        n = len(prompts)
        return torch.randn(n, 4, 8), torch.ones(n, 4, dtype=torch.bool)


def _make():
    return _CountingModel(), _FakeAE(), _FakeEncoder()


KW = dict(device="cpu", dtype=torch.float32, guidance=0.0, width=16, height=16)


def test_returns_final_survivors_per_prompt():
    model, ae, encoder = _make()
    images = sample_pruned(
        model, ae, encoder, ["a fox"], num_seeds=6, keep=[3, 1], steps=6, **KW
    )
    assert len(images) == 1  # keep[-1] = 1 image per prompt

    model, ae, encoder = _make()
    images = sample_pruned(
        model, ae, encoder, ["a fox", "a dog"], num_seeds=4, keep=[2, 1], steps=6, **KW
    )
    assert len(images) == 2  # one survivor per prompt


def test_nfe_budget_honored_and_smaller_than_run_all():
    model, ae, encoder = _make()
    sample_pruned(model, ae, encoder, ["p"], num_seeds=6, keep=[3, 1], steps=6, **KW)
    # Phase 0: 6 candidates x 3 steps; phase 1: 3 candidates x 3 steps -> 27.
    assert model.nfe == nfe_budget(6, [3, 1], 6) == 27
    # Front-loading prunes, so this is cheaper than denoising all 6 fully.
    assert model.nfe < 6 * 6


def test_budget_matches_best_of_n_via_sampling():
    # best-of-N: N independent full trajectories through the real sampler.
    model, ae, encoder = _make()
    sample(model, ae, encoder, ["p"] * 4, steps=6, **KW)
    best_of_n = model.nfe  # 4 images x 6 steps = 24 per-image NFEs
    assert best_of_n == 24
    # PSP explores 6 seeds (>4), pruning to 2 then 1, at the same 24 NFEs.
    assert nfe_budget(6, [2, 1], 6) == best_of_n


def test_single_candidate_degenerates_to_plain_sampler():
    model, ae, encoder = _make()
    images = sample_pruned(
        model, ae, encoder, ["p"], num_seeds=1, keep=[1], steps=6, **KW
    )
    assert len(images) == 1
    assert model.nfe == 6  # no pruning overhead, plain 6-step Euler


def test_rejects_invalid_schedule():
    model, ae, encoder = _make()
    for bad in ([6, 1], [0], []):
        try:
            sample_pruned(model, ae, encoder, ["p"], num_seeds=4, keep=bad, steps=6, **KW)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for keep={bad}")
