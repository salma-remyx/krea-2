"""Integration tests for the Progressive Seed Pruning (PSP) wiring.

These import from the *existing* ``sampling`` module (the call site) and drive
both the baseline path and the ``prune=`` hook with lightweight fakes, so no GPU
or checkpoints are required.
"""

import torch

from sampling import sample
from seed_pruning import latent_structure_score, sample_psp


class _FakeConfig:
    patch = 2


class FakeModel:
    config = _FakeConfig()

    def __call__(self, img, context, t, pos, mask):
        # Zero velocity keeps each trajectory at its seed noise, so the
        # predicted-clean latent differs per seed and pruning is non-degenerate.
        return torch.zeros_like(img)


class FakeAE:
    compression = 8
    channels = 16

    def decode(self, x):
        b, _, h, w = x.shape
        return torch.zeros(b, 3, h * self.compression, w * self.compression)


class FakeEncoder:
    seq = 4
    dim = 8

    def __call__(self, prompts):
        b = len(prompts)
        txt = torch.zeros(b, self.seq, self.dim)
        mask = torch.ones(b, self.seq, dtype=torch.bool)
        return txt, mask


COMMON = {
    "width": 64,
    "height": 64,
    "steps": 4,
    "guidance": 0.0,
    "device": "cpu",
    "dtype": torch.float32,
}


def test_baseline_sample_unchanged_by_hook():
    """prune omitted -> the original sampler contract still holds."""
    imgs = sample(FakeModel(), FakeAE(), FakeEncoder(), ["a cat"], **COMMON)
    assert len(imgs) == 1
    assert imgs[0].size == (64, 64)


def test_prune_returns_requested_count():
    """prune=4 explores 4 seeds and prunes back down to 1 image."""
    imgs = sample(FakeModel(), FakeAE(), FakeEncoder(), ["a cat"], prune=4, **COMMON)
    assert len(imgs) == 1
    assert imgs[0].size == (64, 64)


def test_prune_multi_image():
    """n>1: 2 prompts x explore=3 = 6 candidates pruned to 2 survivors."""
    imgs = sample(
        FakeModel(),
        FakeAE(),
        FakeEncoder(),
        ["a cat"] * 2,
        prune=3,
        width=64,
        height=64,
        steps=6,
        guidance=0.0,
        device="cpu",
        dtype=torch.float32,
    )
    assert len(imgs) == 2


def test_prune_with_cfg_slices_uncond_branch():
    """CFG path: the uncond tensors are sliced alongside the survivors."""
    imgs = sample(
        FakeModel(),
        FakeAE(),
        FakeEncoder(),
        ["a dog"],
        prune=2,
        width=64,
        height=64,
        steps=4,
        guidance=3.5,
        device="cpu",
        dtype=torch.float32,
    )
    assert len(imgs) == 1


def test_reward_callback_is_honored():
    """A user-supplied reward is invoked on the shrinking candidate sets."""
    calls = []

    def reward(x0):
        calls.append(x0.shape[0])
        return x0[:, 0, 0]

    imgs = sample_psp(
        FakeModel(),
        FakeAE(),
        FakeEncoder(),
        ["a cat"],
        explore=4,
        reward=reward,
        **COMMON,
    )
    assert len(imgs) == 1
    assert calls  # reward actually drove the pruning
    assert max(calls) >= 2  # at least one prune started from a multi-candidate set


def test_latent_structure_score_is_per_candidate_scalar():
    x0 = torch.randn(3, 16, 64)  # 3 candidates, 4x4 tokens, 16*2*2 channels
    scores = latent_structure_score(x0, patch=2, h=4, w=4)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
