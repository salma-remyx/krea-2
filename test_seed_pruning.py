"""Integration tests for the Progressive Seed Pruning wiring.

Imports the *existing* ``sampling`` module (the call site) and drives the PSP
dispatch added to ``sample`` with fake model / autoencoder / conditioner, so it
runs on CPU without the real Krea 2 weights. The fakes mimic the real
contracts: the model exposes ``.config.patch`` and a batched forward, the AE
decodes a latent to an RGB image at the compression factor, and the encoder
returns ``(hidden states, mask)``.
"""

import torch
import torch.nn.functional as F
from PIL import Image

from sampling import sample  # existing, non-new module under test
from seed_pruning import nfe, prune_steps


class _Config:
    patch = 2


class _Model(torch.nn.Module):
    """Constant-velocity flow (v = -x); records backbone batch sizes."""

    def __init__(self):
        super().__init__()
        self.config = _Config()
        self.calls = []

    def forward(self, img, context, t, pos, mask):
        self.calls.append(img.shape[0])
        return -img


class _AE(torch.nn.Module):
    compression = 8
    channels = 16

    def decode(self, x):
        # Latent (B, C, h, w) -> RGB image (B, 3, h*8, w*8), like the VAE.
        return F.interpolate(x[:, :3].float(), scale_factor=self.compression, mode="nearest")


class _Encoder(torch.nn.Module):
    def forward(self, texts):
        b = len(texts)
        return torch.zeros(b, 8, 32), torch.ones(b, 8, dtype=torch.bool)


def _reward_spy(bucket):
    def _reward(x0_latents, decode):
        bucket.append(len(x0_latents))
        return torch.arange(len(x0_latents), dtype=torch.float32)

    return _reward


def test_plain_sample_unchanged():
    """Default path (no PSP) still works after the cfg_velocity refactor."""
    model = _Model()
    images = sample(
        model, _AE(), _Encoder(), ["a cat"],
        steps=4, guidance=0, width=32, height=32,
        device="cpu", dtype=torch.float32,
    )
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert model.calls == [1, 1, 1, 1]  # batch 1 every step, no pruning


def test_psp_dispatch_prunes_progressively():
    """The dispatch route narrows the pool across phases at fixed NFE."""
    model = _Model()
    seen = []

    images = sample(
        model, _AE(), _Encoder(), ["a cat"],
        num_candidates=4, prune_to=[2, 1], reward=_reward_spy(seen),
        steps=6, guidance=0, width=32, height=32,
        device="cpu", dtype=torch.float32,
    )

    assert len(images) == 1  # prune_to[-1] == 1 survivor
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (32, 32)
    # Backbone batch sizes shrink each phase: 4,4 | 2,2 | 1,1.
    assert model.calls == [4, 4, 2, 2, 1, 1]
    # Reward scored the pre-prune pool at each checkpoint: 4 then 2.
    assert seen == [4, 2]


def test_psp_default_reward_multi_prompt():
    """The parameter-free default reward runs end-to-end over multiple prompts."""
    images = sample(
        _Model(), _AE(), _Encoder(), ["a cat", "a dog"],
        num_candidates=4, prune_to=[2, 1], steps=6, guidance=0,
        width=32, height=32, device="cpu", dtype=torch.float32,
    )
    assert len(images) == 2  # 2 prompts x 1 survivor
    assert all(isinstance(im, Image.Image) for im in images)


def test_psp_nfe_accounting():
    assert prune_steps(6, [2, 1]) == [1, 3]
    # 4,4,2,2,1,1 backbone calls == 14 cond evaluations.
    assert nfe(6, 4, [2, 1]) == 14
    # Same survivor count as best-of-4 but fewer full denoises -> cheaper; widen
    # the pool to match a best-of-N budget rather than assert a fixed number.
    assert nfe(6, 4, [2, 1]) < 4 * 6
