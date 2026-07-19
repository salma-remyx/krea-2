"""Tests for the stability-guided merge hook, wired through sampling.sample.

These exercise the *integration* (sampling.py -> stability_merge) through the
public ``sample`` interface with tiny stand-ins for the model / autoencoder /
text encoder, so they run on CPU without the multi-GB K2 checkpoints.
"""

from types import SimpleNamespace

import torch
from PIL import Image

from sampling import sample
from stability_merge import StabilityMerge, build_clusters


PATCH = 2
CHANNELS = 16
COMP = 16
DIM = CHANNELS * PATCH * PATCH  # patch-feature width the model sees / returns


class DummyModel:
    """Stand-in for SingleStreamDiT: records the seq lens it was called with."""

    def __init__(self):
        self.config = SimpleNamespace(patch=PATCH, channels=CHANNELS)
        self.seen = []

    def __call__(self, img, context, t, pos, mask):
        self.seen.append(img.shape[1])
        return 0.05 * img  # small velocity -> stable trajectory -> merging engages


class DummyAE:
    compression = COMP
    channels = CHANNELS

    def decode(self, latent):
        b, c, h, w = latent.shape
        return torch.zeros(b, 3, h * self.compression, w * self.compression).to(
            latent.dtype
        )


class DummyEncoder:
    def __call__(self, texts):
        b = len(texts)
        length = 5
        return torch.randn(b, length, DIM), torch.ones(b, length, dtype=torch.bool)


def test_build_clusters_shape_and_assignment():
    img = torch.randn(2, 16, DIM)

    # ratio 0.5 merges half the tokens: 8 distinct clusters, ids in [0, 8).
    cid, k = build_clusters(img, 0.5)
    assert k == 8
    assert cid.unique().numel() == 8
    assert int(cid.max()) == 7

    # ratio 0 is a length-preserving permutation (no tokens merged away).
    cid0, k0 = build_clusters(img, 0.0)
    assert k0 == 16
    assert cid0.unique().numel() == 16


def test_sample_default_path_unchanged():
    """Without the hook, sample() behaves exactly as before (regression)."""
    model = DummyModel()
    images = sample(
        model,
        DummyAE(),
        DummyEncoder(),
        ["a red cube on a blue table"],
        steps=4,
        guidance=4.5,
        width=128,
        height=128,
        device="cpu",
        dtype=torch.float32,
    )
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (128, 128)
    # every model call sees the full 16-token image grid
    assert min(model.seen) == 16


def test_sample_with_merge_reduces_sequence_and_returns_image():
    """The wired hook actually shortens the model's input and still decodes."""
    model = DummyModel()
    merge = StabilityMerge(max_ratio=0.6, warmup=0.0)
    images = sample(
        model,
        DummyAE(),
        DummyEncoder(),
        ["a red cube on a blue table"],
        steps=4,
        guidance=4.5,
        width=128,
        height=128,
        device="cpu",
        dtype=torch.float32,
        merge=merge,
    )
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (128, 128)
    # at least one model call ran on a compressed sequence
    assert min(model.seen) < 16, f"merge never engaged; seen={model.seen}"


def test_sample_with_merge_no_cfg():
    """Hook also works with CFG disabled (guidance=0)."""
    model = DummyModel()
    merge = StabilityMerge(max_ratio=0.6, warmup=0.0)
    images = sample(
        model,
        DummyAE(),
        DummyEncoder(),
        ["a red cube"],
        steps=4,
        guidance=0.0,
        width=128,
        height=128,
        device="cpu",
        dtype=torch.float32,
        merge=merge,
    )
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (128, 128)
