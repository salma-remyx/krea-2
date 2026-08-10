"""Tests for the cluster-aware token cache wired into ``sampling.sample``.

The integration tests drive the *real* ``sampling.sample`` loop with
lightweight fakes for the model / autoencoder / text encoder (no GPU, no
checkpoints), exercising the cache wiring end to end. Controller-level
behavior is covered by the unit tests at the bottom.
"""

from dataclasses import dataclass

import torch
from PIL import Image

from sampling import sample
from token_cache import (
    ClusterAwareTokenCache,
    TokenCacheConfig,
    noise_relative_magnitude,
)


@dataclass
class _Patch:
    patch: int = 2


class _DecayingModel:
    """Fake MM-DiT.

    Returns a velocity whose magnitude shrinks each call, so the per-token
    noise-relative-magnitude significance decays across steps and pushes the
    cache into its skip path. Counts its forward calls.
    """

    def __init__(self, decay: float = 0.1):
        self.config = _Patch()
        self.decay = decay
        self.calls = 0

    def __call__(self, img, context, t, pos, mask):
        self.calls += 1
        scale = self.decay ** (self.calls - 1)
        return scale * torch.randn_like(img)


class _FakeAE:
    compression = 8
    channels = 16

    def decode(self, x):
        # Project the 16-channel latent to 3 channels and upsample back to
        # pixel resolution so the sampler's post-processing produces an image.
        x = x[:, :3].float()
        return torch.nn.functional.interpolate(
            x, scale_factor=self.compression, mode="nearest"
        )


class _FakeEncoder:
    def __call__(self, prompts):
        n = len(prompts)
        txt = torch.randn(n, 4, 8)
        mask = torch.ones(n, 4, dtype=torch.bool)
        return txt, mask


def test_sample_without_cache_runs_every_step():
    torch.manual_seed(0)
    model = _DecayingModel()
    images = sample(
        model,
        _FakeAE(),
        _FakeEncoder(),
        ["a cat"],
        width=64,
        height=64,
        steps=6,
        guidance=0.0,
        seed=0,
        device="cpu",
        dtype=torch.float32,
    )
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (64, 64)
    # No cache: exactly one forward per denoise step.
    assert model.calls == 6


def test_sample_with_cache_skips_forwards():
    torch.manual_seed(0)
    model = _DecayingModel(decay=0.1)
    cfg = TokenCacheConfig(
        enabled=True, cluster=2, keep_frac=0.5, skip_frac=0.5, max_skips=2
    )
    images = sample(
        model,
        _FakeAE(),
        _FakeEncoder(),
        ["a cat"],
        width=64,
        height=64,
        steps=6,
        guidance=0.0,
        seed=0,
        device="cpu",
        dtype=torch.float32,
        token_cache=cfg,
    )
    assert images[0].size == (64, 64)
    # Significance decays each step, so several forwards are served from cache.
    assert 1 <= model.calls < 6


def test_disabled_cache_is_a_noop():
    torch.manual_seed(0)
    model = _DecayingModel()
    cfg = TokenCacheConfig(enabled=False)  # opt-in: off
    images = sample(
        model,
        _FakeAE(),
        _FakeEncoder(),
        ["a cat"],
        width=64,
        height=64,
        steps=4,
        guidance=0.0,
        seed=0,
        device="cpu",
        dtype=torch.float32,
        token_cache=cfg,
    )
    assert images[0].size == (64, 64)
    assert model.calls == 4


def test_noise_relative_magnitude_is_update_over_latent():
    img = torch.ones(2, 16, 4)
    v = torch.ones(2, 16, 4)
    sig = noise_relative_magnitude(img, v, dt=1.0)
    assert sig.shape == (2, 16)
    # ||update|| = ||v|| = sqrt(4) = 2; ||img|| = 2; ratio == 1 for every token.
    assert torch.allclose(sig, torch.ones(2, 16))


def test_blend_caches_low_significance_tokens():
    b, h, w, d = 1, 4, 4, 4
    cfg = TokenCacheConfig(enabled=True, cluster=2, keep_frac=0.5, skip_frac=0.0)
    cache = ClusterAwareTokenCache(cfg, h, w)
    img = torch.randn(b, h * w, d)
    v1 = torch.randn(b, h * w, d)
    out1 = cache.update(img, v1, dt=1.0)
    assert torch.equal(out1, v1)  # first step: nothing cached yet
    # Second step with a tiny velocity -> low significance -> mostly cached.
    v2 = 0.01 * torch.randn(b, h * w, d)
    out2 = cache.update(img, v2, dt=1.0)
    assert (out2 - v1).norm() < (out2 - v2).norm()


def test_skip_path_reuses_cache_and_caps_streak():
    cfg = TokenCacheConfig(
        enabled=True, cluster=2, keep_frac=0.5, skip_frac=0.9, max_skips=2
    )
    cache = ClusterAwareTokenCache(cfg, 4, 4)
    img = torch.randn(1, 16, 4)
    # Prime the cache: a large-magnitude velocity sets a high peak significance.
    cache.update(img, torch.randn(1, 16, 4) * 10.0, dt=1.0)
    # Tiny follow-up -> near-zero significance -> almost nothing kept -> skip.
    cache.update(img, torch.randn(1, 16, 4) * 0.001, dt=1.0)
    assert cache.will_skip() is True
    reused = cache.reuse()
    assert torch.equal(reused, cache.v_prev)
    assert cache.skips == 1
    assert cache.will_skip() is True  # streak 1 < max_skips
    cache.reuse()  # streak 2
    assert cache.will_skip() is False  # streak >= max_skips forces a forward


def test_unprimed_cache_never_skips():
    cache = ClusterAwareTokenCache(TokenCacheConfig(enabled=True), 4, 4)
    assert cache.will_skip() is False
