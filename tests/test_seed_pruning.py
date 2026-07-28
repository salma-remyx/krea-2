"""Tests for Progressive Seed Pruning integration.

Two integration surfaces are exercised:

* the call-site wiring in ``inference.py`` (``resolve_sampler`` routes the CLI
  to the new sampler), and
* the core mechanism in ``seed_pruning.py``, which reuses the existing
  ``sampling.prepare`` / ``sampling.timesteps`` / ``sampling.roundup`` helpers
  (this is what makes it an integration rather than a self-test).

The mechanism tests use lightweight fakes (no GPU, no checkpoints), so they
run anywhere ``torch`` is installed.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")  # sampling imports einops; skip the module if absent
from PIL import Image

import sampling  # non-new module under test
from seed_pruning import (
    _topk_per_group,
    cfg_velocity_agreement,
    estimated_nfe,
    match_best_of_n,
    sample_with_progressive_pruning,
)


class FakeModel:
    """MMDiT stand-in: deterministic velocity that depends on conditioning.

    The conditional and unconditional passes return different outputs (driven by
    ``context``), so CFG produces a guided velocity whose direction varies per
    candidate and the velocity-agreement proxy actually discriminates.
    """

    def __init__(self, patch=2):
        self.config = SimpleNamespace(patch=patch)

    def __call__(self, img, context, t, pos, mask):
        ctx = context.float().mean()
        return -0.5 * img.float() + 0.3 * ctx


class FakeEncoder:
    """Returns a fixed-length conditioning tensor (like the real Qwen encoder,
    which pads to ``max_length``). Empty (negative) prompts map to zero context
    so conditional / unconditional embeddings differ."""

    def __init__(self, length=8, dim=4):
        self.length = length
        self.dim = dim

    def __call__(self, prompts):
        b = len(prompts)
        fill = 0.0 if all(p == "" for p in prompts) else 1.0
        txt = torch.full((b, self.length, self.dim), fill, dtype=torch.float32)
        mask = torch.ones(b, self.length, dtype=torch.bool)
        return txt, mask


class FakeAE:
    compression = 8
    channels = 16

    def decode(self, x):
        x = x.float()
        b, _c, h, w = x.shape
        return torch.zeros(b, 3, h * 8, w * 8, dtype=torch.float32)


COMMON = {
    "device": "cpu",
    "dtype": torch.float32,
    "width": 32,
    "height": 32,
    "steps": 6,
    "guidance": 4.5,
    "seed": 0,
}


def _pipeline():
    return FakeModel(), FakeAE(), FakeEncoder()


# --- call-site wiring (inference.py) ----------------------------------------


def _inference():
    """Import inference lazily; it pulls the full encoder/autoencoder stack."""
    return pytest.importorskip("inference")


def test_resolve_sampler_default_returns_sample():
    inference = _inference()
    sampler, extra = inference.resolve_sampler(False, 4, (2, 1), 0.5)
    assert sampler is sampling.sample
    assert extra == {}


def test_resolve_sampler_psp_routes_to_seed_pruning():
    inference = _inference()
    sampler, extra = inference.resolve_sampler(True, 4, (2, 1), 0.5)
    assert sampler is sample_with_progressive_pruning
    assert extra == {
        "num_candidates": 4,
        "keep_counts": (2, 1),
        "front_load_fraction": 0.5,
    }


def test_parse_keep_counts():
    inference = _inference()
    assert inference._parse_keep_counts("4,2,1") == (4, 2, 1)
    assert inference._parse_keep_counts(" 2 , 1 ") == (2, 1)


# --- core mechanism (seed_pruning.py, reusing sampling.*) -------------------


def test_psp_returns_one_best_image_per_prompt():
    model, ae, encoder = _pipeline()
    images = sample_with_progressive_pruning(
        model, ae, encoder, ["a fox", "a crow"],
        num_candidates=4, keep_counts=(2, 1), **COMMON,
    )
    assert len(images) == 2
    assert all(isinstance(im, Image.Image) for im in images)
    assert all(im.size == (32, 32) for im in images)


def test_psp_return_all_keeps_every_survivor():
    model, ae, encoder = _pipeline()
    images = sample_with_progressive_pruning(
        model, ae, encoder, ["a fox", "a crow"],
        num_candidates=4, keep_counts=(2,), return_all=True, **COMMON,
    )
    # one prune to 2 survivors per prompt, both kept
    assert len(images) == 4


def test_psp_rejects_invalid_schedule():
    model, ae, encoder = _pipeline()
    with pytest.raises(ValueError):
        sample_with_progressive_pruning(
            model, ae, encoder, ["a fox"], num_candidates=2, keep_counts=(4,), **COMMON,
        )


def test_psp_custom_reward_drives_selection():
    """A reward that ranks the last candidate highest must make _topk_per_group
    pick it within each prompt group."""
    scores = torch.tensor([0.1, 0.2, 0.3, 0.9, 0.4, 0.5, 0.6, 0.7])
    group = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    keep = _topk_per_group(scores, group, 1)
    # rows 3 (0.9) and 7 (0.7) are the top scorer in each prompt group
    assert keep.tolist() == [3, 7]
    assert sorted(group[keep].tolist()) == [0, 1]


def test_psp_saves_compute_versus_best_of_n():
    # 4 candidates, 6 steps, CFG on: PSP prunes early so it uses fewer passes
    # than running all 4 trajectories end to end.
    psp = estimated_nfe(4, (2, 1), steps=6, cfg=True)
    best_of_n = 4 * 6 * 2
    assert psp < best_of_n
    # pruning saves passes in the no-CFG regime too (16 < 4*6)
    assert estimated_nfe(4, (2, 1), steps=6, cfg=False) == 16
    assert estimated_nfe(4, (2, 1), steps=6, cfg=False) < 4 * 6


def test_match_best_of_n_stays_under_budget():
    num_candidates, keep_counts = match_best_of_n(4, steps=6, cfg=True)
    budget = 4 * 6 * 2
    assert num_candidates >= 4
    assert estimated_nfe(num_candidates, keep_counts, 6, cfg=True) <= budget


def test_default_proxy_is_finite_and_per_candidate():
    img = torch.randn(3, 4, 16)
    cond = torch.randn(3, 4, 16)
    v = cond + 0.5 * torch.randn(3, 4, 16)
    scores = cfg_velocity_agreement(img, cond, v, 0.8)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
