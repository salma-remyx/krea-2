"""Integration test for the Progressive Seed Pruning wiring.

Imports the (non-new) ``sampling`` module and drives the opt-in ``psp_seeds``
dispatch added to ``sampling.sample``, asserting that PSP front-loads seed
exploration and then prunes at fixed total NFE. Uses lightweight fakes so it
runs without GPU weights.
"""

import torch
from PIL import Image

from progressive_seed_pruning import seeds_for_budget
from sampling import sample  # non-new module: exercises the wiring edit


class _PatchCfg:
    patch = 2


class FakeModel:
    """Mimics the K2 MMDiT call surface; records batch size per forward."""

    def __init__(self):
        self.config = _PatchCfg()
        self.batch_sizes = []

    def __call__(self, img, context, t, pos, mask):
        self.batch_sizes.append(int(img.shape[0]))
        return torch.randn_like(img)


class FakeAE:
    compression = 8
    channels = 4

    def decode(self, x):
        b, _c, h, w = x.shape
        return torch.zeros(b, 3, h, w, dtype=x.dtype)


class FakeEncoder:
    seq = 5
    dim = 8

    def __call__(self, prompts):
        n = len(prompts)
        txt = torch.randn(n, self.seq, self.dim)
        txtmask = torch.ones(n, self.seq, dtype=torch.bool)
        return txt, txtmask


def _run(prompts, **kw):
    torch.manual_seed(0)
    model = FakeModel()
    images = sample(
        model,
        FakeAE(),
        FakeEncoder(),
        prompts,
        device="cpu",
        dtype=torch.float32,
        width=32,
        height=32,
        steps=8,
        guidance=0.0,
        seed=0,
        **kw,
    )
    return model, images


def test_default_path_unchanged_without_psp():
    """Without psp_seeds the existing best-of-N behavior is untouched."""
    _model, images = _run(["a", "b"])
    assert len(images) == 2
    assert all(isinstance(i, Image.Image) for i in images)


def test_psp_front_loads_then_prunes():
    """PSP explores num_seeds candidates early and prunes to `keep` survivors."""
    model, images = _run(["a"], psp_seeds=4, psp_keep=2, psp_prune_at=0.25)

    # one prompt, two survivors -> two images
    assert len(images) == 2

    # 8 steps, prune after 2: batches run 4,4 (explore) then 2 x6 (survivors)
    explore = [b for b in model.batch_sizes if b == 4]
    survive = [b for b in model.batch_sizes if b == 2]
    assert len(explore) == 2          # front-loaded exploration on all seeds
    assert len(survive) == 6          # remaining steps on pruned survivors
    assert max(model.batch_sizes) == 4
    assert model.batch_sizes[-1] == 2


def test_seeds_for_budget_matches_nfe():
    """Wider exploration is compute-matched to a best-of-N baseline."""
    # baseline best-of-4 at 8 steps = 32 evals; PSP must spend the same 32.
    seeds = seeds_for_budget(baseline_n=4, steps=8, prune_at=0.25, keep=2)
    prune_steps = 2
    nfe = seeds * prune_steps + 2 * (8 - prune_steps)
    assert seeds >= 4                 # explores at least as many seeds as baseline
    assert nfe == 4 * 8               # total NFE held fixed
