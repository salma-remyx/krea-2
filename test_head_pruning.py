"""Tests for the training-free head-pruning integration.

These exercise the *call site* (``mmdit.Attention.forward``) and the calibration
in ``head_pruning.py``. The production ``mmdit.attention`` pins the cuDNN SDPA
backend (GPU-only); on CPU we swap in a backend-agnostic stand-in so the wiring
is testable without a GPU. The pruning logic under test is independent of the
kernel choice.
"""

import torch
import torch._dynamo
import torch.nn.functional as F
from einops import rearrange

import head_pruning as hp
import mmdit

# mmdit compiles several submodules with fullgraph=True; the tiny test model
# exercises many RMSNorm feature-sizes, each specializing the compiled fn. The
# real model has stable shapes, but raise the limits so the unit model compiles.
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.cache_size_limit = 256

# Minimal SingleStreamDiT config (small enough to run on CPU; headdim must be >=
# 16 so the position-axis split in SingleStreamDiT is well defined).
CONFIG = mmdit.SingleMMDiTConfig(
    features=64,
    tdim=32,
    txtdim=32,
    heads=4,
    kvheads=2,  # GQA: exercises head-group replication in the score
    multiplier=2,
    layers=2,
    patch=2,
    channels=4,
    txtlayers=2,
    txtheads=2,
    txtkvheads=2,
)


def _cpu_attention(q, k, v, mask=None, scale=None, gqa=False):
    """Backend-agnostic stand-in for the GPU-pinned ``mmdit.attention``."""
    x = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
    )
    return rearrange(x, "B H L D -> B L (H D)")


def _make_inputs(txtlen=4, imglen=4):
    hw = int(imglen**0.5)
    imgids = torch.zeros(hw, hw, 3)
    imgids[..., 1] = torch.arange(hw)[:, None]
    imgids[..., 2] = torch.arange(hw)[None, :]
    imgpos = imgids.view(imglen, 3)
    pos = torch.cat([torch.zeros(txtlen, 3), imgpos], 0).unsqueeze(0)
    mask = torch.ones(1, txtlen + imglen, dtype=torch.bool)
    img = torch.randn(1, imglen, CONFIG.channels * CONFIG.patch**2)
    context = torch.randn(1, txtlen, CONFIG.txtlayers, CONFIG.txtdim)
    t = torch.tensor([0.5])
    return {"img": img, "context": context, "t": t, "pos": pos, "mask": mask}


def test_apply_head_mask_zeros_only_pruned_slices():
    heads, headdim = 4, 3
    out = torch.randn(2, 5, heads * headdim)
    # pruning all heads -> fully zero
    assert torch.equal(
        hp.apply_head_mask(out, torch.zeros(heads), heads), torch.zeros_like(out)
    )
    # keeping all heads -> unchanged
    assert torch.equal(hp.apply_head_mask(out, torch.ones(heads), heads), out)
    # pruning head 1 only zeros its slice, leaves the rest
    mask = torch.tensor([1.0, 0.0, 1.0, 1.0])
    got = hp.apply_head_mask(out, mask, heads)
    reshaped = out.reshape(2, 5, heads, headdim)
    expected = reshaped.clone()
    expected[:, :, 1, :] = 0.0
    assert torch.equal(got, expected.reshape(2, 5, heads * headdim))


def test_attention_forward_all_pruned_is_zero(monkeypatch):
    """Zeroing every head must zero the output (wo has no bias -> wo(0)=0)."""
    monkeypatch.setattr(mmdit, "attention", _cpu_attention)
    torch.manual_seed(0)
    attn = mmdit.Attention(dim=CONFIG.features, heads=CONFIG.heads, kvheads=CONFIG.kvheads)
    qkv = torch.randn(1, 8, CONFIG.features)
    full = attn(qkv)
    assert full.abs().sum() > 0
    hp.set_head_mask(attn, torch.zeros(CONFIG.heads))
    assert attn._head_pruning_active is True
    zeroed = attn(qkv)
    assert torch.allclose(zeroed, torch.zeros_like(zeroed), atol=1e-6)


def test_attention_default_path_is_noop(monkeypatch):
    """All-ones mask must reproduce the unmodified forward exactly."""
    monkeypatch.setattr(mmdit, "attention", _cpu_attention)
    torch.manual_seed(1)
    attn = mmdit.Attention(dim=CONFIG.features, heads=CONFIG.heads, kvheads=CONFIG.kvheads)
    qkv = torch.randn(1, 8, CONFIG.features)
    baseline = attn(qkv)
    hp.set_head_mask(attn, torch.ones(CONFIG.heads))  # active stays False (nothing pruned)
    assert attn._head_pruning_active is False
    assert torch.equal(attn(qkv), baseline)


def test_head_text_attention_score_ranks_text_attenders():
    torch.manual_seed(2)
    txtlen, imglen, dh = 2, 4, 4
    hq = 2
    q = torch.randn(1, hq, txtlen + imglen, dh)
    k = torch.randn(1, hq, txtlen + imglen, dh)
    # head 0: image queries align with text key 0 -> attends to text
    q[:, 0, txtlen:, :] = k[:, 0, 0:1, :]
    # head 1: image queries align with an image key -> attends away from text
    q[:, 1, txtlen:, :] = k[:, 1, 3:4, :]
    score = hp.head_text_attention_score(q, k, mask=None, txtlen=txtlen)
    assert score[0] > score[1]
    assert score.shape[0] == hq


def test_select_prune_masks_global_vs_per_layer():
    scores = [torch.tensor([0.9, 0.1]), torch.tensor([0.8, 0.2])]
    glob = hp._select_prune_masks(scores, 0.5, "global")  # prune top-2: {.9, .8}
    assert [int((m == 0).sum()) for m in glob] == [1, 1]
    assert glob[0].tolist() == [0.0, 1.0]  # .9 pruned, .1 kept
    assert glob[1].tolist() == [0.0, 1.0]  # .8 pruned, .2 kept
    per = hp._select_prune_masks(scores, 0.5, "per_layer")  # top-1 per layer
    assert per[0].tolist() == [0.0, 1.0]
    assert per[1].tolist() == [0.0, 1.0]


def test_calibrate_end_to_end_through_dit(monkeypatch):
    monkeypatch.setattr(mmdit, "attention", _cpu_attention)
    torch.manual_seed(3)
    model = mmdit.SingleStreamDiT(CONFIG).eval()
    inputs = _make_inputs(txtlen=4, imglen=4)

    baseline = model(**inputs)
    # before calibration: no pruning, nothing captured
    assert hp.pruned_head_fraction(model) == 0.0

    result = hp.calibrate_head_masks(model, [inputs], prune_ratio=0.5, scope="global")
    total = CONFIG.layers * CONFIG.heads
    assert result["total_heads"] == total
    assert result["pruned_heads"] == 4
    assert abs(result["realized_ratio"] - 0.5) < 1e-9
    assert hp.pruned_head_fraction(model) == 0.5

    # paper claim: pruned heads are exactly the highest-text-attention heads
    all_scores = torch.cat(result["per_layer_scores"])
    keep = torch.cat(result["per_layer_masks"]).bool()
    pruned_scores = all_scores[~keep]
    kept_scores = all_scores[keep]
    assert pruned_scores.min() >= kept_scores.max()

    # pruning is active and changes the output
    assert model.blocks[0].attn._head_pruning_active is True
    pruned_out = model(**inputs)
    assert not torch.allclose(pruned_out, baseline, atol=1e-6)

    # captures are cleaned up after calibration
    assert model.blocks[0].attn._capture is None

    # clearing restores the original (un-pruned) behavior
    hp.clear_head_masks(model)
    assert hp.pruned_head_fraction(model) == 0.0
    assert torch.allclose(model(**inputs), baseline, atol=1e-6)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
