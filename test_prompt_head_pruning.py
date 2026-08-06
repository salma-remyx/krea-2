"""Tests for prompt_head_pruning.

These exercise the runtime patch against the repo's real ``mmdit`` module:
``mmdit.Attention`` instances are patched and run, and a tiny
``mmdit.SingleStreamDiT`` is wired/unwired at the model level. The capability is
imported from a NEW file but every object it patches lives in the NON-NEW
``mmdit`` module, so this proves the integration rather than self-testing.
"""

import logging

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

import mmdit
import prompt_head_pruning as php

# The repo's mmdit.attention pins the CUDNN SDPA backend (GPU-only). CPU tests
# swap in a backend-agnostic implementation; numerically identical to the repo.
logging.getLogger("torch").setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def _cpu_sdpa(monkeypatch):
    def _attention(q, k, v, mask=None, scale=None, gqa=False):
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
        return rearrange(x, "B H L D -> B L (H D)")

    monkeypatch.setattr(mmdit, "attention", _attention)


def test_prompt_attention_mass_ranks_prompt_reading_head():
    # Head 0 aligns image queries with text keys (reads the prompt); head 1 aligns
    # image queries with image keys (visual synthesis). The scorer must rank 0 > 1.
    b, h, l, d = 1, 2, 6, 8
    text_len = 3
    q = torch.zeros(b, h, l, d)
    k = torch.zeros(b, h, l, d)
    q[:, 0, text_len:, :] = 3.0
    k[:, 0, :text_len, :] = 3.0  # head 0: image-q <-> text-k
    q[:, 1, text_len:, :] = 3.0
    k[:, 1, text_len:, :] = 3.0  # head 1: image-q <-> image-k

    mass = php.prompt_attention_mass(q, k, text_len)

    assert mass.shape == (h,)
    assert mass[0] > 0.5  # head 0 puts most mass on text keys
    assert mass[1] < mass[0]  # head 1 attends less to the prompt


def test_select_pruned_groups_drops_highest_mass_and_keeps_one():
    # heads=8, kvheads=4 (group=2). Group masses: g0 high, g1 low, g2 high, g3 low.
    mass = torch.tensor([0.9, 0.8, 0.05, 0.05, 0.85, 0.8, 0.1, 0.1])
    # group_mass = [0.85, 0.05, 0.825, 0.1] -> top-2 dispensable = g0, g2.
    kept = php.select_pruned_groups(mass, heads=8, kvheads=4, ratio=0.5)

    assert kept.shape == (4,)
    assert kept.dtype == torch.bool
    assert kept.tolist() == [False, True, False, True]  # g0, g2 pruned
    assert kept.any()  # always keeps at least one group

    # ratio=0 keeps everything; ratio=1 still keeps one group.
    assert php.select_pruned_groups(mass, 8, 4, ratio=0.0).all()
    assert php.select_pruned_groups(mass, 8, 4, ratio=1.0).sum().item() == 1


def _reference_with_heads_zeroed(module, qkv, keep_q):
    """Full attention output but with the ``keep_q=False`` query heads zeroed.

    The patched path slices kept heads, runs SDPA on them, then scatters back
    with zeros for pruned heads - so its output must equal this reference.
    """
    q, k, v, gate = module.wq(qkv), module.wk(qkv), module.wv(qkv), module.gate(qkv)
    q = rearrange(q, "B L (H D) -> B H L D", H=module.heads)
    k = rearrange(k, "B L (H D) -> B H L D", H=module.kvheads)
    v = rearrange(v, "B L (H D) -> B H L D", H=module.kvheads)
    q, k, v = module.qknorm(q, k, v)
    x = F.scaled_dot_product_attention(q, k, v, enable_gqa=module.gqa)
    x = x * keep_q.to(x.dtype).view(1, -1, 1, 1)
    merged = rearrange(x, "B H L D -> B L (H D)")
    return module.wo(merged * F.sigmoid(gate))


def test_apply_kept_mask_preserves_shape_and_is_noop_when_unpruned():
    torch.manual_seed(0)
    heads, kvheads, dim = 8, 4, 64
    attn = mmdit.Attention(dim=dim, heads=heads, kvheads=kvheads)
    qkv = torch.randn(1, 10, dim)
    out_full = attn(qkv)

    # Keeping every group is a no-op: patched path falls through to mmdit.attention.
    all_kept = torch.ones(kvheads, dtype=torch.bool)
    php.apply_kept_mask(attn, all_kept)
    assert php.is_patched(attn)
    out_noop = attn(qkv)
    assert out_noop.shape == out_full.shape
    assert torch.allclose(out_noop, out_full, atol=1e-5)


def test_apply_kept_mask_prunes_and_matches_zeroed_reference():
    torch.manual_seed(1)
    heads, kvheads, dim = 8, 2, 64  # group=4
    attn = mmdit.Attention(dim=dim, heads=heads, kvheads=kvheads)
    qkv = torch.randn(1, 10, dim)

    # Prune kv group 1 (query heads 4..7) -> genuine SDPA FLOP reduction.
    kept = torch.tensor([True, False])
    php.apply_kept_mask(attn, kept)

    out = attn(qkv)
    assert out.shape == (1, 10, dim)  # weight shapes / I/O unchanged

    keep_q = kept.repeat_interleave(heads // kvheads)  # (heads,)
    ref = _reference_with_heads_zeroed(attn, qkv, keep_q)
    assert torch.allclose(out, ref, atol=1e-5)

    # Pruning actually changes the output (heads were not inert).
    attn2 = mmdit.Attention(dim=dim, heads=heads, kvheads=kvheads)
    attn2.load_state_dict(attn.state_dict())
    assert not torch.allclose(out, attn2(qkv), atol=1e-4)


def test_model_level_apply_and_remove_over_joint_modules():
    config = mmdit.SingleMMDiTConfig(
        features=128,
        tdim=16,
        txtdim=32,
        heads=8,
        kvheads=2,
        multiplier=2,
        layers=1,
        patch=2,
        channels=4,
        txtheads=4,
        txtkvheads=4,
        txtlayers=2,
    )
    dit = mmdit.SingleStreamDiT(config)
    joint = php.joint_attention_modules(dit)
    assert len(joint) == config.layers  # one joint attn per single-stream block
    assert all(isinstance(m, mmdit.Attention) for m in joint)

    kept = {m: torch.tensor([True, False]) for m in joint}
    php.apply_prompt_head_pruning(dit, kept)
    assert all(php.is_patched(m) for m in joint)
    assert all(getattr(m, php._KEPT_ATTR).tolist() == [True, False] for m in joint)

    php.remove_prompt_head_pruning(dit)
    assert all(not php.is_patched(m) for m in joint)


def test_prune_prompt_heads_calibrates_then_applies_and_restores():
    config = mmdit.SingleMMDiTConfig(
        features=128,
        tdim=16,
        txtdim=32,
        heads=8,
        kvheads=2,
        multiplier=2,
        layers=1,
        patch=2,
        channels=4,
        txtheads=4,
        txtkvheads=4,
        txtlayers=2,
    )
    dit = mmdit.SingleStreamDiT(config)
    joint = php.joint_attention_modules(dit)
    text_len = 4

    def run_forward():
        # Deterministic joint sequence (4 text + 8 image tokens, text first) so
        # calibration is reproducible across the calibrate/prune calls below.
        torch.manual_seed(123)
        for block in dit.blocks:
            x = torch.randn(1, 12, config.features)
            block.attn(x, mask=None)

    # Calibrate -> each module gets a per-head prompt-attention mass vector.
    masses = php.calibrate_prompt_attention(dit, run_forward, text_len)
    assert set(masses) == set(joint)
    assert all(v.shape == (config.heads,) for v in masses.values())

    # prune_prompt_heads calibrates again (deterministic) and applies the masks.
    kept = php.prune_prompt_heads(
        dit, ratio=0.5, run_forward=run_forward, text_len=text_len
    )
    assert set(kept) == set(joint)
    assert all(php.is_patched(m) for m in joint)
    for module, mask in kept.items():
        assert getattr(module, php._KEPT_ATTR).tolist() == mask.tolist()
        assert int(mask.sum()) == 1  # ratio=0.5 over 2 groups keeps one

    # Context manager applies on enter and always restores on exit.
    with php.prompt_head_pruning(
        dit, ratio=0.5, run_forward=run_forward, text_len=text_len
    ):
        assert all(php.is_patched(m) for m in joint)
    assert all(not php.is_patched(m) for m in joint)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
