"""Tests for training-free attention-head pruning.

These build a *real* ``mmdit.SingleStreamDiT`` (tiny config) and exercise the
pruning through the same ``Attention`` modules the inference path uses, so they
cover the integration, not just the new file in isolation.
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel

import head_pruning
import mmdit
from mmdit import SingleMMDiTConfig, SingleStreamDiT, _mask, ropeapply

# The repo's ``mmdit.attention`` forces the CUDNN SDPA backend, which is the right
# choice on GPU but absent on CPU runners. Swap in a portable backend order
# (CUDNN/flash/efficient first, MATH as the fallback) so these logic tests run
# anywhere. The pruning code under test does not depend on the backend.
_ORIG_ATTENTION = mmdit.attention


def _portable_attention(q, k, v, mask=None, scale=None, gqa=False):
    backends = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    with sdpa_kernel(backends):
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=gqa
        )
    return rearrange(x, "B H L D -> B L (H D)")


mmdit.attention = _portable_attention


def _tiny_config():
    return SingleMMDiTConfig(
        features=128,
        tdim=64,
        txtdim=64,
        heads=8,
        kvheads=4,  # GQA: 2 query heads share each kv head
        multiplier=2,
        layers=3,
        patch=2,
        channels=4,
        theta=1e3,
        txtlayers=2,
        txtheads=4,
        txtkvheads=4,
    )


def _make_model():
    return SingleStreamDiT(_tiny_config()).eval()


def _sample_inputs(model, txtlen=5):
    """Build a valid forward batch: patchified latent + text context + pos/mask.

    Returns the 2D key-padding mask the model expects (it expands it internally).
    """
    latent = torch.randn(1, model.config.channels, 8, 8)  # patch=2 -> 16 img tokens
    patch = model.config.patch
    h, w = latent.shape[2:]
    h_, w_ = h // patch, w // patch
    img = rearrange(latent, "b c (h p) (w q) -> b (h w) (c p q)", p=patch, q=patch)

    context = torch.randn(1, txtlen, model.config.txtlayers, model.config.txtdim)
    imglen = img.shape[1]
    mask1d = torch.ones(1, txtlen + imglen, dtype=torch.bool)

    txtpos = torch.zeros(1, txtlen, 3)
    imgids = torch.zeros(h_, w_, 3)
    imgids[..., 1] = torch.arange(h_)[:, None]
    imgids[..., 2] = torch.arange(w_)[None, :]
    imgpos = rearrange(imgids, "h w d -> 1 (h w) d")
    pos = torch.cat((txtpos, imgpos), dim=1)

    t = torch.full((1,), 0.5, dtype=torch.float32)
    return img, context, t, pos, mask1d


def _model_inputs(model, txtlen=5):
    """Convenience: unpack _sample_inputs to model-forward argument order."""
    img, context, t, pos, mask1d = _sample_inputs(model, txtlen)
    return {"img": img, "context": context, "t": t, "pos": pos, "mask": mask1d}


def test_calibration_scores_shape_and_range():
    model = _make_model()
    inputs = _model_inputs(model)
    scores = head_pruning.score_prompt_attention(model, **inputs)
    assert scores.shape == (3, 8)
    assert torch.isfinite(scores).all()
    assert (scores >= 0).all()


def test_prune_mask_selects_dispensable_heads_and_keeps_one_per_layer():
    model = _make_model()
    inputs = _model_inputs(model)
    scores = head_pruning.score_prompt_attention(model, **inputs)

    mask = head_pruning.build_prune_mask(scores, fraction=0.25)
    assert mask.dtype == torch.bool
    assert mask.shape == scores.shape
    # ~25% of 24 heads => 6 pruned.
    assert int(mask.sum()) == 6
    # At least one head survives in every layer.
    assert (~mask).any(dim=1).all()
    # The single highest-scoring head is among the pruned set (correct direction).
    flat_max = scores.argmax()
    assert mask.view(-1)[flat_max]


def test_pruned_forward_matches_reference_and_skips_heads():
    """The runtime attn rewrite must equal full SDPA with pruned heads zeroed."""
    model = _make_model()
    inputs = _model_inputs(model)
    scores = head_pruning.score_prompt_attention(model, **inputs)
    prune_mask = head_pruning.build_prune_mask(scores, fraction=0.25)

    state = head_pruning.apply_head_pruning(model, prune_mask)
    try:
        attn = model.blocks[0].attn
        layer_mask = prune_mask[0]

        pos = inputs["pos"]
        qkv = torch.randn(1, pos.shape[1], model.config.features)
        freqs = model.posemb(pos)
        attn_mask = _mask(inputs["mask"])  # 4D mask, as the block passes internally

        live = attn(qkv, freqs, attn_mask)

        # Reference: full attention over all heads, then zero the pruned ones.
        q = rearrange(attn.wq(qkv), "B L (H D) -> B H L D", H=attn.heads)
        k = rearrange(attn.wk(qkv), "B L (H D) -> B H L D", H=attn.kvheads)
        v = rearrange(attn.wv(qkv), "B L (H D) -> B H L D", H=attn.kvheads)
        q, k, v = attn.qknorm(q, k, v)
        q, k = ropeapply(q, k, freqs)
        full = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, enable_gqa=attn.gqa
        )
        full = full * (~layer_mask)[None, :, None, None].to(full.dtype)
        ref = attn.wo(rearrange(full, "B H L D -> B L (H D)") * torch.sigmoid(attn.gate(qkv)))

        assert torch.allclose(live, ref, atol=1e-5)
        # Pruned head slots carry no signal through wo (verified by equality above);
        # somewhere in the model at least one head was actually pruned.
        assert prune_mask.any()
    finally:
        state.remove()


def test_apply_changes_output_and_remove_restores_it():
    model = _make_model()
    inputs = _model_inputs(model)

    with torch.no_grad():
        baseline = model(**inputs)

    pruner = head_pruning.HeadPruner(fraction=0.25)
    pruner.calibrate(model, **inputs)
    assert pruner.pruned_fraction > 0.0
    pruner.apply(model)
    try:
        with torch.no_grad():
            pruned = model(**inputs)
        assert torch.isfinite(pruned).all()
        # Pruning is a real change to the computation graph.
        assert not torch.allclose(baseline, pruned, atol=1e-5)
    finally:
        pruner.remove()

    with torch.no_grad():
        restored = model(**inputs)
    assert torch.allclose(baseline, restored, atol=1e-6)
