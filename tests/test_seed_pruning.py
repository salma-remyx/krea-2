"""Integration tests for Progressive Seed Pruning (PSP).

These exercise the real integration: ``seed_pruning.sample_psp`` drives the
existing sampler's primitives (``sampling.prepare`` / ``sampling.timesteps``)
and the repo's black-box ``model(img, context, t, pos, mask)`` contract, and the
``--prune`` flag in ``inference.py`` routes the CLI to it.
"""

import os
import sys

# This repo has a flat top-level module layout (no src/), so make the repo root
# importable regardless of how pytest is invoked.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from sampling import prepare, roundup, timesteps  # noqa: E402
import seed_pruning  # noqa: E402


class _FakeModel:
    """Mimics the K2 forward contract: ``model(img, context, t, pos, mask) -> v``.

    Returns a per-candidate velocity that depends on each candidate's current
    latent mean, so trajectories diverge and pruning scores differ.
    """

    class config:
        patch = 2

    def __init__(self):
        # Count candidate-step evaluations (batch size per forward), which is the
        # quantity PSP conserves -- not the number of batched invocations.
        self.evals = 0

    def __call__(self, img, context, t, pos, mask):
        self.evals += img.shape[0]
        per = img.mean(dim=(1, 2))  # [b]
        v = torch.zeros_like(img)
        v[:, :, 0] = per.unsqueeze(1) * 0.1
        return v


class _FakeAE:
    compression = 8
    channels = 16

    def decode(self, x):  # x: [b, c, h, w] -> pixels [b, 3, h*8, w*8]
        return torch.zeros(x.shape[0], 3, x.shape[2] * 8, x.shape[3] * 8)


class _FakeEncoder:
    def __call__(self, prompts):
        b = len(prompts)
        txt = torch.zeros(b, 8, 1, 32)
        mask = torch.ones(b, 8, dtype=torch.bool)
        return txt, mask


def _expected_calls(seeds, keep, prunes, steps, cfg):
    ckpts = seed_pruning._checkpoints(steps, prunes)
    sched = seed_pruning._schedule(seeds, keep, len(ckpts))
    seg_steps = []
    prev = 0
    for b in ckpts + [steps]:
        seg_steps.append(b - prev)
        prev = b
    return (2 if cfg else 1) * sum(s * ln for s, ln in zip(sched, seg_steps))


def test_module_reuses_existing_sampler_primitives():
    # The new capability integrates by reusing sampling.py's helpers unchanged.
    assert seed_pruning.prepare is prepare
    assert seed_pruning.timesteps is timesteps
    assert seed_pruning.roundup is roundup


def test_psp_prunes_and_conserves_compute_under_cfg():
    model = _FakeModel()
    images = seed_pruning.sample_psp(
        model, _FakeAE(), _FakeEncoder(), ["a fox walking in the snow"],
        seeds=4, keep=1, prunes=2,
        device="cpu", dtype=torch.float32,
        width=256, height=256, steps=6, guidance=4.5, seed=0,
    )
    # One survivor per prompt.
    assert len(images) == 1
    assert images[0].size == (256, 256)
    # Fixed-budget accounting holds exactly (cond + uncond per step under CFG).
    assert model.evals == _expected_calls(4, 1, 2, 6, cfg=True)
    # Pruning saved compute vs fully denoising every seed.
    assert model.evals < 2 * 4 * 6


def test_psp_without_prune_is_best_of_seeds():
    model = _FakeModel()
    images = seed_pruning.sample_psp(
        model, _FakeAE(), _FakeEncoder(), ["a cat"],
        seeds=3, keep=1, prunes=0,
        device="cpu", dtype=torch.float32,
        width=256, height=256, steps=4, guidance=4.5, seed=1,
    )
    # Nothing pruned -> all seeds returned at full compute.
    assert len(images) == 3
    assert model.evals == _expected_calls(3, 1, 0, 4, cfg=True)


def test_psp_runs_without_cfg_via_latent_sharpness_default():
    model = _FakeModel()
    images = seed_pruning.sample_psp(
        model, _FakeAE(), _FakeEncoder(), ["a dog"],
        seeds=4, keep=2, prunes=2,
        device="cpu", dtype=torch.float32,
        width=256, height=256, steps=6, guidance=0.0, seed=2,
    )
    # `keep` survivors, single forward per step (no CFG branch).
    assert len(images) == 2
    assert model.evals == _expected_calls(4, 2, 2, 6, cfg=False)


def test_psp_accepts_pluggable_blackbox_reward():
    model = _FakeModel()
    seen = []

    def reward(x0_hat):
        seen.append(int(x0_hat.shape[0]))
        return x0_hat.flatten(1).std(dim=1)  # 1D score per candidate

    images = seed_pruning.sample_psp(
        model, _FakeAE(), _FakeEncoder(), ["a bird"],
        seeds=4, keep=1, prunes=2, reward=reward,
        device="cpu", dtype=torch.float32,
        width=256, height=256, steps=6, guidance=4.5, seed=3,
    )
    assert len(images) == 1
    # Reward scored the pool at each of the two prune checkpoints (4 -> 2).
    assert seen == [4, 2]


class _DummyImg:
    def save(self, path):
        return None


def test_cli_routes_prune_flag(monkeypatch):
    # Stub the heavy text-encoder dep so importing inference needs no weights.
    if "transformers" not in sys.modules:
        import types

        tf = types.ModuleType("transformers")
        tf.AutoTokenizer = object
        tf.Qwen2TokenizerFast = object
        tf.Qwen3VLForConditionalGeneration = object
        sys.modules["transformers"] = tf

    import inference
    from click.testing import CliRunner

    calls = {"sample": 0, "psp": 0}

    monkeypatch.setattr(inference, "_pipeline", lambda *a, **k: ("m", "a", "e"))
    monkeypatch.setattr(inference, "sample", lambda *a, **k: calls.__setitem__("sample", calls["sample"] + 1) or [_DummyImg()])
    monkeypatch.setattr(inference, "sample_psp", lambda *a, **k: calls.__setitem__("psp", calls["psp"] + 1) or [_DummyImg()])

    runner = CliRunner()

    # Default path -> standard sampler.
    res = runner.invoke(inference.main, ["a prompt"])
    assert res.exit_code == 0, res.output
    assert calls == {"sample": 1, "psp": 0}

    # --prune -> PSP path.
    calls["sample"] = calls["psp"] = 0
    res = runner.invoke(
        inference.main, ["a prompt", "--prune", "--psp-seeds", "4", "--psp-keep", "1"]
    )
    assert res.exit_code == 0, res.output
    assert calls == {"sample": 0, "psp": 1}
