"""Integration tests for tuned-schedule sampling (schedule_search + sampling.sample)."""

import json

import numpy as np
import pytest
import torch
from PIL import Image

from sampling import sample, timesteps
from schedule_search import ScheduleSpace, load_schedule, save_schedule, tune


class FakeAE:
    """Smallest autoencoder that satisfies sampling.sample's interface."""

    compression = 8
    channels = 4

    def decode(self, latents):
        b, c, h, w = latents.shape
        return latents.sum(dim=1, keepdim=True).expand(b, 3, h, w).contiguous()


class FakeEncoder:
    """Fixed-length conditioning so the token count never varies mid-search."""

    def __call__(self, prompts):
        n = len(prompts)
        return torch.zeros(n, 7, 16), torch.ones(n, 7, dtype=torch.bool)


class FakeDiT(torch.nn.Module):
    """Constant-velocity field: x_{t+1} = x_t + (t_prev - t) * v is exact."""

    def __init__(self):
        super().__init__()
        self.saw = []
        self.config = torch.nn.ParameterDict({})
        self.config.patch = 2

    def forward(self, *, img, context, t, pos, mask):
        self.saw.append(t[0].item())
        return torch.ones_like(img)


@pytest.fixture
def dit():
    return FakeDiT()


def test_sample_honors_explicit_schedule(dit):
    """sample() must walk exactly the timesteps passed via `schedule`."""
    schedule = [1.0, 0.9, 0.5, 0.2, 0.0]
    images = sample(
        dit,
        FakeAE(),
        FakeEncoder(),
        ["a fox"],
        steps=len(schedule) - 1,
        guidance=0.0,
        device="cpu",
        schedule=schedule,
    )
    # sample() casts t to the latent dtype (bfloat16), so compare at that
    # precision rather than exactly.
    assert dit.saw == pytest.approx(schedule[:-1], abs=2e-3)
    assert isinstance(images[0], Image.Image)


def test_sample_default_schedule_unchanged(dit):
    """With no schedule, sample() still derives the grid from steps/mu."""
    sample(
        dit,
        FakeAE(),
        FakeEncoder(),
        ["a fox"],
        steps=5,
        guidance=0.0,
        mu=1.15,
        device="cpu",
    )
    expected = timesteps(1024, 5, 256, 6400, y1=0.5, y2=1.15, mu=1.15)
    assert dit.saw == pytest.approx(expected[:-1], abs=2e-3)


def test_space_decodes_monotone_and_contains_default():
    """Every candidate is strictly decreasing, and the default schedule is in-space."""
    space = ScheduleSpace(5, mu=1.15)
    assert space.decode(space.seed_weights()) == pytest.approx(space.seed, abs=1e-4)
    assert space.decode(space.seed_weights())[0] == 1.0
    assert space.decode(space.seed_weights())[-1] == 0.0

    rng = np.random.RandomState(0)
    for _ in range(50):
        z = rng.uniform(-6, 6, space.dim).tolist()
        sched = space.decode(z)
        assert len(sched) == space.steps + 1
        assert all(b < a for a, b in zip(sched, sched[1:]))


def test_tune_recovers_known_optimum():
    """tune() maximizes the scorer it is given, starting from the default schedule."""
    target = [1.0, 0.95, 0.7, 0.3, 0.0]

    def scorer(schedule):
        return -sum((a - b) ** 2 for a, b in zip(schedule, target))

    space = ScheduleSpace(4, mu=1.15)
    result = tune(scorer, 4, init=6, iters=25, seed=0, space=space, verbose=False)

    default_score = scorer(space.seed)
    assert result.score > default_score
    # BO is not exact; require a large share of the achievable gain.
    gain = (result.score - default_score) / (0.0 - default_score)
    assert gain > 0.5, f"recovered only {gain:.0%} of the possible improvement"
    # The default schedule is always evaluated, so it is always in the history.
    assert any(s == pytest.approx(space.seed, abs=1e-4) for s, _ in result.history)


def test_tune_runs_end_to_end_through_sample(dit):
    """The search loop renders real candidates through sampling.sample."""
    from schedule_search import mean_pairwise_distance, sampler

    render = sampler(
        dit, FakeAE(), FakeEncoder(), "a fox", steps=4, guidance=0.0, device="cpu"
    )
    best = tune(
        lambda sched: mean_pairwise_distance([render(sched)]),
        4,
        init=3,
        iters=5,
        seed=0,
        verbose=False,
    )
    assert dit.saw, "no denoising steps ran"
    assert best.score >= 0.0
    assert len(best.timesteps) == 5


def test_schedule_roundtrip(tmp_path):
    """save -> load returns the same schedule, and malformed files are rejected."""
    schedule = [1.0, 0.8, 0.4, 0.0]
    path = tmp_path / "turbo_3step.json"
    save_schedule(schedule, path)
    assert json.loads(path.read_text()) == schedule
    assert load_schedule(path) == schedule

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([1.0, 0.8, 0.8, 0.0]))
    with pytest.raises(ValueError, match="strictly decreasing"):
        load_schedule(bad)


def test_timesteps_monotone_by_construction():
    """The derived grid the tuner seeds from is itself a valid schedule."""
    for steps in (2, 5, 8, 52):
        sched = timesteps(1024, steps, 256, 6400, mu=1.15)
        assert sched[0] == 1.0 and sched[-1] == 0.0
        assert all(b < a for a, b in zip(sched, sched[1:])), steps


def test_cli_schedule_flag_reaches_sampler(tmp_path, monkeypatch, dit):
    """`inference.py --schedule FILE` loads the JSON and walks exactly those steps."""
    import inference

    schedule = [1.0, 0.75, 0.25, 0.0]
    path = tmp_path / "tuned.json"
    save_schedule(schedule, path)

    monkeypatch.setattr(inference, "_pipeline", lambda **kwargs: (dit, FakeAE(), FakeEncoder()))
    # The CLI calls sampling.sample on the default cuda device; run it on cpu here.
    monkeypatch.setattr(
        inference, "sample", lambda *a, **kw: sample(*a, **{**kw, "device": "cpu"})
    )

    from click.testing import CliRunner

    result = CliRunner().invoke(
        inference.main,
        ["a fox", "--schedule", str(path), "--cfg", "0.0", "--output", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output or result.exception
    assert dit.saw == pytest.approx(schedule[:-1], abs=2e-3)
