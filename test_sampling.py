"""Integration tests for the ODE-solver wiring in ``sampling.sample``.

These drive the real ``sampling.sample`` path (encode -> integrate -> decode)
with lightweight fakes, asserting that the ``solver`` argument is threaded
through to ``solvers.integrate`` and that the higher-order mean-direction
integrator behaves as expected against the first-order Euler default.
"""

import numpy as np
import torch
from PIL import Image

from sampling import sample
from solvers import integrate


class _FakeConfig:
    patch = 2


class _FakeModel:
    """Flow-velocity stub. ``velocity`` maps ``(img, t) -> velocity`` tensor.

    With CFG disabled in the harness, ``sample`` calls the model exactly once
    per velocity evaluation, so the call count directly reflects the integrator.
    """

    def __init__(self, velocity):
        self.velocity = velocity
        self.config = _FakeConfig()
        self.calls = 0

    def __call__(self, img, context, t, pos, mask):
        self.calls += 1
        return self.velocity(img, t)


class _FakeAE:
    compression = 8
    channels = 4

    def decode(self, x):
        # Tile the latent into pixel space so the decoded image reflects the
        # integrated state (only shape + value-dependence matter for the tests).
        return x[:, :3].repeat_interleave(self.compression, dim=2).repeat_interleave(
            self.compression, dim=3
        )


class _FakeEncoder:
    def __init__(self, txtlen=8, dim=16):
        self.txtlen = txtlen
        self.dim = dim

    def __call__(self, prompts):
        b = len(prompts)
        txt = torch.zeros(b, self.txtlen, self.dim)
        mask = torch.ones(b, self.txtlen, dtype=torch.bool)
        return txt, mask


def _run(model, solver="euler", steps=4, seed=7):
    return sample(
        model,
        _FakeAE(),
        _FakeEncoder(),
        ["a test prompt"],
        device="cpu",
        dtype=torch.float32,
        width=32,
        height=32,
        steps=steps,
        guidance=0.0,  # disable CFG so the velocity equals the model output
        seed=seed,
        solver=solver,
    )


def test_sample_returns_image_for_each_solver():
    """Both solvers run end-to-end and return a valid decoded image."""
    model = _FakeModel(lambda img, t: torch.full_like(img, 0.1))
    for solver in ("euler", "mean_direction"):
        images = _run(model, solver=solver)
        assert len(images) == 1
        assert isinstance(images[0], Image.Image)
        assert images[0].size == (32, 32)


def test_mean_direction_matches_euler_for_constant_velocity():
    """For a state-independent velocity the trapezoidal rule is exact, so the
    two solvers must produce identical samples (wiring is symmetric)."""
    const = _FakeModel(lambda img, t: torch.full_like(img, 0.1))
    euler_img = _run(const, solver="euler")[0]
    const2 = _FakeModel(lambda img, t: torch.full_like(img, 0.1))
    md_img = _run(const2, solver="mean_direction")[0]
    assert np.array_equal(np.asarray(euler_img), np.asarray(md_img))


def test_solver_selection_changes_evaluation_count():
    """``mean_direction`` evaluates velocity twice per step (predictor +
    corrector) while ``euler`` evaluates once. Counting calls proves the
    ``solver`` kwarg is wired through ``sample`` and selects a different
    integrator rather than silently aliasing to Euler."""
    n_steps = 4
    euler_model = _FakeModel(lambda img, t: -img)
    _run(euler_model, solver="euler", steps=n_steps)
    md_model = _FakeModel(lambda img, t: -img)
    _run(md_model, solver="mean_direction", steps=n_steps)
    assert euler_model.calls == n_steps
    assert md_model.calls == 2 * n_steps


def test_unknown_solver_raises():
    """An unknown solver name is rejected at integration time."""
    model = _FakeModel(lambda img, t: torch.full_like(img, 0.1))
    try:
        _run(model, solver="bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown solver")


def test_mean_direction_is_higher_order_than_euler():
    """On dx/dt = -x (analytic x(t) = x0 * e^{-t}) integrated t: 1 -> 0, the
    mean-direction (trapezoidal) solver is more accurate than first-order
    Euler for the same coarse schedule."""

    def velocity(x, t):
        return -x

    ts = [1.0, 0.5, 0.0]
    x0 = torch.tensor([1.0])
    euler_x = integrate(velocity, x0.clone(), ts, solver="euler")
    md_x = integrate(velocity, x0.clone(), ts, solver="mean_direction")
    # x(0) = x0 * e^{-(0-1)} = x0 * e.
    exact = float(x0[0] * np.e)
    assert abs(float(md_x[0]) - exact) < abs(float(euler_x[0]) - exact)
