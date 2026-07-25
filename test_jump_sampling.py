"""Pure unit tests for Truncated Jump Sampling (no torch required).

These verify the paper's core endpoint-decodability identity -- the math that
lets the flow ODE stop early and jump to the decoded clean latent. The
torch-dependent wiring through ``sampling.sample`` is covered separately in
``test_sampling_tjs``.
"""

import math

import pytest

from jump_sampling import decode_endpoint, truncation_steps


# --- endpoint decodability (the paper's core identity) ---


def test_decode_endpoint_formula():
    # x_0 = x_t - t * v for the affine path x_t = t * noise + (1 - t) * x_0.
    assert decode_endpoint(10.0, 2.0, 3.0) == pytest.approx(10.0 - 3.0 * 2.0)


def test_decode_endpoint_recovers_clean_sample():
    # If v = noise - x_0 on the affine path, the decode returns x_0 exactly.
    x_0, noise, t = 3.0, -5.0, 0.7
    x_t = t * noise + (1.0 - t) * x_0
    v = noise - x_0
    assert decode_endpoint(x_t, v, t) == pytest.approx(x_0)


def test_decode_endpoint_matches_final_euler_step():
    # At the schedule's final step tprev -> 0, the Euler update x + (0 - t) * v
    # equals the endpoint decode. This is why TJS gracefully reduces to a full
    # rollout as the exit time -> 0.
    x, v, t = 7.0, 1.5, 0.25
    euler_final = x + (0.0 - t) * v
    assert decode_endpoint(x, v, t) == pytest.approx(euler_final)


def test_decode_endpoint_zero_time_is_identity():
    # At t = 0 the affine path is already at the data endpoint: x_0 = x_t.
    assert decode_endpoint(4.2, 99.0, 0.0) == pytest.approx(4.2)


# --- truncation schedule ---


def test_truncation_steps_rounds_up():
    assert truncation_steps(28, 0.5) == 14
    assert truncation_steps(28, 0.3) == math.ceil(0.3 * 28)
    assert truncation_steps(28, 1.0) == 28


def test_truncation_steps_keeps_at_least_one():
    # Even a tiny fraction keeps one model call so the jump is seeded.
    assert truncation_steps(28, 0.01) == 1
