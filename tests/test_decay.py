"""Tests for the Gaussian time decay function."""

import math

from social_graph.decay import DECAY_CONSTANT_DAYS, THREE_SIGMA_DAYS, exponential_time_decay


def test_decay_at_zero():
    assert exponential_time_decay(0) == 1.0


def test_decay_at_one_constant():
    """At t = decay constant (112 days), should be exp(-1) ≈ 0.3679."""
    result = exponential_time_decay(DECAY_CONSTANT_DAYS)
    assert math.isclose(result, math.exp(-1), rel_tol=1e-9)


def test_decay_at_three_sigma():
    """At t = 3 * decay constant (336 days), should be near zero."""
    result = exponential_time_decay(THREE_SIGMA_DAYS)
    assert result < 0.001


def test_decay_monotonically_decreasing():
    prev = 1.0
    for days in range(1, 400, 10):
        current = exponential_time_decay(days)
        assert current < prev
        prev = current


def test_decay_constants():
    assert DECAY_CONSTANT_DAYS == 112
    assert THREE_SIGMA_DAYS == 336


def test_decay_at_one_day():
    """At 1 day, decay should be almost exactly 1.0."""
    result = exponential_time_decay(1)
    # exp(-(1/112)^2) ≈ 0.99992
    assert result > 0.9999
