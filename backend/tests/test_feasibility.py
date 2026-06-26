"""Tests for the honesty helpers (``app.strategies.feasibility``).

HONESTY / SAFETY (this is a finance tool). These pin the impossible-target guard
and the synthetic-data note:

* :func:`~app.strategies.feasibility.feasibility_warning` flags a goal whose
  implied per-day compounding is physically extreme (e.g. a 100x ask over two
  months) and stays silent for sane, achievable goals — and never raises on
  malformed / non-finite input;
* :data:`~app.strategies.feasibility.SYNTHETIC_DATA_NOTE` plainly states the
  results are simulated on synthetic data and never implies guaranteed profit.
"""

from __future__ import annotations

import math

import pytest

from app.strategies.feasibility import (
    DAILY_COMPOUNDING_WARN_PCT,
    SYNTHETIC_DATA_NOTE,
    feasibility_warning,
)


def test_flags_100x_over_two_months() -> None:
    """A 100x ask over ~2 months (≈61 days) is flagged as infeasible."""
    warn = feasibility_warning(amount=100.0, target_amount=10_000.0, horizon_days=61)
    assert warn is not None
    # The implied daily compounding is ~8%/day — well above the threshold.
    assert "every day" in warn.lower()
    # 100x overall is surfaced so the reader sees the scale of the ask.
    assert "100x" in warn


def test_silent_on_a_sane_goal() -> None:
    """A modest goal (10% over a year) compounds gently — no warning."""
    assert (
        feasibility_warning(amount=1_000.0, target_amount=1_100.0, horizon_days=365)
        is None
    )


def test_silent_when_target_not_an_increase() -> None:
    """A target at or below the start amount is nothing to warn about."""
    assert feasibility_warning(100.0, 100.0, 30) is None
    assert feasibility_warning(100.0, 50.0, 30) is None


@pytest.mark.parametrize(
    "amount,target,days",
    [
        (0.0, 1_000.0, 30),  # non-positive start
        (-100.0, 1_000.0, 30),  # negative start
        (100.0, 1_000.0, 0),  # zero horizon
        (100.0, 1_000.0, -5),  # negative horizon
        (float("nan"), 1_000.0, 30),  # non-finite amount
        (100.0, float("inf"), 30),  # non-finite target
        (100.0, 1_000.0, float("nan")),  # non-finite horizon
    ],
)
def test_never_raises_on_degenerate_input(amount, target, days) -> None:
    """Malformed / non-finite inputs return None, never raise."""
    assert feasibility_warning(amount, target, days) is None


def test_threshold_boundary_behaviour() -> None:
    """Just under the daily threshold is silent; well over it warns."""
    # A 1-day horizon makes the implied daily rate equal to the overall multiple.
    just_under = 1.0 + (DAILY_COMPOUNDING_WARN_PCT - 0.5) / 100.0
    assert feasibility_warning(100.0, 100.0 * just_under, 1) is None
    well_over = 1.0 + (DAILY_COMPOUNDING_WARN_PCT + 5.0) / 100.0
    assert feasibility_warning(100.0, 100.0 * well_over, 1) is not None


def test_synthetic_data_note_is_honest() -> None:
    """The note states results are simulated/synthetic and never guarantees profit."""
    text = SYNTHETIC_DATA_NOTE.lower()
    assert "synthetic" in text
    assert "simulation" in text or "simulated" in text
    assert "not a real forecast" in text
    assert "not financial advice" in text
    assert "guarantee" not in text
