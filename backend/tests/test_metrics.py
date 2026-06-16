"""Tests for :mod:`app.quant.metrics` against closed-form / known answers.

Each test pins a metric to a value derivable by hand from a synthetic series, so
a regression in the formula (not just "it runs") is caught. The implementation
uses the *population* standard deviation (``ddof=0``); the expected values below
are computed the same way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.quant import metrics
from app.quant.returns import TRADING_DAYS, annualize_return

_SQRT_DAYS = math.sqrt(TRADING_DAYS)


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------


def test_beta_asset_equals_market_is_one() -> None:
    """beta of an asset whose returns equal the market's is exactly 1."""
    market = np.array([0.01, -0.02, 0.015, 0.0, -0.005, 0.02, -0.01], dtype=float)
    assert metrics.beta(market, market) == pytest.approx(1.0, abs=1e-9)


def test_beta_double_market_is_two() -> None:
    """beta of an asset moving 2x the market (no noise) is 2."""
    market = np.array([0.01, -0.02, 0.015, 0.0, -0.005, 0.02, -0.01], dtype=float)
    asset = 2.0 * market
    assert metrics.beta(asset, market) == pytest.approx(2.0, abs=1e-9)


def test_beta_zero_variance_market_returns_neutral_default() -> None:
    """A constant (zero-variance) market collapses beta to the 1.0 default."""
    asset = np.array([0.01, -0.02, 0.03, 0.0], dtype=float)
    flat_market = np.zeros(4, dtype=float)
    assert metrics.beta(asset, flat_market) == pytest.approx(1.0)


def test_beta_too_short_returns_default() -> None:
    """Fewer than two aligned observations returns the neutral default."""
    assert metrics.beta([0.01], [0.01]) == pytest.approx(1.0)
    assert metrics.beta([], []) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# annual_volatility
# ---------------------------------------------------------------------------


def test_annual_volatility_matches_sqrt_time_rule() -> None:
    """Annual vol equals population std times sqrt(252)."""
    returns = np.array([0.01, -0.01, 0.02, -0.02, 0.0, 0.015], dtype=float)
    expected = float(np.std(returns)) * _SQRT_DAYS
    assert metrics.annual_volatility(returns) == pytest.approx(expected, rel=1e-12)


def test_annual_volatility_constant_series_is_zero() -> None:
    """A constant series has zero volatility (no divide blow-up)."""
    assert metrics.annual_volatility(np.full(50, 0.003)) == pytest.approx(0.0)


def test_annual_volatility_short_input_is_zero() -> None:
    """Fewer than two observations yields 0.0."""
    assert metrics.annual_volatility([0.01]) == pytest.approx(0.0)
    assert metrics.annual_volatility([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# sharpe / sortino — zero-vol guards and a known value
# ---------------------------------------------------------------------------


def test_sharpe_constant_excess_returns_zero(constant_excess_returns: np.ndarray) -> None:
    """A constant-excess series (zero std of excess) yields Sharpe 0, not inf."""
    assert metrics.sharpe(constant_excess_returns, rf_daily=0.0) == pytest.approx(0.0)


def test_sharpe_known_value() -> None:
    """Sharpe equals mean(excess)/std(excess)*sqrt(252) on a known series."""
    returns = np.array([0.02, -0.01, 0.03, 0.0, 0.01, -0.02, 0.015], dtype=float)
    rf = 0.0
    excess = returns - rf
    expected = (float(np.mean(excess)) / float(np.std(excess))) * _SQRT_DAYS
    assert metrics.sharpe(returns, rf) == pytest.approx(expected, rel=1e-12)


def test_sharpe_short_input_is_zero() -> None:
    """Fewer than two observations yields 0.0."""
    assert metrics.sharpe([0.01], 0.0) == pytest.approx(0.0)


def test_sortino_constant_excess_returns_zero(constant_excess_returns: np.ndarray) -> None:
    """All returns above the MAR -> zero downside deviation -> Sortino 0."""
    # rf=0, every return = +0.001 > 0, so there is no downside.
    assert metrics.sortino(constant_excess_returns, rf_daily=0.0) == pytest.approx(0.0)


def test_sortino_known_value() -> None:
    """Sortino equals mean(excess)/downside_dev*sqrt(252) on a known series."""
    returns = np.array([0.02, -0.01, 0.03, -0.04, 0.01, -0.02, 0.015], dtype=float)
    rf = 0.0
    shortfall = np.minimum(returns - rf, 0.0)
    dd = math.sqrt(float(np.mean(shortfall * shortfall)))
    expected = (float(np.mean(returns - rf)) / dd) * _SQRT_DAYS
    assert metrics.sortino(returns, rf) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# downside_deviation
# ---------------------------------------------------------------------------


def test_downside_deviation_no_downside_is_zero() -> None:
    """When every return meets the MAR the downside deviation is zero."""
    assert metrics.downside_deviation(np.array([0.01, 0.02, 0.0]), mar=0.0) == pytest.approx(0.0)


def test_downside_deviation_known_value() -> None:
    """Only sub-MAR returns contribute; divisor is the full sample size."""
    returns = np.array([0.02, -0.01, -0.03, 0.04], dtype=float)
    shortfall = np.minimum(returns, 0.0)  # [0, -0.01, -0.03, 0]
    expected = math.sqrt(float(np.mean(shortfall * shortfall)))
    assert metrics.downside_deviation(returns, mar=0.0) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# treynor / jensen_alpha
# ---------------------------------------------------------------------------


def test_treynor_known_value() -> None:
    """Treynor = annualized mean excess / beta on a series with known beta."""
    market = np.array([0.01, -0.02, 0.015, 0.0, -0.005, 0.02, -0.01], dtype=float)
    asset = 2.0 * market  # beta == 2 exactly
    rf = 0.0
    expected = (float(np.mean(asset - rf)) * TRADING_DAYS) / 2.0
    assert metrics.treynor(asset, market, rf) == pytest.approx(expected, rel=1e-9)


def test_jensen_alpha_zero_when_asset_tracks_capm() -> None:
    """An asset that exactly equals the market (beta 1) has ~zero alpha at rf=0."""
    market = np.array([0.01, -0.02, 0.015, 0.0, -0.005, 0.02, -0.01], dtype=float)
    # alpha_daily = mean(asset) - [rf + beta*(mean(mkt)-rf)] = 0 when asset==market.
    assert metrics.jensen_alpha(market, market, rf_daily=0.0) == pytest.approx(0.0, abs=1e-9)


def test_jensen_alpha_constant_outperformance() -> None:
    """A constant per-day excess over the market annualizes to that alpha."""
    market = np.array([0.01, -0.02, 0.015, 0.0, -0.005, 0.02, -0.01], dtype=float)
    bonus = 0.001
    asset = market + bonus  # beta still 1, daily alpha == bonus at rf=0
    expected = annualize_return(bonus)
    assert metrics.jensen_alpha(asset, market, rf_daily=0.0) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# information_ratio
# ---------------------------------------------------------------------------


def test_information_ratio_vs_self_is_zero() -> None:
    """Active return against an identical benchmark is zero -> IR 0."""
    series = np.array([0.01, -0.02, 0.03, 0.0, 0.015], dtype=float)
    assert metrics.information_ratio(series, series) == pytest.approx(0.0)


def test_information_ratio_known_value() -> None:
    """IR = mean(active)/std(active)*sqrt(252) on a known active series."""
    port = np.array([0.02, -0.01, 0.03, 0.0, 0.01], dtype=float)
    bench = np.array([0.01, -0.015, 0.02, 0.005, 0.0], dtype=float)
    active = port - bench
    expected = (float(np.mean(active)) / float(np.std(active))) * _SQRT_DAYS
    assert metrics.information_ratio(port, bench) == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------


def test_max_drawdown_known_path(known_drawdown_prices: np.ndarray) -> None:
    """The 120 -> 60 leg is a 50% peak-to-trough decline."""
    assert metrics.max_drawdown(known_drawdown_prices) == pytest.approx(-0.5, rel=1e-12)


def test_max_drawdown_monotonic_increase_is_zero() -> None:
    """A strictly rising path never draws down."""
    rising = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)
    assert metrics.max_drawdown(rising) == pytest.approx(0.0)


def test_max_drawdown_short_input_is_zero() -> None:
    """Fewer than two valid prices yields 0.0."""
    assert metrics.max_drawdown([100.0]) == pytest.approx(0.0)
    assert metrics.max_drawdown([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# calmar
# ---------------------------------------------------------------------------


def test_calmar_known_value(known_drawdown_prices: np.ndarray) -> None:
    """Calmar = annualized return / |max drawdown| with a known -50% drawdown."""
    returns = np.array([0.001, -0.001, 0.002, 0.0, 0.0015], dtype=float)
    ann_ret = annualize_return(float(np.mean(returns)))
    expected = ann_ret / 0.5
    assert metrics.calmar(returns, known_drawdown_prices) == pytest.approx(expected, rel=1e-9)


def test_calmar_no_drawdown_is_zero() -> None:
    """Zero drawdown denominator collapses Calmar to 0 (no blow-up)."""
    returns = np.array([0.001, 0.002, 0.001], dtype=float)
    rising = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)
    assert metrics.calmar(returns, rising) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# defensiveness: nothing raises and everything is finite
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [float("nan"), float("inf")],
        [0.0],
        [float("nan"), 0.01, float("-inf"), 0.02],
    ],
)
def test_metrics_never_raise_on_degenerate_input(bad: list[float]) -> None:
    """Empty / NaN / inf inputs must return finite floats, never raise."""
    arr = np.asarray(bad, dtype=float)
    assert math.isfinite(metrics.annual_volatility(arr))
    assert math.isfinite(metrics.sharpe(arr, 0.0))
    assert math.isfinite(metrics.sortino(arr, 0.0))
    assert math.isfinite(metrics.beta(arr, arr))
    assert math.isfinite(metrics.downside_deviation(arr))
    assert math.isfinite(metrics.treynor(arr, arr, 0.0))
    assert math.isfinite(metrics.jensen_alpha(arr, arr, 0.0))
    assert math.isfinite(metrics.information_ratio(arr, arr))
    assert math.isfinite(metrics.max_drawdown(arr))
    assert math.isfinite(metrics.calmar(arr, arr))
