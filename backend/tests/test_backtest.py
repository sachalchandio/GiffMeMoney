"""Tests for the vectorized backtesting engine (``app.quant.backtest``).

These pin the realized-performance contract:

* a **constant-long** position reproduces the **buy & hold** benchmark exactly
  (the core verification invariant);
* a known position series produces a known CAGR / max-drawdown / total-return;
* win-rate and profit-factor are sane (in range, all-wins -> large finite PF);
* a **flat** position earns ~0 with no trades;
* every metric is finite, and a ``supported=False`` strategy mirrors buy & hold
  with zeroed strategy metrics.
"""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

from app.quant.backtest import (
    BacktestMetrics,
    BacktestResult,
    backtest_positions,
)
from app.quant.returns import TRADING_DAYS


# A simple rising-then-wobbling close series used across several tests.
_CLOSES = np.array(
    [100.0, 101.0, 102.0, 103.0, 104.0, 103.0, 105.0, 106.0, 107.0, 108.0]
)


def _all_metric_values(m: BacktestMetrics) -> list[float]:
    """Return every numeric field value of a :class:`BacktestMetrics`."""
    return [getattr(m, f.name) for f in fields(m)]


# ---------------------------------------------------------------------------
# Constant-long == buy & hold (the core invariant)
# ---------------------------------------------------------------------------


def test_constant_long_reproduces_buy_and_hold() -> None:
    """A constant unit-long position (no cost) equals the buy & hold benchmark."""
    res = backtest_positions(_CLOSES, np.ones(_CLOSES.size), cost=0.0)
    assert isinstance(res, BacktestResult)
    assert res.metrics.total_return == pytest.approx(res.benchmark.total_return)
    assert res.metrics.cagr == pytest.approx(res.benchmark.cagr)
    assert res.metrics.max_drawdown == pytest.approx(res.benchmark.max_drawdown)
    assert res.metrics.ann_vol == pytest.approx(res.benchmark.ann_vol)
    # Buy & hold total return is simply last/first - 1 = 108/100 - 1 = 0.08.
    assert res.benchmark.total_return == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# Known position -> known metrics
# ---------------------------------------------------------------------------


def test_known_drawdown_and_total_return() -> None:
    """A peak-to-trough -50% path gives a -50% max drawdown and -10% total return."""
    prices = np.array([100.0, 110.0, 120.0, 90.0, 60.0, 75.0, 90.0])
    res = backtest_positions(prices, np.ones(prices.size), cost=0.0)
    # Peak 120 -> trough 60 is a 50% decline.
    assert res.benchmark.max_drawdown == pytest.approx(-0.5)
    assert res.metrics.max_drawdown == pytest.approx(-0.5)
    # 90 / 100 - 1 = -0.10 total return.
    assert res.benchmark.total_return == pytest.approx(-0.1)


def test_known_cagr_from_equity() -> None:
    """CAGR matches ``(V_end/V_start)^(252/N) - 1`` for a constant-long curve."""
    res = backtest_positions(_CLOSES, np.ones(_CLOSES.size), cost=0.0)
    n_bars = _CLOSES.size - 1  # number of return bars
    expected = 1.08 ** (TRADING_DAYS / n_bars) - 1.0
    assert res.benchmark.cagr == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Win rate / profit factor sanity
# ---------------------------------------------------------------------------


def test_win_rate_and_profit_factor_all_wins() -> None:
    """An all-up series held long has win-rate 1.0 and a large finite profit factor."""
    up = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    res = backtest_positions(up, np.ones(up.size), cost=0.0)
    assert res.metrics.win_rate == pytest.approx(1.0)
    assert res.metrics.profit_factor > 0.0
    assert math.isfinite(res.metrics.profit_factor)


def test_win_rate_in_unit_interval() -> None:
    """Win-rate and exposure stay within [0, 1] for an arbitrary position series."""
    pos = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1], dtype=float)
    res = backtest_positions(_CLOSES, pos)
    assert 0.0 <= res.metrics.win_rate <= 1.0
    assert 0.0 <= res.metrics.exposure <= 1.0
    assert res.metrics.profit_factor >= 0.0


# ---------------------------------------------------------------------------
# Flat position -> ~0
# ---------------------------------------------------------------------------


def test_flat_position_is_zero() -> None:
    """A flat (all-zero) position earns nothing and trades nothing."""
    res = backtest_positions(_CLOSES, np.zeros(_CLOSES.size))
    assert res.metrics.total_return == pytest.approx(0.0)
    assert res.metrics.cagr == pytest.approx(0.0)
    assert res.metrics.max_drawdown == pytest.approx(0.0)
    assert res.metrics.exposure == pytest.approx(0.0)
    assert res.trades == 0


# ---------------------------------------------------------------------------
# Finiteness & shape
# ---------------------------------------------------------------------------


def test_all_metrics_finite() -> None:
    """Every metric on both legs is a finite float."""
    pos = np.array([1, 1, 0, -1, 0, 1, 1, 0, 1, 1], dtype=float)
    res = backtest_positions(_CLOSES, pos)
    for value in _all_metric_values(res.metrics):
        assert math.isfinite(value)
    for value in _all_metric_values(res.benchmark):
        assert math.isfinite(value)


def test_equity_curve_downsampled_and_normalized() -> None:
    """The equity curve is non-empty, capped at ~120 points and starts at 1.0."""
    res = backtest_positions(_CLOSES, np.ones(_CLOSES.size), cost=0.0)
    curve = res.equity_curve
    assert curve
    assert len(curve) <= 120
    first = curve[0]
    for key in ("t", "strategy", "benchmark"):
        assert key in first
    assert first["strategy"] == pytest.approx(1.0)
    assert first["benchmark"] == pytest.approx(1.0)


def test_trades_counted_on_position_changes() -> None:
    """The trade count equals the number of post-entry position changes."""
    pos = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 1], dtype=float)
    res = backtest_positions(_CLOSES, pos)
    # Changes after bar 0: 0->1, 1->0, 0->1, 1->0, 0->1 = 5 changes.
    assert res.trades == 5


# ---------------------------------------------------------------------------
# supported flag
# ---------------------------------------------------------------------------


def test_unsupported_strategy_mirrors_benchmark() -> None:
    """A ``supported=False`` strategy reports zeroed metrics + a valid benchmark."""
    res = backtest_positions(
        _CLOSES, np.ones(_CLOSES.size), supported=False
    )
    assert res.supported is False
    assert res.metrics.total_return == pytest.approx(0.0)
    # The benchmark leg still reports the real buy & hold performance.
    assert res.benchmark.total_return == pytest.approx(0.08)
    assert res.equity_curve  # the chart still renders


def test_degenerate_input_does_not_raise() -> None:
    """Empty / single-bar inputs yield finite metrics without raising."""
    empty = backtest_positions(np.array([]), np.array([]))
    assert empty.metrics.total_return == pytest.approx(0.0)
    single = backtest_positions(np.array([100.0]), np.array([1.0]))
    assert single.metrics.total_return == pytest.approx(0.0)
    for value in _all_metric_values(single.metrics):
        assert math.isfinite(value)
