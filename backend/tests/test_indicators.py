"""Tests for the V2 OHLCV chart indicators (``app.quant.indicators``).

These pin the new volatility / trend / oscillator / volume indicators that the
V2 technical and mean-reversion strategy signals consume:

    true_range / atr / adx / donchian / supertrend / ichimoku /
    williams_r / stochastic / cci / keltner / obv / obv_slope

Each indicator is checked on a *known* series whose expected value can be
hand-computed, plus the defensive contract (short / empty / flat inputs collapse
to safe finite neutral values, never raise).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.quant import indicators as ind


# ---------------------------------------------------------------------------
# true_range / atr
# ---------------------------------------------------------------------------


def test_true_range_known_series() -> None:
    """TR_0 = high-low; later TR is the max of the three Wilder ranges."""
    h = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
    l = np.array([8.0, 9.0, 9.0, 10.0, 11.0])
    c = np.array([9.0, 11.0, 10.0, 12.0, 13.0])
    tr = ind.true_range(h, l, c)
    assert tr.size == h.size
    # First bar has no previous close: TR_0 = 10 - 8 = 2.
    assert tr[0] == pytest.approx(2.0)
    # Bar 1: max(12-9, |12-9|, |9-9|) = max(3, 3, 0) = 3.
    assert tr[1] == pytest.approx(3.0)
    assert np.all(tr >= 0.0)
    assert np.all(np.isfinite(tr))


def test_atr_length_and_nonnegative() -> None:
    """ATR aligns to the inputs, is non-negative, and is finite."""
    h = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
    l = np.array([8.0, 9.0, 9.0, 10.0, 11.0])
    c = np.array([9.0, 11.0, 10.0, 12.0, 13.0])
    a = ind.atr(h, l, c, n=3)
    assert a.size == c.size
    assert np.all(a >= 0.0)
    assert np.all(np.isfinite(a))


def test_true_range_empty_is_empty() -> None:
    """An empty input yields an empty (never-raising) true-range array."""
    tr = ind.true_range([], [], [])
    assert tr.size == 0


# ---------------------------------------------------------------------------
# adx
# ---------------------------------------------------------------------------


def test_adx_in_range_on_trend() -> None:
    """ADX is in [0, 100]; a strong monotone trend pins it near the top."""
    trend = np.arange(1.0, 80.0)
    value = ind.adx(trend, trend, trend, n=14)
    assert 0.0 <= value <= 100.0
    # A perfectly directional series has maximal directional strength.
    assert value > 90.0


def test_adx_short_input_is_zero() -> None:
    """Too few bars -> no measurable directional strength (0.0)."""
    assert ind.adx([1.0], [1.0], [1.0]) == 0.0
    assert ind.adx([], [], []) == 0.0


# ---------------------------------------------------------------------------
# donchian
# ---------------------------------------------------------------------------


def test_donchian_excludes_current_bar() -> None:
    """Donchian channel uses the prior ``n`` bars (excludes the current bar)."""
    h = np.array([10.0, 11.0, 12.0, 13.0, 20.0])
    l = np.array([5.0, 6.0, 7.0, 8.0, 4.0])
    upper, lower = ind.donchian(h, l, n=3)
    # Prior 3 highs are 11, 12, 13 -> upper 13 (the 20 of the current bar is excluded).
    assert upper == pytest.approx(13.0)
    # Prior 3 lows are 6, 7, 8 -> lower 6.
    assert lower == pytest.approx(6.0)
    assert upper >= lower


def test_donchian_empty_is_zero() -> None:
    """No bars -> a (0, 0) band, never raising."""
    assert ind.donchian([], []) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# supertrend
# ---------------------------------------------------------------------------


def test_supertrend_uptrend_is_bullish() -> None:
    """A strong uptrend yields a +1 (bullish) Supertrend direction."""
    up = np.arange(1.0, 60.0)
    level, direction = ind.supertrend(up, up, up, n=10, mult=3.0)
    assert direction == 1
    assert math.isfinite(level)


def test_supertrend_downtrend_is_bearish() -> None:
    """A strong downtrend yields a -1 (bearish) Supertrend direction."""
    down = np.arange(60.0, 1.0, -1.0)
    _level, direction = ind.supertrend(down, down, down, n=10, mult=3.0)
    assert direction == -1


def test_supertrend_short_input() -> None:
    """A single bar returns ``(last_close, +1)`` without raising."""
    level, direction = ind.supertrend([42.0], [42.0], [42.0])
    assert level == pytest.approx(42.0)
    assert direction == 1


# ---------------------------------------------------------------------------
# ichimoku
# ---------------------------------------------------------------------------


def test_ichimoku_keys_and_cloud_position() -> None:
    """Ichimoku returns the five lines; a rising close sits above the cloud."""
    rising = np.arange(1.0, 60.0)
    res = ind.ichimoku(rising, rising, rising)
    for key in ("tenkan", "kijun", "senkou_a", "senkou_b", "cloud_pos"):
        assert key in res
        assert math.isfinite(res[key])
    assert res["cloud_pos"] == 1.0


def test_ichimoku_below_cloud_is_bearish() -> None:
    """A falling close sits below the cloud -> cloud_pos -1."""
    falling = np.arange(60.0, 1.0, -1.0)
    res = ind.ichimoku(falling, falling, falling)
    assert res["cloud_pos"] == -1.0


def test_ichimoku_empty_is_zeroed() -> None:
    """No bars -> every Ichimoku value is 0.0."""
    res = ind.ichimoku([], [], [])
    assert all(v == 0.0 for v in res.values())


# ---------------------------------------------------------------------------
# williams_r
# ---------------------------------------------------------------------------


def test_williams_r_extremes() -> None:
    """%R is 0 when the close is at the period high, -100 at the period low."""
    h = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    l = np.array([5.0, 6.0, 7.0, 8.0, 5.0])
    close_at_high = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    assert ind.williams_r(h, l, close_at_high, n=5) == pytest.approx(0.0)
    close_at_low = np.array([10.0, 11.0, 12.0, 13.0, 5.0])
    assert ind.williams_r(h, l, close_at_low, n=5) == pytest.approx(-100.0)


def test_williams_r_neutral_on_flat_or_empty() -> None:
    """A flat (zero-range) window or no bars yields the neutral -50."""
    flat = np.full(14, 50.0)
    assert ind.williams_r(flat, flat, flat, n=14) == pytest.approx(-50.0)
    assert ind.williams_r([], [], []) == pytest.approx(-50.0)


# ---------------------------------------------------------------------------
# stochastic
# ---------------------------------------------------------------------------


def test_stochastic_close_at_high_is_100() -> None:
    """%K is 100 when the close sits at the window's high (no smoothing)."""
    h = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    l = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    c = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    pk, pd = ind.stochastic(h, l, c, k=5, d=1)
    assert pk == pytest.approx(100.0)
    assert pd == pytest.approx(100.0)


def test_stochastic_flat_is_neutral() -> None:
    """A flat window and no bars both yield the neutral (50, 50)."""
    flat = np.full(20, 50.0)
    assert ind.stochastic(flat, flat, flat) == (50.0, 50.0)
    assert ind.stochastic([], [], []) == (50.0, 50.0)


# ---------------------------------------------------------------------------
# cci
# ---------------------------------------------------------------------------


def test_cci_flat_is_zero() -> None:
    """A flat (zero-deviation) window yields CCI 0; no bars also 0."""
    flat = np.full(20, 50.0)
    assert ind.cci(flat, flat, flat, n=10) == pytest.approx(0.0)
    assert ind.cci([], [], []) == 0.0


def test_cci_positive_when_above_mean() -> None:
    """A typical price above its trailing SMA gives a positive, finite CCI."""
    c = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0])
    value = ind.cci(c, c, c, n=10)
    assert value > 0.0
    assert math.isfinite(value)
    assert -500.0 <= value <= 500.0


# ---------------------------------------------------------------------------
# keltner
# ---------------------------------------------------------------------------


def test_keltner_bands_ordered() -> None:
    """The Keltner bands satisfy lower <= mid <= upper and are finite."""
    c = np.linspace(100.0, 110.0, 30)
    h = c + 1.0
    l = c - 1.0
    mid, upper, lower = ind.keltner(c, h, l, n=10, mult=2.0)
    assert lower <= mid <= upper
    assert all(math.isfinite(x) for x in (mid, upper, lower))


def test_keltner_empty_is_zeroed() -> None:
    """No bars -> a (0, 0, 0) channel."""
    assert ind.keltner([], [], []) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# obv / obv_slope
# ---------------------------------------------------------------------------


def test_obv_known_series() -> None:
    """OBV adds volume on up-closes and subtracts it on down-closes."""
    close = np.array([10.0, 11.0, 10.0, 12.0])
    volume = np.array([100.0, 200.0, 300.0, 400.0])
    series = ind.obv(close, volume)
    # OBV_0 = 0; +200 (up); -300 (down); +400 (up) -> cumulative [0, 200, -100, 300].
    assert series.tolist() == pytest.approx([0.0, 200.0, -100.0, 300.0])


def test_obv_slope_sign() -> None:
    """A monotonically accumulating OBV has a positive normalized slope."""
    close = np.arange(1.0, 30.0)  # all up-closes
    volume = np.full(close.size, 100.0)
    slope = ind.obv_slope(close, volume, n=20)
    assert slope > 0.0
    assert -10.0 <= slope <= 10.0


def test_obv_slope_flat_is_zero() -> None:
    """A flat OBV (or too few bars) gives a zero slope, never raising."""
    flat = np.full(25, 10.0)
    assert ind.obv_slope(flat, np.full(25, 100.0), n=20) == 0.0
    assert ind.obv_slope([1.0], [100.0]) == 0.0


def test_obv_empty_is_empty() -> None:
    """No aligned bars -> an empty OBV series."""
    assert ind.obv([], []).size == 0
