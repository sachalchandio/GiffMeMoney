"""OHLCV chart indicators for the GiffMeMoney quant engine (V2 expansion).

This module adds the volatility / trend / oscillator / volume indicators that
the new technical and mean-reversion strategy signals consume but that are *not*
already provided by :mod:`app.quant.technical`:

    true_range / atr        — Wilder true range and Average True Range
    adx                     — Average Directional Index (+DI / -DI / ADX), latest
    donchian                — Donchian channel (prior-N high / low), latest
    supertrend              — Supertrend trailing-stop level + trend direction
    ichimoku                — the five Ichimoku lines + cloud position
    williams_r              — Williams %R (-100..0), latest
    stochastic              — slow %K / %D stochastic oscillator, latest
    cci                     — Commodity Channel Index, latest
    keltner                 — Keltner channel (mid / upper / lower), latest
    obv / obv_slope         — On-Balance Volume series + normalized slope

The simple/exponential moving averages (``sma`` / ``ema``) and the
RSI / MACD / Bollinger / momentum / z-score indicators live in
:mod:`app.quant.technical`; ``sma`` and ``ema`` are imported from there rather
than re-implemented.

Every function is numerically defensive. Short, empty, constant, mismatched or
non-finite inputs never raise — they collapse to safe, finite, neutral defaults
(e.g. ADX ``0``, Williams %R ``-50``, stochastic ``50``, CCI ``0``, cloud
position ``0``, OBV slope ``0``). All computation is vectorized with numpy where
possible; the Wilder-smoothed families use a single linear recursion. Functions
that the catalog specifies as "latest value" return the most recent bar's value
as a plain ``float`` (or small tuple/dict thereof).
"""

from __future__ import annotations

import math

import numpy as np

from app.quant.technical import ema

__all__ = [
    "true_range",
    "atr",
    "adx",
    "donchian",
    "supertrend",
    "ichimoku",
    "williams_r",
    "stochastic",
    "cci",
    "keltner",
    "obv",
    "obv_slope",
]

# Smallest denominator treated as non-zero; below this a range/band is
# effectively degenerate and the ratio would blow up, so we collapse to a
# neutral default rather than dividing.
_EPS: float = 1e-12


def _safe_float(value: float, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, falling back to ``default``.

    Args:
        value: Candidate number.
        default: Value substituted when ``value`` is NaN / +-inf.

    Returns:
        ``float(value)`` if finite, else ``default``.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _as_array(values: np.ndarray | list[float]) -> np.ndarray:
    """Coerce an input to a 1-D ``float64`` array (without dropping elements).

    Non-finite values are preserved here (callers that need alignment must keep
    positional correspondence between the high / low / close / volume series);
    they are sanitized to neutral values at the point of use.

    Args:
        values: Sequence of numbers.

    Returns:
        A 1-D ``float64`` array (possibly empty).
    """
    return np.asarray(values, dtype=np.float64).ravel()


def _align_ohlc(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align high / low / close to a common trailing length and sanitize them.

    The three series are truncated to ``min(len)`` keeping the most recent
    observations aligned, then non-finite entries are replaced positionally:
    NaN / inf in ``close`` is forward/zero-filled to a finite value, and
    ``high``/``low`` are repaired toward ``close`` so ``high >= low`` holds.

    Args:
        high: Per-bar high prices.
        low: Per-bar low prices.
        close: Per-bar close prices.

    Returns:
        A triple ``(high, low, close)`` of equal-length finite ``float64``
        arrays (possibly empty).
    """
    h = _as_array(high)
    l = _as_array(low)
    c = _as_array(close)
    n = min(h.size, l.size, c.size)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    h, l, c = h[-n:].copy(), l[-n:].copy(), c[-n:].copy()

    # Sanitize close first: replace non-finite with the previous finite close,
    # then any remaining (leading) non-finite with 0.0.
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    # Where close collapsed to 0 but high/low carry info, leave as-is; downstream
    # math floors denominators so this is safe.
    h = np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
    l = np.nan_to_num(l, nan=0.0, posinf=0.0, neginf=0.0)

    # Repair obvious inversions so high is the upper bound of the bar.
    bad = h < l
    if np.any(bad):
        hi = np.maximum(h, l)
        lo = np.minimum(h, l)
        h, l = hi, lo
    return h, l, c


def _rolling_sma(values: np.ndarray, n: int) -> np.ndarray:
    """Trailing simple moving average of an *arbitrary* finite series.

    Unlike :func:`app.quant.technical.sma` (which filters to strictly-positive
    prices), this smooths a general signal series — oscillator values such as
    fast %K that are legitimately zero or negative are preserved. ``out[t]`` is
    the mean of the trailing ``n`` values ending at ``t``; the leading
    (incomplete-window) region uses the expanding mean.

    Args:
        values: A 1-D finite ``float64`` array (length ``L``).
        n: Window length (clamped to ``[1, L]``).

    Returns:
        A 1-D ``float64`` array of length ``L`` (empty if input is empty).
    """
    L = values.size
    if L == 0:
        return np.empty(0, dtype=np.float64)
    window = max(1, min(int(n), L))
    cumsum = np.cumsum(values)
    out = np.empty(L, dtype=np.float64)
    for i in range(min(window - 1, L)):
        out[i] = cumsum[i] / (i + 1)
    if L >= window:
        rolling = cumsum.copy()
        rolling[window:] = cumsum[window:] - cumsum[:-window]
        out[window - 1 :] = rolling[window - 1 :] / window
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _wilder_rma(values: np.ndarray, n: int) -> np.ndarray:
    """Wilder's running moving average (RMA) of a finite series.

    Wilder smoothing is an EMA with ``alpha = 1 / n``:

        RMA_0 = x_0
        RMA_t = RMA_{t-1} + (x_t - RMA_{t-1}) / n
              = (1 - 1/n) * RMA_{t-1} + (1/n) * x_t

    Args:
        values: A 1-D finite ``float64`` array (length ``L >= 1``).
        n: Smoothing period (clamped to ``>= 1``).

    Returns:
        A 1-D ``float64`` array of length ``L`` of Wilder-smoothed values.
    """
    L = values.size
    period = max(1, int(n))
    out = np.empty(L, dtype=np.float64)
    if L == 0:
        return out
    alpha = 1.0 / period
    out[0] = values[0]
    for t in range(1, L):
        out[t] = out[t - 1] + alpha * (values[t] - out[t - 1])
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def true_range(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
) -> np.ndarray:
    """Wilder's true range series.

    Formula (with ``prevClose`` = the prior bar's close):

        TR_t = max( high_t - low_t,
                    |high_t - prevClose|,
                    |low_t  - prevClose| )

    The first bar has no previous close, so ``TR_0 = high_0 - low_0``.

    Args:
        high: Per-bar high prices.
        low: Per-bar low prices.
        close: Per-bar close prices.

    Returns:
        A 1-D ``float64`` array of non-negative true ranges, aligned to the
        (trailing-aligned) inputs. Empty if no aligned bars; all values finite.
    """
    h, l, c = _align_ohlc(high, low, close)
    L = c.size
    if L == 0:
        return np.empty(0, dtype=np.float64)
    hl = h - l
    if L == 1:
        tr = np.array([hl[0]], dtype=np.float64)
        return np.maximum(np.nan_to_num(tr, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    prev_close = c[:-1]
    hc = np.abs(h[1:] - prev_close)
    lc = np.abs(l[1:] - prev_close)
    tr = np.empty(L, dtype=np.float64)
    tr[0] = hl[0]
    tr[1:] = np.maximum.reduce([hl[1:], hc, lc])
    tr = np.nan_to_num(tr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(tr, 0.0)


def atr(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    n: int = 14,
) -> np.ndarray:
    """Average True Range (Wilder-smoothed true range).

    Formula:

        ATR_t = RMA_n(TR)_t        (RMA = Wilder running average, alpha = 1/n)

    Args:
        high: Per-bar high prices.
        low: Per-bar low prices.
        close: Per-bar close prices.
        n: Smoothing period (default 14, clamped to ``>= 1``).

    Returns:
        A 1-D ``float64`` array of non-negative ATR values aligned to the
        inputs. Empty if no aligned bars; all values finite.
    """
    tr = true_range(high, low, close)
    if tr.size == 0:
        return tr
    out = _wilder_rma(tr, n)
    return np.maximum(out, 0.0)


def adx(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    n: int = 14,
) -> float:
    """Average Directional Index (latest value).

    Computes Wilder's directional movement system and returns the most recent
    ADX. Intermediate steps (all Wilder-smoothed with period ``n``):

        +DM_t = up   if (up > down and up > 0) else 0,   up   = high_t - high_{t-1}
        -DM_t = down if (down > up and down > 0) else 0, down = low_{t-1} - low_t
        +DI   = 100 * RMA(+DM) / ATR
        -DI   = 100 * RMA(-DM) / ATR
        DX    = 100 * |+DI - -DI| / (+DI + -DI)
        ADX   = RMA_n(DX)

    Args:
        high: Per-bar high prices.
        low: Per-bar low prices.
        close: Per-bar close prices.
        n: Wilder period (default 14, clamped to ``>= 1``).

    Returns:
        The latest ADX as a float in ``[0, 100]``. Returns ``0.0`` (no
        directional strength) when there are too few bars or the system is
        degenerate.
    """
    h, l, c = _align_ohlc(high, low, close)
    L = c.size
    if L < 2:
        return 0.0
    period = max(1, int(n))

    up = h[1:] - h[:-1]
    down = l[:-1] - l[1:]
    plus_dm = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0.0), down, 0.0)

    tr = true_range(h, l, c)[1:]  # drop the seed TR to align with DM (length L-1)

    atr_s = _wilder_rma(tr, period)
    plus_dm_s = _wilder_rma(plus_dm, period)
    minus_dm_s = _wilder_rma(minus_dm, period)

    # Avoid divide-by-zero: where ATR is ~0 the bar is flat -> no direction.
    safe_atr = np.where(atr_s > _EPS, atr_s, np.nan)
    plus_di = 100.0 * plus_dm_s / safe_atr
    minus_di = 100.0 * minus_dm_s / safe_atr
    plus_di = np.nan_to_num(plus_di, nan=0.0, posinf=0.0, neginf=0.0)
    minus_di = np.nan_to_num(minus_di, nan=0.0, posinf=0.0, neginf=0.0)

    di_sum = plus_di + minus_di
    dx = np.where(di_sum > _EPS, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)
    dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)

    adx_series = _wilder_rma(dx, period)
    if adx_series.size == 0:
        return 0.0
    value = _safe_float(adx_series[-1], 0.0)
    return min(100.0, max(0.0, value))


def adx_components(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    n: int = 14,
) -> tuple[float, float, float]:
    """Latest ``(+DI, -DI, ADX)`` triple from the directional movement system.

    Convenience companion to :func:`adx` for strategies (e.g. ``adx-trend-strength``)
    that need the directional indicators for the trend *direction* as well as the
    ADX *strength*. See :func:`adx` for the formulas.

    Args:
        high: Per-bar high prices.
        low: Per-bar low prices.
        close: Per-bar close prices.
        n: Wilder period (default 14, clamped to ``>= 1``).

    Returns:
        A tuple ``(plus_di, minus_di, adx)`` of latest-bar floats. ``+DI`` / ``-DI``
        are non-negative; ``adx`` is in ``[0, 100]``. All finite; ``(0, 0, 0)``
        for too-few bars.
    """
    h, l, c = _align_ohlc(high, low, close)
    L = c.size
    if L < 2:
        return 0.0, 0.0, 0.0
    period = max(1, int(n))

    up = h[1:] - h[:-1]
    down = l[:-1] - l[1:]
    plus_dm = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0.0), down, 0.0)

    tr = true_range(h, l, c)[1:]
    atr_s = _wilder_rma(tr, period)
    plus_dm_s = _wilder_rma(plus_dm, period)
    minus_dm_s = _wilder_rma(minus_dm, period)

    safe_atr = np.where(atr_s > _EPS, atr_s, np.nan)
    plus_di = np.nan_to_num(100.0 * plus_dm_s / safe_atr, nan=0.0, posinf=0.0, neginf=0.0)
    minus_di = np.nan_to_num(100.0 * minus_dm_s / safe_atr, nan=0.0, posinf=0.0, neginf=0.0)

    di_sum = plus_di + minus_di
    dx = np.where(di_sum > _EPS, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)
    dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
    adx_series = _wilder_rma(dx, period)

    pdi = max(0.0, _safe_float(plus_di[-1], 0.0))
    mdi = max(0.0, _safe_float(minus_di[-1], 0.0))
    adx_val = min(100.0, max(0.0, _safe_float(adx_series[-1], 0.0)))
    return pdi, mdi, adx_val


def donchian(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    n: int = 20,
) -> tuple[float, float]:
    """Donchian channel (latest upper / lower band).

    The Donchian channel is the highest high and lowest low over the trailing
    ``n`` bars *excluding* the current bar (the classic Turtle breakout uses the
    prior ``n`` days so the current bar can break out of the channel):

        upper = max(high_{t-n} .. high_{t-1})
        lower = min(low_{t-n}  .. low_{t-1})

    When fewer than ``n + 1`` bars exist the window shrinks to all-but-the-last
    bar; with a single bar the band collapses to that bar's high / low.

    Args:
        high: Per-bar high prices.
        low: Per-bar low prices.
        n: Channel look-back (default 20, clamped to ``>= 1``).

    Returns:
        A tuple ``(upper, lower)`` of latest-bar floats with ``upper >= lower``.
        ``(0.0, 0.0)`` when there are no bars.
    """
    h = _as_array(high)
    l = _as_array(low)
    n_bars = min(h.size, l.size)
    if n_bars == 0:
        return 0.0, 0.0
    h = np.nan_to_num(h[-n_bars:], nan=0.0, posinf=0.0, neginf=0.0)
    l = np.nan_to_num(l[-n_bars:], nan=0.0, posinf=0.0, neginf=0.0)
    if n_bars == 1:
        u, lo = float(h[0]), float(l[0])
        return (max(u, lo), min(u, lo))

    window = max(1, int(n))
    # Exclude the current (last) bar: look at the prior `window` bars.
    prior_h = h[:-1]
    prior_l = l[:-1]
    win_h = prior_h[-window:]
    win_l = prior_l[-window:]
    upper = _safe_float(np.max(win_h), 0.0)
    lower = _safe_float(np.min(win_l), 0.0)
    if upper < lower:
        upper, lower = lower, upper
    return upper, lower


def supertrend(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    n: int = 10,
    mult: float = 3.0,
) -> tuple[float, int]:
    """Supertrend trailing-stop level and trend direction (latest value).

    Builds the Supertrend from ATR(``n``) and the ``hl2`` midline, carrying the
    final upper / lower bands forward per the standard rules:

        hl2        = (high + low) / 2
        basicUpper = hl2 + mult * ATR
        basicLower = hl2 - mult * ATR
        finalUpper_t = basicUpper_t  if basicUpper_t < finalUpper_{t-1}
                                        or close_{t-1} > finalUpper_{t-1}
                       else finalUpper_{t-1}
        finalLower_t = basicLower_t  if basicLower_t > finalLower_{t-1}
                                        or close_{t-1} < finalLower_{t-1}
                       else finalLower_{t-1}

    Trend flips up (direction ``+1``, line = finalLower) when close crosses above
    the prior finalUpper, and down (direction ``-1``, line = finalUpper) when
    close crosses below the prior finalLower; otherwise the prior trend carries.

    Args:
        high: Per-bar high prices.
        low: Per-bar low prices.
        close: Per-bar close prices.
        n: ATR period (default 10, clamped to ``>= 1``).
        mult: ATR band multiplier (default 3.0).

    Returns:
        A tuple ``(level, direction)`` where ``level`` is the latest Supertrend
        line (the active trailing stop) and ``direction`` is ``+1`` (bullish) or
        ``-1`` (bearish). For too-few bars returns ``(last_close, +1)``.
    """
    h, l, c = _align_ohlc(high, low, close)
    L = c.size
    if L == 0:
        return 0.0, 1
    if L < 2:
        return float(c[-1]), 1

    m = _safe_float(mult, 3.0)
    atr_series = atr(h, l, c, n)
    hl2 = (h + l) / 2.0
    basic_upper = hl2 + m * atr_series
    basic_lower = hl2 - m * atr_series

    final_upper = np.empty(L, dtype=np.float64)
    final_lower = np.empty(L, dtype=np.float64)
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    for t in range(1, L):
        if basic_upper[t] < final_upper[t - 1] or c[t - 1] > final_upper[t - 1]:
            final_upper[t] = basic_upper[t]
        else:
            final_upper[t] = final_upper[t - 1]
        if basic_lower[t] > final_lower[t - 1] or c[t - 1] < final_lower[t - 1]:
            final_lower[t] = basic_lower[t]
        else:
            final_lower[t] = final_lower[t - 1]

    direction = 1  # +1 bullish (line = lower band), -1 bearish (line = upper)
    line = final_lower[0]
    for t in range(1, L):
        if direction == 1:
            if c[t] < final_lower[t]:
                direction = -1
                line = final_upper[t]
            else:
                line = final_lower[t]
        else:
            if c[t] > final_upper[t]:
                direction = 1
                line = final_lower[t]
            else:
                line = final_upper[t]

    return _safe_float(line, float(c[-1])), int(direction)


def ichimoku(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    tenkan: int = 9,
    kijun: int = 26,
    span_b: int = 52,
    displacement: int = 26,
) -> dict[str, float]:
    """Ichimoku Kinko Hyo lines and cloud position (latest values).

    Lines (each midpoint = (highest high + lowest low) / 2 over its window):

        tenkan   = midpoint over the last ``tenkan`` bars (conversion line)
        kijun    = midpoint over the last ``kijun`` bars (base line)
        senkou_a = (tenkan + kijun) / 2          (leading span A)
        senkou_b = midpoint over the last ``span_b`` bars  (leading span B)

    ``senkou_a`` / ``senkou_b`` form the cloud (kumo). ``cloud_pos`` summarizes
    where the current close sits relative to the cloud:

        +1  close above both spans (above the cloud — bullish)
         0  close inside the cloud (neutral)
        -1  close below both spans (below the cloud — bearish)

    Note: the classic indicator plots the cloud ``displacement`` bars ahead; for
    a per-bar signal the *current* cloud (spans computed from the latest window)
    is compared to the current close. ``displacement`` is accepted for API
    completeness but does not shift the latest-value comparison.

    Args:
        high: Per-bar high prices.
        low: Per-bar low prices.
        close: Per-bar close prices.
        tenkan: Conversion-line window (default 9).
        kijun: Base-line window (default 26).
        span_b: Leading-span-B window (default 52).
        displacement: Cloud displacement (default 26; see note above).

    Returns:
        A dict with finite float keys ``tenkan``, ``kijun``, ``senkou_a``,
        ``senkou_b``, ``cloud_pos`` (``+1``/``0``/``-1`` as a float). For no bars
        every value is ``0.0``.
    """
    h, l, c = _align_ohlc(high, low, close)
    L = c.size
    if L == 0:
        return {
            "tenkan": 0.0,
            "kijun": 0.0,
            "senkou_a": 0.0,
            "senkou_b": 0.0,
            "cloud_pos": 0.0,
        }

    def _midpoint(window: int) -> float:
        w = max(1, min(int(window), L))
        seg_h = h[-w:]
        seg_l = l[-w:]
        return (_safe_float(np.max(seg_h), 0.0) + _safe_float(np.min(seg_l), 0.0)) / 2.0

    ten = _midpoint(tenkan)
    kij = _midpoint(kijun)
    senkou_a = (ten + kij) / 2.0
    senkou_b = _midpoint(span_b)

    last = float(c[-1])
    cloud_top = max(senkou_a, senkou_b)
    cloud_bot = min(senkou_a, senkou_b)
    if last > cloud_top:
        cloud_pos = 1.0
    elif last < cloud_bot:
        cloud_pos = -1.0
    else:
        cloud_pos = 0.0

    return {
        "tenkan": _safe_float(ten, 0.0),
        "kijun": _safe_float(kij, 0.0),
        "senkou_a": _safe_float(senkou_a, 0.0),
        "senkou_b": _safe_float(senkou_b, 0.0),
        "cloud_pos": cloud_pos,
    }


def williams_r(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    n: int = 14,
) -> float:
    """Williams %R over the trailing ``n`` bars (latest value).

    Formula:

        %R = (HighestHigh_n - Close) / (HighestHigh_n - LowestLow_n) * -100

    ``%R`` ranges from ``-100`` (close at the period low — oversold) to ``0``
    (close at the period high — overbought).

    Args:
        high: Per-bar high prices.
        low: Per-bar low prices.
        close: Per-bar close prices.
        n: Look-back period (default 14, clamped to ``>= 1``).

    Returns:
        The latest %R as a float in ``[-100, 0]``. Returns the neutral midpoint
        ``-50.0`` for too-few bars or a flat (zero-range) window.
    """
    h, l, c = _align_ohlc(high, low, close)
    L = c.size
    if L == 0:
        return -50.0
    window = max(1, min(int(n), L))
    hh = _safe_float(np.max(h[-window:]), 0.0)
    ll = _safe_float(np.min(l[-window:]), 0.0)
    rng = hh - ll
    if rng <= _EPS or not math.isfinite(rng):
        return -50.0
    last = float(c[-1])
    value = (hh - last) / rng * -100.0
    value = _safe_float(value, -50.0)
    return min(0.0, max(-100.0, value))


def stochastic(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    k: int = 14,
    d: int = 3,
) -> tuple[float, float]:
    """Slow stochastic oscillator %K / %D (latest values).

    Fast %K over a rolling ``k``-bar window:

        fast%K_t = (Close_t - LowestLow_k) / (HighestHigh_k - LowestLow_k) * 100

    The "slow" oscillator smooths fast %K by an SMA of length ``d`` to get the
    reported (slow) %K, and %D is a further SMA of length ``d`` of slow %K — the
    standard (14, 3, 3) parameterization.

    Args:
        high: Per-bar high prices.
        low: Per-bar low prices.
        close: Per-bar close prices.
        k: %K look-back window (default 14, clamped to ``>= 1``).
        d: Smoothing length for slow %K and for %D (default 3, clamped to
            ``>= 1``).

    Returns:
        A tuple ``(percent_k, percent_d)`` of latest-bar floats in ``[0, 100]``.
        Returns ``(50.0, 50.0)`` (neutral) for too-few bars.
    """
    h, l, c = _align_ohlc(high, low, close)
    L = c.size
    if L == 0:
        return 50.0, 50.0
    k_win = max(1, int(k))
    d_smooth = max(1, int(d))

    # Rolling highest-high / lowest-low over a trailing k-bar window, aligned to
    # each bar. For the leading region (< k bars) use the expanding window.
    fast_k = np.empty(L, dtype=np.float64)
    for t in range(L):
        start = max(0, t - k_win + 1)
        hh = float(np.max(h[start : t + 1]))
        ll = float(np.min(l[start : t + 1]))
        rng = hh - ll
        if rng <= _EPS or not math.isfinite(rng):
            fast_k[t] = 50.0
        else:
            val = (float(c[t]) - ll) / rng * 100.0
            fast_k[t] = min(100.0, max(0.0, _safe_float(val, 50.0)))

    slow_k = _rolling_sma(fast_k, d_smooth)
    percent_d = _rolling_sma(slow_k, d_smooth)

    pk = min(100.0, max(0.0, _safe_float(slow_k[-1], 50.0))) if slow_k.size else 50.0
    pd = min(100.0, max(0.0, _safe_float(percent_d[-1], 50.0))) if percent_d.size else 50.0
    return pk, pd


def cci(
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    close: np.ndarray | list[float],
    n: int = 20,
) -> float:
    """Commodity Channel Index over the trailing ``n`` bars (latest value).

    Formula:

        TP   = (High + Low + Close) / 3                 (typical price)
        CCI  = (TP_t - SMA(TP, n)_t) / (0.015 * MeanDev(TP, n)_t)

    where ``MeanDev`` is the mean absolute deviation of the typical price from
    its SMA over the window. The ``0.015`` constant scales ~70-80% of values into
    ``[-100, +100]`` for normal data.

    Args:
        high: Per-bar high prices.
        low: Per-bar low prices.
        close: Per-bar close prices.
        n: Look-back period (default 20, clamped to ``>= 1``).

    Returns:
        The latest CCI as a float, clamped to ``[-500, 500]`` to bound a
        pathological reading. Returns ``0.0`` for too-few bars or a zero-deviation
        (flat) window.
    """
    h, l, c = _align_ohlc(high, low, close)
    L = c.size
    if L == 0:
        return 0.0
    window = max(1, min(int(n), L))
    tp = (h + l + c) / 3.0
    win = tp[-window:]
    mean_tp = float(np.mean(win))
    mean_dev = float(np.mean(np.abs(win - mean_tp)))
    if mean_dev <= _EPS or not math.isfinite(mean_dev):
        return 0.0
    last_tp = float(tp[-1])
    value = (last_tp - mean_tp) / (0.015 * mean_dev)
    value = _safe_float(value, 0.0)
    return min(500.0, max(-500.0, value))


def keltner(
    close: np.ndarray | list[float],
    high: np.ndarray | list[float],
    low: np.ndarray | list[float],
    n: int = 20,
    mult: float = 2.0,
) -> tuple[float, float, float]:
    """Keltner channel (latest middle / upper / lower bands).

    Formula:

        middle = EMA(Close, n)
        band   = mult * ATR(High, Low, Close, n)
        upper  = middle + band
        lower  = middle - band

    Note: the ATR period here follows ``n`` for a single-parameter call; the
    ``keltner-reversion`` strategy in the catalog uses EMA(20) with ATR(10), so
    callers that need a distinct ATR period should compute :func:`atr` and
    :func:`ema` directly. This helper uses one ``n`` for both, which is the
    common simplified Keltner definition.

    Args:
        close: Per-bar close prices.
        high: Per-bar high prices.
        low: Per-bar low prices.
        n: EMA / ATR period (default 20, clamped to ``>= 1``).
        mult: ATR band multiplier (default 2.0).

    Returns:
        A tuple ``(mid, upper, lower)`` of latest-bar floats with
        ``upper >= mid >= lower``. ``(0.0, 0.0, 0.0)`` for no bars.
    """
    h, l, c = _align_ohlc(high, low, close)
    L = c.size
    if L == 0:
        return 0.0, 0.0, 0.0
    m = _safe_float(mult, 2.0)
    mid_series = ema(c, n)
    atr_series = atr(h, l, c, n)
    mid = _safe_float(mid_series[-1], float(c[-1])) if mid_series.size else float(c[-1])
    band = m * (_safe_float(atr_series[-1], 0.0) if atr_series.size else 0.0)
    band = max(0.0, band)
    upper = mid + band
    lower = mid - band
    return _safe_float(mid, 0.0), _safe_float(upper, 0.0), _safe_float(lower, 0.0)


def obv(
    close: np.ndarray | list[float],
    volume: np.ndarray | list[float],
) -> np.ndarray:
    """On-Balance Volume cumulative series.

    Formula (with ``prevClose`` = the prior bar's close):

        OBV_0 = 0
        OBV_t = OBV_{t-1} + volume_t   if close_t > prevClose
              = OBV_{t-1} - volume_t   if close_t < prevClose
              = OBV_{t-1}              if close_t == prevClose

    Args:
        close: Per-bar close prices.
        volume: Per-bar volumes (non-finite / negative volumes treated as 0).

    Returns:
        A 1-D ``float64`` array of cumulative OBV aligned to the inputs (length
        ``min(len(close), len(volume))``). ``OBV[0] = 0``. Empty if no aligned
        bars; all values finite.
    """
    c = _as_array(close)
    v = _as_array(volume)
    n = min(c.size, v.size)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    c = np.nan_to_num(c[-n:], nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v[-n:], nan=0.0, posinf=0.0, neginf=0.0)
    v = np.maximum(v, 0.0)  # volume is non-negative
    if n == 1:
        return np.zeros(1, dtype=np.float64)

    diff = np.diff(c)
    direction = np.sign(diff)  # +1 up, -1 down, 0 flat
    signed_vol = direction * v[1:]
    out = np.empty(n, dtype=np.float64)
    out[0] = 0.0
    out[1:] = np.cumsum(signed_vol)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def obv_slope(
    close: np.ndarray | list[float],
    volume: np.ndarray | list[float],
    n: int = 20,
) -> float:
    """Normalized slope of OBV over the trailing ``n`` bars (latest value).

    Measures the recent OBV trend, scaled by the typical per-bar OBV change so
    the result is a dimensionless rate comparable across assets:

        slope = (OBV_t - OBV_{t-n}) / (n * mean(|ΔOBV| over the window))

    A positive value indicates accumulation (rising OBV — bullish volume
    confirmation); a negative value indicates distribution.

    Args:
        close: Per-bar close prices.
        volume: Per-bar volumes.
        n: Look-back window (default 20, clamped to ``>= 1``).

    Returns:
        The normalized OBV slope as a float, clamped to ``[-10, 10]``. Returns
        ``0.0`` for too-few bars or a flat (zero-change) OBV window.
    """
    series = obv(close, volume)
    L = series.size
    if L < 2:
        return 0.0
    window = max(1, min(int(n), L - 1))
    recent = series[-(window + 1) :]
    change = recent[-1] - recent[0]
    deltas = np.abs(np.diff(recent))
    mean_abs = float(np.mean(deltas)) if deltas.size else 0.0
    denom = window * mean_abs
    if denom <= _EPS or not math.isfinite(denom):
        return 0.0
    slope = change / denom
    slope = _safe_float(slope, 0.0)
    return min(10.0, max(-10.0, slope))
