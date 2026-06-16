"""Technical-analysis indicators for the GiffMeMoney quant engine.

This module implements the standard chart-based indicators consumed by the
technical and statistical strategy signals: moving averages (SMA / EMA), the
MACD oscillator, the Relative Strength Index (RSI), Bollinger Bands and the
derived %B, the 12-1 momentum factor, and a rolling z-score for mean-reversion.

Every function is numerically defensive. Short or empty price series, constant
series (zero dispersion), and non-finite inputs never raise — they collapse to
safe, finite defaults (neutral readings such as RSI ``50``, %B ``0.5``,
z-score / momentum ``0.0``). Where an indicator is undefined for a window the
function falls back to using the full available history rather than crashing.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "sma",
    "ema",
    "macd",
    "rsi",
    "bollinger",
    "momentum_12_1",
    "zscore",
]


def _clean(prices: np.ndarray | list[float]) -> np.ndarray:
    """Coerce a price input to a clean 1-D float array of finite positives.

    Non-finite entries (NaN / +-inf) and non-positive prices are dropped so the
    indicator math stays well defined.

    Args:
        prices: Sequence of price levels.

    Returns:
        A 1-D ``float64`` array of finite, strictly-positive prices (possibly
        empty).
    """
    arr = np.asarray(prices, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr) & (arr > 0.0)
    return arr[mask]


def _safe_float(value: float, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, falling back to ``default``.

    Args:
        value: Candidate number.
        default: Value substituted when ``value`` is NaN / +-inf.

    Returns:
        ``float(value)`` if finite, else ``default``.
    """
    v = float(value)
    return v if math.isfinite(v) else default


def sma(prices: np.ndarray | list[float], n: int) -> np.ndarray:
    """Simple moving average over a trailing window of ``n`` periods.

    Formula:
        SMA_t = (1 / n) * sum_{i=0}^{n-1} P_{t-i}

    Computed as a rolling mean via cumulative sums. The returned array is aligned
    to the input so that ``out[t]`` is the average of the ``n`` prices ending at
    ``t``; the first ``n - 1`` positions (where a full window is unavailable) are
    filled with the expanding mean of the prices seen so far.

    Args:
        prices: Sequence of price levels (length ``L``).
        n: Window length (clamped to ``[1, L]``).

    Returns:
        A 1-D ``float64`` array of length ``L`` (empty if no valid prices).
    """
    arr = _clean(prices)
    L = arr.size
    if L == 0:
        return np.empty(0, dtype=np.float64)
    window = max(1, min(int(n), L))

    out = np.empty(L, dtype=np.float64)
    cumsum = np.cumsum(arr)
    # Expanding mean for the initial (incomplete) window region.
    for i in range(min(window - 1, L)):
        out[i] = cumsum[i] / (i + 1)
    # Full-window rolling mean for the rest.
    if L >= window:
        rolling = cumsum.copy()
        rolling[window:] = cumsum[window:] - cumsum[:-window]
        out[window - 1 :] = rolling[window - 1 :] / window
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def ema(prices: np.ndarray | list[float], n: int) -> np.ndarray:
    """Exponential moving average with span ``n``.

    Formula:
        alpha = 2 / (n + 1)
        EMA_0 = P_0
        EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}

    Args:
        prices: Sequence of price levels (length ``L``).
        n: Span of the EMA (clamped to ``>= 1``).

    Returns:
        A 1-D ``float64`` array of length ``L`` (empty if no valid prices).
    """
    arr = _clean(prices)
    if arr.size == 0:
        return np.empty(0, dtype=np.float64)
    return _ema_raw(arr, n)


def _ema_raw(values: np.ndarray, n: int) -> np.ndarray:
    """EMA recursion over an arbitrary finite series (no positive-price filter).

    Formula:
        alpha = 2 / (n + 1)
        EMA_0 = x_0
        EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}

    Unlike :func:`ema`, this does not drop non-positive values, so it is safe to
    apply to series that legitimately go negative (e.g. the MACD line). The
    output length always matches the input length.

    Args:
        values: A 1-D ``float64`` array of finite values (length ``L >= 1``).
        n: Span of the EMA (clamped to ``>= 1``).

    Returns:
        A 1-D ``float64`` array of length ``L`` with finite values.
    """
    L = values.size
    span = max(1, int(n))
    alpha = 2.0 / (span + 1.0)
    out = np.empty(L, dtype=np.float64)
    out[0] = values[0]
    for t in range(1, L):
        out[t] = alpha * values[t] + (1.0 - alpha) * out[t - 1]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def macd(
    prices: np.ndarray | list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence Divergence oscillator.

    Formula:
        MACD_t   = EMA(fast)_t - EMA(slow)_t
        Signal_t = EMA(signal) of the MACD line
        Hist_t   = MACD_t - Signal_t

    Args:
        prices: Sequence of price levels (length ``L``).
        fast: Span of the fast EMA (default 12).
        slow: Span of the slow EMA (default 26).
        signal: Span of the signal EMA over the MACD line (default 9).

    Returns:
        A tuple ``(macd_line, signal_line, histogram)`` of 1-D ``float64``
        arrays, each length ``L`` (all empty if no valid prices). All values are
        finite.
    """
    arr = _clean(prices)
    L = arr.size
    if L == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy()

    fast_e = _ema_raw(arr, fast)
    slow_e = _ema_raw(arr, slow)
    macd_line = fast_e - slow_e
    signal_line = _ema_raw(macd_line, signal)
    hist = macd_line - signal_line

    macd_line = np.nan_to_num(macd_line, nan=0.0, posinf=0.0, neginf=0.0)
    signal_line = np.nan_to_num(signal_line, nan=0.0, posinf=0.0, neginf=0.0)
    hist = np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0)
    return macd_line, signal_line, hist


def rsi(prices: np.ndarray | list[float], n: int = 14) -> float:
    """Wilder's Relative Strength Index over the latest ``n`` periods.

    Formula:
        delta_t = P_t - P_{t-1}
        gain    = average of positive deltas over the window
        loss    = average of |negative deltas| over the window
        RS      = gain / loss
        RSI     = 100 - 100 / (1 + RS)

    Uses Wilder's smoothing (an EMA with alpha = 1/n) of the up/down moves over
    the available history. Returns the latest RSI value in ``[0, 100]``.

    Args:
        prices: Sequence of price levels.
        n: Look-back period (default 14, clamped to ``>= 1``).

    Returns:
        The latest RSI as a float in ``[0, 100]``. Returns the neutral value
        ``50.0`` when there are too few prices or all losses are zero (no
        downside) is paired with no upside.
    """
    arr = _clean(prices)
    if arr.size < 2:
        return 50.0
    period = max(1, int(n))

    deltas = np.diff(arr)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)

    # Wilder smoothing (RMA): alpha = 1/period.
    alpha = 1.0 / period
    avg_gain = gains[0]
    avg_loss = losses[0]
    for i in range(1, deltas.size):
        avg_gain = alpha * gains[i] + (1.0 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1.0 - alpha) * avg_loss

    if avg_loss <= 0.0:
        # No downside moves: fully overbought if there were gains, else neutral.
        return 100.0 if avg_gain > 0.0 else 50.0
    rs = avg_gain / avg_loss
    value = 100.0 - 100.0 / (1.0 + rs)
    value = _safe_float(value, 50.0)
    return min(100.0, max(0.0, value))


def bollinger(
    prices: np.ndarray | list[float],
    n: int = 20,
    k: float = 2.0,
) -> tuple[float, float, float, float]:
    """Bollinger Bands and %B for the latest price.

    Formula (over the trailing ``n``-period window ending at the last price):
        mid    = SMA_n
        sigma  = population standard deviation of the window
        upper  = mid + k * sigma
        lower  = mid - k * sigma
        %B     = (P_last - lower) / (upper - lower)

    Args:
        prices: Sequence of price levels.
        n: Window length (default 20, clamped to the available history).
        k: Band width in standard deviations (default 2.0).

    Returns:
        A tuple ``(mid, upper, lower, percent_b)`` of latest-bar floats. For a
        constant (zero-dispersion) window the bands collapse to ``mid`` and
        ``percent_b`` defaults to the neutral ``0.5``. All values are finite.
    """
    arr = _clean(prices)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.5
    window = max(1, min(int(n), arr.size))
    win = arr[-window:]
    last = float(arr[-1])

    mid = float(np.mean(win))
    sigma = float(np.std(win))  # population std (ddof=0)
    width = float(k) * sigma
    upper = mid + width
    lower = mid - width

    band = upper - lower
    if band <= 0.0 or not math.isfinite(band):
        percent_b = 0.5
    else:
        percent_b = (last - lower) / band
    percent_b = _safe_float(percent_b, 0.5)
    # %B can legitimately exceed [0,1] outside the bands; clamp lightly so a
    # pathological value cannot blow up downstream scoring.
    percent_b = min(3.0, max(-2.0, percent_b))

    return (
        _safe_float(mid),
        _safe_float(upper),
        _safe_float(lower),
        percent_b,
    )


def momentum_12_1(prices: np.ndarray | list[float]) -> float:
    """12-1 momentum: trailing 12-month return excluding the most recent month.

    Formula (with 1 month ~= 21 trading days, 12 months ~= 252):
        momentum = P_{t-21} / P_{t-252} - 1

    This is the classic Jegadeesh-Titman / cross-sectional momentum factor: the
    return over the window from 12 months ago to 1 month ago, skipping the last
    month to avoid short-term reversal. Falls back to the full-history return
    when fewer than ~252 prices are available.

    Args:
        prices: Sequence of price levels.

    Returns:
        The 12-1 momentum as a decimal return (e.g. ``0.18`` for +18%). Returns
        ``0.0`` for insufficient/degenerate data. Clamped to a sane range.
    """
    arr = _clean(prices)
    L = arr.size
    if L < 2:
        return 0.0

    skip = 21   # most recent month excluded
    lookback = 252  # 12 months

    if L > lookback:
        start = arr[-(lookback + 1)]
        end = arr[-(skip + 1)]
    else:
        # Not enough history for a full 12-1 window: use the longest window we
        # can while still skipping roughly the last month when possible.
        end_idx = max(0, L - 1 - skip) if L > skip + 1 else L - 1
        start = arr[0]
        end = arr[end_idx]

    if start <= 0.0 or not math.isfinite(start):
        return 0.0
    mom = end / start - 1.0
    mom = _safe_float(mom, 0.0)
    return min(50.0, max(-1.0, mom))


def zscore(prices: np.ndarray | list[float], n: int = 60) -> float:
    """Rolling z-score of the latest price within a trailing ``n``-window.

    Formula (over the window of the last ``n`` prices):
        mu    = mean(window)
        sigma = population standard deviation(window)
        z     = (P_last - mu) / sigma

    A high positive z means the price is stretched above its recent mean
    (mean-reversion bearish); a negative z means it is depressed (bullish).

    Args:
        prices: Sequence of price levels.
        n: Window length (default 60, clamped to the available history).

    Returns:
        The latest z-score as a float, clamped to ``[-10, 10]``. Returns ``0.0``
        for a constant window (zero dispersion) or too few prices.
    """
    arr = _clean(prices)
    if arr.size < 2:
        return 0.0
    window = max(2, min(int(n), arr.size))
    win = arr[-window:]
    last = float(arr[-1])

    mu = float(np.mean(win))
    sigma = float(np.std(win))  # population std
    if sigma <= 0.0 or not math.isfinite(sigma):
        return 0.0
    z = (last - mu) / sigma
    z = _safe_float(z, 0.0)
    return min(10.0, max(-10.0, z))
