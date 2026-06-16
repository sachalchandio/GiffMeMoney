"""Time-series forecasting models: OLS trend and Holt-Winters smoothing.

These two estimators back the ``trend-ols`` statistical strategy. Both are
implemented by hand with numpy (no statsmodels): :func:`ols_trend` fits a
least-squares line to the log-price series and reports its slope, fit quality,
and the implied daily drift; :func:`holt_winters` runs double exponential
smoothing (Holt's linear trend method, no seasonal term) and projects the level
forward.

All functions are numerically defensive: short/empty series, constant series
(zero variance), and non-finite inputs never raise — they return safe finite
defaults (zero slope/drift, ``r2`` of ``0.0``, forecast equal to the last
price).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "ols_trend",
    "holt_winters",
]


def _clean(prices: np.ndarray | list[float]) -> np.ndarray:
    """Coerce a price input to a clean 1-D float array of finite positives.

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


def ols_trend(
    prices: np.ndarray | list[float],
) -> tuple[float, float, float, float]:
    """Ordinary-least-squares trend on the log-price series.

    Regresses log price on a time index ``t = 0, 1, ..., L-1``:

        ln(P_t) = intercept + slope * t + e_t

    fitted via the closed-form OLS estimator. The R^2 is the coefficient of
    determination of that fit. Because the dependent variable is the *log* price,
    ``slope`` is itself the estimated mean daily log return, so the implied daily
    drift is just the slope:

        forecast_drift_daily = slope

    Args:
        prices: Sequence of price levels (length ``L``).

    Returns:
        A tuple ``(slope, intercept, r2, forecast_drift_daily)``:
            * ``slope`` — OLS slope of log price vs. time (~ daily log return).
            * ``intercept`` — OLS intercept (log price at ``t = 0``).
            * ``r2`` — coefficient of determination in ``[0, 1]``.
            * ``forecast_drift_daily`` — implied daily log-return drift (``slope``).
        For fewer than two valid prices or a constant series, returns
        ``(0.0, ln(P_last) or 0.0, 0.0, 0.0)``. All values finite.
    """
    arr = _clean(prices)
    L = arr.size
    if L < 2:
        intercept = _safe_float(math.log(arr[0])) if L == 1 else 0.0
        return 0.0, intercept, 0.0, 0.0

    y = np.log(arr)
    t = np.arange(L, dtype=np.float64)

    t_mean = float(np.mean(t))
    y_mean = float(np.mean(y))
    dt = t - t_mean
    dy = y - y_mean

    denom = float(np.dot(dt, dt))
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0, _safe_float(y_mean), 0.0, 0.0

    slope = float(np.dot(dt, dy)) / denom
    intercept = y_mean - slope * t_mean

    # R^2 = 1 - SS_res / SS_tot.
    ss_tot = float(np.dot(dy, dy))
    if ss_tot <= 0.0 or not math.isfinite(ss_tot):
        r2 = 0.0
    else:
        resid = y - (intercept + slope * t)
        ss_res = float(np.dot(resid, resid))
        r2 = 1.0 - ss_res / ss_tot
        r2 = _safe_float(r2, 0.0)
        r2 = min(1.0, max(0.0, r2))

    slope = _safe_float(slope, 0.0)
    intercept = _safe_float(intercept, 0.0)
    # Drift equals the log-price slope; clamp to a sane daily range.
    drift = min(1.0, max(-1.0, slope))
    return slope, intercept, r2, drift


def holt_winters(
    prices: np.ndarray | list[float],
    alpha: float = 0.3,
    beta: float = 0.1,
    horizon: int = 21,
) -> tuple[float, float, float]:
    """Holt's linear-trend exponential smoothing (double exponential smoothing).

    No seasonal component is used (despite the conventional name). The recursive
    updates over the price series are:

        level_t = alpha * P_t + (1 - alpha) * (level_{t-1} + trend_{t-1})
        trend_t = beta  * (level_t - level_{t-1}) + (1 - beta) * trend_{t-1}

    initialised with ``level_0 = P_0`` and ``trend_0 = P_1 - P_0``. The
    ``h``-step-ahead forecast is the linear extrapolation:

        forecast = level_T + horizon * trend_T

    Args:
        prices: Sequence of price levels (length ``L``).
        alpha: Level smoothing factor in ``[0, 1]`` (default 0.3, clamped).
        beta: Trend smoothing factor in ``[0, 1]`` (default 0.1, clamped).
        horizon: Number of steps ahead to forecast (default 21, clamped ``>= 0``).

    Returns:
        A tuple ``(level, trend, forecast_value)`` of finite floats — the final
        smoothed level, the final smoothed trend (per step), and the
        ``horizon``-step-ahead forecast value. The forecast is floored at a tiny
        positive value so it never goes non-positive. For fewer than two valid
        prices, returns ``(P_last, 0.0, P_last)`` (or zeros for empty input).
    """
    arr = _clean(prices)
    L = arr.size
    if L == 0:
        return 0.0, 0.0, 0.0
    if L == 1:
        p = float(arr[0])
        return p, 0.0, p

    a = min(1.0, max(0.0, _safe_float(alpha, 0.3)))
    b = min(1.0, max(0.0, _safe_float(beta, 0.1)))
    h = max(0, int(horizon))

    level = float(arr[0])
    trend = float(arr[1] - arr[0])
    for t in range(1, L):
        prev_level = level
        level = a * float(arr[t]) + (1.0 - a) * (prev_level + trend)
        trend = b * (level - prev_level) + (1.0 - b) * trend

    level = _safe_float(level, float(arr[-1]))
    trend = _safe_float(trend, 0.0)
    forecast = level + h * trend
    forecast = _safe_float(forecast, level)
    # Prices are positive; keep the forecast strictly positive.
    forecast = max(1e-9, forecast)
    return level, trend, forecast
