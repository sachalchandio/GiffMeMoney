"""Return and annualization helpers plus multi-horizon return projection.

This module is the numerical foundation that the strategy signals and the
analysis engine build on. It converts a price series into period returns,
annualizes daily statistics, and projects an expected-return distribution onto
each of the five product horizons (1D / 1W / 1M / 1Y / 5Y).

All functions are numerically defensive: empty or constant inputs, zero/near-zero
volatility, and non-finite values never raise — they collapse to safe, finite
defaults and outputs are clamped to sane ranges.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

__all__ = [
    "TRADING_DAYS",
    "HORIZON_DAYS",
    "simple_returns",
    "log_returns",
    "annualize_return",
    "annualize_vol",
    "project_horizons",
]

#: Number of trading days in a calendar year (standard finance convention).
TRADING_DAYS: int = 252

#: Mapping of each product horizon label to its length in trading days.
#: 1D=1, 1W=5, 1M=21, 1Y=252, 5Y=1260 (= 5 * 252).
HORIZON_DAYS: dict[str, int] = {
    "1D": 1,
    "1W": 5,
    "1M": 21,
    "1Y": 252,
    "5Y": 1260,
}

# Smallest volatility we treat as non-zero. Below this a series is effectively
# constant and probability/band math would blow up, so we floor to this value.
_VOL_FLOOR: float = 1e-8

# Hard clamp on per-horizon expected return percentage so a pathological drift
# estimate can never emit absurd (e.g. 1e9 %) numbers into the wire DTOs.
_RET_PCT_CLAMP: float = 1.0e6

# z-score for a one-sided 5% / 95% normal quantile (Phi^{-1}(0.95)).
_Z_90: float = 1.645


def _clean_prices(prices: np.ndarray | list[float]) -> np.ndarray:
    """Coerce a price input to a clean 1-D float array of finite positives.

    Non-finite entries (NaN / +-inf) and non-positive prices are dropped so that
    downstream log/ratio math stays well defined.

    Args:
        prices: Sequence of price levels.

    Returns:
        A 1-D ``float64`` array containing only finite, strictly-positive prices
        (possibly empty).
    """
    arr = np.asarray(prices, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr) & (arr > 0.0)
    return arr[mask]


def simple_returns(prices: np.ndarray | list[float]) -> np.ndarray:
    """Compute simple (arithmetic) period returns from a price series.

    Formula:
        r_t = P_t / P_{t-1} - 1

    Args:
        prices: Sequence of price levels (length ``n``).

    Returns:
        A 1-D ``float64`` array of length ``n - 1`` with the simple return for
        each step. Returns an empty array when fewer than two valid prices are
        supplied. Non-finite results are replaced with ``0.0``.
    """
    arr = _clean_prices(prices)
    if arr.size < 2:
        return np.empty(0, dtype=np.float64)
    rets = arr[1:] / arr[:-1] - 1.0
    return np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)


def log_returns(prices: np.ndarray | list[float]) -> np.ndarray:
    """Compute continuously-compounded (log) period returns from prices.

    Formula:
        r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

    Args:
        prices: Sequence of price levels (length ``n``).

    Returns:
        A 1-D ``float64`` array of length ``n - 1`` with the log return for each
        step. Returns an empty array when fewer than two valid prices are
        supplied. Non-finite results are replaced with ``0.0``.
    """
    arr = _clean_prices(prices)
    if arr.size < 2:
        return np.empty(0, dtype=np.float64)
    rets = np.log(arr[1:] / arr[:-1])
    return np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)


def annualize_return(daily_mean: float) -> float:
    """Annualize a mean daily (log) return by geometric compounding.

    Formula:
        R_annual = exp(mu_daily * TRADING_DAYS) - 1

    Treats ``daily_mean`` as a continuously-compounded daily rate so the result
    is the equivalent simple annual return.

    Args:
        daily_mean: Mean daily log return.

    Returns:
        The annualized simple return as a decimal (e.g. ``0.12`` for 12%).
        Non-finite inputs yield ``0.0``; the exponent is clamped to avoid
        overflow.
    """
    if not math.isfinite(daily_mean):
        return 0.0
    exponent = float(daily_mean) * TRADING_DAYS
    # Clamp exponent to keep exp() finite (~e^700 is near float max).
    exponent = max(-700.0, min(700.0, exponent))
    result = math.exp(exponent) - 1.0
    return result if math.isfinite(result) else 0.0


def annualize_vol(daily_std: float) -> float:
    """Annualize a daily return standard deviation via the square-root-of-time rule.

    Formula:
        sigma_annual = sigma_daily * sqrt(TRADING_DAYS)

    Args:
        daily_std: Standard deviation of daily returns.

    Returns:
        The annualized volatility as a decimal. Non-finite or negative inputs
        yield ``0.0``.
    """
    if not math.isfinite(daily_std) or daily_std < 0.0:
        return 0.0
    return float(daily_std) * math.sqrt(TRADING_DAYS)


def project_horizons(mu_daily: float, sigma_daily: float) -> list[dict]:
    """Project a daily return distribution onto each product horizon.

    For a horizon of ``h`` trading days, with daily log-drift ``mu_daily`` and
    daily volatility ``sigma_daily``, the geometric-Brownian projection is:

        expectedReturnPct = (exp(mu_daily * h) - 1) * 100
        spread            = 1.645 * sigma_daily * sqrt(h)        # ~5th/95th pct
        low               = (exp(mu_daily * h - spread) - 1) * 100
        high              = (exp(mu_daily * h + spread) - 1) * 100
        probPositive      = Phi(mu_daily * sqrt(h) / sigma_daily)
        annualizedVol     = sigma_daily * sqrt(252) * 100

    The bands are the lognormal 5th/95th percentiles of the terminal price
    relative to today, expressed as percentage returns. ``probPositive`` is the
    probability that the cumulative log return over the horizon exceeds zero
    under a Normal(mu_daily*h, (sigma_daily*sqrt(h))^2) assumption.

    Args:
        mu_daily: Mean daily log return (drift).
        sigma_daily: Daily return volatility. Floored to a tiny positive value
            so probability/band math stays finite for constant series.

    Returns:
        A list of dicts (one per horizon, in ``HORIZON_DAYS`` order) each shaped
        exactly like the ``ExpectedReturn`` DTO::

            {
              "horizon": str,            # '1D'|'1W'|'1M'|'1Y'|'5Y'
              "expectedReturnPct": float,
              "low": float,              # ~5th percentile return, %
              "high": float,             # ~95th percentile return, %
              "probPositive": float,     # 0..1
              "annualizedVol": float,    # %
            }

        All numbers are finite. ``expectedReturnPct``/``low``/``high`` are
        clamped to a sane range and ``probPositive`` to [0, 1].
    """
    mu = float(mu_daily) if math.isfinite(mu_daily) else 0.0
    sigma = float(sigma_daily) if math.isfinite(sigma_daily) else 0.0
    # Floor volatility to a small positive number to avoid divide-by-zero and to
    # keep band widths/probabilities well defined for (near-)constant series.
    sigma = max(sigma, _VOL_FLOOR)

    annual_vol_pct = sigma * math.sqrt(TRADING_DAYS) * 100.0
    annual_vol_pct = annual_vol_pct if math.isfinite(annual_vol_pct) else 0.0

    out: list[dict] = []
    for label, h in HORIZON_DAYS.items():
        sqrt_h = math.sqrt(h)
        drift_h = mu * h                       # cumulative log drift over horizon
        spread = _Z_90 * sigma * sqrt_h        # half-width of the 5/95 band (log space)

        expected_pct = _expm1_pct(drift_h)
        low_pct = _expm1_pct(drift_h - spread)
        high_pct = _expm1_pct(drift_h + spread)

        # P(cumulative log return > 0) = Phi(mu*h / (sigma*sqrt(h)))
        #                              = Phi(mu*sqrt(h) / sigma)
        z = (mu * sqrt_h) / sigma
        if not math.isfinite(z):
            prob_positive = 0.5
        else:
            prob_positive = float(norm.cdf(z))
        prob_positive = min(1.0, max(0.0, prob_positive))

        out.append(
            {
                "horizon": label,
                "expectedReturnPct": expected_pct,
                "low": low_pct,
                "high": high_pct,
                "probPositive": prob_positive,
                "annualizedVol": annual_vol_pct,
            }
        )
    return out


def _expm1_pct(log_growth: float) -> float:
    """Convert a cumulative log return to a clamped, finite percentage return.

    Formula:
        pct = (exp(log_growth) - 1) * 100

    Args:
        log_growth: Cumulative continuously-compounded return over a horizon.

    Returns:
        The equivalent simple return in percent, clamped to
        ``[-_RET_PCT_CLAMP, _RET_PCT_CLAMP]`` and guaranteed finite. A
        ``log_growth`` of ``-inf`` maps to ``-100`` (total loss).
    """
    if not math.isfinite(log_growth):
        return -100.0 if log_growth < 0 else _RET_PCT_CLAMP
    g = max(-700.0, min(700.0, log_growth))
    pct = (math.exp(g) - 1.0) * 100.0
    if not math.isfinite(pct):
        return _RET_PCT_CLAMP if pct > 0 else -100.0
    return max(-_RET_PCT_CLAMP, min(_RET_PCT_CLAMP, pct))
