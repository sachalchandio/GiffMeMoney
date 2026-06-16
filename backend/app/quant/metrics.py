"""Risk and risk-adjusted performance metrics on daily return series.

Every function in this module takes a 1-D array of *daily* returns (or, for the
drawdown family, a price series) and returns a single ``float``. Where a metric
is conventionally annualized (volatility, Sharpe, Sortino, Treynor, Jensen
alpha, information ratio, Calmar) the annualized value is returned using the
square-root-of-time / linear-in-time conventions from
:mod:`app.quant.returns` (``TRADING_DAYS = 252``).

The formulas implemented are the standard textbook definitions:

    beta              = Cov(r_a, r_m) / Var(r_m)
    annual_volatility = std(r) * sqrt(252)
    sharpe            = mean(r - rf_d) / std(r - rf_d) * sqrt(252)
    sortino           = mean(r - rf_d) / downside_dev(r, rf_d) * sqrt(252)
    downside_dev      = sqrt(mean(min(r - mar, 0)^2))            (daily)
    treynor           = (mean(r - rf_d) * 252) / beta(r, r_m)
    jensen_alpha      = annualized[ mean(r) - (rf_d + beta*(mean(r_m) - rf_d)) ]
    information_ratio = mean(r - r_b) / std(r - r_b) * sqrt(252)
    max_drawdown      = min over t of  P_t / running_max(P)_t - 1   (<= 0)
    calmar            = annual_return(mean log r) / |max_drawdown|

All functions are numerically defensive: empty / too-short inputs, zero
variance, divide-by-zero and non-finite values never raise — they return a
finite, sane default (typically ``0.0``). Sample standard deviations use the
population estimator (``ddof=0``) for stability on short windows; this matches
the closed-form test fixtures (e.g. a constant-excess series gives the
documented Sharpe).
"""

from __future__ import annotations

import math

import numpy as np

from app.quant.returns import TRADING_DAYS, annualize_return

__all__ = [
    "beta",
    "annual_volatility",
    "sharpe",
    "sortino",
    "downside_deviation",
    "treynor",
    "jensen_alpha",
    "information_ratio",
    "max_drawdown",
    "calmar",
]

# Smallest std/denominator we treat as non-zero; below this the series is
# effectively constant and the ratio would blow up, so we collapse to a default.
_EPS: float = 1e-12

#: sqrt(252) precomputed for the square-root-of-time scaling.
_SQRT_DAYS: float = math.sqrt(TRADING_DAYS)


def _clean(returns: np.ndarray | list[float]) -> np.ndarray:
    """Coerce a return input to a clean 1-D float array of finite values.

    Args:
        returns: Sequence of returns.

    Returns:
        A 1-D ``float64`` array with NaN/inf entries removed (possibly empty).
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Truncate two arrays to their common (trailing-aligned) length.

    Both inputs are first cleaned of non-finite values independently, then
    truncated to the same length by keeping the last ``min(len)`` elements of
    each so the most recent observations stay aligned.

    Args:
        a: First return series.
        b: Second return series.

    Returns:
        A pair of equal-length finite arrays (possibly empty).
    """
    ca = _clean(a)
    cb = _clean(b)
    n = min(ca.size, cb.size)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    return ca[-n:], cb[-n:]


def beta(asset_ret: np.ndarray | list[float], market_ret: np.ndarray | list[float]) -> float:
    """Compute the CAPM market beta of an asset.

    Formula:
        beta = Cov(r_asset, r_market) / Var(r_market)

    Both series are aligned to their common trailing length. Covariance and
    variance use the population estimator (``ddof=0``).

    Args:
        asset_ret: Daily returns of the asset.
        market_ret: Daily returns of the market proxy.

    Returns:
        The beta as a ``float``. Returns ``1.0`` (market-neutral default) when
        there are fewer than two aligned observations or the market variance is
        effectively zero.
    """
    a, m = _align(asset_ret, market_ret)
    if a.size < 2:
        return 1.0
    var_m = float(np.var(m))
    if var_m < _EPS:
        return 1.0
    cov = float(np.mean((a - a.mean()) * (m - m.mean())))
    result = cov / var_m
    return result if math.isfinite(result) else 1.0


def annual_volatility(returns: np.ndarray | list[float]) -> float:
    """Annualize the volatility of a daily return series.

    Formula:
        sigma_annual = std(r) * sqrt(252)

    Uses the population standard deviation (``ddof=0``).

    Args:
        returns: Daily returns.

    Returns:
        Annualized volatility as a decimal (e.g. ``0.20`` for 20%). Returns
        ``0.0`` for fewer than two observations or non-finite results.
    """
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    sd = float(np.std(r))
    result = sd * _SQRT_DAYS
    return result if math.isfinite(result) and result >= 0.0 else 0.0


def downside_deviation(returns: np.ndarray | list[float], mar: float = 0.0) -> float:
    """Compute the (daily) downside deviation below a minimum acceptable return.

    Formula:
        downside_dev = sqrt( mean( min(r_t - mar, 0)^2 ) )

    Only returns below ``mar`` contribute; observations at or above ``mar`` enter
    the mean as zero (standard Sortino convention, divisor = full sample size).

    Args:
        returns: Daily returns.
        mar: Minimum acceptable (daily) return threshold. Defaults to ``0.0``.

    Returns:
        The daily downside deviation (non-negative). Returns ``0.0`` for an
        empty series or non-finite results.
    """
    r = _clean(returns)
    if r.size == 0:
        return 0.0
    m = float(mar) if math.isfinite(mar) else 0.0
    shortfall = np.minimum(r - m, 0.0)
    dd = math.sqrt(float(np.mean(shortfall * shortfall)))
    return dd if math.isfinite(dd) and dd >= 0.0 else 0.0


def sharpe(returns: np.ndarray | list[float], rf_daily: float) -> float:
    """Compute the annualized Sharpe ratio of a daily return series.

    Formula:
        excess = r_t - rf_daily
        sharpe = mean(excess) / std(excess) * sqrt(252)

    Uses the population standard deviation of excess returns (``ddof=0``). A
    constant-excess series (zero std) returns ``0.0`` rather than diverging.

    Args:
        returns: Daily returns.
        rf_daily: Daily risk-free rate (decimal).

    Returns:
        The annualized Sharpe ratio. Returns ``0.0`` for fewer than two
        observations, zero volatility, or non-finite results.
    """
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    rf = float(rf_daily) if math.isfinite(rf_daily) else 0.0
    excess = r - rf
    sd = float(np.std(excess))
    if sd < _EPS:
        return 0.0
    result = (float(np.mean(excess)) / sd) * _SQRT_DAYS
    return result if math.isfinite(result) else 0.0


def sortino(returns: np.ndarray | list[float], rf_daily: float) -> float:
    """Compute the annualized Sortino ratio of a daily return series.

    Formula:
        excess  = r_t - rf_daily
        sortino = mean(excess) / downside_dev(r, mar=rf_daily) * sqrt(252)

    The denominator penalizes only downside volatility (returns below the
    risk-free rate), unlike Sharpe which uses total volatility.

    Args:
        returns: Daily returns.
        rf_daily: Daily risk-free rate, used both as the excess-return baseline
            and as the downside MAR.

    Returns:
        The annualized Sortino ratio. Returns ``0.0`` for fewer than two
        observations, zero downside deviation, or non-finite results.
    """
    r = _clean(returns)
    if r.size < 2:
        return 0.0
    rf = float(rf_daily) if math.isfinite(rf_daily) else 0.0
    excess_mean = float(np.mean(r - rf))
    dd = downside_deviation(r, mar=rf)
    if dd < _EPS:
        return 0.0
    result = (excess_mean / dd) * _SQRT_DAYS
    return result if math.isfinite(result) else 0.0


def treynor(
    returns: np.ndarray | list[float],
    market_ret: np.ndarray | list[float],
    rf_daily: float,
) -> float:
    """Compute the annualized Treynor ratio of a daily return series.

    Formula:
        beta    = Cov(r, r_m) / Var(r_m)
        treynor = (mean(r - rf_daily) * 252) / beta

    Excess return is annualized linearly (mean daily excess times 252) and
    divided by systematic risk (beta) rather than total risk.

    Args:
        returns: Daily returns of the asset.
        market_ret: Daily returns of the market proxy.
        rf_daily: Daily risk-free rate (decimal).

    Returns:
        The annualized Treynor ratio. Returns ``0.0`` for fewer than two aligned
        observations, near-zero beta, or non-finite results.
    """
    a, m = _align(returns, market_ret)
    if a.size < 2:
        return 0.0
    rf = float(rf_daily) if math.isfinite(rf_daily) else 0.0
    b = beta(a, m)
    if abs(b) < _EPS:
        return 0.0
    excess_annual = float(np.mean(a - rf)) * TRADING_DAYS
    result = excess_annual / b
    return result if math.isfinite(result) else 0.0


def jensen_alpha(
    returns: np.ndarray | list[float],
    market_ret: np.ndarray | list[float],
    rf_daily: float,
) -> float:
    """Compute the annualized Jensen's alpha of a daily return series.

    Formula:
        beta        = Cov(r, r_m) / Var(r_m)
        alpha_daily = mean(r) - [ rf_daily + beta * (mean(r_m) - rf_daily) ]
        alpha       = annualize_return(alpha_daily)

    The daily alpha is the average return in excess of the CAPM-predicted return
    given the realized market excess return; it is then annualized by geometric
    compounding (consistent with :func:`app.quant.returns.annualize_return`).

    Args:
        returns: Daily returns of the asset.
        market_ret: Daily returns of the market proxy.
        rf_daily: Daily risk-free rate (decimal).

    Returns:
        The annualized Jensen's alpha as a decimal. Returns ``0.0`` for fewer
        than two aligned observations or non-finite results.
    """
    a, m = _align(returns, market_ret)
    if a.size < 2:
        return 0.0
    rf = float(rf_daily) if math.isfinite(rf_daily) else 0.0
    b = beta(a, m)
    predicted_daily = rf + b * (float(np.mean(m)) - rf)
    alpha_daily = float(np.mean(a)) - predicted_daily
    if not math.isfinite(alpha_daily):
        return 0.0
    result = annualize_return(alpha_daily)
    return result if math.isfinite(result) else 0.0


def information_ratio(
    returns: np.ndarray | list[float],
    bench_ret: np.ndarray | list[float],
) -> float:
    """Compute the annualized information ratio versus a benchmark.

    Formula:
        active = r_t - r_benchmark_t            (active return / tracking error)
        IR     = mean(active) / std(active) * sqrt(252)

    The denominator (``std(active)``) is the tracking error; it uses the
    population estimator (``ddof=0``).

    Args:
        returns: Daily returns of the portfolio / asset.
        bench_ret: Daily returns of the benchmark.

    Returns:
        The annualized information ratio. Returns ``0.0`` for fewer than two
        aligned observations, zero tracking error, or non-finite results.
    """
    a, b = _align(returns, bench_ret)
    if a.size < 2:
        return 0.0
    active = a - b
    te = float(np.std(active))
    if te < _EPS:
        return 0.0
    result = (float(np.mean(active)) / te) * _SQRT_DAYS
    return result if math.isfinite(result) else 0.0


def max_drawdown(prices: np.ndarray | list[float]) -> float:
    """Compute the maximum drawdown of a price series.

    Formula:
        running_max_t = max(P_0, ..., P_t)
        drawdown_t    = P_t / running_max_t - 1
        max_drawdown  = min_t drawdown_t                 (<= 0)

    The result is the most negative peak-to-trough fractional decline observed
    over the series.

    Args:
        prices: Sequence of price levels.

    Returns:
        The maximum drawdown as a non-positive fraction (e.g. ``-0.25`` for a
        25% drawdown). Returns ``0.0`` for fewer than two valid prices, when the
        series never declines, or for non-finite results.
    """
    arr = np.asarray(prices, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    # running_max is strictly positive (arr > 0), so division is safe.
    drawdowns = arr / running_max - 1.0
    mdd = float(np.min(drawdowns))
    if not math.isfinite(mdd):
        return 0.0
    # Drawdown is non-positive by construction; clamp tiny positive float noise.
    return min(0.0, mdd)


def calmar(returns: np.ndarray | list[float], prices: np.ndarray | list[float]) -> float:
    """Compute the Calmar ratio: annualized return over maximum drawdown.

    Formula:
        annual_return = annualize_return(mean(r))     (geometric, exp(mu*252)-1)
        calmar        = annual_return / |max_drawdown(prices)|

    Args:
        returns: Daily returns used for the annualized-return numerator.
        prices: Price series used for the maximum-drawdown denominator.

    Returns:
        The Calmar ratio. Returns ``0.0`` when returns are too short, when the
        maximum drawdown is effectively zero (no decline), or for non-finite
        results.
    """
    r = _clean(returns)
    if r.size < 1:
        return 0.0
    ann_ret = annualize_return(float(np.mean(r)))
    mdd = max_drawdown(prices)
    denom = abs(mdd)
    if denom < _EPS:
        return 0.0
    result = ann_ret / denom
    return result if math.isfinite(result) else 0.0
