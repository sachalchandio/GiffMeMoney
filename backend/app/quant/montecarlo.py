"""Geometric Brownian Motion Monte Carlo simulation.

Simulates forward price paths under GBM, summarizes them into percentile bands
over time, a terminal-price histogram, expected return, tail risk (VaR/CVaR),
and the probability of a positive return. The output of :func:`montecarlo_summary`
matches the ``MonteCarloResult`` wire DTO exactly (camelCase keys, ``t`` = step
index, prices for the bands, percentages where the DTO calls for them).

All functions are numerically defensive: empty/short inputs, zero/near-zero
volatility, non-finite drift, and degenerate distributions never raise — they
collapse to safe finite defaults and outputs are clamped to sane ranges.
"""

from __future__ import annotations

import math

import numpy as np

from app.quant.returns import HORIZON_DAYS, TRADING_DAYS

__all__ = [
    "gbm_paths",
    "montecarlo_summary",
]

# Smallest volatility treated as non-zero; below this a series is effectively
# deterministic and distribution math would degenerate.
_VOL_FLOOR: float = 1e-8

# Clamp on the per-day drift/vol so a pathological estimate cannot produce
# infinite prices through the exponential.
_MU_CLAMP: float = 1.0
_SIGMA_CLAMP: float = 2.0

# Hard clamp on returned percentage figures so the wire DTO never sees absurd
# values from a degenerate estimate.
_PCT_CLAMP: float = 1.0e6


def _safe_float(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, falling back to ``default`` otherwise.

    Args:
        x: Candidate value.
        default: Replacement when ``x`` is NaN/inf.

    Returns:
        A finite float.
    """
    xf = float(x)
    return xf if math.isfinite(xf) else default


def gbm_paths(
    s0: float,
    mu_daily: float,
    sigma_daily: float,
    steps: int,
    sims: int,
    seed: int,
) -> np.ndarray:
    """Simulate Geometric Brownian Motion price paths.

    Each step evolves the log price by a normal increment:

        S_{t+1} = S_t * exp((mu_daily - 0.5 * sigma_daily^2) + sigma_daily * Z),
            Z ~ N(0, 1)

    i.e. the discrete Euler-on-log-price (exact) GBM update with a per-step
    Ito drift correction so that ``E[log(S_t/S_0)] = t * (mu_daily - 0.5 sigma^2)``
    and the median path grows at the geometric rate.

    Args:
        s0: Starting price (must be positive; floored to a tiny positive value).
        mu_daily: Mean daily log return (drift), clamped to ``+/-_MU_CLAMP``.
        sigma_daily: Daily volatility, floored to a tiny positive value and
            clamped to ``_SIGMA_CLAMP``.
        steps: Number of forward steps (each one trading day). Coerced to >= 1.
        sims: Number of simulated paths. Coerced to >= 1.
        seed: Seed for the per-call ``numpy`` Generator (deterministic output).

    Returns:
        A ``(sims, steps + 1)`` ``float64`` array of simulated prices; column 0
        is ``s0`` for every path. All entries are finite and positive.
    """
    s0_f = _safe_float(s0, 1.0)
    if s0_f <= 0.0:
        s0_f = 1.0
    n_steps = max(1, int(steps))
    n_sims = max(1, int(sims))

    mu = _safe_float(mu_daily, 0.0)
    mu = max(-_MU_CLAMP, min(_MU_CLAMP, mu))
    sigma = _safe_float(sigma_daily, 0.0)
    sigma = max(_VOL_FLOOR, min(_SIGMA_CLAMP, sigma))

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    drift = mu - 0.5 * sigma * sigma
    shocks = rng.standard_normal(size=(n_sims, n_steps))
    log_increments = drift + sigma * shocks  # (sims, steps)

    log_paths = np.cumsum(log_increments, axis=1)
    log_paths = np.clip(log_paths, -700.0, 700.0)
    prices = s0_f * np.exp(log_paths)

    # Prepend the starting column.
    out = np.empty((n_sims, n_steps + 1), dtype=np.float64)
    out[:, 0] = s0_f
    out[:, 1:] = prices
    out = np.nan_to_num(out, nan=s0_f, posinf=s0_f, neginf=s0_f)
    # Guard against any non-positive values produced by clamping.
    out[out <= 0.0] = _VOL_FLOOR
    return out


def montecarlo_summary(
    s0: float,
    mu_daily: float,
    sigma_daily: float,
    horizon: str,
    sims: int = 2000,
    seed: int = 0,
) -> dict:
    """Run a GBM Monte Carlo and summarize it into the ``MonteCarloResult`` shape.

    The horizon string maps to a number of trading-day steps via
    ``HORIZON_DAYS`` (1D=1, 1W=5, 1M=21, 1Y=252, 5Y=1260). For each step the
    p5/p25/p50/p75/p95 **price** percentiles form the fan ``bands``. The terminal
    column is histogrammed into ``finalDistribution`` bins. Summary statistics
    are computed from terminal returns ``R = S_T / S_0 - 1``:

        expectedReturnPct = mean(R) * 100
        var95Pct          = -quantile(R, 0.05) * 100   (positive loss fraction, %)
        cvar95Pct         = -mean(R | R <= quantile(R, 0.05)) * 100
        probPositive      = mean(R > 0)

    Args:
        s0: Starting price.
        mu_daily: Mean daily log return (drift).
        sigma_daily: Daily volatility.
        horizon: One of ``'1D' | '1W' | '1M' | '1Y' | '5Y'``. Unknown values
            fall back to ``'1Y'``.
        sims: Number of simulated paths (coerced to >= 1).
        seed: RNG seed for reproducibility.

    Returns:
        A dict matching ``MonteCarloResult`` (without ``symbol``, which the
        caller fills in)::

            {
              "horizon": str,
              "sims": int,
              "steps": int,
              "bands": [{"t": int, "p5":.., "p25":.., "p50":.., "p75":.., "p95":..}],
              "finalDistribution": [{"binStart":.., "binEnd":.., "count": int}],
              "expectedReturnPct": float,
              "var95Pct": float,
              "cvar95Pct": float,
              "probPositive": float,
            }

        All numbers are finite; percentages are clamped and ``probPositive`` lies
        in ``[0, 1]``.
    """
    steps = HORIZON_DAYS.get(horizon, HORIZON_DAYS["1Y"])
    horizon_label = horizon if horizon in HORIZON_DAYS else "1Y"
    n_sims = max(1, int(sims))

    s0_f = _safe_float(s0, 1.0)
    if s0_f <= 0.0:
        s0_f = 1.0

    paths = gbm_paths(s0_f, mu_daily, sigma_daily, steps, n_sims, seed)

    # Percentile bands over time (price space). One band entry per step index.
    pct_levels = [5.0, 25.0, 50.0, 75.0, 95.0]
    band_matrix = np.percentile(paths, pct_levels, axis=0)  # (5, steps+1)
    band_matrix = np.nan_to_num(band_matrix, nan=s0_f, posinf=s0_f, neginf=s0_f)

    bands: list[dict] = []
    n_cols = paths.shape[1]
    for t in range(n_cols):
        bands.append(
            {
                "t": t,
                "p5": float(band_matrix[0, t]),
                "p25": float(band_matrix[1, t]),
                "p50": float(band_matrix[2, t]),
                "p75": float(band_matrix[3, t]),
                "p95": float(band_matrix[4, t]),
            }
        )

    # Terminal prices -> returns.
    terminal = paths[:, -1]
    terminal = terminal[np.isfinite(terminal)]
    if terminal.size == 0:
        terminal = np.array([s0_f], dtype=np.float64)

    returns = terminal / s0_f - 1.0

    expected_return_pct = _clamp_pct(float(np.mean(returns)) * 100.0)
    prob_positive = float(np.mean(returns > 0.0))
    prob_positive = min(1.0, max(0.0, prob_positive))

    # 95% VaR / CVaR on the terminal-return distribution (positive loss).
    q05 = float(np.quantile(returns, 0.05))
    var95_pct = _clamp_pct(-q05 * 100.0)
    var95_pct = max(0.0, var95_pct)

    tail = returns[returns <= q05]
    if tail.size == 0:
        cvar_raw = q05
    else:
        cvar_raw = float(np.mean(tail))
    cvar95_pct = _clamp_pct(-cvar_raw * 100.0)
    cvar95_pct = max(var95_pct, max(0.0, cvar95_pct))

    final_distribution = _histogram_bins(terminal)

    return {
        "horizon": horizon_label,
        "sims": int(n_sims),
        "steps": int(steps),
        "bands": bands,
        "finalDistribution": final_distribution,
        "expectedReturnPct": expected_return_pct,
        "var95Pct": var95_pct,
        "cvar95Pct": cvar95_pct,
        "probPositive": prob_positive,
    }


def _histogram_bins(values: np.ndarray, n_bins: int = 40) -> list[dict]:
    """Histogram terminal prices into ``{binStart, binEnd, count}`` dicts.

    Uses ``numpy.histogram`` with ``n_bins`` equal-width bins spanning the data
    range. A degenerate (all-equal or single-value) input produces a single
    symmetric bin centered on the value so the wire shape is always valid.

    Args:
        values: 1-D array of terminal prices (assumed finite, positive).
        n_bins: Desired number of histogram bins.

    Returns:
        A list of ``{"binStart": float, "binEnd": float, "count": int}`` dicts.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [{"binStart": 0.0, "binEnd": 1.0, "count": 0}]

    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        # Degenerate range: build one small symmetric bin around the value.
        center = lo if math.isfinite(lo) else 0.0
        pad = abs(center) * 0.05 + 1.0
        return [
            {
                "binStart": center - pad,
                "binEnd": center + pad,
                "count": int(arr.size),
            }
        ]

    bins = max(1, int(n_bins))
    counts, edges = np.histogram(arr, bins=bins, range=(lo, hi))
    out: list[dict] = []
    for i in range(len(counts)):
        out.append(
            {
                "binStart": float(edges[i]),
                "binEnd": float(edges[i + 1]),
                "count": int(counts[i]),
            }
        )
    return out


def _clamp_pct(pct: float) -> float:
    """Clamp a percentage to a finite, sane range.

    Args:
        pct: Candidate percentage value.

    Returns:
        ``pct`` clamped to ``[-_PCT_CLAMP, _PCT_CLAMP]`` and guaranteed finite
        (non-finite maps to ``0.0``).
    """
    if not math.isfinite(pct):
        return 0.0
    return max(-_PCT_CLAMP, min(_PCT_CLAMP, float(pct)))


# Annual conversion helper retained for callers that pass annual figures; kept
# private to avoid leaking into the public surface but documents the convention.
def _annual_to_daily_drift(annual_return: float) -> float:
    """Convert an annual simple return to a daily log-drift.

    Formula:
        mu_daily = ln(1 + annual_return) / TRADING_DAYS

    Args:
        annual_return: Annual simple return as a decimal.

    Returns:
        Daily continuously-compounded drift; ``0.0`` for non-finite or
        ``<= -100%`` inputs.
    """
    a = _safe_float(annual_return, 0.0)
    if a <= -1.0:
        return 0.0
    return math.log1p(a) / TRADING_DAYS
