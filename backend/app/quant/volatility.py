"""Volatility estimation: EWMA and a hand-rolled GARCH(1,1) MLE.

No external econometrics packages are used. The GARCH(1,1) parameters are fit by
maximizing the Gaussian log-likelihood with ``scipy.optimize.minimize`` under the
stationarity constraint ``alpha + beta < 1`` (and non-negativity). If the
optimizer fails or returns a degenerate fit, the module falls back to an EWMA
volatility estimate so a forecast is always produced.

All functions are numerically defensive: empty/short inputs, zero variance, and
non-finite values never raise — they return safe finite defaults, and volatility
outputs are floored at zero and clamped to a sane upper bound.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from app.quant.returns import TRADING_DAYS

__all__ = [
    "ewma_vol",
    "garch11_fit",
    "garch11_forecast",
]

# Smallest variance/volatility treated as non-zero.
_VAR_FLOOR: float = 1e-12
_VOL_FLOOR: float = 1e-8

# Upper clamp on annualized volatility (1000% — far beyond any real asset).
_VOL_CLAMP: float = 10.0

# Default GARCH parameters used as a sane starting point / fallback structure.
_DEFAULT_ALPHA: float = 0.08
_DEFAULT_BETA: float = 0.90

# Keep alpha + beta strictly below this to enforce stationarity / a finite
# unconditional variance.
_PERSISTENCE_CAP: float = 0.9999


def _clean_returns(returns: np.ndarray | list[float]) -> np.ndarray:
    """Coerce a returns input to a clean 1-D finite float array.

    Args:
        returns: Sequence of period returns.

    Returns:
        A 1-D ``float64`` array containing only finite entries (possibly empty).
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _annualize_vol(daily_vol: float) -> float:
    """Annualize a daily volatility and clamp it to a sane range.

    Formula:
        sigma_annual = sigma_daily * sqrt(TRADING_DAYS)

    Args:
        daily_vol: Daily volatility (standard deviation of daily returns).

    Returns:
        Annualized volatility, floored at ``0.0`` and clamped to ``_VOL_CLAMP``.
    """
    if not math.isfinite(daily_vol) or daily_vol < 0.0:
        return 0.0
    ann = daily_vol * math.sqrt(TRADING_DAYS)
    if not math.isfinite(ann):
        return 0.0
    return max(0.0, min(_VOL_CLAMP, ann))


def ewma_vol(returns: np.ndarray | list[float], lam: float = 0.94) -> float:
    """Annualized EWMA (RiskMetrics) volatility from the latest observation.

    The exponentially-weighted moving-average variance recursion is:

        sigma^2_t = lambda * sigma^2_{t-1} + (1 - lambda) * r_{t-1}^2

    seeded with the sample variance of the returns. The returned value is the
    latest recursive variance, square-rooted and annualized by sqrt(252).

    Args:
        returns: Sequence of daily returns.
        lam: Decay factor in ``(0, 1)``; RiskMetrics uses ``0.94`` for daily
            data. Values outside the open interval are clamped.

    Returns:
        Annualized EWMA volatility as a decimal, floored at ``0.0`` and clamped
        to ``_VOL_CLAMP``. Returns ``0.0`` for empty input.
    """
    arr = _clean_returns(returns)
    if arr.size == 0:
        return 0.0
    if arr.size == 1:
        return _annualize_vol(abs(float(arr[0])))

    lam = float(lam)
    if not math.isfinite(lam):
        lam = 0.94
    lam = min(0.999999, max(1e-6, lam))

    # Seed with sample variance (de-meaned) for a stable start.
    var = float(np.var(arr))
    if not math.isfinite(var) or var < _VAR_FLOOR:
        var = _VAR_FLOOR

    for r in arr:
        var = lam * var + (1.0 - lam) * (float(r) * float(r))
        if not math.isfinite(var) or var < 0.0:
            var = _VAR_FLOOR

    daily_vol = math.sqrt(max(var, 0.0))
    return _annualize_vol(daily_vol)


def _garch_neg_loglik(
    params: np.ndarray, eps: np.ndarray, var0: float
) -> float:
    """Negative Gaussian log-likelihood of a GARCH(1,1) model.

    Conditional variance recursion (zero-mean innovations ``eps``):

        h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}

    Gaussian log-likelihood (dropping constants):

        L = -0.5 * sum_t [ ln(h_t) + eps_t^2 / h_t ]

    Args:
        params: ``[omega, alpha, beta]``.
        eps: De-meaned return innovations (1-D finite array).
        var0: Variance used to seed ``h_0`` (sample variance).

    Returns:
        The negative log-likelihood (to be minimized). Returns a large finite
        penalty for invalid parameterizations so the optimizer steers away.
    """
    omega, alpha, beta = float(params[0]), float(params[1]), float(params[2])
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0 or (alpha + beta) >= 1.0:
        return 1e12

    h = max(var0, _VAR_FLOOR)
    n = eps.size
    nll = 0.0
    e2 = eps * eps
    for t in range(n):
        if h < _VAR_FLOOR:
            h = _VAR_FLOOR
        nll += math.log(h) + e2[t] / h
        # Update variance for the next step.
        h = omega + alpha * e2[t] + beta * h
        if not math.isfinite(h):
            return 1e12
    half = 0.5 * nll
    return half if math.isfinite(half) else 1e12


def garch11_fit(
    returns: np.ndarray | list[float],
) -> tuple[float, float, float]:
    """Fit GARCH(1,1) parameters by constrained maximum likelihood.

    Maximizes the Gaussian log-likelihood of the recursion

        h_t = omega + alpha * eps_{t-1}^2 + beta * h_{t-1}

    over ``(omega, alpha, beta)`` using ``scipy.optimize.minimize`` (SLSQP) with
    bounds ``omega >= 0``, ``alpha, beta in [0, 1)`` and the stationarity
    constraint ``alpha + beta <= 0.9999``. On too-short data, optimizer failure,
    or a degenerate result the function falls back to a sensible default
    parameterization derived from the sample variance and an EWMA-like split, so
    it always returns a valid, stationary triple.

    The likelihood is optimized on **variance-scaled** innovations (returns
    divided by their sample std) so the parameters are O(1) and SLSQP is
    well-conditioned regardless of the absolute return scale; ``omega`` is then
    rescaled back. Several starting points are tried and the lowest-NLL solution
    (including the default fallback) is returned, guarding against poor local
    optima.

    Args:
        returns: Sequence of daily returns (de-meaned internally).

    Returns:
        ``(omega, alpha, beta)`` with ``omega > 0``, ``alpha, beta >= 0`` and
        ``alpha + beta < 1``.
    """
    arr = _clean_returns(returns)
    sample_var = float(np.var(arr)) if arr.size >= 2 else _VAR_FLOOR
    if not math.isfinite(sample_var) or sample_var < _VAR_FLOOR:
        sample_var = _VAR_FLOOR

    def _fallback() -> tuple[float, float, float]:
        """Default stationary GARCH params targeting the sample variance."""
        alpha = _DEFAULT_ALPHA
        beta = _DEFAULT_BETA
        omega = sample_var * (1.0 - alpha - beta)
        omega = max(omega, _VAR_FLOOR)
        return (omega, alpha, beta)

    # Need a reasonable amount of data for a stable MLE.
    if arr.size < 30:
        return _fallback()

    # Work on scaled innovations so the variance is ~1 and parameters are O(1).
    # eps_scaled = eps / std  =>  var(eps_scaled) ~ 1. omega is rescaled by the
    # variance scale at the end (omega has units of variance).
    eps = arr - float(np.mean(arr))
    scale = math.sqrt(sample_var)  # std of the de-meaned series
    eps_scaled = eps / scale
    var0_s = 1.0  # scaled sample variance

    bounds = [(1e-12, None), (0.0, _PERSISTENCE_CAP), (0.0, _PERSISTENCE_CAP)]
    constraints = [
        {
            "type": "ineq",
            # alpha + beta <= _PERSISTENCE_CAP  -> cap - alpha - beta >= 0
            "fun": lambda p: _PERSISTENCE_CAP - p[1] - p[2],
        }
    ]

    # Candidate starting points in scaled space (target unconditional var = 1).
    starts = [
        (1.0 - _DEFAULT_ALPHA - _DEFAULT_BETA, _DEFAULT_ALPHA, _DEFAULT_BETA),
        (1.0, 0.0, 0.0),                 # white-noise / constant-variance
        (0.4, 0.1, 0.5),                 # moderate persistence
        (0.05, 0.05, 0.90),             # high persistence (RiskMetrics-like)
    ]

    # Seed the search with the (scaled) fallback so the result is never worse.
    fb_omega, fb_alpha, fb_beta = _fallback()
    best_params = (fb_omega / sample_var, fb_alpha, fb_beta)  # in scaled space
    best_nll = _garch_neg_loglik(np.array(best_params), eps_scaled, var0_s)

    for s in starts:
        x0 = np.array([max(s[0], 1e-9), s[1], s[2]], dtype=np.float64)
        try:
            res = minimize(
                _garch_neg_loglik,
                x0,
                args=(eps_scaled, var0_s),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-12},
            )
        except Exception:
            continue
        if not np.all(np.isfinite(res.x)):
            continue
        cand = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
        if (
            cand[0] <= 0.0
            or cand[1] < 0.0
            or cand[2] < 0.0
            or (cand[1] + cand[2]) >= 1.0
        ):
            continue
        nll = _garch_neg_loglik(np.array(cand), eps_scaled, var0_s)
        if math.isfinite(nll) and nll < best_nll:
            best_nll = nll
            best_params = cand

    omega_s, alpha, beta = best_params
    # Rescale omega from scaled (variance ~1) space back to raw return variance.
    omega = omega_s * sample_var

    # Final validation; fall back if degenerate or non-stationary.
    if (
        not math.isfinite(omega)
        or not math.isfinite(alpha)
        or not math.isfinite(beta)
        or omega <= 0.0
        or alpha < 0.0
        or beta < 0.0
        or (alpha + beta) >= 1.0
    ):
        return _fallback()

    omega = max(omega, _VAR_FLOOR)
    return (omega, alpha, beta)


def garch11_forecast(
    returns: np.ndarray | list[float], horizon_days: int
) -> float:
    """Forecast annualized volatility ``horizon_days`` ahead with GARCH(1,1).

    Fits GARCH(1,1), runs the conditional-variance recursion through the sample
    to obtain the latest variance ``h_T``, then projects forward. The k-step
    forecast of conditional variance mean-reverts to the unconditional variance
    ``V = omega / (1 - alpha - beta)``:

        E[h_{T+k}] = V + (alpha + beta)^{k-1} * (h_{T+1} - V),
        h_{T+1}    = omega + alpha * eps_T^2 + beta * h_T

    The reported figure is the average per-day forecast variance over the
    horizon, square-rooted and annualized by sqrt(252) — i.e. the expected
    annualized volatility over the next ``horizon_days``. Falls back to
    :func:`ewma_vol` if fitting/forecasting is not viable.

    Args:
        returns: Sequence of daily returns.
        horizon_days: Forecast horizon in trading days (coerced to >= 1).

    Returns:
        Annualized volatility forecast as a decimal, floored at ``0.0`` and
        clamped to ``_VOL_CLAMP``.
    """
    arr = _clean_returns(returns)
    h_days = max(1, int(horizon_days))

    if arr.size < 30:
        return ewma_vol(arr)

    omega, alpha, beta = garch11_fit(arr)
    persistence = alpha + beta

    # Unconditional (long-run) variance.
    denom = 1.0 - persistence
    if denom <= _VAR_FLOOR:
        uncond_var = float(np.var(arr))
    else:
        uncond_var = omega / denom
    if not math.isfinite(uncond_var) or uncond_var < _VAR_FLOOR:
        uncond_var = max(float(np.var(arr)), _VAR_FLOOR)

    # Roll the recursion through the sample to get the latest variance h_T.
    eps = arr - float(np.mean(arr))
    e2 = eps * eps
    h = max(float(np.var(arr)), _VAR_FLOOR)
    for t in range(arr.size):
        h = omega + alpha * e2[t] + beta * h
        if not math.isfinite(h) or h < _VAR_FLOOR:
            h = _VAR_FLOOR

    # h is now the one-step-ahead variance h_{T+1}.
    if not math.isfinite(h) or h < _VAR_FLOOR:
        return ewma_vol(arr)

    # Average expected variance over the horizon (mean-reverting forecast).
    if persistence >= _PERSISTENCE_CAP or persistence <= 0.0:
        # No mean reversion: forecast variance stays at h.
        avg_var = h
    else:
        total = 0.0
        for k in range(1, h_days + 1):
            fc = uncond_var + (persistence ** (k - 1)) * (h - uncond_var)
            if not math.isfinite(fc) or fc < 0.0:
                fc = uncond_var
            total += fc
        avg_var = total / h_days

    if not math.isfinite(avg_var) or avg_var < 0.0:
        return ewma_vol(arr)

    daily_vol = math.sqrt(max(avg_var, 0.0))
    result = _annualize_vol(daily_vol)
    if result <= _VOL_FLOOR:
        # Degenerate result -> fall back to EWMA.
        return ewma_vol(arr)
    return result
