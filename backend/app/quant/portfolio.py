"""Markowitz mean-variance portfolio optimization and the efficient frontier.

This module implements classical Modern Portfolio Theory (Markowitz, 1952) on a
set of assets whose expected returns and covariance are supplied **already
annualized**. It provides:

    * ``portfolio_stats`` — annual return, volatility and Sharpe of a weight vector.
    * ``optimize``        — long-only constrained optimization for three objectives
                            (``max_sharpe``, ``min_volatility``, ``target_return``)
                            via :func:`scipy.optimize.minimize` (SLSQP).
    * ``efficient_frontier`` — the minimum-variance frontier sampled at ``n`` points.
    * ``tangency_portfolio``  — the maximum-Sharpe (tangency) portfolio.
    * ``capital_market_line`` — the CML from the risk-free rate through the tangency.

Core formulas (``w`` = weight vector, ``mu`` = expected returns, ``S`` =
covariance matrix, ``rf`` = annual risk-free rate):

    portfolio return     R_p   = w . mu
    portfolio variance   var_p = w . S . w
    portfolio volatility sig_p = sqrt(var_p)
    Sharpe ratio         SR    = (R_p - rf) / sig_p

Constraints used everywhere: long-only (``0 <= w_i <= 1``) and fully invested
(``sum(w) = 1``).

Numerical defensiveness:
    * Empty / length-1 universes, NaN/inf inputs, mismatched shapes and singular
      covariance never raise. The covariance matrix is symmetrized and a tiny
      ridge (Tikhonov regularization, ``S + eps * I``) is added so it is always
      positive-definite for the quadratic form and any inverse.
    * Optimizer failures fall back to the equal-weight portfolio, which is always
      feasible (long-only, sums to 1).
    * All emitted numbers are finite; weights are clipped to ``[0, 1]`` and
      renormalized to sum to 1.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "portfolio_stats",
    "optimize",
    "efficient_frontier",
    "tangency_portfolio",
    "capital_market_line",
]

# Ridge added to the covariance diagonal to guarantee a positive-definite,
# invertible matrix even when assets are perfectly collinear or variance is zero.
_RIDGE: float = 1e-8

# Smallest volatility / denominator treated as non-zero before a ratio is taken.
_EPS: float = 1e-12

# SLSQP tolerance and iteration budget — tight enough to converge, cheap enough
# to call many times when sweeping the efficient frontier.
_SLSQP_TOL: float = 1e-9
_SLSQP_MAXITER: int = 400


def _clean_mu(mu_annual: np.ndarray | list[float]) -> np.ndarray:
    """Coerce expected returns to a clean finite 1-D float array.

    Non-finite entries (NaN / +-inf) are replaced with ``0.0`` so a single bad
    estimate cannot poison the optimization.

    Args:
        mu_annual: Sequence of annual expected returns, one per asset.

    Returns:
        A 1-D ``float64`` array of finite expected returns (possibly empty).
    """
    arr = np.asarray(mu_annual, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _clean_cov(cov_annual: np.ndarray | list[list[float]], n: int) -> np.ndarray:
    """Coerce a covariance matrix to a clean, symmetric, positive-definite array.

    The input is reshaped/validated to ``n x n``; non-finite entries become
    ``0.0``; the matrix is symmetrized (``(S + S.T) / 2``) and a tiny ridge is
    added to the diagonal (Tikhonov regularization) so the quadratic form and any
    inverse are always well defined:

        S_clean = (S + S.T) / 2 + RIDGE * I

    Args:
        cov_annual: Annual covariance matrix (``n x n`` array-like).
        n: Expected number of assets (rows/cols).

    Returns:
        A symmetric positive-definite ``n x n`` ``float64`` covariance matrix.
        If the input cannot be interpreted as ``n x n`` it is replaced with a
        ridge-only diagonal (independent unit-ish variances), keeping the
        optimization well posed.
    """
    if n <= 0:
        return np.empty((0, 0), dtype=np.float64)
    arr = np.asarray(cov_annual, dtype=np.float64)
    if arr.shape != (n, n):
        # Fall back to a diagonal matrix so optimization stays well posed.
        arr = np.eye(n, dtype=np.float64) * _RIDGE
    else:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = 0.5 * (arr + arr.T)
    arr = arr + np.eye(n, dtype=np.float64) * _RIDGE
    return arr


def _normalize_weights(weights: np.ndarray, n: int) -> np.ndarray:
    """Clip weights to ``[0, 1]`` and renormalize so they sum to 1.

    Guarantees a valid long-only, fully-invested portfolio is returned even if an
    optimizer drifts slightly outside the feasible set. If the clipped weights
    sum to (near) zero the equal-weight portfolio is returned.

    Args:
        weights: Candidate weight vector.
        n: Number of assets (used for the equal-weight fallback).

    Returns:
        A non-negative ``float64`` weight vector of length ``n`` summing to 1.
    """
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size != n or not np.all(np.isfinite(w)):
        return np.full(n, 1.0 / n, dtype=np.float64)
    w = np.clip(w, 0.0, 1.0)
    total = float(w.sum())
    if total <= _EPS:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return w / total


def portfolio_stats(
    weights: np.ndarray | list[float],
    mu_annual: np.ndarray | list[float],
    cov_annual: np.ndarray | list[list[float]],
    rf: float,
) -> tuple[float, float, float]:
    """Compute annual return, volatility and Sharpe ratio of a portfolio.

    Formulas (inputs already annualized):
        R_p   = w . mu
        var_p = w . S . w
        sig_p = sqrt(max(var_p, 0))
        SR    = (R_p - rf) / sig_p        (0 if sig_p ~ 0)

    Args:
        weights: Portfolio weights (need not be pre-normalized; they are clipped
            non-negative and renormalized to sum to 1 before the calculation).
        mu_annual: Annual expected returns per asset.
        cov_annual: Annual covariance matrix (``n x n``).
        rf: Annual risk-free rate (decimal, e.g. ``0.04``).

    Returns:
        A ``(expected_return, volatility, sharpe)`` tuple of finite floats.
        Returns ``(0.0, 0.0, 0.0)`` for an empty universe.
    """
    mu = _clean_mu(mu_annual)
    n = mu.size
    if n == 0:
        return 0.0, 0.0, 0.0
    cov = _clean_cov(cov_annual, n)
    w = _normalize_weights(np.asarray(weights, dtype=np.float64).ravel(), n)
    rf_v = float(rf) if math.isfinite(rf) else 0.0

    ret = float(w @ mu)
    var = float(w @ cov @ w)
    var = max(var, 0.0)
    vol = math.sqrt(var)

    if not math.isfinite(ret):
        ret = 0.0
    if not math.isfinite(vol):
        vol = 0.0

    if vol < _EPS:
        sharpe = 0.0
    else:
        sharpe = (ret - rf_v) / vol
        if not math.isfinite(sharpe):
            sharpe = 0.0
    return ret, vol, sharpe


def _solve_slsqp(
    objective,
    n: int,
    extra_constraints: list[dict] | None = None,
) -> np.ndarray | None:
    """Run SLSQP from the equal-weight start with long-only, sum-to-1 constraints.

    Args:
        objective: Callable ``f(w) -> float`` to minimize.
        n: Number of assets.
        extra_constraints: Optional list of additional SLSQP constraint dicts
            (e.g. a target-return equality) appended to the budget constraint.

    Returns:
        The optimized weight vector (clipped/renormalized) if the solver reports
        success and produces a finite, feasible result; otherwise ``None`` so the
        caller can fall back.
    """
    if n <= 0:
        return None
    x0 = np.full(n, 1.0 / n, dtype=np.float64)
    bounds = [(0.0, 1.0)] * n
    constraints: list[dict] = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
    if extra_constraints:
        constraints.extend(extra_constraints)
    try:
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": _SLSQP_MAXITER, "ftol": _SLSQP_TOL},
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None
    if not getattr(res, "success", False):
        return None
    w = np.asarray(res.x, dtype=np.float64).ravel()
    if w.size != n or not np.all(np.isfinite(w)):
        return None
    return _normalize_weights(w, n)


def optimize(
    mu_annual: np.ndarray | list[float],
    cov_annual: np.ndarray | list[list[float]],
    rf: float,
    objective: str,
    target: float | None = None,
) -> np.ndarray:
    """Solve a long-only mean-variance optimization for the chosen objective.

    All problems are solved with SLSQP under long-only (``0 <= w_i <= 1``) and
    fully-invested (``sum(w) = 1``) constraints. Supported objectives:

        * ``"max_sharpe"``     — maximize ``(w.mu - rf) / sqrt(w.S.w)``
          (implemented as minimizing the negative Sharpe ratio).
        * ``"min_volatility"`` — minimize portfolio variance ``w.S.w``.
        * ``"target_return"``  — minimize variance subject to ``w.mu = target``
          (the target is clamped to the achievable ``[min(mu), max(mu)]`` range so
          the equality constraint is always feasible).

    Args:
        mu_annual: Annual expected returns per asset.
        cov_annual: Annual covariance matrix (``n x n``).
        rf: Annual risk-free rate (decimal).
        objective: One of ``"max_sharpe"``, ``"min_volatility"``,
            ``"target_return"``. Unknown values default to ``"max_sharpe"``.
        target: Required annual target return for ``"target_return"``; ignored
            otherwise. ``None`` with ``"target_return"`` falls back to the
            mean of ``mu``.

    Returns:
        A long-only ``float64`` weight vector of length ``n`` summing to 1. For an
        empty universe an empty array is returned; on optimizer failure the
        equal-weight portfolio is returned as a safe, feasible fallback.
    """
    mu = _clean_mu(mu_annual)
    n = mu.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.ones(1, dtype=np.float64)

    cov = _clean_cov(cov_annual, n)
    rf_v = float(rf) if math.isfinite(rf) else 0.0
    equal = np.full(n, 1.0 / n, dtype=np.float64)

    def _variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    obj = objective if objective in {"max_sharpe", "min_volatility", "target_return"} else "max_sharpe"

    if obj == "min_volatility":
        result = _solve_slsqp(_variance, n)
        return result if result is not None else equal

    if obj == "target_return":
        lo = float(np.min(mu))
        hi = float(np.max(mu))
        if target is None or not math.isfinite(target):
            tgt = float(np.mean(mu))
        else:
            tgt = float(target)
        # Clamp the target into the achievable range so the equality is feasible.
        tgt = min(hi, max(lo, tgt))
        ret_constraint = {"type": "eq", "fun": lambda w, t=tgt: float(w @ mu - t)}
        result = _solve_slsqp(_variance, n, [ret_constraint])
        return result if result is not None else equal

    # Default: max_sharpe — minimize negative Sharpe ratio.
    def _neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mu)
        var = float(w @ cov @ w)
        vol = math.sqrt(var) if var > 0.0 else 0.0
        if vol < _EPS:
            return 0.0
        sr = (ret - rf_v) / vol
        return -sr if math.isfinite(sr) else 0.0

    result = _solve_slsqp(_neg_sharpe, n)
    if result is None:
        return equal
    # Guard against a degenerate corner solution: if max_sharpe collapsed to all
    # weight on one asset but another single asset has a strictly better Sharpe,
    # prefer the best single-asset Sharpe is already covered by the optimizer; we
    # simply return the (normalized) optimizer result here.
    return result


def tangency_portfolio(
    mu_annual: np.ndarray | list[float],
    cov_annual: np.ndarray | list[list[float]],
    rf: float,
) -> np.ndarray:
    """Compute the long-only tangency (maximum-Sharpe) portfolio.

    The tangency portfolio is the point on the efficient frontier touched by the
    capital market line from ``rf``; it maximizes

        SR = (w.mu - rf) / sqrt(w.S.w)

    This is a thin wrapper around :func:`optimize` with ``objective="max_sharpe"``
    (long-only / fully invested), so the unconstrained closed form
    ``w* ~ S^{-1} (mu - rf)`` is replaced by its long-only projection.

    Args:
        mu_annual: Annual expected returns per asset.
        cov_annual: Annual covariance matrix (``n x n``).
        rf: Annual risk-free rate (decimal).

    Returns:
        A long-only ``float64`` weight vector of length ``n`` summing to 1
        (empty for an empty universe).
    """
    return optimize(mu_annual, cov_annual, rf, "max_sharpe")


def efficient_frontier(
    mu_annual: np.ndarray | list[float],
    cov_annual: np.ndarray | list[list[float]],
    rf: float,
    n: int = 40,
) -> list[dict]:
    """Sample the long-only minimum-variance (efficient) frontier.

    The frontier is traced by sweeping the target return from the
    minimum-variance portfolio's return up to the maximum single-asset return,
    solving a ``target_return`` minimum-variance problem at each level, and
    recording ``(volatility, expectedReturn, sharpe)`` for the resulting
    portfolio:

        for target in linspace(R_minvar, max(mu), n):
            w   = optimize(min variance s.t. w.mu = target)
            pt  = (vol(w), ret(w), sharpe(w))

    Starting at the global minimum-variance return keeps every sampled point on
    the *efficient* (upper) half of the frontier.

    Args:
        mu_annual: Annual expected returns per asset.
        cov_annual: Annual covariance matrix (``n x n``).
        rf: Annual risk-free rate (decimal), used only for the per-point Sharpe.
        n: Number of points to sample (clamped to ``>= 2`` when feasible).

    Returns:
        A list of dicts, each shaped like the ``PortfolioPoint`` DTO::

            {"volatility": float, "expectedReturn": float, "sharpe": float}

        sorted by ascending volatility. Returns an empty list for an empty
        universe; for a single asset returns that asset's single point.
    """
    mu = _clean_mu(mu_annual)
    n_assets = mu.size
    if n_assets == 0:
        return []
    cov = _clean_cov(cov_annual, n_assets)

    if n_assets == 1:
        ret, vol, sh = portfolio_stats(np.ones(1), mu, cov, rf)
        return [{"volatility": vol, "expectedReturn": ret, "sharpe": sh}]

    n_points = max(2, int(n)) if n and n > 0 else 40

    # Lower bound of the efficient segment = return of the global min-var portfolio.
    w_minvar = optimize(mu, cov, rf, "min_volatility")
    ret_minvar, _, _ = portfolio_stats(w_minvar, mu, cov, rf)
    hi = float(np.max(mu))
    lo = float(ret_minvar)
    if not math.isfinite(lo):
        lo = float(np.min(mu))
    if hi <= lo:
        # Degenerate spread (all returns ~equal): emit the min-var point only.
        ret, vol, sh = portfolio_stats(w_minvar, mu, cov, rf)
        return [{"volatility": vol, "expectedReturn": ret, "sharpe": sh}]

    targets = np.linspace(lo, hi, n_points)
    points: list[dict] = []
    for tgt in targets:
        w = optimize(mu, cov, rf, "target_return", target=float(tgt))
        ret, vol, sh = portfolio_stats(w, mu, cov, rf)
        if math.isfinite(ret) and math.isfinite(vol) and math.isfinite(sh):
            points.append({"volatility": vol, "expectedReturn": ret, "sharpe": sh})

    if not points:
        ret, vol, sh = portfolio_stats(w_minvar, mu, cov, rf)
        return [{"volatility": vol, "expectedReturn": ret, "sharpe": sh}]

    points.sort(key=lambda p: p["volatility"])
    return points


def capital_market_line(
    rf: float,
    tangency_return: float,
    tangency_vol: float,
    n: int = 40,
) -> list[dict]:
    """Build the capital market line from the risk-free asset to (beyond) tangency.

    The CML is the set of risk/return points attainable by mixing the risk-free
    asset with the tangency portfolio:

        for a fraction ``f`` of wealth in the tangency portfolio,
            vol(f)    = f * sigma_tangency
            return(f) = rf + f * (R_tangency - rf)
            sharpe    = (return(f) - rf) / vol(f) = (R_tan - rf)/sigma_tan  (constant)

    Sampling ``f`` from 0 (all cash, the ``(0, rf)`` intercept) to 1.5 (a modestly
    levered position past the tangency point) yields a straight line whose slope
    equals the tangency Sharpe ratio.

    Args:
        rf: Annual risk-free rate (decimal).
        tangency_return: Annual expected return of the tangency portfolio.
        tangency_vol: Annual volatility of the tangency portfolio.
        n: Number of points to sample along the line (clamped to ``>= 2``).

    Returns:
        A list of dicts shaped like the ``PortfolioPoint`` DTO
        (``{"volatility", "expectedReturn", "sharpe"}``), ordered by ascending
        volatility. If the tangency volatility is effectively zero only the
        risk-free intercept ``(0, rf)`` is returned (a vertical/degenerate line
        cannot be drawn meaningfully).
    """
    rf_v = float(rf) if math.isfinite(rf) else 0.0
    ret_t = float(tangency_return) if math.isfinite(tangency_return) else rf_v
    vol_t = float(tangency_vol) if (math.isfinite(tangency_vol) and tangency_vol > 0.0) else 0.0
    n_points = max(2, int(n)) if n and n > 0 else 40

    if vol_t < _EPS:
        # No risky volatility to leverage: the CML degenerates to the rf point.
        return [{"volatility": 0.0, "expectedReturn": rf_v, "sharpe": 0.0}]

    slope = (ret_t - rf_v) / vol_t  # constant Sharpe along the line
    if not math.isfinite(slope):
        slope = 0.0

    fractions = np.linspace(0.0, 1.5, n_points)
    points: list[dict] = []
    for f in fractions:
        vol = float(f) * vol_t
        ret = rf_v + float(f) * (ret_t - rf_v)
        if not (math.isfinite(vol) and math.isfinite(ret)):
            continue
        sh = 0.0 if vol < _EPS else slope
        if not math.isfinite(sh):
            sh = 0.0
        points.append({"volatility": vol, "expectedReturn": ret, "sharpe": sh})

    if not points:
        return [{"volatility": 0.0, "expectedReturn": rf_v, "sharpe": 0.0}]
    points.sort(key=lambda p: p["volatility"])
    return points
