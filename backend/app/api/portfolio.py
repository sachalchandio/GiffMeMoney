"""``/api/portfolio/optimize`` — Markowitz mean-variance optimization.

The single endpoint accepts a :class:`~app.schemas.PortfolioRequest` (a set of
symbols, an annual risk-free rate, an objective and an optional target return)
and returns a :class:`~app.schemas.PortfolioResult`: the optimal long-only
weights, the portfolio's annual return / volatility / Sharpe, the efficient
frontier and the capital market line.

Inputs to the optimizer are built from the deterministic simulator's price
history via the shared :class:`~app.market.provider.MarketDataProvider`:

    * **Expected returns** ``mu`` (annual) — each asset's mean daily log return
      compounded to a year: ``mu_i = exp(mean(log r_i) * 252) - 1``.
    * **Covariance** ``S`` (annual) — the sample covariance of the trailing,
      length-aligned daily simple returns scaled by ``252`` (linear-in-time):
      ``S = Cov(R_daily) * 252``.

Everything downstream (the SLSQP optimization, frontier sweep and CML) lives in
:mod:`app.quant.portfolio` and is numerically defensive, so a degenerate request
(empty/duplicate symbols, flat history) still yields a valid, finite result.
Unknown symbols return ``404``; an empty symbol list returns ``422``.
"""

from __future__ import annotations

import math

import numpy as np
from fastapi import APIRouter, HTTPException

from app.market.provider import MarketDataProvider, get_provider
from app.quant import portfolio as pf
from app.quant.returns import TRADING_DAYS, log_returns, simple_returns
from app.schemas import (
    PortfolioPoint,
    PortfolioRequest,
    PortfolioResult,
    PortfolioWeight,
)

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["portfolio"])

# How many trailing daily closes to pull per symbol when estimating mu / cov.
_HISTORY_DAYS: int = 1260

# Number of points sampled for the efficient frontier and the capital market line.
_FRONTIER_POINTS: int = 40
_CML_POINTS: int = 40


def _safe(value: float, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, falling back to ``default``.

    Args:
        value: Candidate number.
        default: Substitute for NaN / +-inf / non-numeric input.

    Returns:
        A finite float.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _resolve_symbols(symbols: list[str], provider: MarketDataProvider) -> list[str]:
    """Validate and de-duplicate requested symbols, preserving first-seen order.

    Each symbol is validated against the provider (``get_asset`` raises
    ``KeyError`` for unknown tickers); duplicates are collapsed so an asset is
    never double-counted in the optimization.

    Args:
        symbols: Raw symbol list from the request body.
        provider: The market-data provider used to validate tickers.

    Returns:
        A list of unique, upper-cased, known symbols in request order.

    Raises:
        HTTPException: ``404`` for the first unknown symbol; ``422`` if the
            resolved list is empty.
    """
    seen: set[str] = set()
    resolved: list[str] = []
    for raw in symbols:
        sym = str(raw).strip().upper()
        if not sym or sym in seen:
            continue
        try:
            provider.get_asset(sym)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Unknown symbol: {raw!r}"
            ) from exc
        seen.add(sym)
        resolved.append(sym)
    if not resolved:
        raise HTTPException(
            status_code=422,
            detail="At least one valid symbol is required.",
        )
    return resolved


def _estimate_mu_cov(
    symbols: list[str], provider: MarketDataProvider
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate annualized expected returns and covariance for the symbols.

    Daily closes are pulled per symbol and trailing-aligned to a common length so
    the covariance lines up the same dates. Then:

        mu_i = exp(mean(log r_i) * 252) - 1          (annual expected return)
        S    = Cov(R_daily, ddof=0) * 252            (annual covariance)

    where ``R_daily`` is the matrix of length-aligned daily *simple* returns
    (assets in columns).

    Args:
        symbols: Unique, known symbols (length ``n``).
        provider: The market-data provider supplying price history.

    Returns:
        A ``(mu, cov)`` tuple: ``mu`` is a length-``n`` annual expected-return
        vector and ``cov`` an ``n x n`` annual covariance matrix. Both are finite
        (NaN/inf scrubbed); ``cov`` defaults to a zero matrix when history is too
        short (the optimizer ridge keeps it well posed).
    """
    n = len(symbols)
    mu = np.zeros(n, dtype=np.float64)

    # Per-symbol daily simple returns for covariance, plus the annual mu estimate.
    simple_series: list[np.ndarray] = []
    for i, sym in enumerate(symbols):
        closes = np.asarray(
            provider.history(sym, days=_HISTORY_DAYS), dtype=np.float64
        ).ravel()
        lr = log_returns(closes)
        if lr.size:
            daily_mean = float(np.mean(lr))
            exponent = max(-700.0, min(700.0, daily_mean * TRADING_DAYS))
            ann = math.exp(exponent) - 1.0
            mu[i] = ann if math.isfinite(ann) else 0.0
        simple_series.append(simple_returns(closes))

    # Trailing-align the daily simple-return series to a common length.
    lengths = [s.size for s in simple_series if s.size > 0]
    m = min(lengths) if lengths else 0
    if m >= 2 and n >= 1:
        matrix = np.column_stack([s[-m:] for s in simple_series])
        # rowvar=False -> variables (assets) are columns; ddof=0 population cov.
        daily_cov = np.cov(matrix, rowvar=False, ddof=0)
        daily_cov = np.atleast_2d(np.asarray(daily_cov, dtype=np.float64))
        if daily_cov.shape != (n, n):
            daily_cov = np.zeros((n, n), dtype=np.float64)
        cov = daily_cov * float(TRADING_DAYS)
    else:
        cov = np.zeros((n, n), dtype=np.float64)

    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    return mu, cov


@router.post(
    "/portfolio/optimize",
    response_model=PortfolioResult,
    summary="Markowitz mean-variance portfolio optimization",
)
def optimize_portfolio(request: PortfolioRequest) -> PortfolioResult:
    """Optimize a long-only portfolio over the requested symbols.

    Builds annualized expected-return and covariance estimates from simulated
    price history, then solves the chosen objective (``max_sharpe`` /
    ``min_volatility`` / ``target_return``) under long-only, fully-invested
    constraints. The response also carries the efficient frontier and the
    capital market line (drawn from the risk-free rate through the tangency
    portfolio) for charting.

    Formulas (``w`` weights, ``mu`` annual returns, ``S`` annual covariance):
        return  R_p = w . mu ;  vol  sigma_p = sqrt(w . S . w) ;
        Sharpe  = (R_p - rf) / sigma_p.

    Args:
        request: A :class:`~app.schemas.PortfolioRequest` with the symbol set,
            annual ``riskFreeRate``, ``objective`` and optional ``targetReturn``
            (annual decimal, used only for the ``target_return`` objective).

    Returns:
        A populated :class:`~app.schemas.PortfolioResult` with weights, the
        portfolio's annual return / volatility / Sharpe, the efficient frontier,
        the capital market line and the echoed risk-free rate.

    Raises:
        HTTPException: ``404`` for an unknown symbol; ``422`` for an empty symbol
            list (or otherwise invalid body, via FastAPI validation).
    """
    provider = get_provider()
    symbols = _resolve_symbols(request.symbols, provider)
    rf = _safe(request.risk_free_rate)

    mu, cov = _estimate_mu_cov(symbols, provider)

    # Solve the chosen objective (long-only, fully invested).
    weights = pf.optimize(
        mu_annual=mu,
        cov_annual=cov,
        rf=rf,
        objective=request.objective,
        target=request.target_return,
    )
    ret, vol, sharpe = pf.portfolio_stats(weights, mu, cov, rf)

    # Efficient frontier (minimum-variance frontier, efficient half).
    frontier = pf.efficient_frontier(mu, cov, rf, n=_FRONTIER_POINTS)

    # Capital market line through the tangency (max-Sharpe) portfolio.
    tan_w = pf.tangency_portfolio(mu, cov, rf)
    tan_ret, tan_vol, _ = pf.portfolio_stats(tan_w, mu, cov, rf)
    cml = pf.capital_market_line(rf, tan_ret, tan_vol, n=_CML_POINTS)

    weight_dtos = [
        PortfolioWeight(symbol=sym, weight=_safe(float(w)))
        for sym, w in zip(symbols, np.asarray(weights, dtype=np.float64).ravel())
    ]
    frontier_dtos = [
        PortfolioPoint(
            volatility=_safe(p["volatility"]),
            expected_return=_safe(p["expectedReturn"]),
            sharpe=_safe(p["sharpe"]),
        )
        for p in frontier
    ]
    cml_dtos = [
        PortfolioPoint(
            volatility=_safe(p["volatility"]),
            expected_return=_safe(p["expectedReturn"]),
            sharpe=_safe(p["sharpe"]),
        )
        for p in cml
    ]

    return PortfolioResult(
        weights=weight_dtos,
        expected_return=_safe(ret),
        volatility=_safe(vol),
        sharpe=_safe(sharpe),
        efficient_frontier=frontier_dtos,
        capital_market_line=cml_dtos,
        risk_free_rate=rf,
    )
