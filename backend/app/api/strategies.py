"""``/api/strategies`` — the quant model catalog, rankings, and backtesting.

Endpoints:

    * ``GET /api/strategies`` — the static catalog of every registered model
      (:data:`~app.strategies.registry.STRATEGY_META`, now ~73 entries).
    * ``GET /api/strategies/{id}/rankings`` — every asset ranked by a single
      strategy's signal score (404 if the strategy id is unknown).
    * ``GET /api/strategies/{id}/backtest?symbol=<sym>`` — the realized
      backtest (:class:`~app.schemas.BacktestResultDTO`) of one strategy on one
      asset (V2; 404 for an unknown strategy or symbol).
    * ``GET /api/strategies/leaderboard?symbol=<sym>&limit=20`` — every strategy
      ranked by realized backtest Sharpe / CAGR for one asset (V2; 404 for an
      unknown symbol).

The catalog is served straight from the registry; the per-strategy ranking is
produced by the shared :class:`~app.strategies.engine.AnalysisEngine`. The
backtests run each strategy's vectorized position series from
:data:`~app.strategies.registry.POSITION_FUNCS` through
:func:`~app.quant.backtest.backtest_positions`; strategies without a per-bar
position function are reported with ``supported=False`` (a buy & hold-only
result) rather than erroring.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query

from app.api.recommendations import get_engine
from app.schemas import (
    BacktestResultDTO,
    StrategyLeaderboard,
    StrategyMeta,
    StrategyRanking,
)
from app.api.assets import leaderboard_for_symbol, run_backtest
from app.strategies.registry import META_BY_ID, STRATEGY_META

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["strategies"])


@router.get(
    "/strategies",
    response_model=list[StrategyMeta],
    summary="List the full quant model catalog",
    description=(
        "Return the static metadata for every registered strategy (~73 ids) in "
        "registry order: each `StrategyMeta` carries the id, name, category, "
        "summary, formula, inputs and academic references."
    ),
)
def list_strategies() -> list[StrategyMeta]:
    """Return the static metadata for every registered strategy.

    The catalog mirrors the V2 registry (the base 20 plus the 53 expansion
    strategies, now ~73 ids) in registry order: id, name, category, summary,
    formula, inputs and references.

    Returns:
        A list of :class:`~app.schemas.StrategyMeta` (now >= 70 entries).
    """
    return list(STRATEGY_META)


@router.get(
    "/strategies/leaderboard",
    response_model=StrategyLeaderboard,
    summary="Rank strategies by realized backtest performance for one asset",
    description=(
        "Backtest every registered strategy's vectorized position series on one "
        "asset and return a `StrategyLeaderboard` ranked best-first by Sharpe "
        "ratio (CAGR breaks ties). Strategies without a per-bar position "
        "function (snapshot / fundamental models) are flagged "
        "`supported=false` and sorted to the bottom rather than dropped. "
        "Includes the asset's buy & hold benchmark. Returns `404` for an "
        "unknown symbol."
    ),
)
def get_strategy_leaderboard(
    symbol: str = Query(
        ...,
        description="Asset ticker to backtest every strategy on (case-insensitive).",
        examples=["AAPL"],
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of leaderboard entries to return (1..200).",
        examples=[20],
    ),
) -> StrategyLeaderboard:
    """Rank every strategy by its realized backtest performance for one asset.

    Backtests each strategy's vectorized position series on the asset, then
    ranks the results best-first by Sharpe ratio (CAGR breaks ties). Strategies
    without a per-bar position function (snapshot / fundamental models) are
    reported with ``supported=False`` and sorted to the bottom rather than
    dropped.

    Args:
        symbol: Asset ticker (case-insensitive).
        limit: Maximum number of entries to return (1..200; default 20).

    Returns:
        A :class:`~app.schemas.StrategyLeaderboard` with the asset's buy & hold
        benchmark and the ranked strategy entries.

    Raises:
        HTTPException: ``404`` if the symbol is unknown.
    """
    engine = get_engine()
    try:
        return leaderboard_for_symbol(engine, symbol, limit=limit)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown symbol: {symbol}",
        ) from exc


@router.get(
    "/strategies/{strategy_id}/rankings",
    response_model=StrategyRanking,
    summary="Rank every asset by one strategy's signal score",
    description=(
        "Score the whole universe with a single strategy and return a "
        "`StrategyRanking` sorted by that strategy's signal score, most "
        "bullish first. Returns `404` if `strategy_id` is not a registered "
        "strategy."
    ),
)
def get_strategy_rankings(
    strategy_id: str = Path(
        description="Strategy id from the catalog.",
        examples=["sharpe"],
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of ranked entries to return (1..200).",
        examples=[20],
    ),
) -> StrategyRanking:
    """Rank the whole universe by a single strategy's signal score (descending).

    Args:
        strategy_id: Strategy id from the catalog (e.g. ``'capm'``, ``'rsi'``).
        limit: Maximum number of entries to return (1..200; default 20).

    Returns:
        A :class:`~app.schemas.StrategyRanking` sorted by that strategy's score,
        most bullish first.

    Raises:
        HTTPException: ``404`` if ``strategy_id`` is not a registered strategy.
    """
    engine = get_engine()
    try:
        return engine.strategy_ranking(strategy_id, limit=limit)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown strategy: {strategy_id!r}",
        ) from exc


@router.get(
    "/strategies/{strategy_id}/backtest",
    response_model=BacktestResultDTO,
    summary="Backtest one strategy on one asset",
    description=(
        "Run one strategy's vectorized per-bar position series on one asset's "
        "close history and return a `BacktestResultDTO`: the 14 realized "
        "metrics for both the strategy and a buy & hold benchmark, plus a "
        "downsampled equity curve. Strategies that are not time-backtestable "
        "per-bar return a buy & hold-only result flagged `supported=false` "
        "(not an error). Returns `404` for an unknown strategy id or symbol."
    ),
)
def get_strategy_backtest(
    strategy_id: str = Path(
        description="Strategy id from the catalog.",
        examples=["golden-cross"],
    ),
    symbol: str = Query(
        ...,
        description="Asset ticker to backtest the strategy on (case-insensitive).",
        examples=["AAPL"],
    ),
) -> BacktestResultDTO:
    """Return the realized backtest of one strategy applied to one asset.

    Runs the strategy's vectorized per-bar position series on the asset's close
    history and reports the 14 realized metrics for both the strategy and a buy
    & hold benchmark, plus a downsampled equity curve. Strategies that are not
    time-backtestable per-bar (snapshot / fundamental models) return a buy &
    hold-only result flagged ``supported=False`` (not an error).

    Args:
        strategy_id: Strategy id from the catalog.
        symbol: Asset ticker (case-insensitive).

    Returns:
        A populated :class:`~app.schemas.BacktestResultDTO`.

    Raises:
        HTTPException: ``404`` if ``strategy_id`` is unknown or ``symbol`` is
            not in the universe.
    """
    sid = strategy_id.strip()
    if sid not in META_BY_ID:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown strategy: {strategy_id!r}",
        )
    engine = get_engine()
    try:
        return run_backtest(engine, sid, symbol)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown symbol: {symbol}",
        ) from exc
