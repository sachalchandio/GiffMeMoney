"""Asset-level API routes: listing, detail, candles, analysis, Monte Carlo, backtest.

The router carries no prefix here; ``app.main`` mounts it under ``/api``. Unknown
symbols return ``404 {"detail": ...}``; all response bodies are the frozen
section-4 DTOs from :mod:`app.schemas` (camelCase on the wire).

Routes (see CONTRACT section 5 + STRATEGIES-V2 §8):
    * ``GET /api/assets``                       -> ``Asset[]`` (``assetClass?``)
    * ``GET /api/assets/{symbol}``              -> ``Asset``
    * ``GET /api/assets/{symbol}/candles``      -> ``Candle[]``
    * ``GET /api/assets/{symbol}/analysis``     -> ``AssetAnalysis``
    * ``GET /api/assets/{symbol}/montecarlo``   -> ``MonteCarloResult``
    * ``GET /api/assets/{symbol}/backtest``     -> ``BacktestResultDTO`` (V2)

This module also hosts the shared backtest service helpers
(:func:`run_backtest`, :func:`leaderboard_for_symbol`) reused by the
``/api/strategies`` router so both surfaces share one implementation of "run a
strategy's position series through the backtester".
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.market.provider import get_provider
from app.quant.backtest import BacktestMetrics, BacktestResult, backtest_positions
from app.schemas import (
    Asset,
    AssetAnalysis,
    BacktestEquityPoint,
    BacktestMetricsDTO,
    BacktestResultDTO,
    Candle,
    HORIZONS,
    MonteCarloResult,
    StrategyLeaderboard,
    StrategyLeaderboardEntry,
)
from app.strategies.engine import AnalysisContext, AnalysisEngine
from app.strategies.registry import META_BY_ID, POSITION_FUNCS, STRATEGY_META

__all__ = [
    "router",
    "run_backtest",
    "leaderboard_for_symbol",
]

router = APIRouter(prefix="/assets", tags=["assets"])

# Process-wide singletons shared with the rest of the API. The engine caches
# per-symbol analyses so repeated detail / recommendation calls are cheap.
_provider = get_provider()
_engine = AnalysisEngine(_provider)


@router.get("", response_model=list[Asset])
def list_assets(
    asset_class: Optional[str] = Query(default=None, alias="assetClass"),
) -> list[Asset]:
    """List every asset, optionally filtered by asset class.

    Args:
        asset_class: Optional ``assetClass`` filter — one of ``'equity'``,
            ``'crypto'`` or ``'etf'`` (case-insensitive). An unknown value
            simply yields an empty list.

    Returns:
        A list of :class:`~app.schemas.Asset` snapshots.
    """
    assets = _provider.list_assets()
    if asset_class:
        wanted = asset_class.strip().lower()
        assets = [a for a in assets if str(a.asset_class).lower() == wanted]
    return assets


@router.get("/{symbol}", response_model=Asset)
def get_asset(symbol: str) -> Asset:
    """Return a single :class:`~app.schemas.Asset` snapshot.

    Args:
        symbol: Asset ticker (case-insensitive).

    Returns:
        The :class:`~app.schemas.Asset` snapshot.

    Raises:
        HTTPException: ``404`` if the symbol is unknown.
    """
    try:
        return _provider.get_asset(symbol)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")


@router.get("/{symbol}/candles", response_model=list[Candle])
def get_candles(
    symbol: str,
    interval: str = Query(default="1d"),
    limit: int = Query(default=365, ge=1, le=5000),
) -> list[Candle]:
    """Return up to ``limit`` recent OHLCV candles for a symbol.

    Args:
        symbol: Asset ticker (case-insensitive).
        interval: Candle interval (only daily ``'1d'`` is simulated; accepted
            for forward compatibility with real providers).
        limit: Maximum number of most-recent candles (1..5000).

    Returns:
        A list of :class:`~app.schemas.Candle`, ordered oldest -> newest.

    Raises:
        HTTPException: ``404`` if the symbol is unknown.
    """
    try:
        return _provider.get_candles(symbol, limit=limit)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")


@router.get("/{symbol}/analysis", response_model=AssetAnalysis)
def get_analysis(symbol: str) -> AssetAnalysis:
    """Return the full composite :class:`~app.schemas.AssetAnalysis`.

    Runs every registered quant model for the symbol and blends them into a
    composite score, recommendation, 5-horizon expected returns, risk metrics
    and a narrative rationale.

    Args:
        symbol: Asset ticker (case-insensitive).

    Returns:
        A complete :class:`~app.schemas.AssetAnalysis`.

    Raises:
        HTTPException: ``404`` if the symbol is unknown.
    """
    try:
        return _engine.analyze(symbol)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")


@router.get("/{symbol}/montecarlo", response_model=MonteCarloResult)
def get_montecarlo(
    symbol: str,
    horizon: str = Query(default="1Y"),
    sims: int = Query(default=2000, ge=1, le=100000),
) -> MonteCarloResult:
    """Run a GBM Monte Carlo simulation for a symbol over a horizon.

    Args:
        symbol: Asset ticker (case-insensitive).
        horizon: One of :data:`~app.schemas.HORIZONS`; unknown values fall back
            to ``'1Y'``.
        sims: Number of simulated paths (1..100000).

    Returns:
        A populated :class:`~app.schemas.MonteCarloResult` with price percentile
        bands, the terminal-price distribution, and VaR/CVaR/probPositive.

    Raises:
        HTTPException: ``404`` if the symbol is unknown.
    """
    hz = horizon if horizon in HORIZONS else "1Y"
    try:
        return _engine.montecarlo(symbol, horizon=hz, sims=sims)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")


@router.get("/{symbol}/backtest", response_model=BacktestResultDTO)
def get_backtest(
    symbol: str,
    strategy: str = Query(
        ...,
        description="Strategy id to backtest on this asset (e.g. 'golden-cross').",
    ),
) -> BacktestResultDTO:
    """Return the realized backtest of one strategy applied to this asset.

    Runs the strategy's vectorized per-bar position series on the asset's close
    history and reports the 14 realized metrics for both the strategy and a buy
    & hold benchmark, plus a downsampled equity curve. Strategies that are not
    time-backtestable per-bar (snapshot / fundamental models) return a buy &
    hold-only result flagged ``supported=False`` (not an error).

    Args:
        symbol: Asset ticker (case-insensitive).
        strategy: Strategy id from the catalog.

    Returns:
        A populated :class:`~app.schemas.BacktestResultDTO`.

    Raises:
        HTTPException: ``404`` if the ``strategy`` id is unknown or the
            ``symbol`` is not in the universe.
    """
    sid = strategy.strip()
    if sid not in META_BY_ID:
        raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy!r}")
    try:
        return run_backtest(_engine, sid, symbol)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown symbol: {symbol}")


# ---------------------------------------------------------------------------
# Shared backtest service (reused by the /api/strategies router)
# ---------------------------------------------------------------------------


def _metrics_to_dto(m: BacktestMetrics) -> BacktestMetricsDTO:
    """Convert a :class:`~app.quant.backtest.BacktestMetrics` to its wire DTO.

    Args:
        m: The realized-performance metrics dataclass.

    Returns:
        A :class:`~app.schemas.BacktestMetricsDTO` (all fields finite by the
        backtester's own guards).
    """
    return BacktestMetricsDTO(
        cagr=m.cagr,
        total_return=m.total_return,
        ann_vol=m.ann_vol,
        sharpe=m.sharpe,
        sortino=m.sortino,
        calmar=m.calmar,
        max_drawdown=m.max_drawdown,
        ulcer_index=m.ulcer_index,
        win_rate=m.win_rate,
        profit_factor=m.profit_factor,
        exposure=m.exposure,
        turnover=m.turnover,
        cvar95=m.cvar95,
        beta=m.beta,
        information_ratio=m.information_ratio,
    )


def _result_to_dto(result: BacktestResult) -> BacktestResultDTO:
    """Convert a :class:`~app.quant.backtest.BacktestResult` to its wire DTO.

    The equity-curve dicts carry ``t`` / ``strategy`` / ``benchmark`` keys which
    map straight onto :class:`~app.schemas.BacktestEquityPoint`.

    Args:
        result: The backtest result dataclass.

    Returns:
        A populated :class:`~app.schemas.BacktestResultDTO`.
    """
    curve = [
        BacktestEquityPoint(
            t=int(pt.get("t", 0)),
            strategy=float(pt.get("strategy", 1.0)),
            benchmark=float(pt.get("benchmark", 1.0)),
        )
        for pt in result.equity_curve
    ]
    return BacktestResultDTO(
        symbol=result.symbol,
        strategy_id=result.strategy_id,
        supported=bool(result.supported),
        trades=int(result.trades),
        metrics=_metrics_to_dto(result.metrics),
        benchmark=_metrics_to_dto(result.benchmark),
        equity_curve=curve,
    )


def _backtest_context(
    ctx: AnalysisContext, strategy_id: str
) -> BacktestResult:
    """Run one strategy's position series for a prepared context (no DTO wrap).

    Looks the strategy's vectorized ``positions(...)`` function up in
    :data:`~app.strategies.registry.POSITION_FUNCS`. When present the positions
    are computed from the context's OHLCV arrays and run through
    :func:`~app.quant.backtest.backtest_positions`; when absent (snapshot /
    fundamental strategies) a buy & hold-only result flagged ``supported=False``
    is returned. Never raises — a failing position function degrades to an
    unsupported buy & hold result.

    Args:
        ctx: A prepared :class:`~app.strategies.engine.AnalysisContext`.
        strategy_id: Strategy id (assumed already validated against the catalog).

    Returns:
        A :class:`~app.quant.backtest.BacktestResult`.
    """
    closes = ctx.closes
    highs = ctx.highs if getattr(ctx, "highs", None) is not None else None
    lows = ctx.lows if getattr(ctx, "lows", None) is not None else None
    volumes = ctx.volumes if getattr(ctx, "volumes", None) is not None else None
    rf_daily = float(ctx.rf_daily)
    symbol = ctx.asset.symbol

    pos_fn = POSITION_FUNCS.get(strategy_id)
    if pos_fn is None:
        # Not time-backtestable per-bar: buy & hold-only result.
        return backtest_positions(
            closes,
            positions=[],
            rf_daily=rf_daily,
            symbol=symbol,
            strategy_id=strategy_id,
            supported=False,
            highs=highs,
            lows=lows,
        )

    try:
        positions = pos_fn(closes, highs, lows, volumes, None)
    except Exception:
        # A misbehaving position function must not 500 the endpoint; fall back to
        # an unsupported buy & hold result.
        return backtest_positions(
            closes,
            positions=[],
            rf_daily=rf_daily,
            symbol=symbol,
            strategy_id=strategy_id,
            supported=False,
            highs=highs,
            lows=lows,
        )

    return backtest_positions(
        closes,
        positions=positions,
        rf_daily=rf_daily,
        symbol=symbol,
        strategy_id=strategy_id,
        supported=True,
        highs=highs,
        lows=lows,
    )


def run_backtest(
    engine: AnalysisEngine, strategy_id: str, symbol: str
) -> BacktestResultDTO:
    """Backtest one strategy on one asset and return the wire DTO.

    Builds the asset's analysis context once (a single-symbol provider hit),
    runs the strategy's position series through the backtester, and converts the
    result to a :class:`~app.schemas.BacktestResultDTO`.

    Args:
        engine: The shared :class:`~app.strategies.engine.AnalysisEngine`.
        strategy_id: Strategy id (assumed already validated against the catalog).
        symbol: Asset ticker (case-insensitive).

    Returns:
        A populated :class:`~app.schemas.BacktestResultDTO`.

    Raises:
        KeyError: If the symbol is unknown (propagated so the API can 404).
    """
    ctx = engine.context(symbol)
    result = _backtest_context(ctx, strategy_id)
    return _result_to_dto(result)


def leaderboard_for_symbol(
    engine: AnalysisEngine, symbol: str, limit: int = 20
) -> StrategyLeaderboard:
    """Rank every registered strategy by realized backtest performance for an asset.

    Builds the asset's context once, backtests every strategy's position series
    on it, and ranks the results best-first by Sharpe (CAGR breaks ties).
    Strategies without a per-bar position function are reported ``supported=False``
    and sorted to the bottom. The asset's buy & hold metrics (identical across
    strategies) are returned once as the benchmark bar to beat.

    Args:
        engine: The shared :class:`~app.strategies.engine.AnalysisEngine`.
        symbol: Asset ticker (case-insensitive).
        limit: Maximum number of leaderboard entries to return.

    Returns:
        A populated :class:`~app.schemas.StrategyLeaderboard`.

    Raises:
        KeyError: If the symbol is unknown (propagated so the API can 404).
    """
    ctx = engine.context(symbol)
    asset_symbol = ctx.asset.symbol

    rows: list[tuple[StrategyLeaderboardEntry, bool, float, float]] = []
    benchmark_metrics: BacktestMetrics | None = None

    for meta in STRATEGY_META:
        result = _backtest_context(ctx, meta.id)
        if benchmark_metrics is None:
            benchmark_metrics = result.benchmark
        m = result.metrics
        entry = StrategyLeaderboardEntry(
            rank=0,  # filled in after sorting
            strategy_id=meta.id,
            strategy_name=meta.name,
            category=meta.category,
            supported=bool(result.supported),
            sharpe=float(m.sharpe),
            cagr=float(m.cagr),
            total_return=float(m.total_return),
            max_drawdown=float(m.max_drawdown),
            calmar=float(m.calmar),
            win_rate=float(m.win_rate),
            trades=int(result.trades),
        )
        rows.append((entry, bool(result.supported), float(m.sharpe), float(m.cagr)))

    # Rank: supported strategies first, then by Sharpe desc, then CAGR desc.
    rows.sort(key=lambda r: (r[1], r[2], r[3]), reverse=True)

    lim = max(0, int(limit)) if limit else len(rows)
    entries: list[StrategyLeaderboardEntry] = []
    for rank, (entry, _supported, _sharpe, _cagr) in enumerate(rows[:lim], start=1):
        entry.rank = rank
        entries.append(entry)

    bench_dto = (
        _metrics_to_dto(benchmark_metrics)
        if benchmark_metrics is not None
        else _metrics_to_dto(BacktestMetrics.zeros())
    )
    return StrategyLeaderboard(
        symbol=asset_symbol,
        benchmark=bench_dto,
        entries=entries,
    )
