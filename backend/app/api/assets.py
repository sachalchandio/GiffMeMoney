"""Asset-level API routes: listing, detail, candles, analysis, Monte Carlo.

The router carries no prefix here; ``app.main`` mounts it under ``/api``. Unknown
symbols return ``404 {"detail": ...}``; all response bodies are the frozen
section-4 DTOs from :mod:`app.schemas` (camelCase on the wire).

Routes (see CONTRACT section 5):
    * ``GET /api/assets``                       -> ``Asset[]`` (``assetClass?``)
    * ``GET /api/assets/{symbol}``              -> ``Asset``
    * ``GET /api/assets/{symbol}/candles``      -> ``Candle[]``
    * ``GET /api/assets/{symbol}/analysis``     -> ``AssetAnalysis``
    * ``GET /api/assets/{symbol}/montecarlo``   -> ``MonteCarloResult``
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.market.provider import get_provider
from app.schemas import (
    Asset,
    AssetAnalysis,
    Candle,
    HORIZONS,
    MonteCarloResult,
)
from app.strategies.engine import AnalysisEngine

__all__ = ["router"]

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
