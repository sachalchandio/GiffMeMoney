"""Market-level API routes: health check and the dashboard market summary.

These endpoints back the dashboard's top-level overview and the lightweight
liveness probe used by the frontend / deployment tooling. The router carries no
prefix here; ``app.main`` mounts it under ``/api``.

Routes (see CONTRACT section 5):
    * ``GET /api/health``         -> ``{status, time, universe}``
    * ``GET /api/market/summary`` -> :class:`~app.schemas.MarketSummary`
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.market.provider import get_provider
from app.schemas import MarketSummary
from app.strategies.engine import AnalysisEngine

__all__ = ["router"]

router = APIRouter(tags=["market"])

# Process-wide singletons. The provider is the shared market-data singleton and
# the engine caches per-symbol analyses, so reusing one instance across requests
# keeps the dashboard fast and consistent.
_provider = get_provider()
_engine = AnalysisEngine(_provider)


@router.get("/health")
def health() -> dict[str, object]:
    """Liveness probe reporting server time and universe size.

    Returns:
        A dict ``{"status": "ok", "time": <unix_ms>, "universe": <count>}``
        where ``time`` is the current unix timestamp in milliseconds and
        ``universe`` is the number of assets the provider knows about.
    """
    try:
        universe = len(_provider.list_assets())
    except Exception:
        universe = 0
    return {
        "status": "ok",
        "time": int(time.time() * 1000),
        "universe": universe,
    }


@router.get("/market/summary", response_model=MarketSummary)
def market_summary() -> MarketSummary:
    """Return the dashboard :class:`~app.schemas.MarketSummary`.

    Aggregates breadth (advancers/decliners), per-sector average change,
    synthetic per-class index levels, and the top gainers / losers across the
    universe via :meth:`~app.strategies.engine.AnalysisEngine.market_summary`.

    Returns:
        A populated :class:`~app.schemas.MarketSummary`.
    """
    return _engine.market_summary()
