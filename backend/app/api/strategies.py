"""``/api/strategies`` — the quant model catalog and cross-asset rankings.

Two endpoints:

    * ``GET /api/strategies`` — the static catalog of every registered model
      (:data:`~app.strategies.registry.STRATEGY_META`, >= 18 entries).
    * ``GET /api/strategies/{id}/rankings`` — every asset ranked by a single
      strategy's signal score (404 if the strategy id is unknown).

The catalog is served straight from the registry; the per-strategy ranking is
produced by the shared :class:`~app.strategies.engine.AnalysisEngine`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.api.recommendations import get_engine
from app.schemas import StrategyMeta, StrategyRanking
from app.strategies.registry import STRATEGY_META

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["strategies"])


@router.get(
    "/strategies",
    response_model=list[StrategyMeta],
    summary="List the full quant model catalog",
)
def list_strategies() -> list[StrategyMeta]:
    """Return the static metadata for every registered strategy.

    The catalog mirrors section 7 of the contract (all 20 ids) in declaration
    order: id, name, category, summary, formula, inputs and references.

    Returns:
        A list of :class:`~app.schemas.StrategyMeta` (>= 18 entries).
    """
    return list(STRATEGY_META)


@router.get(
    "/strategies/{strategy_id}/rankings",
    response_model=StrategyRanking,
    summary="Rank every asset by one strategy's signal score",
)
def get_strategy_rankings(
    strategy_id: str,
    limit: int = Query(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of ranked entries to return.",
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
