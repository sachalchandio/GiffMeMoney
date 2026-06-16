"""``/api/recommendations`` — ranked investment ideas across the universe.

This router exposes a single endpoint that asks the
:class:`~app.strategies.engine.AnalysisEngine` to rank every (optionally
asset-class-filtered) asset by its composite score and return the strongest
ideas as :class:`~app.schemas.Recommendation` rows (rank 1 = best).

The heavy lifting (running all quant models, computing the composite score,
blending the 5-horizon expected return, and writing the narrative reasons) lives
in the engine; this module is a thin, defensive HTTP adapter. A process-wide
:class:`~app.strategies.engine.AnalysisEngine` singleton is reused across
requests so its per-symbol analysis cache is shared.
"""

from __future__ import annotations

import threading
from typing import Optional

from fastapi import APIRouter, Query

from app.schemas import AssetClass, Recommendation
from app.strategies.engine import AnalysisEngine

__all__ = ["router", "get_engine"]

router = APIRouter(prefix="/api", tags=["recommendations"])

# Process-wide engine singleton so the per-symbol analysis cache is shared across
# requests (and with the other API routers that reuse :func:`get_engine`).
_ENGINE: AnalysisEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> AnalysisEngine:
    """Return the shared :class:`~app.strategies.engine.AnalysisEngine` singleton.

    The engine wraps the process-wide market-data provider and caches per-symbol
    analyses, so reusing one instance keeps the expensive quant pipeline warm.

    Returns:
        The shared :class:`~app.strategies.engine.AnalysisEngine`.
    """
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = AnalysisEngine()
    return _ENGINE


@router.get(
    "/recommendations",
    response_model=list[Recommendation],
    summary="Ranked recommendations across the universe",
)
def get_recommendations(
    limit: int = Query(
        default=12,
        ge=1,
        le=200,
        description="Maximum number of recommendations to return.",
    ),
    asset_class: Optional[AssetClass] = Query(
        default=None,
        alias="assetClass",
        description="Optional asset-class filter (equity / crypto / etf).",
    ),
) -> list[Recommendation]:
    """Return the top ``limit`` assets ranked by composite score (descending).

    Runs the full quant model suite over the (optionally filtered) universe and
    returns the strongest ideas. Symbols whose analysis fails are silently
    skipped by the engine, so this endpoint never raises on a single bad asset.

    Args:
        limit: Maximum number of recommendations (1..200; default 12).
        asset_class: Optional ``'equity'`` / ``'crypto'`` / ``'etf'`` filter
            (camelCase query alias ``assetClass``).

    Returns:
        A list of :class:`~app.schemas.Recommendation`, rank 1 = best composite
        score. An empty list if nothing matches the filter.
    """
    engine = get_engine()
    cls = str(asset_class) if asset_class is not None else None
    return engine.recommendations(limit=limit, asset_class=cls)
