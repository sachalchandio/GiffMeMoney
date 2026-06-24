"""``/api/bot`` — the simulated auto-trader (paper-trading bot).

HONESTY / SAFETY (this is a finance tool). Every route in this module is a
**SIMULATION on synthetic data**: the auto-trader paper-trades a starting cash
balance over a deterministic historical window — no real money moves and no live
broker is ever contacted. Rotation is **momentum / bandit** style (allocate MORE
to recent winners, LESS to losers) and is hard-capped; the engine **never
martingales** — it never increases a losing sleeve's weight to "recover". Every
:class:`~app.schemas.BotRunResult` carries :data:`~app.schemas.BOT_DISCLAIMER`
and nothing here implies guaranteed profit.

Endpoints:

    * ``GET  /api/bot/modes``     — the five preset :class:`~app.schemas.BotMode`
      presets (from :data:`~app.bot.policies.BOT_MODES`).
    * ``POST /api/bot/backtest``  — run one mode over the (optionally
      class-filtered) candidate universe and return its full
      :class:`~app.schemas.BotRunResult`.
    * ``POST /api/bot/compare``   — run several modes against a shared base
      :class:`~app.schemas.BotConfig` for side-by-side comparison.

This router is a thin, defensive HTTP adapter. The heavy lifting (candidate
selection, the monthly-rebalance walk-forward, the momentum/bandit rotation, risk
controls and attribution) lives in :class:`~app.bot.engine.AutoTraderEngine`. To
keep the compute-heavy backtest fast (anti-stall), a process-wide engine
singleton reuses the shared :class:`~app.strategies.engine.AnalysisEngine` from
:func:`app.api.recommendations.get_engine` so its warm per-symbol analysis cache
is shared across every bot run, and the engine itself restricts each run to a
small candidate set and vectorizes daily mark-to-market.

Error mapping:
    * ``400`` — an unknown / unsupported bot mode id.
    * ``404`` — a symbol referenced by the request is not in the universe.
"""

from __future__ import annotations

import threading
from typing import List

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.recommendations import get_engine
from app.bot.engine import AutoTraderEngine
from app.bot.policies import BOT_MODES, MODE_POLICIES
from app.market.provider import get_provider
from app.schemas import (
    BOT_DISCLAIMER,
    BotConfig,
    BotMode,
    BotRunRequest,
    BotRunResult,
)

__all__ = ["router", "get_bot_engine"]

router = APIRouter(prefix="/api/bot", tags=["bot"])


# Process-wide auto-trader engine singleton. It reuses the shared analysis engine
# (and thus its warm per-symbol composite cache) and the process-wide market-data
# provider so repeated bot runs never re-warm the expensive quant pipeline.
_BOT_ENGINE: AutoTraderEngine | None = None
_BOT_ENGINE_LOCK = threading.Lock()


def get_bot_engine() -> AutoTraderEngine:
    """Return the shared :class:`~app.bot.engine.AutoTraderEngine` singleton.

    The engine is bound to the process-wide market-data provider
    (:func:`app.market.provider.get_provider`) and the shared
    :class:`~app.strategies.engine.AnalysisEngine`
    (:func:`app.api.recommendations.get_engine`), so its candidate selection
    reuses the already-warm per-symbol composite cache rather than recomputing
    the quant suite on every request.

    Returns:
        The shared :class:`~app.bot.engine.AutoTraderEngine`.
    """
    global _BOT_ENGINE
    if _BOT_ENGINE is None:
        with _BOT_ENGINE_LOCK:
            if _BOT_ENGINE is None:
                _BOT_ENGINE = AutoTraderEngine(
                    provider=get_provider(), analysis_engine=get_engine()
                )
    return _BOT_ENGINE


def _validate_mode(mode_id: str) -> None:
    """Reject an unknown bot mode id with ``400``.

    Args:
        mode_id: The requested :data:`~app.schemas.BotModeId`.

    Raises:
        HTTPException: ``400`` if ``mode_id`` is not one of the five presets.
    """
    key = str(mode_id or "").strip().lower()
    if key not in MODE_POLICIES:
        valid = ", ".join(MODE_POLICIES.keys())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown bot mode: {mode_id!r}. Valid modes: {valid}.",
        )


def _validate_symbols(config: BotConfig) -> None:
    """Reject any symbol the config pins that is not in the tradable universe.

    :class:`~app.schemas.BotConfig` selects its candidates by asset-class filter
    rather than by explicit ticker, so there is usually nothing to validate. This
    guard defensively rejects any explicit ``symbols`` the config may carry (now
    or in a future extension) that are not in the universe, mapping the unknown
    ticker to ``404`` rather than silently dropping it.

    Args:
        config: The bot configuration to inspect.

    Raises:
        HTTPException: ``404`` if the config references an unknown symbol.
    """
    requested = getattr(config, "symbols", None)
    if not requested:
        return
    try:
        known = {str(a.symbol).upper() for a in get_provider().list_assets()}
    except Exception:  # pragma: no cover - defensive
        return
    for raw in requested:
        sym = str(raw or "").strip().upper()
        if sym and sym not in known:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown symbol: {raw}",
            )


def _run_config(config: BotConfig) -> BotRunResult:
    """Validate then backtest one :class:`~app.schemas.BotConfig`.

    Args:
        config: The bot configuration to simulate.

    Returns:
        The populated :class:`~app.schemas.BotRunResult` (carries the disclaimer).

    Raises:
        HTTPException: ``400`` for an unknown mode, ``404`` for an unknown symbol.
    """
    _validate_mode(config.mode)
    _validate_symbols(config)
    return get_bot_engine().backtest(config)


# ---------------------------------------------------------------------------
# Compare request body
# ---------------------------------------------------------------------------


class BotCompareRequest(BaseModel):
    """Request body for ``POST /api/bot/compare``.

    Runs several bot modes against a single shared base configuration so their
    results can be lined up side-by-side. Each requested mode id overrides the
    ``mode`` field of ``config`` for that run; everything else in ``config``
    (starting amount, class filter, rebalance cadence, risk limits) is held
    constant across the comparison.

    The two field names (``modes`` / ``config``) are already identical in
    camelCase, so this body deliberately uses a plain :class:`pydantic.BaseModel`
    (no alias generator) — the nested ``config`` is the camelCase
    :class:`~app.schemas.BotConfig` and is unaffected.

    Fields:
        modes: The :data:`~app.schemas.BotModeId` ids to run side-by-side
            (defaults to all five presets when empty).
        config: The shared base :class:`~app.schemas.BotConfig`; its ``mode`` is
            overridden per requested id.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "modes": ["conservative", "balanced", "aggressive"],
                    "config": {"amount": 10000, "mode": "balanced"},
                }
            ]
        }
    )

    modes: List[str] = Field(default_factory=list)
    config: BotConfig


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get(
    "/modes",
    response_model=list[BotMode],
    summary="List the preset auto-trader modes",
    description=(
        "Return the five preset auto-trader modes (Conservative, Balanced, "
        "Aggressive, Adaptive Bandit, All-Weather): each carries its risk badge, "
        "portfolio objective, rotation style and maximum number of sleeves.\n\n"
        "**Simulation disclaimer.** " + BOT_DISCLAIMER + " Rotation allocates "
        "more to recent winners and less to losers (momentum / bandit) and is "
        "hard-capped; the bot never increases a losing sleeve to chase losses."
    ),
)
def list_modes() -> list[BotMode]:
    """Return the five preset :class:`~app.schemas.BotMode` definitions.

    Returns:
        The ordered list of preset modes from
        :data:`~app.bot.policies.BOT_MODES`.
    """
    return list(BOT_MODES)


@router.post(
    "/backtest",
    response_model=BotRunResult,
    summary="Backtest one simulated auto-trader mode",
    description=(
        "Run the simulated paper-trading auto-trader for one mode over the "
        "(optionally asset-class filtered) candidate universe and return its "
        "full result: the bot-vs-benchmark equity curve (with per-bar regime and "
        "drawdown), the simulated trade blotter, per-sleeve attribution "
        "(best → worst), realized metrics and the regime timeline.\n\n"
        "The run is efficient by construction: candidates are scored once with "
        "the shared, warm analysis engine, the book is restricted to a small set, "
        "rebalances happen on a ~21-trading-day (monthly) grid, and daily "
        "mark-to-market is vectorized.\n\n"
        "**Simulation disclaimer.** " + BOT_DISCLAIMER + " Rotation is momentum "
        "/ bandit (more to recent winners, less to losers), hard-capped, and "
        "never martingale.\n\n"
        "**Status codes**\n"
        "- `200` — the populated `BotRunResult`.\n"
        "- `400` — an unknown / unsupported bot mode id.\n"
        "- `404` — the config references a symbol not in the universe."
    ),
    responses={
        200: {"description": "The simulated auto-trader run result."},
        400: {"description": "Unknown / unsupported bot mode."},
        404: {"description": "A referenced symbol is not in the universe."},
    },
)
def run_backtest(body: BotRunRequest) -> BotRunResult:
    """Backtest one simulated auto-trader configuration.

    Args:
        body: The :class:`~app.schemas.BotRunRequest` wrapping the
            :class:`~app.schemas.BotConfig` to simulate.

    Returns:
        The populated :class:`~app.schemas.BotRunResult` (carries the mandatory
        simulation disclaimer).

    Raises:
        HTTPException: ``400`` for an unknown mode; ``404`` for an unknown symbol.
    """
    return _run_config(body.config)


@router.post(
    "/compare",
    response_model=list[BotRunResult],
    summary="Compare several auto-trader modes side-by-side",
    description=(
        "Run several simulated auto-trader modes against one shared base "
        "configuration and return a `BotRunResult` per mode (in the requested "
        "order) for side-by-side comparison. Each requested mode id overrides the "
        "`mode` field of the shared `config`; every other field (starting amount, "
        "class filter, rebalance cadence, risk limits) is held constant. When "
        "`modes` is empty all five presets are run.\n\n"
        "Each run reuses the shared, warm analysis cache, so comparing modes does "
        "not re-warm the quant pipeline.\n\n"
        "**Simulation disclaimer.** " + BOT_DISCLAIMER + " Every mode rotates "
        "momentum / bandit style (more to recent winners, less to losers), "
        "hard-capped, never martingale.\n\n"
        "**Status codes**\n"
        "- `200` — one `BotRunResult` per requested mode.\n"
        "- `400` — one of the requested mode ids is unknown / unsupported.\n"
        "- `404` — the config references a symbol not in the universe."
    ),
    responses={
        200: {"description": "One simulated run result per requested mode."},
        400: {"description": "An unknown / unsupported bot mode in the list."},
        404: {"description": "A referenced symbol is not in the universe."},
    },
)
def compare_modes(body: BotCompareRequest) -> list[BotRunResult]:
    """Run several auto-trader modes against one base config, side-by-side.

    The requested ``modes`` each override the ``mode`` of the shared base
    ``config``; all other config fields are held constant so the runs are
    directly comparable. An empty ``modes`` list runs all five presets.

    Args:
        body: The :class:`BotCompareRequest` (mode ids + shared base config).

    Returns:
        A list of :class:`~app.schemas.BotRunResult`, one per requested mode in
        order (each carrying the mandatory simulation disclaimer).

    Raises:
        HTTPException: ``400`` if any requested mode is unknown; ``404`` if the
            shared config references an unknown symbol.
    """
    base = body.config
    # Validate the shared symbol references once up front (404 short-circuit).
    _validate_symbols(base)

    requested: List[str] = [str(m or "").strip() for m in (body.modes or []) if str(m or "").strip()]
    if not requested:
        requested = [m.id for m in BOT_MODES]

    # Validate every requested mode before running any (400 short-circuit) so a
    # bad id never produces a partial result set.
    for mode_id in requested:
        _validate_mode(mode_id)

    results: list[BotRunResult] = []
    for mode_id in requested:
        run_config = base.model_copy(update={"mode": str(mode_id).strip().lower()})
        results.append(get_bot_engine().backtest(run_config))
    return results
