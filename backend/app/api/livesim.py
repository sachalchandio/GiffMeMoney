"""``/api/livesim`` — the Real-Time mode (live-feeling PAPER simulation).

HONESTY / SAFETY (this is a finance tool). Every route is an **accelerated
SIMULATION on synthetic data**. No real money, no bank, no live broker — $0 real.
Trades are charged realistic costs and returns are kept realistic, so this never
turns $20 into thousands. Every payload carries
:data:`~app.schemas.LIVESIM_DISCLAIMER`.

Endpoints:

    * ``POST /api/livesim/start``       — create a session, return its first state.
    * ``POST /api/livesim/tick``        — advance a session, return the new state.
    * ``GET  /api/livesim/state/{id}``  — read current state without advancing.
    * ``POST /api/livesim/stop/{id}``   — end a session.

The frontend drives ticks on a short interval to produce the real-time feel.
"""

from __future__ import annotations

import math

from fastapi import APIRouter, HTTPException, status

from app.hft.costs import CostModel
from app.livesim.engine import (
    LIVESIM_MAX_VENUES,
    LIVESIM_MIN_VENUES,
    LiveSimSession,
)
from app.livesim.store import get_store
from app.schemas import (
    LIVESIM_DISCLAIMER,
    BotTrade,
    HftCostModel,
    LiveSimDayPoint,
    LiveSimStartRequest,
    LiveSimState,
    LiveSimTickRequest,
    LiveSimVenue,
)

__all__ = ["router"]

router = APIRouter(prefix="/api/livesim", tags=["livesim"])

_EPS = 1e-12
_CURVE_POINTS = 180


def _finite(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _cost_dto(model: CostModel) -> HftCostModel:
    """Map a cost model to its wire DTO."""
    return HftCostModel(
        key=model.key,
        name=model.name,
        half_spread_bps=model.half_spread_bps,
        fee_bps=model.fee_bps,
        impact_coef=model.impact_coef,
        round_trip_bps=round(model.round_trip_bps(), 4),
        note=model.note,
    )


def _downsample(values: list[float], points: int = _CURVE_POINTS) -> list[float]:
    """Down-sample a series to ~``points`` evenly-spaced values (keeps the ends)."""
    n = len(values)
    if n <= points:
        return [round(float(x), 4) for x in values]
    step = (n - 1) / float(points - 1)
    out: list[float] = []
    seen = -1
    for i in range(points):
        idx = int(round(i * step))
        if idx != seen:
            out.append(round(float(values[idx]), 4))
            seen = idx
    return out


def _max_drawdown_pct(curve: list[float]) -> float:
    """Worst peak-to-trough drawdown of an equity curve, in percent (``<= 0``)."""
    peak = -math.inf
    worst = 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > _EPS:
            dd = v / peak - 1.0
            worst = min(worst, dd)
    return round(_finite(worst) * 100.0, 4)


def _state_dto(session: LiveSimSession) -> LiveSimState:
    """Project a live-sim session into its wire DTO."""
    equity = session.equity()
    start = session.start_equity if session.start_equity > _EPS else session.amount
    total_pnl = (equity / start - 1.0) * 100.0 if start > _EPS else 0.0
    day_base = session.day_start_equity if session.day_start_equity > _EPS else start
    day_pnl = (equity / day_base - 1.0) * 100.0 if day_base > _EPS else 0.0

    # How wide the book is currently spread (matches the engine's rule).
    target_n = int(equity / session.dollars_per_venue) if session.dollars_per_venue > _EPS else 0
    target_n = max(LIVESIM_MIN_VENUES, min(target_n, session.max_venues, len(session.venues)))

    venues: list[LiveSimVenue] = []
    active = 0
    for v in sorted(session.venues, key=lambda x: x.score, reverse=True):
        price = v.price
        pos_val = v.units * price
        held = v.units > _EPS
        if held:
            active += 1
        pnl_pct = (price / v.entry_price - 1.0) * 100.0 if held and v.entry_price > _EPS else 0.0
        venues.append(
            LiveSimVenue(
                symbol=v.symbol,
                label=v.label,
                price=round(_finite(price), 6),
                pred_up_prob=round(_finite(v.pred_up, 0.5), 4),
                confidence=round(_finite(abs(v.pred_up - 0.5) * 2.0), 4),
                score=round(_finite(v.score), 4),
                weight_pct=round(_finite(pos_val / equity * 100.0) if equity > _EPS else 0.0, 2),
                position_value=round(_finite(pos_val), 2),
                pnl_pct=round(_finite(pnl_pct), 2),
                held=held,
            )
        )

    trades = [
        BotTrade(
            t=int(tr.get("step", 0)),
            symbol=str(tr.get("symbol", "")),
            side=str(tr.get("side", "buy")),  # type: ignore[arg-type]
            amount=_finite(tr.get("amount", 0.0)),
            strategy=session.signal,
            price=_finite(tr.get("price", 0.0)),
        )
        for tr in reversed(session.trades[-20:])
    ]

    return LiveSimState(
        session_id=session.id,
        step=session.step,
        day=session.day,
        finished=session.finished,
        equity=round(_finite(equity), 2),
        cash=round(_finite(session.cash), 2),
        start_equity=round(_finite(start), 2),
        total_pnl_pct=round(_finite(total_pnl), 4),
        day_pnl_pct=round(_finite(day_pnl), 4),
        max_drawdown_pct=_max_drawdown_pct(session.equity_curve),
        venues_active=active,
        venues_target=target_n,
        venues_max=session.max_venues,
        equity_curve=_downsample(session.equity_curve),
        daily_pnl=[
            LiveSimDayPoint(day=int(d["day"]), pnl_pct=_finite(d["pnlPct"]), equity=_finite(d["equity"]))
            for d in session.daily_pnl[-120:]
        ],
        venues=venues,
        recent_trades=trades,
        cost_model=_cost_dto(session.cost_model),
    )


@router.post(
    "/start",
    response_model=LiveSimState,
    summary="Start a Real-Time-mode paper session",
    description=(
        "Create a live-feeling, multi-venue PAPER session and return its first "
        "state. The book spreads across more venues as equity grows (hard-capped "
        "at 80) and rotates toward the venues scoring best, charging realistic "
        "costs.\n\n**Simulation disclaimer.** " + LIVESIM_DISCLAIMER
    ),
)
def start(body: LiveSimStartRequest) -> LiveSimState:
    """Start a new session and return its initial state."""
    session = get_store().start(
        amount=body.amount,
        signal=body.signal,
        cost_preset=body.cost_preset,
        dollars_per_venue=body.dollars_per_venue,
        max_venues=body.max_venues,
        rebalance_every=body.rebalance_every,
        stop_loss_pct=body.stop_loss_pct,
        max_drawdown_pct=body.max_drawdown_pct,
        steps_per_tick=body.steps_per_tick,
    )
    return _state_dto(session)


@router.post(
    "/tick",
    response_model=LiveSimState,
    summary="Advance a Real-Time-mode session",
    description=(
        "Advance the session forward (learn, re-score, rotate, mark) and return "
        "the new state. The frontend calls this on a short interval to produce "
        "the real-time feel.\n\n**Simulation disclaimer.** " + LIVESIM_DISCLAIMER + "\n\n"
        "**Status codes**\n- `200` — the new state.\n- `404` — unknown session id."
    ),
    responses={200: {"description": "The advanced session state."}, 404: {"description": "Unknown session."}},
)
def tick(body: LiveSimTickRequest) -> LiveSimState:
    """Advance a session by ``steps`` and return its new state."""
    session = get_store().tick(body.session_id, body.steps)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Unknown or expired session. Start a new one.",
        )
    return _state_dto(session)


@router.get(
    "/state/{session_id}",
    response_model=LiveSimState,
    summary="Read a session's current state",
    description="Return the current state without advancing.\n\n**Simulation disclaimer.** " + LIVESIM_DISCLAIMER,
    responses={404: {"description": "Unknown session."}},
)
def get_state(session_id: str) -> LiveSimState:
    """Return a session's current state without advancing it."""
    session = get_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown or expired session.")
    return _state_dto(session)


@router.post(
    "/stop/{session_id}",
    summary="End a Real-Time-mode session",
    description="Remove a session from memory. Idempotent.",
)
def stop(session_id: str) -> dict:
    """End a session. Always returns ``{ok: true}`` (idempotent)."""
    get_store().stop(session_id)
    return {"ok": True}
