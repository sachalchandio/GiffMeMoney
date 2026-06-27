"""``/api/hft`` — the High-Frequency Simulation Lab (paper-only).

HONESTY / SAFETY (this is a finance tool). Every route here is a **SIMULATION on
synthetic data**. The lab exists to answer one question truthfully: *does trading
faster / in smaller portions make more money, or less?* It models the things that
actually decide that — the bid-ask spread, fees, slippage and noise — and shows
that past a low turnover, every extra trade just feeds the spread. It is explicit
that this is **not** microsecond trading: a web app's broker round-trip is ~100ms,
a million times slower than co-located HFT, so the lab simulates *bars*. No real
money moves and every payload carries :data:`~app.schemas.HFT_DISCLAIMER`.

Endpoints:

    * ``GET  /api/hft/cost-presets`` — the transaction-cost presets.
    * ``POST /api/hft/simulate``     — run one configuration (gross vs net vs
      buy-&-hold equity curves + realized metrics).
    * ``POST /api/hft/sweep``        — vary only the trade frequency and return
      the turnover curve, the net-of-cost optimum, and a plain-English verdict.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.hft.costs import COST_PRESETS, CostModel, get_cost_model
from app.hft.execution import SimResult, SimSpec, run_sim
from app.hft.lab import SweepResult, run_sweep
from app.schemas import (
    HFT_DISCLAIMER,
    HftCostModel,
    HftSimMetrics,
    HftSimRequest,
    HftSimResult,
    HftSweepPoint,
    HftSweepRequest,
    HftSweepResult,
)

__all__ = ["router"]

router = APIRouter(prefix="/api/hft", tags=["hft"])


def _cost_dto(model: CostModel) -> HftCostModel:
    """Map a :class:`~app.hft.costs.CostModel` to its wire DTO."""
    return HftCostModel(
        key=model.key,
        name=model.name,
        half_spread_bps=model.half_spread_bps,
        fee_bps=model.fee_bps,
        impact_coef=model.impact_coef,
        round_trip_bps=round(model.round_trip_bps(), 4),
        note=model.note,
    )


def _spec_from(req: HftSimRequest) -> SimSpec:
    """Build a :class:`~app.hft.execution.SimSpec` from a request DTO."""
    return SimSpec(
        symbol=req.symbol,
        amount=req.amount,
        days=req.days,
        bars_per_day=req.bars_per_day,
        signal=req.signal,
        lookback=req.lookback,
        rebalance_interval=req.rebalance_interval,
        deadband=req.deadband,
        target_vol=req.target_vol,
        max_exposure=req.max_exposure,
        allow_short=req.allow_short,
        stop_loss_pct=req.stop_loss_pct,
        take_profit_pct=req.take_profit_pct,
        max_drawdown_pct=req.max_drawdown_pct,
        cooldown_bars=req.cooldown_bars,
        cost_preset=req.cost_preset,
    )


def _sim_result_dto(res: SimResult, cost: CostModel) -> HftSimResult:
    """Map an engine :class:`~app.hft.execution.SimResult` to its wire DTO."""
    m = res.metrics
    return HftSimResult(
        metrics=HftSimMetrics(
            gross_return_pct=m.gross_return_pct,
            net_return_pct=m.net_return_pct,
            cost_drag_pct=m.cost_drag_pct,
            buy_hold_return_pct=m.buy_hold_return_pct,
            vs_buy_hold_pct=m.vs_buy_hold_pct,
            turnover=m.turnover,
            turnover_per_day=m.turnover_per_day,
            trades=m.trades,
            time_in_market_pct=m.time_in_market_pct,
            sharpe_net=m.sharpe_net,
            sharpe_gross=m.sharpe_gross,
            max_drawdown_pct=m.max_drawdown_pct,
            hit_rate_pct=m.hit_rate_pct,
            final_net_value=m.final_net_value,
        ),
        net_curve=res.net_curve,
        gross_curve=res.gross_curve,
        buy_hold_curve=res.buy_hold_curve,
        exposure_curve=res.exposure_curve,
        bars=res.bars,
        bars_per_year=res.bars_per_year,
        cost_model=_cost_dto(cost),
    )


def _sweep_point_dto(p) -> HftSweepPoint:
    """Map an engine sweep point to its wire DTO."""
    return HftSweepPoint(
        interval=p.interval,
        label=p.label,
        turnover=p.turnover,
        turnover_per_day=p.turnover_per_day,
        trades=p.trades,
        gross_return_pct=p.gross_return_pct,
        net_return_pct=p.net_return_pct,
        cost_drag_pct=p.cost_drag_pct,
        sharpe_net=p.sharpe_net,
        max_drawdown_pct=p.max_drawdown_pct,
        vs_buy_hold_pct=p.vs_buy_hold_pct,
    )


def _sweep_result_dto(res: SweepResult) -> HftSweepResult:
    """Map an engine :class:`~app.hft.lab.SweepResult` to its wire DTO."""
    return HftSweepResult(
        points=[_sweep_point_dto(p) for p in res.points],
        optimum_by_net_return=_sweep_point_dto(res.optimum_by_net_return)
        if res.optimum_by_net_return
        else None,
        optimum_by_net_sharpe=_sweep_point_dto(res.optimum_by_net_sharpe)
        if res.optimum_by_net_sharpe
        else None,
        naive_fast=_sweep_point_dto(res.naive_fast) if res.naive_fast else None,
        buy_hold_return_pct=res.buy_hold_return_pct,
        verdict=res.verdict,
    )


@router.get(
    "/cost-presets",
    response_model=list[HftCostModel],
    summary="List the transaction-cost presets",
    description=(
        "Return the transaction-cost presets used by the lab (spread + fee + "
        "slippage), from a frictionless illustration baseline to a high-fee "
        "venue. The round-trip cost in basis points is the toll that makes "
        "turnover bleed money.\n\n**Simulation disclaimer.** " + HFT_DISCLAIMER
    ),
)
def list_cost_presets() -> list[HftCostModel]:
    """Return every transaction-cost preset as a wire DTO."""
    return [_cost_dto(m) for m in COST_PRESETS.values()]


@router.post(
    "/simulate",
    response_model=HftSimResult,
    summary="Run one short-horizon simulation (gross vs net vs buy-&-hold)",
    description=(
        "Simulate one short-horizon strategy over a deterministic synthetic "
        "intraday path and return three aligned equity curves — net (after "
        "costs), gross (the same decisions with zero costs), and buy-&-hold — "
        "plus realized metrics. The gap between gross and net is the **cost "
        "drag**: the price of turnover, paid in full.\n\n"
        "**Simulation disclaimer.** " + HFT_DISCLAIMER + "\n\n"
        "**Status codes**\n- `200` — the populated `HftSimResult`."
    ),
    responses={200: {"description": "The simulation result with equity curves."}},
)
def simulate(body: HftSimRequest) -> HftSimResult:
    """Run one configuration and return its full result (never raises)."""
    cost = get_cost_model(body.cost_preset)
    res = run_sim(_spec_from(body))
    return _sim_result_dto(res, cost)


@router.post(
    "/sweep",
    response_model=HftSweepResult,
    summary="Sweep trade frequency to find the net-of-cost optimum",
    description=(
        "Hold the strategy fixed and vary only **how often it trades**, running "
        "every setting on the same synthetic path. Returns the turnover curve "
        "(turnover, gross/net return, cost drag, Sharpe, drawdown per setting), "
        "the net-of-cost optimum, the risk-adjusted optimum, the naive "
        "re-decide-every-bar point, the buy-&-hold benchmark, and a plain-English "
        "verdict. The honest result on edge-free synthetic data: past a low "
        "turnover, every extra trade just feeds the spread.\n\n"
        "**Simulation disclaimer.** " + HFT_DISCLAIMER + "\n\n"
        "**Status codes**\n- `200` — the populated `HftSweepResult`."
    ),
    responses={200: {"description": "The turnover sweep with optimum + verdict."}},
)
def sweep(body: HftSweepRequest) -> HftSweepResult:
    """Run the turnover sweep and return the curve + optimum + verdict."""
    res = run_sweep(_spec_from(body.base), body.intervals)
    return _sweep_result_dto(res)
