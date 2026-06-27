"""The turnover sweep — finding the TRUE net-of-cost optimum.

This is the lab's headline experiment and the honest answer to *"should I trade
faster / break my money into more portions?"*. It holds the strategy fixed and
varies only **how often it trades** (the ``rebalance_interval`` knob, from
re-deciding every bar down to rarely), running every setting on the *same*
synthetic price path so the comparison is apples-to-apples.

For each setting it records turnover, the gross (cost-free) return, the net
(after-cost) return, the cost drag, risk-adjusted return and drawdown. Then it
reports three reference points:

    * **naive-fast** — re-decide every bar (the limit of "trade more lively /
      split into more portions"); almost always the *worst* net result;
    * **optimum by net return** — where after-cost return actually peaks;
    * **optimum by net Sharpe** — the best *risk-adjusted* setting.

…and a plain-English verdict comparing them to simply buying and holding. On
synthetic data with no real edge, the lesson is reliably the same: past a low
turnover, every extra trade just feeds the spread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from app.hft.execution import SimSpec, run_sim
from app.hft.intraday import generate_intraday

__all__ = ["SweepPoint", "SweepResult", "default_intervals", "run_sweep"]


def _finite(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


@dataclass(frozen=True)
class SweepPoint:
    """One setting on the turnover curve.

    Attributes:
        interval: The ``rebalance_interval`` (bars between re-decisions).
        label: Human-readable trade cadence (e.g. ``"every bar"``, ``"~6×/day"``).
        turnover: Total traded notional / capital at this setting.
        turnover_per_day: Turnover divided by the number of days.
        trades: Number of trades executed.
        gross_return_pct: Return with zero costs.
        net_return_pct: Return after all costs (the truth).
        cost_drag_pct: ``gross - net`` (the toll paid for this much turnover).
        sharpe_net: Annualised net Sharpe.
        max_drawdown_pct: Worst net drawdown (``≤ 0``).
        vs_buy_hold_pct: Net return minus buy-&-hold return.
    """

    interval: int
    label: str
    turnover: float
    turnover_per_day: float
    trades: int
    gross_return_pct: float
    net_return_pct: float
    cost_drag_pct: float
    sharpe_net: float
    max_drawdown_pct: float
    vs_buy_hold_pct: float


@dataclass
class SweepResult:
    """The full turnover-sweep result.

    Attributes:
        points: The curve, ordered from most-active (interval 1) to least.
        optimum_by_net_return: The point with the highest net return.
        optimum_by_net_sharpe: The point with the highest net Sharpe.
        naive_fast: The interval-1 (re-decide every bar) point.
        buy_hold_return_pct: Buy-&-hold return over the same path.
        verdict: A plain-English summary of what the curve shows.
    """

    points: list[SweepPoint] = field(default_factory=list)
    optimum_by_net_return: SweepPoint | None = None
    optimum_by_net_sharpe: SweepPoint | None = None
    naive_fast: SweepPoint | None = None
    buy_hold_return_pct: float = 0.0
    verdict: str = ""


def default_intervals(bars_per_day: int) -> list[int]:
    """A sensible turnover grid spanning hyperactive → patient.

    Args:
        bars_per_day: Bars per day (sets the upper, "slow" end of the grid).

    Returns:
        A sorted, de-duplicated list of rebalance intervals (in bars).
    """
    bpd = max(2, int(bars_per_day))
    raw = [1, 2, 4, 8, 16, 32, bpd, bpd * 2, bpd * 5]
    out = sorted({max(1, int(i)) for i in raw})
    return out


def _label_for(interval: int, bars_per_day: int) -> str:
    """Describe a rebalance interval as a trade cadence."""
    if interval <= 1:
        return "every bar"
    per_day = bars_per_day / float(interval)
    if per_day >= 1.0:
        return f"~{round(per_day)}×/day"
    per_week = per_day * 5.0
    if per_week >= 1.0:
        return f"~{round(per_week)}×/week"
    return f"every {interval} bars"


def run_sweep(base_spec: SimSpec, intervals: list[int] | None = None) -> SweepResult:
    """Run the turnover sweep over a shared price path (never raises).

    Args:
        base_spec: The strategy to hold fixed; only ``rebalance_interval`` varies.
        intervals: Optional explicit grid; defaults to :func:`default_intervals`.

    Returns:
        A populated :class:`SweepResult` (empty/flat on a degenerate base spec).
    """
    try:
        return _run_sweep_impl(base_spec, intervals)
    except Exception:  # pragma: no cover - defensive
        return SweepResult(verdict="Sweep unavailable for this configuration.")


def _run_sweep_impl(base_spec: SimSpec, intervals: list[int] | None) -> SweepResult:
    """Implementation of :func:`run_sweep`."""
    # One shared synthetic path so every setting trades identical prices.
    series = generate_intraday(
        base_spec.symbol,
        days=base_spec.days,
        bars_per_day=base_spec.bars_per_day,
        annual_drift=base_spec.annual_drift,
        annual_vol=base_spec.annual_vol,
    )
    grid = intervals or default_intervals(base_spec.bars_per_day)

    points: list[SweepPoint] = []
    buy_hold = 0.0
    for interval in grid:
        spec = replace(base_spec, rebalance_interval=max(1, int(interval)))
        res = run_sim(spec, series=series)
        m = res.metrics
        buy_hold = m.buy_hold_return_pct  # identical across settings
        points.append(
            SweepPoint(
                interval=int(spec.rebalance_interval),
                label=_label_for(spec.rebalance_interval, base_spec.bars_per_day),
                turnover=m.turnover,
                turnover_per_day=m.turnover_per_day,
                trades=m.trades,
                gross_return_pct=m.gross_return_pct,
                net_return_pct=m.net_return_pct,
                cost_drag_pct=m.cost_drag_pct,
                sharpe_net=m.sharpe_net,
                max_drawdown_pct=m.max_drawdown_pct,
                vs_buy_hold_pct=m.vs_buy_hold_pct,
            )
        )

    if not points:
        return SweepResult(buy_hold_return_pct=round(buy_hold, 4),
                           verdict="No settings produced a result.")

    opt_ret = max(points, key=lambda p: p.net_return_pct)
    opt_sharpe = max(points, key=lambda p: p.sharpe_net)
    fast = min(points, key=lambda p: p.interval)

    return SweepResult(
        points=points,
        optimum_by_net_return=opt_ret,
        optimum_by_net_sharpe=opt_sharpe,
        naive_fast=fast,
        buy_hold_return_pct=round(_finite(buy_hold), 4),
        verdict=_build_verdict(points, fast, opt_ret, opt_sharpe, buy_hold),
    )


def _build_verdict(
    points: list[SweepPoint],
    fast: SweepPoint,
    opt_ret: SweepPoint,
    opt_sharpe: SweepPoint,
    buy_hold: float,
) -> str:
    """Compose an honest, plain-English summary of the turnover curve."""
    parts: list[str] = []

    parts.append(
        f"Re-deciding every bar churned {fast.turnover:.0f}× turnover into a "
        f"{fast.cost_drag_pct:.1f}% cost drag, leaving {fast.net_return_pct:+.1f}% net."
    )

    if opt_ret.interval == fast.interval:
        parts.append("Even the calmest setting tested couldn't out-earn the fastest here.")
    else:
        parts.append(
            f"Net return peaks at the '{opt_ret.label}' setting "
            f"({opt_ret.net_return_pct:+.1f}% net, {opt_ret.cost_drag_pct:.1f}% drag) — "
            f"trading faster than that only feeds the spread."
        )

    if buy_hold >= opt_ret.net_return_pct - 1e-9:
        parts.append(
            f"And buy-&-hold ({buy_hold:+.1f}%) beats every active setting after costs — "
            f"on data with no real edge, doing less wins."
        )
    else:
        parts.append(
            f"The best active setting edges buy-&-hold ({buy_hold:+.1f}%) — but only at "
            f"low turnover, and only gross of the real-world latency you don't have."
        )

    parts.append(
        "Splitting capital into more portions or trading 'more lively' moves you toward "
        "the left (high-turnover) end of this curve — the losing end."
    )
    return " ".join(parts)
