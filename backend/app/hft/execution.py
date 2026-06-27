"""The execution simulator — the honest core of the lab.

Given an intraday price path, a signal, a cost model, position-sizing rules and
risk gates, :func:`run_sim` walks the path bar-by-bar and produces three equity
curves that tell the whole story:

    * **net**   — the strategy actually traded, *with* spreads/fees/slippage;
    * **gross** — the identical decisions with *zero* costs (the counterfactual);
    * **buy & hold** — buy once at the start and sit still.

The gap between gross and net is the **cost drag** — the price of turnover, paid
in full. The gap between net and buy & hold is whether all that activity was
worth it. Because gross replays the *same* actual exposures as net, the drag is
measured exactly, not estimated.

Decision flow at each rebalance bar (every ``rebalance_interval`` bars):

    1. raw exposure from the point-in-time signal (no look-ahead);
    2. **volatility targeting** — scale so the position's annualised vol ≈
       ``target_vol`` (size down in turbulence, up when calm);
    3. **cap** at ``max_exposure`` (a Kelly-style leverage ceiling, ≤ 1 = no
       leverage);
    4. **risk gates** (always on, every bar): a per-position trailing stop and
       take-profit, plus an account drawdown circuit-breaker that flattens the
       book and sits out a cooldown.

Then trade the delta to the target on the net book (charging cost) and on the
gross book (free). Everything is finite and defensive: a degenerate spec yields a
flat, finite result rather than raising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.hft.costs import CostModel, get_cost_model
from app.hft.intraday import IntradaySeries, generate_intraday
from app.hft.signals import raw_exposure

__all__ = ["SimSpec", "SimMetrics", "SimResult", "run_sim"]

_EPS: float = 1e-12


def _finite(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


@dataclass(frozen=True)
class SimSpec:
    """A single simulation configuration.

    Attributes:
        symbol: Instrument name (drives the deterministic synthetic path).
        amount: Starting capital in dollars (``> 0``).
        days: Trading days to simulate.
        bars_per_day: Intraday bars per day (the bar is the fastest the strategy
            can act — honestly minutes, never microseconds).
        signal: Signal id (``"meanrev"`` / ``"momentum"`` / ``"buyhold"``).
        lookback: Trailing window (bars) for the signal.
        rebalance_interval: Bars between target re-evaluations — the **turnover
            knob**. 1 = re-decide every bar (most active); larger = calmer.
        deadband: Minimum change in target exposure required to actually trade
            (suppresses churn on tiny adjustments).
        target_vol: Annualised volatility target for position sizing (``0`` =
            use the raw signal magnitude unscaled).
        max_exposure: Hard cap on |exposure| (``≤ 1`` means no leverage).
        allow_short: Whether the signal may take short (negative) exposure.
        stop_loss_pct: Per-position trailing stop from the favorable extreme
            (percent; ``0`` disables).
        take_profit_pct: Per-position take-profit from entry (percent; ``0``
            disables).
        max_drawdown_pct: Account drawdown circuit-breaker (percent; ``0``
            disables).
        cooldown_bars: Bars to stay flat after a stop / breaker fires.
        cost_preset: Cost-model preset id (see :data:`app.hft.costs.COST_PRESETS`).
        bar_dollar_volume: Liquidity per bar used for the slippage / participation
            term. Large by default, so a small account has ~zero market impact.
        annual_drift: Optional drift override for the synthetic path.
        annual_vol: Optional vol override for the synthetic path.
    """

    symbol: str = "SYNTH"
    amount: float = 20.0
    days: int = 30
    bars_per_day: int = 78
    signal: str = "meanrev"
    lookback: int = 20
    rebalance_interval: int = 1
    deadband: float = 0.05
    target_vol: float = 0.25
    max_exposure: float = 1.0
    allow_short: bool = False
    stop_loss_pct: float = 3.0
    take_profit_pct: float = 0.0
    max_drawdown_pct: float = 15.0
    cooldown_bars: int = 5
    cost_preset: str = "retail-crypto"
    bar_dollar_volume: float = 5_000_000.0
    annual_drift: float | None = None
    annual_vol: float | None = None


@dataclass(frozen=True)
class SimMetrics:
    """Realized metrics for one simulation (all finite).

    Attributes:
        gross_return_pct: Return with zero costs (the counterfactual).
        net_return_pct: Return after all costs (the truth).
        cost_drag_pct: ``gross_return_pct - net_return_pct`` — the toll paid.
        buy_hold_return_pct: Buy-once-and-hold return (one trade).
        vs_buy_hold_pct: ``net_return_pct - buy_hold_return_pct`` (signed).
        turnover: Total traded notional / starting capital (sum of |Δ|).
        turnover_per_day: ``turnover / days``.
        trades: Number of bars on which a trade occurred.
        time_in_market_pct: Fraction of bars holding a non-zero position.
        sharpe_net: Annualised Sharpe of net per-bar returns.
        sharpe_gross: Annualised Sharpe of gross per-bar returns.
        max_drawdown_pct: Worst net peak-to-trough drawdown (``≤ 0``).
        hit_rate_pct: Fraction of in-market bars with a positive position P&L.
        final_net_value: Final net account value.
    """

    gross_return_pct: float
    net_return_pct: float
    cost_drag_pct: float
    buy_hold_return_pct: float
    vs_buy_hold_pct: float
    turnover: float
    turnover_per_day: float
    trades: int
    time_in_market_pct: float
    sharpe_net: float
    sharpe_gross: float
    max_drawdown_pct: float
    hit_rate_pct: float
    final_net_value: float


@dataclass
class SimResult:
    """The full result of one simulation.

    Attributes:
        spec: The :class:`SimSpec` that was run.
        metrics: The realized :class:`SimMetrics`.
        net_curve: Net account value at each (downsampled) bar.
        gross_curve: Gross (cost-free) account value, aligned to ``net_curve``.
        buy_hold_curve: Buy-&-hold value, aligned to ``net_curve``.
        exposure_curve: Actual exposure held, aligned to ``net_curve``.
        bars: Number of return steps simulated.
        bars_per_year: Bars in a trading year (for annualisation).
    """

    spec: SimSpec
    metrics: SimMetrics
    net_curve: list[float] = field(default_factory=list)
    gross_curve: list[float] = field(default_factory=list)
    buy_hold_curve: list[float] = field(default_factory=list)
    exposure_curve: list[float] = field(default_factory=list)
    bars: int = 0
    bars_per_year: int = 0


def _annualised_sharpe(values: np.ndarray, periods_per_year: int) -> float:
    """Annualised Sharpe of the per-step returns of a value series.

    Args:
        values: A strictly-positive account-value series.
        periods_per_year: Steps per year (for sqrt-time scaling).

    Returns:
        A finite Sharpe ratio (0 on degenerate input).
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 3:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        r = v[1:] / v[:-1] - 1.0
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    mu = float(np.mean(r))
    sd = float(np.std(r))
    if sd <= _EPS:
        return 0.0
    return float(mu / sd * math.sqrt(max(1, periods_per_year)))


def _max_drawdown(values: np.ndarray) -> float:
    """Worst peak-to-trough drawdown of a value series (``≤ 0``)."""
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size < 2:
        return 0.0
    peak = np.maximum.accumulate(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = v / np.where(peak > _EPS, peak, _EPS) - 1.0
    dd = np.nan_to_num(dd, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.min(dd))


def _downsample(values: list[float], points: int = 240) -> list[float]:
    """Down-sample a series to ~``points`` evenly-spaced values (keeps ends)."""
    n = len(values)
    if n <= points:
        return [round(float(x), 6) for x in values]
    idx = np.unique(np.linspace(0, n - 1, num=points, dtype=np.int64))
    return [round(float(values[int(i)]), 6) for i in idx]


def _target_exposure(spec: SimSpec, prices: np.ndarray, t: int, bars_per_year: int) -> float:
    """Signal → vol-targeted, capped target exposure (point-in-time)."""
    raw = raw_exposure(
        spec.signal,
        prices,
        t,
        lookback=spec.lookback,
        allow_short=spec.allow_short,
    )
    if abs(raw) <= _EPS:
        return 0.0

    scale = 1.0
    tv = _finite(spec.target_vol)
    if tv > 0.0:
        lb = max(5, int(spec.lookback))
        if t >= lb:
            seg = prices[t - lb + 1 : t + 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                steps = seg[1:] / seg[:-1] - 1.0
            steps = np.nan_to_num(steps, nan=0.0, posinf=0.0, neginf=0.0)
            realized = float(np.std(steps)) * math.sqrt(max(1, bars_per_year))
            if realized > _EPS:
                scale = tv / realized
    cap = _finite(spec.max_exposure, 1.0)
    cap = cap if cap > 0.0 else 1.0
    expo = raw * scale
    return float(max(-cap, min(cap, expo)))


def run_sim(spec: SimSpec, series: IntradaySeries | None = None) -> SimResult:
    """Run one simulation and return its full :class:`SimResult` (never raises).

    Args:
        spec: The :class:`SimSpec` to simulate.
        series: An optional pre-generated :class:`IntradaySeries` (so a sweep can
            reuse one path across configs). When ``None`` a deterministic path is
            generated from the spec.

    Returns:
        A populated :class:`SimResult`. A degenerate spec degrades to a flat,
        finite result.
    """
    try:
        return _run_sim_impl(spec, series)
    except Exception:  # pragma: no cover - defensive (run never raises)
        amount = max(0.0, _finite(spec.amount))
        flat = SimMetrics(
            gross_return_pct=0.0,
            net_return_pct=0.0,
            cost_drag_pct=0.0,
            buy_hold_return_pct=0.0,
            vs_buy_hold_pct=0.0,
            turnover=0.0,
            turnover_per_day=0.0,
            trades=0,
            time_in_market_pct=0.0,
            sharpe_net=0.0,
            sharpe_gross=0.0,
            max_drawdown_pct=0.0,
            hit_rate_pct=0.0,
            final_net_value=round(amount, 2),
        )
        return SimResult(spec=spec, metrics=flat)


def _run_sim_impl(spec: SimSpec, series: IntradaySeries | None) -> SimResult:
    """Implementation of :func:`run_sim` (wrapped for defensiveness)."""
    amount = _finite(spec.amount)
    if amount <= 0.0:
        return run_sim_flat(spec)

    if series is None:
        series = generate_intraday(
            spec.symbol,
            days=spec.days,
            bars_per_day=spec.bars_per_day,
            annual_drift=spec.annual_drift,
            annual_vol=spec.annual_vol,
        )
    prices = np.asarray(series.prices, dtype=np.float64).ravel()
    n = prices.size
    if n < 3:
        return run_sim_flat(spec)
    bpy = series.bars_per_year
    days = max(1, series.days)

    cost: CostModel = get_cost_model(spec.cost_preset)
    bar_dv = max(1.0, _finite(spec.bar_dollar_volume, 5_000_000.0))
    interval = max(1, int(spec.rebalance_interval))
    deadband = max(0.0, _finite(spec.deadband))
    stop = max(0.0, _finite(spec.stop_loss_pct)) / 100.0
    take = max(0.0, _finite(spec.take_profit_pct)) / 100.0
    max_dd = max(0.0, _finite(spec.max_drawdown_pct)) / 100.0
    cooldown_bars = max(0, int(spec.cooldown_bars))

    # --- books -------------------------------------------------------------
    net_cash, net_units = amount, 0.0
    gross_cash, gross_units = amount, 0.0
    # Buy & hold: enter fully at bar 0, paying one entry cost on the net side.
    bh_entry_cost = cost.cost_of(amount, amount / bar_dv)
    bh_units = (amount - bh_entry_cost) / prices[0] if prices[0] > _EPS else 0.0

    target_expo = 0.0          # current target exposure (held between rebalances)
    pos_dir = 0                # -1/0/+1 sign of the current net position
    entry_px = 0.0             # vwap of the current position
    fav_px = 0.0               # favorable extreme since entry (for trailing stop)
    cooldown = 0               # bars remaining flat after a stop/breaker

    net_values = np.empty(n, dtype=np.float64)
    gross_values = np.empty(n, dtype=np.float64)
    bh_values = np.empty(n, dtype=np.float64)
    exposures = np.empty(n, dtype=np.float64)

    total_traded = 0.0
    trades = 0
    net_peak = amount
    in_market_bars = 0
    hit_bars = 0

    prev_net_value = amount

    for t in range(n):
        px = float(prices[t])
        net_value = net_cash + net_units * px
        gross_value = gross_cash + gross_units * px
        bh_value = bh_units * px

        net_peak = max(net_peak, net_value)
        dd = (net_value / net_peak - 1.0) if net_peak > _EPS else 0.0

        # --- risk gates (evaluated every bar on the live net position) ------
        gate_flat = False
        if pos_dir != 0 and px > 0.0:
            # Update favorable extreme.
            if pos_dir > 0:
                fav_px = max(fav_px, px)
                trail = (px / fav_px - 1.0) if fav_px > _EPS else 0.0  # ≤ 0
                gain = (px / entry_px - 1.0) if entry_px > _EPS else 0.0
                if stop > 0.0 and trail <= -stop:
                    gate_flat = True
                if take > 0.0 and gain >= take:
                    gate_flat = True
            else:
                fav_px = min(fav_px, px) if fav_px > 0.0 else px
                trail = (px / fav_px - 1.0) if fav_px > _EPS else 0.0  # ≥ 0 is adverse
                gain = (entry_px / px - 1.0) if px > _EPS else 0.0
                if stop > 0.0 and trail >= stop:
                    gate_flat = True
                if take > 0.0 and gain >= take:
                    gate_flat = True
        if max_dd > 0.0 and dd <= -max_dd:
            gate_flat = True

        if gate_flat:
            cooldown = cooldown_bars
            target_expo = 0.0

        # --- decide target exposure on the rebalance grid -------------------
        is_rebalance = (t % interval == 0)
        if cooldown > 0:
            target_expo = 0.0
            cooldown -= 1
        elif is_rebalance and not gate_flat:
            target_expo = _target_exposure(spec, prices, t, bpy)

        # --- trade the net book toward the target (charge cost) -------------
        if px > _EPS:
            cur_notional = net_units * px
            target_notional = target_expo * net_value
            delta = target_notional - cur_notional
            if abs(delta) >= deadband * max(net_value, _EPS) and abs(delta) > 1e-9:
                participation = abs(delta) / bar_dv
                c = cost.cost_of(delta, participation)
                net_units += delta / px
                net_cash -= delta + c
                total_traded += abs(delta)
                trades += 1
                # Update position bookkeeping (entry vwap / direction / extreme).
                new_notional = net_units * px
                new_dir = 1 if new_notional > _EPS else (-1 if new_notional < -_EPS else 0)
                if new_dir == 0:
                    pos_dir, entry_px, fav_px = 0, 0.0, 0.0
                elif new_dir != pos_dir or pos_dir == 0:
                    pos_dir, entry_px, fav_px = new_dir, px, px
                # else: same direction, keep entry vwap-ish (simplified — entry
                # holds; trailing extreme already tracked).

            # --- gross book: replay the SAME target exposure, zero cost ------
            g_cur = gross_units * px
            g_target = target_expo * gross_value
            g_delta = g_target - g_cur
            if abs(g_delta) >= deadband * max(gross_value, _EPS) and abs(g_delta) > 1e-9:
                gross_units += g_delta / px
                gross_cash -= g_delta

        # --- record -----------------------------------------------------------
        net_value = net_cash + net_units * px
        gross_value = gross_cash + gross_units * px
        bh_value = bh_units * px
        net_values[t] = net_value
        gross_values[t] = gross_value
        bh_values[t] = bh_value
        exposures[t] = (net_units * px / net_value) if net_value > _EPS else 0.0

        # In-market + hit-rate bookkeeping (based on the position held INTO t).
        if abs(exposures[t]) > 1e-4:
            in_market_bars += 1
        if t > 0 and abs(exposures[t - 1] if t > 0 else 0.0) > 1e-4:
            if net_value > prev_net_value:
                hit_bars += 1
        prev_net_value = net_value

    # --- metrics -----------------------------------------------------------
    final_net = float(net_values[-1])
    final_gross = float(gross_values[-1])
    final_bh = float(bh_values[-1])

    gross_ret = (final_gross / amount - 1.0) if amount > _EPS else 0.0
    net_ret = (final_net / amount - 1.0) if amount > _EPS else 0.0
    bh_ret = (final_bh / amount - 1.0) if amount > _EPS else 0.0
    turnover = total_traded / amount if amount > _EPS else 0.0

    in_market_denom = max(1, in_market_bars)
    metrics = SimMetrics(
        gross_return_pct=round(_finite(gross_ret) * 100.0, 4),
        net_return_pct=round(_finite(net_ret) * 100.0, 4),
        cost_drag_pct=round(_finite(gross_ret - net_ret) * 100.0, 4),
        buy_hold_return_pct=round(_finite(bh_ret) * 100.0, 4),
        vs_buy_hold_pct=round(_finite(net_ret - bh_ret) * 100.0, 4),
        turnover=round(_finite(turnover), 4),
        turnover_per_day=round(_finite(turnover / days), 4),
        trades=int(trades),
        time_in_market_pct=round(100.0 * in_market_bars / max(1, n), 2),
        sharpe_net=round(_annualised_sharpe(net_values, bpy), 4),
        sharpe_gross=round(_annualised_sharpe(gross_values, bpy), 4),
        max_drawdown_pct=round(_max_drawdown(net_values) * 100.0, 4),
        hit_rate_pct=round(100.0 * hit_bars / in_market_denom, 2),
        final_net_value=round(_finite(final_net), 2),
    )

    return SimResult(
        spec=spec,
        metrics=metrics,
        net_curve=_downsample(list(net_values)),
        gross_curve=_downsample(list(gross_values)),
        buy_hold_curve=_downsample(list(bh_values)),
        exposure_curve=_downsample(list(exposures)),
        bars=int(n - 1),
        bars_per_year=int(bpy),
    )


def run_sim_flat(spec: SimSpec) -> SimResult:
    """Return a flat, finite result for a degenerate spec (no capital / no path)."""
    amount = max(0.0, _finite(spec.amount))
    metrics = SimMetrics(
        gross_return_pct=0.0,
        net_return_pct=0.0,
        cost_drag_pct=0.0,
        buy_hold_return_pct=0.0,
        vs_buy_hold_pct=0.0,
        turnover=0.0,
        turnover_per_day=0.0,
        trades=0,
        time_in_market_pct=0.0,
        sharpe_net=0.0,
        sharpe_gross=0.0,
        max_drawdown_pct=0.0,
        hit_rate_pct=0.0,
        final_net_value=round(amount, 2),
    )
    return SimResult(
        spec=spec,
        metrics=metrics,
        net_curve=[round(amount, 6)],
        gross_curve=[round(amount, 6)],
        buy_hold_curve=[round(amount, 6)],
        exposure_curve=[0.0],
        bars=0,
        bars_per_year=0,
    )
