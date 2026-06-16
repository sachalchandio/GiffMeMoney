"""Vectorized event-free backtesting engine for the GiffMeMoney strategy library.

This module turns a *position series* (a target exposure in ``[-1, 1]`` for each
bar, produced by a timing strategy's ``positions(...)`` function) into a realized
strategy **equity curve** and computes the 14 catalog backtest metrics from it,
alongside a **buy & hold** benchmark on the same asset. It is what makes the
rules-based strategies report *proven* historical performance rather than guessed
forward projections.

Trading model
-------------
Positions are applied with a one-bar lag (you trade at the close of bar ``t`` and
earn bar ``t+1``'s return), and turnover is charged a linear cost::

    asset_return[t]    = close[t] / close[t-1] - 1
    strategy_return[t] = position[t-1] * asset_return[t] - cost * |position[t] - position[t-1]|
    equity[t]          = prod_{s<=t} (1 + strategy_return[s])

with ``cost`` defaulting to 5 bps (``0.0005``) per unit of absolute position
change. The buy & hold benchmark holds a constant unit position (``cost`` only on
the initial entry, which is negligible and excluded so a constant-long strategy
reproduces buy & hold exactly).

The 14 metrics implemented (exactly per the catalog ``backtestMetrics``):
``cagr, total_return, ann_vol, sharpe, sortino, calmar, max_drawdown,
ulcer_index, win_rate, profit_factor, exposure, turnover, cvar95, beta,
information_ratio``.

Everything is vectorized with numpy and numerically defensive: short / empty /
constant / NaN inputs collapse to finite, sane defaults and never raise. Risk and
risk-adjusted helpers are reused from :mod:`app.quant.metrics` and
:mod:`app.quant.returns` so the realized-performance math matches the rest of the
engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.quant import metrics as _metrics
from app.quant.returns import TRADING_DAYS

__all__ = [
    "BacktestMetrics",
    "BacktestResult",
    "backtest_positions",
    "DEFAULT_COST",
]

# Smallest std/denominator we treat as non-zero; below this a series is
# effectively constant and a ratio would diverge, so we collapse to a default.
_EPS: float = 1e-12

# Default one-way turnover cost per unit of |Δposition|: 5 basis points.
DEFAULT_COST: float = 0.0005

# Target number of points in the downsampled equity curve sent over the wire.
_EQUITY_CURVE_POINTS: int = 120

# Hard clamp on reported ratio metrics so a pathological denominator can never
# emit absurd numbers into the DTOs.
_RATIO_CLAMP: float = 1.0e6


def _safe(value: float, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, else ``default``.

    Args:
        value: Candidate number.
        default: Substituted when ``value`` is NaN / +-inf.

    Returns:
        ``float(value)`` if finite, otherwise ``default``.
    """
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _clamp_ratio(value: float) -> float:
    """Clamp a ratio metric to ``[-_RATIO_CLAMP, _RATIO_CLAMP]`` and finitize."""
    f = _safe(value, 0.0)
    return max(-_RATIO_CLAMP, min(_RATIO_CLAMP, f))


@dataclass
class BacktestMetrics:
    """The 14 realized-performance metrics of an equity curve.

    Returns are decimals (e.g. ``0.12`` = 12%); the API/DTO layer converts the
    percentage-natured fields to percent. Drawdown is non-positive. All fields
    are guaranteed finite.

    Attributes:
        cagr: Compound annual growth rate, ``(V_end/V_start)^(252/N) - 1``.
        total_return: Cumulative return over the whole period, ``V_end/V_start - 1``.
        ann_vol: Annualized volatility of daily returns, ``sqrt(252)*std(r)``.
        sharpe: Annualized Sharpe, ``sqrt(252)*mean(r-rf)/std(r-rf)``.
        sortino: Annualized Sortino (downside-deviation denominator).
        calmar: ``CAGR / |MaxDrawdown|``.
        max_drawdown: Largest peak-to-trough decline of equity (<= 0).
        ulcer_index: RMS of percent drawdowns over time (>= 0).
        win_rate: Fraction of bars with a positive strategy return, ``[0, 1]``.
        profit_factor: ``sum(gains) / |sum(losses)|`` (>= 0).
        exposure: Fraction of bars with a non-zero position, ``[0, 1]``.
        turnover: Annualized one-way turnover, ``(1/years)*sum|Δpos|/2``.
        cvar95: Expected shortfall of the worst 5% of daily returns (>= 0, a loss).
        beta: Beta of strategy returns to the benchmark returns.
        information_ratio: Annualized IR vs the benchmark.
    """

    cagr: float = 0.0
    total_return: float = 0.0
    ann_vol: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    ulcer_index: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    exposure: float = 0.0
    turnover: float = 0.0
    cvar95: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0

    @classmethod
    def zeros(cls) -> "BacktestMetrics":
        """Return an all-zero metrics object (a flat, do-nothing strategy)."""
        return cls()


@dataclass
class BacktestResult:
    """Outcome of backtesting one position series against buy & hold.

    Attributes:
        symbol: The asset symbol (upper-cased convention, but not enforced here).
        strategy_id: The strategy id that produced the positions.
        metrics: Realized metrics of the strategy equity curve.
        benchmark: Realized metrics of the buy & hold curve on the same asset.
        equity_curve: Down-sampled (~120 pts) curve of dicts
            ``{"t": int, "strategy": float, "benchmark": float}`` where both
            series start at ``1.0`` (growth-of-1 normalized equity).
        trades: Number of position changes (entries/exits/flips).
        supported: ``True`` when the strategy is per-bar backtestable; ``False``
            for snapshot/fundamental strategies (the result then mirrors buy &
            hold with empty/neutral strategy metrics).
    """

    symbol: str
    strategy_id: str
    metrics: BacktestMetrics
    benchmark: BacktestMetrics
    equity_curve: list[dict] = field(default_factory=list)
    trades: int = 0
    supported: bool = True


def _clean_closes(closes: np.ndarray | list[float]) -> np.ndarray:
    """Coerce a close-price input to a 1-D float array (non-finite -> NaN kept out later).

    Unlike the strict price cleaners elsewhere we keep length so positions stay
    aligned to bars; non-finite / non-positive prices are forward-handled in the
    return computation.

    Args:
        closes: Sequence of close prices.

    Returns:
        A 1-D ``float64`` array (possibly empty).
    """
    return np.asarray(closes, dtype=np.float64).ravel()


def _asset_returns(closes: np.ndarray) -> np.ndarray:
    """Compute bar-over-bar simple returns aligned so ``ret[t]`` pairs with bar ``t``.

    ``ret[0]`` is defined as ``0.0`` (no prior bar) so the returns array has the
    same length as ``closes`` and indexes the same bars as the position array.
    Non-finite / non-positive transitions yield a ``0.0`` return rather than NaN.

    Args:
        closes: Clean close-price array of length ``n``.

    Returns:
        A length-``n`` array of simple returns with ``ret[0] == 0.0``.
    """
    n = closes.size
    rets = np.zeros(n, dtype=np.float64)
    if n < 2:
        return rets
    prev = closes[:-1]
    cur = closes[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        step = cur / prev - 1.0
    valid = np.isfinite(step) & np.isfinite(prev) & (prev > 0.0)
    step = np.where(valid, step, 0.0)
    rets[1:] = step
    return rets


def _normalize_positions(positions: np.ndarray | list[float], n: int) -> np.ndarray:
    """Coerce a raw position series to a clean length-``n`` array in ``[-1, 1]``.

    Non-finite entries are treated as flat (``0.0``); values outside ``[-1, 1]``
    are clipped. If the input is shorter than ``n`` it is left-padded with zeros
    (no position before history starts); if longer it is trailing-truncated.

    Args:
        positions: Target exposure per bar.
        n: Number of bars to align to (length of the close series).

    Returns:
        A length-``n`` ``float64`` array of finite positions in ``[-1, 1]``.
    """
    pos = np.asarray(positions, dtype=np.float64).ravel()
    pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    if pos.size == 0:
        return out
    if pos.size >= n:
        out[:] = pos[-n:]
    else:
        out[-pos.size:] = pos
    return np.clip(out, -1.0, 1.0)


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    """Return the per-bar drawdown of an equity curve (``equity/cummax - 1``, <= 0)."""
    if equity.size == 0:
        return equity
    running_max = np.maximum.accumulate(equity)
    running_max = np.where(running_max > _EPS, running_max, _EPS)
    dd = equity / running_max - 1.0
    return np.minimum(dd, 0.0)


def _compute_metrics(
    strat_rets: np.ndarray,
    equity: np.ndarray,
    positions: np.ndarray,
    bench_rets: np.ndarray,
    rf_daily: float,
    n_days: int,
) -> BacktestMetrics:
    """Compute all 14 metrics from a strategy return / equity / position series.

    Args:
        strat_rets: Per-bar strategy returns (net of costs), length ``n``.
        equity: Growth-of-1 equity curve, length ``n``, ``equity[0] == 1.0``.
        positions: Per-bar position series in ``[-1, 1]``, length ``n``.
        bench_rets: Per-bar benchmark (buy & hold) returns for beta / IR, length ``n``.
        rf_daily: Daily risk-free rate (decimal).
        n_days: Number of return bars used for the annualization exponent.

    Returns:
        A fully populated, finite :class:`BacktestMetrics`.
    """
    m = BacktestMetrics()
    if strat_rets.size == 0 or equity.size == 0:
        return m

    rf = _safe(rf_daily, 0.0)

    # --- Total return & CAGR from the equity curve -------------------------
    v_end = _safe(float(equity[-1]), 1.0)
    v_start = _safe(float(equity[0]), 1.0)
    if v_start <= _EPS:
        v_start = 1.0
    total_return = v_end / v_start - 1.0
    m.total_return = _clamp_ratio(total_return)

    # CAGR = (V_end/V_start)^(252/N) - 1 ; N = number of return bars.
    n_eff = max(1, int(n_days))
    ratio = v_end / v_start
    if ratio > _EPS and math.isfinite(ratio):
        try:
            cagr = ratio ** (TRADING_DAYS / n_eff) - 1.0
        except (OverflowError, ValueError):
            cagr = 0.0
    else:
        # Equity hit zero/negative -> total loss annualized.
        cagr = -1.0
    m.cagr = _clamp_ratio(cagr)

    # --- Volatility, Sharpe, Sortino (reuse metrics.py) --------------------
    m.ann_vol = _safe(_metrics.annual_volatility(strat_rets), 0.0)
    m.sharpe = _clamp_ratio(_metrics.sharpe(strat_rets, rf))
    m.sortino = _clamp_ratio(_metrics.sortino(strat_rets, rf))

    # --- Max drawdown & Calmar (drawdown from equity, CAGR numerator) ------
    dd = _drawdown_series(equity)
    mdd = _safe(float(np.min(dd)), 0.0) if dd.size else 0.0
    m.max_drawdown = min(0.0, mdd)
    denom = abs(m.max_drawdown)
    m.calmar = _clamp_ratio(m.cagr / denom) if denom > _EPS else 0.0

    # --- Ulcer Index: RMS of percent drawdowns -----------------------------
    if dd.size:
        d_pct = dd * 100.0
        ui = math.sqrt(float(np.mean(d_pct * d_pct)))
        m.ulcer_index = _safe(ui, 0.0) if ui >= 0.0 else 0.0

    # --- Win rate over active (non-flat-return) bars -----------------------
    # Count only bars that actually moved capital; a perpetually flat series
    # has no trades and a 0 win rate, which is the desired "did nothing" signal.
    nonzero = strat_rets[np.abs(strat_rets) > _EPS]
    if nonzero.size > 0:
        m.win_rate = float(np.count_nonzero(nonzero > 0.0)) / float(nonzero.size)
    m.win_rate = min(1.0, max(0.0, m.win_rate))

    # --- Profit factor: gross gains / |gross losses| -----------------------
    gains = float(np.sum(strat_rets[strat_rets > 0.0]))
    losses = float(np.sum(strat_rets[strat_rets < 0.0]))  # <= 0
    abs_losses = abs(losses)
    if abs_losses > _EPS:
        m.profit_factor = _clamp_ratio(gains / abs_losses)
    elif gains > _EPS:
        # All wins, no losses -> capped large but finite profit factor.
        m.profit_factor = _RATIO_CLAMP
    else:
        m.profit_factor = 0.0

    # --- Exposure: fraction of bars with a non-zero position ---------------
    m.exposure = float(np.count_nonzero(np.abs(positions) > _EPS)) / float(positions.size)
    m.exposure = min(1.0, max(0.0, m.exposure))

    # --- Turnover: annualized half-sum of |Δposition| ----------------------
    if positions.size >= 2:
        delta = np.abs(np.diff(positions))
        years = max(n_eff / TRADING_DAYS, 1.0 / TRADING_DAYS)
        m.turnover = _clamp_ratio((float(np.sum(delta)) / 2.0) / years)

    # --- CVaR / Expected Shortfall (95%) of daily strategy returns ---------
    m.cvar95 = _cvar95(strat_rets)

    # --- Beta & Information Ratio vs benchmark (reuse metrics.py) ----------
    m.beta = _safe(_metrics.beta(strat_rets, bench_rets), 0.0)
    m.information_ratio = _clamp_ratio(_metrics.information_ratio(strat_rets, bench_rets))

    return m


def _cvar95(returns: np.ndarray) -> float:
    """Expected shortfall at 95%: mean loss over the worst 5% of daily returns.

    Formula:
        CVaR_95 = -mean( r_t | r_t <= Quantile_{0.05}(r) )

    Returned as a non-negative magnitude (a loss). A series with no observations
    or no downside in the tail returns ``0.0``.

    Args:
        returns: Per-bar strategy returns.

    Returns:
        The 95% expected shortfall as a non-negative decimal.
    """
    r = returns[np.isfinite(returns)]
    if r.size == 0:
        return 0.0
    q05 = float(np.quantile(r, 0.05))
    tail = r[r <= q05]
    if tail.size == 0:
        tail = np.array([q05], dtype=np.float64)
    es = -float(np.mean(tail))
    es = _safe(es, 0.0)
    return max(0.0, es)


def _downsample_equity(
    strat_equity: np.ndarray,
    bench_equity: np.ndarray,
    points: int = _EQUITY_CURVE_POINTS,
) -> list[dict]:
    """Down-sample two aligned equity curves to ~``points`` ``{t, strategy, benchmark}`` dicts.

    The first and last bars are always kept; the interior is sampled on an even
    index grid. ``t`` is the original bar index so the frontend can align it to
    the candle series.

    Args:
        strat_equity: Strategy growth-of-1 equity curve.
        bench_equity: Benchmark growth-of-1 equity curve (same length).
        points: Target number of output points.

    Returns:
        A list of dicts (length <= ``points``) with finite floats.
    """
    n = int(min(strat_equity.size, bench_equity.size))
    if n == 0:
        return []
    if n <= points:
        idx = np.arange(n)
    else:
        idx = np.unique(
            np.linspace(0, n - 1, num=points, dtype=np.int64)
        )
    out: list[dict] = []
    for i in idx:
        i = int(i)
        out.append(
            {
                "t": i,
                "strategy": _safe(float(strat_equity[i]), 1.0),
                "benchmark": _safe(float(bench_equity[i]), 1.0),
            }
        )
    return out


def backtest_positions(
    closes: np.ndarray | list[float],
    positions: np.ndarray | list[float],
    rf_daily: float = 0.0,
    *,
    symbol: str = "",
    strategy_id: str = "",
    cost: float = DEFAULT_COST,
    supported: bool = True,
    benchmark: str = "bh",
    highs: np.ndarray | list[float] | None = None,
    lows: np.ndarray | list[float] | None = None,
) -> BacktestResult:
    """Backtest a position series against buy & hold and compute the 14 metrics.

    The strategy daily return is the one-bar-lagged position times the asset's
    return, minus a linear turnover cost::

        r_strat[t] = position[t-1] * asset_return[t] - cost * |position[t] - position[t-1]|

    The benchmark holds a constant unit long position (no turnover cost), so a
    **constant-long strategy reproduces buy & hold exactly** (the verification
    invariant). Both equity curves are growth-of-1 (start at ``1.0``).

    Args:
        closes: Close-price series of length ``n``.
        positions: Target exposure per bar in ``[-1, 1]`` (or ``{0, 1}``), aligned
            to ``closes``. Shorter inputs are left-padded with zeros, longer
            inputs trailing-truncated.
        rf_daily: Daily risk-free rate (decimal) used for Sharpe / Sortino.
        symbol: Asset symbol carried onto the result.
        strategy_id: Strategy id carried onto the result.
        cost: One-way turnover cost per unit of ``|Δposition|`` (default 5 bps).
        supported: ``False`` for snapshot/fundamental strategies — the result
            then reports buy & hold for both legs (no per-bar timing) and the
            flag is propagated so the API can label it.
        benchmark: Reserved for future benchmark choices; only ``'bh'`` (buy &
            hold the same asset) is implemented.
        highs: Optional high series (unused here; accepted for API symmetry with
            indicator-driven position functions).
        lows: Optional low series (unused here; accepted for API symmetry).

    Returns:
        A :class:`BacktestResult` with strategy + benchmark :class:`BacktestMetrics`,
        a ~120-point down-sampled equity curve, the trade count, and the
        ``supported`` flag. Never raises; degenerate inputs yield zeroed metrics
        and an empty/short curve.
    """
    c = _clean_closes(closes)
    n = c.size

    # Benchmark = buy & hold: constant unit long, no turnover cost.
    asset_rets = _asset_returns(c)
    bench_pos = np.ones(n, dtype=np.float64)
    # Lagged exposure: hold from bar 1 onward (entry at bar 0 earns nothing).
    bench_lagged = np.zeros(n, dtype=np.float64)
    if n >= 1:
        bench_lagged[1:] = bench_pos[:-1]
    bench_rets = bench_lagged * asset_rets
    bench_equity = np.cumprod(1.0 + bench_rets) if n else np.array([], dtype=np.float64)
    bench_metrics = _compute_metrics(
        strat_rets=bench_rets,
        equity=bench_equity,
        positions=bench_pos,
        bench_rets=bench_rets,
        rf_daily=rf_daily,
        n_days=max(1, n - 1),
    )

    if not supported or n < 2:
        # Non-backtestable strategy (or too little data): mirror buy & hold for the
        # benchmark leg and report neutral/zero strategy metrics. Curve uses the
        # buy & hold series for both legs so the chart still renders.
        curve = _downsample_equity(
            bench_equity if n else np.array([1.0]),
            bench_equity if n else np.array([1.0]),
        )
        return BacktestResult(
            symbol=symbol,
            strategy_id=strategy_id,
            metrics=BacktestMetrics.zeros() if not supported else bench_metrics,
            benchmark=bench_metrics,
            equity_curve=curve,
            trades=0,
            supported=supported,
        )

    pos = _normalize_positions(positions, n)
    cost_rate = max(0.0, _safe(cost, DEFAULT_COST))

    # One-bar-lagged exposure: position decided at close[t] earns ret[t+1].
    lagged = np.zeros(n, dtype=np.float64)
    lagged[1:] = pos[:-1]

    # Turnover cost charged on the bar the position changes (including initial entry).
    delta = np.zeros(n, dtype=np.float64)
    delta[0] = abs(pos[0])
    delta[1:] = np.abs(np.diff(pos))
    turnover_cost = cost_rate * delta

    strat_rets = lagged * asset_rets - turnover_cost
    strat_rets = np.nan_to_num(strat_rets, nan=0.0, posinf=0.0, neginf=0.0)
    strat_equity = np.cumprod(1.0 + strat_rets)
    # Guard equity against going non-positive from a pathological single-bar loss.
    strat_equity = np.where(np.isfinite(strat_equity), strat_equity, _EPS)
    strat_equity = np.maximum(strat_equity, _EPS)

    strat_metrics = _compute_metrics(
        strat_rets=strat_rets,
        equity=strat_equity,
        positions=pos,
        bench_rets=bench_rets,
        rf_daily=rf_daily,
        n_days=max(1, n - 1),
    )

    trades = int(np.count_nonzero(delta[1:] > _EPS))
    curve = _downsample_equity(strat_equity, bench_equity)

    return BacktestResult(
        symbol=symbol,
        strategy_id=strategy_id,
        metrics=strat_metrics,
        benchmark=bench_metrics,
        equity_curve=curve,
        trades=trades,
        supported=True,
    )
