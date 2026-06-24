"""The auto-trader engine: a simulated, paper-traded momentum/bandit bot.

HONESTY / SAFETY (this is a finance tool). :class:`AutoTraderEngine` runs a
**SIMULATION on synthetic data**: it paper-trades a starting cash balance over a
deterministic historical window from :mod:`app.market.simulator`. No real money
moves, no live broker is ever contacted, and every result carries the mandatory
:data:`~app.schemas.BOT_DISCLAIMER`. Rotation is **momentum / bandit** style
(allocate MORE to recent winners, LESS to losers) and is hard-capped; the engine
**never martingales** — it never increases a losing sleeve's weight to recover.
Nothing here implies guaranteed profit.

The backtest is deliberately efficient (anti-stall):

    * A small candidate set is chosen ONCE up front by ranking the universe with
      the (cached) :class:`~app.strategies.engine.AnalysisEngine` composite, so
      the heavy quant pipeline runs at most once per symbol and is reused.
    * Daily mark-to-market is fully vectorized over a pre-aligned close matrix.
    * Rebalances happen on a ~21-trading-day (monthly) grid, not daily.
    * Regime detection and the mean-variance optimizer run on the small candidate
      set only, a handful of times.

At each rebalance the engine:

    1. **Market analysis** — classifies the regime via
       :func:`app.quant.projection.detect_regime` on a synthetic index and scores
       the candidates by the engine's composite (already cached).
    2. **Selection** — keeps the top ``max_names`` candidates by composite.
    3. **Base weights** — per the mode objective via :mod:`app.quant.portfolio`
       (min-variance / max-Sharpe), a momentum rule, or risk-parity.
    4. **Rotation** — tilts the base weights toward sleeves with strong trailing
       realized performance using a softmax (winners up, losers DOWN; capped;
       never martingale).
    5. **Risk** — exits a sleeve down more than ``stop_loss_pct`` from entry, and
       raises cash to a defensive floor when portfolio drawdown exceeds
       ``max_drawdown_pct``.

Mark-to-market is recorded daily; trades and per-sleeve realized P&L are logged.
The benchmark is an equal-weight buy & hold of the same candidate set. Every
public method is fully defensive: a degenerate config can never raise.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from app.bot.attribution import SleeveStat, build_attribution
from app.bot.policies import ModePolicy, RotationPolicy, get_mode, get_policy
from app.market.provider import MarketDataProvider, get_provider
from app.quant import metrics as _metrics
from app.quant import portfolio as _portfolio
from app.quant import projection
from app.quant.returns import TRADING_DAYS
from app.schemas import (
    BOT_DISCLAIMER,
    BotConfig,
    BotEquityPoint,
    BotMetrics,
    BotRunResult,
    BotTrade,
)
from app.strategies.engine import AnalysisEngine

__all__ = ["AutoTraderEngine"]

# Trailing window of daily closes the backtest runs over (~3.6 years). Kept
# short enough to stay fast, long enough for several regimes and rebalances.
_BACKTEST_DAYS: int = 920

# How many extra candidates beyond ``max_names`` to keep in the working set so
# the per-rebalance re-selection has room to rotate. Capped tightly for speed.
_CANDIDATE_SLACK: int = 4

# Hard ceiling on the working candidate set regardless of the above (anti-stall).
_MAX_CANDIDATES: int = 12

# Linear one-way trading cost charged on |Δ dollar weight| at each rebalance
# (5 bps), mirroring the strategy backtester's convention.
_COST_RATE: float = 0.0005

# Defensive cash floor the drawdown circuit-breaker raises to (fraction held in
# cash once the portfolio drawdown breaches ``max_drawdown_pct``).
_DRAWDOWN_CASH_FLOOR: float = 0.5

# Target number of equity-curve points sent over the wire (down-sampled).
_EQUITY_POINTS: int = 180

# Smallest denominator treated as non-zero.
_EPS: float = 1e-12


def _finite(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


@dataclass
class _Candidate:
    """A working candidate sleeve: its symbol, composite score, and close series.

    Attributes:
        symbol: The asset ticker (upper-cased).
        score: The engine composite score in ``[-100, 100]`` (selection signal).
        closes: The aligned trailing close series for the backtest window.
    """

    symbol: str
    score: float
    closes: np.ndarray


class AutoTraderEngine:
    """Run a simulated, paper-traded auto-trader backtest over synthetic data.

    Args:
        provider: A :class:`~app.market.provider.MarketDataProvider` (defaults to
            the process-wide singleton).
        analysis_engine: The shared :class:`~app.strategies.engine.AnalysisEngine`
            whose cached composite scores drive candidate selection (defaults to
            a fresh engine bound to the same provider).
    """

    def __init__(
        self,
        provider: MarketDataProvider | None = None,
        analysis_engine: AnalysisEngine | None = None,
    ) -> None:
        """Bind the engine to a provider + analysis engine."""
        self._provider: MarketDataProvider = provider or get_provider()
        self._engine: AnalysisEngine = analysis_engine or AnalysisEngine(self._provider)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backtest(self, config: BotConfig) -> BotRunResult:
        """Backtest one :class:`~app.schemas.BotConfig` and return the full result.

        Runs the simulated paper-trader over the trailing backtest window and
        produces the bot-vs-benchmark equity curve, the trade blotter, per-sleeve
        attribution, realized metrics, and the regime timeline. Never raises: a
        degenerate config (no candidates, no history, …) degrades to a flat,
        finite result carrying the disclaimer.

        Args:
            config: The bot configuration to simulate.

        Returns:
            A populated :class:`~app.schemas.BotRunResult`.
        """
        policy = get_policy(config.mode)
        amount = _finite(config.amount, 0.0)
        if amount <= 0.0:
            amount = 0.0

        candidates = self._select_candidates(config, policy)
        if not candidates or amount <= 0.0:
            return self._empty_result(config, policy)

        try:
            return self._run(config, policy, candidates, amount)
        except Exception:  # pragma: no cover - defensive (run never raises)
            return self._empty_result(config, policy)

    # ------------------------------------------------------------------
    # Candidate selection (runs the cached analysis engine at most once/symbol)
    # ------------------------------------------------------------------

    def _select_candidates(
        self, config: BotConfig, policy: ModePolicy
    ) -> list[_Candidate]:
        """Pick a small, class-filtered candidate set ranked by composite score.

        Scores the (optionally class-filtered) universe with the cached analysis
        engine, keeps the strongest ``max_names + slack`` (capped), and pulls each
        one's aligned trailing close series for the backtest window.

        Args:
            config: The bot config (asset-class filter).
            policy: The mode policy (``max_names``).

        Returns:
            A list of :class:`_Candidate` (possibly empty), best score first, all
            sharing the same aligned close length.
        """
        classes = (
            {str(c).lower() for c in config.asset_classes}
            if config.asset_classes
            else None
        )
        try:
            assets = self._provider.list_assets()
        except Exception:  # pragma: no cover - defensive
            assets = []

        scored: list[tuple[str, float]] = []
        for a in assets:
            try:
                if classes is not None and str(a.asset_class).lower() not in classes:
                    continue
                analysis = self._engine.analyze(a.symbol)
                scored.append(
                    (str(a.symbol).upper(), _finite(analysis.composite_score))
                )
            except Exception:
                continue

        if not scored:
            return []

        scored.sort(key=lambda kv: kv[1], reverse=True)
        keep = min(
            _MAX_CANDIDATES, int(policy.mode.max_names) + _CANDIDATE_SLACK
        )
        top = scored[: max(1, keep)]

        # Pull + trailing-align the close series for the working set.
        series: dict[str, np.ndarray] = {}
        min_len = None
        for sym, _score in top:
            try:
                closes = np.asarray(
                    self._provider.history(sym, days=_BACKTEST_DAYS), dtype=np.float64
                ).ravel()
            except Exception:
                continue
            closes = np.nan_to_num(closes, nan=0.0, posinf=0.0, neginf=0.0)
            if closes.size < 2 or not np.any(closes > 0.0):
                continue
            series[sym] = closes
            min_len = closes.size if min_len is None else min(min_len, closes.size)

        if not series or not min_len or min_len < 2:
            return []

        out: list[_Candidate] = []
        score_by_sym = dict(top)
        for sym, closes in series.items():
            aligned = closes[-min_len:]
            # Forward-fill any non-positive prices so mark-to-market is finite.
            aligned = self._sanitize_prices(aligned)
            out.append(
                _Candidate(symbol=sym, score=score_by_sym.get(sym, 0.0), closes=aligned)
            )
        # Keep best-score-first ordering.
        out.sort(key=lambda c: c.score, reverse=True)
        return out

    @staticmethod
    def _sanitize_prices(closes: np.ndarray) -> np.ndarray:
        """Forward-fill non-positive / non-finite prices to a positive series.

        Args:
            closes: A raw close series.

        Returns:
            A strictly-positive, finite close series of the same length.
        """
        c = np.array(closes, dtype=np.float64).ravel()
        last = 0.0
        for i in range(c.size):
            v = c[i]
            if math.isfinite(v) and v > 0.0:
                last = v
            else:
                c[i] = last if last > 0.0 else 1.0
        if last <= 0.0:
            c[:] = 1.0
        # Backfill the leading zeros (before the first valid price) with the first
        # valid value so the series starts positive.
        first_valid = next((x for x in c if x > 0.0), 1.0)
        c[c <= 0.0] = first_valid
        return c

    # ------------------------------------------------------------------
    # The simulation
    # ------------------------------------------------------------------

    def _run(
        self,
        config: BotConfig,
        policy: ModePolicy,
        candidates: list[_Candidate],
        amount: float,
    ) -> BotRunResult:
        """Walk the backtest forward, rebalancing monthly, marking daily.

        Args:
            config: The bot config (risk params).
            policy: The mode policy (objective + rotation).
            candidates: The aligned candidate set.
            amount: Starting paper capital (``> 0``).

        Returns:
            A populated :class:`~app.schemas.BotRunResult`.
        """
        symbols = [c.symbol for c in candidates]
        k = len(symbols)
        # Price matrix P[t, j] for candidate j at bar t (all strictly positive).
        prices = np.column_stack([c.closes for c in candidates])
        n = prices.shape[0]

        rf_daily = self._rf_daily()
        rebalance_days = max(1, int(config.rebalance_days or 21))
        stop_loss = max(0.0, _finite(config.stop_loss_pct, 25.0)) / 100.0
        max_dd = max(0.0, _finite(config.max_drawdown_pct, 35.0)) / 100.0

        # Synthetic index for regime detection = equal-weight candidate index.
        index_closes = self._equal_weight_index(prices)

        # --- portfolio state ---
        cash = float(amount)
        units = np.zeros(k, dtype=np.float64)         # units held per sleeve
        entry_price = np.zeros(k, dtype=np.float64)   # last entry price per sleeve
        sleeve_pnl = np.zeros(k, dtype=np.float64)    # realized + closed marked P&L
        sleeve_trades = np.zeros(k, dtype=np.int64)
        sleeve_wins = np.zeros(k, dtype=np.int64)
        sleeve_legs = np.zeros(k, dtype=np.int64)
        # Marked value of each sleeve at the previous rebalance (for win/leg + reward).
        last_rebalance_value = np.zeros(k, dtype=np.float64)

        trades: list[BotTrade] = []
        equity: list[BotEquityPoint] = []
        regime_timeline: list[str] = []
        now_ms = int(time.time() * 1000)
        day_ms = 86_400_000
        # Anchor the timeline so the last bar is "now".
        t0 = now_ms - (n - 1) * day_ms

        # Benchmark: equal-weight buy & hold of the candidate set, same start cash.
        bench_units = (amount / float(k)) / prices[0]
        bench_value0 = float(np.dot(bench_units, prices[0]))

        peak_value = float(amount)

        for t in range(n):
            px = prices[t]
            # Mark current sleeve values.
            sleeve_value = units * px
            total_value = cash + float(np.sum(sleeve_value))
            peak_value = max(peak_value, total_value)
            drawdown = (total_value / peak_value - 1.0) if peak_value > _EPS else 0.0

            # Regime at this bar (uses history up to and including t).
            regime = projection.detect_regime(index_closes[: t + 1])
            regime_label = str(regime.get("regime", "neutral"))

            is_rebalance = (t % rebalance_days == 0) or (t == 0)
            if is_rebalance and t < n:
                regime_timeline.append(regime_label)
                # Settle the leg that just ended: realize per-sleeve P&L vs the
                # last rebalance mark and credit win/leg counters.
                if t > 0:
                    for j in range(k):
                        if last_rebalance_value[j] > _EPS or units[j] > _EPS:
                            # Per-leg win/loss is judged on the marked value change
                            # since the last rebalance (realized $ P&L is tracked
                            # separately on sells/liquidation for attribution).
                            leg_pnl = sleeve_value[j] - last_rebalance_value[j]
                            sleeve_legs[j] += 1
                            if leg_pnl > 0.0:
                                sleeve_wins[j] += 1

                cash, units, entry_price, sleeve_pnl, sleeve_trades = self._rebalance(
                    t=t,
                    px=px,
                    cash=cash,
                    units=units,
                    entry_price=entry_price,
                    sleeve_pnl=sleeve_pnl,
                    sleeve_trades=sleeve_trades,
                    prices=prices,
                    candidates=candidates,
                    policy=policy,
                    rf_daily=rf_daily,
                    stop_loss=stop_loss,
                    max_dd=max_dd,
                    drawdown=drawdown,
                    regime=regime,
                    trades=trades,
                    t_ms=t0 + t * day_ms,
                )
                # Re-mark after trading for the leg baseline.
                last_rebalance_value = units * px

            # Record the (post-trade) equity point.
            sleeve_value = units * px
            total_value = cash + float(np.sum(sleeve_value))
            peak_value = max(peak_value, total_value)
            drawdown = (total_value / peak_value - 1.0) if peak_value > _EPS else 0.0
            bench_value = float(np.dot(bench_units, px))
            equity.append(
                BotEquityPoint(
                    t=t0 + t * day_ms,
                    bot_value=round(_finite(total_value), 2),
                    benchmark_value=round(_finite(bench_value), 2),
                    drawdown_pct=round(_finite(drawdown) * 100.0, 4),
                    regime=regime_label,
                )
            )

        # Final liquidation realize (mark remaining sleeves to last price as P&L
        # relative to entry) so attribution reflects open positions.
        last_px = prices[-1]
        for j in range(k):
            if units[j] > _EPS and entry_price[j] > _EPS:
                sleeve_pnl[j] += units[j] * (last_px[j] - entry_price[j])

        bot_values = np.array([e.bot_value for e in equity], dtype=np.float64)
        bench_values = np.array([e.benchmark_value for e in equity], dtype=np.float64)
        metrics = self._metrics(
            bot_values, bench_values, amount, rf_daily, sleeve_legs, sleeve_wins
        )

        stats = [
            SleeveStat(
                key=symbols[j],
                realized_pnl=_finite(sleeve_pnl[j]),
                trades=int(sleeve_trades[j]),
                wins=int(sleeve_wins[j]),
                legs=int(sleeve_legs[j]),
            )
            for j in range(k)
        ]
        attribution = build_attribution(stats)
        best = attribution[0].key if attribution and attribution[0].verdict == "best" else None
        worst = (
            attribution[-1].key
            if attribution and attribution[-1].verdict == "worst"
            else None
        )

        return BotRunResult(
            mode=policy.mode,
            config=config,
            equity_curve=self._downsample(equity),
            trades=trades,
            attribution=attribution,
            metrics=metrics,
            best_strategy=best,
            worst_strategy=worst,
            regime_timeline=regime_timeline,
            disclaimer=BOT_DISCLAIMER,
        )

    # ------------------------------------------------------------------
    # One rebalance: select → base weights → rotate → risk → trade
    # ------------------------------------------------------------------

    def _rebalance(
        self,
        *,
        t: int,
        px: np.ndarray,
        cash: float,
        units: np.ndarray,
        entry_price: np.ndarray,
        sleeve_pnl: np.ndarray,
        sleeve_trades: np.ndarray,
        prices: np.ndarray,
        candidates: list[_Candidate],
        policy: ModePolicy,
        rf_daily: float,
        stop_loss: float,
        max_dd: float,
        drawdown: float,
        regime: dict,
        trades: list[BotTrade],
        t_ms: int,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute a single rebalance and return the new portfolio state.

        Returns:
            A ``(cash, units, entry_price, sleeve_pnl, sleeve_trades)`` tuple with
            the updated portfolio state after trading to the target weights.
        """
        k = len(candidates)

        # Current marked value (pre-trade) — realize P&L on any unit we sell.
        sleeve_value = units * px
        total_value = cash + float(np.sum(sleeve_value))
        if total_value <= _EPS:
            return cash, units, entry_price, sleeve_pnl, sleeve_trades

        # --- (1) market analysis + selection: rank candidates by composite,
        #         keep the top max_names for THIS rebalance ---
        max_names = max(1, int(policy.mode.max_names))
        order = sorted(range(k), key=lambda j: candidates[j].score, reverse=True)
        selected = order[: min(max_names, k)]

        # --- (2) base weights over the selected sleeves per the mode objective ---
        base = self._base_weights(selected, prices, t, rf_daily, policy)

        # --- (3) rotation: momentum / bandit tilt (winners up, losers DOWN) ---
        tilted = self._rotate(
            selected, base, prices, t, sleeve_legs=None, policy=policy
        )

        # --- (4a) per-sleeve stop-loss: any held sleeve down > stop_loss from its
        #          entry is forced to weight 0 (exit), regardless of selection ---
        target = np.zeros(k, dtype=np.float64)
        for idx, j in enumerate(selected):
            target[j] = tilted[idx]
        if stop_loss > 0.0:
            for j in range(k):
                if units[j] > _EPS and entry_price[j] > _EPS:
                    dd_j = px[j] / entry_price[j] - 1.0
                    if dd_j < -stop_loss:
                        target[j] = 0.0

        # Renormalize the (post-stop) target weights to sum to <= 1.
        tsum = float(np.sum(target))
        if tsum > _EPS:
            target = target / tsum
        else:
            target = np.zeros(k, dtype=np.float64)

        # --- (4b) drawdown circuit-breaker: if portfolio drawdown breached the
        #          limit, scale risk down to a defensive cash floor (raise cash) ---
        invest_frac = 1.0
        if max_dd > 0.0 and drawdown < -max_dd:
            invest_frac = 1.0 - _DRAWDOWN_CASH_FLOOR
        target = target * invest_frac

        # --- trade to the target dollar weights ---
        target_dollars = target * total_value
        target_units = np.where(px > _EPS, target_dollars / px, 0.0)

        new_units = np.array(units, dtype=np.float64)
        new_entry = np.array(entry_price, dtype=np.float64)
        new_cash = cash
        new_pnl = np.array(sleeve_pnl, dtype=np.float64)
        new_trades = np.array(sleeve_trades, dtype=np.int64)

        for j in range(k):
            delta_units = target_units[j] - units[j]
            if abs(delta_units) * px[j] < 1e-6:
                continue
            trade_dollars = delta_units * px[j]
            cost = _COST_RATE * abs(trade_dollars)
            if delta_units > 0.0:
                # BUY: update weighted-average entry price.
                prev_cost = units[j] * new_entry[j]
                new_units[j] = units[j] + delta_units
                if new_units[j] > _EPS:
                    new_entry[j] = (prev_cost + delta_units * px[j]) / new_units[j]
                new_cash -= trade_dollars + cost
                side = "buy"
            else:
                # SELL: realize P&L on the units sold vs entry.
                sold = -delta_units
                if new_entry[j] > _EPS:
                    new_pnl[j] += sold * (px[j] - new_entry[j])
                new_units[j] = max(0.0, units[j] + delta_units)
                new_cash += (-trade_dollars) - cost
                if new_units[j] <= _EPS:
                    new_entry[j] = 0.0
                side = "sell"
            new_pnl[j] -= cost
            new_trades[j] += 1
            trades.append(
                BotTrade(
                    t=t_ms,
                    symbol=candidates[j].symbol,
                    side=side,  # type: ignore[arg-type]
                    amount=round(_finite(abs(trade_dollars)), 2),
                    strategy=policy.mode.name,
                    price=round(_finite(px[j]), 6),
                )
            )

        if not math.isfinite(new_cash):
            new_cash = cash
        new_cash = max(0.0, new_cash)
        return new_cash, new_units, new_entry, new_pnl, new_trades

    # ------------------------------------------------------------------
    # Base weights per objective
    # ------------------------------------------------------------------

    def _base_weights(
        self,
        selected: list[int],
        prices: np.ndarray,
        t: int,
        rf_daily: float,
        policy: ModePolicy,
    ) -> np.ndarray:
        """Base portfolio weights over the selected sleeves per the mode objective.

        Uses a trailing window ending at bar ``t`` to estimate annualized returns
        and covariance, then dispatches on the objective:

            * ``min_volatility`` / ``max_sharpe`` → :func:`app.quant.portfolio.optimize`
            * ``momentum`` → weights proportional to positive trailing momentum
            * ``risk_parity`` → inverse-volatility weights
            * ``bandit`` → equal base (the rotation softmax does the allocation)

        Args:
            selected: Indices (into the candidate list) of the chosen sleeves.
            prices: The full price matrix.
            t: The current bar index.
            rf_daily: Daily risk-free rate.
            policy: The mode policy.

        Returns:
            A weight vector over ``selected`` (length ``len(selected)``) summing
            to 1, all non-negative and finite.
        """
        m = len(selected)
        if m == 0:
            return np.empty(0, dtype=np.float64)
        if m == 1:
            return np.ones(1, dtype=np.float64)

        # Trailing window of daily returns ending at bar t (~1 year, capped to data).
        win = min(TRADING_DAYS, t)
        if win < 5:
            return np.full(m, 1.0 / m, dtype=np.float64)
        seg = prices[t - win : t + 1][:, selected]  # (win+1, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = seg[1:] / seg[:-1] - 1.0
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

        objective = policy.objective

        if objective in ("min_volatility", "max_sharpe"):
            mu = np.mean(rets, axis=0) * TRADING_DAYS
            cov = np.cov(rets, rowvar=False) * TRADING_DAYS
            obj = "min_volatility" if objective == "min_volatility" else "max_sharpe"
            w = _portfolio.optimize(mu, cov, rf_daily * TRADING_DAYS, obj)
            return self._clean_weights(w, m)

        if objective == "risk_parity":
            vol = np.std(rets, axis=0)
            inv = np.where(vol > _EPS, 1.0 / vol, 0.0)
            if float(np.sum(inv)) <= _EPS:
                return np.full(m, 1.0 / m, dtype=np.float64)
            return self._clean_weights(inv / np.sum(inv), m)

        if objective == "momentum":
            # Trailing total return over the window; only positive momentum gets
            # weight, and we never short. If nothing is positive, hold equal cash-
            # like equal weight (the rotation/stop-loss handle the rest).
            total_ret = seg[-1] / seg[0] - 1.0
            pos = np.clip(total_ret, 0.0, None)
            if float(np.sum(pos)) <= _EPS:
                return np.full(m, 1.0 / m, dtype=np.float64)
            return self._clean_weights(pos / np.sum(pos), m)

        # bandit / default: equal base; the softmax rotation allocates.
        return np.full(m, 1.0 / m, dtype=np.float64)

    @staticmethod
    def _clean_weights(w: np.ndarray, m: int) -> np.ndarray:
        """Clip to non-negative, finite weights summing to 1 (equal-weight fallback)."""
        arr = np.asarray(w, dtype=np.float64).ravel()
        if arr.size != m or not np.all(np.isfinite(arr)):
            return np.full(m, 1.0 / m, dtype=np.float64)
        arr = np.clip(arr, 0.0, 1.0)
        s = float(np.sum(arr))
        if s <= _EPS:
            return np.full(m, 1.0 / m, dtype=np.float64)
        return arr / s

    # ------------------------------------------------------------------
    # Rotation: momentum / bandit softmax tilt (NEVER martingale)
    # ------------------------------------------------------------------

    def _rotate(
        self,
        selected: list[int],
        base: np.ndarray,
        prices: np.ndarray,
        t: int,
        sleeve_legs,
        policy: ModePolicy,
    ) -> np.ndarray:
        """Tilt base weights toward sleeves with strong trailing performance.

        Each sleeve's reward is its trailing risk-adjusted return over the policy
        lookback, clipped to ``[-1, 1]``. The tilt is ``exp(temperature * reward)``
        — monotone increasing, so a winner's weight can only RISE and a loser's
        only FALL relative to the base (momentum, never martingale). For the
        bandit policy an optimistic exploration bonus (scaled by trailing
        volatility) is added so under-sampled / higher-uncertainty sleeves keep a
        chance. After tilting, each weight is capped at ``max_weight`` and the
        vector is renormalized.

        Args:
            selected: Indices of the selected sleeves.
            base: Base weights over the selected sleeves (sums to 1).
            prices: The full price matrix.
            t: The current bar index.
            sleeve_legs: Unused placeholder (kept for signature symmetry).
            policy: The mode policy (rotation params).

        Returns:
            The tilted weight vector over ``selected`` (sums to 1).
        """
        rot: RotationPolicy = policy.rotation
        m = len(selected)
        if m == 0:
            return base
        if rot.temperature <= 0.0:
            # Rebalance-only: no tilt, just enforce the cap.
            return self._cap_and_normalize(base, rot.max_weight)

        lb = max(5, int(rot.lookback_days))
        win = min(lb, t)
        if win < 5:
            return self._cap_and_normalize(base, rot.max_weight)

        seg = prices[t - win : t + 1][:, selected]  # (win+1, m)
        with np.errstate(divide="ignore", invalid="ignore"):
            rets = seg[1:] / seg[:-1] - 1.0
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

        mean_d = np.mean(rets, axis=0)
        std_d = np.std(rets, axis=0)
        # Trailing annualized risk-adjusted return (Sharpe-like), bounded reward.
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpe_like = np.where(std_d > _EPS, mean_d / std_d, 0.0) * math.sqrt(
                TRADING_DAYS
            )
        sharpe_like = np.nan_to_num(sharpe_like, nan=0.0, posinf=0.0, neginf=0.0)
        # Squash to [-1, 1] so the softmax temperature has a stable scale.
        reward = np.tanh(sharpe_like / 2.0)

        if rot.bandit:
            # Thompson-style optimism: higher trailing vol ⇒ more uncertainty ⇒
            # a small exploration bonus (still bounded, still momentum-respecting).
            explore = np.tanh(std_d * math.sqrt(TRADING_DAYS))
            reward = np.clip(reward + 0.25 * explore, -1.0, 1.0)

        tilt = np.exp(rot.temperature * reward)
        tilt = np.nan_to_num(tilt, nan=1.0, posinf=1.0, neginf=1.0)
        w = base * tilt
        s = float(np.sum(w))
        if s <= _EPS:
            w = base.copy()
        else:
            w = w / s
        return self._cap_and_normalize(w, rot.max_weight)

    @staticmethod
    def _cap_and_normalize(w: np.ndarray, max_weight: float) -> np.ndarray:
        """Cap each weight at ``max_weight`` and renormalize to sum to 1.

        Applies the cap iteratively (clamping spills onto the uncapped names) so
        the result respects the concentration guard while still summing to 1.

        Args:
            w: A non-negative weight vector summing to ~1.
            max_weight: The per-name cap in ``(0, 1]``.

        Returns:
            A capped, normalized weight vector summing to 1.
        """
        arr = np.asarray(w, dtype=np.float64).ravel()
        m = arr.size
        if m == 0:
            return arr
        cap = float(max_weight) if math.isfinite(max_weight) and max_weight > 0.0 else 1.0
        # If the cap is infeasible (cap * m < 1) fall back to equal weight.
        if cap * m <= 1.0 + 1e-9:
            return np.full(m, 1.0 / m, dtype=np.float64)
        arr = np.clip(arr, 0.0, None)
        s = float(np.sum(arr))
        if s <= _EPS:
            return np.full(m, 1.0 / m, dtype=np.float64)
        arr = arr / s
        # Iterative water-filling to respect the cap.
        for _ in range(m):
            over = arr > cap
            if not np.any(over):
                break
            excess = float(np.sum(arr[over] - cap))
            arr[over] = cap
            under = ~over
            under_sum = float(np.sum(arr[under]))
            if under_sum <= _EPS:
                arr[:] = 1.0 / m
                break
            arr[under] += excess * (arr[under] / under_sum)
        s = float(np.sum(arr))
        return arr / s if s > _EPS else np.full(m, 1.0 / m, dtype=np.float64)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _metrics(
        self,
        bot_values: np.ndarray,
        bench_values: np.ndarray,
        amount: float,
        rf_daily: float,
        sleeve_legs: np.ndarray,
        sleeve_wins: np.ndarray,
    ) -> BotMetrics:
        """Compute the realized :class:`~app.schemas.BotMetrics` for the run.

        Args:
            bot_values: The bot's total value at each bar.
            bench_values: The benchmark value at each bar.
            amount: Starting capital.
            rf_daily: Daily risk-free rate.
            sleeve_legs: Per-sleeve completed leg counts.
            sleeve_wins: Per-sleeve profitable leg counts.

        Returns:
            A populated, finite :class:`~app.schemas.BotMetrics`.
        """
        n = bot_values.size
        if n < 2 or amount <= 0.0:
            return BotMetrics(
                total_return_pct=0.0,
                cagr_pct=0.0,
                sharpe=0.0,
                sortino=0.0,
                max_drawdown_pct=0.0,
                win_rate_pct=0.0,
                vs_benchmark_pct=0.0,
                final_value=round(_finite(float(bot_values[-1]) if n else amount), 2),
            )

        final = float(bot_values[-1])
        total_return = final / amount - 1.0

        with np.errstate(divide="ignore", invalid="ignore"):
            daily = bot_values[1:] / bot_values[:-1] - 1.0
        daily = np.nan_to_num(daily, nan=0.0, posinf=0.0, neginf=0.0)

        years = max(n / float(TRADING_DAYS), 1.0 / TRADING_DAYS)
        ratio = final / amount
        if ratio > _EPS and math.isfinite(ratio):
            try:
                cagr = ratio ** (1.0 / years) - 1.0
            except (OverflowError, ValueError):
                cagr = 0.0
        else:
            cagr = -1.0

        sharpe = _metrics.sharpe(daily, rf_daily)
        sortino = _metrics.sortino(daily, rf_daily)
        mdd = _metrics.max_drawdown(bot_values)

        total_legs = int(np.sum(sleeve_legs))
        total_wins = int(np.sum(sleeve_wins))
        win_rate = (total_wins / total_legs) if total_legs > 0 else 0.0

        bench_final = float(bench_values[-1]) if bench_values.size else amount
        bench_return = bench_final / amount - 1.0
        vs_bench = (total_return - bench_return) * 100.0

        return BotMetrics(
            total_return_pct=round(_finite(total_return) * 100.0, 4),
            cagr_pct=round(_finite(cagr) * 100.0, 4),
            sharpe=round(_finite(sharpe), 4),
            sortino=round(_finite(sortino), 4),
            max_drawdown_pct=round(_finite(mdd) * 100.0, 4),
            win_rate_pct=round(_finite(win_rate) * 100.0, 2),
            vs_benchmark_pct=round(_finite(vs_bench), 4),
            final_value=round(_finite(final), 2),
        )

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _rf_daily(self) -> float:
        """Return the mean daily risk-free rate from the factor series (else 0)."""
        try:
            factors = self._provider.factor_history(days=_BACKTEST_DAYS)
            rf = np.asarray(factors.get("rf", np.empty(0)), dtype=np.float64).ravel()
            v = float(np.mean(rf)) if rf.size else 0.0
            return v if math.isfinite(v) else 0.0
        except Exception:  # pragma: no cover - defensive
            return 0.0

    @staticmethod
    def _equal_weight_index(prices: np.ndarray) -> np.ndarray:
        """Build an equal-weight, base-100 synthetic index from a price matrix.

        Each column is normalized to its first value, averaged across columns, and
        scaled to 100 at the start — a clean series for regime detection.

        Args:
            prices: The ``(T, K)`` candidate price matrix (strictly positive).

        Returns:
            A length-``T`` strictly-positive index series.
        """
        p0 = prices[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            normed = prices / np.where(p0 > _EPS, p0, 1.0)
        normed = np.nan_to_num(normed, nan=1.0, posinf=1.0, neginf=1.0)
        idx = np.mean(normed, axis=1) * 100.0
        idx = np.where(idx > _EPS, idx, _EPS)
        return idx

    @staticmethod
    def _downsample(
        equity: list[BotEquityPoint], points: int = _EQUITY_POINTS
    ) -> list[BotEquityPoint]:
        """Down-sample an equity curve to ~``points`` evenly-spaced points.

        The first and last bars are always kept. ``t``/regime/values are taken
        from the original points (no interpolation), so the curve stays faithful.

        Args:
            equity: The full per-bar equity curve.
            points: Target number of output points.

        Returns:
            A list of :class:`~app.schemas.BotEquityPoint` of length <= ``points``.
        """
        n = len(equity)
        if n <= points:
            return equity
        idx = np.unique(np.linspace(0, n - 1, num=points, dtype=np.int64))
        return [equity[int(i)] for i in idx]

    def _empty_result(self, config: BotConfig, policy: ModePolicy) -> BotRunResult:
        """Build a flat, finite result for a degenerate config (no candidates).

        Args:
            config: The bot config (echoed back).
            policy: The mode policy.

        Returns:
            A :class:`~app.schemas.BotRunResult` with a flat equity curve at the
            starting amount, no trades, and zeroed metrics — still carrying the
            disclaimer.
        """
        amount = max(0.0, _finite(config.amount, 0.0))
        now_ms = int(time.time() * 1000)
        equity = [
            BotEquityPoint(
                t=now_ms,
                bot_value=round(amount, 2),
                benchmark_value=round(amount, 2),
                drawdown_pct=0.0,
                regime="neutral",
            )
        ]
        metrics = BotMetrics(
            total_return_pct=0.0,
            cagr_pct=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown_pct=0.0,
            win_rate_pct=0.0,
            vs_benchmark_pct=0.0,
            final_value=round(amount, 2),
        )
        return BotRunResult(
            mode=policy.mode,
            config=config,
            equity_curve=equity,
            trades=[],
            attribution=[],
            metrics=metrics,
            best_strategy=None,
            worst_strategy=None,
            regime_timeline=[],
            disclaimer=BOT_DISCLAIMER,
        )
