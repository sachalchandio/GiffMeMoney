"""The Real-Time mode engine — a ticking, multi-venue paper book.

A :class:`LiveSimSession` holds up to 80 *venues* (markets), each with its own
deterministic intraday path and a self-updating :class:`~app.livesim.predictor.
OnlinePredictor`. Calling :func:`tick` advances simulated time: every venue
learns from the latest move, gets re-scored, and on a cadence the book
**reallocates** — spreading wider as equity grows (the user's mental model) and
rotating capital toward the venues doing best, charging a realistic cost on every
trade. Per-step marks feed an equity curve and a **daily** profit/loss series.

HONESTY: accelerated SIMULATION on synthetic data, $0 real. Costs are charged so
churn bleeds; drift/vol are plausible so returns stay realistic (no $20→$4k). The
predictor genuinely updates itself but, lacking a real edge, hovers near a coin
flip — and says so via its confidence. The engine never raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.hft.costs import CostModel, get_cost_model
from app.hft.intraday import generate_intraday
from app.hft.signals import raw_exposure
from app.livesim.predictor import OnlinePredictor, features_from_path

__all__ = [
    "Venue",
    "LiveSimSession",
    "LIVESIM_MAX_VENUES",
    "LIVESIM_MIN_VENUES",
    "create_session",
    "tick",
]

#: Hard ceiling on venues the book can ever spread across (the user's cap).
LIVESIM_MAX_VENUES: int = 80
#: Floor so even a tiny balance is diversified a little.
LIVESIM_MIN_VENUES: int = 5

# Internal sim resolution.
_DAYS: int = 180
_BARS_PER_DAY: int = 20
_EQUITY_CURVE_CAP: int = 1200
_TRADES_CAP: int = 40
_EPS: float = 1e-12

# Real tickers we prefer for the first venues (the rest are synthetic SIM-xx).
_REAL_SYMBOLS: tuple[str, ...] = (
    "BTC", "ETH", "SOL", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META",
    "JPM", "XRP", "ADA", "DOGE", "AVAX", "LINK", "MATIC", "DOT", "AMD", "NFLX",
)


def _finite(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _venue_params(index: int) -> tuple[float, float]:
    """Deterministic (annual_drift, annual_vol) giving each venue its own character."""
    # A cheap, stable pseudo-random in [0, 1) from the index.
    r1 = ((index * 2654435761) % 1000) / 1000.0
    r2 = ((index * 40503 + 12345) % 1000) / 1000.0
    drift = -0.04 + 0.16 * r1   # roughly [-4%, +12%] annual
    vol = 0.25 + 0.75 * r2      # roughly [25%, 100%] annual
    return drift, vol


@dataclass
class Venue:
    """One market the book can trade (paper).

    Attributes:
        symbol: Ticker / synthetic id.
        label: Human-readable name.
        path: The venue's deterministic intraday price path.
        cursor: Current index into ``path``.
        predictor: The venue's self-updating predictor.
        units: Paper units currently held.
        entry_price: Average entry price of the current holding.
        score: Latest blended score (signal + prediction), in ``[-1, 1]``.
        pred_up: Latest predicted probability the next move is up.
    """

    symbol: str
    label: str
    path: np.ndarray
    cursor: int
    predictor: OnlinePredictor
    units: float = 0.0
    entry_price: float = 0.0
    score: float = 0.0
    pred_up: float = 0.5

    @property
    def price(self) -> float:
        """Current marked price at the cursor."""
        i = min(max(self.cursor, 0), self.path.size - 1)
        return float(self.path[i])


@dataclass
class LiveSimSession:
    """A live-sim session: configuration, venues, book state, and history."""

    id: str
    # config
    amount: float
    signal: str
    cost_preset: str
    dollars_per_venue: float
    max_venues: int
    rebalance_every: int
    stop_loss_pct: float
    max_drawdown_pct: float
    steps_per_tick: int
    # venues + book
    venues: list[Venue]
    cash: float
    start_equity: float
    peak_equity: float
    day_start_equity: float
    step: int = 0
    day: int = 0
    finished: bool = False
    # history
    equity_curve: list[float] = field(default_factory=list)
    daily_pnl: list[dict] = field(default_factory=list)
    trades: list[dict] = field(default_factory=list)

    @property
    def cost_model(self) -> CostModel:
        """The resolved cost model for this session."""
        return get_cost_model(self.cost_preset)

    @property
    def steps_per_day(self) -> int:
        """Steps that constitute one simulated day."""
        return _BARS_PER_DAY

    def equity(self) -> float:
        """Current total paper equity (cash + marked positions)."""
        held = sum(v.units * v.price for v in self.venues)
        return _finite(self.cash + held, self.cash)


def _seed_from_id(session_id: str) -> int:
    """Derive a stable int offset from a session id string."""
    return abs(hash(session_id)) % (10**6)


def create_session(
    session_id: str,
    *,
    amount: float = 20.0,
    signal: str = "momentum",
    cost_preset: str = "retail-crypto",
    dollars_per_venue: float = 1.0,
    max_venues: int = 20,
    rebalance_every: int = 3,
    stop_loss_pct: float = 6.0,
    max_drawdown_pct: float = 20.0,
    steps_per_tick: int = 4,
) -> LiveSimSession:
    """Build a fresh live-sim session (never raises).

    Args:
        session_id: Stable id (drives the deterministic venue seeds).
        amount: Starting paper capital.
        signal: Short-horizon signal id (``momentum`` / ``meanrev``).
        cost_preset: Transaction-cost preset id.
        dollars_per_venue: Equity per venue used to decide how wide to spread.
        max_venues: Requested venue ceiling (clamped to ``[MIN, 80]``).
        rebalance_every: Steps between reallocations.
        stop_loss_pct: Per-venue stop (exit if down more than this from entry).
        max_drawdown_pct: Portfolio circuit-breaker (raise cash past this).
        steps_per_tick: How many steps one ``tick`` advances (the "speed").

    Returns:
        A ready :class:`LiveSimSession`.
    """
    amt = _finite(amount, 20.0)
    if amt <= 0.0:
        amt = 20.0
    cap = int(min(LIVESIM_MAX_VENUES, max(LIVESIM_MIN_VENUES, int(max_venues or 20))))
    seed = _seed_from_id(session_id)

    venues: list[Venue] = []
    for i in range(cap):
        if i < len(_REAL_SYMBOLS):
            sym = _REAL_SYMBOLS[i]
            label = sym
        else:
            sym = f"SIM-{i + 1:02d}"
            label = f"Sim venue {i + 1}"
        drift, vol = _venue_params(i + seed)
        series = generate_intraday(
            sym, days=_DAYS, bars_per_day=_BARS_PER_DAY,
            annual_drift=drift, annual_vol=vol, seed_offset=seed,
        )
        venues.append(
            Venue(
                symbol=sym,
                label=label,
                path=series.prices,
                cursor=1,
                predictor=OnlinePredictor(),
            )
        )

    return LiveSimSession(
        id=session_id,
        amount=amt,
        signal=str(signal or "momentum"),
        cost_preset=str(cost_preset or "retail-crypto"),
        dollars_per_venue=max(0.25, _finite(dollars_per_venue, 1.0)),
        max_venues=cap,
        rebalance_every=max(1, int(rebalance_every or 3)),
        stop_loss_pct=max(0.0, _finite(stop_loss_pct, 6.0)),
        max_drawdown_pct=max(0.0, _finite(max_drawdown_pct, 20.0)),
        steps_per_tick=int(min(20, max(1, steps_per_tick or 4))),
        venues=venues,
        cash=amt,
        start_equity=amt,
        peak_equity=amt,
        day_start_equity=amt,
        equity_curve=[round(amt, 4)],
    )


def tick(session: LiveSimSession, steps: int | None = None) -> LiveSimSession:
    """Advance the session by ``steps`` (default ``steps_per_tick``). Never raises."""
    try:
        n = int(steps) if steps is not None else session.steps_per_tick
    except (TypeError, ValueError):
        n = session.steps_per_tick
    n = max(1, min(60, n))
    try:
        for _ in range(n):
            if session.finished:
                break
            _advance_one(session)
    except Exception:  # pragma: no cover - defensive (tick never raises)
        session.finished = True
    return session


def _advance_one(session: LiveSimSession) -> None:
    """Advance exactly one simulated step: learn, rescore, maybe rebalance, mark."""
    # 1) Each venue: predict, advance, learn from the realised move, rescore.
    for v in session.venues:
        if v.cursor + 1 >= v.path.size:
            session.finished = True
            return
        t = v.cursor
        feats = features_from_path(v.path, t)
        p_up = v.predictor.predict_proba(feats)
        v.cursor += 1
        outcome = 1.0 if v.path[v.cursor] > v.path[t] else 0.0
        v.predictor.update(feats, outcome)
        v.pred_up = p_up
        sig = raw_exposure(session.signal, v.path, v.cursor, allow_short=False)  # [0,1]
        # Blend the rule-based signal with the model's edge over a coin flip.
        v.score = float(max(-1.0, min(1.0, 0.6 * sig + 0.4 * ((p_up - 0.5) * 2.0))))

    session.step += 1

    # 2) Rebalance on cadence.
    if session.step % session.rebalance_every == 0:
        _rebalance(session)

    # 3) Mark-to-market + history.
    equity = session.equity()
    session.peak_equity = max(session.peak_equity, equity)
    session.equity_curve.append(round(equity, 4))
    if len(session.equity_curve) > _EQUITY_CURVE_CAP:
        session.equity_curve = session.equity_curve[-_EQUITY_CURVE_CAP:]

    # 4) Daily boundary → record a daily P&L point.
    if session.step % session.steps_per_day == 0:
        session.day += 1
        prev = session.day_start_equity if session.day_start_equity > _EPS else session.start_equity
        day_pnl_pct = (equity / prev - 1.0) * 100.0 if prev > _EPS else 0.0
        session.daily_pnl.append(
            {"day": session.day, "pnlPct": round(_finite(day_pnl_pct), 4), "equity": round(equity, 2)}
        )
        session.day_start_equity = equity


def _rebalance(session: LiveSimSession) -> None:
    """Reallocate the book: spread by equity, rotate into winners, gate risk."""
    venues = session.venues
    equity = session.equity()
    if equity <= _EPS:
        return

    # How wide to spread, growing with equity, hard-capped at the venue count.
    target_n = int(equity / session.dollars_per_venue)
    target_n = max(LIVESIM_MIN_VENUES, min(target_n, session.max_venues, len(venues)))

    # Drawdown circuit-breaker: raise cash if we've fallen too far from the peak.
    invest_frac = 0.98  # always keep a small cash buffer
    dd = (equity / session.peak_equity - 1.0) if session.peak_equity > _EPS else 0.0
    if session.max_drawdown_pct > 0.0 and dd <= -(session.max_drawdown_pct / 100.0):
        invest_frac = 0.40

    stop = session.stop_loss_pct / 100.0

    # Rank by score; only invest where the score is positive (predicted to rise).
    ranked = sorted(venues, key=lambda v: v.score, reverse=True)
    chosen = [v for v in ranked[:target_n] if v.score > 0.0]

    # Score-proportional target weights with a per-name cap (concentration guard).
    target_w: dict[str, float] = {}
    if chosen:
        cap_w = max(1.0 / len(chosen), min(0.25, 3.0 / len(chosen)))
        raw = np.array([max(v.score, 0.0) for v in chosen], dtype=np.float64)
        if float(raw.sum()) <= _EPS:
            raw = np.ones(len(chosen), dtype=np.float64)
        w = raw / raw.sum()
        w = np.minimum(w, cap_w)
        s = float(w.sum())
        w = (w / s) if s > _EPS else np.full(len(chosen), 1.0 / len(chosen))
        for v, wi in zip(chosen, w):
            target_w[v.symbol] = float(wi) * invest_frac

    cost = session.cost_model
    # Trade every venue toward its target dollar weight (0 if not chosen / stopped).
    for v in venues:
        price = v.price
        if price <= _EPS:
            continue
        # Per-venue stop-loss: if held and underwater past the stop, force exit.
        weight = target_w.get(v.symbol, 0.0)
        if v.units > _EPS and v.entry_price > _EPS and stop > 0.0:
            if price / v.entry_price - 1.0 < -stop:
                weight = 0.0
        target_dollars = weight * equity
        cur_dollars = v.units * price
        delta = target_dollars - cur_dollars
        if abs(delta) < 0.01:  # ignore sub-cent dust trades
            continue
        c = cost.cost_of(delta)
        if delta > 0.0:  # BUY → update average entry
            prev_cost = v.units * v.entry_price
            v.units += delta / price
            v.entry_price = (prev_cost + delta) / v.units if v.units > _EPS else price
            session.cash -= delta + c
            side = "buy"
        else:  # SELL
            v.units = max(0.0, v.units + delta / price)
            session.cash += (-delta) - c
            if v.units <= _EPS:
                v.entry_price = 0.0
            side = "sell"
        session.trades.append(
            {
                "step": session.step,
                "symbol": v.symbol,
                "side": side,
                "amount": round(abs(_finite(delta)), 2),
                "price": round(_finite(price), 6),
            }
        )

    if not math.isfinite(session.cash):
        session.cash = 0.0
    session.cash = max(0.0, session.cash)
    if len(session.trades) > _TRADES_CAP:
        session.trades = session.trades[-_TRADES_CAP:]
