"""Mean-reversion & technical strategy builders (V2 expansion).

This module implements the ten mean-reversion / technical timing strategies from
``docs/research/strategy-catalog.json`` assigned to ``builders_meanrev_technical``:

    connors-rsi2              — Connors RSI(2) pullback (trend-filtered)
    connors-cumulative-rsi2   — Connors cumulative RSI(2) (multi-day confirmation)
    zscore-reversion          — rolling z-score statistical mean reversion
    pairs-trading             — relative-value spread vs the best-correlated peer
    bollinger-squeeze         — BandWidth squeeze + volatility breakout
    stochastic-oscillator     — slow %K/%D reversion
    williams-r                — Williams %R reversion
    cci-reversion             — Commodity Channel Index reversion
    keltner-reversion         — Keltner (ATR-band) mean reversion
    obv-volume-trend          — On-Balance Volume trend confirmation / divergence

Each builder consumes an :class:`~app.strategies.engine.AnalysisContext` and
returns a fully-validated :class:`~app.schemas.StrategySignal` via
:func:`app.strategies.base.make_signal`. ``computeSignal`` is implemented
*exactly* as written in the catalog. Score convention is **positive = bullish**;
confidence is in ``[0, 1]``.

The module exports two module-level dicts (mirroring the other ``builders_*``
modules):

    * :data:`BUILDERS` — ``dict[str, tuple[StrategyMeta, builder_fn]]`` for all 10
      ids, in catalog priority order.
    * :data:`POSITION_FUNCS` — ``dict[str, Callable]`` of *vectorized* per-bar
      position series (values in ``[-1, 1]``) for the timing strategies that are
      time-backtestable per the V2 backtest contract.

Robustness contract: every builder is numerically defensive. Short / empty / NaN
histories collapse to a safe, neutral ``HOLD`` (score ``0``) rather than raising.
The OHLC / volume arrays and the cross-sectional ``UniverseStats`` are read off
the context defensively (the engine attaches ``highs`` / ``lows`` / ``volumes`` /
``universe`` in the V2 build; when absent they degrade gracefully — highs/lows
fall back to ``closes`` and the cross-sectional pairs strategy returns a
no-tradable-pair neutral signal).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from app.quant import indicators, returns
from app.quant import technical
from app.schemas import StrategyMeta, StrategySignal
from app.strategies.base import clamp, make_signal

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from app.strategies.engine import AnalysisContext, UniverseStats  # noqa: F401

__all__ = ["BUILDERS", "POSITION_FUNCS"]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _safe(value: float, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, falling back to ``default``."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _closes(ctx: "AnalysisContext") -> np.ndarray:
    """Return the asset close series as a clean 1-D float array."""
    arr = np.asarray(getattr(ctx, "closes", np.empty(0)), dtype=np.float64).ravel()
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _ohlc(ctx: "AnalysisContext") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(highs, lows, closes)`` aligned to the close series.

    The V2 engine attaches ``highs`` / ``lows`` arrays to the context. When they
    are missing or mis-sized (e.g. while the integration agent has not yet wired
    them up), this falls back to using the close series for both bands so the
    oscillator math stays well defined (a degenerate but safe zero-range bar).

    Returns:
        A triple of equal-length finite ``float64`` arrays (possibly empty).
    """
    c = _closes(ctx)
    n = c.size
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy()

    raw_h = getattr(ctx, "highs", None)
    raw_l = getattr(ctx, "lows", None)
    h = np.asarray(raw_h, dtype=np.float64).ravel() if raw_h is not None else np.empty(0)
    l = np.asarray(raw_l, dtype=np.float64).ravel() if raw_l is not None else np.empty(0)

    if h.size >= n:
        h = np.nan_to_num(h[-n:], nan=0.0, posinf=0.0, neginf=0.0)
    else:
        h = c.copy()
    if l.size >= n:
        l = np.nan_to_num(l[-n:], nan=0.0, posinf=0.0, neginf=0.0)
    else:
        l = c.copy()

    # Guarantee high >= close >= low so range math is consistent even when the
    # bands are proxied from close.
    hi = np.maximum.reduce([h, l, c])
    lo = np.minimum.reduce([h, l, c])
    return hi, lo, c


def _volumes(ctx: "AnalysisContext", n: int) -> np.ndarray:
    """Return the volume series aligned to length ``n`` (synthetic if absent).

    When the engine has not attached real volumes, a flat unit-volume series is
    returned so OBV degrades to a sign-of-close-change accumulation rather than
    raising. All values are finite and non-negative.
    """
    raw_v = getattr(ctx, "volumes", None)
    if raw_v is not None:
        v = np.asarray(raw_v, dtype=np.float64).ravel()
        if v.size >= n and n > 0:
            v = np.nan_to_num(v[-n:], nan=0.0, posinf=0.0, neginf=0.0)
            return np.maximum(v, 0.0)
    return np.ones(max(0, n), dtype=np.float64)


def _is_crypto(ctx: "AnalysisContext") -> bool:
    """True when the asset is a crypto asset (fat tails -> lower confidence)."""
    try:
        ac = str(getattr(ctx.asset, "asset_class", "")).lower()
    except Exception:
        return False
    return ac == "crypto"


def _last(arr: np.ndarray, default: float = 0.0) -> float:
    """Latest finite element of an array, or ``default`` when empty."""
    if arr.size == 0:
        return default
    return _safe(arr[-1], default)


def _trend_sign(closes: np.ndarray, n: int = 200) -> int:
    """Trend filter: ``+1`` if last close > SMA(n) else ``-1`` (0 if no data)."""
    if closes.size == 0:
        return 0
    s = technical.sma(closes, n)
    if s.size == 0:
        return 0
    last = float(closes[-1])
    sma_last = float(s[-1])
    if not math.isfinite(sma_last):
        return 0
    return 1 if last > sma_last else -1


def _rsi_series(closes: np.ndarray, n: int = 2) -> np.ndarray:
    """Vectorized Wilder RSI series (one value per bar) over period ``n``.

    Mirrors :func:`app.quant.technical.rsi` but returns the whole series (the
    latest value equals ``technical.rsi``). Used for RSI(2) systems that need the
    prior bar's RSI (cumulative / position series). Defensive: short series →
    all-50 neutral.
    """
    arr = np.asarray(closes, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    L = arr.size
    if L < 2:
        return np.full(max(L, 1), 50.0, dtype=np.float64)
    period = max(1, int(n))
    deltas = np.diff(arr)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)
    alpha = 1.0 / period
    out = np.full(L, 50.0, dtype=np.float64)
    avg_gain = gains[0]
    avg_loss = losses[0]
    for i in range(1, deltas.size):
        avg_gain = alpha * gains[i] + (1.0 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1.0 - alpha) * avg_loss
        if avg_loss <= 0.0:
            out[i + 1] = 100.0 if avg_gain > 0.0 else 50.0
        else:
            rs = avg_gain / avg_loss
            val = 100.0 - 100.0 / (1.0 + rs)
            out[i + 1] = val if math.isfinite(val) else 50.0
    # The recursion above fills out[2:]; out[0] keeps the neutral seed and out[1]
    # is the single-delta RSI from the first up/down move.
    g0 = float(gains[0])
    l0 = float(losses[0])
    if l0 > 0.0:
        rs0 = g0 / l0
        out[1] = 100.0 - 100.0 / (1.0 + rs0)
    else:
        out[1] = 100.0 if g0 > 0.0 else 50.0
    return np.nan_to_num(out, nan=50.0, posinf=100.0, neginf=0.0)


def _zscore_series(closes: np.ndarray, n: int = 20) -> np.ndarray:
    """Rolling z-score of close over a trailing ``n``-window (per bar).

    ``z_t = (close_t - SMA_n) / std_n``; the leading region uses the expanding
    window. Zero-dispersion windows yield ``0``. Returns a per-bar array aligned
    to ``closes``.
    """
    arr = np.asarray(closes, dtype=np.float64).ravel()
    L = arr.size
    if L == 0:
        return np.empty(0, dtype=np.float64)
    window = max(2, int(n))
    out = np.zeros(L, dtype=np.float64)
    for t in range(L):
        start = max(0, t - window + 1)
        win = arr[start : t + 1]
        if win.size < 2:
            out[t] = 0.0
            continue
        mu = float(np.mean(win))
        sigma = float(np.std(win))
        if sigma <= 0.0 or not math.isfinite(sigma):
            out[t] = 0.0
        else:
            z = (float(arr[t]) - mu) / sigma
            out[t] = z if math.isfinite(z) else 0.0
    return np.clip(out, -10.0, 10.0)


def _lag1_autocorr(series: np.ndarray) -> float:
    """Lag-1 autocorrelation of a series (finite, in ``[-1, 1]``).

    Negative autocorrelation indicates mean-reverting behaviour (good for the
    reversion strategies); positive indicates trending/persistence.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    a = x[:-1]
    b = x[1:]
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 0.0 or sb <= 0.0:
        return 0.0
    cov = float(np.mean((a - np.mean(a)) * (b - np.mean(b))))
    rho = cov / (sa * sb)
    return clamp(rho, -1.0, 1.0)


def _meta_from_catalog(
    strategy_id: str,
    name: str,
    category: str,
    summary: str,
    formula: str,
    inputs: list[str],
    sources: list[str],
) -> StrategyMeta:
    """Assemble a :class:`StrategyMeta` carrying the catalog summary + sources."""
    return StrategyMeta(
        id=strategy_id,
        name=name,
        category=category,  # type: ignore[arg-type]
        summary=summary,
        formula=formula,
        inputs=inputs,
        references=list(sources),
    )


# Catalog sources (carried verbatim into StrategyMeta.references) -----------

_SRC_CONNORS_RSI2 = [
    "Connors & Alvarez, 'Short Term Trading Strategies That Work' (2009), RSI(2) chapter",
    "StockCharts ChartSchool RSI(2) - https://chartschool.stockcharts.com/table-of-contents/trading-strategies-and-models/trading-strategies/rsi-2",
]
_SRC_CONNORS_CUM = [
    "Connors & Alvarez, 'Short Term Trading Strategies That Work' (2009), ch. 9 (Cumulative RSI)",
    "Easycators - Cumulative RSI-2 - https://easycators.com/thinkscript/cumulative-rsi-2-trading-strategy/",
]
_SRC_ZSCORE = [
    "Gatev, Goetzmann & Rouwenhorst (2006), 'Pairs Trading', Review of Financial Studies - https://doi.org/10.1093/rfs/hhj020",
    "Ernest Chan, 'Algorithmic Trading' (Wiley, 2013) - z-score/half-life",
]
_SRC_PAIRS = [
    "Gatev, Goetzmann & Rouwenhorst (2006), 'Pairs Trading: Performance of a Relative-Value Arbitrage Rule', Review of Financial Studies - https://doi.org/10.1093/rfs/hhj020",
    "Ernest Chan, 'Algorithmic Trading' (Wiley, 2013)",
]
_SRC_BB_SQUEEZE = [
    "John Bollinger (2001), 'Bollinger on Bollinger Bands' - The Squeeze & Volatility Breakout Method I",
    "StockCharts ChartSchool - Bollinger Band Squeeze",
]
_SRC_STOCH = [
    "George C. Lane - Stochastic Oscillator (late 1950s)",
    "StockCharts ChartSchool - Stochastic Oscillator (Fast/Slow/Full)",
]
_SRC_WILLIAMS = [
    "Larry Williams (1979), 'How I Made One Million Dollars Last Year Trading Commodities'",
    "StockCharts ChartSchool - Williams %R",
]
_SRC_CCI = [
    "Donald R. Lambert (1980), 'Commodity Channel Index', Commodities Magazine",
    "StockCharts ChartSchool - CCI",
]
_SRC_KELTNER = [
    "Chester W. Keltner (1960), 'How To Make Money in Commodities'",
    "StockCharts ChartSchool - Keltner Channels",
]
_SRC_OBV = [
    "Joseph Granville (1963), \"Granville's New Key to Stock Market Profits\" - OBV",
    "StockCharts ChartSchool - On Balance Volume (OBV)",
]


# ---------------------------------------------------------------------------
# 1. Connors RSI(2) -- mean-reversion / Technical
# ---------------------------------------------------------------------------

_META_CONNORS_RSI2 = _meta_from_catalog(
    "connors-rsi2",
    "Connors RSI(2) Mean Reversion",
    "Technical",
    (
        "Larry Connors' 2-period RSI pullback system: buy short-term oversold dips "
        "inside an established uptrend (Close>SMA200), exit on the snap-back. "
        "~75-80% historical win rates on equity indices with 1-3 day holds."
    ),
    "trend=sign(Close-SMA200); long if RSI(2)<10, short if RSI(2)>90; signal ~ depth of the extreme",
    ["price history (SMA200 / SMA5 filters)", "RSI(2)"],
    _SRC_CONNORS_RSI2,
)


def _build_connors_rsi2(ctx: "AnalysisContext") -> StrategySignal:
    """Connors RSI(2): trend-filtered short-term oversold/overbought pullback.

    Implements the catalog ``computeSignal`` exactly:

        trend = +1 if Close>SMA200 else -1
        if trend==+1 and RSI2<10: signal = +100 * max(0, (10-RSI2)/10)
        if trend==-1 and RSI2>90: signal = -100 * max(0, (RSI2-90)/10)
        outside the zones the signal decays toward 0; once Close crosses above
        SMA5 (the exit) the signal is driven toward 0.
        confidence = 0.5 + 0.3*(RSI2<5 or RSI2>95) + 0.2*(trend agreement);
        crypto is reduced (fat tails).
    """
    meta = _META_CONNORS_RSI2
    c = _closes(ctx)
    if c.size < 3:
        return _neutral(meta, "need >= 3 closes for RSI(2)")
    rsi2 = technical.rsi(c, n=2)
    trend = _trend_sign(c, 200)
    sma5 = technical.sma(c, 5)
    above_sma5 = bool(sma5.size and float(c[-1]) > float(sma5[-1]))

    score = 0.0
    if trend == 1 and rsi2 < 10.0:
        score = 100.0 * max(0.0, (10.0 - rsi2) / 10.0)
    elif trend == -1 and rsi2 > 90.0:
        score = -100.0 * max(0.0, (rsi2 - 90.0) / 10.0)
    else:
        # Outside the arming zones: small decayed tilt toward reversion so the
        # signal is informative but not high-conviction.
        score = clamp((50.0 - rsi2) / 50.0 * 20.0, -20.0, 20.0)

    # Exit logic: a long that has snapped back above SMA5 is being closed -> pull
    # a bullish reading toward neutral; mirror for shorts below SMA5.
    if above_sma5 and score > 0.0:
        score *= 0.3
    elif (not above_sma5) and score < 0.0:
        score *= 0.3

    extreme = rsi2 < 5.0 or rsi2 > 95.0
    in_zone = (trend == 1 and rsi2 < 10.0) or (trend == -1 and rsi2 > 90.0)
    confidence = 0.5 + 0.3 * (1.0 if extreme else 0.0) + 0.2 * (1.0 if in_zone else 0.0)
    if _is_crypto(ctx):
        confidence -= 0.15
    confidence = clamp(confidence, 0.0, 1.0)

    horizons = _short_horizon_projection(ctx, score)
    trend_word = "uptrend (Close>SMA200)" if trend == 1 else "downtrend (Close<SMA200)"
    rationale = (
        f"RSI(2) at {rsi2:.1f} in a {trend_word}; "
        + (
            "deep oversold inside an uptrend -> mean-reversion long."
            if score > 5.0
            else "stretched overbought inside a downtrend -> mean-reversion short."
            if score < -5.0
            else "not in an actionable oversold/overbought zone (near neutral)."
        )
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "rsi2": rsi2,
            "trend": float(trend),
            "aboveSma5": 1.0 if above_sma5 else 0.0,
        },
        horizons=horizons,
    )


def _positions_connors_rsi2(
    closes: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """Vectorized Connors RSI(2) long/flat/short position series in ``[-1, 1]``.

    Per-bar rule (on-close): enter long (``+1``) when Close>SMA200 and RSI2<10,
    exit (back to flat) when Close>SMA5 or RSI2>65; mirror for shorts
    (Close<SMA200 and RSI2>90, cover when Close<SMA5 or RSI2<35). The position is
    carried forward between entry and exit. Defensive: short series → all-flat.
    """
    p = params or {}
    arm_lo = float(p.get("oversold_arm", 10.0))
    arm_hi = float(p.get("overbought_arm", 90.0))
    exit_long_rsi = float(p.get("rsi_exit_long", 65.0))
    exit_short_rsi = float(p.get("rsi_exit_short", 35.0))
    c = np.asarray(closes, dtype=np.float64).ravel()
    L = c.size
    if L < 3:
        return np.zeros(max(L, 0), dtype=np.float64)
    rsi2 = _rsi_series(c, 2)
    sma200 = technical.sma(c, 200)
    sma5 = technical.sma(c, 5)
    # Align (technical.sma drops non-positive prices; for synthetic positive
    # series lengths match, but guard anyway).
    n = min(L, rsi2.size, sma200.size, sma5.size)
    pos = np.zeros(L, dtype=np.float64)
    state = 0.0
    for t in range(L - n, L):
        rt = rsi2[t - (L - n)] if rsi2.size == n else rsi2[t]
        s200 = sma200[t - (L - n)] if sma200.size == n else sma200[t]
        s5 = sma5[t - (L - n)] if sma5.size == n else sma5[t]
        price = c[t]
        if state == 0.0:
            if price > s200 and rt < arm_lo:
                state = 1.0
            elif price < s200 and rt > arm_hi:
                state = -1.0
        elif state == 1.0:
            if price > s5 or rt > exit_long_rsi:
                state = 0.0
        elif state == -1.0:
            if price < s5 or rt < exit_short_rsi:
                state = 0.0
        pos[t] = state
    return pos


# ---------------------------------------------------------------------------
# 2. Connors Cumulative RSI(2) -- mean-reversion / Technical
# ---------------------------------------------------------------------------

_META_CONNORS_CUM = _meta_from_catalog(
    "connors-cumulative-rsi2",
    "Connors Cumulative RSI(2)",
    "Technical",
    (
        "Connors & Alvarez refinement (ch. 9): sum the last X RSI(2) readings so a "
        "single noisy value cannot trigger; entries require multiple consecutive "
        "oversold days. ~88% reported accuracy on SPY (1993-2008) with ~3.7-day holds."
    ),
    "CumRSI = RSI2_t + RSI2_{t-1}; long if Close>SMA200 and CumRSI<35; short if Close<SMA200 and CumRSI>165",
    ["price history (SMA200/SMA5 filters)", "RSI(2) (summed over X=2 days)"],
    _SRC_CONNORS_CUM,
)


def _build_connors_cumulative_rsi2(ctx: "AnalysisContext") -> StrategySignal:
    """Connors cumulative RSI(2): multi-day-confirmed oversold/overbought.

    Implements the catalog ``computeSignal`` exactly (X=2):

        CumRSI = RSI2_t + RSI2_{t-1}
        long:  if Close>SMA200 and CumRSI<35  -> signal = +100*clamp((35-CumRSI)/35,0,1)
        short: if Close<SMA200 and CumRSI>165 -> signal = -100*clamp((CumRSI-165)/35,0,1)
        else 0.
        confidence base 0.55, +0.25 if in the strongest decile, +0.2 trend agreement.
    """
    meta = _META_CONNORS_CUM
    c = _closes(ctx)
    if c.size < 4:
        return _neutral(meta, "need >= 4 closes for cumulative RSI(2)")
    rsi_series = _rsi_series(c, 2)
    if rsi_series.size < 2:
        return _neutral(meta, "RSI(2) series too short")
    cum_rsi = float(rsi_series[-1] + rsi_series[-2])
    trend = _trend_sign(c, 200)

    score = 0.0
    if trend == 1 and cum_rsi < 35.0:
        score = 100.0 * clamp((35.0 - cum_rsi) / 35.0, 0.0, 1.0)
    elif trend == -1 and cum_rsi > 165.0:
        score = -100.0 * clamp((cum_rsi - 165.0) / 35.0, 0.0, 1.0)
    else:
        score = 0.0

    # Strongest decile: deeply oversold (CumRSI<25) or deeply overbought (>175).
    strongest = cum_rsi < 25.0 or cum_rsi > 175.0
    in_zone = (trend == 1 and cum_rsi < 35.0) or (trend == -1 and cum_rsi > 165.0)
    confidence = 0.55 + 0.25 * (1.0 if strongest else 0.0) + 0.2 * (1.0 if in_zone else 0.0)
    if _is_crypto(ctx):
        confidence -= 0.1
    confidence = clamp(confidence, 0.0, 1.0)

    horizons = _short_horizon_projection(ctx, score)
    trend_word = "uptrend" if trend == 1 else "downtrend" if trend == -1 else "flat trend"
    rationale = (
        f"Cumulative RSI(2) (2-day sum) at {cum_rsi:.1f} in a {trend_word}; "
        + (
            "multi-day oversold confirmation inside an uptrend -> long."
            if score > 5.0
            else "multi-day overbought inside a downtrend -> short."
            if score < -5.0
            else "no multi-day extreme confirmed (neutral)."
        )
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "cumRsi2": cum_rsi,
            "rsi2": float(rsi_series[-1]),
            "trend": float(trend),
        },
        horizons=horizons,
    )


def _positions_connors_cumulative_rsi2(
    closes: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """Vectorized cumulative-RSI(2) long/flat/short position series in ``[-1,1]``.

    Enter long when Close>SMA200 and CumRSI(X=2)<35; exit when CumRSI>65 or
    Close>SMA5. Mirror for shorts (Close<SMA200, CumRSI>165; cover at CumRSI<135
    or Close<SMA5). Defensive: short series → all-flat.
    """
    p = params or {}
    long_entry = float(p.get("long_entry_cum", 35.0))
    long_exit = float(p.get("long_exit_cum", 65.0))
    short_entry = float(p.get("short_entry_cum", 165.0))
    short_exit = float(p.get("short_exit_cum", 135.0))
    c = np.asarray(closes, dtype=np.float64).ravel()
    L = c.size
    if L < 4:
        return np.zeros(max(L, 0), dtype=np.float64)
    rsi2 = _rsi_series(c, 2)
    sma200 = technical.sma(c, 200)
    sma5 = technical.sma(c, 5)
    pos = np.zeros(L, dtype=np.float64)
    state = 0.0
    for t in range(1, L):
        cum = float(rsi2[t] + rsi2[t - 1]) if rsi2.size == L else 100.0
        s200 = sma200[t] if sma200.size == L else c[t]
        s5 = sma5[t] if sma5.size == L else c[t]
        price = c[t]
        if state == 0.0:
            if price > s200 and cum < long_entry:
                state = 1.0
            elif price < s200 and cum > short_entry:
                state = -1.0
        elif state == 1.0:
            if cum > long_exit or price > s5:
                state = 0.0
        elif state == -1.0:
            if cum < short_exit or price < s5:
                state = 0.0
        pos[t] = state
    return pos


# ---------------------------------------------------------------------------
# 3. Rolling Z-Score Mean Reversion -- Statistical
# ---------------------------------------------------------------------------

_META_ZSCORE = _meta_from_catalog(
    "zscore-reversion",
    "Rolling Z-Score Mean Reversion",
    "Statistical",
    (
        "Canonical statistical mean-reversion: standardize price by its rolling "
        "mean and stdev; enter when |z|>~2 (price stretched), exit as z reverts "
        "toward 0. The engine behind Bollinger %B; the spread version underlies "
        "pairs trading."
    ),
    "z = (Close - SMA(Close,N)) / std(Close,N), N=20; signal = clamp(-z/3*100, -100, 100)",
    ["price history (rolling N=20 mean/std)"],
    _SRC_ZSCORE,
)


def _build_zscore_reversion(ctx: "AnalysisContext") -> StrategySignal:
    """Rolling z-score mean reversion (N=20).

    Implements the catalog ``computeSignal`` exactly:

        signal = clamp(-z/3*100, -100, 100)   (z=-2 -> +66.7, -3 -> +100, +2 -> -66.7)
        only |z|>=2 is conviction; |z|<1 is noise.
        confidence = clamp(0.3 + 0.2*min(|z|/3,1) + 0.2*reversion_quality, 0, 0.9)
        where reversion_quality comes from negative lag-1 autocorrelation;
        trending/positive-autocorr assets (some crypto) get lower confidence.
    """
    meta = _META_ZSCORE
    c = _closes(ctx)
    if c.size < 3:
        return _neutral(meta, "need >= 3 closes for a rolling z-score")
    z = technical.zscore(c, n=20)
    score = clamp(-z / 3.0 * 100.0, -100.0, 100.0)

    # reversion_quality in [0,1]: 1 when lag-1 autocorr of returns is strongly
    # negative (clean mean reversion), 0 when positive (trending/persistent).
    rets = returns.simple_returns(c)
    rho = _lag1_autocorr(rets)
    reversion_quality = clamp(-rho, 0.0, 1.0)  # negative rho -> positive quality
    confidence = clamp(
        0.3 + 0.2 * min(abs(z) / 3.0, 1.0) + 0.2 * reversion_quality, 0.0, 0.9
    )

    horizons = _short_horizon_projection(ctx, score)
    if abs(z) >= 2.0:
        tone = "stretched (actionable reversion)"
    elif abs(z) >= 1.0:
        tone = "mildly stretched"
    else:
        tone = "near its mean (noise zone)"
    rationale = (
        f"Price z-score {z:+.2f} over a 20-day window ({tone}); "
        f"{'depressed below the mean (bullish)' if z < 0 else 'above the mean (bearish)'}. "
        f"Lag-1 return autocorr {rho:+.2f} -> reversion quality {reversion_quality:.2f}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "zscore": z,
            "lag1Autocorr": rho,
            "reversionQuality": reversion_quality,
        },
        horizons=horizons,
    )


def _positions_zscore_reversion(
    closes: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """Vectorized z-score reversion long/flat/short position series in ``[-1,1]``.

    Enter long when z<=-entry_z; exit when z>=exit_z (revert to mean). Enter short
    when z>=+entry_z; cover when z<=exit_z. N=20 window. Defensive: short → flat.
    """
    p = params or {}
    entry_z = float(p.get("entry_z", 2.0))
    exit_z = float(p.get("exit_z", 0.0))
    c = np.asarray(closes, dtype=np.float64).ravel()
    L = c.size
    if L < 3:
        return np.zeros(max(L, 0), dtype=np.float64)
    z = _zscore_series(c, int(p.get("window", 20)))
    pos = np.zeros(L, dtype=np.float64)
    state = 0.0
    for t in range(L):
        zt = z[t]
        if state == 0.0:
            if zt <= -entry_z:
                state = 1.0
            elif zt >= entry_z:
                state = -1.0
        elif state == 1.0:
            if zt >= exit_z:
                state = 0.0
        elif state == -1.0:
            if zt <= exit_z:
                state = 0.0
        pos[t] = state
    return pos


# ---------------------------------------------------------------------------
# 4. Pairs Trading (Statistical Arbitrage) -- Statistical
# ---------------------------------------------------------------------------

_META_PAIRS = _meta_from_catalog(
    "pairs-trading",
    "Pairs Trading (Statistical Arbitrage)",
    "Statistical",
    (
        "Market-neutral relative-value: find two historically co-moving assets, "
        "trade the spread when it diverges (short the rich, long the cheap), exit "
        "when it converges. Gatev, Goetzmann & Rouwenhorst (2006) document "
        "persistent profits."
    ),
    "spread = log(self) - beta*log(partner); z = zscore(spread,N); signal = clamp(-z/2.5*100, -100, 100)",
    ["self price history", "best-correlated universe peer price history"],
    _SRC_PAIRS,
)


def _universe_closes(ctx: "AnalysisContext") -> dict[str, np.ndarray]:
    """Return ``{symbol: closes}`` for universe peers, defensively.

    The V2 engine builds a :class:`UniverseStats`; this looks for a ``closes``
    mapping on it (the natural place to expose per-symbol price series for the
    cross-sectional pairs strategy). Returns an empty dict when unavailable so
    the builder degrades to a no-tradable-pair neutral signal.
    """
    uni = getattr(ctx, "universe", None)
    if uni is None:
        return {}
    for attr in ("closes", "closes_by_symbol", "price_series", "histories"):
        m = getattr(uni, attr, None)
        if isinstance(m, dict) and m:
            out: dict[str, np.ndarray] = {}
            for k, v in m.items():
                arr = np.asarray(v, dtype=np.float64).ravel()
                arr = arr[np.isfinite(arr) & (arr > 0.0)]
                if arr.size >= 30:
                    out[str(k).upper()] = arr
            if out:
                return out
    return {}


def _build_pairs_trading(ctx: "AnalysisContext") -> StrategySignal:
    """Pairs trading: spread z-score of the asset vs its best-correlated peer.

    Implements the catalog ``computeSignal``:

        - find the asset's best-correlated partner in the universe;
        - hedge ratio beta = OLS of log(self) on log(partner);
        - spread = log(price_self) - beta*log(price_partner); z = zscore(spread,N);
        - signal = clamp(-z/2.5*100, -100, 100)  (self cheap vs partner -> bullish);
        - confidence = clamp(0.3 + 0.3*pair_correlation + 0.2*min(|z|/2.5,1), 0, 0.85);
        - zero confidence if no partner with correlation>0.7 (no tradable pair).
    """
    meta = _META_PAIRS
    self_sym = str(getattr(ctx.asset, "symbol", "")).upper()
    c = _closes(ctx)
    c = c[c > 0.0]
    if c.size < 30:
        return _neutral(meta, "need >= 30 self closes")

    peers = _universe_closes(ctx)
    # Drop the self symbol from candidate partners.
    peers.pop(self_sym, None)
    if not peers:
        # No cross-sectional data available -> no tradable pair (zero conviction).
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.0,
            "No universe peer data available to form a tradable pair (market-neutral spread not computable).",
            meta.formula,
            metrics={"pairCorrelation": 0.0, "spreadZ": 0.0},
            horizons=[],
        )

    self_log = np.log(c)
    best_corr = -2.0
    best_sym = ""
    best_partner_log = np.empty(0)
    self_ret = np.diff(self_log)
    for sym, pc in peers.items():
        n = min(self_log.size, pc.size)
        if n < 30:
            continue
        a = self_log[-n:]
        b = np.log(pc[-n:])
        ar = np.diff(a)
        br = np.diff(b)
        if ar.size < 2 or br.size < 2:
            continue
        sa = float(np.std(ar))
        sb = float(np.std(br))
        if sa <= 0.0 or sb <= 0.0:
            continue
        corr = float(np.mean((ar - ar.mean()) * (br - br.mean())) / (sa * sb))
        if math.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_sym = sym
            best_partner_log = b

    if best_sym == "" or not math.isfinite(best_corr) or best_corr <= 0.7:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.0,
            (
                f"No universe peer is sufficiently co-moving with {self_sym} "
                f"(best correlation {max(best_corr, 0.0):.2f} <= 0.70) -> no tradable pair."
            ),
            meta.formula,
            metrics={"pairCorrelation": clamp(max(best_corr, 0.0), 0.0, 1.0), "spreadZ": 0.0},
            horizons=[],
        )

    n = min(self_log.size, best_partner_log.size)
    a = self_log[-n:]
    b = best_partner_log[-n:]
    # Hedge ratio beta via OLS of a on b (with intercept).
    bb_var = float(np.var(b))
    if bb_var <= 0.0:
        beta = 1.0
    else:
        beta = float(np.cov(a, b, ddof=0)[0, 1] / bb_var)
    if not math.isfinite(beta):
        beta = 1.0
    spread = a - beta * b
    # z-score the spread over the last N (use 60, clamp to available).
    window = min(60, spread.size)
    win = spread[-window:]
    mu = float(np.mean(win))
    sigma = float(np.std(win))
    if sigma <= 0.0 or not math.isfinite(sigma):
        z = 0.0
    else:
        z = clamp((float(spread[-1]) - mu) / sigma, -10.0, 10.0)

    score = clamp(-z / 2.5 * 100.0, -100.0, 100.0)
    confidence = clamp(
        0.3 + 0.3 * clamp(best_corr, 0.0, 1.0) + 0.2 * min(abs(z) / 2.5, 1.0),
        0.0,
        0.85,
    )

    rationale = (
        f"Spread of {self_sym} vs best-correlated peer {best_sym} "
        f"(corr {best_corr:.2f}, hedge beta {beta:.2f}) sits at z {z:+.2f}; "
        + (
            f"{self_sym} is cheap relative to {best_sym} -> long the spread (bullish on {self_sym})."
            if z < 0
            else f"{self_sym} is rich relative to {best_sym} -> short the spread (bearish on {self_sym})."
            if z > 0
            else "spread is at fair value (neutral)."
        )
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "spreadZ": z,
            "hedgeBeta": beta,
            "pairCorrelation": clamp(best_corr, 0.0, 1.0),
        },
        horizons=[],  # market-neutral relative value: no per-asset drift projection
    )


# ---------------------------------------------------------------------------
# 5. Bollinger Band Squeeze (Volatility Breakout) -- Technical
# ---------------------------------------------------------------------------

_META_BB_SQUEEZE = _meta_from_catalog(
    "bollinger-squeeze",
    "Bollinger Band Squeeze (Volatility Breakout)",
    "Technical",
    (
        "Bollinger's 'The Squeeze': BandWidth contracting to a multi-month low "
        "signals a low-volatility coil that precedes high-volatility expansion "
        "('low vol begets high vol'). The first band break after a squeeze signals "
        "direction; beware the 'head fake'."
    ),
    "BandWidth=(UpperBB-LowerBB)/MiddleBB; squeeze=1-percentile(BandWidth,125); signal=+/-100*squeeze on a band break",
    ["price history", "volume", "Bollinger(20,2) bands & BandWidth"],
    _SRC_BB_SQUEEZE,
)


def _bandwidth_series(closes: np.ndarray, n: int = 20, k: float = 2.0) -> np.ndarray:
    """Rolling Bollinger BandWidth ``(upper-lower)/mid`` per bar (defensive)."""
    arr = np.asarray(closes, dtype=np.float64).ravel()
    L = arr.size
    if L == 0:
        return np.empty(0, dtype=np.float64)
    window = max(2, int(n))
    out = np.zeros(L, dtype=np.float64)
    for t in range(L):
        start = max(0, t - window + 1)
        win = arr[start : t + 1]
        if win.size < 2:
            out[t] = 0.0
            continue
        mid = float(np.mean(win))
        sigma = float(np.std(win))
        if mid <= 0.0 or not math.isfinite(mid):
            out[t] = 0.0
        else:
            out[t] = (2.0 * float(k) * sigma) / mid
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _build_bollinger_squeeze(ctx: "AnalysisContext") -> StrategySignal:
    """Bollinger squeeze + volatility breakout.

    Implements the catalog ``computeSignal``:

        squeeze_strength = 1 - percentile(BandWidth over 125 days)
        signal = +100*squeeze_strength when Close>UpperBB; -100*squeeze_strength
                 when Close<LowerBB; in a squeeze with no break -> 0 (armed).
        Multiply by min(1, vol/(1.5*avgvol20)).
        confidence = 0.4 + 0.3*squeeze_strength + 0.2*volume_confirm
                     + 0.1*(break aligns with SMA200).
    """
    meta = _META_BB_SQUEEZE
    c = _closes(ctx)
    if c.size < 5:
        return _neutral(meta, "need >= 5 closes for Bollinger bands")
    mid, upper, lower, _pb = technical.bollinger(c, n=20, k=2.0)
    bw_series = _bandwidth_series(c, 20, 2.0)
    look = min(125, bw_series.size)
    recent_bw = bw_series[-look:]
    cur_bw = float(bw_series[-1])
    # percentile of the current BandWidth within the lookback (0..1).
    if recent_bw.size > 1:
        pct = float(np.mean(recent_bw <= cur_bw))
    else:
        pct = 0.5
    squeeze_strength = clamp(1.0 - pct, 0.0, 1.0)

    last = float(c[-1])
    breakout = 0
    if upper > lower:
        if last > upper:
            breakout = 1
        elif last < lower:
            breakout = -1

    # Volume confirmation factor.
    vols = _volumes(ctx, c.size)
    if vols.size >= 21:
        avg_vol20 = float(np.mean(vols[-21:-1])) if vols.size > 21 else float(np.mean(vols[:-1]))
    else:
        avg_vol20 = float(np.mean(vols)) if vols.size else 0.0
    cur_vol = float(vols[-1]) if vols.size else 0.0
    if avg_vol20 > 0.0:
        vol_ratio = cur_vol / avg_vol20
        vol_factor = clamp(vol_ratio / 1.5, 0.0, 1.0)
        volume_confirm = clamp(min(1.0, vol_ratio / 1.5), 0.0, 1.0)
    else:
        # No real volume series available -> assume neutral confirmation.
        vol_factor = 1.0
        volume_confirm = 0.5

    score = 0.0
    if breakout == 1:
        score = 100.0 * squeeze_strength * vol_factor
    elif breakout == -1:
        score = -100.0 * squeeze_strength * vol_factor
    # else: in/just-after a squeeze with no break -> 0 ("armed").

    # SMA200 alignment of the break.
    trend = _trend_sign(c, 200)
    aligns = 1.0 if (breakout != 0 and breakout == trend) else 0.0

    confidence = clamp(
        0.4 + 0.3 * squeeze_strength + 0.2 * volume_confirm + 0.1 * aligns, 0.0, 1.0
    )

    horizons = _short_horizon_projection(ctx, score)
    if breakout == 1:
        state = "broke above the upper band on a squeeze -> bullish volatility expansion"
    elif breakout == -1:
        state = "broke below the lower band on a squeeze -> bearish volatility expansion"
    elif squeeze_strength > 0.5:
        state = "coiled in a squeeze (armed, no break yet)"
    else:
        state = "no squeeze (bands not contracted)"
    rationale = (
        f"BandWidth {cur_bw:.3f} is at the {pct * 100:.0f}th percentile of the last "
        f"{look} days (squeeze strength {squeeze_strength:.2f}); price {state}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "bandWidth": cur_bw,
            "bandWidthPercentile": pct,
            "squeezeStrength": squeeze_strength,
            "breakout": float(breakout),
            "volumeConfirm": volume_confirm,
        },
        horizons=horizons,
    )


def _positions_bollinger_squeeze(
    closes: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """Vectorized squeeze-breakout position series in ``[-1, 1]``.

    Go long after a squeeze (BandWidth <= 20th percentile over 125 bars) when
    Close>UpperBB; go short on Close<LowerBB; flatten back to the middle band
    when price re-enters the channel (Close crosses the middle SMA). Defensive:
    short series → all-flat.
    """
    p = params or {}
    pctile = float(p.get("squeeze_percentile", 0.2))
    c = np.asarray(closes, dtype=np.float64).ravel()
    L = c.size
    if L < 5:
        return np.zeros(max(L, 0), dtype=np.float64)
    bw = _bandwidth_series(c, int(p.get("bb_period", 20)), float(p.get("bb_stdev_mult", 2.0)))
    sma20 = technical.sma(c, int(p.get("bb_period", 20)))
    pos = np.zeros(L, dtype=np.float64)
    state = 0.0
    look = int(p.get("bandwidth_lookback", 125))
    for t in range(L):
        start = max(0, t - look + 1)
        win = bw[start : t + 1]
        if win.size > 1:
            rank = float(np.mean(win <= bw[t]))
        else:
            rank = 0.5
        squeezed = rank <= pctile
        mid = sma20[t] if sma20.size == L else c[t]
        win_p = c[max(0, t - int(p.get("bb_period", 20)) + 1) : t + 1]
        sigma = float(np.std(win_p)) if win_p.size > 1 else 0.0
        upper = mid + float(p.get("bb_stdev_mult", 2.0)) * sigma
        lower = mid - float(p.get("bb_stdev_mult", 2.0)) * sigma
        price = c[t]
        if state == 0.0:
            if squeezed and price > upper:
                state = 1.0
            elif squeezed and price < lower:
                state = -1.0
        elif state == 1.0:
            if price < mid:
                state = 0.0
        elif state == -1.0:
            if price > mid:
                state = 0.0
        pos[t] = state
    return pos


# ---------------------------------------------------------------------------
# 6. Stochastic Oscillator (%K/%D) Reversion -- Technical
# ---------------------------------------------------------------------------

_META_STOCH = _meta_from_catalog(
    "stochastic-oscillator",
    "Stochastic Oscillator (%K/%D) Reversion",
    "Technical",
    (
        "George Lane's Stochastic measures close relative to the recent high-low "
        "range. Mean-reversion: buy when %K/%D cross up out of oversold (<20), "
        "sell when they cross down out of overbought (>80). Lane stressed "
        "crossovers and divergence over bare threshold touches."
    ),
    "slow %K,%D(14,3,3); score=(50-%K)/50*100; +-15 crossover boost in zone; cap counter-trend at +/-40",
    ["high/low/close history", "slow %K / %D"],
    _SRC_STOCH,
)


def _build_stochastic_oscillator(ctx: "AnalysisContext") -> StrategySignal:
    """Stochastic %K/%D reversion.

    Implements the catalog ``computeSignal``:

        score_level = (50-%K)/50*100   (%K=0 -> +100, 100 -> -100)
        crossover boost: +15 if %K>%D in the oversold zone, -15 if %K<%D in
            the overbought zone.
        score = clamp(score_level + adj, -100, 100); counter-trend capped at +-40.
        confidence = 0.4 + 0.25*(in-zone extreme) + 0.2*(crossover confirmed)
                     + 0.15*(trend agreement).
    """
    meta = _META_STOCH
    h, l, c = _ohlc(ctx)
    if c.size < 3:
        return _neutral(meta, "need >= 3 bars for the stochastic")
    pk, pd = indicators.stochastic(h, l, c, k=14, d=3)
    score_level = (50.0 - pk) / 50.0 * 100.0

    in_oversold = pk < 20.0
    in_overbought = pk > 80.0
    crossover = 0.0
    if in_oversold and pk > pd:
        crossover = 15.0
    elif in_overbought and pk < pd:
        crossover = -15.0
    score = clamp(score_level + crossover, -100.0, 100.0)

    # Trend gate: cap counter-trend reads at +-40.
    trend = _trend_sign(c, 200)
    if trend == -1 and score > 40.0:
        score = 40.0
    elif trend == 1 and score < -40.0:
        score = -40.0

    in_zone_extreme = in_oversold or in_overbought
    crossover_confirmed = crossover != 0.0
    trend_agreement = 1.0 if (
        (score > 0 and trend == 1) or (score < 0 and trend == -1)
    ) else 0.0
    confidence = (
        0.4
        + 0.25 * (1.0 if in_zone_extreme else 0.0)
        + 0.2 * (1.0 if crossover_confirmed else 0.0)
        + 0.15 * trend_agreement
    )
    if _is_crypto(ctx):
        confidence -= 0.1
    confidence = clamp(confidence, 0.0, 1.0)

    horizons = _short_horizon_projection(ctx, score)
    if in_oversold:
        tone = "oversold (<20)"
    elif in_overbought:
        tone = "overbought (>80)"
    else:
        tone = "mid-range"
    rationale = (
        f"Slow stochastic %K={pk:.0f}/%D={pd:.0f} ({tone}); "
        + (
            "%K crossed above %D out of oversold -> mean-reversion long."
            if crossover > 0
            else "%K crossed below %D out of overbought -> mean-reversion short."
            if crossover < 0
            else "range position implies a "
            + ("bullish" if score > 0 else "bearish" if score < 0 else "neutral")
            + " reversion tilt."
        )
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"percentK": pk, "percentD": pd, "trend": float(trend)},
        horizons=horizons,
    )


def _positions_stochastic_oscillator(
    closes: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """Vectorized slow-stochastic reversion position series in ``[-1, 1]``.

    Long when slow %K crosses above %D while both <20; exit when %K>80 or %K
    crosses below %D in overbought. Mirror for shorts. Defensive: short → flat.
    Highs/lows fall back to closes when absent.
    """
    p = params or {}
    oversold = float(p.get("oversold", 20.0))
    overbought = float(p.get("overbought", 80.0))
    c = np.asarray(closes, dtype=np.float64).ravel()
    L = c.size
    if L < 3:
        return np.zeros(max(L, 0), dtype=np.float64)
    h = np.asarray(highs, dtype=np.float64).ravel() if highs is not None else c
    l = np.asarray(lows, dtype=np.float64).ravel() if lows is not None else c
    if h.size != L:
        h = c
    if l.size != L:
        l = c
    hi = np.maximum.reduce([h, l, c])
    lo = np.minimum.reduce([h, l, c])

    k_win = int(p.get("k_period", 14))
    d_smooth = int(p.get("d_period", 3))
    # Build the full slow %K / %D series.
    fast_k = np.full(L, 50.0, dtype=np.float64)
    for t in range(L):
        start = max(0, t - k_win + 1)
        hh = float(np.max(hi[start : t + 1]))
        ll = float(np.min(lo[start : t + 1]))
        rng = hh - ll
        if rng > 1e-12 and math.isfinite(rng):
            fast_k[t] = min(100.0, max(0.0, (float(c[t]) - ll) / rng * 100.0))
    slow_k = _sma_generic(fast_k, d_smooth)
    pd = _sma_generic(slow_k, d_smooth)

    pos = np.zeros(L, dtype=np.float64)
    state = 0.0
    for t in range(1, L):
        kt, dt = slow_k[t], pd[t]
        kp, dp = slow_k[t - 1], pd[t - 1]
        cross_up = kp <= dp and kt > dt
        cross_dn = kp >= dp and kt < dt
        if state == 0.0:
            if cross_up and kt < oversold:
                state = 1.0
            elif cross_dn and kt > overbought:
                state = -1.0
        elif state == 1.0:
            if kt > overbought or (cross_dn and kt > overbought):
                state = 0.0
        elif state == -1.0:
            if kt < oversold or (cross_up and kt < oversold):
                state = 0.0
        pos[t] = state
    return pos


def _sma_generic(values: np.ndarray, n: int) -> np.ndarray:
    """Trailing SMA of a general (possibly non-positive) finite series."""
    L = values.size
    if L == 0:
        return np.empty(0, dtype=np.float64)
    window = max(1, min(int(n), L))
    cumsum = np.cumsum(values)
    out = np.empty(L, dtype=np.float64)
    for i in range(min(window - 1, L)):
        out[i] = cumsum[i] / (i + 1)
    if L >= window:
        rolling = cumsum.copy()
        rolling[window:] = cumsum[window:] - cumsum[:-window]
        out[window - 1 :] = rolling[window - 1 :] / window
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# 7. Williams %R Reversion -- Technical
# ---------------------------------------------------------------------------

_META_WILLIAMS = _meta_from_catalog(
    "williams-r",
    "Williams %R Reversion",
    "Technical",
    (
        "Larry Williams' %R bounded oscillator (0 to -100) shows where close sits "
        "in the recent high-low range. Mean-reversion: buy oversold (<-80), sell "
        "overbought (>-20), exit toward midline. Effectively an inverted Fast "
        "Stochastic %K."
    ),
    "%R(14); score=(-50-%R)/50*100 (-100->+100, 0->-100, -50->0); cap bullish at +40 if Close<SMA200",
    ["high/low/close history", "Williams %R(14)"],
    _SRC_WILLIAMS,
)


def _build_williams_r(ctx: "AnalysisContext") -> StrategySignal:
    """Williams %R reversion.

    Implements the catalog ``computeSignal``:

        score = (-50-%R)/50*100   (%R=-100 -> +100 bullish, 0 -> -100, -50 -> 0)
        if Close<SMA200 cap bullish at +40.
        confidence = 0.4 + 0.3*(|%R+50|/50) + 0.2*(crossover turn confirmed)
                     + 0.1*(trend agreement); crypto -0.1.
    """
    meta = _META_WILLIAMS
    h, l, c = _ohlc(ctx)
    if c.size < 3:
        return _neutral(meta, "need >= 3 bars for Williams %R")
    wr = indicators.williams_r(h, l, c, n=14)
    score = (-50.0 - wr) / 50.0 * 100.0
    score = clamp(score, -100.0, 100.0)

    trend = _trend_sign(c, 200)
    if trend == -1 and score > 40.0:
        score = 40.0

    # Crossover turn-up confirmation: previous bar's %R below -80 and current
    # turning up (above the previous), or mirror at the top.
    turn_confirmed = 0.0
    if c.size >= 4:
        prev_wr = indicators.williams_r(h[:-1], l[:-1], c[:-1], n=14)
        if prev_wr <= -80.0 and wr > prev_wr:
            turn_confirmed = 1.0
        elif prev_wr >= -20.0 and wr < prev_wr:
            turn_confirmed = 1.0

    trend_agreement = 1.0 if (
        (score > 0 and trend == 1) or (score < 0 and trend == -1)
    ) else 0.0
    confidence = (
        0.4
        + 0.3 * (abs(wr + 50.0) / 50.0)
        + 0.2 * turn_confirmed
        + 0.1 * trend_agreement
    )
    if _is_crypto(ctx):
        confidence -= 0.1
    confidence = clamp(confidence, 0.0, 1.0)

    horizons = _short_horizon_projection(ctx, score)
    if wr <= -80.0:
        tone = "oversold (<=-80)"
    elif wr >= -20.0:
        tone = "overbought (>=-20)"
    else:
        tone = "mid-range"
    rationale = (
        f"Williams %R(14) at {wr:.0f} ({tone}); "
        f"{'depressed -> mean-reversion long' if score > 0 else 'stretched -> mean-reversion short' if score < 0 else 'neutral'}"
        + (" (capped vs a downtrend)." if (trend == -1 and score >= 40.0) else ".")
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"williamsR": wr, "trend": float(trend), "turnConfirmed": turn_confirmed},
        horizons=horizons,
    )


def _positions_williams_r(
    closes: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """Vectorized Williams %R reversion position series in ``[-1, 1]``.

    Long when %R turns up out of oversold (crosses above -80); exit at the -50
    midline or -20. Mirror for shorts (turns down from overbought). Defensive:
    short series → flat; highs/lows fall back to closes.
    """
    p = params or {}
    oversold = float(p.get("oversold", -80.0))
    overbought = float(p.get("overbought", -20.0))
    exit_mid = float(p.get("exit_mid", -50.0))
    n = int(p.get("wr_period", 14))
    c = np.asarray(closes, dtype=np.float64).ravel()
    L = c.size
    if L < 3:
        return np.zeros(max(L, 0), dtype=np.float64)
    h = np.asarray(highs, dtype=np.float64).ravel() if highs is not None else c
    l = np.asarray(lows, dtype=np.float64).ravel() if lows is not None else c
    if h.size != L:
        h = c
    if l.size != L:
        l = c
    hi = np.maximum.reduce([h, l, c])
    lo = np.minimum.reduce([h, l, c])

    wr = np.full(L, -50.0, dtype=np.float64)
    for t in range(L):
        start = max(0, t - n + 1)
        hh = float(np.max(hi[start : t + 1]))
        ll = float(np.min(lo[start : t + 1]))
        rng = hh - ll
        if rng > 1e-12 and math.isfinite(rng):
            wr[t] = min(0.0, max(-100.0, (hh - float(c[t])) / rng * -100.0))

    pos = np.zeros(L, dtype=np.float64)
    state = 0.0
    for t in range(1, L):
        cross_up = wr[t - 1] <= oversold and wr[t] > oversold
        cross_dn = wr[t - 1] >= overbought and wr[t] < overbought
        if state == 0.0:
            if cross_up:
                state = 1.0
            elif cross_dn:
                state = -1.0
        elif state == 1.0:
            if wr[t] >= exit_mid:
                state = 0.0
        elif state == -1.0:
            if wr[t] <= exit_mid:
                state = 0.0
        pos[t] = state
    return pos


# ---------------------------------------------------------------------------
# 8. Commodity Channel Index (CCI) Reversion -- Technical
# ---------------------------------------------------------------------------

_META_CCI = _meta_from_catalog(
    "cci-reversion",
    "Commodity Channel Index (CCI) Reversion",
    "Technical",
    (
        "Donald Lambert's CCI (1980) measures how far typical price is from its MA "
        "in mean-deviation units, scaled so ~70-80% of values fall within +-100. "
        "Reversion variant: fade readings beyond +-100, targeting a return toward 0."
    ),
    "CCI(20)=(TP-SMA(TP))/(0.015*MeanDev); score=clamp(-CCI/200*100,-100,100); full only for cross-back through +/-100",
    ["high/low/close history (typical price)", "CCI(20)"],
    _SRC_CCI,
)


def _build_cci_reversion(ctx: "AnalysisContext") -> StrategySignal:
    """CCI(20) reversion.

    Implements the catalog ``computeSignal``:

        score = clamp(-CCI/200*100, -100, 100)  (CCI=-200 -> +100, -100 -> +50, +200 -> -100)
        high conviction only |CCI|>100; require a cross-back through +-100 for the
            full score else 60%.
        trend gate caps counter-trend at +-40.
        confidence = 0.4 + 0.3*min(|CCI|/200,1) + 0.2*(cross-back confirmed)
                     + 0.1*(trend agreement); crypto -0.1.
    """
    meta = _META_CCI
    h, l, c = _ohlc(ctx)
    if c.size < 3:
        return _neutral(meta, "need >= 3 bars for the CCI")
    cci_now = indicators.cci(h, l, c, n=20)
    raw_score = clamp(-cci_now / 200.0 * 100.0, -100.0, 100.0)

    # Cross-back-through +-100 confirmation: prior bar beyond the band and the
    # current bar moving back toward 0 across the +-100 line.
    cross_back = 0.0
    if c.size >= 4:
        cci_prev = indicators.cci(h[:-1], l[:-1], c[:-1], n=20)
        if cci_prev <= -100.0 and cci_now > cci_prev:
            cross_back = 1.0
        elif cci_prev >= 100.0 and cci_now < cci_prev:
            cross_back = 1.0

    # Full score only on a confirmed cross-back; otherwise 60% magnitude when
    # |CCI|>100 (still beyond the band but not yet confirmed reverting).
    if abs(cci_now) > 100.0 and cross_back == 0.0:
        score = raw_score * 0.6
    else:
        score = raw_score

    trend = _trend_sign(c, 200)
    if trend == -1 and score > 40.0:
        score = 40.0
    elif trend == 1 and score < -40.0:
        score = -40.0

    trend_agreement = 1.0 if (
        (score > 0 and trend == 1) or (score < 0 and trend == -1)
    ) else 0.0
    confidence = (
        0.4
        + 0.3 * min(abs(cci_now) / 200.0, 1.0)
        + 0.2 * cross_back
        + 0.1 * trend_agreement
    )
    if _is_crypto(ctx):
        confidence -= 0.1
    confidence = clamp(confidence, 0.0, 1.0)

    horizons = _short_horizon_projection(ctx, score)
    if cci_now <= -100.0:
        tone = "oversold (CCI<=-100)"
    elif cci_now >= 100.0:
        tone = "overbought (CCI>=+100)"
    else:
        tone = "inside the normal +-100 band"
    rationale = (
        f"CCI(20) at {cci_now:+.0f} ({tone}); "
        + (
            "confirmed cross-back toward the mean -> "
            if cross_back
            else "beyond the band but not yet confirmed reverting -> "
            if abs(cci_now) > 100.0
            else "near the mean -> "
        )
        + ("bullish reversion." if score > 5.0 else "bearish reversion." if score < -5.0 else "neutral.")
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"cci": cci_now, "crossBack": cross_back, "trend": float(trend)},
        horizons=horizons,
    )


def _positions_cci_reversion(
    closes: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """Vectorized CCI reversion position series in ``[-1, 1]``.

    Long when CCI turns up across -100 (crosses above -100 from below); exit at
    0. Short when CCI turns down across +100; cover at 0. Defensive: short series
    → flat; highs/lows fall back to closes.
    """
    p = params or {}
    n = int(p.get("cci_period", 20))
    oversold = float(p.get("oversold", -100.0))
    overbought = float(p.get("overbought", 100.0))
    exit_level = float(p.get("exit", 0.0))
    c = np.asarray(closes, dtype=np.float64).ravel()
    L = c.size
    if L < 3:
        return np.zeros(max(L, 0), dtype=np.float64)
    h = np.asarray(highs, dtype=np.float64).ravel() if highs is not None else c
    l = np.asarray(lows, dtype=np.float64).ravel() if lows is not None else c
    if h.size != L:
        h = c
    if l.size != L:
        l = c
    hi = np.maximum.reduce([h, l, c])
    lo = np.minimum.reduce([h, l, c])
    tp = (hi + lo + c) / 3.0

    cci_s = np.zeros(L, dtype=np.float64)
    for t in range(L):
        start = max(0, t - n + 1)
        win = tp[start : t + 1]
        mean_tp = float(np.mean(win))
        mean_dev = float(np.mean(np.abs(win - mean_tp)))
        if mean_dev > 1e-12 and math.isfinite(mean_dev):
            cci_s[t] = clamp((float(tp[t]) - mean_tp) / (0.015 * mean_dev), -500.0, 500.0)

    pos = np.zeros(L, dtype=np.float64)
    state = 0.0
    for t in range(1, L):
        cross_up = cci_s[t - 1] <= oversold and cci_s[t] > oversold
        cross_dn = cci_s[t - 1] >= overbought and cci_s[t] < overbought
        if state == 0.0:
            if cross_up:
                state = 1.0
            elif cross_dn:
                state = -1.0
        elif state == 1.0:
            if cci_s[t] >= exit_level:
                state = 0.0
        elif state == -1.0:
            if cci_s[t] <= exit_level:
                state = 0.0
        pos[t] = state
    return pos


# ---------------------------------------------------------------------------
# 9. Keltner Channel Mean Reversion (ATR bands) -- Technical
# ---------------------------------------------------------------------------

_META_KELTNER = _meta_from_catalog(
    "keltner-reversion",
    "Keltner Channel Mean Reversion (ATR bands)",
    "Technical",
    (
        "Keltner Channels (Chester Keltner, 1960; Raschke/Colby refinements) plot "
        "ATR-based bands around an EMA. In range-bound markets, outer-band touches "
        "flag overshoots that revert to the middle EMA. Because bands use ATR not "
        "close-stdev, they pair with Bollinger to distinguish reversion vs breakout."
    ),
    "EMA20, ATR(10); p=(Close-EMA20)/(2*ATR); score=clamp(-p*100,-100,100); squeeze guard scales by 0.3",
    ["high/low/close history", "EMA(20)", "ATR(10)", "Bollinger(20,2) for squeeze guard"],
    _SRC_KELTNER,
)


def _build_keltner_reversion(ctx: "AnalysisContext") -> StrategySignal:
    """Keltner (ATR-band) mean reversion with a TTM-squeeze guard.

    Implements the catalog ``computeSignal``:

        EMA20, ATR(10); p = (Close-EMA20)/(2*ATR)  (~-1 at lower band, +1 at upper)
        score = clamp(-p*100, -100, 100).
        squeeze guard: if Bollinger(20,2) width < Keltner width scale score by 0.3
            and flag the breakout regime.
        trend gate caps counter-trend at +-40.
        confidence = 0.4 + 0.3*min(|p|,1.5)/1.5 + 0.2*(1-squeeze_flag)
                     + 0.1*(trend agreement).
    """
    meta = _META_KELTNER
    h, l, c = _ohlc(ctx)
    if c.size < 3:
        return _neutral(meta, "need >= 3 bars for the Keltner channel")
    ema20 = technical.ema(c, 20)
    ema_last = _last(ema20, float(c[-1]))
    atr_series = indicators.atr(h, l, c, n=10)
    atr_last = max(0.0, _last(atr_series, 0.0))
    last = float(c[-1])

    if atr_last <= 0.0:
        p = 0.0
    else:
        p = (last - ema_last) / (2.0 * atr_last)
    score = clamp(-p * 100.0, -100.0, 100.0)

    # Squeeze guard: BB(20,2) width vs Keltner width (2*ATR). When the Bollinger
    # band sits inside the Keltner channel -> TTM squeeze (breakout likely).
    _mid, bb_up, bb_lo, _pb = technical.bollinger(c, n=20, k=2.0)
    bb_width = abs(bb_up - bb_lo)
    kc_width = 2.0 * (2.0 * atr_last)  # full Keltner channel width (upper-lower)
    squeeze_flag = 1.0 if (kc_width > 0.0 and bb_width < kc_width) else 0.0
    if squeeze_flag:
        score *= 0.3

    trend = _trend_sign(c, 200)
    if trend == -1 and score > 40.0:
        score = 40.0
    elif trend == 1 and score < -40.0:
        score = -40.0

    trend_agreement = 1.0 if (
        (score > 0 and trend == 1) or (score < 0 and trend == -1)
    ) else 0.0
    confidence = clamp(
        0.4 + 0.3 * (min(abs(p), 1.5) / 1.5) + 0.2 * (1.0 - squeeze_flag) + 0.1 * trend_agreement,
        0.0,
        1.0,
    )

    horizons = _short_horizon_projection(ctx, score)
    band_pos = "above the upper band" if p > 1.0 else "below the lower band" if p < -1.0 else "inside the channel"
    rationale = (
        f"Close is {p:+.2f} ATR-band units from its EMA20 ({band_pos}); "
        + (
            "TTM squeeze detected (Bollinger inside Keltner) -> reversion suppressed (breakout regime)."
            if squeeze_flag
            else "an overshoot expected to revert toward the EMA -> "
            + ("bullish." if score > 5.0 else "bearish." if score < -5.0 else "neutral.")
        )
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "bandPosition": _safe(p),
            "ema20": _safe(ema_last),
            "atr10": _safe(atr_last),
            "squeezeFlag": squeeze_flag,
            "trend": float(trend),
        },
        horizons=horizons,
    )


def _positions_keltner_reversion(
    closes: np.ndarray,
    highs: np.ndarray | None = None,
    lows: np.ndarray | None = None,
    volumes: np.ndarray | None = None,
    params: dict | None = None,
) -> np.ndarray:
    """Vectorized Keltner reversion position series in ``[-1, 1]``.

    Long when Close<=LowerKC (EMA20 - 2*ATR(10)); exit at the EMA20. Short at
    Close>=UpperKC; cover at EMA20. Defensive: short series → flat; highs/lows
    fall back to closes.
    """
    p = params or {}
    ema_n = int(p.get("ema_period", 20))
    atr_n = int(p.get("atr_period", 10))
    mult = float(p.get("atr_mult", 2.0))
    c = np.asarray(closes, dtype=np.float64).ravel()
    L = c.size
    if L < 3:
        return np.zeros(max(L, 0), dtype=np.float64)
    h = np.asarray(highs, dtype=np.float64).ravel() if highs is not None else c
    l = np.asarray(lows, dtype=np.float64).ravel() if lows is not None else c
    if h.size != L:
        h = c
    if l.size != L:
        l = c
    hi = np.maximum.reduce([h, l, c])
    lo = np.minimum.reduce([h, l, c])

    ema_s = technical.ema(c, ema_n)
    atr_s = indicators.atr(hi, lo, c, n=atr_n)
    pos = np.zeros(L, dtype=np.float64)
    state = 0.0
    for t in range(L):
        mid = ema_s[t] if ema_s.size == L else c[t]
        a = atr_s[t] if atr_s.size == L else 0.0
        upper = mid + mult * a
        lower = mid - mult * a
        price = c[t]
        if state == 0.0:
            if price <= lower:
                state = 1.0
            elif price >= upper:
                state = -1.0
        elif state == 1.0:
            if price >= mid:
                state = 0.0
        elif state == -1.0:
            if price <= mid:
                state = 0.0
        pos[t] = state
    return pos


# ---------------------------------------------------------------------------
# 10. On-Balance Volume Trend Confirmation -- Technical
# ---------------------------------------------------------------------------

_META_OBV = _meta_from_catalog(
    "obv-volume-trend",
    "On-Balance Volume Trend Confirmation",
    "Technical",
    (
        "Joe Granville's On-Balance Volume cumulates signed volume (add volume on "
        "up days, subtract on down days) to confirm price trends and spot "
        "divergences: OBV making new highs with price confirms the trend; OBV "
        "diverging warns of a reversal."
    ),
    "OBV signed-volume sum; score=clamp(sign(obv_slope)*min(|obv_slope_z|,2.5)*40,-100,100); halve on divergence",
    ["close + volume history", "OBV slope (20d) vs price slope (20d)"],
    _SRC_OBV,
)


def _build_obv_volume_trend(ctx: "AnalysisContext") -> StrategySignal:
    """On-Balance Volume trend confirmation / divergence.

    Implements the catalog ``computeSignal``:

        obv_slope = normalized slope of OBV over the last ~20 days.
        price_slope = close/close[-20] - 1.
        score = clamp(sign(obv_slope)*min(|obv_slope_z|,2.5)*40, -100, 100);
        divergence penalty: if sign(obv_slope)!=sign(price_slope) halve the
            magnitude and flag divergence.
        confidence = clamp(0.35 + 0.3*min(|obv_slope_z|/2.5,1)
                     + 0.2*(agreement?1:0), 0, 0.85). Requires volume.
    """
    meta = _META_OBV
    c = _closes(ctx)
    c = c[c > 0.0]
    if c.size < 5:
        return _neutral(meta, "need >= 5 closes for OBV")
    vols = _volumes(ctx, c.size)
    has_real_volume = getattr(ctx, "volumes", None) is not None

    obv_series = indicators.obv(c, vols)
    # Normalized OBV slope (the indicators helper already normalizes by mean
    # per-bar OBV change), which serves directly as a z-like dislocation rate.
    obv_slope_norm = indicators.obv_slope(c, vols, n=20)

    # obv_slope_z: standardize the recent OBV daily changes' trend. We derive a
    # z by comparing the recent 20-bar OBV change to the std of OBV daily deltas.
    look = min(20, obv_series.size - 1)
    if look >= 1:
        deltas = np.diff(obv_series)
        recent_change = float(obv_series[-1] - obv_series[-(look + 1)])
        std_delta = float(np.std(deltas)) if deltas.size else 0.0
        if std_delta > 0.0 and math.isfinite(std_delta):
            obv_slope_z = recent_change / (std_delta * math.sqrt(look))
        else:
            obv_slope_z = 0.0
    else:
        obv_slope_z = 0.0
    obv_slope_z = clamp(obv_slope_z, -10.0, 10.0)

    # Price slope over the same ~20-day window.
    pl = min(20, c.size - 1)
    if pl >= 1 and float(c[-(pl + 1)]) > 0.0:
        price_slope = float(c[-1]) / float(c[-(pl + 1)]) - 1.0
    else:
        price_slope = 0.0

    sign_obv = 1.0 if obv_slope_z > 0 else -1.0 if obv_slope_z < 0 else 0.0
    score = clamp(sign_obv * min(abs(obv_slope_z), 2.5) * 40.0, -100.0, 100.0)

    # Divergence penalty.
    sign_price = 1.0 if price_slope > 0 else -1.0 if price_slope < 0 else 0.0
    divergence = sign_obv != 0.0 and sign_price != 0.0 and sign_obv != sign_price
    agreement = sign_obv != 0.0 and sign_price != 0.0 and sign_obv == sign_price
    if divergence:
        score *= 0.5

    confidence = clamp(
        0.35 + 0.3 * min(abs(obv_slope_z) / 2.5, 1.0) + 0.2 * (1.0 if agreement else 0.0),
        0.0,
        0.85,
    )
    if not has_real_volume:
        # No real volume series -> OBV is a sign-of-price proxy; trim confidence.
        confidence = clamp(confidence * 0.6, 0.0, 0.5)

    horizons = _short_horizon_projection(ctx, score)
    if divergence:
        diag = "bearish divergence (OBV and price disagree) -> reversal warning, magnitude halved"
    elif agreement:
        diag = (
            "OBV confirms the price trend"
            + (" (accumulation -> bullish)" if score > 0 else " (distribution -> bearish)")
        )
    else:
        diag = "flat volume trend (no confirmation)"
    rationale = (
        f"OBV slope (20d) z={obv_slope_z:+.2f}, normalized slope {obv_slope_norm:+.2f}; "
        f"price 20d slope {price_slope * 100:+.1f}% -> {diag}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "obvSlopeZ": obv_slope_z,
            "obvSlopeNorm": obv_slope_norm,
            "priceSlope": price_slope,
            "divergence": 1.0 if divergence else 0.0,
        },
        horizons=horizons,
    )


# ---------------------------------------------------------------------------
# Shared neutral + short-horizon projection helpers
# ---------------------------------------------------------------------------

def _neutral(meta: StrategyMeta, reason: str) -> StrategySignal:
    """Neutral ``HOLD`` signal for a strategy that could not run (no raise)."""
    return make_signal(
        meta.id, meta.name, meta.category, 0.0, 0.1,
        f"Insufficient or degenerate data for {meta.name}: {reason}.",
        meta.formula,
        metrics={},
        horizons=[],
    )


def _short_horizon_projection(ctx: "AnalysisContext", score: float) -> list[dict]:
    """Project a short-horizon tactical tilt implied by a reversion score.

    The mean-reversion / technical timing strategies imply a *short-horizon*
    drift (1-5 day snap-back) proportional to the score's conviction, sized by
    the asset's realized daily volatility (the catalog's "use realized hold-period
    vol for sizing"). A flat/neutral score (|score|<~5) implies no drift, so an
    empty horizon list is returned and the engine blends only the projecting
    signals.

    The implied daily log-drift is conservative: a full +-100 score maps to a
    modest ~+-15 bps/day tilt so these tactical signals do not dominate the
    blended long-run projection.

    Args:
        ctx: The analysis context (for realized volatility).
        score: The strategy score in ``[-100, 100]``.

    Returns:
        A list of 5 horizon dicts (per :func:`returns.project_horizons`) when the
        score is actionable, else an empty list.
    """
    s = _safe(score, 0.0)
    if abs(s) < 5.0:
        return []
    closes = _closes(ctx)
    lr = returns.log_returns(closes)
    if lr.size == 0:
        sigma_daily = 1e-4
    else:
        sigma_daily = float(np.std(lr))
        if not math.isfinite(sigma_daily) or sigma_daily <= 0.0:
            sigma_daily = 1e-4
    # Conviction -> small daily drift: +-100 score -> ~+-0.0015 daily log drift.
    mu_daily = clamp((s / 100.0) * 0.0015, -0.0015, 0.0015)
    return returns.project_horizons(mu_daily, sigma_daily)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

#: All 10 builders keyed by id, in catalog priority order (P1, then P2s, then
#: P3s), each value a ``(StrategyMeta, builder_fn)`` tuple.
BUILDERS: dict[str, tuple[StrategyMeta, Callable[["AnalysisContext"], StrategySignal]]] = {
    "connors-rsi2": (_META_CONNORS_RSI2, _build_connors_rsi2),
    "connors-cumulative-rsi2": (_META_CONNORS_CUM, _build_connors_cumulative_rsi2),
    "zscore-reversion": (_META_ZSCORE, _build_zscore_reversion),
    "pairs-trading": (_META_PAIRS, _build_pairs_trading),
    "bollinger-squeeze": (_META_BB_SQUEEZE, _build_bollinger_squeeze),
    "stochastic-oscillator": (_META_STOCH, _build_stochastic_oscillator),
    "williams-r": (_META_WILLIAMS, _build_williams_r),
    "cci-reversion": (_META_CCI, _build_cci_reversion),
    "keltner-reversion": (_META_KELTNER, _build_keltner_reversion),
    "obv-volume-trend": (_META_OBV, _build_obv_volume_trend),
}

#: Vectorized per-bar position series (values in ``[-1, 1]``) for the
#: time-backtestable timing strategies in this module. ``pairs-trading`` is a
#: cross-sectional / market-neutral relative-value strategy and ``obv-volume-trend``
#: is a confirmation overlay rather than a standalone per-bar timing rule, so
#: neither exposes a single-asset position series here.
POSITION_FUNCS: dict[str, Callable] = {
    "connors-rsi2": _positions_connors_rsi2,
    "connors-cumulative-rsi2": _positions_connors_cumulative_rsi2,
    "zscore-reversion": _positions_zscore_reversion,
    "bollinger-squeeze": _positions_bollinger_squeeze,
    "stochastic-oscillator": _positions_stochastic_oscillator,
    "williams-r": _positions_williams_r,
    "cci-reversion": _positions_cci_reversion,
    "keltner-reversion": _positions_keltner_reversion,
}
