"""Momentum & trend strategy builders (V2 expansion).

This module implements the 15 momentum / trend strategies assigned to the
``builders_momentum_trend`` group of the V2 strategy expansion
(``docs/STRATEGIES-V2.md`` §5). Each ``computeSignal`` is taken verbatim from
``docs/research/strategy-catalog.json`` and turned into a builder with the frozen
signature ``(ctx) -> StrategySignal``.

Implemented ids::

    dual-momentum, tsmom, cross-sectional-momentum, 52w-high,
    relative-strength-rotation, frog-in-the-pan-momentum, faber-taa,
    donchian-turtle, golden-cross, dual-ma-crossover, supertrend, ichimoku,
    adx-trend-strength, ma-ribbon, absolute-momentum-overlay

Module exports
--------------
``BUILDERS``: ``dict[str, tuple[StrategyMeta, Callable[[ctx], StrategySignal]]]``
    One entry per id — the metadata (category mapped to the existing
    :data:`~app.schemas.StrategyCategory` literal; summary / sources carried from
    the catalog) plus the builder function.
``POSITION_FUNCS``: ``dict[str, Callable]``
    Vectorized position-series generators (``positions(closes, highs, lows,
    volumes, params) -> np.ndarray`` in ``[-1, 1]`` aligned to ``closes``) for
    the *backtestable* single-asset timing strategies here (tsmom, faber-taa,
    donchian-turtle, golden-cross, dual-ma-crossover, supertrend, ichimoku,
    adx-trend-strength, ma-ribbon, absolute-momentum-overlay). Cross-sectional
    strategies (dual-momentum, cross-sectional-momentum, 52w-high,
    relative-strength-rotation, frog-in-the-pan-momentum) are not per-bar
    single-asset backtestable and are intentionally absent.

Conventions (frozen)
--------------------
* Score in ``[-100, 100]`` (positive = bullish); confidence in ``[0, 1]``.
* Cross-sectional builders read ranks from ``ctx.universe`` (the
  :class:`~app.strategies.engine.UniverseStats` injected by the engine). When the
  universe is not yet available (e.g. an older context) they degrade to a
  defensive single-asset estimate from ``ctx.closes`` so they never raise.
* Indicator builders read OHLCV from ``ctx.closes`` / ``ctx.highs`` / ``ctx.lows``
  / ``ctx.volumes``; when highs/lows are unavailable the close series is reused.
* :class:`~app.strategies.engine.AnalysisContext` and ``UniverseStats`` are
  referenced under :data:`typing.TYPE_CHECKING` only, to avoid a circular import.

Every builder is numerically defensive: short / empty / NaN inputs collapse to
safe, finite, neutral readings rather than raising.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from app.quant import indicators, returns, technical
from app.schemas import StrategyMeta, StrategySignal
from app.strategies.base import clamp, make_signal

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from app.strategies.engine import AnalysisContext, UniverseStats

__all__ = ["BUILDERS", "POSITION_FUNCS"]


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

_TD: int = returns.TRADING_DAYS  # 252


def _clean(arr: np.ndarray | list[float] | None) -> np.ndarray:
    """Coerce an input to a finite 1-D ``float64`` array (positions preserved).

    Args:
        arr: Sequence of numbers, or ``None``.

    Returns:
        A 1-D ``float64`` array with NaN/inf replaced by ``0.0`` (empty if the
        input is ``None`` / empty).
    """
    if arr is None:
        return np.empty(0, dtype=np.float64)
    out = np.asarray(arr, dtype=np.float64).ravel()
    if out.size == 0:
        return out
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _closes(ctx: "AnalysisContext") -> np.ndarray:
    """Return the asset's close series as a finite ``float64`` array."""
    return _clean(getattr(ctx, "closes", None))


def _highs(ctx: "AnalysisContext") -> np.ndarray:
    """Return the asset's high series, falling back to closes when absent.

    The V2 engine adds ``highs`` to :class:`AnalysisContext`; an older context
    (or one missing the field) degrades gracefully to the close series so
    indicator builders still run.
    """
    h = _clean(getattr(ctx, "highs", None))
    if h.size:
        return h
    return _closes(ctx)


def _lows(ctx: "AnalysisContext") -> np.ndarray:
    """Return the asset's low series, falling back to closes when absent."""
    l = _clean(getattr(ctx, "lows", None))
    if l.size:
        return l
    return _closes(ctx)


def _volumes(ctx: "AnalysisContext") -> np.ndarray:
    """Return the asset's volume series (empty array when absent)."""
    return _clean(getattr(ctx, "volumes", None))


def _symbol(ctx: "AnalysisContext") -> str:
    """Return the upper-cased symbol for ``ctx`` (best effort)."""
    asset = getattr(ctx, "asset", None)
    sym = getattr(asset, "symbol", "") if asset is not None else ""
    return str(sym).strip().upper()


def _universe(ctx: "AnalysisContext") -> "UniverseStats | None":
    """Return the cross-sectional :class:`UniverseStats`, or ``None`` if absent."""
    return getattr(ctx, "universe", None)


def _rf_daily(ctx: "AnalysisContext") -> float:
    """Return the scalar daily risk-free rate (0.0 when missing / non-finite)."""
    rf = float(getattr(ctx, "rf_daily", 0.0) or 0.0)
    return rf if math.isfinite(rf) else 0.0


def _history_conf(closes: np.ndarray, need: int = _TD) -> float:
    """Confidence floor from history length: ``min(1, n / need)``.

    Args:
        closes: The price series.
        need: Number of bars for full confidence (default one year).

    Returns:
        A value in ``[0, 1]``.
    """
    n = int(closes.size)
    if need <= 0:
        return 1.0
    return clamp(n / float(need), 0.0, 1.0)


def _trailing_return(closes: np.ndarray, lookback: int) -> float:
    """Trailing total return over ``lookback`` bars: ``Close[-1]/Close[-1-lb]-1``.

    Falls back to the full-history return when fewer than ``lookback + 1`` bars
    are available. Returns ``0.0`` for degenerate input.

    Args:
        closes: Price series.
        lookback: Number of bars to look back.

    Returns:
        The trailing simple return (decimal), finite.
    """
    c = closes
    if c.size < 2:
        return 0.0
    lb = max(1, int(lookback))
    if c.size > lb:
        start = float(c[-(lb + 1)])
    else:
        start = float(c[0])
    end = float(c[-1])
    if start <= 0.0 or not math.isfinite(start):
        return 0.0
    r = end / start - 1.0
    return r if math.isfinite(r) else 0.0


def _ewma_annual_vol(closes: np.ndarray, lam: float = 0.94) -> float:
    """RiskMetrics EWMA annualized volatility of daily returns.

    ``sigma_annual = sqrt(252 * EWMA(r_daily^2, lambda))`` with the standard
    RiskMetrics decay ``lambda = 0.94``.

    Args:
        closes: Price series.
        lam: EWMA decay (default 0.94).

    Returns:
        Annualized volatility (decimal), floored to a tiny positive value.
    """
    r = returns.simple_returns(closes)
    if r.size == 0:
        return 1e-4
    lam = float(lam)
    if not math.isfinite(lam) or not (0.0 < lam < 1.0):
        lam = 0.94
    var = float(r[0] * r[0])
    for i in range(1, r.size):
        var = lam * var + (1.0 - lam) * float(r[i] * r[i])
    sigma_daily = math.sqrt(max(var, 0.0))
    sigma_annual = sigma_daily * math.sqrt(_TD)
    if not math.isfinite(sigma_annual) or sigma_annual <= 0.0:
        return 1e-4
    return sigma_annual


def _annual_vol(closes: np.ndarray) -> float:
    """Plain annualized volatility ``sqrt(252) * std(daily returns)`` (floored)."""
    r = returns.simple_returns(closes)
    if r.size == 0:
        return 1e-4
    sigma = float(np.std(r)) * math.sqrt(_TD)
    if not math.isfinite(sigma) or sigma <= 0.0:
        return 1e-4
    return sigma


def _drift_daily_from_annual(annual_return: float) -> float:
    """Convert an annual simple return to a daily log drift, clamped sanely.

    ``mu_daily = ln(1 + annual_return) / 252`` clamped to ``[-0.02, 0.02]`` so a
    pathological estimate cannot produce absurd horizon projections.

    Args:
        annual_return: Expected annual simple return (decimal).

    Returns:
        Daily log drift (finite), clamped to ``[-0.02, 0.02]``.
    """
    ar = float(annual_return)
    if not math.isfinite(ar) or 1.0 + ar <= 0.0:
        return 0.0
    mu = math.log1p(ar) / float(_TD)
    return clamp(mu, -0.02, 0.02)


def _project(mu_daily: float, sigma_daily: float) -> list[dict]:
    """Project the 5 horizons from a daily drift / vol (thin wrapper)."""
    return returns.project_horizons(mu_daily, sigma_daily)


def _xsec_z(
    universe: "UniverseStats | None",
    metric: str,
    symbol: str,
    fallback: float,
    all_vals: dict[str, float] | None = None,
) -> tuple[float, float, int]:
    """Cross-sectional z-score of ``symbol`` for ``metric`` across the universe.

    Reads the universe's per-symbol metric dict (e.g. ``momentum_12_1``), forms
    ``z = (x - mean) / std`` across all valid symbols, and returns the symbol's
    z together with the cross-sectional dispersion and the count of valid peers.

    Args:
        universe: The :class:`UniverseStats` (may be ``None``).
        metric: Name of the metric dict on the universe.
        symbol: The (upper-cased) symbol to score.
        fallback: Value to use for ``symbol`` when the universe lacks it.
        all_vals: Optional pre-fetched metric dict (used when the metric is not a
            plain attribute, e.g. for a derived value).

    Returns:
        A tuple ``(z, std, n_valid)``. When the universe is unavailable or the
        cross-section is degenerate, ``z`` is ``0.0``.
    """
    vals: dict[str, float] = {}
    if all_vals is not None:
        vals = all_vals
    elif universe is not None:
        raw = getattr(universe, metric, None)
        if isinstance(raw, dict):
            vals = raw
    if not vals:
        return 0.0, 0.0, 0
    arr = np.array(
        [float(v) for v in vals.values() if math.isfinite(float(v))],
        dtype=np.float64,
    )
    if arr.size == 0:
        return 0.0, 0.0, 0
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    x = float(vals.get(symbol, fallback))
    if not math.isfinite(x):
        x = fallback
    if std <= 1e-12 or not math.isfinite(std):
        return 0.0, 0.0, int(arr.size)
    z = (x - mean) / std
    if not math.isfinite(z):
        z = 0.0
    return clamp(z, -6.0, 6.0), std, int(arr.size)


# ---------------------------------------------------------------------------
# Catalog metadata (sources carried from strategy-catalog.json)
# ---------------------------------------------------------------------------

_META: dict[str, StrategyMeta] = {
    "dual-momentum": StrategyMeta(
        id="dual-momentum",
        name="Dual Momentum / GEM (Antonacci)",
        category="Portfolio",
        summary=(
            "Combines relative momentum (stronger of US vs ex-US equities) with "
            "absolute momentum (hold equities only if they beat T-bills, else a "
            "defensive sleeve). Antonacci (2014): GEM matched/beat the S&P 500 "
            "with ~half the max drawdown."
        ),
        formula="abs_mom = r_12m - rf_12m; rel = r_12m - best_peer; score ~ 50*sign(abs)+50*sign(rel)",
        inputs=["price history", "risk-free rate", "cross-sectional peers"],
        references=[
            "Antonacci (2014), 'Dual Momentum Investing', McGraw-Hill",
            "Antonacci (2013), 'Risk Premia Harvesting Through Dual Momentum', SSRN 2042750",
        ],
    ),
    "tsmom": StrategyMeta(
        id="tsmom",
        name="Time-Series (Absolute) Momentum with Vol Scaling",
        category="Technical",
        summary=(
            "Trade each asset on the sign of its own trailing ~12-month excess "
            "return, sizing inversely to ex-ante volatility to target constant "
            "risk. Moskowitz, Ooi & Pedersen (2012)."
        ),
        formula="raw = (r_12m - rf_12m) / sigma_annual; score = 100*tanh(raw)",
        inputs=["price history", "risk-free rate"],
        references=[
            "Moskowitz, Ooi & Pedersen (2012), 'Time Series Momentum', JFE 104(2):228-250",
            "https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum",
        ],
    ),
    "cross-sectional-momentum": StrategyMeta(
        id="cross-sectional-momentum",
        name="Cross-Sectional Momentum (12-1, Jegadeesh-Titman)",
        category="Factor",
        summary=(
            "The relative-strength momentum anomaly: rank assets by trailing "
            "12-month return skipping the most recent month, long winners short "
            "losers. Jegadeesh & Titman (1993); basis of the UMD/MOM factor."
        ),
        formula="r_12_1 = Close[t-21]/Close[t-252]-1; z across universe; score = clip(z*40,-100,100)",
        inputs=["price history", "cross-sectional universe"],
        references=[
            "Jegadeesh & Titman (1993), 'Returns to Buying Winners and Selling Losers', Journal of Finance 48(1):65-91",
            "https://alphaarchitect.com/momentum-factor-investing-30-years-of-out-of-sample-data/",
        ],
    ),
    "52w-high": StrategyMeta(
        id="52w-high",
        name="52-Week High Proximity / Breakout (George-Hwang)",
        category="Technical",
        summary=(
            "Rank assets by how close price is to its 52-week high; nearness "
            "predicts higher future returns and subsumes much of Jegadeesh-Titman "
            "momentum (anchoring). George & Hwang (2004)."
        ),
        formula="PR = Close / HH252; score = clip((PR-0.85)/0.15*100, -100, 100)",
        inputs=["price history (highs)"],
        references=[
            "George & Hwang (2004), 'The 52-Week High and Momentum Investing', Journal of Finance 59(5):2145-2176",
            "Darvas (1960), 'How I Made $2,000,000 in the Stock Market'",
        ],
    ),
    "relative-strength-rotation": StrategyMeta(
        id="relative-strength-rotation",
        name="Relative-Strength Rotation (Faber)",
        category="Technical",
        summary=(
            "Rank a universe by recent relative strength (average of 1/3/6/12-month "
            "returns) and hold the top-N each month, gated by a long-term trend "
            "filter on the broad market. Faber & Richardson (2010)."
        ),
        formula="rs = mean(r_21, r_63, r_126, r_252); z across universe; score = clip(z*40,-100,100), gated by SPY>SMA200",
        inputs=["price history", "cross-sectional universe", "broad-market trend"],
        references=[
            "Faber & Richardson (2010), 'Relative Strength Strategies for Investing', Cambria",
            "Faber (2007), 'A Quantitative Approach to Tactical Asset Allocation'",
        ],
    ),
    "frog-in-the-pan-momentum": StrategyMeta(
        id="frog-in-the-pan-momentum",
        name="Risk-Adjusted (Frog-in-the-Pan) Momentum",
        category="Factor",
        summary=(
            "Refines 12-1 momentum by preferring smooth, high-consistency trends "
            "over jumpy ones. Da, Gurun & Warachka (2014): continuous information "
            "produces stronger, more persistent momentum."
        ),
        formula="score = clip(z_mom*40*m, -100, 100); m = smoothness from IR / information-discreteness",
        inputs=["price history", "cross-sectional universe"],
        references=[
            "Da, Gurun & Warachka (2014), 'Frog in the Pan: Continuous Information and Momentum', Review of Financial Studies 27(7):2171-2218",
        ],
    ),
    "faber-taa": StrategyMeta(
        id="faber-taa",
        name="Faber 10-Month / 200-Day SMA Timing",
        category="Technical",
        summary=(
            "Mebane Faber's tactical timing: hold the asset only when its close is "
            "above its 10-month (~200-day) SMA, else go to cash. Equity-like "
            "returns with bond-like drawdowns. Faber (2006)."
        ),
        formula="d = (P - SMA200)/SMA200; score = clamp(d/0.10*100, -100, 100)",
        inputs=["price history"],
        references=[
            "Faber (2007), 'A Quantitative Approach to Tactical Asset Allocation', Journal of Wealth Management, SSRN 962461",
            "https://mebfaber.com/timing-model/",
        ],
    ),
    "donchian-turtle": StrategyMeta(
        id="donchian-turtle",
        name="Donchian Channel / Turtle Breakout",
        category="Technical",
        summary=(
            "The classic Turtle trend-following system (Dennis & Eckhardt) on "
            "Donchian channels: enter long on a new N-day high, exit on the "
            "opposite shorter-channel low. System 1: 20/10; System 2: 55/20."
        ),
        formula="midline = (HH20+LL20)/2; score = clamp((P-midline)/(0.5*(HH20-LL20))*100, -100, 100)",
        inputs=["price history (highs/lows)"],
        references=[
            "Curtis Faith (2007), 'Way of the Turtle', McGraw-Hill",
            "'The Original Turtle Trading Rules' - https://www.turtletrader.com/rules/",
        ],
    ),
    "golden-cross": StrategyMeta(
        id="golden-cross",
        name="Golden Cross / Death Cross (50/200 SMA)",
        category="Technical",
        summary=(
            "Long-term trend regime filter: a Golden Cross (50-day SMA above "
            "200-day) is bullish, a Death Cross (50 below 200) bearish. Trims "
            "drawdowns more reliably than it boosts raw return."
        ),
        formula="spread = (SMA50 - SMA200)/SMA200; score = clamp(spread/0.10*100, -100, 100)",
        inputs=["price history"],
        references=[
            "Corporate Finance Institute - Death Cross - https://corporatefinanceinstitute.com/resources/capital-markets/death-cross/",
            "QuantifiedStrategies, 'Does the Death Cross Actually Work?' (65-yr backtest)",
        ],
    ),
    "dual-ma-crossover": StrategyMeta(
        id="dual-ma-crossover",
        name="Dual Moving-Average Crossover (Fast/Slow)",
        category="Technical",
        summary=(
            "Foundational systematic trend rule: long when a fast MA crosses above "
            "a slow MA, exit/short when below. Brock, Lakonishok & LeBaron (1992); "
            "AQR confirms trend premia. Defaults 20/100 SMA."
        ),
        formula="spread = (fastMA - slowMA)/slowMA; score = clamp(spread/0.06*100, -100, 100)",
        inputs=["price history"],
        references=[
            "Brock, Lakonishok & LeBaron (1992), 'Simple Technical Trading Rules', Journal of Finance 47(5):1731-1764",
            "Hurst, Ooi & Pedersen (AQR, 2017), 'A Century of Evidence on Trend-Following Investing'",
        ],
    ),
    "supertrend": StrategyMeta(
        id="supertrend",
        name="Supertrend (ATR Trailing Trend)",
        category="Technical",
        summary=(
            "Volatility-adaptive trend indicator (Olivier Seban): an ATR-scaled "
            "band trails price; closing above flips the trend bullish, below "
            "bearish. Period 10, multiplier 3. Self-adjusts to volatility."
        ),
        formula="score = clamp((close - Supertrend)/(3*ATR)*100, -100, 100)",
        inputs=["price history (highs/lows)"],
        references=[
            "Olivier Seban - SuperTrend indicator",
            "TradingView Supertrend docs - https://www.tradingview.com/support/solutions/43000634738-supertrend/",
        ],
    ),
    "ichimoku": StrategyMeta(
        id="ichimoku",
        name="Ichimoku Kinko Hyo (Cloud)",
        category="Technical",
        summary=(
            "Goichi Hosoda's all-in-one trend system: Tenkan/Kijun lines, a Kumo "
            "(cloud) of leading spans, and the lagging Chikou span give trend, "
            "support/resistance and momentum. Price above the cloud is bullish."
        ),
        formula="raw = sum of {price>cloud, Tenkan>Kijun, SpanA>SpanB, close>close[26]} votes / 4; score = raw*(0.6+0.4|W|)*100",
        inputs=["price history (highs/lows)"],
        references=[
            "Goichi Hosoda - Ichimoku Kinko Hyo",
            "StockCharts ChartSchool - Ichimoku Cloud",
        ],
    ),
    "adx-trend-strength": StrategyMeta(
        id="adx-trend-strength",
        name="ADX Trend-Strength Filter (Wilder DMI)",
        category="Technical",
        summary=(
            "Wilder's Average Directional Index measures trend STRENGTH (not "
            "direction); +DI/-DI give direction. ADX>25 confirms a trend worth "
            "following, ADX<20 flags a choppy range. A regime gauge."
        ),
        formula="direction = sign(+DI - -DI); strength = clip((ADX-20)/30, 0, 1); score = clamp(direction*strength*100, -100, 100)",
        inputs=["price history (highs/lows)"],
        references=[
            "J. Welles Wilder (1978), 'New Concepts in Technical Trading Systems' - ADX/DMI",
            "StockCharts ChartSchool - ADX/DMI",
        ],
    ),
    "ma-ribbon": StrategyMeta(
        id="ma-ribbon",
        name="Moving-Average Ribbon Alignment",
        category="Technical",
        summary=(
            "A multi-MA 'ribbon' (10/20/50/100/200) read as one trend gauge: "
            "stacked-in-order and fanning apart = strong trend; compressing or "
            "crossing = fading/reversing. Generalizes single MA crossovers (GMMA)."
        ),
        formula="ordered = (bull pairs - bear pairs)/total; W = mean(|SMA_i - SMA_{i+1}|)/price; score = clamp(ordered*(0.6+0.4*min(W/0.04,1))*100, -100, 100)",
        inputs=["price history"],
        references=[
            "Daryl Guppy - Guppy Multiple Moving Average (GMMA)",
            "QuantifiedStrategies - EMA Ribbon strategy - https://www.quantifiedstrategies.com/exponential-moving-average-ribbon/",
        ],
    ),
    "absolute-momentum-overlay": StrategyMeta(
        id="absolute-momentum-overlay",
        name="Absolute Momentum Trend Overlay (Antonacci)",
        category="Technical",
        summary=(
            "Single-asset trend filter: be invested only when an asset's trailing "
            "12-month return exceeds the risk-free rate, else go to cash. "
            "Antonacci (2013) - cuts volatility and drawdown; a universal overlay."
        ),
        formula="excess = r_12m - rf_12m; score = clip(100*tanh((excess/sigma)*1.5), -100, 100)",
        inputs=["price history", "risk-free rate"],
        references=[
            "Antonacci (2013), 'Absolute Momentum: A Simple Rule-Based Strategy and Universal Trend-Following Overlay', SSRN 2244633",
        ],
    ),
}


def _meta(strategy_id: str) -> StrategyMeta:
    """Return the :class:`StrategyMeta` for ``strategy_id``."""
    return _META[strategy_id]


# ---------------------------------------------------------------------------
# Signal builders
# ---------------------------------------------------------------------------

def _build_dual_momentum(ctx: "AnalysisContext") -> StrategySignal:
    """Dual Momentum / GEM signal (Antonacci).

    Per the catalog ``computeSignal``: from trailing 12m total return and 12m
    risk-free, ``abs_mom = r_12m - rf_12m`` (absolute filter) and the relative
    rank is ``r_12m`` minus the best competitor's 12m return (read from the
    universe's ``momentum_12_1``-style ranks). The equity score is
    ``clip(50*sign(abs_mom) + 50*sign(rel_winner), -100, 100)``; confidence rises
    with history and with ``|abs_mom|`` far from the hurdle.
    """
    meta = _meta("dual-momentum")
    closes = _closes(ctx)
    sym = _symbol(ctx)
    if closes.size < 2:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "Insufficient price history for dual momentum.", meta.formula,
            metrics={}, horizons=[],
        )

    r_12m = _trailing_return(closes, _TD)
    rf_12m = _rf_daily(ctx) * _TD
    abs_mom = r_12m - rf_12m

    # Relative momentum: this asset's 12m return vs the best peer in the universe.
    universe = _universe(ctx)
    best_peer = r_12m
    n_peers = 1
    if universe is not None:
        mom = getattr(universe, "momentum_12_1", None)
        if isinstance(mom, dict) and mom:
            vals = [float(v) for v in mom.values() if math.isfinite(float(v))]
            if vals:
                best_peer = max(vals)
                n_peers = len(vals)
    rel_winner = r_12m - best_peer  # >= 0 only if this asset IS the best peer

    sign_abs = 1.0 if abs_mom > 0 else (-1.0 if abs_mom < 0 else 0.0)
    sign_rel = 1.0 if rel_winner >= 0 else -1.0
    score = clamp(50.0 * sign_abs + 50.0 * sign_rel, -100.0, 100.0)

    sigma_annual = _annual_vol(closes)
    hurdle_gap = abs(abs_mom) / sigma_annual if sigma_annual > 0 else 0.0
    confidence = clamp(
        _history_conf(closes) * clamp(0.4 + 0.6 * min(1.0, hurdle_gap), 0.4, 1.0),
        0.1, 1.0,
    )

    horizons: list[dict] = []
    if abs_mom > 0:
        # Risk-on: project the asset's own trailing risk premium.
        mu = _drift_daily_from_annual(min(0.35, max(0.0, r_12m)))
        horizons = _project(mu, sigma_annual / math.sqrt(_TD))
    else:
        # Risk-off: defensive, project ~risk-free at low vol.
        horizons = _project(_rf_daily(ctx), 0.06 / math.sqrt(_TD))

    rationale = (
        f"12m return {r_12m * 100:+.1f}% vs 12m T-bill {rf_12m * 100:.1f}% "
        f"-> absolute momentum {abs_mom * 100:+.1f}% "
        f"({'risk-on, hold equity' if abs_mom > 0 else 'risk-off, hold defensive'}); "
        f"relative to the best of {n_peers} peer(s) it is "
        f"{'the leader' if rel_winner >= 0 else f'{rel_winner * 100:.1f}% behind'}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "r12m": r_12m,
            "rf12m": rf_12m,
            "absMom": abs_mom,
            "relWinner": rel_winner,
            "bestPeer": best_peer,
        },
        horizons=horizons,
    )


def _build_tsmom(ctx: "AnalysisContext") -> StrategySignal:
    """Time-series momentum with vol scaling signal (Moskowitz-Ooi-Pedersen).

    Catalog ``computeSignal``: ``r_12m = Close[t]/Close[t-252]-1``; ``rf_12m`` =
    trailing 12m risk-free; ``excess = r_12m - rf_12m``;
    ``sigma_annual = sqrt(252*EWMA(r_daily^2, 0.94))``; ``raw = excess/sigma_annual``;
    ``score = clip(tanh(raw)*100, -100, 100)``;
    ``confidence = min(1, history/252) * clip(|excess|/sigma_annual, 0.3, 1)``.
    """
    meta = _meta("tsmom")
    closes = _closes(ctx)
    if closes.size < 2:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "Insufficient price history for time-series momentum.", meta.formula,
            metrics={}, horizons=[],
        )

    r_12m = _trailing_return(closes, _TD)
    rf_12m = _rf_daily(ctx) * _TD
    excess = r_12m - rf_12m
    sigma_annual = _ewma_annual_vol(closes, lam=0.94)
    raw = excess / sigma_annual if sigma_annual > 0 else 0.0
    score = clamp(math.tanh(raw) * 100.0, -100.0, 100.0)

    confidence = clamp(
        _history_conf(closes) * clamp(abs(raw), 0.3, 1.0),
        0.1, 1.0,
    )

    # Positive trend predicts continued positive excess return ~1yr.
    mu = _drift_daily_from_annual(clamp(excess, -0.40, 0.60))
    horizons = _project(mu, sigma_annual / math.sqrt(_TD))

    rationale = (
        f"Trailing 12m excess return {excess * 100:+.1f}% vs ex-ante vol "
        f"{sigma_annual * 100:.0f}% -> risk-adjusted trend {raw:+.2f} "
        f"({'long' if score > 0 else 'short/flat'}); position sized inverse to vol."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "r12m": r_12m,
            "excess": excess,
            "sigmaAnnual": sigma_annual,
            "raw": raw,
        },
        horizons=horizons,
    )


def _build_cross_sectional_momentum(ctx: "AnalysisContext") -> StrategySignal:
    """Cross-sectional 12-1 momentum signal (Jegadeesh-Titman).

    Catalog ``computeSignal``: ``r_12_1 = Close[t-21]/Close[t-252]-1`` per asset;
    ``z = (r_12_1 - mean)/std`` across the universe; ``score = clip(z*40, -100, 100)``
    (+2.5 sigma ~ +100); ``confidence = min(1, n_valid/24) * min(1, history/252)``,
    reduced when history < 252 or cross-sectional std is near zero.
    """
    meta = _meta("cross-sectional-momentum")
    closes = _closes(ctx)
    sym = _symbol(ctx)
    own_mom = technical.momentum_12_1(closes)  # 12-1 with skip-month spec

    universe = _universe(ctx)
    z, std, n_valid = _xsec_z(universe, "momentum_12_1", sym, fallback=own_mom)
    if universe is None or n_valid == 0:
        # No cross-section available: fall back to a self-referential read so the
        # signal is still informative (scaled own momentum).
        z = clamp(own_mom / 0.20, -3.0, 3.0)
        n_valid = 1

    score = clamp(z * 40.0, -100.0, 100.0)
    conf = clamp(
        min(1.0, n_valid / 24.0) * _history_conf(closes),
        0.1, 1.0,
    )
    if std <= 1e-9 and universe is not None:
        conf *= 0.5

    # Winners' next-month expected excess return is positive; map score to drift.
    mu = _drift_daily_from_annual(clamp(score / 100.0 * 0.15, -0.25, 0.35))
    horizons = _project(mu, _annual_vol(closes) / math.sqrt(_TD))

    rationale = (
        f"12-1 momentum {own_mom * 100:+.1f}% ranks at z={z:+.2f} across "
        f"{n_valid} asset(s) "
        f"({'a winner (long)' if score > 0 else 'a loser (short/avoid)'})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, conf, rationale, meta.formula,
        metrics={
            "momentum12_1": own_mom,
            "zScore": z,
            "crossSectionStd": std,
            "nValid": float(n_valid),
        },
        horizons=horizons,
    )


def _build_52w_high(ctx: "AnalysisContext") -> StrategySignal:
    """52-week-high proximity / breakout signal (George-Hwang).

    Catalog ``computeSignal``: ``high252 = max(High[t-252..t])``;
    ``PR = Close/high252``; ``score = clip((PR-0.85)/0.15*100, -100, 100)``
    (PR=1.0 ~ +100, 0.85 ~ 0, <=0.70 ~ -100); flips strongly negative on a new
    52-week low. Confidence from history, reduced if highly volatile/noisy.
    """
    meta = _meta("52w-high")
    closes = _closes(ctx)
    highs = _highs(ctx)
    lows = _lows(ctx)
    if closes.size == 0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No price history for 52-week-high proximity.", meta.formula,
            metrics={}, horizons=[],
        )

    last = float(closes[-1])
    win = min(_TD, highs.size)
    high252 = float(np.max(highs[-win:])) if win > 0 else last
    low_win = min(_TD, lows.size)
    low252 = float(np.min(lows[-low_win:])) if low_win > 0 else last
    pr = last / high252 if high252 > 0 else 1.0
    pr = clamp(pr, 0.0, 1.5)
    score = clamp((pr - 0.85) / 0.15 * 100.0, -100.0, 100.0)
    # Flip strongly negative at / below a new 52-week low.
    if low252 > 0 and last <= low252 * 1.001:
        score = min(score, -90.0)

    vol = _annual_vol(closes)
    # Reduce confidence when very volatile/noisy (vol > ~60% annual).
    noise_penalty = clamp(1.0 - max(0.0, vol - 0.60) / 0.80, 0.5, 1.0)
    confidence = clamp(_history_conf(closes) * noise_penalty * (0.5 + 0.4 * abs(score) / 100.0), 0.1, 0.95)

    # Near-high names carry a positive tilt that persists (George-Hwang).
    mu = _drift_daily_from_annual(clamp((pr - 0.85) * 0.8, -0.25, 0.30))
    horizons = _project(mu, vol / math.sqrt(_TD))

    rationale = (
        f"Price {last:.2f} is {pr * 100:.0f}% of its 52-week high {high252:.2f} "
        f"(proximity ratio {pr:.2f}); "
        f"{'near the high (bullish breakout tilt)' if score > 0 else 'well below the high (weak)'}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "proximityRatio": pr,
            "high252": high252,
            "low252": low252,
            "price": last,
        },
        horizons=horizons,
    )


def _build_relative_strength_rotation(ctx: "AnalysisContext") -> StrategySignal:
    """Relative-strength rotation signal (Faber).

    Catalog ``computeSignal``: ``rs = mean(r_21, r_63, r_126, r_252)``;
    ``z = (rs - mean)/std`` across the rotation universe;
    ``base = clip(z*40, -100, 100)``. If SPY < 200d SMA multiply by 0.0-0.3
    (risk-off). Confidence from history times a trend-filter agreement factor.
    """
    meta = _meta("relative-strength-rotation")
    closes = _closes(ctx)
    sym = _symbol(ctx)
    if closes.size < 2:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "Insufficient price history for relative-strength rotation.", meta.formula,
            metrics={}, horizons=[],
        )

    rs = float(
        np.mean(
            [
                _trailing_return(closes, 21),
                _trailing_return(closes, 63),
                _trailing_return(closes, 126),
                _trailing_return(closes, 252),
            ]
        )
    )

    # Cross-sectional z of the composite RS. The universe exposes momentum_6m and
    # momentum_12_1; build a composite-RS dict if the universe lets us, else use
    # the universe momentum_12_1 ranks as a proxy.
    universe = _universe(ctx)
    z, std, n_valid = (0.0, 0.0, 0)
    if universe is not None:
        # Proxy composite RS by blending the available cross-sectional momentum
        # dicts (12-1 and 6m) so the ranking is genuinely cross-sectional.
        mom12 = getattr(universe, "momentum_12_1", None)
        mom6 = getattr(universe, "momentum_6m", None)
        comp: dict[str, float] = {}
        if isinstance(mom12, dict):
            for k, v in mom12.items():
                comp[k] = comp.get(k, 0.0) + 0.5 * float(v)
        if isinstance(mom6, dict):
            for k, v in mom6.items():
                comp[k] = comp.get(k, 0.0) + 0.5 * float(v)
        if comp:
            comp.setdefault(sym, rs)
            z, std, n_valid = _xsec_z(universe, "", sym, fallback=rs, all_vals=comp)
    if n_valid == 0:
        z = clamp(rs / 0.15, -3.0, 3.0)
        n_valid = 1

    base = clamp(z * 40.0, -100.0, 100.0)

    # Trend gate: only hold equity sleeves while the broad market is in an uptrend.
    # Proxy the SPY>SMA200 filter by this asset's own 200d trend when no market
    # series is available (own price above its 200d SMA == risk-on).
    sma200 = technical.sma(closes, 200)
    own_above = bool(sma200.size and closes.size and closes[-1] > sma200[-1])
    market_ret = _clean(getattr(ctx, "market_ret", None))
    risk_on = own_above
    gate = 1.0 if risk_on else 0.2  # 0.0-0.3 risk-off; use 0.2
    score = clamp(base * gate, -100.0, 100.0)

    trend_agrees = (base >= 0) == risk_on
    confidence = clamp(
        _history_conf(closes) * (1.0 if trend_agrees else 0.5),
        0.1, 0.95,
    )

    mu = _drift_daily_from_annual(clamp(score / 100.0 * 0.15, -0.25, 0.30))
    horizons = _project(mu, _annual_vol(closes) / math.sqrt(_TD))

    rationale = (
        f"Composite relative strength {rs * 100:+.1f}% ranks at z={z:+.2f} "
        f"({base:+.0f} base); trend filter is "
        f"{'risk-on' if risk_on else 'risk-off (scaled down)'} -> score {score:+.0f}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "compositeRs": rs,
            "zScore": z,
            "base": base,
            "gate": gate,
            "nValid": float(n_valid),
        },
        horizons=horizons,
    )


def _build_frog_in_the_pan(ctx: "AnalysisContext") -> StrategySignal:
    """Frog-in-the-Pan (risk-adjusted) momentum signal (Da-Gurun-Warachka).

    Catalog ``computeSignal``: ``z_mom`` = cross-sectional z of ``r_12_1``;
    ``IR = r_12_1/(daily_std*sqrt(252))``; consistency ``C = (up_days-down_days)/total``
    signed by trend; smoothness multiplier ``m = clip(0.5+0.5*normalized(IR or -ID), 0.3, 1.0)``;
    ``score = clip(z_mom*40*m, -100, 100)``; confidence rises with smoothness.
    """
    meta = _meta("frog-in-the-pan-momentum")
    closes = _closes(ctx)
    sym = _symbol(ctx)
    if closes.size < 2:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "Insufficient price history for frog-in-the-pan momentum.", meta.formula,
            metrics={}, horizons=[],
        )

    own_mom = technical.momentum_12_1(closes)
    universe = _universe(ctx)
    z_mom, std, n_valid = _xsec_z(universe, "momentum_12_1", sym, fallback=own_mom)
    if universe is None or n_valid == 0:
        z_mom = clamp(own_mom / 0.20, -3.0, 3.0)
        n_valid = 1

    # Information ratio over the formation window (t-252..t-21 when available).
    formation = closes
    if closes.size > _TD:
        formation = closes[-(_TD + 1):-21] if closes.size > _TD + 1 else closes[-(_TD + 1):]
    rets = returns.simple_returns(formation)
    daily_std = float(np.std(rets)) if rets.size else 0.0
    formation_vol = daily_std * math.sqrt(_TD)
    ir = own_mom / formation_vol if formation_vol > 0 else 0.0

    # Information discreteness ID = sign(PRET)*(%down - %up); low ID = smooth.
    if rets.size:
        up_days = float(np.sum(rets > 0))
        down_days = float(np.sum(rets < 0))
        total = float(rets.size)
        pct_up = up_days / total
        pct_down = down_days / total
        consistency = (up_days - down_days) / total
        sign_pret = 1.0 if own_mom >= 0 else -1.0
        info_discreteness = sign_pret * (pct_down - pct_up)
    else:
        consistency = 0.0
        info_discreteness = 0.0

    # Smoothness: higher IR and lower ID -> smoother (stronger). Normalize IR by a
    # reference of ~1.0 Sharpe-like scale.
    norm = clamp(0.5 * (clamp(ir, -2.0, 2.0) / 2.0) + 0.5 * (-info_discreteness), -1.0, 1.0)
    m = clamp(0.5 + 0.5 * norm, 0.3, 1.0)
    score = clamp(z_mom * 40.0 * m, -100.0, 100.0)

    confidence = clamp(
        _history_conf(closes) * clamp(0.4 + 0.5 * m, 0.4, 0.95),
        0.1, 0.95,
    )

    mu = _drift_daily_from_annual(clamp(score / 100.0 * 0.15, -0.25, 0.30))
    horizons = _project(mu, _annual_vol(closes) / math.sqrt(_TD))

    rationale = (
        f"12-1 momentum z={z_mom:+.2f} refined by smoothness x{m:.2f} "
        f"(information ratio {ir:+.2f}, consistency {consistency:+.2f}) -> "
        f"{'a smooth winner (stronger)' if score > 20 else 'a loser/jumpy trend' if score < -20 else 'neutral'}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "zMom": z_mom,
            "informationRatio": ir,
            "consistency": consistency,
            "infoDiscreteness": info_discreteness,
            "smoothness": m,
        },
        horizons=horizons,
    )


def _build_faber_taa(ctx: "AnalysisContext") -> StrategySignal:
    """Faber 10-month / 200-day SMA timing signal.

    Catalog ``computeSignal``: ``P`` = latest close, ``SMA200`` = 200-day SMA,
    ``d = (P-SMA200)/SMA200``; ``score = clamp(d/0.10*100, -100, 100)`` (10%+ above
    saturates +100, 10%+ below -100); confidence rises when the SMA slopes with
    the signal: ``clamp(0.4 + |normalized 20d SMA slope|*k, 0.4, 0.9)``.
    """
    meta = _meta("faber-taa")
    closes = _closes(ctx)
    if closes.size == 0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No price history for Faber SMA timing.", meta.formula,
            metrics={}, horizons=[],
        )

    p = float(closes[-1])
    sma200_series = technical.sma(closes, 200)
    sma200 = float(sma200_series[-1]) if sma200_series.size else p
    d = (p - sma200) / sma200 if sma200 > 0 else 0.0
    score = clamp(d / 0.10 * 100.0, -100.0, 100.0)

    # SMA slope over the last 20 bars, normalized by price.
    slope_norm = 0.0
    if sma200_series.size >= 21 and p > 0:
        slope_norm = (float(sma200_series[-1]) - float(sma200_series[-21])) / (20.0 * p)
    slope_agrees = (slope_norm >= 0) == (score >= 0)
    confidence = clamp(0.4 + abs(slope_norm) * 1000.0 * (1.0 if slope_agrees else 0.3), 0.4, 0.9)
    if closes.size < 200:
        confidence *= 0.6

    # Above SMA200: project the asset's trailing risk premium; below: risk-free.
    if score > 0:
        mu = _drift_daily_from_annual(clamp(_trailing_return(closes, _TD), -0.25, 0.35))
        sigma = _annual_vol(closes) / math.sqrt(_TD)
    else:
        mu = _rf_daily(ctx)
        sigma = 0.02 / math.sqrt(_TD)
    horizons = _project(mu, sigma)

    rationale = (
        f"Close {p:.2f} is {d * 100:+.1f}% vs its 200-day SMA {sma200:.2f} "
        f"({'above -> hold (risk-on)' if score > 0 else 'below -> cash (risk-off)'}); "
        f"SMA200 slope {'confirms' if slope_agrees else 'diverges from'} the signal."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "price": p,
            "sma200": sma200,
            "distance": d,
            "smaSlope": slope_norm,
        },
        horizons=horizons,
    )


def _build_donchian_turtle(ctx: "AnalysisContext") -> StrategySignal:
    """Donchian channel / Turtle breakout signal.

    Catalog continuous variant: ``midline = (HH20+LL20)/2``;
    ``score = clamp(((P-midline)/(0.5*(HH-LL)))*100, -100, 100)``. A fresh 20/55-day
    high pushes the score toward +100, a fresh downside breakout toward -100.
    Confidence: ``clamp(0.3 + min(channel_width_in_N, 5)/5*0.6, 0.3, 0.9)``.
    """
    meta = _meta("donchian-turtle")
    closes = _closes(ctx)
    highs = _highs(ctx)
    lows = _lows(ctx)
    if closes.size == 0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No price history for Donchian breakout.", meta.formula,
            metrics={}, horizons=[],
        )

    p = float(closes[-1])
    hh20, ll20 = indicators.donchian(highs, lows, n=20)  # prior-20 channel
    hh55, _ = indicators.donchian(highs, lows, n=55)
    _, ll10 = indicators.donchian(lows, lows, n=10)  # not used for score; kept for state read

    midline = 0.5 * (hh20 + ll20)
    half_range = 0.5 * (hh20 - ll20)
    if half_range > 1e-9:
        score = clamp((p - midline) / half_range * 100.0, -100.0, 100.0)
    else:
        score = 0.0

    # Freshness: a fresh upside breakout (close above prior HH20) saturates +100.
    if hh20 > 0 and p > hh20:
        score = max(score, 80.0)
    if hh55 > 0 and p > hh55:
        score = 100.0
    if ll20 > 0 and p < ll20:
        score = min(score, -80.0)

    atr_series = indicators.atr(highs, lows, closes, n=20)
    atr_n = float(atr_series[-1]) if atr_series.size else 0.0
    channel_width_in_n = (hh20 - ll20) / atr_n if atr_n > 1e-9 else 0.0
    confidence = clamp(0.3 + min(channel_width_in_n, 5.0) / 5.0 * 0.6, 0.3, 0.9)

    # On a confirmed breakout, project continuation at the trailing momentum rate.
    mu = _drift_daily_from_annual(clamp(_trailing_return(closes, 63) * 4.0, -0.25, 0.35)) if score > 0 else 0.0
    sigma = (atr_n / p / math.sqrt(1.0)) if p > 0 and atr_n > 0 else _annual_vol(closes) / math.sqrt(_TD)
    horizons = _project(mu, sigma) if score > 0 else []

    rationale = (
        f"Close {p:.2f} vs 20-day Donchian channel [{ll20:.2f}, {hh20:.2f}] "
        f"(midline {midline:.2f}); "
        f"{'fresh upside breakout (long)' if score >= 80 else 'in-channel' if abs(score) < 80 else 'downside breakout (exit/short)'}; "
        f"channel width {channel_width_in_n:.1f}N."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "price": p,
            "hh20": hh20,
            "ll20": ll20,
            "hh55": hh55,
            "ll10": ll10,
            "atr": atr_n,
            "channelWidthN": channel_width_in_n,
        },
        horizons=horizons,
    )


def _build_golden_cross(ctx: "AnalysisContext") -> StrategySignal:
    """Golden Cross / Death Cross (50/200 SMA) signal.

    Catalog ``computeSignal``: ``spread = (SMA50-SMA200)/SMA200``;
    ``score = clamp(spread/0.10*100, -100, 100)``; a freshness boost pushes |score|
    toward 100 if a cross occurred within the last 5 bars; confidence:
    ``clamp(0.4 + min(|spread|/0.05, 1)*0.3 + (SMA200 slope agrees ? 0.2 : 0), 0.4, 0.95)``.
    """
    meta = _meta("golden-cross")
    closes = _closes(ctx)
    if closes.size == 0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No price history for golden cross.", meta.formula,
            metrics={}, horizons=[],
        )

    sma50 = technical.sma(closes, 50)
    sma200 = technical.sma(closes, 200)
    s50 = float(sma50[-1]) if sma50.size else float(closes[-1])
    s200 = float(sma200[-1]) if sma200.size else float(closes[-1])
    spread = (s50 - s200) / s200 if s200 > 0 else 0.0
    score = clamp(spread / 0.10 * 100.0, -100.0, 100.0)

    # Freshness: was there a cross in the last 5 bars?
    fresh = False
    if sma50.size >= 6 and sma200.size >= 6:
        diff = sma50[-6:] - sma200[-6:]
        signs = np.sign(diff)
        if np.any(signs[:-1] != signs[-1]):
            fresh = True
            score = math.copysign(max(abs(score), 70.0), score if score != 0 else 1.0)

    slope200 = 0.0
    if sma200.size >= 21:
        slope200 = float(sma200[-1]) - float(sma200[-21])
    slope_agrees = (slope200 >= 0) == (score >= 0)
    confidence = clamp(
        0.4 + min(abs(spread) / 0.05, 1.0) * 0.3 + (0.2 if slope_agrees else 0.0),
        0.4, 0.95,
    )
    if closes.size < 200:
        confidence *= 0.7

    if score > 0:
        mu = _drift_daily_from_annual(clamp(_trailing_return(closes, _TD), -0.25, 0.35))
        sigma = _annual_vol(closes) / math.sqrt(_TD)
    else:
        mu = _rf_daily(ctx)
        sigma = 0.02 / math.sqrt(_TD)
    horizons = _project(mu, sigma)

    rationale = (
        f"SMA50 {s50:.2f} is {spread * 100:+.1f}% vs SMA200 {s200:.2f} "
        f"({'golden cross (bullish)' if score > 0 else 'death cross (bearish)'}"
        f"{', fresh' if fresh else ''}); SMA200 slope "
        f"{'confirms' if slope_agrees else 'diverges'}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "sma50": s50,
            "sma200": s200,
            "spread": spread,
            "slope200": slope200,
            "fresh": 1.0 if fresh else 0.0,
        },
        horizons=horizons,
    )


def _build_dual_ma_crossover(ctx: "AnalysisContext") -> StrategySignal:
    """Dual moving-average crossover (fast/slow) signal.

    Catalog ``computeSignal``: with fast=20, slow=100 SMA,
    ``spread = (fastMA-slowMA)/slowMA``; ``score = clamp(spread/0.06*100, -100, 100)``;
    a freshness boost if the crossover is within ~3 bars; confidence:
    ``clamp(0.35 + min(|spread|/0.03, 1)*0.35 + (slowMA slope agrees ? 0.2 : 0), 0.35, 0.9)``.
    """
    meta = _meta("dual-ma-crossover")
    closes = _closes(ctx)
    if closes.size == 0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No price history for dual MA crossover.", meta.formula,
            metrics={}, horizons=[],
        )

    fast = technical.sma(closes, 20)
    slow = technical.sma(closes, 100)
    f = float(fast[-1]) if fast.size else float(closes[-1])
    s = float(slow[-1]) if slow.size else float(closes[-1])
    spread = (f - s) / s if s > 0 else 0.0
    score = clamp(spread / 0.06 * 100.0, -100.0, 100.0)

    fresh = False
    if fast.size >= 4 and slow.size >= 4:
        diff = fast[-4:] - slow[-4:]
        signs = np.sign(diff)
        if np.any(signs[:-1] != signs[-1]):
            fresh = True
            score = math.copysign(max(abs(score), 60.0), score if score != 0 else 1.0)

    slow_slope = 0.0
    if slow.size >= 21:
        slow_slope = float(slow[-1]) - float(slow[-21])
    slope_agrees = (slow_slope >= 0) == (score >= 0)
    confidence = clamp(
        0.35 + min(abs(spread) / 0.03, 1.0) * 0.35 + (0.2 if slope_agrees else 0.0),
        0.35, 0.9,
    )

    if score > 0:
        mu = _drift_daily_from_annual(clamp(_trailing_return(closes, 100), -0.25, 0.35))
        sigma = _annual_vol(closes) / math.sqrt(_TD)
    else:
        mu = _rf_daily(ctx)
        sigma = 0.02 / math.sqrt(_TD)
    horizons = _project(mu, sigma)

    rationale = (
        f"Fast SMA20 {f:.2f} is {spread * 100:+.1f}% vs slow SMA100 {s:.2f} "
        f"({'fast above slow (long)' if score > 0 else 'fast below slow (exit/short)'}"
        f"{', fresh cross' if fresh else ''})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "fastMa": f,
            "slowMa": s,
            "spread": spread,
            "slowSlope": slow_slope,
            "fresh": 1.0 if fresh else 0.0,
        },
        horizons=horizons,
    )


def _build_supertrend(ctx: "AnalysisContext") -> StrategySignal:
    """Supertrend (ATR trailing trend) signal.

    Catalog ``computeSignal``: with the Supertrend line and ATR(10, mult 3),
    ``score = clamp(((close-Supertrend)/(3*ATR))*100, -100, 100)`` (one full band
    away saturates +-100); confidence:
    ``clamp(0.4 + min(ATR_pct_rank, 1)*0.2 + persistence_bars/20*0.3, 0.4, 0.9)``.
    """
    meta = _meta("supertrend")
    closes = _closes(ctx)
    highs = _highs(ctx)
    lows = _lows(ctx)
    if closes.size == 0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No price history for Supertrend.", meta.formula,
            metrics={}, horizons=[],
        )

    p = float(closes[-1])
    line, direction = indicators.supertrend(highs, lows, closes, n=10, mult=3.0)
    atr_series = indicators.atr(highs, lows, closes, n=10)
    atr_n = float(atr_series[-1]) if atr_series.size else 0.0
    band = 3.0 * atr_n
    if band > 1e-9:
        score = clamp((p - line) / band * 100.0, -100.0, 100.0)
    else:
        score = 100.0 * (1.0 if direction >= 0 else -1.0)

    # ATR percent-rank within its own history (a rough vol regime gauge).
    atr_pct_rank = 0.5
    if atr_series.size > 10:
        atr_pct_rank = float(np.mean(atr_series[-_TD:] <= atr_n)) if atr_series.size else 0.5
    # Persistence: how many recent bars the trend has held its side of the line.
    persistence = _supertrend_persistence(highs, lows, closes)
    confidence = clamp(
        0.4 + min(atr_pct_rank, 1.0) * 0.2 + min(persistence, 20) / 20.0 * 0.3,
        0.4, 0.9,
    )

    mu = _drift_daily_from_annual(clamp(_trailing_return(closes, 63) * 4.0, -0.25, 0.35)) if direction >= 0 else 0.0
    sigma = (atr_n / p) if p > 0 and atr_n > 0 else _annual_vol(closes) / math.sqrt(_TD)
    horizons = _project(mu, sigma) if direction >= 0 else []

    rationale = (
        f"Close {p:.2f} is {'above' if direction >= 0 else 'below'} the Supertrend "
        f"line {line:.2f} (trend {'up' if direction >= 0 else 'down'}, "
        f"{abs(p - line):.2f} = {abs(score):.0f}% of a {band:.2f} band); held "
        f"{persistence} bar(s)."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "price": p,
            "supertrend": line,
            "direction": float(direction),
            "atr": atr_n,
            "persistenceBars": float(persistence),
        },
        horizons=horizons,
    )


def _supertrend_persistence(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, max_bars: int = 30
) -> int:
    """Count how many recent bars the Supertrend direction has stayed constant.

    Recomputes the Supertrend direction over progressively longer trailing
    windows; the count of trailing bars sharing the latest direction is a cheap
    persistence proxy. Bounded by ``max_bars`` for cost.

    Args:
        highs: High series.
        lows: Low series.
        closes: Close series.
        max_bars: Maximum look-back (default 30).

    Returns:
        An integer persistence count in ``[0, max_bars]``.
    """
    n = int(closes.size)
    if n < 3:
        return 0
    _, latest_dir = indicators.supertrend(highs, lows, closes, n=10, mult=3.0)
    count = 0
    limit = min(max_bars, n - 2)
    for k in range(1, limit + 1):
        _, d = indicators.supertrend(highs[: n - k], lows[: n - k], closes[: n - k], n=10, mult=3.0)
        if d == latest_dir:
            count += 1
        else:
            break
    return count


def _build_ichimoku(ctx: "AnalysisContext") -> StrategySignal:
    """Ichimoku Kinko Hyo (cloud) signal.

    Catalog ``computeSignal``: component votes +1 each for {price>cloud,
    Tenkan>Kijun, SpanA>SpanB, close>close[26]}, -1 for the bearish mirror;
    ``raw = sum/4`` in [-1,1]; cloud-distance strength
    ``W = clip((close-cloud_mid)/(cloud_thickness or price*0.05), -1, 1)``;
    ``score = clamp(raw*(0.6+0.4*|W|)*100, -100, 100)``; confidence:
    ``clamp(0.4 + 0.3*|raw| + 0.2*min(cloud_thickness/(price*0.03), 1), 0.4, 0.9)``.
    """
    meta = _meta("ichimoku")
    closes = _closes(ctx)
    highs = _highs(ctx)
    lows = _lows(ctx)
    if closes.size == 0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No price history for Ichimoku.", meta.formula,
            metrics={}, horizons=[],
        )

    lines = indicators.ichimoku(highs, lows, closes)
    tenkan = float(lines["tenkan"])
    kijun = float(lines["kijun"])
    span_a = float(lines["senkou_a"])
    span_b = float(lines["senkou_b"])
    cloud_pos = float(lines["cloud_pos"])
    p = float(closes[-1])

    votes = 0
    votes += 1 if cloud_pos > 0 else (-1 if cloud_pos < 0 else 0)
    votes += 1 if tenkan > kijun else (-1 if tenkan < kijun else 0)
    votes += 1 if span_a > span_b else (-1 if span_a < span_b else 0)
    # Chikou: today's close vs the close 26 bars ago.
    if closes.size > 26:
        votes += 1 if p > float(closes[-27]) else (-1 if p < float(closes[-27]) else 0)
    raw = clamp(votes / 4.0, -1.0, 1.0)

    cloud_top = max(span_a, span_b)
    cloud_bot = min(span_a, span_b)
    cloud_mid = 0.5 * (cloud_top + cloud_bot)
    cloud_thickness = cloud_top - cloud_bot
    denom = cloud_thickness if cloud_thickness > 1e-9 else p * 0.05
    w = clamp((p - cloud_mid) / denom, -1.0, 1.0) if denom > 0 else 0.0
    score = clamp(raw * (0.6 + 0.4 * abs(w)) * 100.0, -100.0, 100.0)

    confidence = clamp(
        0.4 + 0.3 * abs(raw) + 0.2 * min(cloud_thickness / (p * 0.03) if p > 0 else 0.0, 1.0),
        0.4, 0.9,
    )

    mu = _drift_daily_from_annual(clamp(_trailing_return(closes, 63) * 4.0, -0.25, 0.35)) if score > 0 else 0.0
    horizons = _project(mu, _annual_vol(closes) / math.sqrt(_TD)) if score > 0 else []

    rationale = (
        f"Price {p:.2f} is {'above' if cloud_pos > 0 else 'below' if cloud_pos < 0 else 'inside'} "
        f"the Kumo [{cloud_bot:.2f}, {cloud_top:.2f}]; Tenkan {tenkan:.2f} "
        f"{'>' if tenkan > kijun else '<='} Kijun {kijun:.2f}; component vote "
        f"{raw:+.2f}, cloud strength {w:+.2f}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "tenkan": tenkan,
            "kijun": kijun,
            "senkouA": span_a,
            "senkouB": span_b,
            "cloudPos": cloud_pos,
            "vote": raw,
            "cloudStrength": w,
        },
        horizons=horizons,
    )


def _build_adx_trend_strength(ctx: "AnalysisContext") -> StrategySignal:
    """ADX trend-strength filter signal (Wilder DMI).

    Catalog ``computeSignal``: from ADX(14), +DI, -DI: ``direction = sign(+DI - -DI)``;
    ``strength = clip((ADX-20)/30, 0, 1)`` (ADX=20 -> 0, 50 -> 1);
    ``score = clamp(direction*strength*100, -100, 100)``; confidence:
    ``clamp(0.4 + 0.4*strength + 0.1*(rising ADX ? 1 : 0), 0.4, 0.9)``.
    """
    meta = _meta("adx-trend-strength")
    closes = _closes(ctx)
    highs = _highs(ctx)
    lows = _lows(ctx)
    if closes.size < 2:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "Insufficient price history for ADX.", meta.formula,
            metrics={}, horizons=[],
        )

    plus_di, minus_di, adx_val = indicators.adx_components(highs, lows, closes, n=14)
    direction = 1.0 if plus_di > minus_di else (-1.0 if minus_di > plus_di else 0.0)
    strength = clamp((adx_val - 20.0) / 30.0, 0.0, 1.0)
    score = clamp(direction * strength * 100.0, -100.0, 100.0)

    # Rising ADX? compare to the ADX 5 bars ago.
    rising = False
    if closes.size > 7:
        prev_adx = indicators.adx(highs[:-5], lows[:-5], closes[:-5], n=14)
        rising = adx_val > prev_adx
    confidence = clamp(0.4 + 0.4 * strength + (0.1 if rising else 0.0), 0.4, 0.9)

    # ADX is best as a regime multiplier; a strong directional trend implies drift.
    if score > 0:
        mu = _drift_daily_from_annual(clamp(_trailing_return(closes, 63) * 4.0 * strength, -0.25, 0.35))
        horizons = _project(mu, _annual_vol(closes) / math.sqrt(_TD))
    else:
        horizons = []

    if adx_val >= 25:
        regime = "trending"
    elif adx_val < 20:
        regime = "range-bound (chop)"
    else:
        regime = "transitional"
    rationale = (
        f"ADX(14) {adx_val:.0f} ({regime}); +DI {plus_di:.0f} vs -DI {minus_di:.0f} "
        f"-> {'bullish' if direction > 0 else 'bearish' if direction < 0 else 'no'} "
        f"direction at strength {strength:.2f}{' (rising)' if rising else ''}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "adx": adx_val,
            "plusDi": plus_di,
            "minusDi": minus_di,
            "direction": direction,
            "strength": strength,
        },
        horizons=horizons,
    )


def _build_ma_ribbon(ctx: "AnalysisContext") -> StrategySignal:
    """Moving-average ribbon alignment signal (GMMA-style).

    Catalog ``computeSignal``: with SMAs {10,20,50,100,200},
    ``ordered_score = (bullish-ordered pairs - bearish-ordered pairs)/total pairs``
    in [-1,1]; ``W = mean(|SMA_i - SMA_{i+1}|)/price`` (fan width);
    ``score = clamp(ordered_score*(0.6+0.4*min(W/0.04, 1))*100, -100, 100)``;
    confidence: ``clamp(0.3 + |ordered_score|*0.4 + min(W/0.04, 1)*0.25, 0.3, 0.9)``.
    """
    meta = _meta("ma-ribbon")
    closes = _closes(ctx)
    if closes.size == 0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No price history for MA ribbon.", meta.formula,
            metrics={}, horizons=[],
        )

    windows = [10, 20, 50, 100, 200]
    smas: list[float] = []
    for w in windows:
        s = technical.sma(closes, w)
        smas.append(float(s[-1]) if s.size else float(closes[-1]))

    # Pairwise ordering: a "bullish" pair is shorter-window SMA > longer-window.
    bull_pairs = 0
    bear_pairs = 0
    total_pairs = 0
    for i in range(len(smas) - 1):
        total_pairs += 1
        if smas[i] > smas[i + 1]:
            bull_pairs += 1
        elif smas[i] < smas[i + 1]:
            bear_pairs += 1
    ordered_score = (bull_pairs - bear_pairs) / total_pairs if total_pairs else 0.0

    p = float(closes[-1])
    gaps = [abs(smas[i] - smas[i + 1]) for i in range(len(smas) - 1)]
    fan_w = (float(np.mean(gaps)) / p) if (gaps and p > 0) else 0.0
    sep_factor = min(fan_w / 0.04, 1.0)
    score = clamp(ordered_score * (0.6 + 0.4 * sep_factor) * 100.0, -100.0, 100.0)

    confidence = clamp(
        0.3 + abs(ordered_score) * 0.4 + sep_factor * 0.25,
        0.3, 0.9,
    )

    if score > 0:
        mu = _drift_daily_from_annual(clamp(_trailing_return(closes, _TD) * sep_factor, -0.25, 0.35))
        horizons = _project(mu, _annual_vol(closes) / math.sqrt(_TD))
    else:
        horizons = []

    rationale = (
        f"MA ribbon {bull_pairs}/{total_pairs} pairs stacked bullishly "
        f"(alignment {ordered_score:+.2f}, fan width {fan_w * 100:.1f}% of price) -> "
        f"{'strong uptrend' if score > 40 else 'strong downtrend' if score < -40 else 'tangled/weak'}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "sma10": smas[0],
            "sma20": smas[1],
            "sma50": smas[2],
            "sma100": smas[3],
            "sma200": smas[4],
            "orderedScore": ordered_score,
            "fanWidth": fan_w,
        },
        horizons=horizons,
    )


def _build_absolute_momentum_overlay(ctx: "AnalysisContext") -> StrategySignal:
    """Absolute momentum trend overlay signal (Antonacci).

    Catalog ``computeSignal``: ``r_12m = Close[t]/Close[t-252]-1``;
    ``rf_12m`` = trailing 12m risk-free; ``excess = r_12m - rf_12m``;
    ``sigma = sqrt(252*var of daily returns)``;
    ``score = clip(100*tanh((excess/sigma)*1.5), -100, 100)``;
    ``confidence = min(1, history/252)``, rising with ``|excess|/sigma``.
    """
    meta = _meta("absolute-momentum-overlay")
    closes = _closes(ctx)
    if closes.size < 2:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "Insufficient price history for absolute momentum overlay.", meta.formula,
            metrics={}, horizons=[],
        )

    r_12m = _trailing_return(closes, _TD)
    rf_12m = _rf_daily(ctx) * _TD
    excess = r_12m - rf_12m
    sigma = _annual_vol(closes)
    ratio = excess / sigma if sigma > 0 else 0.0
    score = clamp(100.0 * math.tanh(ratio * 1.5), -100.0, 100.0)

    confidence = clamp(
        _history_conf(closes) * clamp(0.4 + 0.6 * min(1.0, abs(ratio)), 0.4, 1.0),
        0.1, 1.0,
    )

    if score > 0:
        mu = _drift_daily_from_annual(clamp(excess, -0.25, 0.35))
        sigma_d = sigma / math.sqrt(_TD)
    else:
        mu = _rf_daily(ctx)
        sigma_d = 0.02 / math.sqrt(_TD)
    horizons = _project(mu, sigma_d)

    rationale = (
        f"Trailing 12m return {r_12m * 100:+.1f}% beats the 12m cash hurdle "
        f"{rf_12m * 100:.1f}% by {excess * 100:+.1f}% (vol {sigma * 100:.0f}%) -> "
        f"{'risk-on (stay invested)' if score > 0 else 'risk-off (go to cash)'}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "r12m": r_12m,
            "rf12m": rf_12m,
            "excess": excess,
            "sigmaAnnual": sigma,
            "ratio": ratio,
        },
        horizons=horizons,
    )


# ---------------------------------------------------------------------------
# Vectorized position-series generators (for the backtester)
# ---------------------------------------------------------------------------
#
# Each ``positions(closes, highs, lows, volumes, params) -> np.ndarray`` returns a
# target-exposure series in {0, 1} (long-or-flat trend systems) or [-1, 1] aligned
# to ``closes``. They never raise; degenerate input yields an all-zero series.

def _pos_array(closes: np.ndarray | list[float] | None) -> np.ndarray:
    """Clean ``closes`` into a finite positive ``float64`` array for positions."""
    arr = np.asarray(closes, dtype=np.float64).ravel() if closes is not None else np.empty(0)
    if arr.size == 0:
        return arr.astype(np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _positions_tsmom(
    closes: np.ndarray | list[float],
    highs: np.ndarray | list[float] | None = None,
    lows: np.ndarray | list[float] | None = None,
    volumes: np.ndarray | list[float] | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Time-series momentum positions: long when trailing 252d return > 0, else short.

    ``position[t] = +1`` if ``Close[t]/Close[t-lookback] - 1 > 0`` else ``-1``
    (long/short). The first ``lookback`` bars (no full window) are flat (0).
    """
    c = _pos_array(closes)
    n = c.size
    pos = np.zeros(n, dtype=np.float64)
    if n < 2:
        return pos
    lb = int((params or {}).get("lookback", _TD))
    lb = max(1, lb)
    for t in range(n):
        ref_idx = t - lb
        if ref_idx < 0 or c[ref_idx] <= 0.0:
            continue
        r = c[t] / c[ref_idx] - 1.0
        pos[t] = 1.0 if r > 0.0 else -1.0
    return pos


def _positions_faber_taa(
    closes: np.ndarray | list[float],
    highs: np.ndarray | list[float] | None = None,
    lows: np.ndarray | list[float] | None = None,
    volumes: np.ndarray | list[float] | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Faber TAA positions: long (1) when close > 200d SMA, else cash (0)."""
    c = _pos_array(closes)
    n = c.size
    pos = np.zeros(n, dtype=np.float64)
    if n == 0:
        return pos
    window = int((params or {}).get("window", 200))
    sma = technical.sma(c, window)
    m = min(n, sma.size)
    pos[-m:] = np.where(c[-m:] > sma[-m:], 1.0, 0.0)
    return pos


def _positions_golden_cross(
    closes: np.ndarray | list[float],
    highs: np.ndarray | list[float] | None = None,
    lows: np.ndarray | list[float] | None = None,
    volumes: np.ndarray | list[float] | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Golden-cross positions: long (1) when SMA50 > SMA200, else cash (0)."""
    c = _pos_array(closes)
    n = c.size
    pos = np.zeros(n, dtype=np.float64)
    if n == 0:
        return pos
    p = params or {}
    sfast = technical.sma(c, int(p.get("fast", 50)))
    sslow = technical.sma(c, int(p.get("slow", 200)))
    m = min(n, sfast.size, sslow.size)
    pos[-m:] = np.where(sfast[-m:] > sslow[-m:], 1.0, 0.0)
    return pos


def _positions_dual_ma(
    closes: np.ndarray | list[float],
    highs: np.ndarray | list[float] | None = None,
    lows: np.ndarray | list[float] | None = None,
    volumes: np.ndarray | list[float] | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Dual-MA crossover positions: long (1) when fast SMA > slow SMA, else cash."""
    c = _pos_array(closes)
    n = c.size
    pos = np.zeros(n, dtype=np.float64)
    if n == 0:
        return pos
    p = params or {}
    sfast = technical.sma(c, int(p.get("fast", 20)))
    sslow = technical.sma(c, int(p.get("slow", 100)))
    m = min(n, sfast.size, sslow.size)
    pos[-m:] = np.where(sfast[-m:] > sslow[-m:], 1.0, 0.0)
    return pos


def _positions_absolute_momentum(
    closes: np.ndarray | list[float],
    highs: np.ndarray | list[float] | None = None,
    lows: np.ndarray | list[float] | None = None,
    volumes: np.ndarray | list[float] | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Absolute-momentum overlay positions: long (1) when 252d return > 0, else cash."""
    c = _pos_array(closes)
    n = c.size
    pos = np.zeros(n, dtype=np.float64)
    if n < 2:
        return pos
    lb = max(1, int((params or {}).get("lookback", _TD)))
    for t in range(n):
        ref_idx = t - lb
        if ref_idx < 0 or c[ref_idx] <= 0.0:
            continue
        pos[t] = 1.0 if (c[t] / c[ref_idx] - 1.0) > 0.0 else 0.0
    return pos


def _positions_donchian(
    closes: np.ndarray | list[float],
    highs: np.ndarray | list[float] | None = None,
    lows: np.ndarray | list[float] | None = None,
    volumes: np.ndarray | list[float] | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Donchian Turtle System-1 positions: long on a 20d high, exit on a 10d low.

    State machine: enter long (1) when close breaks above the prior-20-day high,
    exit to flat (0) when close breaks below the prior-10-day low. Holds between.
    """
    c = _pos_array(closes)
    h = _pos_array(highs) if highs is not None else c
    l = _pos_array(lows) if lows is not None else c
    n = c.size
    pos = np.zeros(n, dtype=np.float64)
    if n < 2:
        return pos
    if h.size != n:
        h = c
    if l.size != n:
        l = c
    p = params or {}
    entry = max(1, int(p.get("entry", 20)))
    exit_ = max(1, int(p.get("exit", 10)))
    state = 0.0
    for t in range(1, n):
        lo_entry = max(0, t - entry)
        lo_exit = max(0, t - exit_)
        hh = float(np.max(h[lo_entry:t])) if t > lo_entry else float(h[t])
        ll = float(np.min(l[lo_exit:t])) if t > lo_exit else float(l[t])
        if state <= 0.0 and c[t] > hh:
            state = 1.0
        elif state > 0.0 and c[t] < ll:
            state = 0.0
        pos[t] = state
    return pos


def _positions_supertrend(
    closes: np.ndarray | list[float],
    highs: np.ndarray | list[float] | None = None,
    lows: np.ndarray | list[float] | None = None,
    volumes: np.ndarray | list[float] | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Supertrend positions: long (1) while trend up, flat (0) while trend down.

    Replays the Supertrend direction bar-by-bar (long when close is above the
    trailing line). Long-or-flat (no shorting) to match the timing-overlay use.
    """
    c = _pos_array(closes)
    h = _pos_array(highs) if highs is not None else c
    l = _pos_array(lows) if lows is not None else c
    n = c.size
    pos = np.zeros(n, dtype=np.float64)
    if n < 2:
        return pos
    if h.size != n:
        h = c
    if l.size != n:
        l = c
    p = params or {}
    period = int(p.get("n", 10))
    mult = float(p.get("mult", 3.0))

    m = float(mult)
    atr_series = indicators.atr(h, l, c, period)
    hl2 = (h + l) / 2.0
    basic_upper = hl2 + m * atr_series
    basic_lower = hl2 - m * atr_series
    final_upper = np.empty(n, dtype=np.float64)
    final_lower = np.empty(n, dtype=np.float64)
    final_upper[0] = basic_upper[0]
    final_lower[0] = basic_lower[0]
    for t in range(1, n):
        if basic_upper[t] < final_upper[t - 1] or c[t - 1] > final_upper[t - 1]:
            final_upper[t] = basic_upper[t]
        else:
            final_upper[t] = final_upper[t - 1]
        if basic_lower[t] > final_lower[t - 1] or c[t - 1] < final_lower[t - 1]:
            final_lower[t] = basic_lower[t]
        else:
            final_lower[t] = final_lower[t - 1]
    direction = 1
    pos[0] = 1.0
    for t in range(1, n):
        if direction == 1:
            if c[t] < final_lower[t]:
                direction = -1
        else:
            if c[t] > final_upper[t]:
                direction = 1
        pos[t] = 1.0 if direction == 1 else 0.0
    return pos


def _positions_ichimoku(
    closes: np.ndarray | list[float],
    highs: np.ndarray | list[float] | None = None,
    lows: np.ndarray | list[float] | None = None,
    volumes: np.ndarray | list[float] | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """Ichimoku positions: long (1) when price is above the cloud, else cash (0).

    Computes the leading spans from each bar's trailing windows and goes long
    when the close is above both spans (above the cloud), flat otherwise.
    """
    c = _pos_array(closes)
    h = _pos_array(highs) if highs is not None else c
    l = _pos_array(lows) if lows is not None else c
    n = c.size
    pos = np.zeros(n, dtype=np.float64)
    if n == 0:
        return pos
    if h.size != n:
        h = c
    if l.size != n:
        l = c
    p = params or {}
    tenkan_n = int(p.get("tenkan", 9))
    kijun_n = int(p.get("kijun", 26))
    span_b_n = int(p.get("span_b", 52))

    def _mid(arr_h: np.ndarray, arr_l: np.ndarray, t: int, w: int) -> float:
        start = max(0, t - w + 1)
        return (float(np.max(arr_h[start : t + 1])) + float(np.min(arr_l[start : t + 1]))) / 2.0

    for t in range(n):
        ten = _mid(h, l, t, tenkan_n)
        kij = _mid(h, l, t, kijun_n)
        span_a = (ten + kij) / 2.0
        span_b = _mid(h, l, t, span_b_n)
        cloud_top = max(span_a, span_b)
        pos[t] = 1.0 if c[t] > cloud_top else 0.0
    return pos


def _positions_adx_trend(
    closes: np.ndarray | list[float],
    highs: np.ndarray | list[float] | None = None,
    lows: np.ndarray | list[float] | None = None,
    volumes: np.ndarray | list[float] | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """ADX-gated trend positions: long (1) when +DI>-DI and ADX>threshold, else cash.

    Uses the directional movement system: go long only when the trend is up
    (+DI > -DI) AND strong enough (ADX above the threshold, default 25). Replays
    bar-by-bar over expanding windows (bounded cost via a coarse step is avoided
    here for correctness; the series length equals ``closes``).
    """
    c = _pos_array(closes)
    h = _pos_array(highs) if highs is not None else c
    l = _pos_array(lows) if lows is not None else c
    n = c.size
    pos = np.zeros(n, dtype=np.float64)
    if n < 3:
        return pos
    if h.size != n:
        h = c
    if l.size != n:
        l = c
    p = params or {}
    period = int(p.get("n", 14))
    threshold = float(p.get("threshold", 25.0))

    # Vectorized DMI over the full series, then take running latest values.
    up = h[1:] - h[:-1]
    down = l[:-1] - l[1:]
    plus_dm = np.where((up > down) & (up > 0.0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0.0), down, 0.0)
    tr = indicators.true_range(h, l, c)[1:]
    atr_s = indicators._wilder_rma(tr, period)
    plus_dm_s = indicators._wilder_rma(plus_dm, period)
    minus_dm_s = indicators._wilder_rma(minus_dm, period)
    safe_atr = np.where(atr_s > 1e-12, atr_s, np.nan)
    plus_di = np.nan_to_num(100.0 * plus_dm_s / safe_atr, nan=0.0, posinf=0.0, neginf=0.0)
    minus_di = np.nan_to_num(100.0 * minus_dm_s / safe_atr, nan=0.0, posinf=0.0, neginf=0.0)
    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 1e-12, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)
    adx_s = indicators._wilder_rma(dx, period)  # length n-1, aligned to bars 1..n-1

    long_mask = (plus_di > minus_di) & (adx_s > threshold)
    pos[1:] = np.where(long_mask, 1.0, 0.0)
    return pos


def _positions_ma_ribbon(
    closes: np.ndarray | list[float],
    highs: np.ndarray | list[float] | None = None,
    lows: np.ndarray | list[float] | None = None,
    volumes: np.ndarray | list[float] | None = None,
    params: dict[str, Any] | None = None,
) -> np.ndarray:
    """MA-ribbon positions: long (1) when the ribbon is fully bull-stacked, else cash.

    Long when SMA10 > SMA20 > SMA50 > SMA100 > SMA200 (full bullish alignment),
    flat otherwise. Long-or-cash trend overlay.
    """
    c = _pos_array(closes)
    n = c.size
    pos = np.zeros(n, dtype=np.float64)
    if n == 0:
        return pos
    windows = (params or {}).get("windows", [10, 20, 50, 100, 200])
    sma_series = [technical.sma(c, int(w)) for w in windows]
    m = min([n] + [s.size for s in sma_series])
    if m == 0:
        return pos
    stacked = np.ones(m, dtype=bool)
    for i in range(len(sma_series) - 1):
        a = sma_series[i][-m:]
        b = sma_series[i + 1][-m:]
        stacked &= a > b
    pos[-m:] = np.where(stacked, 1.0, 0.0)
    return pos


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

#: id -> (StrategyMeta, builder fn). The engine/registry merges this into the
#: global registry; ``build_signals`` runs each guarded.
BUILDERS: dict[str, tuple[StrategyMeta, Callable[["AnalysisContext"], StrategySignal]]] = {
    "dual-momentum": (_META["dual-momentum"], _build_dual_momentum),
    "tsmom": (_META["tsmom"], _build_tsmom),
    "cross-sectional-momentum": (_META["cross-sectional-momentum"], _build_cross_sectional_momentum),
    "52w-high": (_META["52w-high"], _build_52w_high),
    "relative-strength-rotation": (_META["relative-strength-rotation"], _build_relative_strength_rotation),
    "frog-in-the-pan-momentum": (_META["frog-in-the-pan-momentum"], _build_frog_in_the_pan),
    "faber-taa": (_META["faber-taa"], _build_faber_taa),
    "donchian-turtle": (_META["donchian-turtle"], _build_donchian_turtle),
    "golden-cross": (_META["golden-cross"], _build_golden_cross),
    "dual-ma-crossover": (_META["dual-ma-crossover"], _build_dual_ma_crossover),
    "supertrend": (_META["supertrend"], _build_supertrend),
    "ichimoku": (_META["ichimoku"], _build_ichimoku),
    "adx-trend-strength": (_META["adx-trend-strength"], _build_adx_trend_strength),
    "ma-ribbon": (_META["ma-ribbon"], _build_ma_ribbon),
    "absolute-momentum-overlay": (_META["absolute-momentum-overlay"], _build_absolute_momentum_overlay),
}

#: id -> vectorized position-series generator for the backtester. Cross-sectional
#: strategies (dual-momentum, cross-sectional-momentum, 52w-high,
#: relative-strength-rotation, frog-in-the-pan-momentum) are NOT per-bar
#: single-asset backtestable and are intentionally omitted.
POSITION_FUNCS: dict[str, Callable[..., np.ndarray]] = {
    "tsmom": _positions_tsmom,
    "faber-taa": _positions_faber_taa,
    "golden-cross": _positions_golden_cross,
    "dual-ma-crossover": _positions_dual_ma,
    "absolute-momentum-overlay": _positions_absolute_momentum,
    "donchian-turtle": _positions_donchian,
    "supertrend": _positions_supertrend,
    "ichimoku": _positions_ichimoku,
    "adx-trend-strength": _positions_adx_trend,
    "ma-ribbon": _positions_ma_ribbon,
}
