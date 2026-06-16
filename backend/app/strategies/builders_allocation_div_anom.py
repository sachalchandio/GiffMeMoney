"""Allocation / risk / dividend / anomaly strategy builders.

This module implements the fourth catalog group from
``docs/research/strategy-catalog.json`` (see ``docs/STRATEGIES-V2.md`` §5):

    Allocation / risk
        * ``all-weather-risk-parity`` — Ray Dalio All-Weather sleeve overlay.
        * ``vol-target`` — volatility targeting (scale exposure inversely to vol).
        * ``risk-parity-inverse-vol`` — naive inverse-volatility risk parity.
        * ``min-variance`` — minimum-variance portfolio weight tilt.
        * ``permanent-portfolio`` — Harry Browne 25/25/25/25 sleeve overlay.

    Anomaly / factor
        * ``low-vol-anomaly`` — long the lowest-volatility names.
        * ``betting-against-beta`` — long low-beta, short high-beta (BAB).
        * ``seasonality`` — Halloween / Sell-in-May + turn-of-month calendar tilt.

    Dividend / income
        * ``chowder-rule`` — yield + dividend-growth total-return screen.
        * ``dividend-safety`` — yield + payout/coverage sustainability.
        * ``dividend-growth-aristocrats`` — durable dividend growth / yield-on-cost.
        * ``shareholder-yield`` — Meb Faber total cash returned to holders.
        * ``dogs-of-dow`` — high-yield blue-chip mean-reversion.
        * ``small-dogs-of-dow`` — the low-priced 5 concentrated Dogs variant.

Each builder consumes an :class:`~app.strategies.engine.AnalysisContext` (imported
only under :data:`typing.TYPE_CHECKING` to avoid a circular import) and returns a
fully-validated :class:`~app.schemas.StrategySignal` via
:func:`app.strategies.base.make_signal`. Cross-sectional strategies read the
universe-wide metrics from ``ctx.universe`` (a ``UniverseStats`` injected by the
engine; see §1). Because that field is added to ``AnalysisContext`` by the
integration agent, every access here is *defensive*: when the universe (or any
optional context attribute such as ``now`` / ``highs``) is absent, the builder
falls back to a self-contained estimate so the module imports and runs in
isolation and never raises.

Score convention is **positive = bullish** everywhere; scores are clamped to
``[-100, 100]`` and confidence to ``[0, 1]`` by :func:`make_signal`. Builders that
imply a forward drift attach 5-horizon projections via
:func:`app.quant.returns.project_horizons`; pure risk/allocation overlays leave
``horizons`` empty.

Module exports:
    * :data:`BUILDERS` — ``dict[str, tuple[StrategyMeta, builder_fn]]`` for the 14
      ids, consumed by :mod:`app.strategies.registry`.
    * :data:`POSITION_FUNCS` — vectorized position series for backtestable timing
      strategies in this group. None of these 14 are per-bar timing strategies
      (they are cross-sectional / fundamental / calendar overlays), so this is an
      empty dict.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from app.quant import metrics, returns, volatility
from app.schemas import StrategyMeta, StrategySignal
from app.strategies.base import clamp, make_signal, squash

if TYPE_CHECKING:  # pragma: no cover - typing-only imports (avoid circular import)
    from app.strategies.engine import AnalysisContext, UniverseStats

__all__ = ["BUILDERS", "POSITION_FUNCS"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Equity / ETF growth-sleeve symbols recognised by the allocation overlays.
_GROWTH_SYMBOLS: frozenset[str] = frozenset({"SPY", "QQQ", "VTI"})
#: Gold / commodity sleeve.
_GOLD_SYMBOLS: frozenset[str] = frozenset({"GLD"})

#: Annualised long-run risk premia per All-Weather / Permanent-Portfolio sleeve,
#: used only to give the allocation overlays a mild, plausible projection drift.
_SLEEVE_DRIFT: dict[str, float] = {
    "growth": 0.07,      # equities
    "gold": 0.04,        # gold / commodities
    "bond": 0.02,        # bond proxy (cash-like here)
    "cash": 0.01,        # cash
    "crypto": 0.10,      # crypto growth sleeve
}


# ---------------------------------------------------------------------------
# Defensive context accessors
# ---------------------------------------------------------------------------

def _universe(ctx: "AnalysisContext") -> "UniverseStats | None":
    """Return the engine's :class:`UniverseStats` if present, else ``None``.

    The ``universe`` field is added to :class:`AnalysisContext` by the
    integration agent; accessing it defensively lets this module import and run
    before that wiring exists.

    Args:
        ctx: The analysis context.

    Returns:
        The cross-sectional :class:`UniverseStats`, or ``None`` when unavailable.
    """
    uni = getattr(ctx, "universe", None)
    return uni


def _symbol(ctx: "AnalysisContext") -> str:
    """Return the upper-cased symbol for ``ctx`` (defensive)."""
    try:
        return str(ctx.asset.symbol).strip().upper()
    except Exception:  # pragma: no cover - defensive
        return ""


def _asset_class(ctx: "AnalysisContext") -> str:
    """Return the lower-cased asset class for ``ctx`` (defensive)."""
    try:
        return str(ctx.asset.asset_class).strip().lower()
    except Exception:  # pragma: no cover - defensive
        return ""


def _price(ctx: "AnalysisContext") -> float:
    """Return a positive price for ``ctx`` (falls back to the seed base price)."""
    try:
        p = float(ctx.asset.price)
        if math.isfinite(p) and p > 0.0:
            return p
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        bp = float(ctx.seed.base_price)
        if math.isfinite(bp) and bp > 0.0:
            return bp
    except Exception:  # pragma: no cover - defensive
        pass
    return 1.0


def _history_days(ctx: "AnalysisContext") -> int:
    """Return the number of available return observations (history length)."""
    try:
        r = np.asarray(ctx.returns, dtype=np.float64).ravel()
        return int(r[np.isfinite(r)].size)
    except Exception:  # pragma: no cover - defensive
        return 0


def _annual_vol(ctx: "AnalysisContext") -> float:
    """Return the asset's annualised volatility (universe value if available)."""
    sym = _symbol(ctx)
    uni = _universe(ctx)
    if uni is not None:
        try:
            av = uni.annual_vol.get(sym)
            if av is not None and math.isfinite(float(av)) and float(av) > 0.0:
                return float(av)
        except Exception:  # pragma: no cover - defensive
            pass
    vol = metrics.annual_volatility(ctx.returns)
    return float(vol) if math.isfinite(vol) and vol > 0.0 else 0.0


def _beta(ctx: "AnalysisContext") -> float:
    """Return the asset's market beta (universe value if available)."""
    sym = _symbol(ctx)
    uni = _universe(ctx)
    if uni is not None:
        try:
            bv = uni.beta.get(sym)
            if bv is not None and math.isfinite(float(bv)):
                return float(bv)
        except Exception:  # pragma: no cover - defensive
            pass
    b = metrics.beta(ctx.returns, ctx.market_ret)
    return float(b) if math.isfinite(b) else 1.0


def _universe_percentile(
    ctx: "AnalysisContext", metric: str, default: float = 0.5
) -> float | None:
    """Return ``ctx``'s 0..1 percentile for ``metric`` (1 = highest value).

    Args:
        ctx: The analysis context.
        metric: A :class:`UniverseStats` metric name (e.g. ``"annual_vol"``).
        default: Returned when the universe is present but the lookup misbehaves.

    Returns:
        The percentile in ``[0, 1]``, ``default`` on a soft failure, or ``None``
        when no universe is attached (so the caller can use a self-contained
        fallback instead of a meaningless 0.5).
    """
    uni = _universe(ctx)
    if uni is None:
        return None
    try:
        p = float(uni.percentile(metric, _symbol(ctx)))
        if math.isfinite(p):
            return clamp(p, 0.0, 1.0)
    except Exception:  # pragma: no cover - defensive
        pass
    return default


def _now_month_day(ctx: "AnalysisContext") -> tuple[int, int]:
    """Resolve the current ``(month, day)`` for the calendar strategies.

    The engine may inject a deterministic ``now`` (tests inject a fixed value);
    otherwise we fall back to the system clock. ``ctx.now`` may be a
    :class:`datetime.datetime`/:class:`datetime.date` or an object exposing
    ``month``/``day`` attributes.

    Args:
        ctx: The analysis context.

    Returns:
        A ``(month, day)`` tuple with ``month`` in ``1..12`` and ``day`` in
        ``1..31``.
    """
    now = getattr(ctx, "now", None)
    if now is not None:
        try:
            month = int(getattr(now, "month"))
            day = int(getattr(now, "day"))
            if 1 <= month <= 12 and 1 <= day <= 31:
                return month, day
        except Exception:  # pragma: no cover - defensive
            pass
    today = _dt.date.today()
    return today.month, today.day


# ---------------------------------------------------------------------------
# Strategy metadata
# ---------------------------------------------------------------------------

_META_ALL_WEATHER = StrategyMeta(
    id="all-weather-risk-parity",
    name="Ray Dalio All-Weather (Risk Parity)",
    category="Portfolio",
    summary=(
        "Bridgewater's strategic allocation balancing risk (not dollars) across "
        "economic seasons: ~30% growth/equities, 40% long bonds, 15% intermediate "
        "bonds, 7.5% gold, 7.5% commodities. The signal is each asset's "
        "over/underweight versus its All-Weather sleeve target."
    ),
    formula="score = clamp(100 * (target_w - current_w) / target_w, -100, 100)",
    inputs=["asset class", "sleeve mapping", "sleeve target weights"],
    references=[
        "Ray Dalio / Bridgewater - All-Weather principles",
        "Tony Robbins, 'Money: Master the Game' (2014) - retail All-Weather weights",
    ],
)

_META_VOL_TARGET = StrategyMeta(
    id="vol-target",
    name="Volatility Targeting",
    category="Portfolio",
    summary=(
        "Scale exposure inversely to forecast volatility to hold portfolio risk at "
        "a constant target. A calm regime (forecast vol below the long-run average) "
        "is mildly bullish on a risk-adjusted basis; a vol spike is bearish."
    ),
    formula="ratio = long_run_vol / forecast_vol; score = clip((ratio - 1) * 100, -100, 100)",
    inputs=["daily returns", "EWMA/GARCH forecast vol", "long-run vol"],
    references=[
        "Moreira & Muir (2017), 'Volatility-Managed Portfolios', Journal of Finance 72(4)",
        "Harvey et al. (2018), 'The Impact of Volatility Targeting', JPM",
    ],
)

_META_RISK_PARITY = StrategyMeta(
    id="risk-parity-inverse-vol",
    name="Risk Parity / Inverse-Volatility Weighting",
    category="Portfolio",
    summary=(
        "Equalize each asset's risk contribution by weighting inversely to "
        "volatility (naive risk parity). Low-vol assets earn a weight above "
        "equal-weight; the signal is the inverse-vol weight versus 1/N."
    ),
    formula="w_i ~ (1/sigma_i)/sum(1/sigma_j); score = clamp(100*(w_i - 1/N)/(1/N), -100, 100)",
    inputs=["annualised vol of each universe asset"],
    references=[
        "Maillard, Roncalli & Teiletche (2010), 'Equally Weighted Risk Contribution Portfolios', JPM",
        "Qian (2005), 'Risk Parity Portfolios', PanAgora",
    ],
)

_META_MIN_VARIANCE = StrategyMeta(
    id="min-variance",
    name="Minimum-Variance Portfolio",
    category="Portfolio",
    summary=(
        "The leftmost point of the efficient frontier: lowest total variance "
        "regardless of expected return. Low-beta, low-vol, low-correlation names "
        "earn the highest weights (the low-volatility anomaly). Proxied by a "
        "low-beta + low-vol rank when the full optimizer is unavailable."
    ),
    formula="minimize w'Sigma w s.t. sum(w)=1, w>=0; score = clamp(100*(w_i - 1/N)/(1/N), -100, 100)",
    inputs=["covariance proxy: beta percentile + vol percentile"],
    references=[
        "Clarke, de Silva & Thorley (2006), 'Minimum-Variance Portfolios in the US Equity Market', JPM",
        "Ledoit & Wolf (2004), 'Honey, I Shrunk the Sample Covariance Matrix'",
    ],
)

_META_PERMANENT = StrategyMeta(
    id="permanent-portfolio",
    name="Harry Browne Permanent Portfolio",
    category="Portfolio",
    summary=(
        "Harry Browne's 25% stocks / 25% long bonds / 25% cash / 25% gold - one "
        "sleeve thrives in each economic regime. The signal is each asset's "
        "over/underweight versus its 25% sleeve target."
    ),
    formula="score = clamp(100 * (0.25 - current_w) / 0.25, -100, 100) within sleeve",
    inputs=["asset class", "sleeve mapping", "25% sleeve targets"],
    references=[
        "Harry Browne, 'Fail-Safe Investing' (1999)",
        "Craig Rowland & J.M. Lawson, 'The Permanent Portfolio' (Wiley, 2012)",
    ],
)

_META_LOW_VOL = StrategyMeta(
    id="low-vol-anomaly",
    name="Low-Volatility Anomaly",
    category="Statistical",
    summary=(
        "Low-volatility (and low-beta) assets have historically earned higher "
        "risk-adjusted - and often higher absolute - returns than high-vol assets, "
        "contradicting CAPM. The signal ranks the universe by trailing vol: lowest "
        "vol is most bullish."
    ),
    formula="score = clip((low_vol_pct - 0.5) * 200, -100, 100)",
    inputs=["trailing 252-day annualised vol across the universe"],
    references=[
        "Ang, Hodrick, Xing & Zhang (2006), 'The Cross-Section of Volatility and Expected Returns', JoF 61(1)",
        "Baker, Bradley & Wurgler (2011), 'Benchmarks as Limits to Arbitrage', FAJ",
    ],
)

_META_BAB = StrategyMeta(
    id="betting-against-beta",
    name="Betting Against Beta (BAB)",
    category="Factor",
    summary=(
        "Frazzini & Pedersen (2014): low-beta assets earn higher risk-adjusted "
        "returns than high-beta because leverage-constrained investors bid up "
        "high-beta. BAB goes long low-beta, short high-beta. The signal ranks the "
        "universe by beta: low beta is bullish."
    ),
    formula="score = clip(-(beta - median_beta)/beta_std * 40, -100, 100)",
    inputs=["per-asset beta across the universe"],
    references=[
        "Frazzini & Pedersen (2014), 'Betting Against Beta', JFE 111(1):1-25",
    ],
)

_META_SEASONALITY = StrategyMeta(
    id="seasonality",
    name="Calendar Seasonality (Sell-in-May / Halloween + TOM)",
    category="Statistical",
    summary=(
        "Well-documented calendar effects: the Halloween / 'Sell in May' indicator "
        "(Nov-Apr returns far exceed May-Oct) and the turn-of-month effect (returns "
        "cluster around month-end / month-start). Strongest bullish when both "
        "windows align."
    ),
    formula="score = clamp(halloween_score + tom_score, -100, 100)",
    inputs=["current month", "current day-of-month"],
    references=[
        "Bouman & Jacobsen (2002), 'The Halloween Indicator, Sell in May and Go Away', AER 92(5)",
        "Ariel (1987), 'A Monthly Effect in Stock Returns', JFE",
    ],
)

_META_CHOWDER = StrategyMeta(
    id="chowder-rule",
    name="Chowder Rule (Dividend Growth)",
    category="Fundamental",
    summary=(
        "Dividend-growth screen (Seeking Alpha 'Chowder' / David Fish CCC lists): "
        "the Chowder Number = current yield + 5yr dividend-growth rate approximates "
        "potential total return with a margin of safety against sector thresholds."
    ),
    formula="chowder = yield% + dgr%; score = squash(chowder - T, 6)",
    inputs=["dividend", "price", "dividend growth", "EPS payout"],
    references=[
        "Sure Dividend, 'The Chowder Rule Explained'",
        "David Fish CCC Lists, DripInvesting.org",
    ],
)

_META_DIV_SAFETY = StrategyMeta(
    id="dividend-safety",
    name="Dividend Safety (Yield + Payout/Coverage)",
    category="Fundamental",
    summary=(
        "Income safety screen (CFA Institute; Simply Safe Dividends): combine an "
        "attractive yield with a sustainable payout. Dividends covered by a low "
        "fraction of earnings and FCF, with manageable leverage, are far less "
        "likely to be cut."
    ),
    formula="raw = (2*safety - 1)*0.6 + yieldAttr*0.4; score = clamp(100*raw, -100, 100)",
    inputs=["dividend", "price", "EPS", "FCF/share", "debt/equity", "current ratio"],
    references=[
        "CFA Institute, 'Analysis of Dividend Safety'",
        "Simply Safe Dividends - Dividend Safety Score methodology",
    ],
)

_META_DIV_GROWTH = StrategyMeta(
    id="dividend-growth-aristocrats",
    name="Dividend Growth / Aristocrats (Yield-on-Cost)",
    category="Fundamental",
    summary=(
        "The dividend-growth discipline behind the S&P 500 Dividend Aristocrats "
        "(25+ consecutive annual increases): own firms that reliably raise "
        "dividends so yield-on-cost compounds. Favours healthy 5-12% growth with "
        "payout headroom."
    ),
    formula="yoc10 = yield*(1+g)^10; score = clamp(0.6*growthScore + 0.4*yocBoost, -100, 100)",
    inputs=["dividend", "price", "dividend growth", "EPS payout"],
    references=[
        "S&P Dow Jones Indices, 'S&P 500 Dividend Aristocrats' methodology",
        "David Fish, 'U.S. Dividend Champions' (CCC)",
        "Gordon (1959) growth model",
    ],
)

_META_SHY = StrategyMeta(
    id="shareholder-yield",
    name="Shareholder Yield (Mebane Faber)",
    category="Fundamental",
    summary=(
        "Meb Faber (2013): total cash returned = dividend yield + net buyback yield "
        "+ net debt-paydown yield. Outperforms dividend yield alone because "
        "buybacks and deleveraging are tax-advantaged and harder to fake."
    ),
    formula="proxy = divYield + max(0,(fcf-div))/price + clamp(1-d/e,-0.5,0.5)*0.03; score = squash(proxy-0.04, 0.05)",
    inputs=["dividend", "price", "FCF/share", "debt/equity"],
    references=[
        "Mebane T. Faber, 'Shareholder Yield' (2013)",
        "https://en.wikipedia.org/wiki/Shareholder_yield",
    ],
)

_META_DOGS = StrategyMeta(
    id="dogs-of-dow",
    name="Dogs of the Dow (High-Yield Blue Chips)",
    category="Fundamental",
    summary=(
        "Michael O'Higgins (1991): each January buy the 10 highest-yielding Dow "
        "stocks, equal-weight, hold one year. An elevated yield on a quality blue "
        "chip signals a temporarily depressed price (value / mean-reversion)."
    ),
    formula="score = squash(yield_percentile - 0.5, 0.35); damp x0.6 if payout>1",
    inputs=["dividend", "price", "EPS payout", "cross-sectional yield percentile"],
    references=[
        "Michael O'Higgins & John Downes, 'Beating the Dow' (1991)",
        "https://en.wikipedia.org/wiki/Dogs_of_the_Dow",
    ],
)

_META_SMALL_DOGS = StrategyMeta(
    id="small-dogs-of-dow",
    name="Small Dogs of the Dow (Low-Priced 5)",
    category="Fundamental",
    summary=(
        "O'Higgins' concentrated variant: from the 10 Dogs take the 5 lowest-priced, "
        "equal-weight. Lower-priced shares historically show larger percentage "
        "moves, amplifying the rebound at the cost of diversification."
    ),
    formula="combined = 0.5*(yield_pct-0.5) + 0.5*((1-price_pct)-0.5); score = squash(combined, 0.3)",
    inputs=["dividend", "price", "cross-sectional yield + inverse-price ranks"],
    references=[
        "Michael O'Higgins & John Downes, 'Beating the Dow' (1991, rev.)",
        "AAII 'Dogs of the Dow: Low Priced 5' screen",
    ],
)


# ---------------------------------------------------------------------------
# Allocation / risk builders
# ---------------------------------------------------------------------------

def _build_all_weather(ctx: "AnalysisContext") -> StrategySignal:
    """All-Weather risk-parity sleeve overlay (per-asset over/underweight).

    Maps the asset to an All-Weather sleeve (growth/bond/gold/commodity) and
    scores its over/underweight versus a naive equal-cap current weight. Since we
    cannot observe the user's live portfolio, "current weight" is proxied by the
    equal-cap weight ``1/N`` across the universe (or ``1/24`` when no universe is
    attached) — so an asset that *should* be a large sleeve (heavy bonds/gold)
    reads as under-target and accumulate-worthy, while a thin sleeve reads
    over-target.
    """
    meta = _META_ALL_WEATHER
    sym = _symbol(ctx)
    cls = _asset_class(ctx)

    n = 24
    uni = _universe(ctx)
    if uni is not None:
        try:
            n = max(1, len(uni.symbols))
        except Exception:  # pragma: no cover - defensive
            n = 24
    current_w = 1.0 / float(n)

    # Map symbol/class -> (sleeve, target weight, confidence).
    if sym in _GROWTH_SYMBOLS or cls in ("equity", "etf"):
        if sym in _GOLD_SYMBOLS:
            sleeve, target, conf = "gold", 0.075, 0.7
        elif sym in _GROWTH_SYMBOLS:
            sleeve, target, conf = "growth", 0.30, 0.7
        elif cls == "equity":
            sleeve, target, conf = "growth", 0.30, 0.5
        else:  # other ETFs (bond-less universe) proxy the bond sleeve
            sleeve, target, conf = "bond", 0.55, 0.4
    elif sym in _GOLD_SYMBOLS:
        sleeve, target, conf = "gold", 0.075, 0.7
    elif cls == "crypto":
        sleeve, target, conf = "crypto", 0.075, 0.4
    else:
        # Unmapped name -> neutral.
        return make_signal(
            meta.id, meta.name, meta.category,
            score=0.0, confidence=0.2,
            rationale=f"{sym} maps to no All-Weather sleeve - neutral overlay.",
            formula=meta.formula,
            metrics={"currentWeight": current_w, "targetWeight": 0.0},
            horizons=[],
        )

    score = clamp(100.0 * (target - current_w) / target, -100.0, 100.0)
    rationale = (
        f"{sym} maps to the All-Weather {sleeve} sleeve (target {target * 100:.1f}% "
        f"vs equal-cap {current_w * 100:.1f}%): "
        f"{'under-target - accumulate' if score > 0 else 'over-target - trim'} "
        f"({score:+.0f})."
    )

    horizons = returns.project_horizons(
        _SLEEVE_DRIFT.get(sleeve, 0.0) / returns.TRADING_DAYS,
        max(_annual_vol(ctx) / math.sqrt(returns.TRADING_DAYS), 1e-4),
    ) if sleeve in ("growth", "crypto") else []

    return make_signal(
        meta.id, meta.name, meta.category, score, conf, rationale, meta.formula,
        metrics={
            "currentWeight": current_w,
            "targetWeight": target,
            "sleeveDrift": _SLEEVE_DRIFT.get(sleeve, 0.0),
        },
        horizons=horizons,
    )


def _build_vol_target(ctx: "AnalysisContext") -> StrategySignal:
    """Volatility-targeting signal: long-run vs forecast vol ratio.

    ``forecast_vol`` is the annualised EWMA(lambda=0.94) volatility; ``long_run_vol``
    is the annualised stdev over the full history. A ratio above 1 (forecast vol
    below the long-run average = a calm regime) is mildly bullish on a
    risk-adjusted basis; a ratio below 1 (a vol spike) is bearish.
    """
    meta = _META_VOL_TARGET
    r = np.asarray(ctx.returns, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    if r.size < 5:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.15,
            "Insufficient history to estimate a volatility regime.",
            meta.formula, metrics={}, horizons=[],
        )

    forecast_vol = float(volatility.ewma_vol(r, lam=0.94))
    long_run_vol = float(metrics.annual_volatility(r))
    if forecast_vol <= 1e-8 or long_run_vol <= 1e-8:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.2,
            "Degenerate (near-zero) volatility - no actionable vol regime.",
            meta.formula,
            metrics={"forecastVol": forecast_vol, "longRunVol": long_run_vol},
            horizons=[],
        )

    ratio = long_run_vol / forecast_vol
    score = clamp((ratio - 1.0) * 100.0, -100.0, 100.0)
    history_days = _history_days(ctx)
    confidence = clamp(0.4 + 0.3 * min(1.0, history_days / 252.0), 0.0, 0.8)

    target_weight = clamp(0.12 / forecast_vol, 0.0, 2.0)  # target_vol ~ 12%
    rationale = (
        f"Forecast vol {forecast_vol * 100:.1f}% vs long-run {long_run_vol * 100:.1f}% "
        f"(ratio {ratio:.2f}): {'calm regime - scale up' if ratio > 1 else 'stressed regime - scale down'}. "
        f"Vol-target weight ~{target_weight:.2f}x for a 12% target."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "forecastVol": forecast_vol,
            "longRunVol": long_run_vol,
            "ratio": ratio,
            "targetWeight": target_weight,
        },
        horizons=[],
    )


def _build_risk_parity(ctx: "AnalysisContext") -> StrategySignal:
    """Inverse-volatility (naive risk-parity) weight tilt vs equal-weight.

    Each asset's risk-parity weight is ``(1/sigma_i)/sum_j(1/sigma_j)``; the
    signal compares it to ``1/N``. Low-vol assets sit above equal-weight
    (accumulate); high-vol assets below (trim). When the full universe is
    available the sum is exact, otherwise the asset's vol is compared to a
    universe-typical vol prior.
    """
    meta = _META_RISK_PARITY
    sigma_i = _annual_vol(ctx)
    if sigma_i <= 1e-8:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.2,
            "Zero/undefined volatility - no inverse-vol weight.",
            meta.formula, metrics={}, horizons=[],
        )

    uni = _universe(ctx)
    inv_sum = 0.0
    n = 24
    if uni is not None:
        try:
            vols = [
                float(v)
                for v in uni.annual_vol.values()
                if v is not None and math.isfinite(float(v)) and float(v) > 1e-8
            ]
            n = max(1, len(uni.symbols))
            inv_sum = sum(1.0 / v for v in vols)
        except Exception:  # pragma: no cover - defensive
            inv_sum = 0.0
    if inv_sum <= 0.0:
        # Fallback: compare to a typical universe vol (~30% annual).
        typical_vol = 0.30
        iv_w = (1.0 / sigma_i) / ((1.0 / sigma_i) + (n - 1) / typical_vol)
    else:
        iv_w = (1.0 / sigma_i) / inv_sum

    eq_w = 1.0 / float(n)
    score = clamp(100.0 * (iv_w - eq_w) / eq_w, -100.0, 100.0)
    history_days = _history_days(ctx)
    confidence = clamp(0.5 + 0.3 * min(1.0, history_days / 252.0), 0.0, 0.85)

    rationale = (
        f"Annualised vol {sigma_i * 100:.1f}% -> inverse-vol weight {iv_w * 100:.1f}% "
        f"vs equal-weight {eq_w * 100:.1f}%: "
        f"{'low-vol, above equal-weight (accumulate)' if score > 0 else 'high-vol, below equal-weight (trim)'} "
        f"({score:+.0f})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "annualVol": sigma_i,
            "inverseVolWeight": iv_w,
            "equalWeight": eq_w,
        },
        horizons=[],
    )


def _build_min_variance(ctx: "AnalysisContext") -> StrategySignal:
    """Minimum-variance weight tilt, proxied by low beta + low vol.

    A full Ledoit-Wolf min-variance optimiser is portfolio-level; here we use the
    documented proxy: rank the universe by combined *low* beta and *low* vol (and
    treat that as the min-variance weight tilt versus equal-weight). Low-beta,
    low-vol names get the highest implied weight; high-beta/high-vol names near
    zero weight. Estimation error caps confidence.
    """
    meta = _META_MIN_VARIANCE
    sigma_i = _annual_vol(ctx)
    beta_i = _beta(ctx)

    # Low-vol and low-beta percentiles (1 = lowest vol / lowest beta = best here).
    vol_pct = _universe_percentile(ctx, "annual_vol")
    beta_pct = _universe_percentile(ctx, "beta")
    if vol_pct is None or beta_pct is None:
        # Self-contained fallback: map absolute vol/beta onto a 0..1 "lowness".
        low_vol = clamp(1.0 - sigma_i / 0.60, 0.0, 1.0)        # 0% vol ->1, 60% ->0
        low_beta = clamp(1.0 - abs(beta_i) / 2.0, 0.0, 1.0)    # |b|=0 ->1, 2 ->0
    else:
        low_vol = 1.0 - vol_pct
        low_beta = 1.0 - beta_pct

    # Combined "min-variance suitability" in [0,1]; 0.5 = equal-weight neutral.
    suitability = 0.6 * low_vol + 0.4 * low_beta
    score = clamp((suitability - 0.5) * 200.0, -100.0, 100.0)
    history_days = _history_days(ctx)
    confidence = clamp(0.45 + 0.3 * min(1.0, history_days / 252.0), 0.0, 0.8)

    rationale = (
        f"Low-vol score {low_vol:.2f} (vol {sigma_i * 100:.1f}%) and low-beta score "
        f"{low_beta:.2f} (beta {beta_i:.2f}) give a min-variance suitability of "
        f"{suitability:.2f}: {'above equal-weight (accumulate)' if score > 0 else 'below equal-weight (avoid)'} "
        f"({score:+.0f})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "annualVol": sigma_i,
            "beta": beta_i,
            "lowVolScore": low_vol,
            "lowBetaScore": low_beta,
            "suitability": suitability,
        },
        horizons=[],
    )


def _build_permanent_portfolio(ctx: "AnalysisContext") -> StrategySignal:
    """Permanent-Portfolio sleeve overlay (25/25/25/25 over/underweight).

    Maps the asset to one of Browne's four sleeves (stocks / long bonds / cash /
    gold) and scores its over/underweight versus the 25% sleeve target, using the
    equal-cap current weight ``1/N`` as the proxy current holding (so heavy
    sleeves read accumulate, thin sleeves trim). Crypto/unmapped names are
    neutral.
    """
    meta = _META_PERMANENT
    sym = _symbol(ctx)
    cls = _asset_class(ctx)

    n = 24
    uni = _universe(ctx)
    if uni is not None:
        try:
            n = max(1, len(uni.symbols))
        except Exception:  # pragma: no cover - defensive
            n = 24
    current_w = 1.0 / float(n)
    target = 0.25

    if sym in _GOLD_SYMBOLS:
        sleeve, conf = "gold", 0.7
    elif sym in _GROWTH_SYMBOLS or cls == "equity" or cls == "etf":
        sleeve, conf = "stocks", 0.7 if (sym in _GROWTH_SYMBOLS or cls == "equity") else 0.5
    elif cls == "crypto":
        # Crypto has no Permanent-Portfolio home -> neutral, low confidence.
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.2,
            f"{sym} (crypto) has no Permanent-Portfolio sleeve - neutral overlay.",
            meta.formula,
            metrics={"currentWeight": current_w, "targetWeight": 0.0},
            horizons=[],
        )
    else:
        sleeve, conf = "cash", 0.4

    score = clamp(100.0 * (target - current_w) / target, -100.0, 100.0)
    sleeve_drift = {"stocks": "growth", "gold": "gold", "cash": "cash"}.get(sleeve, "cash")
    horizons = (
        returns.project_horizons(
            _SLEEVE_DRIFT["growth"] / returns.TRADING_DAYS,
            max(_annual_vol(ctx) / math.sqrt(returns.TRADING_DAYS), 1e-4),
        )
        if sleeve == "stocks"
        else []
    )
    rationale = (
        f"{sym} maps to the Permanent-Portfolio {sleeve} sleeve (target 25% vs "
        f"equal-cap {current_w * 100:.1f}%): "
        f"{'under-target - accumulate' if score > 0 else 'over-target - trim'} "
        f"({score:+.0f})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, conf, rationale, meta.formula,
        metrics={
            "currentWeight": current_w,
            "targetWeight": target,
            "sleeveDrift": _SLEEVE_DRIFT.get(sleeve_drift, 0.0),
        },
        horizons=horizons,
    )


# ---------------------------------------------------------------------------
# Anomaly / factor builders
# ---------------------------------------------------------------------------

def _build_low_vol_anomaly(ctx: "AnalysisContext") -> StrategySignal:
    """Low-volatility anomaly: lowest-vol names are most bullish.

    Ranks the universe by trailing annualised volatility. ``low_vol_pct`` is the
    percentile of *lowness* (1 = lowest vol); the score maps it onto ``[-100,
    100]`` so the lowest-vol asset is +100 and the highest-vol asset -100, lightly
    blended with the low-beta percentile when available.
    """
    meta = _META_LOW_VOL
    sigma_i = _annual_vol(ctx)
    vol_pct = _universe_percentile(ctx, "annual_vol")
    beta_pct = _universe_percentile(ctx, "beta")

    if vol_pct is None:
        # Self-contained fallback when no universe: map absolute vol to lowness.
        low_vol_pct = clamp(1.0 - sigma_i / 0.60, 0.0, 1.0)
        blended = low_vol_pct
        used_universe = False
    else:
        low_vol_pct = 1.0 - vol_pct
        if beta_pct is not None:
            low_beta_pct = 1.0 - beta_pct
            blended = 0.7 * low_vol_pct + 0.3 * low_beta_pct
        else:
            blended = low_vol_pct
        used_universe = True

    score = clamp((blended - 0.5) * 200.0, -100.0, 100.0)
    history_days = _history_days(ctx)
    confidence = clamp(0.5 + 0.3 * min(1.0, history_days / 252.0), 0.0, 0.8)

    rationale = (
        f"Annualised vol {sigma_i * 100:.1f}% sits in the "
        f"{(1.0 - low_vol_pct) * 100:.0f}th vol percentile "
        f"({'lowest-vol = bullish' if score > 0 else 'highest-vol = bearish'}); the "
        f"low-volatility anomaly favours low-vol names ({score:+.0f})."
        + ("" if used_universe else " [no universe attached - absolute-vol fallback]")
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "annualVol": sigma_i,
            "lowVolPercentile": low_vol_pct,
            "blendedRank": blended,
        },
        horizons=[],
    )


def _build_betting_against_beta(ctx: "AnalysisContext") -> StrategySignal:
    """Betting-Against-Beta: low beta bullish, high beta bearish.

    Cross-sectionally ranks the universe by beta; ``low_beta_pct = 1 -
    percentile(beta)``. The score maps it so the lowest-beta asset is +100 and the
    highest-beta asset -100 (``score = (low_beta_pct - 0.5) * 200``). Confidence
    rises with history length (a better-estimated beta).
    """
    meta = _META_BAB
    beta_i = _beta(ctx)
    beta_pct = _universe_percentile(ctx, "beta")

    if beta_pct is None:
        # Self-contained fallback: center on a market beta of 1.0.
        low_beta_pct = clamp(1.0 - beta_i / 2.0, 0.0, 1.0)  # beta 0 ->1, 2 ->0
        used_universe = False
    else:
        low_beta_pct = 1.0 - beta_pct
        used_universe = True

    score = clamp((low_beta_pct - 0.5) * 200.0, -100.0, 100.0)
    history_days = _history_days(ctx)
    confidence = clamp(0.5 + 0.3 * min(1.0, history_days / 252.0), 0.0, 0.8)

    rationale = (
        f"Beta {beta_i:.2f} ranks in the {(1.0 - low_beta_pct) * 100:.0f}th beta "
        f"percentile: BAB is "
        f"{'long low-beta here (bullish)' if score > 0 else 'short high-beta here (bearish)'} "
        f"({score:+.0f})."
        + ("" if used_universe else " [no universe attached - beta-vs-1.0 fallback]")
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "beta": beta_i,
            "lowBetaPercentile": low_beta_pct,
        },
        horizons=[],
    )


def _build_seasonality(ctx: "AnalysisContext") -> StrategySignal:
    """Calendar seasonality: Halloween (Sell-in-May) + turn-of-month tilt.

    From the current ``(month, day)`` (injected as ``ctx.now`` by the engine, else
    the system clock): the Halloween component is +40 in Nov-Apr and -40 in
    May-Oct; the turn-of-month component is +40 within the last trading day or
    first 3 days of the month, else 0. The score is their clamped sum. Only
    applies to equities/ETFs; crypto is neutral (the effect is an equity-market
    anomaly).
    """
    meta = _META_SEASONALITY
    cls = _asset_class(ctx)
    if cls == "crypto":
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.2,
            "Calendar seasonality is an equity-market effect - neutral for crypto.",
            meta.formula, metrics={}, horizons=[],
        )

    month, day = _now_month_day(ctx)
    # Halloween / Sell-in-May: favourable Nov(11)-Dec(12)-Jan..Apr(1-4).
    in_halloween = month in (11, 12, 1, 2, 3, 4)
    halloween_score = 40.0 if in_halloween else -40.0
    # Turn-of-month: first 3 days of the month or the last (~28th onward).
    in_tom = day <= 3 or day >= 28
    tom_score = 40.0 if in_tom else 0.0

    score = clamp(halloween_score + tom_score, -100.0, 100.0)
    # Higher confidence when both windows align; calendar effects are modest.
    aligned = in_halloween and in_tom
    confidence = 0.5 if aligned else 0.4 if in_halloween or in_tom else 0.35

    month_name = _dt.date(2000, month, 1).strftime("%B")
    horizons: list = []
    if in_halloween:
        # Small additive seasonal drift in the favourable window (low magnitude).
        sigma_daily = max(_annual_vol(ctx) / math.sqrt(returns.TRADING_DAYS), 1e-4)
        seasonal_annual = 0.03  # ~3%/yr favourable-window tilt
        horizons = returns.project_horizons(
            math.log1p(seasonal_annual) / returns.TRADING_DAYS, sigma_daily
        )

    rationale = (
        f"Current month {month_name} ({'Nov-Apr favourable' if in_halloween else 'May-Oct weak'}) "
        f"and day {day} ({'in turn-of-month window' if in_tom else 'mid-month'}): "
        f"calendar tilt {score:+.0f}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "month": float(month),
            "day": float(day),
            "halloweenScore": halloween_score,
            "tomScore": tom_score,
        },
        horizons=horizons,
    )


# ---------------------------------------------------------------------------
# Dividend / income builders
# ---------------------------------------------------------------------------

def _build_chowder(ctx: "AnalysisContext") -> StrategySignal:
    """Chowder Rule: yield + dividend growth vs a sector threshold.

    ``chowder = yield% + dgr%`` against threshold ``T`` (15 if yield<3%, else 12;
    8 for utility-like high-yield slow-growth). The margin over ``T`` is squashed
    (scale 6) into the score; an EPS payout above 0.9 halves the score (a likely
    cut). Non-payers are neutral.
    """
    meta = _META_CHOWDER
    f = ctx.fundamentals
    price = _price(ctx)
    dividend = float(f.dividend)
    if dividend <= 0.0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.15,
            "No dividend - the Chowder Rule does not apply.",
            meta.formula, metrics={"dividend": dividend}, horizons=[],
        )

    yield_pct = 100.0 * dividend / price
    dgr_pct = 100.0 * float(f.dividend_growth)
    chowder = yield_pct + dgr_pct

    # Threshold selection.
    utility_like = yield_pct >= 4.0 and dgr_pct <= 6.0
    if utility_like:
        threshold = 8.0
    elif yield_pct < 3.0:
        threshold = 15.0
    else:
        threshold = 12.0

    margin = chowder - threshold
    score = squash(margin, scale=6.0)

    eps = float(f.eps)
    payout = dividend / eps if eps > 0.0 else float("inf")
    if math.isfinite(payout) and payout > 0.9:
        score *= 0.5

    confidence = clamp(0.4 + 0.4 * min(1.0, abs(margin) / 6.0), 0.0, 1.0)
    if math.isfinite(payout) and 0.3 <= payout <= 0.6:
        confidence = min(confidence, 0.7) if confidence > 0.7 else max(confidence, 0.5)
        confidence = clamp(confidence, 0.0, 0.7)

    # Chowder number itself is an expected-total-return estimate (capped 20%).
    horizons = returns.project_horizons(
        math.log1p(clamp(chowder / 100.0, -0.5, 0.20)) / returns.TRADING_DAYS,
        max(_annual_vol(ctx) / math.sqrt(returns.TRADING_DAYS), 1e-4),
    )

    rationale = (
        f"Chowder number {chowder:.1f} (yield {yield_pct:.1f}% + dividend growth "
        f"{dgr_pct:.1f}%) vs threshold {threshold:.0f} -> margin {margin:+.1f} "
        f"({'passes' if margin >= 0 else 'fails'})"
        + (f"; EPS payout {payout * 100:.0f}% high -> score halved." if (math.isfinite(payout) and payout > 0.9) else ".")
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "yieldPct": yield_pct,
            "dgrPct": dgr_pct,
            "chowder": chowder,
            "threshold": threshold,
            "margin": margin,
            "payout": payout if math.isfinite(payout) else 0.0,
        },
        horizons=horizons,
    )


def _build_dividend_safety(ctx: "AnalysisContext") -> StrategySignal:
    """Dividend-safety signal: yield attractiveness x coverage sustainability.

    Builds a 0..1 safety score by penalising a stretched EPS payout (>0.6), weak
    FCF coverage (FCF payout >1), high leverage (d/e >1.5) and thin liquidity
    (current ratio <1). Combines it with a yield-attractiveness term: ``raw =
    (2*safety - 1)*0.6 + yieldAttr*0.4``. Non-payers neutral; missing EPS/FCF caps
    magnitude.
    """
    meta = _META_DIV_SAFETY
    f = ctx.fundamentals
    price = _price(ctx)
    dividend = float(f.dividend)
    if dividend <= 0.0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No dividend - dividend-safety screen does not apply.",
            meta.formula, metrics={"dividend": dividend}, horizons=[],
        )

    div_yield = dividend / price
    eps = float(f.eps)
    fcf_ps = float(f.fcf_per_share)
    eps_payout = dividend / eps if eps > 0.0 else float("inf")
    fcf_payout = dividend / fcf_ps if fcf_ps > 0.0 else float("inf")

    safety = 1.0
    if math.isfinite(eps_payout) and eps_payout > 0.6:
        safety -= min(0.5, (eps_payout - 0.6) / 0.4)
    if math.isfinite(fcf_payout) and fcf_payout > 1.0:
        safety -= min(0.4, fcf_payout - 1.0)
    if float(f.debt_to_equity) > 1.5:
        safety -= 0.15
    if float(f.current_ratio) < 1.0:
        safety -= 0.1
    safety = clamp(safety, 0.0, 1.0)

    yield_attr = clamp((div_yield - 0.02) / 0.04, -1.0, 1.0)
    raw = (2.0 * safety - 1.0) * 0.6 + yield_attr * 0.4
    score = clamp(100.0 * raw, -100.0, 100.0)

    # If we cannot verify coverage (no EPS / FCF), cap the magnitude.
    if eps <= 0.0 or fcf_ps <= 0.0:
        score = clamp(score, -30.0, 30.0)

    confidence = clamp(0.4 + 0.4 * safety, 0.0, 0.85)

    rationale = (
        f"Yield {div_yield * 100:.1f}% with safety score {safety:.2f} "
        f"(EPS payout {eps_payout * 100:.0f}%" + (
            f", FCF payout {fcf_payout * 100:.0f}%" if math.isfinite(fcf_payout) else ", FCF n/a"
        ) + f"); {'well-covered, attractive' if score > 0 else 'stretched/at-risk'} ({score:+.0f})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "dividendYield": div_yield,
            "epsPayout": eps_payout if math.isfinite(eps_payout) else 0.0,
            "fcfPayout": fcf_payout if math.isfinite(fcf_payout) else 0.0,
            "safetyScore": safety,
            "yieldAttractiveness": yield_attr,
        },
        horizons=[],
    )


def _build_dividend_growth(ctx: "AnalysisContext") -> StrategySignal:
    """Dividend-growth / Aristocrats signal: yield-on-cost compounding.

    ``yoc10 = yield0*(1+g)^10`` (10-year yield-on-cost). ``growthScore =
    squash(g-0.04, 0.05)`` and ``yocBoost = squash(yoc10 - 2*yield0, 0.03)``; a
    payout above 0.65 scales the growth term down (less headroom). Non-payers /
    non-growers are at most slightly negative. Projects a Gordon-style ``yield+g``
    drift (capped 15%/yr).
    """
    meta = _META_DIV_GROWTH
    f = ctx.fundamentals
    price = _price(ctx)
    dividend = float(f.dividend)
    g = float(f.dividend_growth)

    if dividend <= 0.0 or g <= 0.0:
        return make_signal(
            meta.id, meta.name, meta.category,
            score=-10.0 if dividend <= 0.0 else 0.0,
            confidence=0.2,
            rationale=(
                "No dividend - excluded from the dividend-growth discipline."
                if dividend <= 0.0
                else "Dividend not growing - no compounding-income tilt."
            ),
            formula=meta.formula,
            metrics={"dividend": dividend, "dividendGrowth": g},
            horizons=[],
        )

    yield0 = dividend / price
    yoc10 = yield0 * (1.0 + g) ** 10
    growth_score = squash(g - 0.04, scale=0.05)
    yoc_boost = squash(yoc10 - 2.0 * yield0, scale=0.03)

    eps = float(f.eps)
    payout = dividend / eps if eps > 0.0 else float("inf")
    if math.isfinite(payout) and payout > 0.65:
        growth_score *= max(0.3, 1.0 - (payout - 0.65) / 0.35)

    score = clamp(0.6 * growth_score + 0.4 * yoc_boost, -100.0, 100.0)
    confidence = clamp(0.4 + 0.4 * min(1.0, abs(g) / 0.10), 0.0, 0.7)

    # Gordon-style total-return drift = yield + growth, capped at 15%/yr.
    drift_annual = clamp(yield0 + g, -0.10, 0.15)
    horizons = returns.project_horizons(
        math.log1p(drift_annual) / returns.TRADING_DAYS,
        max(_annual_vol(ctx) / math.sqrt(returns.TRADING_DAYS), 1e-4),
    )

    rationale = (
        f"Dividend growth {g * 100:.1f}% on a {yield0 * 100:.1f}% yield -> 10-year "
        f"yield-on-cost {yoc10 * 100:.1f}% "
        f"({'durable grower - compounding income' if score > 0 else 'stalled/stretched growth'}; "
        f"{score:+.0f})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "yield0": yield0,
            "dividendGrowth": g,
            "yieldOnCost10y": yoc10,
            "growthScore": growth_score,
            "yocBoost": yoc_boost,
            "payout": payout if math.isfinite(payout) else 0.0,
        },
        horizons=horizons,
    )


def _build_shareholder_yield(ctx: "AnalysisContext") -> StrategySignal:
    """Shareholder-yield signal (Meb Faber): total cash returned to holders.

    ``proxy = divYield + max(0,(fcf_per_share - dividend))/price + clamp(1 -
    debt/equity, -0.5, 0.5)*0.03`` (dividend + buyback proxy from covered FCF
    surplus + a small debt-paydown tilt). The score is ``squash(proxy - 0.04,
    0.05)`` (~9% proxy -> +76). Confidence is capped at 0.55 because the
    buyback/debt legs are proxied. Crypto/ETF with no FCF are neutral.
    """
    meta = _META_SHY
    f = ctx.fundamentals
    price = _price(ctx)
    dividend = float(f.dividend)
    fcf_ps = float(f.fcf_per_share)

    # No FCF data (crypto / ETF) -> neutral.
    if fcf_ps <= 0.0 and dividend <= 0.0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No free cash flow or dividend - shareholder-yield proxy unavailable.",
            meta.formula, metrics={}, horizons=[],
        )

    div_yield = dividend / price
    buyback_proxy = max(0.0, fcf_ps - dividend) / price
    debt_tilt = clamp(1.0 - float(f.debt_to_equity), -0.5, 0.5) * 0.03
    proxy = div_yield + buyback_proxy + debt_tilt

    score = squash(proxy - 0.04, scale=0.05)
    confidence = clamp(0.35 + 0.4 * min(1.0, proxy / 0.10), 0.0, 0.55)

    # Small additive 1Y+ drift premium (proxy capped at 12%/yr).
    drift_annual = clamp(proxy, -0.05, 0.12)
    horizons = returns.project_horizons(
        math.log1p(drift_annual) / returns.TRADING_DAYS,
        max(_annual_vol(ctx) / math.sqrt(returns.TRADING_DAYS), 1e-4),
    )

    rationale = (
        f"Shareholder-yield proxy {proxy * 100:.1f}% (dividend {div_yield * 100:.1f}% + "
        f"buyback proxy {buyback_proxy * 100:.1f}% + debt tilt {debt_tilt * 100:+.1f}%): "
        f"{'high total cash returned (bullish)' if score > 0 else 'low (bearish)'} ({score:+.0f})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "dividendYield": div_yield,
            "buybackProxy": buyback_proxy,
            "debtTilt": debt_tilt,
            "shareholderYieldProxy": proxy,
        },
        horizons=horizons,
    )


def _build_dogs_of_dow(ctx: "AnalysisContext") -> StrategySignal:
    """Dogs of the Dow: high cross-sectional yield is bullish (mean-reversion).

    Computes the asset's dividend-yield percentile among dividend payers in the
    universe and squashes ``percentile - 0.5`` (scale 0.35) into the score (top
    decile ~ +76). A trailing EPS payout above 1.0 dampens the score x0.6 (the
    yield is high because earnings collapsed - a likely cut). Non-payers neutral.
    """
    meta = _META_DOGS
    f = ctx.fundamentals
    price = _price(ctx)
    dividend = float(f.dividend)
    if dividend <= 0.0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No dividend - excluded from the Dogs of the Dow yield ranking.",
            meta.formula, metrics={"dividend": dividend}, horizons=[],
        )

    div_yield = dividend / price
    percentile = _yield_percentile_among_payers(ctx, div_yield)

    score = squash(percentile - 0.5, scale=0.35)

    eps = float(f.eps)
    payout = dividend / eps if eps > 0.0 else float("inf")
    if math.isfinite(payout) and payout > 1.0:
        score *= 0.6

    confidence = clamp(0.35 + 0.4 * abs(percentile - 0.5) * 2.0, 0.0, 1.0)

    # Mild +2%/yr value/mean-reversion drift for top-decile Dogs.
    drift_annual = 0.02 if percentile >= 0.8 else 0.0
    horizons = returns.project_horizons(
        math.log1p(drift_annual) / returns.TRADING_DAYS,
        max(_annual_vol(ctx) / math.sqrt(returns.TRADING_DAYS), 1e-4),
    ) if drift_annual != 0.0 else []

    rationale = (
        f"Dividend yield {div_yield * 100:.1f}% ranks in the {percentile * 100:.0f}th "
        f"percentile among payers: {'a high-yield Dog (depressed price, mean-reversion bullish)' if score > 0 else 'a low-yield name (bearish)'} "
        f"({score:+.0f})"
        + (f"; EPS payout {payout * 100:.0f}% > 100% -> score dampened." if (math.isfinite(payout) and payout > 1.0) else ".")
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "dividendYield": div_yield,
            "yieldPercentile": percentile,
            "payout": payout if math.isfinite(payout) else 0.0,
        },
        horizons=horizons,
    )


def _build_small_dogs_of_dow(ctx: "AnalysisContext") -> StrategySignal:
    """Small Dogs of the Dow: high yield AND low nominal price is strongest.

    Stage 1 is the dividend-yield percentile (as Dogs). Stage 2 adds the
    inverse-price rank (lower nominal price = stronger). For high-yield names
    (top tertile) the combined score is ``0.5*(yield_pct-0.5) +
    0.5*((1-price_pct)-0.5)``; otherwise the yield term alone (x0.4). Confidence
    is capped at 0.5 (low diversification). Non-payers neutral.
    """
    meta = _META_SMALL_DOGS
    f = ctx.fundamentals
    price = _price(ctx)
    dividend = float(f.dividend)
    if dividend <= 0.0:
        return make_signal(
            meta.id, meta.name, meta.category, 0.0, 0.1,
            "No dividend - excluded from the Small Dogs ranking.",
            meta.formula, metrics={"dividend": dividend}, horizons=[],
        )

    div_yield = dividend / price
    yield_pct = _yield_percentile_among_payers(ctx, div_yield)
    price_pct = _price_percentile_among_payers(ctx, price)

    is_dog = yield_pct >= (2.0 / 3.0)
    if is_dog:
        combined = 0.5 * (yield_pct - 0.5) + 0.5 * ((1.0 - price_pct) - 0.5)
    else:
        combined = 0.4 * (yield_pct - 0.5)

    score = squash(combined, scale=0.3)
    confidence = clamp(0.3 + 0.4 * min(1.0, abs(combined) / 0.5), 0.0, 0.5)

    # Same value/mean-reversion tilt as Dogs but stronger; +2%/yr for selected.
    selected = is_dog and price_pct <= 0.5
    drift_annual = 0.02 if selected else 0.0
    horizons = returns.project_horizons(
        math.log1p(drift_annual) / returns.TRADING_DAYS,
        max(_annual_vol(ctx) / math.sqrt(returns.TRADING_DAYS) * 1.1, 1e-4),  # widen vol x1.1
    ) if drift_annual != 0.0 else []

    rationale = (
        f"Yield {div_yield * 100:.1f}% ({yield_pct * 100:.0f}th pct) and nominal price "
        f"{price:.2f} ({price_pct * 100:.0f}th price pct): "
        f"{'a high-yield, low-priced Small Dog (strong rebound tilt)' if (selected and score > 0) else ('a high-yield Dog' if is_dog else 'not a Dog')} "
        f"({score:+.0f})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "dividendYield": div_yield,
            "yieldPercentile": yield_pct,
            "pricePercentile": price_pct,
            "combined": combined,
            "nominalPrice": price,
        },
        horizons=horizons,
    )


# ---------------------------------------------------------------------------
# Cross-sectional percentile helpers (dividend strategies)
# ---------------------------------------------------------------------------

def _yield_percentile_among_payers(
    ctx: "AnalysisContext", own_yield: float
) -> float:
    """Return the asset's dividend-yield percentile among universe payers.

    Uses the engine's ``UniverseStats.percentile('dividend_yield', symbol)`` when
    available (0..1, 1 = highest yield). With no universe attached, falls back to
    a calibrated absolute mapping of the asset's own yield onto a percentile (a
    ~6% yield maps near the top decile), so the strategy still differentiates.

    Args:
        ctx: The analysis context.
        own_yield: The asset's own dividend yield (decimal), used by the fallback.

    Returns:
        A yield percentile in ``[0, 1]`` (1 = highest yield).
    """
    pct = _universe_percentile(ctx, "dividend_yield")
    if pct is not None:
        return pct
    # Absolute fallback: 0% -> 0, ~6% -> ~0.9 (top decile), saturating.
    return clamp(own_yield / 0.06 * 0.9, 0.0, 1.0)


def _price_percentile_among_payers(
    ctx: "AnalysisContext", own_price: float
) -> float:
    """Return the asset's nominal-price percentile among universe payers.

    Uses ``UniverseStats`` price data when present; otherwise maps the nominal
    price onto a percentile via a log scale (a few dollars -> low, several hundred
    -> high) so the Small-Dogs low-price tilt remains meaningful without a
    universe.

    Args:
        ctx: The analysis context.
        own_price: The asset's nominal price.

    Returns:
        A price percentile in ``[0, 1]`` (1 = highest nominal price).
    """
    uni = _universe(ctx)
    if uni is not None:
        # UniverseStats does not expose a price metric directly; rebuild the
        # cross-sectional rank from the live asset snapshots if reachable.
        prices: list[float] = []
        try:
            symbols = list(uni.symbols)
            for sym in symbols:
                p = _universe_price(uni, sym)
                if p is not None and math.isfinite(p) and p > 0.0:
                    prices.append(p)
        except Exception:  # pragma: no cover - defensive
            prices = []
        if len(prices) >= 2:
            arr = np.asarray(prices, dtype=np.float64)
            below = float(np.sum(arr < own_price))
            equal = float(np.sum(arr == own_price))
            pct = (below + 0.5 * equal) / float(arr.size)
            return clamp(pct, 0.0, 1.0)
    # Log-scale absolute fallback: $1 -> ~0, $1000 -> ~1.
    p = max(own_price, 1e-6)
    return clamp(math.log10(p) / 3.0, 0.0, 1.0)


def _universe_price(uni: "UniverseStats", sym: str) -> float | None:
    """Best-effort nominal price lookup for ``sym`` from ``UniverseStats``.

    The frozen ``UniverseStats`` API (§1) does not include a price dict, so this
    reads an optional ``price`` mapping if a future engine adds one; otherwise it
    returns ``None`` and the caller uses its absolute-price fallback.

    Args:
        uni: The universe-stats object.
        sym: Upper-cased symbol.

    Returns:
        The nominal price, or ``None`` when unavailable.
    """
    price_map = getattr(uni, "price", None)
    if isinstance(price_map, dict):
        val = price_map.get(sym)
        if val is not None:
            try:
                fv = float(val)
                return fv if math.isfinite(fv) else None
            except (TypeError, ValueError):  # pragma: no cover - defensive
                return None
    return None


# ---------------------------------------------------------------------------
# Registry exports
# ---------------------------------------------------------------------------

#: ``id -> (StrategyMeta, builder_fn)`` for every strategy in this group.
BUILDERS: dict[str, tuple[StrategyMeta, Callable[["AnalysisContext"], StrategySignal]]] = {
    "all-weather-risk-parity": (_META_ALL_WEATHER, _build_all_weather),
    "vol-target": (_META_VOL_TARGET, _build_vol_target),
    "risk-parity-inverse-vol": (_META_RISK_PARITY, _build_risk_parity),
    "min-variance": (_META_MIN_VARIANCE, _build_min_variance),
    "permanent-portfolio": (_META_PERMANENT, _build_permanent_portfolio),
    "low-vol-anomaly": (_META_LOW_VOL, _build_low_vol_anomaly),
    "betting-against-beta": (_META_BAB, _build_betting_against_beta),
    "seasonality": (_META_SEASONALITY, _build_seasonality),
    "chowder-rule": (_META_CHOWDER, _build_chowder),
    "dividend-safety": (_META_DIV_SAFETY, _build_dividend_safety),
    "dividend-growth-aristocrats": (_META_DIV_GROWTH, _build_dividend_growth),
    "shareholder-yield": (_META_SHY, _build_shareholder_yield),
    "dogs-of-dow": (_META_DOGS, _build_dogs_of_dow),
    "small-dogs-of-dow": (_META_SMALL_DOGS, _build_small_dogs_of_dow),
}


#: Vectorized per-bar position series for backtestable *timing* strategies in
#: this group. The 14 strategies here are cross-sectional / fundamental /
#: calendar overlays (not per-bar timing rules), so there are none.
POSITION_FUNCS: dict[str, Callable] = {}
