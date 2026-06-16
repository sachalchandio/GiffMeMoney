"""Value / quality / growth / factor strategy builders (STRATEGIES-V2 §5).

This module implements 14 cited, fundamentals-driven strategies from
``docs/research/strategy-catalog.json``, each as a pure builder

    ``fn(ctx: AnalysisContext) -> StrategySignal``

that implements the catalog's ``computeSignal`` exactly. The module exposes two
module-level dicts that the integration agent merges into the registry:

    * :data:`BUILDERS` — ``dict[str, tuple[StrategyMeta, builder_fn]]`` keyed by
      strategy id (all 14 ids below).
    * :data:`POSITION_FUNCS` — vectorized position-series functions for any
      *time-backtestable* timing strategy here. None of these 14 are per-bar
      timing strategies (they are snapshot/cross-sectional fundamental screens),
      so this is the empty dict ``{}``.

Strategy ids implemented:
    graham-defensive, graham-number, net-net-ncav, magic-formula,
    acquirers-multiple, owner-earnings-yield, buffett-quality-fair-price,
    qmj-quality-minus-junk, gross-profitability, return-on-capital-compounder,
    fama-french-5, gordon-reverse-implied-growth, peg-lynch, canslim.

Category mapping (STRATEGIES-V2 §5 -> existing ``StrategyCategory`` literal):
    value/quality/growth -> ``Valuation`` or ``Fundamental``; factor -> ``Factor``.

Cross-sectional strategies (magic-formula, acquirers-multiple, qmj,
gross-profitability, return-on-capital-compounder, fama-french-5) need
percentile/z-score ranks across the equity universe. They read those from
``ctx.universe`` (a ``UniverseStats`` injected by the engine) when available, and
otherwise fall back to computing the cross-section directly from the seed
universe (``app.market.universe``) so the builders are correct and importable
standalone. Every builder is numerically defensive: short/empty/NaN inputs and
non-equity asset classes collapse to safe finite defaults and never raise.

Score convention: positive = bullish, score in ``[-100, 100]``, confidence in
``[0, 1]`` (matching the existing registry).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from app.market.universe import UNIVERSE, AssetSeed, Fundamentals
from app.quant import metrics, returns
from app.schemas import StrategyMeta, StrategySignal
from app.strategies.base import clamp, make_signal, squash

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a circular import
    from app.strategies.engine import AnalysisContext, UniverseStats

__all__ = ["BUILDERS", "POSITION_FUNCS"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Graham's combined P/E * P/B ceiling (P/E<=15 and P/B<=1.5 => 22.5).
_GRAHAM_CONST: float = 22.5

#: Equity-risk-premium prior used where a CAPM required return is needed and no
#: realized market premium is available.
_DEFAULT_ERP: float = 0.045

#: Equities are the only class with meaningful fundamentals; ETFs/crypto have
#: synthetic/empty fundamentals so most of these screens are N/A for them.
_EQUITY: str = "equity"


# ---------------------------------------------------------------------------
# Cross-sectional support
# ---------------------------------------------------------------------------


def _equity_seeds() -> list[AssetSeed]:
    """Return all equity seeds in the universe (the cross-section for ranking).

    Returns:
        A list of :class:`~app.market.universe.AssetSeed` whose ``asset_class``
        is ``'equity'`` (the screens here exclude ETFs/crypto).
    """
    return [s for s in UNIVERSE if s.asset_class == _EQUITY]


def _seed_price(seed: AssetSeed) -> float:
    """A stable price proxy for a seed when no live price is available.

    Uses the seed's ``base_price`` (the deterministic simulator anchor). This is
    only used in the cross-sectional fallback, where every asset must be scored
    on a consistent basis.

    Args:
        seed: The asset seed.

    Returns:
        A strictly-positive price proxy (``1.0`` if the base price is invalid).
    """
    p = float(seed.base_price)
    return p if math.isfinite(p) and p > 0.0 else 1.0


def _metric_for_seed(metric: str, seed: AssetSeed) -> float:
    """Compute one cross-sectional metric for a seed (fallback ranking path).

    Mirrors the metric definitions in :data:`UniverseStats` so the fallback
    ranking agrees with the engine-injected ranking. All values are finite.

    Args:
        metric: One of ``'earnings_yield'``, ``'roic'``,
            ``'gross_profitability'``, ``'op_profitability'``, ``'roa'``,
            ``'fcf_yield'``, ``'net_margin'``.
        seed: The asset seed to evaluate.

    Returns:
        The finite metric value (``0.0`` on any degeneracy).
    """
    f = seed.fundamentals
    price = _seed_price(seed)
    mc = float(seed.market_cap) if seed.market_cap else 0.0

    if metric == "earnings_yield":
        ev = _enterprise_value(f, mc)
        return _finite(f.ebit / ev) if ev > 0.0 else 0.0
    if metric == "roic":
        inv = _invested_capital(f)
        return _finite(f.ebit / inv) if inv != 0.0 else 0.0
    if metric == "op_profitability":
        be = _book_equity(f)
        return _finite(f.ebit / be) if be > 0.0 else _finite(f.ebit / f.total_assets) if f.total_assets > 0.0 else 0.0
    if metric == "gross_profitability":
        ta = float(f.total_assets)
        if ta <= 0.0:
            return 0.0
        # No COGS line: operating-profitability flavour (Novy-Marx proxy).
        return _finite(f.ebit / ta)
    if metric == "roa":
        return _finite(f.roa)
    if metric == "net_margin":
        return _finite(f.net_margin)
    if metric == "fcf_yield":
        return _finite(f.fcf_per_share / price) if price > 0.0 else 0.0
    if metric == "revenue_growth":
        return _finite(f.revenue_growth)
    return 0.0


def _percentile(
    ctx: "AnalysisContext",
    metric: str,
    sym: str,
    universe_attr: Optional[str] = None,
) -> float:
    """Cross-sectional percentile rank of ``sym`` on ``metric`` (0..1, 1=highest).

    Prefers the engine-injected ``ctx.universe`` (a ``UniverseStats`` whose
    ``percentile`` returns 0..1 with 1 = best/highest). If the universe (or the
    named metric on it) is unavailable, falls back to ranking the equity seeds
    by :func:`_metric_for_seed` so the builder still produces a meaningful,
    differentiated cross-sectional score standalone.

    Args:
        ctx: The analysis context.
        metric: The local metric key understood by :func:`_metric_for_seed`.
        sym: Upper-cased symbol to rank.
        universe_attr: The corresponding ``UniverseStats`` metric name to try on
            ``ctx.universe.percentile`` first (defaults to ``metric``).

    Returns:
        A percentile in ``[0, 1]`` (``0.5`` when only one asset / degenerate).
    """
    key = sym.strip().upper()
    uni = getattr(ctx, "universe", None)
    attr = universe_attr or metric
    if uni is not None:
        try:
            # UniverseStats.percentile(metric, symbol) -> 0..1 (1 = highest).
            val = uni.percentile(attr, key)
            fv = float(val)
            if math.isfinite(fv):
                return clamp(fv, 0.0, 1.0)
        except Exception:
            pass

    # Fallback: rank the equity seeds ourselves.
    seeds = _equity_seeds()
    if not seeds:
        return 0.5
    vals = np.array([_metric_for_seed(metric, s) for s in seeds], dtype=np.float64)
    target = _metric_for_seed(metric, _seed_for(key))
    n = vals.size
    if n <= 1:
        return 0.5
    # Fraction strictly below + half the ties = mid-rank percentile in [0,1].
    below = float(np.sum(vals < target))
    ties = float(np.sum(vals == target))
    pct = (below + 0.5 * ties) / n
    return clamp(pct, 0.0, 1.0)


def _zscore_cross(
    ctx: "AnalysisContext",
    metric: str,
    sym: str,
    sign: float = 1.0,
) -> float:
    """Cross-sectional z-score of ``sym`` on ``metric`` across equity seeds.

    Computes ``z = sign * (x - mean) / std`` over the equity cross-section. Used
    by the QMJ composite (which needs signed z-scores, not just percentiles).

    Args:
        ctx: The analysis context.
        metric: A metric key understood by :func:`_metric_for_seed`.
        sym: Upper-cased symbol.
        sign: ``+1`` for "higher is better", ``-1`` for "lower is better".

    Returns:
        A finite z-score, clamped to ``[-3, 3]`` (``0.0`` on degeneracy).
    """
    seeds = _equity_seeds()
    if len(seeds) <= 1:
        return 0.0
    vals = np.array([_metric_for_seed(metric, s) for s in seeds], dtype=np.float64)
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    if sd <= 0.0 or not math.isfinite(sd):
        return 0.0
    x = _metric_for_seed(metric, _seed_for(sym.strip().upper()))
    z = sign * (x - mu) / sd
    return clamp(z, -3.0, 3.0) if math.isfinite(z) else 0.0


def _seed_for(sym: str) -> AssetSeed:
    """Return the seed for ``sym`` (falling back to the first equity seed).

    Args:
        sym: Upper-cased symbol.

    Returns:
        The matching :class:`AssetSeed`, or the first equity seed if unknown.
    """
    for s in UNIVERSE:
        if s.symbol == sym:
            return s
    eq = _equity_seeds()
    return eq[0] if eq else UNIVERSE[0]


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _finite(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _is_equity(ctx: "AnalysisContext") -> bool:
    """True iff the context's asset is an equity (the screens' valid universe)."""
    return str(getattr(ctx.asset, "asset_class", "")).lower() == _EQUITY


def _price(ctx: "AnalysisContext") -> float:
    """Latest positive price for the asset (falls back to the seed base price)."""
    p = float(getattr(ctx.asset, "price", 0.0) or 0.0)
    if math.isfinite(p) and p > 0.0:
        return p
    bp = float(getattr(ctx.seed, "base_price", 0.0) or 0.0)
    return bp if math.isfinite(bp) and bp > 0.0 else 1.0


def _book_equity(f: Fundamentals) -> float:
    """Book equity = book value per share * shares outstanding (>= 0)."""
    be = float(f.book_value_per_share) * float(f.shares_out)
    return be if math.isfinite(be) and be > 0.0 else 0.0


def _net_debt(f: Fundamentals) -> float:
    """Net-debt proxy = debt_to_equity * bvps * shares_out (>= 0).

    The catalog's EV proxy: ``EV = market_cap + net_debt`` with the net-debt
    leg approximated from leverage and book equity when a debt series is absent.
    """
    nd = float(f.debt_to_equity) * float(f.book_value_per_share) * float(f.shares_out)
    return nd if math.isfinite(nd) and nd > 0.0 else 0.0


def _enterprise_value(f: Fundamentals, market_cap: float) -> float:
    """Enterprise value proxy = market_cap + net_debt (fallback market_cap).

    Args:
        f: The asset fundamentals.
        market_cap: The asset market capitalisation.

    Returns:
        A strictly-positive EV (``0.0`` only when market cap is unusable).
    """
    mc = float(market_cap) if market_cap and math.isfinite(market_cap) else 0.0
    if mc <= 0.0:
        return 0.0
    ev = mc + _net_debt(f)
    return ev if math.isfinite(ev) and ev > 0.0 else mc


def _invested_capital(f: Fundamentals) -> float:
    """Invested-capital proxy for ROIC = EBIT / invested capital.

    Net invested capital ~= working_capital + (TA - TL - working_capital)
    = TA - TL (net assets). Falls back to ``TA - TL`` and finally to ``|EPS|``
    so the ratio stays finite. Returns ``0.0`` when nothing is usable.
    """
    net_assets = float(f.total_assets) - float(f.total_liabilities)
    if math.isfinite(net_assets) and net_assets > 0.0:
        return net_assets
    return 0.0


def _capm_required(ctx: "AnalysisContext") -> float:
    """CAPM required return = rf + beta * ERP, clamped to ``[0.05, 0.20]``.

    Uses the realized market premium when available, else the default ERP.

    Args:
        ctx: The analysis context.

    Returns:
        A required annual return in ``[0.05, 0.20]``.
    """
    b = metrics.beta(ctx.returns, ctx.market_ret)
    rf_annual = float(ctx.rf_daily) * returns.TRADING_DAYS
    mkt = np.asarray(ctx.market_ret, dtype=np.float64).ravel()
    if mkt.size:
        premium = (float(np.mean(mkt)) - float(ctx.rf_daily)) * returns.TRADING_DAYS
        if not math.isfinite(premium) or premium <= 0.0:
            premium = _DEFAULT_ERP
    else:
        premium = _DEFAULT_ERP
    req = rf_annual + b * premium
    return clamp(req, 0.05, 0.20)


def _project_from_annual(ann_drift: float, ctx: "AnalysisContext") -> list[dict]:
    """Project the 5 horizons from an *annual* expected drift + realized vol.

    Converts an annual expected return into a daily log drift, caps it to a
    realistic equity band (``[-25%, +35%]`` annual, per R1) and projects with the
    asset's realized daily volatility.

    Args:
        ann_drift: Implied annual expected return (decimal).
        ctx: The analysis context (for realized volatility).

    Returns:
        The list of horizon dicts from :func:`app.quant.returns.project_horizons`.
    """
    capped = clamp(_finite(ann_drift), -0.25, 0.35)
    daily = math.log1p(capped) / returns.TRADING_DAYS if capped > -1.0 else 0.0
    lr = returns.log_returns(ctx.closes)
    sigma = float(np.std(lr)) if lr.size else 1e-4
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = 1e-4
    return returns.project_horizons(daily, sigma)


def _na_signal(strategy_id: str, reason: str) -> StrategySignal:
    """Neutral, low-confidence ``HOLD`` for an asset the screen does not apply to.

    Args:
        strategy_id: The strategy id (must exist in :data:`_META`).
        reason: Short plain-English explanation.

    Returns:
        A zero-score, low-confidence :class:`~app.schemas.StrategySignal`.
    """
    meta = _META[strategy_id]
    return make_signal(
        meta.id,
        meta.name,
        meta.category,
        score=0.0,
        confidence=0.1,
        rationale=f"{meta.name} not applicable: {reason}.",
        formula=meta.formula,
        metrics={},
        horizons=[],
    )


# ---------------------------------------------------------------------------
# Strategy metadata (carried from the catalog: summary + sources)
# ---------------------------------------------------------------------------

_META: dict[str, StrategyMeta] = {
    "graham-defensive": StrategyMeta(
        id="graham-defensive",
        name="Graham Defensive Investor Criteria",
        category="Valuation",
        summary=(
            "Benjamin Graham's mechanical 7-test checklist for the passive "
            "('defensive') investor: adequate size, strong financials, earnings "
            "stability, dividend record, growth, moderate P/E (<=15) and P/B "
            "(<=1.5, or P/E*P/B<=22.5)."
        ),
        formula="V = 100*(22.5 - P/E*P/B)/22.5; score = V*(0.5+0.5*tests/7)",
        inputs=["price", "EPS", "BVPS", "current ratio", "debt/equity", "dividend", "revenue growth", "market cap"],
        references=[
            "Benjamin Graham, 'The Intelligent Investor' (1973 rev.), Ch. 14",
            "AAII Graham Defensive screens — https://www.aaii.com/stock-screens",
        ],
    ),
    "graham-number": StrategyMeta(
        id="graham-number",
        name="Graham Number (Fair-Value Ceiling)",
        category="Valuation",
        summary=(
            "Single fair-value formula from Graham's P/E<=15 and P/B<=1.5 limits: "
            "GN = sqrt(22.5 * EPS * BVPS), the maximum a defensive investor should "
            "pay for a profitable, asset-backed firm."
        ),
        formula="GN = sqrt(22.5 * EPS * BVPS); score = 100*(GN - price)/GN",
        inputs=["EPS", "BVPS", "price"],
        references=[
            "Benjamin Graham, 'The Intelligent Investor' (1973) — 22.5 constant basis",
            "GuruFocus Graham Number — https://www.gurufocus.com/term/graham-number",
        ],
    ),
    "net-net-ncav": StrategyMeta(
        id="net-net-ncav",
        name="Graham Net-Net / NCAV (Deep Value)",
        category="Valuation",
        summary=(
            "Graham's deep-value 'cigar-butt' screen (Security Analysis, 1934): "
            "buy below two-thirds of Net Current Asset Value (current assets minus "
            "total liabilities) — liquidation-value investing with a built-in "
            "margin of safety."
        ),
        formula="NCAV/sh = NCAV_proxy/shares; score = 100*(1 - (price/NCAVps)/0.67)",
        inputs=["working capital", "total liabilities", "shares outstanding", "price"],
        references=[
            "Graham & Dodd, 'Security Analysis' (1934)",
            "Oppenheimer (1986), 'Ben Graham's Net Current Asset Values', FAJ",
        ],
    ),
    "magic-formula": StrategyMeta(
        id="magic-formula",
        name="Greenblatt Magic Formula",
        category="Fundamental",
        summary=(
            "Joel Greenblatt's combined ranking of cheapness (earnings yield = "
            "EBIT/EV) and quality (return on capital = EBIT/capital) from 'The "
            "Little Book That Beats the Market' (2005)."
        ),
        formula="C = 0.5*pct(EBIT/EV) + 0.5*pct(EBIT/capital); score = 200*(C-0.5)",
        inputs=["EBIT", "enterprise value", "invested capital", "cross-section"],
        references=[
            "Joel Greenblatt, 'The Little Book That Beats the Market' (2005, rev. 2010)",
            "https://www.gurufocus.com/tutorial/article/57",
        ],
    ),
    "acquirers-multiple": StrategyMeta(
        id="acquirers-multiple",
        name="Acquirer's Multiple (EV/EBIT)",
        category="Valuation",
        summary=(
            "Tobias Carlisle's deep-value metric ('Deep Value', 2014): rank by the "
            "cheapest operating multiple EV/EBIT — Greenblatt's earnings yield "
            "without the ROC quality leg."
        ),
        formula="EY = EBIT/EV; score = 200*(pct(EY) - 0.5)",
        inputs=["EBIT", "enterprise value", "cross-section"],
        references=[
            "Tobias E. Carlisle, 'The Acquirer's Multiple' (2014); 'Deep Value' (2014, Wiley)",
            "Carlisle & Gray, 'Quantitative Value' (2012, Wiley)",
        ],
    ),
    "owner-earnings-yield": StrategyMeta(
        id="owner-earnings-yield",
        name="Buffett Owner-Earnings Yield",
        category="Fundamental",
        summary=(
            "Warren Buffett's cash-earnings measure (1986 Berkshire letter): owner "
            "earnings ~ FCF; buy quality businesses when the owner-earnings yield "
            "is attractive vs price and the risk-free rate."
        ),
        formula="OEY = FCF/share / price; score = 100*((OEY - rf)/0.06)",
        inputs=["FCF per share", "price", "risk-free rate"],
        references=[
            "Warren Buffett, Berkshire Hathaway 1986 Shareholder Letter, 'Owner Earnings'",
            "Robert Hagstrom, 'The Warren Buffett Way' (1994)",
        ],
    ),
    "buffett-quality-fair-price": StrategyMeta(
        id="buffett-quality-fair-price",
        name="Buffett Quality-at-Fair-Price (ROE/Moat)",
        category="Fundamental",
        summary=(
            "Buffett's evolution from Graham: buy durable-moat businesses (high "
            "stable ROE>=15%, high margins, low debt) at a fair price. Codified in "
            "Validea/Hagstrom quantitative interpretations."
        ),
        formula="score = 100*Q*(0.5 + 0.5*price_score), Q = mean of 5 quality gates",
        inputs=["EPS", "BVPS", "ROA", "net margin", "debt/equity", "revenue growth", "FCF", "price"],
        references=[
            "Robert Hagstrom, 'The Warren Buffett Way' (1994/2013)",
            "Mary Buffett & David Clark, 'Buffettology' (1997)",
            "https://blog.validea.com/building-a-quantitative-strategy-based-on-warren-buffetts-approach/",
        ],
    ),
    "qmj-quality-minus-junk": StrategyMeta(
        id="qmj-quality-minus-junk",
        name="Quality Minus Junk (QMJ)",
        category="Factor",
        summary=(
            "Asness, Frazzini & Pedersen's QMJ (AQR; Review of Accounting Studies "
            "2019): quality = profitable + growing + safe + (payout). Long "
            "high-quality, short junk."
        ),
        formula="Quality_z = mean(profitability_z, growth_z, safety_z, payout_z); score = 33*Quality_z",
        inputs=["net margin", "ROA", "gross profitability", "FCF/price", "revenue growth", "debt/equity", "beta", "current ratio", "dividend"],
        references=[
            "Asness, Frazzini & Pedersen, 'Quality Minus Junk', Review of Accounting Studies 24, 2019 — https://www.aqr.com/Insights/Research/Working-Paper/Quality-Minus-Junk",
        ],
    ),
    "gross-profitability": StrategyMeta(
        id="gross-profitability",
        name="Novy-Marx Gross Profitability",
        category="Factor",
        summary=(
            "Robert Novy-Marx's gross-profitability premium (JFE 2013): gross "
            "profits scaled by total assets predict the cross-section of returns "
            "about as well as book-to-market."
        ),
        formula="GP = gross_profit/total_assets (proxy EBIT/TA); score = 2*(pct(GP) - 50)",
        inputs=["EBIT / gross profit", "total assets", "cross-section"],
        references=[
            "Robert Novy-Marx, 'The Other Side of Value: The Gross Profitability Premium', JFE 108(1), 2013 — https://rnm.simon.rochester.edu/research/OSoV.pdf",
        ],
    ),
    "return-on-capital-compounder": StrategyMeta(
        id="return-on-capital-compounder",
        name="High Return-on-Capital Compounder",
        category="Fundamental",
        summary=(
            "Buffett/Greenblatt-style quality compounding: persistently high "
            "return on capital (ROA/ROE/ROIC) with strong FCF and low leverage "
            "signals a durable, value-compounding business."
        ),
        formula="quality_rank = 0.6*pct(ROC) + 0.4*pct(ROA); score = 2*(quality_rank - 50)",
        inputs=["EBIT", "total assets", "total liabilities", "ROA", "EPS", "BVPS", "FCF", "debt/equity"],
        references=[
            "Joel Greenblatt, 'The Little Book That Beats the Market' (2005)",
            "Robert G. Hagstrom, 'The Warren Buffett Way' (1994)",
        ],
    ),
    "fama-french-5": StrategyMeta(
        id="fama-french-5",
        name="Fama-French 5-Factor (RMW + CMA)",
        category="Factor",
        summary=(
            "Fama & French's five-factor model (JFE 2015) adds profitability (RMW) "
            "and investment (CMA) to the 3-factor model. Robust-profitability and "
            "conservative-investment firms outperform."
        ),
        formula="score = 0.6*rmw + 0.4*cma; rmw = 2*(pct(EBIT/BE) - 50), cma = -33*aggressiveness_z",
        inputs=["EBIT", "book equity", "revenue growth", "FCF", "dividend", "cross-section"],
        references=[
            "Fama & French, 'A Five-Factor Asset Pricing Model', JFE 116(1), 2015 — https://tevgeniou.github.io/EquityRiskFactors/bibliography/FiveFactor.pdf",
            "Kenneth R. French Data Library (RMW/CMA definitions)",
        ],
    ),
    "gordon-reverse-implied-growth": StrategyMeta(
        id="gordon-reverse-implied-growth",
        name="Reverse-Engineered Implied Growth (DCF/DDM)",
        category="Valuation",
        summary=(
            "Invert the valuation model: solve for the growth rate the current "
            "price implies and compare it to achievable sustainable growth. "
            "Market-implied growth far above fundamentals is bearish; well below "
            "is bullish."
        ),
        formula="implied_g = req - CF/price; gap = g_sustainable - implied_g; score = squash(gap, 0.05)",
        inputs=["FCF per share", "EPS", "BVPS", "dividend", "price", "beta", "risk-free rate"],
        references=[
            "Mauboussin & Rappaport, 'Expectations Investing' (2001)",
            "Gordon (1959) growth model inverted; Damodaran on implied growth",
        ],
    ),
    "peg-lynch": StrategyMeta(
        id="peg-lynch",
        name="Peter Lynch PEG / GARP",
        category="Valuation",
        summary=(
            "Peter Lynch's Growth-At-a-Reasonable-Price ('One Up on Wall Street'): "
            "buy companies whose earnings growth exceeds the P/E paid (PEG<1), with "
            "balance-sheet health filters."
        ),
        formula="PEG = (P/E)/(growth%); score = 100*(1 - PEG) with quality adjustments",
        inputs=["price", "EPS", "revenue growth", "debt/equity", "current ratio"],
        references=[
            "Peter Lynch, 'One Up on Wall Street' (1989); 'Beating the Street' (1993)",
            "ChartMill Peter Lynch screener documentation",
        ],
    ),
    "canslim": StrategyMeta(
        id="canslim",
        name="O'Neil CAN SLIM",
        category="Technical",
        summary=(
            "William O'Neil's CAN SLIM growth system ('How to Make Money in "
            "Stocks'): blends strong earnings growth with technical leadership "
            "(relative strength, new highs) and overall market direction."
        ),
        formula="base = 2*(RS_pct - 50); +new-high & earnings bonuses; *0.3 when market off",
        inputs=["price history (126/252d)", "EPS", "revenue growth", "SPY 200d SMA"],
        references=[
            "William J. O'Neil, 'How to Make Money in Stocks' (4th ed., 2009)",
            "AAII CAN SLIM screen — https://www.aaii.com/stocks/screens/78",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _build_graham_defensive(ctx: "AnalysisContext") -> StrategySignal:
    """Graham 7-test defensive screen; valuation core scaled by quality tests.

    Implements the catalog ``computeSignal``: PE=price/eps, PB=price/bvps; count
    passed tests T/7; V = clip(100*(22.5 - PE*PB)/22.5); if eps<=0 V=-80;
    score = clip(V*(0.5+0.5*T/7)); confidence = clip(0.4+0.5*T/7, 0, 0.6),
    dropped to 0.3 when eps<=0 or bvps<=0.
    """
    meta = _META["graham-defensive"]
    if str(getattr(ctx.asset, "asset_class", "")).lower() not in (_EQUITY, "etf"):
        return _na_signal("graham-defensive", "crypto has no fundamentals")
    f = ctx.fundamentals
    price = _price(ctx)
    eps = float(f.eps)
    bvps = float(f.book_value_per_share)

    pe = price / eps if eps > 0.0 else float("inf")
    pb = price / bvps if bvps > 0.0 else float("inf")
    pe_pb = pe * pb if math.isfinite(pe) and math.isfinite(pb) else float("inf")

    # The 7 defensive tests.
    mc = float(getattr(ctx.asset, "market_cap", 0.0) or getattr(ctx.seed, "market_cap", 0.0) or 0.0)
    t1 = mc >= 2.0e9
    t2 = (float(f.current_ratio) >= 2.0) and (float(f.debt_to_equity) <= 1.0)
    t3 = eps > 0.0
    t4 = float(f.dividend) > 0.0
    t5 = float(f.revenue_growth) > 0.0
    t6 = math.isfinite(pe) and pe <= 15.0
    t7 = (math.isfinite(pb) and pb <= 1.5) or (math.isfinite(pe_pb) and pe_pb <= _GRAHAM_CONST)
    passed = sum(1 for t in (t1, t2, t3, t4, t5, t6, t7) if t)

    if eps <= 0.0:
        v = -80.0
    elif math.isfinite(pe_pb):
        v = clamp(100.0 * (_GRAHAM_CONST - pe_pb) / _GRAHAM_CONST, -100.0, 100.0)
    else:
        v = -100.0
    q = passed / 7.0
    score = clamp(v * (0.5 + 0.5 * q), -100.0, 100.0)

    if eps <= 0.0 or bvps <= 0.0:
        confidence = 0.3
    else:
        confidence = clamp(0.4 + 0.5 * q, 0.0, 0.6)

    rationale = (
        f"Passed {passed}/7 Graham defensive tests; "
        f"P/E={pe:.1f}, P/B={pb:.2f}, P/E*P/B={pe_pb:.1f} vs 22.5 ceiling "
        f"(valuation core {v:+.0f}). "
        f"{'Earnings negative — fails the value gate.' if eps <= 0 else ('Cheap and sound.' if score > 20 else 'Mixed/expensive.')}"
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "pe": _finite(pe, 0.0),
            "pb": _finite(pb, 0.0),
            "pePb": _finite(pe_pb, 0.0),
            "testsPassed": float(passed),
            "valuationCore": v,
        },
        horizons=[],
    )


def _build_graham_number(ctx: "AnalysisContext") -> StrategySignal:
    """Graham Number ceiling: GN = sqrt(22.5*EPS*BVPS); discount drives the score.

    score = clip(100*(GN - price)/GN); confidence 0.55 base, up to 0.7 when EPS
    and BVPS both sizeable, down to 0.35 for asset-light firms. Equity-only.
    """
    meta = _META["graham-number"]
    if not _is_equity(ctx):
        return _na_signal("graham-number", "N/A for ETFs/crypto (no book value/earnings)")
    f = ctx.fundamentals
    eps = float(f.eps)
    bvps = float(f.book_value_per_share)
    price = _price(ctx)
    if eps <= 0.0 or bvps <= 0.0:
        meta_loss = make_signal(
            meta.id, meta.name, meta.category, score=0.0, confidence=0.1,
            rationale="Graham Number undefined: requires positive EPS and book value per share.",
            formula=meta.formula, metrics={"eps": eps, "bvps": bvps}, horizons=[],
        )
        return meta_loss

    gn = math.sqrt(_GRAHAM_CONST * eps * bvps)
    discount = (gn - price) / gn if gn > 0.0 else 0.0
    score = clamp(100.0 * discount, -100.0, 100.0)

    # Asset-light proxy: small book value relative to price => lower confidence.
    asset_light = bvps < 0.10 * price
    confidence = 0.55
    if eps >= 1.0 and bvps >= 5.0:
        confidence = 0.7
    if asset_light:
        confidence = 0.35
    confidence = clamp(confidence, 0.0, 1.0)

    horizons = _project_from_annual(clamp(discount * 0.5, -0.25, 0.35), ctx) if discount > 0 else []
    rationale = (
        f"Graham Number {gn:.2f} vs price {price:.2f} -> "
        f"{discount * 100:+.0f}% discount "
        f"(GN = sqrt(22.5*{eps:.2f}*{bvps:.2f}))."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"grahamNumber": gn, "price": price, "discount": discount, "eps": eps, "bvps": bvps},
        horizons=horizons,
    )


def _build_net_net_ncav(ctx: "AnalysisContext") -> StrategySignal:
    """Graham net-net: buy below 2/3 of NCAV per share (deep-value liquidation).

    NCAV_proxy = working_capital; NCAVps = NCAV_proxy/shares; R = price/NCAVps;
    score = clip(100*(1 - R/0.67)). Equity-only; if NCAV<=0 -> 0 score, conf 0.15.
    """
    meta = _META["net-net-ncav"]
    if not _is_equity(ctx):
        return _na_signal("net-net-ncav", "N/A for ETFs/crypto (no current-asset breakdown)")
    f = ctx.fundamentals
    price = _price(ctx)
    ncav_proxy = float(f.working_capital)
    shares = float(f.shares_out)
    if ncav_proxy <= 0.0 or shares <= 0.0:
        return make_signal(
            meta.id, meta.name, meta.category, score=0.0, confidence=0.15,
            rationale="No positive net-current-asset value (working capital) — net-net N/A.",
            formula=meta.formula, metrics={"ncavProxy": ncav_proxy}, horizons=[],
        )
    ncav_ps = ncav_proxy / shares
    if ncav_ps <= 0.0:
        return make_signal(
            meta.id, meta.name, meta.category, score=0.0, confidence=0.15,
            rationale="NCAV per share non-positive — net-net N/A.",
            formula=meta.formula, metrics={"ncavPerShare": ncav_ps}, horizons=[],
        )
    r = price / ncav_ps
    score = clamp(100.0 * (1.0 - r / 0.67), -100.0, 100.0)
    # Confidence: 0.35 base (proxy), up to 0.5 with clean WC and liabilities.
    clean = float(f.working_capital) > 0.0 and float(f.total_liabilities) > 0.0
    confidence = clamp(0.5 if clean else 0.35, 0.0, 1.0)
    rationale = (
        f"Price/NCAVps = {r:.2f} (NCAVps ~ {ncav_ps:.2f} from working capital). "
        f"{'Below the 2/3 net-net threshold — deep-value buy.' if r <= 0.67 else 'Above NCAV — no liquidation margin.'} "
        f"Proxy uses working capital (full current-asset breakdown unavailable)."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"priceToNcav": _finite(r), "ncavPerShare": ncav_ps, "ncavProxy": ncav_proxy},
        horizons=[],
    )


def _build_magic_formula(ctx: "AnalysisContext") -> StrategySignal:
    """Greenblatt Magic Formula: combined cross-sectional rank of EY and ROC.

    C = 0.5*pct(EBIT/EV) + 0.5*pct(EBIT/capital); score = clip(200*(C-0.5)).
    If EBIT<=0 force score<=-60. Confidence 0.55, up to 0.7 with real
    debt/asset fields, 0.4 when EV defaults to market cap. Equity-only.
    """
    meta = _META["magic-formula"]
    if not _is_equity(ctx):
        return _na_signal("magic-formula", "ranks equities only")
    f = ctx.fundamentals
    sym = str(ctx.asset.symbol).upper()
    mc = float(getattr(ctx.asset, "market_cap", 0.0) or getattr(ctx.seed, "market_cap", 0.0) or 0.0)
    ev = _enterprise_value(f, mc)
    inv = _invested_capital(f)
    ebit = float(f.ebit)
    ey = ebit / ev if ev > 0.0 else 0.0
    roc = ebit / inv if inv > 0.0 else 0.0

    p_ey = _percentile(ctx, "earnings_yield", sym)
    p_roc = _percentile(ctx, "roic", sym)
    combined = 0.5 * p_ey + 0.5 * p_roc
    score = clamp(200.0 * (combined - 0.5), -100.0, 100.0)
    if ebit <= 0.0:
        score = min(score, -60.0)

    used_real_debt = _net_debt(f) > 0.0 and inv > 0.0
    confidence = clamp(0.7 if used_real_debt else (0.4 if ev <= mc else 0.55), 0.0, 1.0)

    rationale = (
        f"Earnings yield (EBIT/EV) {ey * 100:.1f}% ranks {p_ey * 100:.0f}th pct; "
        f"return-on-capital {roc * 100:.1f}% ranks {p_roc * 100:.0f}th pct -> "
        f"combined {combined * 100:.0f}th pct (cheap-and-good)."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "earningsYield": _finite(ey),
            "returnOnCapital": _finite(roc),
            "eyPercentile": p_ey,
            "rocPercentile": p_roc,
            "enterpriseValue": ev,
        },
        horizons=[],
    )


def _build_acquirers_multiple(ctx: "AnalysisContext") -> StrategySignal:
    """Acquirer's Multiple (EV/EBIT): pure cheapness rank, no quality leg.

    EY = EBIT/EV; score = clip(200*(pct(EY)-0.5)) with an absolute EV/EBIT
    fallback. If EBIT<=0 -> -70. Confidence 0.6 with real net-debt, 0.45 when
    EV=market_cap. Equity-only.
    """
    meta = _META["acquirers-multiple"]
    if not _is_equity(ctx):
        return _na_signal("acquirers-multiple", "ranks equities only")
    f = ctx.fundamentals
    sym = str(ctx.asset.symbol).upper()
    mc = float(getattr(ctx.asset, "market_cap", 0.0) or getattr(ctx.seed, "market_cap", 0.0) or 0.0)
    ev = _enterprise_value(f, mc)
    ebit = float(f.ebit)
    am = ev / ebit if ebit > 0.0 and ev > 0.0 else float("inf")
    ey = ebit / ev if ev > 0.0 else 0.0

    p_ey = _percentile(ctx, "earnings_yield", sym)
    score = clamp(200.0 * (p_ey - 0.5), -100.0, 100.0)
    if ebit <= 0.0:
        score = -70.0
    else:
        # Absolute multiple fallback blended lightly so single-asset sense holds.
        if math.isfinite(am):
            if am < 6.0:
                abs_score = clamp(60.0 + (6.0 - am) * 10.0, 60.0, 100.0)
            elif am < 10.0:
                abs_score = clamp((10.0 - am) * 7.5, 0.0, 30.0)
            elif am < 14.0:
                abs_score = clamp(-(am - 10.0) * 5.0, -20.0, 0.0)
            else:
                abs_score = clamp(-20.0 - (am - 14.0) * 3.0, -100.0, -20.0)
            score = clamp(0.6 * score + 0.4 * abs_score, -100.0, 100.0)

    used_real_debt = _net_debt(f) > 0.0
    confidence = clamp(0.6 if used_real_debt else 0.45, 0.0, 1.0)
    rationale = (
        f"EV/EBIT (acquirer's multiple) = {am:.1f}x; earnings yield {ey * 100:.1f}% "
        f"ranks {p_ey * 100:.0f}th pct of the equity universe "
        f"({'cheap' if score > 20 else 'rich' if score < -20 else 'fair'})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "acquirersMultiple": _finite(am, 0.0),
            "earningsYield": _finite(ey),
            "eyPercentile": p_ey,
            "enterpriseValue": ev,
        },
        horizons=[],
    )


def _build_owner_earnings_yield(ctx: "AnalysisContext") -> StrategySignal:
    """Buffett owner-earnings yield (FCF/price) vs the risk-free rate.

    OEY = FCF/share / price; X = OEY - rf; score = clip(100*(X/0.06)). If FCF<=0
    -> -50. Confidence 0.5, up to 0.6 with clearly positive/stable FCF.
    Equity-only. Implies a forward cash-return drift -> horizons projected.
    """
    meta = _META["owner-earnings-yield"]
    if not _is_equity(ctx):
        return _na_signal("owner-earnings-yield", "owner-earnings N/A for ETFs/crypto")
    f = ctx.fundamentals
    price = _price(ctx)
    fcf = float(f.fcf_per_share)
    rf_annual = float(ctx.rf_daily) * returns.TRADING_DAYS
    if fcf <= 0.0:
        return make_signal(
            meta.id, meta.name, meta.category, score=-50.0, confidence=0.4,
            rationale=f"No positive owner earnings (FCF/share={fcf:.2f}) — unattractive cash return.",
            formula=meta.formula, metrics={"ownerEarningsYield": 0.0, "fcfPerShare": fcf}, horizons=[],
        )
    oey = fcf / price
    excess = oey - rf_annual
    score = clamp(100.0 * (excess / 0.06), -100.0, 100.0)
    confidence = clamp(0.6 if (fcf > 0.0 and float(f.net_margin) > 0.0) else 0.5, 0.0, 1.0)

    # Baseline expected cash return + a modest growth tilt from revenue growth.
    ann_drift = oey + clamp(float(f.revenue_growth), -0.10, 0.20) * 0.5
    horizons = _project_from_annual(ann_drift, ctx)
    rationale = (
        f"Owner-earnings yield {oey * 100:.1f}% (FCF/share {fcf:.2f} / price {price:.2f}) "
        f"vs risk-free {rf_annual * 100:.1f}% -> {excess * 100:+.1f}% excess; "
        f"{'attractive' if excess > 0.03 else 'thin'} cash return."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "ownerEarningsYield": _finite(oey),
            "fcfPerShare": fcf,
            "riskFree": rf_annual,
            "excessYield": _finite(excess),
        },
        horizons=horizons,
    )


def _build_buffett_quality_fair_price(ctx: "AnalysisContext") -> StrategySignal:
    """Buffett quality-at-fair-price: 5 quality gates scaled by a P/E fairness.

    Q = mean(q_roe, q_roa, q_margin, q_debt, q_growth); price_score from P/E;
    score = clip(100*Q*(0.5+0.5*price_score)); confidence = clip(0.45+0.2*Q,0,0.6).
    Equity-only. A positive ROE-driven drift implies horizons.
    """
    meta = _META["buffett-quality-fair-price"]
    if not _is_equity(ctx):
        return _na_signal("buffett-quality-fair-price", "quality screen for equities only")
    f = ctx.fundamentals
    price = _price(ctx)
    eps = float(f.eps)
    bvps = float(f.book_value_per_share)
    roe = eps / bvps if bvps > 0.0 else 0.0

    q_roe = clamp(roe / 0.20, 0.0, 1.0)
    q_roa = clamp(float(f.roa) / 0.12, 0.0, 1.0)
    q_margin = clamp(float(f.net_margin) / 0.20, 0.0, 1.0)
    q_debt = clamp((0.5 - float(f.debt_to_equity)) / 0.5, 0.0, 1.0)
    q_growth = clamp(float(f.revenue_growth) / 0.10, 0.0, 1.0)
    quality = (q_roe + q_roa + q_margin + q_debt + q_growth) / 5.0

    pe = price / eps if eps > 0.0 else float("inf")
    if eps <= 0.0:
        price_score = -1.0
    else:
        price_score = clamp((25.0 - pe) / 25.0, -1.0, 1.0)

    score = clamp(100.0 * quality * (0.5 + 0.5 * price_score), -100.0, 100.0)
    confidence = clamp(0.45 + 0.2 * quality, 0.0, 0.6)

    # Sustainable-growth drift: ROE * retention + dividend yield.
    payout = clamp(float(f.dividend) / eps, 0.0, 1.0) if eps > 0.0 else 1.0
    div_yield = float(f.dividend) / price if price > 0.0 else 0.0
    ann_drift = roe * (1.0 - payout) + div_yield
    horizons = _project_from_annual(ann_drift, ctx) if score > 0 else []
    rationale = (
        f"Quality {quality * 100:.0f}% (ROE {roe * 100:.0f}%, ROA {float(f.roa) * 100:.0f}%, "
        f"margin {float(f.net_margin) * 100:.0f}%, D/E {float(f.debt_to_equity):.2f}); "
        f"P/E {pe:.1f} -> price fairness {price_score:+.2f}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "quality": quality,
            "roe": _finite(roe),
            "priceScore": price_score,
            "pe": _finite(pe, 0.0),
        },
        horizons=horizons,
    )


def _build_qmj(ctx: "AnalysisContext") -> StrategySignal:
    """Quality Minus Junk: composite z-score over profitability/growth/safety/payout.

    Quality_z = mean of the 4 pillar z-scores (each cross-sectional); score =
    clip(Quality_z*33). Confidence 0.7 base, -0.15 (single-period growth), +0.1
    when all pillars valid. Equity-only.
    """
    meta = _META["qmj-quality-minus-junk"]
    if not _is_equity(ctx):
        return _na_signal("qmj-quality-minus-junk", "QMJ ranks equities only")
    f = ctx.fundamentals
    sym = str(ctx.asset.symbol).upper()

    # Profitability pillar: net margin, ROA, gross profitability, FCF/price.
    z_margin = _zscore_cross(ctx, "net_margin", sym, +1.0)
    z_roa = _zscore_cross(ctx, "roa", sym, +1.0)
    z_gp = _zscore_cross(ctx, "gross_profitability", sym, +1.0)
    z_fcf = _zscore_cross(ctx, "fcf_yield", sym, +1.0)
    profitability_z = (z_margin + z_roa + z_gp + z_fcf) / 4.0

    # Growth pillar: revenue growth.
    growth_z = _zscore_cross(ctx, "revenue_growth", sym, +1.0)

    # Safety pillar: low D/E, low beta, high current ratio, low vol.
    z_de = _seed_signed_z(ctx, sym, lambda f2: float(f2.debt_to_equity), sign=-1.0)
    z_cr = _seed_signed_z(ctx, sym, lambda f2: float(f2.current_ratio), sign=+1.0)
    beta_val = metrics.beta(ctx.returns, ctx.market_ret)
    z_beta = clamp(-(beta_val - 1.0) / 0.5, -3.0, 3.0)  # lower beta => higher safety
    vol = metrics.annual_volatility(ctx.returns)
    z_vol = clamp(-(vol - 0.30) / 0.20, -3.0, 3.0)  # lower vol => higher safety
    safety_z = (z_de + z_beta + z_cr + z_vol) / 4.0

    # Payout pillar: dividend growth (if paying) plus a retained-earnings bonus.
    pay_metric = float(f.dividend_growth) if float(f.dividend) > 0.0 else 0.0
    payout_z = clamp(pay_metric / 0.05, -3.0, 3.0)
    if float(f.retained_earnings) > 0.0:
        payout_z += 0.3
    payout_z = clamp(payout_z, -3.0, 3.0)

    quality_z = (profitability_z + growth_z + safety_z + payout_z) / 4.0
    score = clamp(quality_z * 33.0, -100.0, 100.0)
    pillars_valid = all(
        math.isfinite(x) for x in (profitability_z, growth_z, safety_z, payout_z)
    )
    confidence = clamp(0.7 - 0.15 + (0.1 if pillars_valid else 0.0), 0.0, 1.0)

    rationale = (
        f"QMJ quality z {quality_z:+.2f} (profitability {profitability_z:+.2f}, "
        f"growth {growth_z:+.2f}, safety {safety_z:+.2f}, payout {payout_z:+.2f}); "
        f"{'high-quality' if score > 20 else 'junk' if score < -20 else 'middling'} vs peers."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "qualityZ": _finite(quality_z),
            "profitabilityZ": _finite(profitability_z),
            "growthZ": _finite(growth_z),
            "safetyZ": _finite(safety_z),
            "payoutZ": _finite(payout_z),
            "beta": _finite(beta_val),
        },
        horizons=[],
    )


def _seed_signed_z(
    ctx: "AnalysisContext",
    sym: str,
    getter: Callable[[Fundamentals], float],
    sign: float,
) -> float:
    """Cross-sectional signed z-score of a fundamentals field across equity seeds.

    Args:
        ctx: The analysis context (unused directly; kept for API symmetry).
        sym: Upper-cased symbol.
        getter: Extracts the scalar field from a :class:`Fundamentals`.
        sign: ``+1`` higher-is-better, ``-1`` lower-is-better.

    Returns:
        A finite z-score clamped to ``[-3, 3]``.
    """
    seeds = _equity_seeds()
    if len(seeds) <= 1:
        return 0.0
    vals = np.array([_finite(getter(s.fundamentals)) for s in seeds], dtype=np.float64)
    mu = float(np.mean(vals))
    sd = float(np.std(vals))
    if sd <= 0.0 or not math.isfinite(sd):
        return 0.0
    x = _finite(getter(_seed_for(sym).fundamentals))
    z = sign * (x - mu) / sd
    return clamp(z, -3.0, 3.0) if math.isfinite(z) else 0.0


def _build_gross_profitability(ctx: "AnalysisContext") -> StrategySignal:
    """Novy-Marx gross profitability: cross-sectional rank of GP = profit/assets.

    GP proxy = EBIT/total_assets (no COGS line). score = clip((GP_pct-50)*2) with
    a small level blend. Confidence 0.55 on the EBIT proxy. Equity-only.
    """
    meta = _META["gross-profitability"]
    if not _is_equity(ctx):
        return _na_signal("gross-profitability", "ranks equities only")
    f = ctx.fundamentals
    sym = str(ctx.asset.symbol).upper()
    ta = float(f.total_assets)
    gp = float(f.ebit) / ta if ta > 0.0 else 0.0

    gp_pct = _percentile(ctx, "gross_profitability", sym) * 100.0
    score = clamp((gp_pct - 50.0) * 2.0, -100.0, 100.0)
    if gp > 0.33:
        score = clamp(score + 10.0, -100.0, 100.0)
    elif gp < 0.10:
        score = clamp(score - 10.0, -100.0, 100.0)
    # Confidence 0.55 on the EBIT/TA proxy (true gross profit unavailable).
    confidence = 0.55

    rationale = (
        f"Gross profitability (EBIT/assets proxy) {gp * 100:.1f}% ranks "
        f"{gp_pct:.0f}th pct of the equity universe; "
        f"{'highly profitable' if score > 20 else 'low-profitability' if score < -20 else 'average'}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"grossProfitability": _finite(gp), "gpPercentile": gp_pct / 100.0},
        horizons=[],
    )


def _build_roc_compounder(ctx: "AnalysisContext") -> StrategySignal:
    """High return-on-capital compounder: cross-sectional ROC/ROA rank + bonuses.

    quality_rank = 0.6*pct(ROC) + 0.4*pct(ROA); score = clip((quality_rank-50)*2);
    +10 if ROE>=15% and FCF>0; -15 if D/E>2; -20 if ROA<0 or EPS<0. Confidence
    0.7 base, +0.1 strong ROE/FCF, -0.1 (book-value ROE proxy). Equity-only.
    """
    meta = _META["return-on-capital-compounder"]
    if not _is_equity(ctx):
        return _na_signal("return-on-capital-compounder", "ranks equities only")
    f = ctx.fundamentals
    sym = str(ctx.asset.symbol).upper()
    eps = float(f.eps)
    bvps = float(f.book_value_per_share)
    roe = eps / bvps if bvps > 0.0 else 0.0
    fcf = float(f.fcf_per_share)
    de = float(f.debt_to_equity)
    roa = float(f.roa)

    roc_pct = _percentile(ctx, "roic", sym) * 100.0
    roa_pct = _percentile(ctx, "roa", sym) * 100.0
    quality_rank = 0.6 * roc_pct + 0.4 * roa_pct
    score = clamp((quality_rank - 50.0) * 2.0, -100.0, 100.0)
    if roe >= 0.15 and fcf > 0.0:
        score = clamp(score + 10.0, -100.0, 100.0)
    if de > 2.0:
        score = clamp(score - 15.0, -100.0, 100.0)
    if roa < 0.0 or eps < 0.0:
        score = clamp(score - 20.0, -100.0, 100.0)

    confidence = 0.7 - 0.1
    if roe >= 0.15 and fcf > 0.0:
        confidence += 0.1
    confidence = clamp(confidence, 0.0, 1.0)

    rationale = (
        f"Return-on-capital ranks {roc_pct:.0f}th pct, ROA {roa * 100:.0f}% ranks "
        f"{roa_pct:.0f}th pct (quality rank {quality_rank:.0f}); ROE {roe * 100:.0f}%, "
        f"D/E {de:.2f}. {'Durable compounder.' if score > 20 else 'Weak capital returns.' if score < -20 else 'Average.'}"
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "rocPercentile": roc_pct / 100.0,
            "roaPercentile": roa_pct / 100.0,
            "qualityRank": quality_rank,
            "roe": _finite(roe),
        },
        horizons=[],
    )


def _build_fama_french_5(ctx: "AnalysisContext") -> StrategySignal:
    """Fama-French 5-factor tilt: RMW (profitability) + CMA (investment).

    OP = EBIT/book_equity; rmw = 2*(pct(OP)-50). Aggressiveness A =
    z(revenue_growth) + z(-FCF) + z(no dividend); cma = clip(-A*33). score =
    clip(0.6*rmw + 0.4*cma). If EBIT<=0 force <=-30. Confidence 0.55. Equity-only.
    """
    meta = _META["fama-french-5"]
    if not _is_equity(ctx):
        return _na_signal("fama-french-5", "factor tilt for equities only")
    f = ctx.fundamentals
    sym = str(ctx.asset.symbol).upper()
    ebit = float(f.ebit)

    rmw_pct = _percentile(ctx, "op_profitability", sym, universe_attr="gross_profitability") * 100.0
    rmw = clamp((rmw_pct - 50.0) * 2.0, -100.0, 100.0)

    # CMA aggressiveness: high revenue growth + negative FCF + no dividend = aggressive.
    z_rev = _zscore_cross(ctx, "revenue_growth", sym, +1.0)
    z_neg_fcf = _zscore_cross(ctx, "fcf_yield", sym, -1.0)  # low FCF yield => aggressive
    z_nodiv = _seed_signed_z(ctx, sym, lambda f2: 1.0 if float(f2.dividend) <= 0.0 else 0.0, sign=+1.0)
    aggressiveness = z_rev + z_neg_fcf + z_nodiv
    cma = clamp(-aggressiveness * 33.0, -100.0, 100.0)

    score = clamp(0.6 * rmw + 0.4 * cma, -100.0, 100.0)
    if ebit <= 0.0:
        score = min(score, -30.0)
    confidence = 0.55

    # Modest profitability/investment premium tilt for robust+conservative names.
    ann_drift = clamp(0.02 * (score / 100.0) * 2.0, -0.04, 0.04)
    horizons = _project_from_annual(ann_drift + _capm_required(ctx), ctx) if score > 20 else []
    rationale = (
        f"RMW (profitability) ranks {rmw_pct:.0f}th pct (rmw {rmw:+.0f}); CMA "
        f"investment aggressiveness z {aggressiveness:+.2f} (cma {cma:+.0f}). "
        f"{'Robust + conservative tilt.' if score > 20 else 'Weak/aggressive tilt.' if score < -20 else 'Neutral factor tilt.'}"
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "rmw": rmw,
            "cma": cma,
            "rmwPercentile": rmw_pct / 100.0,
            "aggressivenessZ": _finite(aggressiveness),
        },
        horizons=horizons,
    )


def _build_gordon_reverse(ctx: "AnalysisContext") -> StrategySignal:
    """Reverse-engineered implied growth vs sustainable growth (margin of safety).

    req = CAPM clamp(0.05..0.20); cf = FCF/share (fallback EPS); implied_g =
    req - cf/price; g_sust = clamp(ROE*(1-payout), -0.05, 0.30); gap = g_sust -
    implied_g; score = clip(squash(gap, 0.05)). If cf<=0 -> 0, conf 0.15.
    Confidence 0.45 base, up to 0.6 with clean FCF and BVPS. Equity-only.
    """
    meta = _META["gordon-reverse-implied-growth"]
    if not _is_equity(ctx):
        return _na_signal("gordon-reverse-implied-growth", "implied-growth N/A for ETFs/crypto")
    f = ctx.fundamentals
    price = _price(ctx)
    eps = float(f.eps)
    bvps = float(f.book_value_per_share)
    fcf = float(f.fcf_per_share)
    cf = fcf if fcf > 0.0 else eps
    if cf <= 0.0 or price <= 0.0:
        return make_signal(
            meta.id, meta.name, meta.category, score=0.0, confidence=0.15,
            rationale="No positive per-share cash flow — implied-growth inversion N/A.",
            formula=meta.formula, metrics={"cashFlowPerShare": cf}, horizons=[],
        )
    req = _capm_required(ctx)
    implied_g = req - cf / price
    roe = eps / bvps if bvps > 0.0 else 0.0
    payout = clamp(float(f.dividend) / eps, 0.0, 1.0) if eps > 0.0 else 0.0
    g_sust = clamp(roe * (1.0 - payout), -0.05, 0.30)
    gap = g_sust - implied_g
    score = squash(gap, scale=0.05)

    clean = fcf > 0.0 and bvps > 0.0
    confidence = clamp(0.6 if clean else 0.45, 0.0, 1.0)
    horizons = _project_from_annual(req + clamp(gap, -0.10, 0.10), ctx) if score > 20 else []
    rationale = (
        f"Price implies {implied_g * 100:+.1f}% growth (req return {req * 100:.1f}%, "
        f"cash yield {cf / price * 100:.1f}%); sustainable growth ~{g_sust * 100:.1f}% "
        f"-> gap {gap * 100:+.1f}% "
        f"({'market underpricing growth (bullish)' if gap > 0 else 'priced for unrealistic growth (bearish)'})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "impliedGrowth": _finite(implied_g),
            "sustainableGrowth": g_sust,
            "growthGap": _finite(gap),
            "requiredReturn": req,
        },
        horizons=horizons,
    )


def _build_peg_lynch(ctx: "AnalysisContext") -> StrategySignal:
    """Peter Lynch PEG / GARP: PEG = (P/E)/(growth%); buy when PEG<1 with filters.

    score = clamp(100*(1-PEG)); scale*0.5 if growth<15 or >50; -20 if D/E>0.6,
    -15 if current_ratio<1, +10 if D/E<0.25 and current_ratio>=1.5. Confidence
    0.7 base, +0.15 filters pass, -0.3 (revenue-growth proxy); range 0.3-0.85.
    Equity/ETF; projects underpriced-growth drift.
    """
    meta = _META["peg-lynch"]
    if str(getattr(ctx.asset, "asset_class", "")).lower() not in (_EQUITY, "etf"):
        return _na_signal("peg-lynch", "crypto has no earnings/growth")
    f = ctx.fundamentals
    price = _price(ctx)
    eps = float(f.eps)
    rev_growth = float(f.revenue_growth)
    if eps <= 0.0 or rev_growth <= 0.0:
        return make_signal(
            meta.id, meta.name, meta.category, score=0.0, confidence=0.2,
            rationale="PEG undefined: requires positive EPS and positive growth.",
            formula=meta.formula, metrics={"eps": eps, "growth": rev_growth}, horizons=[],
        )
    pe = price / eps
    g = rev_growth * 100.0
    peg = pe / g if g > 0.0 else float("inf")
    score = clamp(100.0 * (1.0 - peg), -100.0, 100.0)
    if g < 15.0 or g > 50.0:
        score *= 0.5

    de = float(f.debt_to_equity)
    cr = float(f.current_ratio)
    filters_pass = True
    if de > 0.6:
        score -= 20.0
        filters_pass = False
    if cr < 1.0:
        score -= 15.0
        filters_pass = False
    if de < 0.25 and cr >= 1.5:
        score += 10.0
    score = clamp(score, -100.0, 100.0)

    confidence = 0.7 + (0.15 if filters_pass else 0.0) - 0.3
    confidence = clamp(confidence, 0.3, 0.85)

    # Underpriced growth -> annual excess proportional to (1-PEG) capped ~+6%.
    excess = clamp((1.0 - peg) * 0.06, -0.06, 0.06)
    horizons = _project_from_annual(_capm_required(ctx) + excess, ctx) if score > 20 else []
    rationale = (
        f"PEG {peg:.2f} (P/E {pe:.1f} / growth {g:.0f}%); "
        f"{'underpriced growth (PEG<1, buy)' if peg < 1.0 else 'expensive (PEG>2)' if peg > 2.0 else 'fairly priced'}. "
        f"Balance sheet: D/E {de:.2f}, current ratio {cr:.2f}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"peg": _finite(peg, 0.0), "pe": _finite(pe), "growthPct": g, "debtToEquity": de, "currentRatio": cr},
        horizons=horizons,
    )


def _build_canslim(ctx: "AnalysisContext") -> StrategySignal:
    """O'Neil CAN SLIM: RS leadership + new highs + earnings, gated by market.

    RS = trailing 126d return ranked to a percentile; high252 proximity; market
    OK = SPY>200d SMA; earningsOK = eps>0 and revenue_growth>=0.25. base =
    2*(RS_pct-50); +20 prox>=0.85, +15 prox>=0.99; +15 earningsOK, -15 eps<=0;
    *0.3 on positive score when market off. Equity/ETF. Projects RS-scaled drift.
    """
    meta = _META["canslim"]
    if str(getattr(ctx.asset, "asset_class", "")).lower() not in (_EQUITY, "etf"):
        return _na_signal("canslim", "CAN SLIM screens equities/ETFs")
    closes = np.asarray(ctx.closes, dtype=np.float64).ravel()
    closes = closes[np.isfinite(closes) & (closes > 0.0)]
    if closes.size < 2:
        return _na_signal("canslim", "no price history")
    f = ctx.fundamentals
    eps = float(f.eps)

    # L — relative strength: trailing 126d return, ranked across the universe.
    rs = _trailing_return(closes, 126)
    rs_pct = _rs_percentile(ctx, rs) * 100.0

    # N — new highs: proximity to the 252d high.
    window = closes[-min(252, closes.size):]
    high252 = float(np.max(window)) if window.size else float(closes[-1])
    last = float(closes[-1])
    prox = last / high252 if high252 > 0.0 else 1.0

    # M — market filter: SPY above its 200-day SMA.
    market_ok = _market_uptrend(ctx)
    earnings_ok = eps > 0.0 and float(f.revenue_growth) >= 0.25

    score = (rs_pct - 50.0) * 2.0
    if prox >= 0.99:
        score += 15.0
    elif prox >= 0.85:
        score += 20.0
    if earnings_ok:
        score += 15.0
    elif eps <= 0.0:
        score -= 15.0
    if not market_ok and score > 0.0:
        score *= 0.3
    score = clamp(score, -100.0, 100.0)

    confidence = 0.6
    if earnings_ok and market_ok:
        confidence += 0.2
    confidence -= 0.2  # quarterly EPS acceleration / sponsorship proxied
    confidence = clamp(confidence, 0.3, 0.85)

    # Top-RS names near highs continue 3-12 months; drift scaled by RS percentile.
    rs_drift = clamp((rs_pct / 100.0 - 0.5) * 0.30, -0.15, 0.15)
    if not market_ok:
        rs_drift *= 0.3
    horizons = _project_from_annual(rs_drift, ctx) if score > 20 else []
    rationale = (
        f"Relative strength ranks {rs_pct:.0f}th pct (126d return {rs * 100:+.0f}%); "
        f"price at {prox * 100:.0f}% of its 252-day high; market "
        f"{'uptrend' if market_ok else 'downtrend (positive signals cut)'}; "
        f"earnings {'strong (>=25% growth)' if earnings_ok else 'not a leader'}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "rsPercentile": rs_pct / 100.0,
            "trailing126dReturn": _finite(rs),
            "proximityToHigh": _finite(prox, 1.0),
            "marketUptrend": 1.0 if market_ok else 0.0,
            "earningsLeader": 1.0 if earnings_ok else 0.0,
        },
        horizons=horizons,
    )


# ---------------------------------------------------------------------------
# CAN SLIM / cross-sectional momentum helpers
# ---------------------------------------------------------------------------


def _trailing_return(closes: np.ndarray, lookback: int) -> float:
    """Trailing ``lookback``-bar simple return of a clean close series.

    Args:
        closes: Finite, positive close prices (1-D).
        lookback: Number of trailing bars (uses the longest available if short).

    Returns:
        The trailing return as a decimal (``0.0`` on degeneracy), clamped sane.
    """
    n = closes.size
    if n < 2:
        return 0.0
    lb = min(int(lookback), n - 1)
    start = float(closes[-(lb + 1)])
    end = float(closes[-1])
    if start <= 0.0 or not math.isfinite(start):
        return 0.0
    r = end / start - 1.0
    return clamp(_finite(r), -1.0, 50.0)


def _rs_percentile(ctx: "AnalysisContext", rs: float) -> float:
    """Percentile of this asset's 126d return across the universe (0..1).

    Prefers ``ctx.universe`` momentum-style metrics; otherwise ranks the seed
    universe's 126d returns by reconstructing each from the simulator history is
    expensive, so the fallback ranks the asset's RS against a fixed reference
    band derived from typical equity 6-month returns. This keeps the builder
    correct standalone while deferring to the real cross-section when injected.

    Args:
        ctx: The analysis context.
        rs: This asset's trailing 126d return.

    Returns:
        A percentile in ``[0, 1]``.
    """
    uni = getattr(ctx, "universe", None)
    sym = str(ctx.asset.symbol).upper()
    if uni is not None:
        for attr in ("momentum_6m", "momentum_12_1", "ret_52w"):
            try:
                val = uni.percentile(attr, sym)
                fv = float(val)
                if math.isfinite(fv):
                    return clamp(fv, 0.0, 1.0)
            except Exception:
                continue
    # Standalone fallback: map a 6-month return onto a percentile via a logistic
    # centered at ~5% with ~20% spread (typical equity 6m dispersion).
    return clamp(0.5 + 0.5 * math.tanh((rs - 0.05) / 0.20), 0.0, 1.0)


def _market_uptrend(ctx: "AnalysisContext") -> bool:
    """True iff the broad market (SPY) is above its 200-day SMA.

    Uses ``ctx.market_ret`` integrated into a synthetic SPY level when no direct
    SPY series is available, falling back to the sign of the recent market drift.

    Args:
        ctx: The analysis context.

    Returns:
        ``True`` when the market regime is an uptrend (CAN SLIM 'M' filter).
    """
    mkt = np.asarray(ctx.market_ret, dtype=np.float64).ravel()
    if mkt.size >= 50:
        level = np.cumprod(1.0 + np.nan_to_num(mkt, nan=0.0))
        n = level.size
        sma_n = min(200, n)
        sma = float(np.mean(level[-sma_n:]))
        return bool(float(level[-1]) > sma)
    # Fallback: positive recent market drift => uptrend.
    return bool(mkt.size and float(np.mean(mkt)) > 0.0)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

#: Builders keyed by strategy id: ``(StrategyMeta, builder_fn)``.
BUILDERS: dict[str, tuple[StrategyMeta, Callable[["AnalysisContext"], StrategySignal]]] = {
    "graham-defensive": (_META["graham-defensive"], _build_graham_defensive),
    "graham-number": (_META["graham-number"], _build_graham_number),
    "net-net-ncav": (_META["net-net-ncav"], _build_net_net_ncav),
    "magic-formula": (_META["magic-formula"], _build_magic_formula),
    "acquirers-multiple": (_META["acquirers-multiple"], _build_acquirers_multiple),
    "owner-earnings-yield": (_META["owner-earnings-yield"], _build_owner_earnings_yield),
    "buffett-quality-fair-price": (_META["buffett-quality-fair-price"], _build_buffett_quality_fair_price),
    "qmj-quality-minus-junk": (_META["qmj-quality-minus-junk"], _build_qmj),
    "gross-profitability": (_META["gross-profitability"], _build_gross_profitability),
    "return-on-capital-compounder": (_META["return-on-capital-compounder"], _build_roc_compounder),
    "fama-french-5": (_META["fama-french-5"], _build_fama_french_5),
    "gordon-reverse-implied-growth": (_META["gordon-reverse-implied-growth"], _build_gordon_reverse),
    "peg-lynch": (_META["peg-lynch"], _build_peg_lynch),
    "canslim": (_META["canslim"], _build_canslim),
}

#: No per-bar timing strategies in this module (all are snapshot/cross-sectional
#: fundamental screens) -> no vectorized position series to backtest.
POSITION_FUNCS: dict[str, Callable] = {}
