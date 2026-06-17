"""The analysis engine: turn one symbol into a full composite ``AssetAnalysis``.

The engine ties together the market-data provider, the quant layer, and the
strategy registry. For a symbol it:

    1. Builds an :class:`AnalysisContext` (aligned histories, factors,
       fundamentals, OHLC arrays, and the cross-sectional :class:`UniverseStats`).
    2. Runs **all** registered strategy builders to get one
       :class:`~app.schemas.StrategySignal` per model (now ~73).
    3. Computes the :class:`~app.schemas.RiskMetrics` (beta, vol, Sharpe,
       Sortino, VaR95, CVaR95, max drawdown, Calmar).
    4. Blends a single 5-horizon ``expectedReturns`` (confidence-weighted mean
       across the projecting signals — always exactly 5 entries).
    5. Forms a composite score (confidence-weighted mean of signal scores,
       lightly shrunk toward zero by disagreement) and the resulting stance.
    6. Generates a narrative ``rationaleSummary`` and 3-5 ``topReasons`` from the
       strongest-contributing signals.

It also serves the cross-asset views the API needs: ranked
``recommendations``, a single-strategy ``strategy_ranking``, a dashboard
``market_summary``, and a per-symbol ``montecarlo`` result.

Every public method is defensive: a single bad symbol can never raise out of
``analyze`` (and therefore out of any aggregate). Per-symbol analyses are cached,
and the cross-sectional :class:`UniverseStats` is built **once per engine pass**
and cached alongside (invalidated together with the analysis cache).

The :class:`UniverseStats` exposes the cross-sectional metrics the V2
cross-sectional / factor / allocation strategies need (earnings yield, ROIC,
profitability, momentum, vol, beta, dividend / shareholder / FCF yields, P/E,
P/B, PEG, 52-week-high distance, …) keyed by upper-cased symbol, plus
``percentile(metric, symbol)`` (0..1, 1 = highest) and ``rank(metric, symbol,
ascending)`` helpers. It also carries per-symbol ``closes`` and ``price`` maps so
the cross-sectional pairs-trading / price-rank strategies can run.
"""

from __future__ import annotations

import datetime as _dt
import math
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app.market.provider import MarketDataProvider, get_provider
from app.market.universe import AssetSeed, Fundamentals, get_seed
from app.quant import metrics, montecarlo, projection, returns
from app.quant import risk as _risk
from app.quant import technical
from app.quant.returns import HORIZON_DAYS
from app.schemas import (
    Asset,
    AssetAnalysis,
    Breadth,
    ExpectedReturn,
    HORIZONS,
    IndexLevel,
    MarketSummary,
    MonteCarloBand,
    MonteCarloBin,
    MonteCarloResult,
    RankingEntry,
    Recommendation,
    RegimeInfo,
    RiskMetrics,
    SectorPerf,
    StrategyRanking,
    StrategySignal,
)
from app.strategies.base import clamp, stance_from_score
from app.strategies.registry import (
    META_BY_ID,
    build_signals,
)

__all__ = [
    "AnalysisContext",
    "AnalysisEngine",
    "UniverseStats",
    "build_universe_stats",
]


# How many trailing days of history the engine pulls for analysis.
_ANALYSIS_DAYS: int = 1260

# How many trailing OHLC candles to pull for indicator-based strategies. One
# year+ is enough for ATR/ADX/Donchian/Ichimoku/52-week-high while staying cheap.
_CANDLE_LIMIT: int = 400

# Trading days used for annualization (mirrors quant/returns.TRADING_DAYS).
_TD: int = returns.TRADING_DAYS

# Equity-risk premium used to build the CAPM shrinkage prior for the projection
# engine (R1). Mirrors ``app.quant.projection._ERP_ANNUAL`` (~4.5%).
_ERP_ANNUAL: float = 0.045

# Default annual risk-free rate when none can be inferred from the factor series.
_DEFAULT_RF_ANNUAL: float = 0.04

# Strategy ids whose backtest reliability / historical hit-rate is meaningfully
# above the registry average; their confidence is given a small reliability
# bonus when feeding the composite (R4 "strategy reliability") and the drift
# ensemble. Kept cheap (a static prior) so ``analyze()`` never runs a live
# backtest (anti-stall).
_RELIABLE_STRATEGIES: frozenset[str] = frozenset(
    {
        "momentum",
        "trend-ols",
        "tsmom",
        "dual-momentum",
        "absolute-momentum-overlay",
        "faber-taa",
        "golden-cross",
        "fama-french",
        "fama-french-5",
        "low-vol-anomaly",
        "betting-against-beta",
        "qmj-quality-minus-junk",
        "gross-profitability",
        "magic-formula",
    }
)


# ---------------------------------------------------------------------------
# Cross-sectional universe statistics
# ---------------------------------------------------------------------------


@dataclass
class UniverseStats:
    """Cross-sectional metrics across the whole universe, computed once per pass.

    All metric dicts are keyed by **upper-cased** symbol. Missing / degenerate
    values are stored as finite ``0.0`` rather than NaN. The percentile helper
    returns a value in ``[0, 1]`` (1 = best / highest) and the rank helper a
    1-based integer rank.

    Attributes:
        symbols: All universe symbols (upper-cased), in declaration order.
        asset_class: ``symbol -> asset class`` ('equity'/'crypto'/'etf').
        sector: ``symbol -> sector`` label.
        earnings_yield: EBIT / EV (EV = market cap + net debt; fallback cap).
        roic: EBIT / invested capital (net assets proxy).
        op_profitability: EBIT / book equity (RMW-style operating profitability).
        gross_profitability: gross-profit proxy / total assets (EBIT/TA proxy).
        roa: Return on assets (decimal).
        net_margin: Net income / sales (decimal).
        revenue_growth: Year-over-year revenue growth (decimal).
        momentum_12_1: Trailing 12-1 month price momentum.
        momentum_6m: Trailing ~6-month price return.
        ret_52w: ``price / 52w-high - 1`` (``<= 0``).
        annual_vol: Annualized volatility of daily returns (decimal).
        beta: Market beta of daily returns.
        dividend_yield: Annual dividend / price.
        shareholder_yield: (dividend + net-buyback proxy) / price.
        fcf_yield: Free cash flow per share / price.
        pe: Price / EPS (0 when EPS <= 0).
        pb: Price / book value per share (0 when BVPS <= 0).
        peg: (P/E) / (revenue growth %) (0 when undefined).
        closes: ``symbol -> trailing closes`` (numpy array) for pairs-trading.
        price: ``symbol -> latest price``.
    """

    symbols: list[str] = field(default_factory=list)
    asset_class: dict[str, str] = field(default_factory=dict)
    sector: dict[str, str] = field(default_factory=dict)
    earnings_yield: dict[str, float] = field(default_factory=dict)
    roic: dict[str, float] = field(default_factory=dict)
    op_profitability: dict[str, float] = field(default_factory=dict)
    gross_profitability: dict[str, float] = field(default_factory=dict)
    roa: dict[str, float] = field(default_factory=dict)
    net_margin: dict[str, float] = field(default_factory=dict)
    revenue_growth: dict[str, float] = field(default_factory=dict)
    momentum_12_1: dict[str, float] = field(default_factory=dict)
    momentum_6m: dict[str, float] = field(default_factory=dict)
    ret_52w: dict[str, float] = field(default_factory=dict)
    annual_vol: dict[str, float] = field(default_factory=dict)
    beta: dict[str, float] = field(default_factory=dict)
    dividend_yield: dict[str, float] = field(default_factory=dict)
    shareholder_yield: dict[str, float] = field(default_factory=dict)
    fcf_yield: dict[str, float] = field(default_factory=dict)
    pe: dict[str, float] = field(default_factory=dict)
    pb: dict[str, float] = field(default_factory=dict)
    peg: dict[str, float] = field(default_factory=dict)
    closes: dict[str, np.ndarray] = field(default_factory=dict)
    price: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def _metric_dict(self, metric: str) -> dict[str, float] | None:
        """Return the per-symbol metric dict named ``metric`` (or ``None``).

        Only numeric scalar metric dicts are eligible (the ``closes`` /
        ``symbols`` / class / sector maps are excluded from ranking).
        """
        if metric in (
            "symbols",
            "closes",
            "price",
            "asset_class",
            "sector",
        ):
            return None
        d = getattr(self, metric, None)
        return d if isinstance(d, dict) else None

    def percentile(self, metric: str, symbol: str) -> float:
        """Cross-sectional percentile of ``symbol`` for ``metric`` (0..1).

        The percentile is the mid-rank fraction (strictly-below + half the ties)
        across every symbol with a finite value for ``metric``; 1 means the
        symbol has the highest value in the universe.

        Args:
            metric: Name of a numeric metric dict on this object.
            symbol: Symbol to rank (case-insensitive).

        Returns:
            A percentile in ``[0, 1]`` (``0.5`` when the metric / symbol is
            unavailable or the cross-section is degenerate).
        """
        d = self._metric_dict(metric)
        if not d:
            return 0.5
        key = symbol.strip().upper()
        if key not in d:
            return 0.5
        vals = np.array(
            [v for v in d.values() if math.isfinite(float(v))], dtype=np.float64
        )
        n = vals.size
        if n <= 1:
            return 0.5
        target = float(d[key])
        if not math.isfinite(target):
            return 0.5
        below = float(np.sum(vals < target))
        ties = float(np.sum(vals == target))
        pct = (below + 0.5 * ties) / float(n)
        return clamp(pct, 0.0, 1.0)

    def rank(self, metric: str, symbol: str, ascending: bool = False) -> int:
        """1-based rank of ``symbol`` for ``metric`` across the universe.

        Args:
            metric: Name of a numeric metric dict on this object.
            symbol: Symbol to rank (case-insensitive).
            ascending: When ``True`` rank 1 = lowest value; otherwise rank 1 =
                highest value.

        Returns:
            A 1-based rank (``0`` when the metric / symbol is unavailable).
        """
        d = self._metric_dict(metric)
        if not d:
            return 0
        key = symbol.strip().upper()
        if key not in d:
            return 0
        items = [
            (s, float(v)) for s, v in d.items() if math.isfinite(float(v))
        ]
        if not items:
            return 0
        items.sort(key=lambda kv: kv[1], reverse=not ascending)
        for idx, (s, _v) in enumerate(items, start=1):
            if s == key:
                return idx
        return 0


def _finite(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _enterprise_value(f: Fundamentals, market_cap: float) -> float:
    """Enterprise-value proxy = market cap + net-debt proxy (fallback cap).

    Net debt is approximated from leverage and book equity
    (``debt_to_equity * BVPS * shares_out``) when no debt series exists.
    """
    mc = float(market_cap) if market_cap and math.isfinite(market_cap) else 0.0
    if mc <= 0.0:
        return 0.0
    nd = float(f.debt_to_equity) * float(f.book_value_per_share) * float(f.shares_out)
    nd = nd if math.isfinite(nd) and nd > 0.0 else 0.0
    ev = mc + nd
    return ev if math.isfinite(ev) and ev > 0.0 else mc


def _invested_capital(f: Fundamentals) -> float:
    """Invested-capital proxy = net assets (total assets - total liabilities)."""
    net_assets = float(f.total_assets) - float(f.total_liabilities)
    return net_assets if math.isfinite(net_assets) and net_assets > 0.0 else 0.0


def build_universe_stats(
    provider: MarketDataProvider,
    symbols: list[str] | None = None,
) -> UniverseStats:
    """Compute the cross-sectional :class:`UniverseStats` for the universe.

    Pulls each asset's snapshot, fundamentals and trailing closes once, derives
    every cross-sectional metric vectorially, and returns a fully-populated
    :class:`UniverseStats`. Intended to be called **once per engine pass** and
    cached; it is defensive (a single failing symbol is skipped, never raised).

    Args:
        provider: The market-data provider.
        symbols: Optional explicit symbol list; defaults to the provider's
            full universe.

    Returns:
        A populated :class:`UniverseStats`.
    """
    stats = UniverseStats()

    try:
        assets = provider.list_assets()
    except Exception:  # pragma: no cover - defensive
        assets = []
    asset_by_symbol: dict[str, Asset] = {}
    for a in assets:
        try:
            asset_by_symbol[str(a.symbol).upper()] = a
        except Exception:  # pragma: no cover - defensive
            continue

    if symbols is None:
        syms = [str(a.symbol).upper() for a in assets]
    else:
        syms = [str(s).upper() for s in symbols]

    # Shared market-factor total returns (for cross-sectional beta).
    try:
        factors = provider.factor_history(days=_ANALYSIS_DAYS)
        mkt = np.asarray(factors.get("mkt", np.empty(0)), dtype=np.float64).ravel()
    except Exception:  # pragma: no cover - defensive
        mkt = np.empty(0, dtype=np.float64)

    for sym in syms:
        try:
            seed = get_seed(sym)
        except Exception:  # pragma: no cover - defensive
            continue
        asset = asset_by_symbol.get(sym)
        f: Fundamentals = seed.fundamentals
        cls = str(seed.asset_class).lower()

        # Price + trailing closes.
        try:
            closes = np.asarray(
                provider.history(sym, days=_ANALYSIS_DAYS), dtype=np.float64
            ).ravel()
        except Exception:  # pragma: no cover - defensive
            closes = np.empty(0, dtype=np.float64)
        closes = closes[np.isfinite(closes) & (closes > 0.0)]
        if asset is not None and asset.price and math.isfinite(float(asset.price)) and asset.price > 0:
            price = float(asset.price)
        elif closes.size:
            price = float(closes[-1])
        else:
            price = float(seed.base_price) if seed.base_price else 1.0

        mc = 0.0
        if asset is not None and asset.market_cap:
            mc = _finite(asset.market_cap)
        if mc <= 0.0 and seed.market_cap:
            mc = _finite(seed.market_cap)

        # Daily returns for vol / beta / momentum.
        ret = returns.simple_returns(closes) if closes.size else np.empty(0)

        stats.symbols.append(sym)
        stats.asset_class[sym] = cls
        stats.sector[sym] = str(seed.sector or "Other")

        # --- valuation / quality fundamentals ---
        ev = _enterprise_value(f, mc)
        ebit = float(f.ebit)
        stats.earnings_yield[sym] = _finite(ebit / ev) if ev > 0.0 else 0.0
        inv = _invested_capital(f)
        stats.roic[sym] = _finite(ebit / inv) if inv > 0.0 else 0.0
        be = float(f.book_value_per_share) * float(f.shares_out)
        if math.isfinite(be) and be > 0.0:
            stats.op_profitability[sym] = _finite(ebit / be)
        elif float(f.total_assets) > 0.0:
            stats.op_profitability[sym] = _finite(ebit / float(f.total_assets))
        else:
            stats.op_profitability[sym] = 0.0
        ta = float(f.total_assets)
        stats.gross_profitability[sym] = _finite(ebit / ta) if ta > 0.0 else 0.0
        stats.roa[sym] = _finite(f.roa)
        stats.net_margin[sym] = _finite(f.net_margin)
        stats.revenue_growth[sym] = _finite(f.revenue_growth)
        stats.fcf_yield[sym] = _finite(f.fcf_per_share / price) if price > 0.0 else 0.0

        eps = float(f.eps)
        bvps = float(f.book_value_per_share)
        stats.pe[sym] = _finite(price / eps) if eps > 0.0 else 0.0
        stats.pb[sym] = _finite(price / bvps) if bvps > 0.0 else 0.0
        growth_pct = float(f.revenue_growth) * 100.0
        if eps > 0.0 and growth_pct > 0.0:
            stats.peg[sym] = _finite((price / eps) / growth_pct)
        else:
            stats.peg[sym] = 0.0

        # --- dividend / shareholder yields ---
        div_yield = _finite(f.dividend / price) if price > 0.0 else 0.0
        stats.dividend_yield[sym] = div_yield
        # Net-buyback proxy: retained-earnings growth funds modest buybacks for
        # cash-generative, low-payout names. Proxy buyback yield from FCF not
        # paid as dividends (capped), shareholder yield = dividend + buyback.
        fcf = float(f.fcf_per_share)
        retained_fcf = max(0.0, fcf - float(f.dividend))
        buyback_yield = _finite(min(retained_fcf, fcf) / price) * 0.3 if price > 0.0 else 0.0
        stats.shareholder_yield[sym] = clamp(div_yield + buyback_yield, -1.0, 1.0)

        # --- momentum / vol / beta / 52w ---
        stats.momentum_12_1[sym] = _finite(technical.momentum_12_1(closes))
        stats.momentum_6m[sym] = _finite(_trailing_return(closes, 126))
        if closes.size:
            win = closes[-min(_TD, closes.size):]
            high52 = float(np.max(win)) if win.size else price
            stats.ret_52w[sym] = clamp(_finite(price / high52 - 1.0) if high52 > 0 else 0.0, -1.0, 0.0)
        else:
            stats.ret_52w[sym] = 0.0
        vol = metrics.annual_volatility(ret) if ret.size else 0.0
        stats.annual_vol[sym] = _finite(vol)
        b = metrics.beta(ret, mkt) if ret.size and mkt.size else 1.0
        stats.beta[sym] = _finite(b, 1.0)

        # --- raw series for the pairs / price-rank strategies ---
        if closes.size >= 30:
            stats.closes[sym] = closes
        stats.price[sym] = price

    return stats


def _trailing_return(closes: np.ndarray, lookback: int) -> float:
    """Trailing ``lookback``-bar simple return of a clean close series."""
    n = closes.size
    if n < 2:
        return 0.0
    lb = min(int(lookback), n - 1)
    start = float(closes[-(lb + 1)])
    end = float(closes[-1])
    if start <= 0.0 or not math.isfinite(start):
        return 0.0
    r = end / start - 1.0
    return float(r) if math.isfinite(r) else 0.0


@dataclass
class AnalysisContext:
    """All per-asset inputs a strategy builder needs, with aligned arrays.

    The return / factor arrays are trailing-aligned to a common length so any
    pairwise statistic (beta, regressions, …) lines up the most recent
    observations.

    Attributes:
        asset: The :class:`~app.schemas.Asset` snapshot (price, change, …).
        seed: The static :class:`~app.market.universe.AssetSeed`.
        closes: Daily closing prices (length ``returns + 1``).
        returns: Daily simple returns of the asset.
        market_ret: Daily *total* market-factor returns (aligned length).
        smb: Daily SMB-factor returns (aligned length).
        hml: Daily HML-factor returns (aligned length).
        rf_daily: Scalar daily risk-free rate (mean of the rf series).
        fundamentals: The asset's :class:`~app.market.universe.Fundamentals`.
        market_cap: Market capitalisation (0.0 when unknown).
        universe: Cross-sectional :class:`UniverseStats` for the whole universe
            (shared across every per-asset context in a pass; V2 additive).
        all_symbols: Every universe symbol (upper-cased; V2 additive).
        highs: Daily high prices aligned to ``closes`` (V2 additive; falls back
            to ``closes`` when OHLC is unavailable).
        lows: Daily low prices aligned to ``closes`` (V2 additive).
        volumes: Daily volumes aligned to ``closes`` (V2 additive; empty array
            when unavailable).
        now: Deterministic "current time" for calendar strategies (the engine
            reads the system clock; tests inject a fixed value; V2 additive).
    """

    asset: Asset
    seed: AssetSeed
    closes: np.ndarray
    returns: np.ndarray
    market_ret: np.ndarray
    smb: np.ndarray
    hml: np.ndarray
    rf_daily: float
    fundamentals: Fundamentals
    market_cap: float
    universe: UniverseStats = field(default_factory=UniverseStats)
    all_symbols: list[str] = field(default_factory=list)
    highs: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    lows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    volumes: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    now: _dt.datetime | None = None


class AnalysisEngine:
    """Run the full quant model suite over the universe and cache the results.

    Args:
        provider: A :class:`~app.market.provider.MarketDataProvider`. Defaults to
            the process-wide singleton from
            :func:`~app.market.provider.get_provider`.
    """

    def __init__(self, provider: MarketDataProvider | None = None) -> None:
        """Initialise the engine with a market-data provider and empty cache."""
        self._provider: MarketDataProvider = provider or get_provider()
        self._cache: dict[str, AssetAnalysis] = {}
        # Per-symbol raw signal lists, cached so the cross-sectional percentile
        # pre-pass (``_absolute_score``) and the full ``analyze`` do NOT each run
        # the 73 builders independently (the builders are pure for a fixed pass).
        # Keyed by upper-cased symbol; invalidated with the analysis cache.
        self._signals_cache: dict[str, list[StrategySignal]] = {}
        # Per-symbol projection parameters (the EXACT capped/shrunk daily drift and
        # the per-horizon daily vol that ``project()`` derived) so ``montecarlo()``
        # can run from the same numbers and agree with ``analyze()`` (R3). Keyed by
        # upper-cased symbol; invalidated together with the analysis cache.
        self._proj_params: dict[str, dict] = {}
        self._universe: UniverseStats | None = None
        self._all_symbols: list[str] | None = None
        # Cross-sectional composite percentiles (R5 relative component), built
        # once on first ``recommendations()``-style pass and cached.
        self._composite_percentiles: dict[str, float] | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Universe statistics (built once per pass, cached)
    # ------------------------------------------------------------------

    def universe_stats(self) -> UniverseStats:
        """Return the cached cross-sectional :class:`UniverseStats`.

        Built once on first use and reused across every per-asset analysis until
        :meth:`clear_cache` invalidates it (together with the analysis cache).

        Returns:
            The shared :class:`UniverseStats`.
        """
        cached = self._universe
        if cached is not None:
            return cached
        with self._lock:
            if self._universe is None:
                self._universe = build_universe_stats(self._provider)
                self._all_symbols = list(self._universe.symbols)
            return self._universe

    def clear_cache(self) -> None:
        """Drop the per-symbol analysis cache and the universe stats together."""
        with self._lock:
            self._cache.clear()
            self._signals_cache.clear()
            self._proj_params.clear()
            self._composite_percentiles = None
            self._universe = None
            self._all_symbols = None

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def context(self, symbol: str, now: _dt.datetime | None = None) -> AnalysisContext:
        """Build an :class:`AnalysisContext` for ``symbol``.

        Pulls the asset snapshot, daily closes, OHLC candle arrays, factor
        histories and fundamentals, attaches the shared :class:`UniverseStats`,
        then trailing-aligns the factor/return arrays to a common length.

        Args:
            symbol: Asset ticker (case-insensitive).
            now: Optional deterministic "current time" for calendar strategies.

        Returns:
            A populated :class:`AnalysisContext`.

        Raises:
            KeyError: If the symbol is unknown (propagated so the API can 404).
        """
        seed = get_seed(symbol)
        asset = self._provider.get_asset(symbol)
        closes = np.asarray(
            self._provider.history(symbol, days=_ANALYSIS_DAYS), dtype=np.float64
        ).ravel()
        asset_ret = returns.simple_returns(closes)

        # OHLC + volume arrays for indicator-based strategies (defensive).
        highs, lows, volumes = self._ohlc(symbol, closes)

        factors = self._provider.factor_history(days=_ANALYSIS_DAYS)
        mkt = np.asarray(factors.get("mkt", np.empty(0)), dtype=np.float64).ravel()
        smb = np.asarray(factors.get("smb", np.empty(0)), dtype=np.float64).ravel()
        hml = np.asarray(factors.get("hml", np.empty(0)), dtype=np.float64).ravel()
        rf_arr = np.asarray(factors.get("rf", np.empty(0)), dtype=np.float64).ravel()

        # Trailing-align the asset returns and factor series to a common length.
        lengths = [a.size for a in (asset_ret, mkt, smb, hml) if a.size > 0]
        n = min(lengths) if lengths else 0
        if n > 0:
            asset_ret = asset_ret[-n:]
            mkt = mkt[-n:] if mkt.size else np.zeros(n)
            smb = smb[-n:] if smb.size else np.zeros(n)
            hml = hml[-n:] if hml.size else np.zeros(n)
        else:
            asset_ret = np.empty(0, dtype=np.float64)
            mkt = np.empty(0, dtype=np.float64)
            smb = np.empty(0, dtype=np.float64)
            hml = np.empty(0, dtype=np.float64)

        rf_daily = float(np.mean(rf_arr)) if rf_arr.size else 0.0
        if not math.isfinite(rf_daily):
            rf_daily = 0.0

        market_cap = float(asset.market_cap) if asset.market_cap else 0.0
        if not math.isfinite(market_cap):
            market_cap = 0.0

        universe = self.universe_stats()
        all_symbols = list(self._all_symbols or universe.symbols)

        return AnalysisContext(
            asset=asset,
            seed=seed,
            closes=closes,
            returns=asset_ret,
            market_ret=mkt,
            smb=smb,
            hml=hml,
            rf_daily=rf_daily,
            fundamentals=seed.fundamentals,
            market_cap=market_cap,
            universe=universe,
            all_symbols=all_symbols,
            highs=highs,
            lows=lows,
            volumes=volumes,
            now=now,
        )

    def _ohlc(
        self, symbol: str, closes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return aligned (highs, lows, volumes) arrays for ``symbol``.

        Pulls recent OHLCV candles from the provider; on any failure the highs /
        lows fall back to the close series and the volume array is empty, so
        indicator builders always have *something* finite to work with.

        Args:
            symbol: Asset ticker.
            closes: The already-pulled close series (used as the fallback).

        Returns:
            A ``(highs, lows, volumes)`` tuple of finite ``float64`` arrays.
        """
        try:
            candles = self._provider.get_candles(symbol, limit=_CANDLE_LIMIT)
        except Exception:  # pragma: no cover - defensive
            candles = []
        if candles:
            highs = np.asarray([float(c.h) for c in candles], dtype=np.float64)
            lows = np.asarray([float(c.l) for c in candles], dtype=np.float64)
            volumes = np.asarray([float(c.v) for c in candles], dtype=np.float64)
            highs = np.nan_to_num(highs, nan=0.0, posinf=0.0, neginf=0.0)
            lows = np.nan_to_num(lows, nan=0.0, posinf=0.0, neginf=0.0)
            volumes = np.nan_to_num(volumes, nan=0.0, posinf=0.0, neginf=0.0)
            return highs, lows, volumes
        # Fallback: reuse closes for highs/lows, no volume.
        c = np.asarray(closes, dtype=np.float64).ravel()
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        return c.copy(), c.copy(), np.empty(0, dtype=np.float64)

    # ------------------------------------------------------------------
    # Core per-asset analysis
    # ------------------------------------------------------------------

    def analyze(self, symbol: str, now: _dt.datetime | None = None) -> AssetAnalysis:
        """Produce the full composite :class:`~app.schemas.AssetAnalysis`.

        Runs all strategy signals (now ~73), computes risk metrics, blends the
        5-horizon expected returns, forms the composite score / stance, and
        writes the narrative. The result is cached per symbol (keyed by the
        upper-cased ticker) only when ``now`` is not injected (so deterministic
        test injections never poison the shared cache).

        Composite score (signals with confidence ``c_i`` and score ``s_i``)::

            base   = sum(c_i * s_i) / sum(c_i)             (confidence-weighted mean)
            disagr = std(s_i)                              (cross-signal dispersion)
            shrink = 1 / (1 + disagr / 60)                 (more disagreement -> closer to 0)
            composite = clamp(base * shrink, -100, 100)

        Args:
            symbol: Asset ticker (case-insensitive).
            now: Optional deterministic "current time" for calendar strategies
                (e.g. seasonality). When provided the result is not cached.

        Returns:
            A complete :class:`~app.schemas.AssetAnalysis`. Unknown symbols
            propagate a ``KeyError`` (the only case that raises); any internal
            modelling problem degrades gracefully to safe defaults.

        Raises:
            KeyError: If the symbol is unknown.
        """
        key = symbol.strip().upper()
        cacheable = now is None
        if cacheable:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        # Building the context validates the symbol (KeyError if unknown).
        ctx = self.context(symbol, now=now)

        # Build outside the lock: ``_build_analysis`` may itself need the engine
        # (e.g. the lazy cross-sectional percentile map / proj-param cache), so
        # holding the non-reentrant lock across the build would deadlock. The
        # double-check below keeps the cache consistent; building twice on a rare
        # race is harmless (the result is deterministic for a given symbol).
        if cacheable:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
        analysis = self._build_analysis(ctx)
        if cacheable:
            with self._lock:
                existing = self._cache.get(key)
                if existing is not None:
                    return existing
                self._cache[key] = analysis
        return analysis

    def _build_analysis(self, ctx: AnalysisContext) -> AssetAnalysis:
        """Assemble the full :class:`~app.schemas.AssetAnalysis` from a context.

        Runs every signal builder, computes risk metrics, blends horizons, forms
        the composite, writes the narrative, and stamps the strategy count +
        disclaimer. Never raises (builders are individually guarded upstream).

        Args:
            ctx: The fully-populated analysis context.

        Returns:
            A complete :class:`~app.schemas.AssetAnalysis`.
        """
        signals = self._signals_for(ctx)
        risk_metrics = self._risk_metrics(ctx)
        # R1/R2/R3/R6: one projection engine drives the horizons, the scenario
        # fan, the downside (CVaR/prob-of-loss) and the regime — and stashes the
        # exact drift+vol so montecarlo() agrees with the analysis.
        expected_returns, regime = self._project_horizons(ctx, signals)
        composite, confidence = self._composite(ctx, signals, regime)
        recommendation = stance_from_score(composite)
        summary, reasons = self._narrative(ctx, signals, composite, recommendation)

        return AssetAnalysis(
            asset=ctx.asset,
            composite_score=composite,
            recommendation=recommendation,
            confidence=confidence,
            expected_returns=expected_returns,
            risk_metrics=risk_metrics,
            signals=signals,
            rationale_summary=summary,
            top_reasons=reasons,
            updated_at=int(time.time() * 1000),
            regime=regime,
            strategy_count=len(signals),
        )

    def _signals_for(self, ctx: AnalysisContext) -> list[StrategySignal]:
        """Run (and cache) the 73 signal builders for one context.

        The builders are pure for a fixed engine pass, so caching by symbol lets
        the cross-sectional percentile pre-pass (:meth:`_absolute_score`) and the
        full :meth:`_build_analysis` share one set of signals instead of each
        running all 73 builders — roughly halving the cold first-analyze cost.
        Injected-``now`` contexts (e.g. seasonality tests) bypass the cache so a
        deterministic month never poisons the shared signals.

        Args:
            ctx: The analysis context.

        Returns:
            The list of strategy signals for the asset.
        """
        if ctx.now is not None:
            return build_signals(ctx)
        key = str(ctx.asset.symbol).strip().upper()
        cached = self._signals_cache.get(key)
        if cached is not None:
            return cached
        signals = build_signals(ctx)
        # Plain dict assignment is atomic in CPython; builds happen outside the
        # analyze lock so no extra locking is needed.
        self._signals_cache[key] = signals
        return signals

    def _risk_metrics(self, ctx: AnalysisContext) -> RiskMetrics:
        """Compute the annualized :class:`~app.schemas.RiskMetrics` for an asset.

        Args:
            ctx: The analysis context.

        Returns:
            A populated :class:`~app.schemas.RiskMetrics` (all finite floats).
        """
        r = ctx.returns
        prices = ctx.closes
        rf_d = float(ctx.rf_daily)
        beta = metrics.beta(r, ctx.market_ret)
        vol = metrics.annual_volatility(r)
        sharpe = metrics.sharpe(r, rf_d)
        sortino = metrics.sortino(r, rf_d)
        # VaR/CVaR reported as positive loss fractions for the wire DTO.
        var95 = _risk.historical_var(r, conf=0.95)
        cvar95 = _risk.cvar(r, conf=0.95)
        mdd = metrics.max_drawdown(prices)
        calmar = metrics.calmar(r, prices)
        return RiskMetrics(
            beta=self._safe(beta),
            annual_vol=self._safe(vol),
            sharpe=self._safe(sharpe),
            sortino=self._safe(sortino),
            var95=self._safe(var95),
            cvar95=self._safe(cvar95),
            max_drawdown=self._safe(mdd),
            calmar=self._safe(calmar),
        )

    def _project_horizons(
        self, ctx: AnalysisContext, signals: list[StrategySignal]
    ) -> tuple[list[ExpectedReturn], RegimeInfo | None]:
        """Build the credible 5-horizon curve + regime via the projection engine.

        Replaces the old confidence-weighted blend of every signal's own
        lognormal bands (which produced +800% 5Y "expected" returns with
        +26,000% upper bands) with the single financially-credible engine in
        :mod:`app.quant.projection` (STRATEGIES-V2 §4 / R1-R6):

            1. Gather each *projecting* signal's **implied daily log-drift** (R1)
               with its confidence into ``signal_drifts`` (the magnitude of each
               signal is recovered from its own 1D horizon, see
               :meth:`_implied_daily_drift`).
            2. Build the CAPM shrinkage prior ``rf + β·ERP`` as a daily log-drift.
            3. Call :func:`projection.project`, which ensembles + James–Stein
               shrinks + regime-tilts + **caps** the drift, forecasts GARCH vol
               with a term structure, draws fat-tailed / bootstrapped bands,
               builds the bull/base/bear fan and the 95% CVaR, and classifies the
               regime.
            4. Cache the **exact** drift + per-horizon daily vol so
               :meth:`montecarlo` runs from the same numbers (R3: analysis 1Y and
               Monte-Carlo 1Y expected returns agree within ~1pp).

        Always returns exactly five :class:`~app.schemas.ExpectedReturn` (one per
        :data:`~app.schemas.HORIZONS`), each carrying the V2 scenario / CVaR
        fields, plus the :class:`~app.schemas.RegimeInfo`.

        Args:
            ctx: The analysis context (closes, returns, beta inputs, asset class).
            signals: All strategy signals for the asset.

        Returns:
            A ``(expected_returns, regime)`` tuple.
        """
        rets = np.asarray(ctx.returns, dtype=np.float64).ravel()
        closes = np.asarray(ctx.closes, dtype=np.float64).ravel()

        # 1) implied daily drifts (+confidence) from the projecting signals.
        signal_drifts: list[tuple[float, float]] = []
        for sig in signals:
            if not sig.horizons:
                continue
            mu = self._implied_daily_drift(sig)
            if mu is None:
                continue
            conf = float(sig.confidence)
            if not math.isfinite(conf) or conf <= 0.0:
                continue
            signal_drifts.append((mu, conf))

        # 2) CAPM daily-log-drift prior (rf + beta * ERP).
        beta = self._safe(metrics.beta(rets, ctx.market_ret), 1.0)
        capm_drift_daily = self._capm_drift_daily(ctx, beta)

        asset_class = str(ctx.seed.asset_class).lower()

        # 3) run the credible projection engine.
        try:
            projs, regime_dict = projection.project(
                closes=closes,
                returns=rets,
                signal_drifts=signal_drifts,
                rf_daily=float(ctx.rf_daily),
                beta=beta,
                capm_drift_daily=capm_drift_daily,
                asset_class=asset_class,
            )
        except Exception:  # pragma: no cover - defensive (project never raises)
            projs, regime_dict = [], projection.detect_regime(closes)

        # 4) cache the EXACT drift (returned by project) + per-horizon daily vol
        #    for montecarlo() (R3).
        exact_drift = self._safe(
            (regime_dict or {}).get("drift_daily"), 0.0
        )
        self._store_proj_params(ctx, exact_drift, projs)

        out: list[ExpectedReturn] = []
        by_label = {p.horizon: p for p in projs}
        for h in HORIZONS:
            p = by_label.get(h)
            if p is not None:
                out.append(ExpectedReturn(**p.to_dict()))
            else:
                # Defensive neutral entry so the DTO always has 5 horizons.
                out.append(
                    ExpectedReturn(
                        horizon=h,  # type: ignore[arg-type]
                        expected_return_pct=0.0,
                        low=0.0,
                        high=0.0,
                        prob_positive=0.5,
                        annualized_vol=0.0,
                        bull_pct=0.0,
                        base_pct=0.0,
                        bear_pct=0.0,
                        cvar_pct=0.0,
                    )
                )

        regime = self._regime_info(regime_dict)
        return out, regime

    @staticmethod
    def _implied_daily_drift(sig: StrategySignal) -> float | None:
        """Recover a projecting signal's implied daily log-drift from its 1D band.

        Every projecting builder produces its ``horizons`` via
        :func:`app.quant.returns.project_horizons(mu_daily, sigma)`, whose 1D
        ``expectedReturnPct`` is ``(exp(mu_daily) - 1) * 100``. Inverting that
        recovers the strategy's own forward drift, which the projection engine
        then ensembles + shrinks + caps (so a single optimistic strategy can no
        longer push the blended median to +800%).

        Args:
            sig: A strategy signal (assumed to carry per-horizon projections).

        Returns:
            The implied daily log-drift, or ``None`` when no usable 1D / 1Y
            horizon is present.
        """
        one_d = next((h for h in sig.horizons if h.horizon == "1D"), None)
        if one_d is not None:
            er = float(one_d.expected_return_pct) / 100.0
            if math.isfinite(er) and er > -1.0:
                mu = math.log1p(er)
                return mu if math.isfinite(mu) else None
        # Fallback: recover from the 1Y horizon (drift over 252 days).
        one_y = next((h for h in sig.horizons if h.horizon == "1Y"), None)
        if one_y is not None:
            er = float(one_y.expected_return_pct) / 100.0
            if math.isfinite(er) and er > -1.0:
                mu = math.log1p(er) / float(_TD)
                return mu if math.isfinite(mu) else None
        return None

    def _capm_drift_daily(self, ctx: AnalysisContext, beta: float) -> float:
        """Build the CAPM daily log-drift prior ``rf + β·ERP`` for shrinkage (R1).

        Args:
            ctx: The analysis context (for the daily risk-free rate).
            beta: The asset's market beta.

        Returns:
            A finite daily log-drift used as the James–Stein shrinkage target.
        """
        rf_d = self._safe(ctx.rf_daily, 0.0)
        rf_annual = (
            projection.daily_to_annual_drift(rf_d) if rf_d != 0.0 else _DEFAULT_RF_ANNUAL
        )
        b = self._safe(beta, 1.0)
        capm_annual = rf_annual + b * _ERP_ANNUAL
        return projection.annual_to_daily_drift(capm_annual)

    def _store_proj_params(
        self,
        ctx: AnalysisContext,
        drift_daily: float,
        projs: list[projection.HorizonProjection],
    ) -> None:
        """Cache the exact drift + per-horizon daily vol used by ``project()`` (R3).

        ``montecarlo()`` reuses these so it draws GBM paths from the SAME drift
        and vol the analysis horizons were built on (the per-horizon annualized
        GARCH vol converted back to a daily vol), guaranteeing the analysis and
        Monte-Carlo expected returns agree within sampling noise (~1pp).

        Args:
            ctx: The analysis context (for the symbol key).
            drift_daily: The EXACT capped/shrunk daily log-drift that
                :func:`projection.project` used (read from its returned regime).
            projs: The projections produced by :func:`projection.project`.
        """
        # Per-horizon daily vol = annualized horizon vol / sqrt(252).
        vol_daily_by_h: dict[str, float] = {}
        for p in projs:
            ann_vol = self._safe(p.annualized_vol, 0.0) / 100.0
            vol_daily = ann_vol / math.sqrt(_TD) if ann_vol > 0.0 else 1e-4
            vol_daily_by_h[p.horizon] = vol_daily if vol_daily > 0.0 else 1e-4
        key = str(ctx.asset.symbol).strip().upper()
        # A plain dict assignment is atomic in CPython; building happens outside
        # the analyze lock, so no extra locking is needed here.
        self._proj_params[key] = {
            "drift_daily": self._safe(drift_daily, 0.0),
            "vol_daily_by_h": vol_daily_by_h,
        }

    @staticmethod
    def _regime_info(regime_dict: dict | None) -> RegimeInfo | None:
        """Coerce a :func:`projection.detect_regime` dict to a ``RegimeInfo``.

        Args:
            regime_dict: The regime dict (or ``None``).

        Returns:
            A validated :class:`~app.schemas.RegimeInfo`, or ``None`` on failure.
        """
        if not regime_dict:
            return None
        try:
            regime = str(regime_dict.get("regime", "neutral"))
            if regime not in ("bull", "bear", "neutral"):
                regime = "neutral"
            vol_regime = str(regime_dict.get("vol_regime", "normal"))
            if vol_regime not in ("low", "normal", "high"):
                vol_regime = "normal"
            trend = float(regime_dict.get("trend", 0.0))
            score = float(regime_dict.get("score", 0.0))
            return RegimeInfo(
                regime=regime,  # type: ignore[arg-type]
                trend=trend if math.isfinite(trend) else 0.0,
                vol_regime=vol_regime,  # type: ignore[arg-type]
                score=score if math.isfinite(score) else 0.0,
            )
        except Exception:  # pragma: no cover - defensive
            return None

    def _composite(
        self,
        ctx: AnalysisContext,
        signals: list[StrategySignal],
        regime: RegimeInfo | None = None,
    ) -> tuple[float, float]:
        """Compute the calibrated composite score and differentiated confidence.

        Replaces the old over-aggressive disagreement shrink (which collapsed
        nearly every asset to ``HOLD``) and the flat ``~0.3`` confidence with the
        STRATEGIES-V2 R4/R5 design.

        **Composite (R5).** A *reliability-weighted* mean of the signal scores —
        each signal weighted by ``confidence × reliability`` (reliability gives a
        small bonus to strategies with strong historical hit-rates, computed from
        a static prior so no live backtest runs) — with a **milder** dispersion
        penalty than before. The absolute score is then blended with a
        cross-sectional component (the asset's percentile among the universe's
        composites, recentred to ``[-100, 100]``) so the universe yields a
        realistic STANCE MIX instead of 24×HOLD::

            base   = Σ(w_i·s_i) / Σ w_i ,  w_i = conf_i · reliability_i
            disagr = weighted std of s_i
            shrink = 1 / (1 + disagr / 110)            (milder than the old /60)
            abs_sc = base · shrink
            rel_sc = (percentile(asset) - 0.5) · 2 · 100   (cross-sectional)
            composite = 0.65·abs_sc + 0.35·rel_sc

        **Confidence (R4).** Differentiated across ≈ ``[0.2, 0.9]`` and driven by
        four independent factors, each in ``[0, 1]``:

            * **consensus** — the (weighted) share of signals agreeing with the
              composite direction *and* of non-trivial magnitude;
            * **dispersion** — lower cross-signal score std ⇒ higher;
            * **data quality** — history length + whether real fundamentals exist
              for the asset class (equities richer than crypto/ETF);
            * **regime clarity** — ``|regime score|`` (a clear bull/bear is more
              actionable than a muddy neutral tape).

        Two assets with clearly different agreement therefore report clearly
        different confidence.

        Args:
            ctx: The analysis context (for data-quality inputs + the symbol).
            signals: All strategy signals for the asset.
            regime: The projection regime (for the regime-clarity factor).

        Returns:
            A ``(composite_score, confidence)`` tuple. ``composite_score`` is in
            ``[-100, 100]`` and ``confidence`` in ``[0, 1]``.
        """
        if not signals:
            return 0.0, 0.2

        scores = np.array([float(s.score) for s in signals], dtype=np.float64)
        scores = np.nan_to_num(scores, nan=0.0, posinf=100.0, neginf=-100.0)
        confs = np.array([float(s.confidence) for s in signals], dtype=np.float64)
        confs = np.where(np.isfinite(confs) & (confs > 0.0), confs, 1e-6)
        reliab = np.array(
            [
                1.25 if s.strategy_id in _RELIABLE_STRATEGIES else 1.0
                for s in signals
            ],
            dtype=np.float64,
        )
        weights = confs * reliab

        total_w = float(np.sum(weights))
        if total_w <= 0.0:
            base = float(np.mean(scores))
            wmean = base
            wstd = float(np.std(scores)) if scores.size > 1 else 0.0
        else:
            wmean = float(np.sum(weights * scores) / total_w)
            base = wmean
            # Weighted dispersion of the signal scores around the weighted mean.
            wvar = float(np.sum(weights * (scores - wmean) ** 2) / total_w)
            wstd = math.sqrt(max(0.0, wvar))

        # R5: a MILDER dispersion penalty so a real net tilt survives disagreement.
        shrink = 1.0 / (1.0 + wstd / 110.0)
        abs_score = clamp(base * shrink, -100.0, 100.0)

        # R5 cross-sectional component: where does this asset's absolute score sit
        # among the universe's? Blending in the relative rank spreads the stance
        # mix so the universe is not uniformly HOLD. Built lazily + cached.
        pctile = self._composite_percentile(ctx, abs_score)
        rel_score = clamp((pctile - 0.5) * 2.0 * 100.0, -100.0, 100.0)
        composite = clamp(0.65 * abs_score + 0.35 * rel_score, -100.0, 100.0)

        # ---- R4 differentiated confidence ----
        direction = 1.0 if base >= 0.0 else -1.0
        # Consensus: weighted share of the *active* (|score| > 10) signals that
        # agree in direction with the net tilt. By construction this lies in
        # ``[0.5, 1.0]`` (the winning side is always the majority), so it is
        # STRETCHED onto ``[0, 1]`` to use the full dynamic range — this is the
        # most discriminating confidence factor across assets.
        active_mask = np.abs(scores) > 10.0
        agree_mask = active_mask & ((scores * direction) > 0.0)
        agree_w = float(np.sum(weights[agree_mask]))
        active_w = float(np.sum(weights[active_mask]))
        raw_consensus = (agree_w / active_w) if active_w > 0.0 else 0.5
        # ``raw_consensus`` is in ``[0.5, 1.0]`` by construction but, with 73
        # diverse models, in practice spans only ~``[0.52, 0.82]`` across the
        # universe (rarely does every active model line up). Remap that realistic
        # band onto the full ``[0, 1]`` so genuinely different agreement produces
        # clearly different confidence (R4: spread > 0.3).
        consensus = clamp((raw_consensus - 0.52) / 0.30, 0.0, 1.0)

        # Dispersion factor: tight agreement (low std) -> high. Centred so a
        # ~45-point std (the universe's typical level) maps near 0.5 and the band
        # uses its full range across the realistic ~40-70 wstd spread.
        dispersion_factor = clamp(1.0 - (wstd - 35.0) / 45.0, 0.0, 1.0)

        # Data quality: history length + real fundamentals for the asset class.
        data_quality = self._data_quality(ctx)

        # Regime clarity: |score| of the regime read (clear bull/bear is clearer);
        # stretched a little so a moderate regime conviction still contributes.
        regime_clarity = 0.0
        if regime is not None:
            regime_clarity = clamp(abs(float(regime.score)) / 0.6, 0.0, 1.0)

        # Blend the four factors (weights sum to 1) into [0,1]. Consensus carries
        # the most weight (it discriminates assets best across the universe); the
        # others modulate it.
        raw = (
            0.58 * consensus
            + 0.17 * dispersion_factor
            + 0.13 * data_quality
            + 0.12 * regime_clarity
        )
        raw = clamp(raw, 0.0, 1.0)
        # Map onto the actionable [0.18, 0.95] band so confidence is never a flat
        # 0.3 and never an over-confident 1.0, while giving a clearly > 0.3 spread
        # across assets with clearly different signal agreement (R4 re-audit gate).
        confidence = clamp(0.18 + 0.77 * raw, 0.0, 1.0)
        return composite, confidence

    def _data_quality(self, ctx: AnalysisContext) -> float:
        """Estimate a ``[0, 1]`` data-quality score for confidence (R4).

        Combines the available return-history length (more history ⇒ more
        trustworthy statistics) with whether the asset class has *real*
        fundamentals (equities carry balance-sheet data the value/quality models
        use; crypto/ETF do not, so their analysis leans on fewer informative
        signals).

        Args:
            ctx: The analysis context.

        Returns:
            A finite quality score in ``[0, 1]``.
        """
        n = int(np.asarray(ctx.returns, dtype=np.float64).ravel().size)
        # 1 year of history -> ~0.7, 5 years -> ~1.0; <1 month -> low.
        history_q = clamp(n / float(_TD), 0.0, 1.0)
        history_q = clamp(0.2 + 0.8 * history_q, 0.0, 1.0)

        cls = str(ctx.seed.asset_class).lower()
        f = ctx.fundamentals
        has_fundamentals = (
            cls == "equity"
            and (abs(float(f.total_assets)) > 1.0 or float(f.eps) != 0.0)
        )
        class_q = 1.0 if has_fundamentals else 0.6 if cls == "etf" else 0.55

        return clamp(0.5 * history_q + 0.5 * class_q, 0.0, 1.0)

    def _composite_percentile(self, ctx: AnalysisContext, abs_score: float) -> float:
        """Cross-sectional percentile of ``abs_score`` among universe composites.

        Drives the R5 relative component. The per-symbol *absolute* composite
        scores are computed once (cheaply, from cached signal stats — no extra
        analysis pass) and cached; this asset's percentile (0..1, 1 = highest) is
        read from that map. When the map cannot be built (single-asset / degenerate
        universe) a neutral ``0.5`` is returned so the relative term contributes
        nothing.

        Args:
            ctx: The analysis context (for the symbol).
            abs_score: This asset's absolute composite score (a fallback when the
                symbol is missing from the cached map).

        Returns:
            A percentile in ``[0, 1]``.
        """
        table = self._composite_percentiles
        if table is None:
            table = self._build_composite_percentiles()
        key = str(ctx.asset.symbol).strip().upper()
        if key in table:
            return clamp(table[key], 0.0, 1.0)
        # Symbol not in the precomputed map (e.g. an injected ``now`` analysis):
        # the cached map holds percentiles (not raw scores) so a neutral 0.5 is
        # the correct, contribution-free fallback for the relative term.
        return 0.5

    def _build_composite_percentiles(self) -> dict[str, float]:
        """Compute + cache cross-sectional composite percentiles for the universe.

        Builds each universe symbol's *absolute* composite score from cached
        signal statistics — reusing already-cached analyses where available and
        cheaply computing the absolute score otherwise — then converts the
        cross-section to per-symbol percentiles (0..1, 1 = highest). The map is
        cached on the engine and invalidated with :meth:`clear_cache`.

        This runs the full universe **at most once** (lazily, on the first
        confidence/composite computation that needs it) and never recurses into
        :meth:`_composite` (it computes only the cheap *absolute* score), so it
        does not re-sweep the universe repeatedly.

        Returns:
            A ``symbol -> percentile`` map (possibly empty / degenerate).
        """
        cached_table = self._composite_percentiles
        if cached_table is not None:
            return cached_table

        # Build the cross-section WITHOUT holding the lock (each ``_absolute_score``
        # needs the universe stats, whose builder also takes the lock — holding it
        # here would deadlock the non-reentrant lock). Store under the lock below.
        scores: dict[str, float] = {}
        try:
            symbols = list(self._all_symbols or [])
            if not symbols:
                uni = self.universe_stats()
                symbols = list(uni.symbols)
        except Exception:  # pragma: no cover - defensive
            symbols = []
        for sym in symbols:
            key = str(sym).strip().upper()
            existing = self._cache.get(key)
            if existing is not None:
                scores[key] = float(existing.composite_score)
                continue
            abs_sc = self._absolute_score(sym)
            if abs_sc is not None:
                scores[key] = abs_sc

        table: dict[str, float] = {}
        if len(scores) >= 2:
            vals = np.array(list(scores.values()), dtype=np.float64)
            n = vals.size
            for k, v in scores.items():
                below = float(np.sum(vals < v))
                ties = float(np.sum(vals == v))
                table[k] = clamp((below + 0.5 * ties) / float(n), 0.0, 1.0)

        with self._lock:
            if self._composite_percentiles is None:
                self._composite_percentiles = table
            return self._composite_percentiles

    def _absolute_score(self, symbol: str) -> float | None:
        """Cheaply compute one symbol's *absolute* composite score for ranking.

        Runs the signal builders for ``symbol`` and forms the reliability-weighted
        mean with the milder dispersion shrink — the SAME formula as the absolute
        component of :meth:`_composite`, but WITHOUT the cross-sectional blend or
        confidence (so it cannot recurse). Used only to seed the cross-sectional
        percentile map.

        Args:
            symbol: Asset ticker.

        Returns:
            The absolute composite score in ``[-100, 100]``, or ``None`` on
            failure.
        """
        try:
            ctx = self.context(symbol)
            signals = self._signals_for(ctx)
        except Exception:  # pragma: no cover - defensive
            return None
        if not signals:
            return None
        scores = np.array([float(s.score) for s in signals], dtype=np.float64)
        scores = np.nan_to_num(scores, nan=0.0, posinf=100.0, neginf=-100.0)
        confs = np.array([float(s.confidence) for s in signals], dtype=np.float64)
        confs = np.where(np.isfinite(confs) & (confs > 0.0), confs, 1e-6)
        reliab = np.array(
            [1.25 if s.strategy_id in _RELIABLE_STRATEGIES else 1.0 for s in signals],
            dtype=np.float64,
        )
        weights = confs * reliab
        total_w = float(np.sum(weights))
        if total_w <= 0.0:
            wmean = float(np.mean(scores))
            wstd = float(np.std(scores)) if scores.size > 1 else 0.0
        else:
            wmean = float(np.sum(weights * scores) / total_w)
            wvar = float(np.sum(weights * (scores - wmean) ** 2) / total_w)
            wstd = math.sqrt(max(0.0, wvar))
        shrink = 1.0 / (1.0 + wstd / 110.0)
        return clamp(wmean * shrink, -100.0, 100.0)

    def _narrative(
        self,
        ctx: AnalysisContext,
        signals: list[StrategySignal],
        composite: float,
        recommendation: str,
    ) -> tuple[str, list[str]]:
        """Build the rationale summary and 3-5 top reasons from strong signals.

        The reasons are taken from the signals whose ``|score| * confidence``
        contribution is largest and that agree in direction with the composite
        stance (falling back to the strongest signals overall if too few agree).

        Args:
            ctx: The analysis context (for the asset name / class).
            signals: All strategy signals.
            composite: The composite score.
            recommendation: The composite stance label.

        Returns:
            A ``(rationale_summary, top_reasons)`` tuple.
        """
        # Rank signals by contribution magnitude.
        ranked = sorted(
            signals,
            key=lambda s: abs(float(s.score)) * float(s.confidence),
            reverse=True,
        )
        direction = 1.0 if composite >= 0 else -1.0
        agreeing = [
            s
            for s in ranked
            if (float(s.score) >= 0) == (direction >= 0) and abs(float(s.score)) > 5.0
        ]
        chosen = agreeing[:5] if len(agreeing) >= 3 else ranked[:5]
        if not chosen:
            chosen = ranked[:3]

        reasons: list[str] = []
        for s in chosen[:5]:
            reasons.append(f"{s.strategy_name}: {s.rationale}")
        if len(reasons) < 3:
            for s in ranked:
                line = f"{s.strategy_name}: {s.rationale}"
                if line not in reasons:
                    reasons.append(line)
                if len(reasons) >= 3:
                    break

        verb = {
            "STRONG_BUY": "a high-conviction buy",
            "BUY": "a buy",
            "HOLD": "a hold",
            "SELL": "a sell",
            "STRONG_SELL": "a high-conviction sell",
        }.get(recommendation, "a hold")
        n_bull = sum(1 for s in signals if float(s.score) > 5.0)
        n_bear = sum(1 for s in signals if float(s.score) < -5.0)
        summary = (
            f"{ctx.asset.name} ({ctx.asset.symbol}) scores {composite:+.0f}/100 "
            f"across {len(signals)} quant models, making it {verb}. "
            f"{n_bull} model(s) lean bullish and {n_bear} bearish; the strongest "
            f"driver is {chosen[0].strategy_name.lower()}." if chosen else
            f"{ctx.asset.name} ({ctx.asset.symbol}) scores {composite:+.0f}/100 "
            f"across {len(signals)} quant models, making it {verb}."
        )
        return summary, reasons[:5]

    # ------------------------------------------------------------------
    # Cross-asset views
    # ------------------------------------------------------------------

    def recommendations(
        self, limit: int = 12, asset_class: str | None = None
    ) -> list[Recommendation]:
        """Rank the universe by composite score and return the top ``limit``.

        Args:
            limit: Maximum number of recommendations to return.
            asset_class: Optional filter (``'equity'`` / ``'crypto'`` / ``'etf'``).

        Returns:
            A list of :class:`~app.schemas.Recommendation`, rank 1 = best
            composite score (descending). Symbols that fail analysis are skipped.
        """
        analyses = self._all_analyses(asset_class)
        analyses.sort(key=lambda a: float(a.composite_score), reverse=True)
        lim = max(0, int(limit)) if limit else len(analyses)
        out: list[Recommendation] = []
        for rank, a in enumerate(analyses[:lim], start=1):
            out.append(self._to_recommendation(a, rank))
        return out

    def strategy_ranking(
        self, strategy_id: str, limit: int = 20
    ) -> StrategyRanking:
        """Rank every asset by a single strategy's signal score.

        Args:
            strategy_id: Strategy id (must exist in the registry).
            limit: Maximum number of entries to return.

        Returns:
            A :class:`~app.schemas.StrategyRanking` sorted by that strategy's
            score (descending).

        Raises:
            KeyError: If ``strategy_id`` is not a registered strategy.
        """
        sid = strategy_id.strip()
        if sid not in META_BY_ID:
            raise KeyError(f"Unknown strategy: {strategy_id!r}")

        entries: list[tuple[Asset, float, str]] = []
        for sym in self._symbols():
            try:
                analysis = self.analyze(sym)
            except Exception:
                continue
            match = next(
                (s for s in analysis.signals if s.strategy_id == sid), None
            )
            if match is None:
                continue
            entries.append((analysis.asset, float(match.score), match.stance))

        entries.sort(key=lambda e: e[1], reverse=True)
        lim = max(0, int(limit)) if limit else len(entries)
        ranking_entries = [
            RankingEntry(asset=a, score=score, stance=stance)  # type: ignore[arg-type]
            for (a, score, stance) in entries[:lim]
        ]
        return StrategyRanking(strategy_id=sid, entries=ranking_entries)

    def market_summary(self) -> MarketSummary:
        """Build the dashboard :class:`~app.schemas.MarketSummary`.

        Aggregates breadth (advancers/decliners by ``change24hPct``), per-sector
        average change, synthetic index levels per asset class, and the top
        gainers / losers by 24h change.

        Returns:
            A populated :class:`~app.schemas.MarketSummary`.
        """
        assets = self._list_assets()
        advancers = sum(1 for a in assets if a.change24h_pct > 0.01)
        decliners = sum(1 for a in assets if a.change24h_pct < -0.01)
        unchanged = len(assets) - advancers - decliners
        breadth = Breadth(
            advancers=advancers, decliners=decliners, unchanged=unchanged
        )

        # Per-sector average 24h change.
        sector_acc: dict[str, list[float]] = {}
        for a in assets:
            sec = a.sector or "Other"
            sector_acc.setdefault(sec, []).append(float(a.change24h_pct))
        sectors = [
            SectorPerf(
                sector=sec,
                change_pct=self._safe(float(np.mean(vals)) if vals else 0.0),
                count=len(vals),
            )
            for sec, vals in sorted(sector_acc.items())
        ]

        # Synthetic per-class indices: average price level and average change.
        indices = self._build_indices(assets)

        # Top movers (reuse the Recommendation shape via lightweight build).
        movers = sorted(assets, key=lambda a: float(a.change24h_pct), reverse=True)
        top_gainers = [
            self._mover_recommendation(a, rank)
            for rank, a in enumerate(movers[:5], start=1)
        ]
        losers = sorted(assets, key=lambda a: float(a.change24h_pct))
        top_losers = [
            self._mover_recommendation(a, rank)
            for rank, a in enumerate(losers[:5], start=1)
        ]

        return MarketSummary(
            as_of=int(time.time() * 1000),
            breadth=breadth,
            top_gainers=top_gainers,
            top_losers=top_losers,
            sectors=sectors,
            indices=indices,
        )

    def montecarlo(
        self, symbol: str, horizon: str = "1Y", sims: int = 2000
    ) -> MonteCarloResult:
        """Run a GBM Monte Carlo from the SAME drift + vol the analysis used (R3).

        The drift and per-horizon volatility are NOT re-estimated here from raw
        log returns (which is what made the old ``/montecarlo`` 1Y disagree with
        the ``/analysis`` 1Y — e.g. +9.6% vs +13.5% in the live audit). Instead
        the engine reuses the EXACT capped/shrunk daily log-drift and the
        horizon's annualized GARCH vol (converted to a daily vol) that
        :meth:`_project_horizons` derived and cached, via
        :func:`app.quant.projection.mc_summary`. The result is that the analysis
        and Monte-Carlo expected returns for a horizon agree within sampling
        noise (~1pp).

        Args:
            symbol: Asset ticker (case-insensitive).
            horizon: One of :data:`~app.schemas.HORIZONS`; unknown -> ``'1Y'``.
            sims: Number of simulated paths.

        Returns:
            A populated :class:`~app.schemas.MonteCarloResult`.

        Raises:
            KeyError: If the symbol is unknown.
        """
        # Running the analysis (cached) populates the projection params and
        # validates the symbol (KeyError if unknown).
        self.analyze(symbol)
        ctx = self.context(symbol)
        key = str(ctx.asset.symbol).strip().upper()
        hz = horizon if horizon in HORIZONS else "1Y"

        params = self._proj_params.get(key)
        if params is not None:
            mu_daily = self._safe(params.get("drift_daily"), 0.0)
            vol_map = params.get("vol_daily_by_h", {}) or {}
            sigma_daily = self._safe(vol_map.get(hz), 0.0)
        else:  # pragma: no cover - defensive (analyze always stores params)
            mu_daily, sigma_daily = 0.0, 0.0
        if not math.isfinite(mu_daily):
            mu_daily = 0.0
        if not math.isfinite(sigma_daily) or sigma_daily <= 0.0:
            # Fallback to realized daily vol so the fan is never degenerate.
            lr = returns.log_returns(ctx.closes)
            sigma_daily = float(np.std(lr)) if lr.size else 1e-4
            if not math.isfinite(sigma_daily) or sigma_daily <= 0.0:
                sigma_daily = 1e-4

        s0 = float(ctx.asset.price) if ctx.asset.price > 0 else float(ctx.seed.base_price)
        seed = abs(hash(ctx.asset.symbol)) % (2**32)
        summary = projection.mc_summary(
            s0=s0,
            drift_daily=mu_daily,
            vol_daily=sigma_daily,
            horizon=hz,
            sims=max(1, int(sims)),
            seed=seed,
        )

        bands = [MonteCarloBand(**b) for b in summary["bands"]]
        final_dist = [MonteCarloBin(**b) for b in summary["finalDistribution"]]
        return MonteCarloResult(
            symbol=ctx.asset.symbol,
            horizon=summary["horizon"],  # type: ignore[arg-type]
            sims=int(summary["sims"]),
            steps=int(summary["steps"]),
            bands=bands,
            final_distribution=final_dist,
            expected_return_pct=self._safe(summary["expectedReturnPct"]),
            var95_pct=self._safe(summary["var95Pct"]),
            cvar95_pct=self._safe(summary["cvar95Pct"]),
            prob_positive=clamp(summary["probPositive"], 0.0, 1.0),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _symbols(self) -> list[str]:
        """Return all symbols available from the provider's universe.

        Returns:
            A list of ticker strings.
        """
        return [a.symbol for a in self._provider.list_assets()]

    def _list_assets(self) -> list[Asset]:
        """Return the provider's full :class:`~app.schemas.Asset` snapshot list."""
        return list(self._provider.list_assets())

    def _all_analyses(self, asset_class: str | None) -> list[AssetAnalysis]:
        """Analyze every (optionally class-filtered) symbol, skipping failures.

        Args:
            asset_class: Optional class filter.

        Returns:
            A list of successfully-built :class:`~app.schemas.AssetAnalysis`.
        """
        cls = asset_class.strip().lower() if asset_class else None
        out: list[AssetAnalysis] = []
        for a in self._provider.list_assets():
            if cls is not None and str(a.asset_class).lower() != cls:
                continue
            try:
                out.append(self.analyze(a.symbol))
            except Exception:
                continue
        return out

    def _to_recommendation(
        self, analysis: AssetAnalysis, rank: int
    ) -> Recommendation:
        """Convert a full analysis into a ranked :class:`~app.schemas.Recommendation`.

        Args:
            analysis: A completed asset analysis.
            rank: 1-based rank position.

        Returns:
            A populated :class:`~app.schemas.Recommendation`.
        """
        one_year = next(
            (h for h in analysis.expected_returns if h.horizon == "1Y"), None
        )
        er_1y = float(one_year.expected_return_pct) if one_year else 0.0
        headline = analysis.rationale_summary.split(". ")[0]
        return Recommendation(
            rank=rank,
            asset=analysis.asset,
            composite_score=float(analysis.composite_score),
            recommendation=analysis.recommendation,
            confidence=float(analysis.confidence),
            expected_return1y_pct=self._safe(er_1y),
            headline=headline,
            reasons=list(analysis.top_reasons),
        )

    def _mover_recommendation(self, asset: Asset, rank: int) -> Recommendation:
        """Build a lightweight :class:`~app.schemas.Recommendation` for a mover.

        Used by :meth:`market_summary` for top gainers/losers, where a full
        analysis is unnecessary; the 1Y expected return is approximated by the
        24h change so the shape is populated meaningfully.

        Args:
            asset: The mover's :class:`~app.schemas.Asset` snapshot.
            rank: 1-based rank within the gainers/losers list.

        Returns:
            A populated :class:`~app.schemas.Recommendation`.
        """
        change = float(asset.change24h_pct)
        stance = "BUY" if change > 0 else "SELL" if change < 0 else "HOLD"
        headline = (
            f"{asset.symbol} is {'up' if change >= 0 else 'down'} "
            f"{abs(change):.2f}% over 24h."
        )
        return Recommendation(
            rank=rank,
            asset=asset,
            composite_score=clamp(change * 5.0, -100.0, 100.0),
            recommendation=stance,  # type: ignore[arg-type]
            confidence=0.3,
            expected_return1y_pct=self._safe(change),
            headline=headline,
            reasons=[headline],
        )

    def _build_indices(self, assets: list[Asset]) -> list[IndexLevel]:
        """Build synthetic per-asset-class index levels and changes.

        Each index level is the (price-weighted-ish) average of its class's
        prices scaled to a readable level; the change is the average 24h change.

        Args:
            assets: The full asset snapshot list.

        Returns:
            A list of :class:`~app.schemas.IndexLevel`, one per non-empty class.
        """
        groups: dict[str, list[Asset]] = {}
        for a in assets:
            groups.setdefault(str(a.asset_class), []).append(a)

        names = {
            "equity": "GiffMe Equity Index",
            "crypto": "GiffMe Crypto Index",
            "etf": "GiffMe ETF Index",
        }
        out: list[IndexLevel] = []
        for cls, members in groups.items():
            if not members:
                continue
            avg_change = float(np.mean([float(m.change24h_pct) for m in members]))
            # A stable, readable synthetic level from the mean change.
            level = 1000.0 * (1.0 + avg_change / 100.0)
            out.append(
                IndexLevel(
                    name=names.get(cls, f"GiffMe {cls.title()} Index"),
                    level=self._safe(level),
                    change_pct=self._safe(avg_change),
                )
            )
        out.sort(key=lambda i: i.name)
        return out

    @staticmethod
    def _safe(value: float, default: float = 0.0) -> float:
        """Return ``value`` as a finite float, falling back to ``default``.

        Args:
            value: Candidate number.
            default: Substitute for NaN / +-inf.

        Returns:
            A finite float.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default
