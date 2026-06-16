"""The analysis engine: turn one symbol into a full composite ``AssetAnalysis``.

The engine ties together the market-data provider, the quant layer, and the
strategy registry. For a symbol it:

    1. Builds an :class:`AnalysisContext` (aligned histories, factors,
       fundamentals).
    2. Runs **all** registered strategy builders to get one
       :class:`~app.schemas.StrategySignal` per model (>= 18).
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
``analyze`` (and therefore out of any aggregate). Per-symbol analyses are cached.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from app.market.provider import MarketDataProvider, get_provider
from app.market.universe import AssetSeed, Fundamentals, get_seed
from app.quant import metrics, montecarlo, returns
from app.quant import risk as _risk
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

__all__ = ["AnalysisContext", "AnalysisEngine"]


# How many trailing days of history the engine pulls for analysis.
_ANALYSIS_DAYS: int = 1260


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
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def context(self, symbol: str) -> AnalysisContext:
        """Build an :class:`AnalysisContext` for ``symbol``.

        Pulls the asset snapshot, daily closes, factor histories and
        fundamentals, then trailing-aligns the factor/return arrays to a common
        length.

        Args:
            symbol: Asset ticker (case-insensitive).

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
        )

    # ------------------------------------------------------------------
    # Core per-asset analysis
    # ------------------------------------------------------------------

    def analyze(self, symbol: str) -> AssetAnalysis:
        """Produce the full composite :class:`~app.schemas.AssetAnalysis`.

        Runs all strategy signals, computes risk metrics, blends the 5-horizon
        expected returns, forms the composite score / stance, and writes the
        narrative. The result is cached per symbol (keyed by the upper-cased
        ticker).

        Composite score (signals with confidence ``c_i`` and score ``s_i``)::

            base   = sum(c_i * s_i) / sum(c_i)             (confidence-weighted mean)
            disagr = std(s_i)                              (cross-signal dispersion)
            shrink = 1 / (1 + disagr / 60)                 (more disagreement -> closer to 0)
            composite = clamp(base * shrink, -100, 100)

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            A complete :class:`~app.schemas.AssetAnalysis`. Unknown symbols
            propagate a ``KeyError`` (the only case that raises); any internal
            modelling problem degrades gracefully to safe defaults.

        Raises:
            KeyError: If the symbol is unknown.
        """
        key = symbol.strip().upper()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # Building the context validates the symbol (KeyError if unknown).
        ctx = self.context(symbol)

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

            signals = build_signals(ctx)
            risk_metrics = self._risk_metrics(ctx)
            expected_returns = self._blend_horizons(signals)
            composite, confidence = self._composite(signals)
            recommendation = stance_from_score(composite)
            summary, reasons = self._narrative(
                ctx, signals, composite, recommendation
            )

            analysis = AssetAnalysis(
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
            )
            self._cache[key] = analysis
            return analysis

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

    def _blend_horizons(
        self, signals: list[StrategySignal]
    ) -> list[ExpectedReturn]:
        """Blend projecting signals into one 5-horizon expected-return curve.

        For each horizon the engine takes the confidence-weighted mean (across
        all signals that produced a projection for that horizon) of every field:
        ``expectedReturnPct``, ``low``, ``high``, ``probPositive`` and
        ``annualizedVol``. Always returns exactly five entries (one per
        :data:`~app.schemas.HORIZONS`); when no signal projects, a neutral
        zero-return / zero-band curve is returned so the DTO is always complete.

        Args:
            signals: All strategy signals for the asset.

        Returns:
            A list of exactly five :class:`~app.schemas.ExpectedReturn`.
        """
        # Accumulate weighted sums per horizon label.
        acc: dict[str, dict[str, float]] = {
            h: {
                "expectedReturnPct": 0.0,
                "low": 0.0,
                "high": 0.0,
                "probPositive": 0.0,
                "annualizedVol": 0.0,
                "weight": 0.0,
            }
            for h in HORIZONS
        }

        for sig in signals:
            if not sig.horizons:
                continue
            w = float(sig.confidence)
            if not math.isfinite(w) or w <= 0.0:
                w = 1e-6
            for hr in sig.horizons:
                bucket = acc.get(hr.horizon)
                if bucket is None:
                    continue
                bucket["expectedReturnPct"] += w * float(hr.expected_return_pct)
                bucket["low"] += w * float(hr.low)
                bucket["high"] += w * float(hr.high)
                bucket["probPositive"] += w * float(hr.prob_positive)
                bucket["annualizedVol"] += w * float(hr.annualized_vol)
                bucket["weight"] += w

        out: list[ExpectedReturn] = []
        for h in HORIZONS:
            bucket = acc[h]
            wt = bucket["weight"]
            if wt > 0.0:
                er = ExpectedReturn(
                    horizon=h,  # type: ignore[arg-type]
                    expected_return_pct=self._safe(bucket["expectedReturnPct"] / wt),
                    low=self._safe(bucket["low"] / wt),
                    high=self._safe(bucket["high"] / wt),
                    prob_positive=clamp(bucket["probPositive"] / wt, 0.0, 1.0),
                    annualized_vol=self._safe(bucket["annualizedVol"] / wt),
                )
            else:
                er = ExpectedReturn(
                    horizon=h,  # type: ignore[arg-type]
                    expected_return_pct=0.0,
                    low=0.0,
                    high=0.0,
                    prob_positive=0.5,
                    annualized_vol=0.0,
                )
            out.append(er)
        return out

    def _composite(self, signals: list[StrategySignal]) -> tuple[float, float]:
        """Compute the composite score and aggregate confidence from signals.

        See :meth:`analyze` for the formula. The aggregate confidence is the mean
        signal confidence scaled down a little when signals disagree strongly, so
        a noisy consensus reports lower confidence.

        Args:
            signals: All strategy signals for the asset.

        Returns:
            A ``(composite_score, confidence)`` tuple. ``composite_score`` is in
            ``[-100, 100]`` and ``confidence`` in ``[0, 1]``.
        """
        if not signals:
            return 0.0, 0.0
        scores = np.array([float(s.score) for s in signals], dtype=np.float64)
        confs = np.array([float(s.confidence) for s in signals], dtype=np.float64)
        confs = np.where(np.isfinite(confs) & (confs > 0.0), confs, 1e-6)
        scores = np.nan_to_num(scores, nan=0.0, posinf=100.0, neginf=-100.0)

        total_w = float(np.sum(confs))
        if total_w <= 0.0:
            base = float(np.mean(scores))
        else:
            base = float(np.sum(confs * scores) / total_w)

        disagreement = float(np.std(scores)) if scores.size > 1 else 0.0
        shrink = 1.0 / (1.0 + disagreement / 60.0)
        composite = clamp(base * shrink, -100.0, 100.0)

        mean_conf = float(np.mean(confs))
        # Penalize confidence when the spread of scores is wide.
        conf_penalty = 1.0 / (1.0 + disagreement / 100.0)
        confidence = clamp(mean_conf * conf_penalty, 0.0, 1.0)
        return composite, confidence

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
        """Run a GBM Monte Carlo for one symbol and shape the result DTO.

        Drift and volatility are estimated from the symbol's realized daily log
        returns; the spot is the latest close. The horizon maps to a number of
        trading-day steps via :data:`app.quant.returns.HORIZON_DAYS`.

        Args:
            symbol: Asset ticker (case-insensitive).
            horizon: One of :data:`~app.schemas.HORIZONS`; unknown -> ``'1Y'``.
            sims: Number of simulated paths.

        Returns:
            A populated :class:`~app.schemas.MonteCarloResult`.

        Raises:
            KeyError: If the symbol is unknown.
        """
        ctx = self.context(symbol)
        lr = returns.log_returns(ctx.closes)
        if lr.size:
            mu_daily = float(np.mean(lr))
            sigma_daily = float(np.std(lr))
        else:
            mu_daily, sigma_daily = 0.0, 1e-4
        if not math.isfinite(mu_daily):
            mu_daily = 0.0
        if not math.isfinite(sigma_daily) or sigma_daily <= 0.0:
            sigma_daily = 1e-4

        s0 = float(ctx.asset.price) if ctx.asset.price > 0 else float(ctx.seed.base_price)
        hz = horizon if horizon in HORIZONS else "1Y"
        seed = abs(hash(ctx.asset.symbol)) % (2**32)
        summary = montecarlo.montecarlo_summary(
            s0=s0,
            mu_daily=mu_daily,
            sigma_daily=sigma_daily,
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
