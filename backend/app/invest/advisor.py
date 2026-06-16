"""Allocation advisor: "where should I invest this money right now?".

The :class:`AllocationAdvisor` answers a single question — given a dollar
``amount`` and a ``riskTolerance`` — by combining the two existing analytical
layers of the app:

    1. **Ranking** — the :class:`~app.strategies.engine.AnalysisEngine` scores
       every (optionally class-filtered) candidate by its composite score. The
       advisor analyzes each candidate **exactly once** and ranks them.
    2. **Sizing** — the strongest ``N`` picks (``N`` by risk: conservative 4 /
       balanced 6 / aggressive 8) are sized by classical Markowitz mean-variance
       optimization (:func:`app.quant.portfolio.optimize`) using annualized
       expected returns (geometric) and an annual covariance built from each
       pick's daily price history.

The objective is risk-driven: ``min_volatility`` for a conservative profile and
``max_sharpe`` otherwise. The resulting weights become per-item dollar amounts
(``weight * amount``); each item carries its composite score, blended 1Y
expected-return percent and a short rationale. Finally the basket's blended
annual expected return / volatility / Sharpe and a weight-blended 5-horizon
``ExpectedReturn`` curve are produced.

Efficiency / anti-stall: analysis is the expensive step, so the advisor caps the
candidate set (engine recommendations are reused from the engine's per-symbol
cache) and never re-analyzes a symbol. It analyzes only the candidate universe
once, not repeatedly.

The advisor is read-only and defensive: every emitted number is finite, weights
sum to ~1, and a candidate whose history is unusable is simply dropped rather
than raising.
"""

from __future__ import annotations

import math

import numpy as np

from app.market.provider import MarketDataProvider
from app.quant.returns import TRADING_DAYS, log_returns, simple_returns
from app.schemas import (
    AdviceItem,
    AllocationAdvice,
    Asset,
    AssetAnalysis,
    AssetClass,
    ExpectedReturn,
    HORIZONS,
    RiskTolerance,
)
from app.strategies.engine import AnalysisEngine

__all__ = ["AllocationAdvisor"]

# How many top-ranked candidates to size into the basket, per risk profile.
_PICKS_BY_RISK: dict[str, int] = {
    "conservative": 4,
    "balanced": 6,
    "aggressive": 8,
}

# Markowitz objective per risk profile. Conservative minimizes volatility;
# balanced / aggressive maximize the Sharpe ratio (aggressive simply sizes more
# names so single-asset concentration is diluted across a wider basket).
_OBJECTIVE_BY_RISK: dict[str, str] = {
    "conservative": "min_volatility",
    "balanced": "max_sharpe",
    "aggressive": "max_sharpe",
}

# Annual risk-free rate assumed for the Sharpe objective / blended stats.
_RISK_FREE: float = 0.04

# How many trailing daily closes to pull per pick when estimating mu / cov.
_HISTORY_DAYS: int = 1260

# How many candidates to consider before picking the top N. Bounded so the
# advisor analyzes a modest set once (the engine caches per symbol regardless).
_CANDIDATE_LIMIT: int = 24


def _safe(value: float, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, falling back to ``default``.

    Args:
        value: Candidate number.
        default: Substitute for NaN / +-inf / non-numeric input.

    Returns:
        A finite float.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


class AllocationAdvisor:
    """Recommend how to split a dollar amount across the strongest assets.

    Args:
        engine: The shared :class:`~app.strategies.engine.AnalysisEngine` used to
            rank candidates by composite score (its per-symbol analysis cache is
            reused, so candidates are analyzed at most once).
        provider: A :class:`~app.market.provider.MarketDataProvider` supplying the
            daily price history used to estimate expected returns and covariance.
    """

    def __init__(self, engine: AnalysisEngine, provider: MarketDataProvider) -> None:
        """Store the collaborators (the advisor holds no mutable state)."""
        self._engine = engine
        self._provider = provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def advise(
        self,
        amount: float,
        risk_tolerance: RiskTolerance,
        asset_classes: list[AssetClass] | None = None,
    ) -> AllocationAdvice:
        """Build a Markowitz-sized allocation for ``amount`` at a risk profile.

        Pipeline:
            1. Validate ``amount`` (``> 0``, finite) and resolve the risk profile.
            2. Rank the candidate universe (optionally class-filtered) by composite
               score, analyzing each candidate exactly once.
            3. Take the top ``N`` picks (``N`` by risk).
            4. Estimate annualized geometric expected returns ``mu`` and annual
               covariance ``S`` from each pick's price history.
            5. Optimize weights (``min_volatility`` for conservative, else
               ``max_sharpe``) under long-only / fully-invested constraints.
            6. Emit per-item legs (``amount = weight * amount``, composite score,
               1Y expected return, rationale) plus the basket's blended annual
               expected return / volatility / Sharpe and a weight-blended
               5-horizon ``ExpectedReturn`` curve.

        Args:
            amount: Dollar amount to allocate (must be ``> 0`` and finite).
            risk_tolerance: ``'conservative'`` / ``'balanced'`` / ``'aggressive'``;
                drives the pick count and the optimizer objective.
            asset_classes: Optional list of asset classes to restrict candidates
                to; ``None`` (or empty) considers every class.

        Returns:
            A fully-populated :class:`~app.schemas.AllocationAdvice`. When no
            candidate can be analyzed/priced the items list is empty and the
            blended stats / horizons are neutral zeros.

        Raises:
            ValueError: If ``amount`` is non-positive or non-finite (HTTP 400).
        """
        try:
            amt = float(amount)
        except (TypeError, ValueError):
            raise ValueError("Amount must be a number.") from None
        if not math.isfinite(amt):
            raise ValueError("Amount must be a finite number.")
        if amt <= 0.0:
            raise ValueError("Amount must be greater than zero.")

        risk = str(risk_tolerance).strip().lower()
        if risk not in _PICKS_BY_RISK:
            risk = "balanced"
        n_picks = _PICKS_BY_RISK[risk]
        objective = _OBJECTIVE_BY_RISK[risk]

        # ---- 1. Rank candidates by composite score (analyze each once). ----
        ranked = self._ranked_analyses(asset_classes, n_picks)
        if not ranked:
            return AllocationAdvice(
                items=[],
                expected_return=0.0,
                expected_vol=0.0,
                sharpe=0.0,
                horizons=self._neutral_horizons(),
                risk_tolerance=risk,  # type: ignore[arg-type]
                amount=round(amt, 2),
            )

        picks = ranked[:n_picks]

        # ---- 2. Estimate annualized mu / cov from each pick's history. ----
        symbols = [a.asset.symbol for a in picks]
        mu, cov = self._estimate_mu_cov(symbols)

        # ---- 3. Optimize the long-only weights for the chosen objective. ----
        weights = self._optimize(mu, cov, objective)

        # ---- 4. Build the per-item legs. ----
        items = self._build_items(picks, weights, amt)

        # ---- 5. Blended basket stats + weight-blended horizon curve. ----
        from app.quant import portfolio as pf

        ret, vol, sharpe = pf.portfolio_stats(weights, mu, cov, _RISK_FREE)
        horizons = self._blend_horizons(picks, weights)

        return AllocationAdvice(
            items=items,
            expected_return=_safe(ret),
            expected_vol=_safe(vol),
            sharpe=_safe(sharpe),
            horizons=horizons,
            risk_tolerance=risk,  # type: ignore[arg-type]
            amount=round(amt, 2),
        )

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def _ranked_analyses(
        self, asset_classes: list[AssetClass] | None, n_picks: int
    ) -> list[AssetAnalysis]:
        """Rank the (optionally class-filtered) candidates by composite score.

        Each candidate symbol is analyzed at most once (the engine caches by
        symbol). The candidate set is capped at :data:`_CANDIDATE_LIMIT` so the
        advisor never sweeps an unbounded universe; we only need the top
        ``n_picks`` after sorting.

        Args:
            asset_classes: Optional list of classes to keep; ``None``/empty keeps
                all classes.
            n_picks: The number of picks the caller will ultimately take (used
                only as a lower bound on how many we must rank — we always rank
                the full bounded candidate set so the top N is correct).

        Returns:
            A list of :class:`~app.schemas.AssetAnalysis` sorted by descending
            composite score (failed/unanalyzable symbols are skipped).
        """
        wanted: set[str] | None
        if asset_classes:
            wanted = {str(c).strip().lower() for c in asset_classes}
        else:
            wanted = None

        candidates: list[str] = []
        for asset in self._provider.list_assets():
            if wanted is not None and str(asset.asset_class).lower() not in wanted:
                continue
            candidates.append(asset.symbol)
            if len(candidates) >= _CANDIDATE_LIMIT:
                break

        analyses: list[AssetAnalysis] = []
        for symbol in candidates:
            try:
                analyses.append(self._engine.analyze(symbol))
            except Exception:
                # A single bad/unknown symbol must never sink the whole advice.
                continue

        analyses.sort(key=lambda a: _safe(a.composite_score), reverse=True)
        return analyses

    # ------------------------------------------------------------------
    # Estimation & optimization
    # ------------------------------------------------------------------

    def _estimate_mu_cov(
        self, symbols: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate annualized expected returns and covariance for the picks.

        Mirrors the optimizer endpoint's estimator:

            mu_i = exp(mean(log r_i) * TRADING_DAYS) - 1   (annual geometric)
            S    = Cov(R_daily, ddof=0) * TRADING_DAYS     (annual covariance)

        Daily simple-return series are trailing-aligned to a common length so the
        covariance lines up the same dates.

        Args:
            symbols: The selected pick symbols (length ``n``).

        Returns:
            A ``(mu, cov)`` tuple: ``mu`` a length-``n`` annual expected-return
            vector and ``cov`` an ``n x n`` annual covariance matrix, both finite
            (NaN/inf scrubbed). ``cov`` defaults to zeros when history is too
            short (the optimizer ridge keeps it well posed).
        """
        n = len(symbols)
        mu = np.zeros(n, dtype=np.float64)
        simple_series: list[np.ndarray] = []

        for i, sym in enumerate(symbols):
            try:
                closes = np.asarray(
                    self._provider.history(sym, days=_HISTORY_DAYS),
                    dtype=np.float64,
                ).ravel()
            except Exception:
                closes = np.empty(0, dtype=np.float64)
            lr = log_returns(closes)
            if lr.size:
                daily_mean = float(np.mean(lr))
                exponent = max(-700.0, min(700.0, daily_mean * TRADING_DAYS))
                ann = math.exp(exponent) - 1.0
                mu[i] = ann if math.isfinite(ann) else 0.0
            simple_series.append(simple_returns(closes))

        lengths = [s.size for s in simple_series if s.size > 0]
        m = min(lengths) if lengths else 0
        if m >= 2 and n >= 1:
            matrix = np.column_stack([s[-m:] for s in simple_series])
            daily_cov = np.cov(matrix, rowvar=False, ddof=0)
            daily_cov = np.atleast_2d(np.asarray(daily_cov, dtype=np.float64))
            if daily_cov.shape != (n, n):
                daily_cov = np.zeros((n, n), dtype=np.float64)
            cov = daily_cov * float(TRADING_DAYS)
        else:
            cov = np.zeros((n, n), dtype=np.float64)

        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        return mu, cov

    def _optimize(
        self, mu: np.ndarray, cov: np.ndarray, objective: str
    ) -> np.ndarray:
        """Solve the long-only mean-variance weights for the objective.

        Args:
            mu: Annual expected-return vector (length ``n``).
            cov: Annual covariance matrix (``n x n``).
            objective: ``'min_volatility'`` or ``'max_sharpe'``.

        Returns:
            A long-only ``float64`` weight vector of length ``n`` summing to ~1
            (equal-weight fallback on optimizer failure, per the quant layer).
        """
        from app.quant import portfolio as pf

        weights = pf.optimize(
            mu_annual=mu,
            cov_annual=cov,
            rf=_RISK_FREE,
            objective=objective,
        )
        return np.asarray(weights, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Item & horizon construction
    # ------------------------------------------------------------------

    def _build_items(
        self,
        picks: list[AssetAnalysis],
        weights: np.ndarray,
        amount: float,
    ) -> list[AdviceItem]:
        """Turn (analysis, weight) pairs into per-asset allocation legs.

        Args:
            picks: The selected analyses (aligned with ``weights`` by index).
            weights: The optimized weight vector.
            amount: The total dollar amount being allocated.

        Returns:
            A list of :class:`~app.schemas.AdviceItem`, one per pick, ordered by
            descending weight (largest allocation first).
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
        items: list[AdviceItem] = []
        for i, analysis in enumerate(picks):
            weight = _safe(float(w[i])) if i < w.size else 0.0
            weight = min(1.0, max(0.0, weight))
            dollars = round(weight * amount, 2)
            items.append(
                AdviceItem(
                    asset=analysis.asset,
                    weight=round(weight, 6),
                    amount=dollars,
                    composite_score=_safe(analysis.composite_score),
                    expected_return1y_pct=self._one_year_pct(analysis),
                    rationale=self._rationale(analysis),
                )
            )
        items.sort(key=lambda it: it.weight, reverse=True)
        return items

    def _one_year_pct(self, analysis: AssetAnalysis) -> float:
        """Return the asset's blended 1Y expected-return percent.

        Args:
            analysis: A completed asset analysis.

        Returns:
            The ``'1Y'`` horizon's ``expectedReturnPct`` as a finite float, or
            ``0.0`` when the asset has no 1Y projection.
        """
        one_year = next(
            (h for h in analysis.expected_returns if h.horizon == "1Y"), None
        )
        return _safe(one_year.expected_return_pct) if one_year else 0.0

    def _rationale(self, analysis: AssetAnalysis) -> str:
        """Build a short plain-English reason for a pick.

        Uses the engine's strongest top reason when present, otherwise a compact
        score-based fallback so the field is always meaningful.

        Args:
            analysis: A completed asset analysis.

        Returns:
            A short rationale string.
        """
        if analysis.top_reasons:
            return analysis.top_reasons[0]
        score = _safe(analysis.composite_score)
        return (
            f"{analysis.asset.symbol} scores {score:+.0f}/100 "
            f"({analysis.recommendation.replace('_', ' ').title()}) across the model suite."
        )

    def _blend_horizons(
        self, picks: list[AssetAnalysis], weights: np.ndarray
    ) -> list[ExpectedReturn]:
        """Weight-blend the picks' 5-horizon curves into one basket curve.

        For each of the five horizons every field (``expectedReturnPct``,
        ``low``, ``high``, ``probPositive``, ``annualizedVol``) is the
        weight-weighted mean across the picks that project that horizon. Always
        returns exactly five entries (one per :data:`~app.schemas.HORIZONS`); a
        horizon with no weighted contribution falls back to a neutral entry.

        Args:
            picks: The selected analyses (aligned with ``weights`` by index).
            weights: The optimized weight vector.

        Returns:
            A list of exactly five :class:`~app.schemas.ExpectedReturn`.
        """
        w = np.asarray(weights, dtype=np.float64).ravel()
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

        for i, analysis in enumerate(picks):
            weight = _safe(float(w[i])) if i < w.size else 0.0
            if weight <= 0.0:
                continue
            for hr in analysis.expected_returns:
                bucket = acc.get(hr.horizon)
                if bucket is None:
                    continue
                bucket["expectedReturnPct"] += weight * _safe(hr.expected_return_pct)
                bucket["low"] += weight * _safe(hr.low)
                bucket["high"] += weight * _safe(hr.high)
                bucket["probPositive"] += weight * _safe(hr.prob_positive)
                bucket["annualizedVol"] += weight * _safe(hr.annualized_vol)
                bucket["weight"] += weight

        out: list[ExpectedReturn] = []
        for h in HORIZONS:
            bucket = acc[h]
            wt = bucket["weight"]
            if wt > 0.0:
                prob = min(1.0, max(0.0, bucket["probPositive"] / wt))
                out.append(
                    ExpectedReturn(
                        horizon=h,  # type: ignore[arg-type]
                        expected_return_pct=_safe(bucket["expectedReturnPct"] / wt),
                        low=_safe(bucket["low"] / wt),
                        high=_safe(bucket["high"] / wt),
                        prob_positive=_safe(prob, 0.5),
                        annualized_vol=_safe(bucket["annualizedVol"] / wt),
                    )
                )
            else:
                out.append(
                    ExpectedReturn(
                        horizon=h,  # type: ignore[arg-type]
                        expected_return_pct=0.0,
                        low=0.0,
                        high=0.0,
                        prob_positive=0.5,
                        annualized_vol=0.0,
                    )
                )
        return out

    @staticmethod
    def _neutral_horizons() -> list[ExpectedReturn]:
        """Return a neutral zero-return 5-horizon curve for an empty basket.

        Returns:
            Exactly five :class:`~app.schemas.ExpectedReturn`, all zero return /
            zero band with a 0.5 probability of being positive.
        """
        return [
            ExpectedReturn(
                horizon=h,  # type: ignore[arg-type]
                expected_return_pct=0.0,
                low=0.0,
                high=0.0,
                prob_positive=0.5,
                annualized_vol=0.0,
            )
            for h in HORIZONS
        ]
