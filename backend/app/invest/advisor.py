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
from app.strategies.feasibility import feasibility_warning

__all__ = ["AllocationAdvisor"]

# How many top-ranked candidates to size into the basket, per risk profile.
_PICKS_BY_RISK: dict[str, int] = {
    "conservative": 4,
    "balanced": 6,
    "aggressive": 8,
}

# Markowitz objective per risk profile. Conservative minimizes downside
# (historical CVaR / expected shortfall) so the risky sleeve actively avoids the
# deep-loss tail; balanced / aggressive maximize the Sharpe ratio (aggressive
# simply sizes more names so single-asset concentration is diluted across a wider
# basket). When CVaR has no usable history the optimizer falls back to
# min_volatility, so conservative is never *more* volatile than equal weight.
_OBJECTIVE_BY_RISK: dict[str, str] = {
    "conservative": "min_cvar",
    "balanced": "max_sharpe",
    "aggressive": "max_sharpe",
}

# Per-name weight cap by risk profile — no single name may dominate the risky
# sleeve. Conservative caps tightest (broadest diversification); aggressive
# allows more concentration. The optimizer floors any cap to 1/n for feasibility.
_MAX_WEIGHT_BY_RISK: dict[str, float] = {
    "conservative": 0.30,
    "balanced": 0.35,
    "aggressive": 0.45,
}

# Cash-sleeve policy. ``base`` is the maximum risky fraction a profile will ever
# deploy (the rest is parked as cash); ``floor`` is the minimum risky fraction
# after regime/conviction de-risking. Conservative caps risky exposure (<= 0.60)
# and can de-risk to 0.30; aggressive deploys nearly fully.
_RISKY_BASE_BY_RISK: dict[str, float] = {
    "conservative": 0.60,
    "balanced": 0.85,
    "aggressive": 1.00,
}
_RISKY_FLOOR_BY_RISK: dict[str, float] = {
    "conservative": 0.30,
    "balanced": 0.45,
    "aggressive": 0.60,
}

# Annual risk-free rate assumed for the Sharpe objective / blended stats.
_RISK_FREE: float = 0.04

# Tail confidence for the CVaR (expected-shortfall) objective + basket CVaR.
_CVAR_BETA: float = 0.95

# How many trailing daily closes to pull per pick when estimating mu / cov.
_HISTORY_DAYS: int = 1260

# How many candidates to consider before picking the top N. Bounded so the
# advisor analyzes a modest set once (the engine caches per symbol regardless).
_CANDIDATE_LIMIT: int = 24

# Dust thresholds: a risky leg below BOTH a tiny weight and a tiny notional is
# dropped (and the remaining risky weights renormalized) so the advice never
# emits a $0.00 / sub-cent allocation leg.
_MIN_LEG_WEIGHT: float = 1e-3
_MIN_LEG_NOTIONAL: float = 1.0

# Default sizing horizon (trading days) — a 1Y view. A shorter request scales
# mu/cov down to that horizon so e.g. a 2-month ask is sized on 2-month risk.
_DEFAULT_HORIZON_DAYS: float = 252.0


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
        target_amount: float | None = None,
        horizon_days: float | None = None,
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
            target_amount: Optional goal amount the caller hopes to reach; used
                only to surface a ``targetWarning`` when the implied per-day
                compounding is physically extreme (it never changes the basket).
            horizon_days: Optional days to reach ``target_amount`` in (pairs with
                ``target_amount`` for the feasibility check).

        Returns:
            A fully-populated :class:`~app.schemas.AllocationAdvice` (always
            carrying ``syntheticData=True`` and an optional ``targetWarning``).
            When no candidate can be analyzed/priced the items list is empty and
            the blended stats / horizons are neutral zeros.

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
        max_weight = _MAX_WEIGHT_BY_RISK[risk]

        # Sizing horizon: a 1Y view by default, but a shorter ``horizon_days``
        # request sizes mu/cov (and thus the basket) on that horizon's risk.
        size_h = (
            float(horizon_days)
            if (horizon_days is not None and math.isfinite(horizon_days) and horizon_days > 0.0)
            else _DEFAULT_HORIZON_DAYS
        )

        # Honesty guard: flag an impossible target (never alters the basket).
        target_warning = (
            feasibility_warning(amt, target_amount, horizon_days)
            if target_amount is not None and horizon_days is not None
            else None
        )

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
                cash_weight=1.0,
                cash_amount=round(amt, 2),
                synthetic_data=True,
                target_warning=target_warning,
            )

        picks = ranked[:n_picks]

        # ---- 2. Estimate horizon-scaled mu / cov / return matrix per pick. ----
        symbols = [a.asset.symbol for a in picks]
        mu, cov, returns_matrix = self._estimate_mu_cov(symbols, size_h)

        # ---- 3. Optimize the long-only, capped weights for the objective. ----
        weights = self._optimize(
            mu, cov, objective, max_weight=max_weight, returns_matrix=returns_matrix
        )

        # ---- 4. Cash sleeve: how much of ``amt`` to actually deploy. ----
        # A risky fraction in [0, 1] driven by the risk profile AND the top
        # pick's regime / conviction. The remainder is parked as cash, so the
        # advice is never blindly "100% invested".
        risky_fraction = self._risky_fraction(risk, picks, weights, mu, cov)
        risky_weights = weights * risky_fraction

        # ---- 5. Build the per-item legs (drop dust + renormalize risky). ----
        items, kept_weights = self._build_items(
            picks, risky_weights, amt, risky_fraction
        )

        # Recompute the deployed risky fraction from the *kept* legs so the cash
        # sleeve exactly reconciles with what the items actually sum to.
        deployed = float(np.sum(kept_weights))
        cash_weight = max(0.0, 1.0 - deployed)
        cash_amount = round(cash_weight * amt, 2)

        # ---- 6. Blended risky-sleeve stats + downside-carrying horizon curve. ----
        from app.quant import portfolio as pf

        # Stats describe the *risky sleeve* itself (the relative weights of the
        # deployed names), so normalize the kept weights back to sum ~1.
        if deployed > 0.0:
            norm_weights = kept_weights / deployed
        else:
            norm_weights = kept_weights
        ret, vol, sharpe = pf.portfolio_stats(norm_weights, mu, cov, _RISK_FREE)
        horizons = self._blend_horizons(picks, norm_weights)

        return AllocationAdvice(
            items=items,
            expected_return=_safe(ret),
            expected_vol=_safe(vol),
            sharpe=_safe(sharpe),
            horizons=horizons,
            risk_tolerance=risk,  # type: ignore[arg-type]
            amount=round(amt, 2),
            cash_weight=round(cash_weight, 6),
            cash_amount=cash_amount,
            synthetic_data=True,
            target_warning=target_warning,
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
        self, symbols: list[str], horizon_days: float = _DEFAULT_HORIZON_DAYS
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate horizon-scaled expected returns, covariance and a return matrix.

        Mirrors the optimizer endpoint's estimator, but scales the result to the
        requested ``horizon_days`` rather than a fixed one-year window so a short
        request (e.g. ~42 days ≈ 2 months) is sized on its own horizon's risk:

            mu_i = exp(mean(log r_i) * h) - 1       (geometric, over h days)
            S    = Cov(R_daily, ddof=0) * h         (covariance, scaled by h)

        With ``h = TRADING_DAYS`` (the default) this is exactly the previous
        annual estimator. Daily simple-return series are trailing-aligned to a
        common length so the covariance (and the returns matrix) line up the same
        dates.

        Args:
            symbols: The selected pick symbols (length ``n``).
            horizon_days: Sizing horizon in trading days (default 252 = 1Y).
                Clamped to ``[1, _HISTORY_DAYS]``.

        Returns:
            A ``(mu, cov, returns_matrix)`` tuple: ``mu`` a length-``n`` expected
            return vector over the horizon, ``cov`` an ``n x n`` covariance scaled
            to the horizon, and ``returns_matrix`` the trailing-aligned ``(m, n)``
            daily simple-return matrix (for the CVaR objective), all finite. The
            matrix is empty (``(0, n)``) when history is too short.
        """
        n = len(symbols)
        h = float(horizon_days) if math.isfinite(horizon_days) else _DEFAULT_HORIZON_DAYS
        h = max(1.0, min(float(_HISTORY_DAYS), h))
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
                exponent = max(-700.0, min(700.0, daily_mean * h))
                horizon_ret = math.exp(exponent) - 1.0
                mu[i] = horizon_ret if math.isfinite(horizon_ret) else 0.0
            simple_series.append(simple_returns(closes))

        lengths = [s.size for s in simple_series if s.size > 0]
        m = min(lengths) if lengths else 0
        if m >= 2 and n >= 1:
            matrix = np.column_stack([s[-m:] for s in simple_series])
            daily_cov = np.cov(matrix, rowvar=False, ddof=0)
            daily_cov = np.atleast_2d(np.asarray(daily_cov, dtype=np.float64))
            if daily_cov.shape != (n, n):
                daily_cov = np.zeros((n, n), dtype=np.float64)
            cov = daily_cov * h
            returns_matrix = np.nan_to_num(
                np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            cov = np.zeros((n, n), dtype=np.float64)
            returns_matrix = np.empty((0, n), dtype=np.float64)

        mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        return mu, cov, returns_matrix

    def _optimize(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        objective: str,
        max_weight: float | None = None,
        returns_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve the long-only, capped weights for the objective.

        Args:
            mu: Horizon expected-return vector (length ``n``).
            cov: Horizon covariance matrix (``n x n``).
            objective: ``'min_volatility'`` / ``'max_sharpe'`` / ``'min_cvar'``.
            max_weight: Optional per-name cap (floored to ``1/n`` by the quant
                layer for feasibility).
            returns_matrix: Optional ``(m, n)`` daily-return matrix used by the
                ``'min_cvar'`` objective (ignored otherwise).

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
            max_weight=max_weight,
            returns_matrix=returns_matrix,
            cvar_beta=_CVAR_BETA,
        )
        return np.asarray(weights, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Cash sleeve (risky fraction)
    # ------------------------------------------------------------------

    def _risky_fraction(
        self,
        risk: str,
        picks: list[AssetAnalysis],
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
    ) -> float:
        """Fraction of the request to actually deploy into risky assets, in [0, 1].

        Combines the risk profile with the prevailing regime and conviction so the
        advice de-risks into cash when the signal is weak or the regime is bearish:

            * Each profile has a ``base`` maximum risky exposure (conservative
              caps at 0.60) and a ``floor`` minimum (conservative 0.30).
            * **Conviction** — the weight-blended top composite score (in
              ``[-100, 100]``) maps to a multiplier in roughly ``[0.5, 1.0]``: a
              strong basket deploys near ``base``, a weak/negative one shrinks
              toward ``floor``.
            * **Regime** — a bear regime (weight-blended regime score < 0) trims
              risky exposure further; a bull regime nudges it up. Neutral leaves it.

        The result is clamped to ``[floor, base]`` (and ``[0, 1]``).

        Args:
            risk: The resolved risk profile key.
            picks: The selected analyses (aligned with ``weights``).
            weights: The optimized (pre-cash) risky weights, summing to ~1.
            mu: Horizon expected-return vector (unused directly; kept for symmetry).
            cov: Horizon covariance matrix (unused directly; kept for symmetry).

        Returns:
            A finite risky fraction in ``[0, 1]``.
        """
        base = _RISKY_BASE_BY_RISK.get(risk, _RISKY_BASE_BY_RISK["balanced"])
        floor = _RISKY_FLOOR_BY_RISK.get(risk, _RISKY_FLOOR_BY_RISK["balanced"])
        floor = min(floor, base)

        w = np.asarray(weights, dtype=np.float64).ravel()
        wsum = float(w.sum())

        # Weight-blended composite score (conviction) and regime score across the
        # deployed names; fall back to the top-ranked pick when weights are flat.
        if wsum > 0.0:
            comp = 0.0
            regime_score = 0.0
            regime_wsum = 0.0
            for i, analysis in enumerate(picks):
                wi = _safe(float(w[i])) if i < w.size else 0.0
                if wi <= 0.0:
                    continue
                comp += wi * _safe(analysis.composite_score)
                reg = analysis.regime
                if reg is not None:
                    regime_score += wi * _safe(reg.score)
                    regime_wsum += wi
            comp /= wsum
            regime_score = (regime_score / regime_wsum) if regime_wsum > 0.0 else 0.0
        else:
            top = picks[0] if picks else None
            comp = _safe(top.composite_score) if top is not None else 0.0
            regime_score = (
                _safe(top.regime.score) if (top is not None and top.regime is not None) else 0.0
            )

        # Conviction multiplier: composite in [-100, 100] -> ~[0.5, 1.0].
        conviction = 0.75 + 0.25 * max(-1.0, min(1.0, comp / 100.0))
        conviction = max(0.5, min(1.0, conviction))

        # Regime tilt: bearish (score < 0) trims, bullish nudges up. Bounded.
        regime_tilt = 1.0 + 0.20 * max(-1.0, min(1.0, regime_score))
        regime_tilt = max(0.7, min(1.1, regime_tilt))

        fraction = base * conviction * regime_tilt
        fraction = max(floor, min(base, fraction))
        return max(0.0, min(1.0, _safe(fraction, floor)))

    # ------------------------------------------------------------------
    # Item & horizon construction
    # ------------------------------------------------------------------

    def _build_items(
        self,
        picks: list[AssetAnalysis],
        risky_weights: np.ndarray,
        amount: float,
        risky_fraction: float,
    ) -> tuple[list[AdviceItem], np.ndarray]:
        """Turn (analysis, risky-weight) pairs into per-asset legs, dropping dust.

        ``risky_weights`` are already scaled by the risky fraction (so each entry
        is a share of the *total* request, and ``sum ~= risky_fraction``). A leg
        is dropped when it is dust — below BOTH a tiny weight (:data:`_MIN_LEG_WEIGHT`)
        and a tiny notional (:data:`_MIN_LEG_NOTIONAL`) — so the advice never emits
        a $0.00 / sub-dollar leg. The kept legs are then **renormalized** back to
        the intended ``risky_fraction`` total (dropped dust is redistributed across
        the survivors, not silently leaked into cash).

        Args:
            picks: The selected analyses (aligned with ``risky_weights`` by index).
            risky_weights: Per-pick weights already scaled by the risky fraction.
            amount: The total dollar amount being allocated.
            risky_fraction: The intended total deployed fraction (the kept legs are
                renormalized to sum to this).

        Returns:
            A ``(items, kept_weights)`` tuple. ``items`` is a list of
            :class:`~app.schemas.AdviceItem` (one per surviving pick, ordered by
            descending weight); ``kept_weights`` is the length-``len(picks)`` array
            of the renormalized share-of-total weight per pick (0 for dropped legs),
            aligned with ``picks`` by index.
        """
        w = np.asarray(risky_weights, dtype=np.float64).ravel()
        n = len(picks)
        raw = np.zeros(n, dtype=np.float64)
        for i in range(n):
            wi = _safe(float(w[i])) if i < w.size else 0.0
            raw[i] = min(1.0, max(0.0, wi))

        # Drop dust: a leg below BOTH the min weight and the min notional.
        keep = np.zeros(n, dtype=bool)
        for i in range(n):
            notional = raw[i] * amount
            if raw[i] >= _MIN_LEG_WEIGHT or notional >= _MIN_LEG_NOTIONAL:
                keep[i] = True

        kept_weights = np.where(keep, raw, 0.0)
        kept_total = float(kept_weights.sum())
        target_total = max(0.0, min(1.0, _safe(risky_fraction)))
        # Renormalize survivors back to the intended risky fraction so dropped
        # dust is redistributed (not leaked into cash). Guard a zero total.
        if kept_total > 0.0 and target_total > 0.0:
            kept_weights = kept_weights * (target_total / kept_total)

        items: list[AdviceItem] = []
        for i, analysis in enumerate(picks):
            if not keep[i]:
                continue
            weight = _safe(float(kept_weights[i]))
            dollars = round(weight * amount, 2)
            if dollars <= 0.0:
                # A rounding edge could still zero a kept leg — skip it and drop
                # its (negligible) weight so items never carry a $0.00 entry.
                kept_weights[i] = 0.0
                continue
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
        return items, kept_weights

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
        """Weight-blend the picks' 5-horizon curves into one risky-sleeve curve.

        For each of the five horizons every base field (``expectedReturnPct``,
        ``low``, ``high``, ``probPositive``, ``annualizedVol``) AND the V2
        downside fan (``bull_pct``, ``base_pct``, ``bear_pct``, ``cvar_pct``) is
        the weight-weighted mean across the picks that project that horizon. The
        four downside fields are tracked with their own running weight so a pick
        that left them ``None`` (pre-V2) is simply skipped for that field rather
        than poisoning the blend — and the basket therefore reports a finite
        ``cvarPct`` (not ``None``) whenever at least one pick supplies it.

        Always returns exactly five entries (one per :data:`~app.schemas.HORIZONS`);
        a horizon with no weighted contribution falls back to a neutral entry.

        Args:
            picks: The selected analyses (aligned with ``weights`` by index).
            weights: The (risky-sleeve-normalized) weight vector.

        Returns:
            A list of exactly five :class:`~app.schemas.ExpectedReturn`, with the
            bull/base/bear/cvar downside fan populated.
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
                # Downside fan accumulators, each with its own weight so a None
                # field on one pick does not skew the mean.
                "bull": 0.0,
                "bull_w": 0.0,
                "base": 0.0,
                "base_w": 0.0,
                "bear": 0.0,
                "bear_w": 0.0,
                "cvar": 0.0,
                "cvar_w": 0.0,
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
                # Carry the V2 downside fan, guarding None per field.
                if hr.bull_pct is not None:
                    bucket["bull"] += weight * _safe(hr.bull_pct)
                    bucket["bull_w"] += weight
                if hr.base_pct is not None:
                    bucket["base"] += weight * _safe(hr.base_pct)
                    bucket["base_w"] += weight
                if hr.bear_pct is not None:
                    bucket["bear"] += weight * _safe(hr.bear_pct)
                    bucket["bear_w"] += weight
                if hr.cvar_pct is not None:
                    bucket["cvar"] += weight * _safe(hr.cvar_pct)
                    bucket["cvar_w"] += weight

        out: list[ExpectedReturn] = []
        for h in HORIZONS:
            bucket = acc[h]
            wt = bucket["weight"]
            if wt > 0.0:
                prob = min(1.0, max(0.0, bucket["probPositive"] / wt))
                bull = bucket["bull"] / bucket["bull_w"] if bucket["bull_w"] > 0.0 else None
                base = bucket["base"] / bucket["base_w"] if bucket["base_w"] > 0.0 else None
                bear = bucket["bear"] / bucket["bear_w"] if bucket["bear_w"] > 0.0 else None
                cvar = (
                    max(0.0, bucket["cvar"] / bucket["cvar_w"])
                    if bucket["cvar_w"] > 0.0
                    else None
                )
                out.append(
                    ExpectedReturn(
                        horizon=h,  # type: ignore[arg-type]
                        expected_return_pct=_safe(bucket["expectedReturnPct"] / wt),
                        low=_safe(bucket["low"] / wt),
                        high=_safe(bucket["high"] / wt),
                        prob_positive=_safe(prob, 0.5),
                        annualized_vol=_safe(bucket["annualizedVol"] / wt),
                        bull_pct=_safe(bull) if bull is not None else None,
                        base_pct=_safe(base) if base is not None else None,
                        bear_pct=_safe(bear) if bear is not None else None,
                        cvar_pct=_safe(cvar) if cvar is not None else None,
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
