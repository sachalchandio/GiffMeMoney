"""Tests for the strategy registry and the analysis engine (Tests set 2).

These tests pin the contract-level guarantees of the quant pipeline:

* the registry catalogs at least 18 strategies (section 7) and every catalog id
  has a matching signal builder;
* :meth:`~app.strategies.engine.AnalysisEngine.analyze` returns a complete
  :class:`~app.schemas.AssetAnalysis` for *every* universe symbol — exactly five
  blended ``expectedReturns`` (one per horizon), at least 18 signals, a composite
  score inside ``[-100, 100]`` and a valid :data:`~app.schemas.Stance`;
* the engine never raises for any known symbol and rejects unknown ones with a
  ``KeyError``;
* :meth:`~app.strategies.engine.AnalysisEngine.recommendations` is ranked by
  composite score (descending) and honours the asset-class filter;
* :meth:`~app.strategies.engine.AnalysisEngine.strategy_ranking` works for a
  known id, is sorted descending, and rejects an unknown id.

The engine is exercised against the real :class:`SimulatedProvider`, so the tests
prove the production code path end to end (no mocks).
"""

from __future__ import annotations

import math

import pytest

from app.schemas import HORIZONS, AssetAnalysis, StrategyRanking
from app.strategies.engine import AnalysisEngine
from app.strategies.registry import (
    META_BY_ID,
    SIGNAL_BUILDERS,
    STRATEGY_META,
    build_signals,
)

# Valid discrete stances from the contract (section 4 / stance thresholds).
VALID_STANCES = {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}

# The eight strategy categories from the contract (section 4).
VALID_CATEGORIES = {
    "Valuation",
    "Factor",
    "Risk-Adjusted",
    "Technical",
    "Statistical",
    "Portfolio",
    "Fundamental",
    "Derivatives",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def engine() -> AnalysisEngine:
    """Return a fresh engine wrapping the real simulated provider."""
    return AnalysisEngine()


@pytest.fixture(scope="module")
def symbols(engine: AnalysisEngine) -> list[str]:
    """Return every symbol the engine's provider knows about."""
    return [a.symbol for a in engine._provider.list_assets()]


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


def test_registry_has_at_least_18_strategies() -> None:
    """The catalog must register at least 18 quant models (section 7)."""
    assert len(STRATEGY_META) >= 18


def test_registry_ids_are_unique() -> None:
    """Every strategy id in the catalog is unique."""
    ids = [m.id for m in STRATEGY_META]
    assert len(ids) == len(set(ids))
    assert len(META_BY_ID) == len(STRATEGY_META)


def test_every_strategy_has_a_builder() -> None:
    """Each catalog id has exactly one matching signal builder, and vice versa."""
    meta_ids = {m.id for m in STRATEGY_META}
    builder_ids = set(SIGNAL_BUILDERS.keys())
    assert meta_ids == builder_ids


def test_strategy_meta_fields_are_populated() -> None:
    """Catalog metadata fields are non-empty and categories are valid."""
    for meta in STRATEGY_META:
        assert meta.id and isinstance(meta.id, str)
        assert meta.name and isinstance(meta.name, str)
        assert meta.category in VALID_CATEGORIES
        assert meta.summary
        assert meta.formula
        assert meta.inputs  # at least one declared input
        assert meta.references  # at least one reference


def test_contract_core_models_present() -> None:
    """The headline models named in section 7 are all registered."""
    expected = {
        "capm",
        "fama-french",
        "dcf",
        "ddm",
        "markowitz",
        "sharpe",
        "sortino",
        "momentum",
        "mean-reversion",
        "macd",
        "rsi",
        "bollinger",
        "montecarlo",
        "garch",
        "black-scholes",
        "var",
        "kelly",
        "piotroski",
        "altman-z",
        "trend-ols",
    }
    assert expected.issubset(set(META_BY_ID.keys()))


# ---------------------------------------------------------------------------
# build_signals (registry-level)
# ---------------------------------------------------------------------------


def test_build_signals_one_per_strategy(engine: AnalysisEngine) -> None:
    """``build_signals`` returns exactly one signal per registered strategy."""
    ctx = engine.context("AAPL")
    signals = build_signals(ctx)
    assert len(signals) == len(STRATEGY_META)
    ids = [s.strategy_id for s in signals]
    assert ids == [m.id for m in STRATEGY_META]  # catalog order preserved


def test_build_signals_values_are_well_formed(engine: AnalysisEngine) -> None:
    """Every signal has a clamped score, confidence, valid stance and category."""
    ctx = engine.context("MSFT")
    for sig in build_signals(ctx):
        assert -100.0 <= sig.score <= 100.0
        assert math.isfinite(sig.score)
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.stance in VALID_STANCES
        assert sig.category in VALID_CATEGORIES
        assert sig.rationale
        assert sig.formula
        # Projecting models attach all five horizons; others attach none.
        assert len(sig.horizons) in (0, len(HORIZONS))
        for h in sig.horizons:
            assert h.horizon in HORIZONS


# ---------------------------------------------------------------------------
# engine.analyze — per-symbol guarantees
# ---------------------------------------------------------------------------


def test_analyze_returns_five_horizons_and_enough_signals(
    engine: AnalysisEngine,
) -> None:
    """analyze() yields exactly 5 expectedReturns and >= 18 signals."""
    analysis = engine.analyze("AAPL")
    assert isinstance(analysis, AssetAnalysis)
    assert len(analysis.expected_returns) == 5
    assert [h.horizon for h in analysis.expected_returns] == HORIZONS
    assert len(analysis.signals) >= 18


def test_analyze_composite_score_and_recommendation(engine: AnalysisEngine) -> None:
    """Composite score is in [-100, 100] and recommendation is a valid stance."""
    analysis = engine.analyze("NVDA")
    assert -100.0 <= analysis.composite_score <= 100.0
    assert math.isfinite(analysis.composite_score)
    assert analysis.recommendation in VALID_STANCES
    assert 0.0 <= analysis.confidence <= 1.0


def test_analyze_has_rationale_and_top_reasons(engine: AnalysisEngine) -> None:
    """The narrative summary is present and there are 3-5 top reasons."""
    analysis = engine.analyze("MSFT")
    assert analysis.rationale_summary
    assert 3 <= len(analysis.top_reasons) <= 5


def test_analyze_risk_metrics_are_finite(engine: AnalysisEngine) -> None:
    """Every risk metric is a finite float."""
    rm = engine.analyze("AAPL").risk_metrics
    for value in (
        rm.beta,
        rm.annual_vol,
        rm.sharpe,
        rm.sortino,
        rm.var95,
        rm.cvar95,
        rm.max_drawdown,
        rm.calmar,
    ):
        assert math.isfinite(value)


def test_analyze_expected_returns_well_formed(engine: AnalysisEngine) -> None:
    """Blended horizons have finite numbers and probabilities in [0, 1]."""
    for h in engine.analyze("SPY").expected_returns:
        assert math.isfinite(h.expected_return_pct)
        assert math.isfinite(h.low)
        assert math.isfinite(h.high)
        assert math.isfinite(h.annualized_vol)
        assert 0.0 <= h.prob_positive <= 1.0


def test_analyze_is_cached(engine: AnalysisEngine) -> None:
    """Repeated analysis of the same symbol returns the cached object."""
    first = engine.analyze("AAPL")
    second = engine.analyze("aapl")  # case-insensitive cache key
    assert first is second


def test_analyze_never_raises_for_any_symbol(
    engine: AnalysisEngine, symbols: list[str]
) -> None:
    """analyze() succeeds for every universe symbol with the full guarantees."""
    assert symbols  # the universe is non-empty
    for sym in symbols:
        analysis = engine.analyze(sym)
        assert len(analysis.expected_returns) == 5
        assert len(analysis.signals) >= 18
        assert -100.0 <= analysis.composite_score <= 100.0
        assert analysis.recommendation in VALID_STANCES


def test_analyze_unknown_symbol_raises_keyerror(engine: AnalysisEngine) -> None:
    """An unknown symbol propagates a KeyError (so the API can 404)."""
    with pytest.raises(KeyError):
        engine.analyze("ZZZZ_NOT_A_SYMBOL")


# ---------------------------------------------------------------------------
# engine.recommendations
# ---------------------------------------------------------------------------


def test_recommendations_sorted_descending(engine: AnalysisEngine) -> None:
    """Recommendations are ranked by composite score, best first."""
    recs = engine.recommendations(limit=12)
    assert recs
    scores = [r.composite_score for r in recs]
    assert scores == sorted(scores, reverse=True)
    # Ranks are 1..n in order.
    assert [r.rank for r in recs] == list(range(1, len(recs) + 1))


def test_recommendations_respect_limit(engine: AnalysisEngine) -> None:
    """The ``limit`` argument caps the number of returned recommendations."""
    recs = engine.recommendations(limit=3)
    assert len(recs) <= 3


def test_recommendations_filter_by_asset_class(engine: AnalysisEngine) -> None:
    """An asset-class filter only returns assets of that class."""
    crypto = engine.recommendations(limit=50, asset_class="crypto")
    assert crypto
    assert all(r.asset.asset_class == "crypto" for r in crypto)


def test_recommendations_fields_well_formed(engine: AnalysisEngine) -> None:
    """Recommendation rows carry valid stance and finite 1Y expected return."""
    for r in engine.recommendations(limit=5):
        assert r.recommendation in VALID_STANCES
        assert -100.0 <= r.composite_score <= 100.0
        assert math.isfinite(r.expected_return1y_pct)
        assert r.headline


# ---------------------------------------------------------------------------
# engine.strategy_ranking
# ---------------------------------------------------------------------------


def test_strategy_ranking_known_id(engine: AnalysisEngine) -> None:
    """A known strategy id yields a descending cross-asset ranking."""
    ranking = engine.strategy_ranking("capm", limit=20)
    assert isinstance(ranking, StrategyRanking)
    assert ranking.strategy_id == "capm"
    assert ranking.entries
    scores = [e.score for e in ranking.entries]
    assert scores == sorted(scores, reverse=True)
    for entry in ranking.entries:
        assert -100.0 <= entry.score <= 100.0
        assert entry.stance in VALID_STANCES


def test_strategy_ranking_respects_limit(engine: AnalysisEngine) -> None:
    """The ranking honours the ``limit`` cap."""
    ranking = engine.strategy_ranking("rsi", limit=5)
    assert len(ranking.entries) <= 5


def test_strategy_ranking_unknown_id_raises(engine: AnalysisEngine) -> None:
    """An unknown strategy id raises a KeyError."""
    with pytest.raises(KeyError):
        engine.strategy_ranking("not-a-real-strategy")


# ---------------------------------------------------------------------------
# engine.market_summary
# ---------------------------------------------------------------------------


def test_market_summary_shape(engine: AnalysisEngine) -> None:
    """market_summary() reports consistent breadth and populated sections."""
    summary = engine.market_summary()
    total_assets = len(engine._provider.list_assets())
    breadth = summary.breadth
    assert (
        breadth.advancers + breadth.decliners + breadth.unchanged == total_assets
    )
    assert summary.sectors
    assert summary.indices
    assert len(summary.top_gainers) <= 5
    assert len(summary.top_losers) <= 5
