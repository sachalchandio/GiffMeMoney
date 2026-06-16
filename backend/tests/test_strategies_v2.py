"""Tests for the V2 strategy expansion (registry growth + cross-sectional + calendar).

These pin the additive STRATEGIES-V2 guarantees on top of the existing suite:

* the registry now catalogs **>= 70** strategies, every id has a matching builder,
  a populated :class:`~app.schemas.StrategyMeta`, and at least one source
  (``references``);
* :meth:`~app.strategies.engine.AnalysisEngine.analyze` still returns exactly five
  blended ``expectedReturns`` and now **>= 70** signals, and never raises for any
  universe symbol;
* cross-sectional / factor strategies (magic-formula, cross-sectional-momentum,
  low-vol-anomaly, betting-against-beta, 52-week-high, …) produce *different*
  scores across different assets (they read ``ctx.universe``);
* the calendar ``seasonality`` strategy is **deterministic given an injected
  month** (Nov-Apr favourable, May-Oct weak) and independent of the system clock.

The tests run against the real :class:`~app.market.provider.SimulatedProvider`
but deliberately use small symbol subsets (no full-universe sweeps and no
all-strategy backtests) so they stay fast.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest

from app.schemas import HORIZONS, AssetAnalysis
from app.strategies.engine import AnalysisEngine
from app.strategies.registry import (
    META_BY_ID,
    POSITION_FUNCS,
    SIGNAL_BUILDERS,
    STRATEGY_META,
)

VALID_STANCES = {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}

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

# A small, stable subset spanning all three asset classes for the fast checks.
_SUBSET = ["AAPL", "MSFT", "JPM", "BTC", "ETH", "SPY", "GLD"]


@pytest.fixture(scope="module")
def engine() -> AnalysisEngine:
    """Return a fresh engine wrapping the real simulated provider."""
    return AnalysisEngine()


@pytest.fixture(scope="module")
def all_symbols(engine: AnalysisEngine) -> list[str]:
    """Return every universe symbol (for the never-raises sweep)."""
    return [a.symbol for a in engine._provider.list_assets()]


# ---------------------------------------------------------------------------
# Registry growth + completeness
# ---------------------------------------------------------------------------


def test_registry_has_at_least_70_strategies() -> None:
    """The V2 expansion grows the registry to at least 70 strategies."""
    assert len(STRATEGY_META) >= 70
    assert len(SIGNAL_BUILDERS) >= 70


def test_every_id_has_builder_meta_and_sources() -> None:
    """Every catalog id has a builder, a populated meta, and >= 1 source."""
    meta_ids = {m.id for m in STRATEGY_META}
    assert meta_ids == set(SIGNAL_BUILDERS.keys())
    assert len(META_BY_ID) == len(STRATEGY_META)  # ids are unique
    for meta in STRATEGY_META:
        assert meta.id and isinstance(meta.id, str)
        assert meta.name
        assert meta.category in VALID_CATEGORIES
        assert meta.summary
        assert meta.formula
        assert meta.inputs
        assert meta.references  # the carried-through research sources
        assert meta.id in SIGNAL_BUILDERS


def test_v2_strategies_are_registered() -> None:
    """A representative spread of the new V2 ids is present in the registry."""
    expected = {
        "magic-formula",
        "graham-defensive",
        "qmj-quality-minus-junk",
        "gross-profitability",
        "cross-sectional-momentum",
        "tsmom",
        "52w-high",
        "dual-momentum",
        "golden-cross",
        "supertrend",
        "ichimoku",
        "connors-rsi2",
        "bollinger-squeeze",
        "williams-r",
        "cci-reversion",
        "low-vol-anomaly",
        "betting-against-beta",
        "seasonality",
        "dogs-of-dow",
        "shareholder-yield",
    }
    assert expected.issubset(set(META_BY_ID.keys()))


def test_position_funcs_registered_for_timing_strategies() -> None:
    """Timing strategies expose vectorized position functions for backtesting."""
    assert len(POSITION_FUNCS) >= 10
    for sid in ("golden-cross", "supertrend", "donchian-turtle", "tsmom"):
        assert sid in POSITION_FUNCS
        assert callable(POSITION_FUNCS[sid])


# ---------------------------------------------------------------------------
# analyze() — still 5 horizons, now >= 70 signals
# ---------------------------------------------------------------------------


def test_analyze_returns_five_horizons_and_seventy_signals(
    engine: AnalysisEngine,
) -> None:
    """analyze() yields exactly 5 expectedReturns and now >= 70 signals."""
    analysis = engine.analyze("AAPL")
    assert isinstance(analysis, AssetAnalysis)
    assert len(analysis.expected_returns) == 5
    assert [h.horizon for h in analysis.expected_returns] == HORIZONS
    assert len(analysis.signals) >= 70
    assert analysis.strategy_count == len(analysis.signals)


def test_analyze_never_raises_across_universe(
    engine: AnalysisEngine, all_symbols: list[str]
) -> None:
    """analyze() succeeds for every universe symbol with the full guarantees."""
    assert all_symbols
    for sym in all_symbols:
        analysis = engine.analyze(sym)
        assert len(analysis.expected_returns) == 5
        assert len(analysis.signals) >= 70
        assert -100.0 <= analysis.composite_score <= 100.0
        assert analysis.recommendation in VALID_STANCES


def test_every_signal_well_formed(engine: AnalysisEngine) -> None:
    """Every signal across the subset is clamped, finite and valid."""
    for sym in _SUBSET:
        for sig in engine.analyze(sym).signals:
            assert -100.0 <= sig.score <= 100.0
            assert math.isfinite(sig.score)
            assert 0.0 <= sig.confidence <= 1.0
            assert sig.stance in VALID_STANCES
            assert sig.category in VALID_CATEGORIES
            assert sig.rationale
            # Projecting models attach all 5 horizons; others attach none.
            assert len(sig.horizons) in (0, len(HORIZONS))


# ---------------------------------------------------------------------------
# Cross-sectional strategies differ across assets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "strategy_id",
    [
        "cross-sectional-momentum",
        "magic-formula",
        "low-vol-anomaly",
        "betting-against-beta",
        "52w-high",
    ],
)
def test_cross_sectional_scores_differ_across_assets(
    engine: AnalysisEngine, strategy_id: str
) -> None:
    """A cross-sectional strategy must score different assets differently.

    These strategies rank a symbol within ``ctx.universe``; if they returned the
    same score for every asset they would not be reading the cross-section. We
    build the contexts directly (one builder call per asset) to avoid running the
    full per-asset analysis.
    """
    builder = SIGNAL_BUILDERS[strategy_id]
    scores = []
    for sym in _SUBSET:
        ctx = engine.context(sym)
        scores.append(round(float(builder(ctx).score), 6))
    # At least two distinct score values across the subset.
    assert len(set(scores)) >= 2


# ---------------------------------------------------------------------------
# Seasonality is deterministic given an injected month
# ---------------------------------------------------------------------------


def test_seasonality_deterministic_given_injected_month(
    engine: AnalysisEngine,
) -> None:
    """Seasonality follows the injected month: Nov-Apr bullish, May-Oct bearish."""
    builder = SIGNAL_BUILDERS["seasonality"]

    nov = builder(engine.context("AAPL", now=dt.datetime(2025, 11, 15)))
    jul = builder(engine.context("AAPL", now=dt.datetime(2025, 7, 15)))
    # November (Halloween window) is favourable; July (Sell-in-May) is weak.
    assert nov.score > 0.0
    assert jul.score < 0.0
    assert nov.score > jul.score


def test_seasonality_repeatable_for_same_month(engine: AnalysisEngine) -> None:
    """The same injected month yields an identical seasonality score (no clock)."""
    builder = SIGNAL_BUILDERS["seasonality"]
    when = dt.datetime(2025, 12, 1)
    first = builder(engine.context("MSFT", now=when)).score
    second = builder(engine.context("MSFT", now=when)).score
    assert first == pytest.approx(second)


def test_seasonality_neutral_for_crypto(engine: AnalysisEngine) -> None:
    """Seasonality is an equity effect: crypto is neutral regardless of month."""
    builder = SIGNAL_BUILDERS["seasonality"]
    sig = builder(engine.context("BTC", now=dt.datetime(2025, 11, 15)))
    assert sig.score == pytest.approx(0.0)


def test_analyze_with_injected_now_is_not_cached(engine: AnalysisEngine) -> None:
    """Injecting ``now`` bypasses the shared cache so test months never leak."""
    a = engine.analyze("AAPL", now=dt.datetime(2025, 11, 15))
    b = engine.analyze("AAPL", now=dt.datetime(2025, 7, 15))
    # Different objects (uncached) — the November vs July seasonality differs.
    assert a is not b
