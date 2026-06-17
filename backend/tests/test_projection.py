"""Tests for the financially-credible projection engine + the R1-R8 re-audit gate.

Two layers:

1. **Fast unit tests** on :func:`app.quant.projection.project` /
   :func:`~app.quant.projection.detect_regime` over small synthetic series.
   These pin the structural §4 / §9 guarantees that hold for *any* asset:

   * :func:`project` returns exactly the 5 contract horizons (in order);
   * ``bull >= base >= bear`` at every horizon (an ordered scenario fan);
   * the confidence band widens with the horizon (cumulative uncertainty);
   * ``prob_positive`` is a probability in ``[0, 1]``;
   * ``cvar`` (a downside loss %) never exceeds the base case;
   * :func:`detect_regime` returns a valid label and finite, bounded numbers;
   * nothing emits NaN/inf, and the engine never raises on degenerate input.

2. **The full-universe REGRESSION AUDIT** (``test_reaudit_*``) — a single, real
   24-asset :meth:`~app.strategies.engine.AnalysisEngine.analyze` pass (built
   once, module-scoped, ~30-40s — acceptable ONCE) that asserts the mandatory
   financial-rigor fixes from ``docs/STRATEGIES-V2.md`` §0 actually landed and
   that the live-audit pathologies are gone:

   * **R1** — no asset projects an implausible 5Y *expected* (base) return:
     equities/ETF ≤ ~250%, crypto ≤ ~400%. (Kills the +380%/+823%/+457% medians.)
   * **R2** — every horizon ``high`` ≤ its credible cap and ``low`` ≥ −95; all
     finite. (Kills the +26,000% upper bands.)
   * **R3** — for a spread of assets, the analysis 1Y expected return and the
     Monte-Carlo 1Y expected return agree within ~1.5pp (one engine).
   * **R4** — confidence spans a > 0.3 range across the 24 assets (no more flat
     ~0.3 on everything).
   * **R5** — a realistic stance MIX: ≥ 5 BUY-or-better and ≥ 3 SELL-or-worse
     (not 24×HOLD).
   * **R6** — every analysis surfaces honest downside: finite 1Y CVaR, max
     drawdown, prob-of-loss, and a bear scenario strictly below the base case.
   * **R8** — zero NaN/inf anywhere in any analysis, and the disclaimer is present.

The audit asset list is fixed and exercises every class (equity / crypto / ETF).
The single analyze pass is shared across the audit tests via a module-scoped
fixture so the universe is swept exactly once.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.quant import projection
from app.quant.returns import HORIZON_DAYS, TRADING_DAYS
from app.schemas import HORIZONS, AssetAnalysis
from app.strategies.engine import AnalysisEngine

# ---------------------------------------------------------------------------
# Credible-cap anchors (mirrors projection._HIGH_CAP_BY_HORIZON / the §0 R1
# caps). The engine caps the annualized *drift*, not the 5Y total directly, so
# the realized 5Y base ceiling is the compounded drift cap; we assert against a
# small headroom above it so the test pins "credible", not the exact internal.
# ---------------------------------------------------------------------------

# §0 R1 annualized drift caps: equities/ETF +35%, crypto +60%. Compounded over
# 5 years that is ~349% (equity) and ~948% (crypto) at the *upper drift bound*;
# the §0 prose anchors the believable 5Y *expected* band well below that, so we
# assert the audit ceilings the prompt names (equities/ETF ≤ ~250%, crypto ≤
# ~400%). These are deliberately generous yet still kill the live-audit medians.
_R1_5Y_CAP_EQUITY_PCT = 250.0
_R1_5Y_CAP_CRYPTO_PCT = 400.0

# §0 R2 per-horizon high caps (percent) — the displayed 95th-pct ceilings.
_R2_HIGH_CAP_PCT = {
    "1D": 15.0,
    "1W": 35.0,
    "1M": 55.0,
    "1Y": 60.0,
    "5Y": 400.0,
}
_R2_LOW_FLOOR_PCT = -95.0

# The full fixed audit universe spanning every asset class (the seed universe is
# exactly 24: 14 equities, 6 crypto, 4 ETFs).
_AUDIT_SYMBOLS = [
    # equities (14)
    "AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "BAC", "V", "JNJ",
    "PFE", "XOM", "CVX", "AMZN", "KO", "CAT",
    # crypto (6)
    "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE",
    # etf (4)
    "SPY", "QQQ", "VTI", "GLD",
]

# A small subset for the R3 analysis-vs-MC consistency check (one per class).
_R3_SYMBOLS = ["AAPL", "BTC", "SPY"]

# Stance thresholds (mirrors app.strategies.base.stance_from_score):
#   BUY-or-better  := composite >= 20  (BUY / STRONG_BUY)
#   SELL-or-worse  := composite <= -20 (SELL / STRONG_SELL)
_BUY_OR_BETTER = {"BUY", "STRONG_BUY"}
_SELL_OR_WORSE = {"SELL", "STRONG_SELL"}


def _all_finite(*values: float) -> bool:
    """Return ``True`` iff every argument is a finite float (no NaN/inf)."""
    return all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values)


# ---------------------------------------------------------------------------
# Synthetic series for the fast unit tests
# ---------------------------------------------------------------------------


@pytest.fixture
def trending_series() -> tuple[np.ndarray, np.ndarray]:
    """A mildly-upward GBM-style price path + its daily returns (~400 days).

    Deterministic (seeded) so the projection numbers are reproducible. ~400
    observations is enough to exercise the block-bootstrap band path
    (``>= _MIN_BOOTSTRAP_OBS``).
    """
    rng = np.random.default_rng(7)
    n = 400
    mu, sigma = 0.0004, 0.012
    rets = rng.normal(mu, sigma, size=n)
    closes = 100.0 * np.cumprod(1.0 + rets)
    closes = np.concatenate(([100.0], closes))
    return closes.astype(np.float64), rets.astype(np.float64)


# ---------------------------------------------------------------------------
# project(): structural guarantees (§4 / §9)
# ---------------------------------------------------------------------------


def test_project_returns_five_horizons_in_order(
    trending_series: tuple[np.ndarray, np.ndarray],
) -> None:
    """project() yields exactly the 5 contract horizons in canonical order."""
    closes, rets = trending_series
    projs, regime = projection.project(
        closes=closes, returns=rets, signal_drifts=[], rf_daily=0.0001,
        beta=1.0, asset_class="equity",
    )
    assert [p.horizon for p in projs] == HORIZONS
    assert len(projs) == len(HORIZONS)
    assert isinstance(regime, dict)


def test_project_scenario_fan_is_ordered(
    trending_series: tuple[np.ndarray, np.ndarray],
) -> None:
    """bull >= base >= bear at every horizon (an ordered scenario fan)."""
    closes, rets = trending_series
    projs, _ = projection.project(
        closes=closes, returns=rets, signal_drifts=[], rf_daily=0.0001,
        beta=1.0, asset_class="equity",
    )
    for p in projs:
        assert p.bull_pct >= p.base_pct - 1e-9, p.horizon
        assert p.base_pct >= p.bear_pct - 1e-9, p.horizon
        # base_pct is the displayed expected return.
        assert p.base_pct == pytest.approx(p.expected_return_pct, abs=1e-9)


def test_project_confidence_band_widens_with_horizon(
    trending_series: tuple[np.ndarray, np.ndarray],
) -> None:
    """The 5/95 confidence band widens monotonically as the horizon lengthens."""
    closes, rets = trending_series
    projs, _ = projection.project(
        closes=closes, returns=rets, signal_drifts=[], rf_daily=0.0001,
        beta=1.0, asset_class="equity",
    )
    widths = [p.high - p.low for p in projs]
    for shorter, longer in zip(widths, widths[1:]):
        # Allow a hair of slack only for the 5Y cap saturation; otherwise strict.
        assert longer >= shorter - 1e-6


def test_project_prob_positive_in_unit_interval(
    trending_series: tuple[np.ndarray, np.ndarray],
) -> None:
    """prob_positive is a genuine probability in [0, 1] at every horizon."""
    closes, rets = trending_series
    projs, _ = projection.project(
        closes=closes, returns=rets, signal_drifts=[], rf_daily=0.0001,
        beta=1.0, asset_class="equity",
    )
    for p in projs:
        assert 0.0 <= p.prob_positive <= 1.0


def test_project_cvar_does_not_exceed_base(
    trending_series: tuple[np.ndarray, np.ndarray],
) -> None:
    """CVaR (a downside loss %) is never larger than the base-case return.

    ``cvar_pct`` is a *positive loss* percentage (expected shortfall); the base
    case is a signed total return. A credible projection's expected shortfall
    must be a worse outcome than the central case, i.e. ``-cvar <= base`` ⇔
    ``cvar <= base`` only fails when base itself is below the tail — which never
    happens because the tail mean is by construction the worst 5%. We assert the
    economically-meaningful form: the tail loss outcome (-cvar) is <= base.
    """
    closes, rets = trending_series
    projs, _ = projection.project(
        closes=closes, returns=rets, signal_drifts=[], rf_daily=0.0001,
        beta=1.0, asset_class="equity",
    )
    for p in projs:
        assert p.cvar_pct >= 0.0, p.horizon
        # The downside (tail) outcome must be no better than the base case.
        assert -p.cvar_pct <= p.base_pct + 1e-9, p.horizon


def test_project_all_values_finite(
    trending_series: tuple[np.ndarray, np.ndarray],
) -> None:
    """No projection field is ever NaN/inf (R8 at the engine-primitive level)."""
    closes, rets = trending_series
    projs, regime = projection.project(
        closes=closes, returns=rets, signal_drifts=[], rf_daily=0.0001,
        beta=1.0, asset_class="crypto",
    )
    for p in projs:
        assert _all_finite(
            p.expected_return_pct, p.low, p.high, p.prob_positive,
            p.annualized_vol, p.bull_pct, p.base_pct, p.bear_pct, p.cvar_pct,
        ), p.horizon
        assert _all_finite(p.cagr_pct())
    assert _all_finite(regime.get("trend", 0.0), regime.get("score", 0.0))


def test_project_caps_drift_even_with_wild_signals() -> None:
    """A single absurd bullish strategy cannot push the 5Y base above the cap (R1).

    Feeds an implausibly large daily drift at full confidence; the James-Stein
    shrinkage + asset-class cap must still pin the 5Y *expected* return inside the
    credible band (this is the core of the +823% fix).
    """
    rng = np.random.default_rng(3)
    rets = rng.normal(0.0003, 0.02, size=400)
    closes = 100.0 * np.cumprod(1.0 + rets)
    # 0.02 daily log-drift ~ +154%/yr if uncapped — wildly optimistic.
    wild = [(0.02, 1.0)] * 5
    projs, _ = projection.project(
        closes=closes.astype(np.float64), returns=rets.astype(np.float64),
        signal_drifts=wild, rf_daily=0.0001, beta=1.2, asset_class="crypto",
    )
    five_y = next(p for p in projs if p.horizon == "5Y")
    assert five_y.base_pct <= _R1_5Y_CAP_CRYPTO_PCT
    assert five_y.high <= _R2_HIGH_CAP_PCT["5Y"] + 1e-6


def test_project_degenerate_input_never_raises() -> None:
    """Empty / constant / tiny inputs collapse to safe finite values, never raise."""
    for closes, rets in (
        (np.array([], dtype=np.float64), np.array([], dtype=np.float64)),
        (np.array([100.0], dtype=np.float64), np.array([], dtype=np.float64)),
        (np.full(10, 50.0), np.zeros(9)),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.5])),
    ):
        projs, regime = projection.project(
            closes=closes, returns=rets, signal_drifts=[], rf_daily=0.0,
            asset_class="equity",
        )
        assert len(projs) == len(HORIZONS)
        for p in projs:
            assert _all_finite(
                p.expected_return_pct, p.low, p.high, p.prob_positive,
                p.bull_pct, p.bear_pct, p.cvar_pct,
            )
        assert regime["regime"] in ("bull", "bear", "neutral")


# ---------------------------------------------------------------------------
# detect_regime(): valid label + bounded, finite numbers
# ---------------------------------------------------------------------------


def test_detect_regime_returns_valid_label() -> None:
    """detect_regime() labels are valid and trend/score are finite & bounded."""
    rng = np.random.default_rng(11)
    up = 100.0 * np.cumprod(1.0 + rng.normal(0.0015, 0.01, size=300))
    down = 100.0 * np.cumprod(1.0 + rng.normal(-0.0015, 0.01, size=300))
    for closes in (up, down, np.full(300, 100.0), np.array([100.0, 101.0])):
        reg = projection.detect_regime(closes.astype(np.float64))
        assert reg["regime"] in ("bull", "bear", "neutral")
        assert reg["vol_regime"] in ("low", "normal", "high")
        assert _all_finite(reg["trend"], reg["score"])
        assert -1.0 <= reg["trend"] <= 1.0
        assert -1.0 <= reg["score"] <= 1.0


def test_detect_regime_uptrend_is_bullish() -> None:
    """A clear, calm uptrend classifies as a bull regime with a positive score."""
    rng = np.random.default_rng(21)
    up = 100.0 * np.cumprod(1.0 + rng.normal(0.002, 0.008, size=300))
    reg = projection.detect_regime(up.astype(np.float64))
    assert reg["regime"] == "bull"
    assert reg["score"] > 0.0


# ---------------------------------------------------------------------------
# THE FULL-UNIVERSE RE-AUDIT GATE (R1-R8) — one analyze pass, module-scoped
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def audit_engine() -> AnalysisEngine:
    """A fresh engine over the real simulated provider (shared by the audit)."""
    return AnalysisEngine()


@pytest.fixture(scope="module")
def audit(audit_engine: AnalysisEngine) -> dict[str, AssetAnalysis]:
    """Run ONE 24-asset analyze pass and return ``symbol -> AssetAnalysis``.

    Module-scoped so the universe is swept exactly once; every R1-R8 audit test
    reads from this single shared map (~30-40s for the cold pass).
    """
    out: dict[str, AssetAnalysis] = {}
    for sym in _AUDIT_SYMBOLS:
        out[sym] = audit_engine.analyze(sym)
    return out


def _horizon(a: AssetAnalysis, label: str):
    """Return the analysis's ExpectedReturn for ``label`` (or ``None``)."""
    return next((h for h in a.expected_returns if h.horizon == label), None)


def _asset_class(a: AssetAnalysis) -> str:
    """Return the lower-cased asset class for an analysis."""
    return str(a.asset.asset_class).lower()


def test_audit_pass_covers_full_universe(
    audit: dict[str, AssetAnalysis], audit_engine: AnalysisEngine,
) -> None:
    """Sanity: the audit pass analyzed all 24 fixed symbols with 5 horizons each."""
    assert len(audit) == len(_AUDIT_SYMBOLS) == 24
    for sym, a in audit.items():
        assert [h.horizon for h in a.expected_returns] == HORIZONS
        assert len(a.signals) >= 70


def test_reaudit_R1_no_implausible_5y_expected(
    audit: dict[str, AssetAnalysis],
) -> None:
    """R1: every asset's 5Y expected (base) return is within the credible cap.

    Equities/ETF ≤ ~250%, crypto ≤ ~400%. NONE may exceed it — this is the kill
    test for the live-audit +380%/+823%/+457% 5Y "expected" returns.
    """
    offenders: list[str] = []
    for sym, a in audit.items():
        five_y = _horizon(a, "5Y")
        assert five_y is not None and math.isfinite(five_y.expected_return_pct)
        cap = (
            _R1_5Y_CAP_CRYPTO_PCT
            if _asset_class(a) == "crypto"
            else _R1_5Y_CAP_EQUITY_PCT
        )
        if five_y.expected_return_pct > cap:
            offenders.append(
                f"{sym} ({_asset_class(a)}): 5Y={five_y.expected_return_pct:.1f}% > {cap:.0f}%"
            )
    assert not offenders, "R1 implausible 5Y expected returns: " + "; ".join(offenders)


def test_reaudit_R2_believable_bands(audit: dict[str, AssetAnalysis]) -> None:
    """R2: every horizon high <= its cap, low >= -95, and all band values finite.

    Kills the +26,000% upper bands from the live audit.
    """
    offenders: list[str] = []
    for sym, a in audit.items():
        for h in a.expected_returns:
            cap = _R2_HIGH_CAP_PCT[h.horizon]
            if not _all_finite(h.low, h.high, h.expected_return_pct):
                offenders.append(f"{sym} {h.horizon}: non-finite band")
                continue
            if h.high > cap + 1e-6:
                offenders.append(f"{sym} {h.horizon}: high={h.high:.1f}% > {cap:.0f}%")
            if h.low < _R2_LOW_FLOOR_PCT - 1e-6:
                offenders.append(f"{sym} {h.horizon}: low={h.low:.1f}% < {_R2_LOW_FLOOR_PCT}%")
    assert not offenders, "R2 implausible bands: " + "; ".join(offenders)


# Sims used for the R3 consistency check. The analysis 1Y uses the analytic GBM
# mean (``exp(mu*252)-1``) while ``montecarlo`` *estimates* the same expectation
# by sampling; for the same drift+vol the only difference is Monte-Carlo standard
# error, which for high-vol crypto is multiple points at the default 2000 sims.
# 50k sims drives that estimation error well below 1pp for every asset, so the
# check measures genuine engine *consistency* (same drift+vol → same expected
# return — R3's design goal) rather than MC sampling noise.
_R3_SIMS = 50_000


def test_reaudit_R3_analysis_matches_montecarlo_1y(
    audit: dict[str, AssetAnalysis], audit_engine: AnalysisEngine,
) -> None:
    """R3: analysis 1Y and Monte-Carlo 1Y expected returns agree within ~1.5pp.

    Checks a spread of assets (one per class) — the analysis horizons and the
    ``/montecarlo`` result must come from the SAME drift + vol (one engine). The
    Monte Carlo is run with enough paths (:data:`_R3_SIMS`) that the residual is
    estimation noise, not a drift/vol mismatch.
    """
    gaps: dict[str, float] = {}
    for sym in _R3_SYMBOLS:
        a = audit[sym]
        one_y = _horizon(a, "1Y")
        assert one_y is not None
        mc = audit_engine.montecarlo(sym, "1Y", sims=_R3_SIMS)
        gap = abs(float(one_y.expected_return_pct) - float(mc.expected_return_pct))
        assert math.isfinite(gap)
        gaps[sym] = gap
    bad = {s: g for s, g in gaps.items() if g > 1.5}
    assert not bad, (
        "R3 analysis-vs-MC 1Y disagreement > 1.5pp: "
        + "; ".join(f"{s}: {g:.2f}pp" for s, g in bad.items())
    )


def test_reaudit_R4_confidence_spread(audit: dict[str, AssetAnalysis]) -> None:
    """R4: confidence spans a > 0.3 range across the 24 assets (not flat ~0.3)."""
    confs = [float(a.confidence) for a in audit.values()]
    assert all(0.0 <= c <= 1.0 and math.isfinite(c) for c in confs)
    spread = max(confs) - min(confs)
    assert spread > 0.3, (
        f"R4 confidence spread too narrow: max={max(confs):.3f} "
        f"min={min(confs):.3f} spread={spread:.3f}"
    )
    # And it must not be the old flat ~0.3 on everything.
    assert len(set(round(c, 2) for c in confs)) >= 5


def test_reaudit_R5_actionable_stance_mix(audit: dict[str, AssetAnalysis]) -> None:
    """R5: a realistic stance MIX — >= 5 BUY-or-better AND >= 3 SELL-or-worse."""
    stances = [a.recommendation for a in audit.values()]
    buys = sum(1 for s in stances if s in _BUY_OR_BETTER)
    sells = sum(1 for s in stances if s in _SELL_OR_WORSE)
    holds = sum(1 for s in stances if s == "HOLD")
    assert buys >= 5, f"R5 only {buys} BUY-or-better (need >= 5); mix={stances}"
    assert sells >= 3, f"R5 only {sells} SELL-or-worse (need >= 3); mix={stances}"
    # Defensive: the universe must not collapse to all-HOLD.
    assert holds < len(stances)


def test_reaudit_R6_honest_downside(audit: dict[str, AssetAnalysis]) -> None:
    """R6: every analysis surfaces finite 1Y CVaR / maxDD / prob_positive + a bear.

    The bear scenario must be strictly below the base case (an explicit downside).
    """
    offenders: list[str] = []
    for sym, a in audit.items():
        one_y = _horizon(a, "1Y")
        assert one_y is not None
        cvar = one_y.cvar_pct
        bear = one_y.bear_pct
        base = one_y.base_pct
        if cvar is None or not math.isfinite(cvar):
            offenders.append(f"{sym}: 1Y cvar non-finite")
        if not math.isfinite(a.risk_metrics.max_drawdown):
            offenders.append(f"{sym}: maxDrawdown non-finite")
        if not (0.0 <= one_y.prob_positive <= 1.0 and math.isfinite(one_y.prob_positive)):
            offenders.append(f"{sym}: 1Y prob_positive invalid")
        if bear is None or base is None or not (bear < base + 1e-9):
            offenders.append(f"{sym}: bear ({bear}) not < base ({base})")
        # A genuine downside scenario must be a loss-or-flat, not another gain.
        if bear is not None and bear > 0.0 + 1e-6:
            offenders.append(f"{sym}: 1Y bear is positive ({bear:.1f}%)")
    assert not offenders, "R6 downside failures: " + "; ".join(offenders)


def test_reaudit_R8_no_nan_inf_anywhere(audit: dict[str, AssetAnalysis]) -> None:
    """R8: zero NaN/inf anywhere in any analysis, and the disclaimer is present."""
    offenders: list[str] = []
    for sym, a in audit.items():
        nums: list[float] = [
            a.composite_score, a.confidence,
            a.risk_metrics.beta, a.risk_metrics.annual_vol, a.risk_metrics.sharpe,
            a.risk_metrics.sortino, a.risk_metrics.var95, a.risk_metrics.cvar95,
            a.risk_metrics.max_drawdown, a.risk_metrics.calmar,
        ]
        for h in a.expected_returns:
            nums += [
                h.expected_return_pct, h.low, h.high, h.prob_positive,
                h.annualized_vol,
            ]
            nums += [
                v for v in (h.bull_pct, h.base_pct, h.bear_pct, h.cvar_pct)
                if v is not None
            ]
        for sig in a.signals:
            nums += [sig.score, sig.confidence]
            nums += [v for v in sig.metrics.values()]
            for h in sig.horizons:
                nums += [
                    h.expected_return_pct, h.low, h.high, h.prob_positive,
                    h.annualized_vol,
                ]
        if not all(math.isfinite(float(v)) for v in nums):
            offenders.append(sym)
        if not a.disclaimer:
            offenders.append(f"{sym}: missing disclaimer")
    assert not offenders, "R8 non-finite / missing-disclaimer assets: " + ", ".join(offenders)
