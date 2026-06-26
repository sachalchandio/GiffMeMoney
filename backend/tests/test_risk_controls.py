"""Behavioral proofs for the audited strategy / risk fixes.

HONESTY / SAFETY (this is a finance tool). Everything exercised here is a
**SIMULATION on synthetic data** — there is no real market edge and nothing
predicts real-world results. These tests do not assert that any control makes
money; they assert that each *protection actually fires* as designed.

This module consolidates the behavioral guarantees for the verified flaws in the
audit (cross-referenced by their ids):

* **C2 (post-buy loss controls)** — a stop-loss auto-sells a losing position on
  ``risk/apply``; a portfolio max-drawdown breach reduces exposure to a cash
  floor (sells worst-first, never a full liquidation).
* **C3 (advisor not always 100% invested)** — the advisor parks real cash
  (``cashAmount > 0``) for a conservative profile, and the deployed risky
  fraction is strictly *smaller* for conservative than for aggressive at the
  same signal; a bear regime de-risks further than a bull regime.
* **C4 (basket downside fan)** — the blended 1Y horizon carries a finite
  ``cvarPct`` (no longer dropped to ``None``) and an honest ``bearPct < basePct``.
* **C5 (no dust legs)** — even a tiny ``$20`` request emits no ``$0.00`` legs.
* **NEW optimizer tail/cap** — the quant optimizer honors a per-name weight cap
  and the ``min_cvar`` objective underweights a fat-tail asset.
* **NEW feasibility honesty** — a 100x / 2-month target is flagged.
* **NEW no look-ahead** — the bot's point-in-time selection score depends only on
  data up to ``t`` and genuinely disagrees with the full-history (look-ahead)
  order.
* **C7 (no divide warnings)** — the directional indicators emit no
  ``RuntimeWarning`` on a flat series.

Anti-stall / speed: the money-path tests drive the services directly over a
**fresh, isolated** :class:`~app.invest.store.AccountStore` with a price-override
provider (no engine, a handful of symbols). The advisor tests use the smallest
profile where possible and reuse the shared engine's per-symbol cache; no
full-universe sweep is performed.
"""

from __future__ import annotations

import math
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from app.invest.advisor import AllocationAdvisor
from app.invest.portfolio_service import PortfolioService
from app.invest.store import AccountStore
from app.invest.wallet import WalletService
from app.market.provider import MarketDataProvider, get_provider
from app.quant import indicators as ind
from app.quant import portfolio as pf
from app.schemas import AllocationItem, CardIn, RiskPolicy

# A canonical Luhn-valid Visa test PAN and a far-future expiry.
VALID_VISA = "4111111111111111"
EXP_MONTH, EXP_YEAR = 12, 2030

# A small, stable set of always-priceable equity symbols (no universe scan).
SAMPLE_SYMBOLS = ["AAPL", "MSFT", "JPM"]


def _card() -> CardIn:
    """Build a valid :class:`~app.schemas.CardIn` for the test Visa PAN."""
    return CardIn(
        number=VALID_VISA,
        exp_month=EXP_MONTH,
        exp_year=EXP_YEAR,
        cvc="123",
        holder="Ada Lovelace",
    )


# ---------------------------------------------------------------------------
# A price-overridable provider so a held position's mark can be pushed up/down.
# ---------------------------------------------------------------------------


class StubPriceProvider(MarketDataProvider):
    """A provider whose per-symbol last price is settable in tests.

    Delegates everything except :meth:`latest_price` to a wrapped real
    (deterministic) provider so asset snapshots stay valid, while letting a test
    push a held position's mark to trigger the risk rules.
    """

    def __init__(self, inner: MarketDataProvider) -> None:
        self._inner = inner
        self._overrides: dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        """Override the last price for ``symbol`` (canonical upper-case key)."""
        self._overrides[symbol.strip().upper()] = float(price)

    # -- overridden -----------------------------------------------------
    def latest_price(self, symbol: str) -> float:
        key = symbol.strip().upper()
        if key in self._overrides:
            return self._overrides[key]
        return self._inner.latest_price(symbol)

    # -- delegated ------------------------------------------------------
    def list_assets(self):  # pragma: no cover - trivial delegation
        return self._inner.list_assets()

    def get_asset(self, symbol: str):
        return self._inner.get_asset(symbol)

    def get_candles(self, symbol: str, limit: int = 365):  # pragma: no cover
        return self._inner.get_candles(symbol, limit)

    def history(self, symbol: str, days: int = 365):  # pragma: no cover
        return self._inner.history(symbol, days)

    def market_history(self, days: int = 365):  # pragma: no cover
        return self._inner.market_history(days)

    def factor_history(self, days: int = 365):  # pragma: no cover
        return self._inner.factor_history(days)

    def fundamentals(self, symbol: str):  # pragma: no cover
        return self._inner.fundamentals(symbol)


@pytest.fixture
def store() -> AccountStore:
    """A fresh, isolated in-memory account store (never the singleton)."""
    return AccountStore()


@pytest.fixture
def stub_provider() -> StubPriceProvider:
    """A price-overridable provider wrapping the real deterministic one."""
    return StubPriceProvider(get_provider())


@pytest.fixture
def wallet_service(store: AccountStore) -> WalletService:
    """A wallet service over the fresh store, used only to fund accounts."""
    from app.invest.payments import SimulatedPaymentProvider

    return WalletService(store, SimulatedPaymentProvider(), get_provider())


@pytest.fixture
def risk_service(
    store: AccountStore, stub_provider: StubPriceProvider
) -> PortfolioService:
    """A :class:`PortfolioService` over the fresh store + stub-price provider."""
    return PortfolioService(store, stub_provider)


def _entry_price(svc: PortfolioService, account: str, symbol: str) -> float:
    """Return the position's blended entry (avg-cost) price."""
    pos = next(p for p in svc.get_state(account).positions if p.symbol == symbol)
    return pos.avg_price


# ---------------------------------------------------------------------------
# Shared advisor fixture: one real basket per profile (cache warmed once).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def advisor() -> AllocationAdvisor:
    """One advisor bound to the shared engine + provider (per-symbol cache reuse)."""
    from app.api.recommendations import get_engine

    return AllocationAdvisor(get_engine(), get_provider())


# ===========================================================================
# C2 — post-buy loss controls actually fire (stop-loss + max-drawdown)
# ===========================================================================


def test_stop_loss_auto_sells_a_losing_position(
    wallet_service: WalletService,
    stub_provider: StubPriceProvider,
    risk_service: PortfolioService,
) -> None:
    """C2: a position pushed below the stop-loss is auto-sold on risk/apply.

    Before the fix the invest path had NO post-buy loss controls — a losing
    position could never be exited mechanically. This proves the stop-loss now
    liquidates the position, records a ``stop_loss`` action at a realized loss,
    and returns the (reduced) proceeds to cash.
    """
    wallet_service.deposit("acct", 1000.0, _card(), save_card=False)
    risk_service.invest("acct", [AllocationItem(symbol="AAPL", amount=300.0)])
    cash_after_buy = risk_service.get_state("acct").wallet.cash_balance

    risk_service.set_risk_policy("acct", RiskPolicy(stop_loss_pct=10.0))
    entry = _entry_price(risk_service, "acct", "AAPL")
    # Push the mark 25% below entry — well past the 10% stop.
    stub_provider.set_price("AAPL", entry * 0.75)

    result = risk_service.evaluate_risk("acct")

    assert result.triggered is True
    actions = [a for a in result.actions if a.action == "stop_loss"]
    assert len(actions) == 1
    assert actions[0].symbol == "AAPL"
    assert actions[0].realized_pnl < 0.0  # exited at a loss
    # Position is gone and the proceeds returned to cash.
    assert all(p.symbol != "AAPL" for p in result.state.positions)
    assert result.state.wallet.cash_balance > cash_after_buy


def test_max_drawdown_reduces_exposure_to_a_cash_floor(
    wallet_service: WalletService,
    stub_provider: StubPriceProvider,
    risk_service: PortfolioService,
) -> None:
    """C2: a portfolio drawdown breach cuts exposure (worst-first), not all.

    Three fully-invested positions; a hard crash drives the portfolio well past
    the 20% drawdown limit. The circuit-breaker raises cash by selling the worst
    position first, reducing exposure toward a defensive cash floor while leaving
    at least one position standing (de-risk, not liquidate).
    """
    wallet_service.deposit("acct", 1000.0, _card(), save_card=False)
    risk_service.invest(
        "acct",
        [
            AllocationItem(symbol="AAPL", amount=500.0),
            AllocationItem(symbol="MSFT", amount=250.0),
            AllocationItem(symbol="JPM", amount=250.0),
        ],
    )
    risk_service.get_state("acct")  # establish the peak at the entry marks
    risk_service.set_risk_policy("acct", RiskPolicy(max_drawdown_pct=20.0))

    aapl = _entry_price(risk_service, "acct", "AAPL")
    msft = _entry_price(risk_service, "acct", "MSFT")
    jpm = _entry_price(risk_service, "acct", "JPM")
    # Crash AAPL hardest (the worst), nick the others — total value falls > 20%.
    stub_provider.set_price("AAPL", aapl * 0.30)
    stub_provider.set_price("MSFT", msft * 0.95)
    stub_provider.set_price("JPM", jpm * 0.95)

    cash_before = risk_service.get_state("acct").wallet.cash_balance
    result = risk_service.evaluate_risk("acct")

    assert result.triggered is True
    drawdown_actions = [a for a in result.actions if a.action == "drawdown"]
    assert drawdown_actions, "max-drawdown breaker did not fire"
    # The worst position (AAPL) is the first sold to reduce exposure.
    assert drawdown_actions[0].symbol == "AAPL"
    assert all(p.symbol != "AAPL" for p in result.state.positions)
    # De-risk, not full liquidation: a position survives the floor.
    assert result.state.positions, "drawdown breaker liquidated everything"
    # Exposure reduced: cash rose.
    assert result.state.wallet.cash_balance > cash_before


# ===========================================================================
# C3 — advisor holds CASH; conservative deploys less than aggressive; bear de-risks
# ===========================================================================


def test_advisor_holds_cash_for_conservative(advisor: AllocationAdvisor) -> None:
    """C3: the conservative advice is NOT 100% invested — it parks real cash.

    Before the fix every weight summed to 1 (always fully invested). Now a cash
    sleeve holds the undeployed remainder, so ``cashAmount > 0`` and the risky
    legs sum to strictly less than the full amount.
    """
    advice = advisor.advise(1000.0, "conservative")
    assert advice.cash_amount > 0.0
    assert advice.cash_weight > 0.0
    risky = sum(it.weight for it in advice.items)
    assert risky < 1.0 - 1e-6  # not fully invested
    # Risky legs + cash reconcile to the whole book.
    assert risky + advice.cash_weight == pytest.approx(1.0, abs=1e-2)
    assert (
        sum(it.amount for it in advice.items) + advice.cash_amount
        == pytest.approx(1000.0, abs=1.0)
    )


def test_conservative_deploys_less_than_aggressive(
    advisor: AllocationAdvisor,
) -> None:
    """C3: the deployed risky fraction is smaller for conservative than aggressive.

    Same request, two profiles: the conservative profile caps risky exposure
    (more cash), the aggressive profile deploys nearly fully — so the conservative
    risky fraction is strictly below the aggressive one.
    """
    cons = advisor.advise(1000.0, "conservative")
    aggr = advisor.advise(1000.0, "aggressive")
    cons_risky = sum(it.weight for it in cons.items)
    aggr_risky = sum(it.weight for it in aggr.items)
    assert cons_risky < aggr_risky
    # And the conservative book holds strictly more cash.
    assert cons.cash_amount > aggr.cash_amount


def test_bear_regime_derisks_more_than_bull(advisor: AllocationAdvisor) -> None:
    """C3: at an identical signal a bear regime deploys less than a bull regime.

    Exercises :meth:`AllocationAdvisor._risky_fraction` directly with two picks
    that differ ONLY in their regime score (same strong composite). The bear
    case must produce a strictly smaller deployed fraction, and conservative must
    deploy strictly less than aggressive at the same (bullish) signal.
    """

    def pick(comp: float, regime_score: float) -> SimpleNamespace:
        # _risky_fraction reads only composite_score and regime.score.
        return SimpleNamespace(
            composite_score=comp, regime=SimpleNamespace(score=regime_score)
        )

    weights = np.array([0.5, 0.5])
    mu = np.array([0.1, 0.1])
    cov = np.eye(2) * 0.04

    bull = [pick(50.0, 0.8), pick(50.0, 0.8)]
    bear = [pick(50.0, -0.8), pick(50.0, -0.8)]

    f_bull = advisor._risky_fraction("conservative", bull, weights, mu, cov)
    f_bear = advisor._risky_fraction("conservative", bear, weights, mu, cov)
    assert f_bear < f_bull, "a bear regime should de-risk below a bull regime"

    # Same strong-bull signal: conservative caps risky exposure below aggressive.
    f_cons = advisor._risky_fraction("conservative", bull, weights, mu, cov)
    f_aggr = advisor._risky_fraction("aggressive", bull, weights, mu, cov)
    assert f_cons < f_aggr


# ===========================================================================
# C4 — basket 1Y CVaR is finite (not None) and the downside fan is honest
# ===========================================================================


def test_basket_1y_cvar_is_finite_and_bear_below_base(
    advisor: AllocationAdvisor,
) -> None:
    """C4: the blended 1Y horizon carries a finite CVaR with bear < base.

    Before the fix ``_blend_horizons`` dropped the bull/base/bear/cvar fan, so
    the basket's ``cvarPct`` was always ``None``. Now the blend carries each
    field with its own weight, so the 1Y horizon reports a finite CVaR and an
    honest scenario fan (a bear case strictly worse than the base case).
    """
    advice = advisor.advise(1000.0, "balanced")
    one_year = next(h for h in advice.horizons if h.horizon == "1Y")

    assert one_year.cvar_pct is not None, "basket 1Y CVaR was dropped to None (C4)"
    assert math.isfinite(one_year.cvar_pct)
    assert one_year.cvar_pct >= 0.0  # CVaR is a positive loss figure

    assert one_year.base_pct is not None and one_year.bear_pct is not None
    assert one_year.bear_pct < one_year.base_pct, "bear scenario must be worse than base"


def test_all_five_horizons_carry_the_downside_fan(
    advisor: AllocationAdvisor,
) -> None:
    """C4: every one of the five horizons carries a populated downside fan.

    Keeping the canonical five horizons is a standing invariant; here each one
    additionally carries the (no-longer-dropped) bull/base/bear/cvar fields.
    """
    advice = advisor.advise(1000.0, "balanced")
    assert {h.horizon for h in advice.horizons} == {"1D", "1W", "1M", "1Y", "5Y"}
    for h in advice.horizons:
        for field in (h.bull_pct, h.base_pct, h.bear_pct, h.cvar_pct):
            assert field is not None
            assert math.isfinite(field)


# ===========================================================================
# C5 — no $0.00 dust legs, even on a tiny request
# ===========================================================================


def test_small_request_emits_no_zero_dollar_legs(advisor: AllocationAdvisor) -> None:
    """C5: a tiny ``$20`` request never emits a ``$0.00`` (sub-cent) allocation leg.

    Before the fix small amounts produced dust legs that rounded to ``$0.00``.
    Now dust legs are dropped and the survivors renormalized, so every emitted
    leg carries a strictly-positive weight and dollar amount, and the legs plus
    cash still reconcile to the request.
    """
    advice = advisor.advise(20.0, "aggressive")
    assert advice.items, "advisor returned no legs for a small request"
    for it in advice.items:
        assert it.amount > 0.0, f"{it.asset.symbol} leg is a $0.00 dust leg"
        assert it.weight > 0.0
    assert (
        sum(it.amount for it in advice.items) + advice.cash_amount
        == pytest.approx(20.0, abs=0.05)
    )


# ===========================================================================
# NEW optimizer — per-name weight cap + CVaR (tail) objective
# ===========================================================================


def test_optimizer_honors_per_name_weight_cap() -> None:
    """NEW: a per-name cap keeps any single name from dominating the basket.

    One asset has a far higher expected return, so an uncapped max-Sharpe solver
    would pile weight onto it. With ``max_weight=0.35`` no weight may exceed the
    cap, while the vector still sums to 1 (long-only, fully invested).
    """
    mu = np.array([0.30, 0.05, 0.05, 0.05])
    cov = np.eye(4) * 0.04
    w = pf.optimize(mu, cov, 0.04, "max_sharpe", max_weight=0.35)
    assert float(np.max(w)) <= 0.35 + 1e-6
    assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)
    assert np.all(w >= -1e-12)


def test_min_cvar_objective_underweights_a_fat_tail_asset() -> None:
    """NEW: the CVaR objective steers weight away from a deep-loss tail.

    Asset 0 carries occasional catastrophic crash days (a fat negative tail);
    asset 1 is steady. ``min_volatility`` (a symmetric variance penalty) would
    not strongly distinguish them, but ``min_cvar`` (expected-shortfall) must
    underweight the tail-heavy name.
    """
    rng = np.random.default_rng(0)
    r0 = rng.normal(0.001, 0.01, 500)
    r0[:25] = -0.20  # crash days → a deep left tail
    r1 = rng.normal(0.001, 0.01, 500)
    R = np.column_stack([r0, r1])
    mu = R.mean(axis=0)
    cov = np.cov(R, rowvar=False)

    w = pf.optimize(mu, cov, 0.0, "min_cvar", returns_matrix=R, cvar_beta=0.95)
    assert float(np.sum(w)) == pytest.approx(1.0, abs=1e-6)
    assert w[1] > w[0], "min_cvar should underweight the fat-tail asset"
    # The historical CVaR of the chosen book is finite and a positive loss.
    cvar = pf.portfolio_cvar(w, R, 0.95)
    assert math.isfinite(cvar) and cvar >= 0.0


# ===========================================================================
# NEW feasibility — an impossible target is flagged through advise()
# ===========================================================================


def test_advise_flags_a_100x_two_month_target(advisor: AllocationAdvisor) -> None:
    """NEW: a 100x / ~2-month ask surfaces a targetWarning (the basket is unchanged).

    The feasibility guard never alters the allocation — it only attaches an
    honest warning so the advice cannot imply an impossible target is achievable.
    A sane request, by contrast, carries no warning.
    """
    flagged = advisor.advise(
        100.0, "conservative", target_amount=10_000.0, horizon_days=61
    )
    assert flagged.target_warning is not None
    assert "every day" in flagged.target_warning.lower()
    # Honesty flag is always set; the basket itself is still produced.
    assert flagged.synthetic_data is True

    sane = advisor.advise(
        1000.0, "conservative", target_amount=1100.0, horizon_days=365
    )
    assert sane.target_warning is None


# ===========================================================================
# NEW no look-ahead — bot point-in-time selection ignores the future
# ===========================================================================


def test_bot_pit_score_uses_only_data_up_to_t() -> None:
    """NEW: the point-in-time selection score at ``t`` ignores all data after ``t``.

    Corrupting every bar strictly AFTER ``t`` must not change the as-of-``t``
    score — the defining property of a no-look-ahead selection signal (the old
    engine reused a full-history composite that peeked at the future).
    """
    from app.bot.engine import AutoTraderEngine

    engine = AutoTraderEngine()
    rng = np.random.default_rng(3)
    n = 120
    cols = [
        100.0 * np.cumprod(1.0 + (d + rng.normal(0.0, 0.005, size=n)))
        for d in (0.004, 0.001, -0.003)
    ]
    prices = np.column_stack(cols)
    t = 60

    before = engine._pit_scores(prices, t)
    future = prices.copy()
    future[t + 1 :] *= 5.0  # wild spike strictly after t
    after = engine._pit_scores(future, t)

    assert np.allclose(before, after)
    assert np.all(np.isfinite(before))


def test_bot_pit_ranking_differs_from_full_history_ranking() -> None:
    """NEW: the point-in-time ranking genuinely disagrees with the look-ahead one.

    Sleeve A is FALLING through the early window and only rallies AFTER bar ``t``,
    so it wins the full-history (future-peeking) race but must rank LAST as-of
    ``t``. Proving the two orders disagree shows the selection is computed from
    data up to ``t`` — not the frozen full-history score the old engine reused at
    every past rebalance.
    """
    from app.bot.engine import AutoTraderEngine

    engine = AutoTraderEngine()
    rng = np.random.default_rng(5)
    n = 120
    t = 50  # an early rebalance, strictly before A's late rally

    def series(daily: np.ndarray) -> np.ndarray:
        return 100.0 * np.cumprod(1.0 + daily)

    a_daily = np.concatenate(
        [
            -0.004 + rng.normal(0.0, 0.003, size=t + 1),  # falling up to & incl. t
            0.05 + rng.normal(0.0, 0.003, size=n - t - 1),  # explosive after t
        ]
    )
    b_daily = 0.006 + rng.normal(0.0, 0.003, size=n)  # early leader
    c_daily = 0.002 + rng.normal(0.0, 0.003, size=n)  # stable middle
    prices = np.column_stack([series(a_daily), series(b_daily), series(c_daily)])

    pit = engine._pit_scores(prices, t)
    pit_order = sorted(range(3), key=lambda j: pit[j], reverse=True)

    full_hist = prices[-1] / prices[0] - 1.0  # the look-ahead stand-in
    full_order = sorted(range(3), key=lambda j: full_hist[j], reverse=True)

    assert full_order[0] == 0  # A is the full-history (look-ahead) winner
    assert pit_order[-1] == 0  # but A ranks LAST as-of t (it was falling)
    assert pit_order != full_order


# ===========================================================================
# C7 — directional indicators emit no divide RuntimeWarning on a flat series
# ===========================================================================


def test_directional_indicators_no_runtime_warning_on_flat_series() -> None:
    """C7: ``adx`` / ``adx_components`` emit no divide RuntimeWarning on a flat tape.

    The directional divides are wrapped in ``np.errstate`` like the seven sibling
    sites. Promoting RuntimeWarning to an error proves a flat series (ATR ~0)
    runs clean while the neutral, zero output is unchanged.
    """
    flat = np.full(40, 50.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        value = ind.adx(flat, flat, flat, n=14)
        pdi, mdi, adx_val = ind.adx_components(flat, flat, flat, n=14)
    assert value == pytest.approx(0.0)
    assert (pdi, mdi, adx_val) == pytest.approx((0.0, 0.0, 0.0))
