"""Tests for the simulated brokerage / Invest extension (docs/INVEST.md).

These cover the in-memory paper-trading sandbox end to end:

* **Payments** — Luhn validity, brand detection, and that a saved card is stored
  **masked only** (no raw PAN ever reaches the store).
* **Wallet** — deposit credits cash, withdraw rejects amounts over the balance,
  and the wallet always reconciles (``cash + invested == total``).
* **Portfolio** — invest splits cash across multiple symbols, rejects over-spend,
  records the correct average cost basis, and sell realizes P&L and returns cash.
* **History** — the backfilled curve returns a total series of the requested
  length plus one per-position series of matching length and shape.
* **Advisor** — weights sum to ~1, per-item dollar amounts sum to ~= the request,
  and exactly five horizons are returned.
* **API smoke** — every new ``/api`` route returns 200 with camelCase keys, the
  invest router is mounted under ``/api``, and the contract error codes hold.

Anti-stall / speed: service-level tests drive the invest services directly over a
**fresh, isolated** :class:`~app.invest.store.AccountStore` (never the shared
singleton), so they touch no engine and only price a handful of symbols. The
single advisor test uses the smallest pick set (``conservative`` -> 4 picks) and
the API tests give every test a **fresh ``X-Account-Id``** so account state never
leaks between tests. No full-universe analysis sweep is performed.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.invest.payments import (
    SimulatedPaymentProvider,
    brand_for,
    luhn_valid,
    mask,
    tokenize,
)
from app.invest.portfolio_service import PortfolioService
from app.invest.store import AccountStore
from app.invest.wallet import WalletService
from app.main import app
from app.market.provider import MarketDataProvider, get_provider
from app.schemas import AllocationItem, CardIn, RiskPolicy

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------

# A canonical Luhn-valid Visa test PAN (starts with 4) and a one-off invalid PAN.
VALID_VISA = "4111111111111111"
INVALID_PAN = "4111111111111112"
VALID_MASTERCARD = "5555555555554444"
VALID_AMEX = "378282246310005"
VALID_DISCOVER = "6011111111111117"

# A far-future expiry so the card never expires relative to the test clock.
EXP_MONTH = 12
EXP_YEAR = 2030

# A small, stable set of always-priceable equity symbols for the money-path
# tests — avoids any full-universe scan.
SAMPLE_SYMBOLS = ["AAPL", "MSFT", "JPM"]


def _card(number: str = VALID_VISA) -> CardIn:
    """Build a valid :class:`~app.schemas.CardIn` for ``number``."""
    return CardIn(
        number=number,
        exp_month=EXP_MONTH,
        exp_year=EXP_YEAR,
        cvc="123",
        holder="Ada Lovelace",
    )


@pytest.fixture
def store() -> AccountStore:
    """A fresh, isolated in-memory account store (never the singleton)."""
    return AccountStore()


@pytest.fixture
def wallet_service(store: AccountStore) -> WalletService:
    """A :class:`WalletService` over the fresh store, sandbox payments and the
    real (deterministic) market provider."""
    return WalletService(store, SimulatedPaymentProvider(), get_provider())


@pytest.fixture
def portfolio_service(store: AccountStore) -> PortfolioService:
    """A :class:`PortfolioService` over the fresh store and market provider."""
    return PortfolioService(store, get_provider())


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """A module-scoped TestClient that runs the app lifespan (tick loop)."""
    with TestClient(app) as test_client:
        yield test_client


def _fresh_account() -> str:
    """Return a unique account id so each API test starts from empty state."""
    return f"test-{uuid.uuid4().hex[:12]}"


def _headers() -> dict[str, str]:
    """Return request headers carrying a fresh ``X-Account-Id``."""
    return {"X-Account-Id": _fresh_account()}


# ---------------------------------------------------------------------------
# Payments: Luhn, brand detection, masking
# ---------------------------------------------------------------------------


def test_luhn_valid_accepts_good_card() -> None:
    """A Luhn-valid PAN (with or without spaces) passes the checksum."""
    assert luhn_valid(VALID_VISA) is True
    assert luhn_valid("4111 1111 1111 1111") is True
    assert luhn_valid(VALID_MASTERCARD) is True
    assert luhn_valid(VALID_AMEX) is True


def test_luhn_rejects_bad_card() -> None:
    """A PAN that fails the mod-10 checksum (or is too short) is rejected."""
    assert luhn_valid(INVALID_PAN) is False
    assert luhn_valid("1234567890123456") is False
    assert luhn_valid("411") is False
    assert luhn_valid("") is False


def test_brand_detection() -> None:
    """Brand is derived from the leading digits of the PAN."""
    assert brand_for(VALID_VISA) == "visa"
    assert brand_for(VALID_MASTERCARD) == "mastercard"
    assert brand_for("2221000000000009") == "mastercard"
    assert brand_for(VALID_AMEX) == "amex"
    assert brand_for(VALID_DISCOVER) == "discover"
    assert brand_for("9999999999999999") == "unknown"
    assert brand_for("") == "unknown"


def test_mask_shows_only_last_four() -> None:
    """Masking reveals only the last four digits."""
    assert mask(VALID_VISA) == "•••• 1111"
    assert mask("4111 1111 1111 1234") == "•••• 1234"


def test_tokenize_drops_raw_pan() -> None:
    """Tokenizing a card retains only masked data — never the raw PAN/CVC."""
    saved = tokenize(_card(VALID_VISA))
    assert saved.brand == "visa"
    assert saved.last4 == "1111"
    # No field on the masked card carries the full PAN or the CVC.
    blob = saved.model_dump_json()
    assert VALID_VISA not in blob
    assert "123" not in saved.last4  # the cvc digits never leak into last4


# ---------------------------------------------------------------------------
# SimulatedPaymentProvider
# ---------------------------------------------------------------------------


def test_charge_rejects_invalid_card() -> None:
    """A charge on a Luhn-invalid card raises ValueError."""
    provider = SimulatedPaymentProvider()
    with pytest.raises(ValueError):
        provider.charge(_card(INVALID_PAN), 100.0, "demo")


def test_charge_rejects_non_positive_and_over_limit() -> None:
    """A charge with amount <= 0 or above the sandbox cap raises ValueError."""
    provider = SimulatedPaymentProvider()
    with pytest.raises(ValueError):
        provider.charge(_card(), 0.0, "demo")
    with pytest.raises(ValueError):
        provider.charge(_card(), 25_000.0, "demo")


def test_charge_succeeds_for_valid_card() -> None:
    """A valid card + sane amount yields a completed deposit transaction."""
    provider = SimulatedPaymentProvider()
    txn = provider.charge(_card(), 250.0, "demo")
    assert txn.type == "deposit"
    assert txn.status == "completed"
    assert txn.amount == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# Wallet service: deposit, save-card masking, withdraw, reconciliation
# ---------------------------------------------------------------------------


def test_deposit_credits_cash(wallet_service: WalletService) -> None:
    """A deposit credits the cash balance by the deposited amount."""
    wallet, txn = wallet_service.deposit("acct", 500.0, _card(), save_card=False)
    assert wallet.cash_balance == pytest.approx(500.0)
    assert txn.type == "deposit"
    # No card was saved.
    assert wallet.saved_cards == []


def test_save_card_stores_masked_card_only(
    wallet_service: WalletService, store: AccountStore
) -> None:
    """Saving a card stores a MASKED SavedCard — the raw PAN is nowhere in state."""
    wallet, _ = wallet_service.deposit("acct", 100.0, _card(VALID_VISA), save_card=True)
    assert len(wallet.saved_cards) == 1
    saved = wallet.saved_cards[0]
    assert saved.brand == "visa"
    assert saved.last4 == "1111"

    # The raw PAN must not appear anywhere in the persisted account state.
    account = store.get_account("acct")
    for card in account.saved_cards:
        dumped = card.model_dump_json()
        assert VALID_VISA not in dumped
        assert not hasattr(card, "number")
        assert not hasattr(card, "cvc")


def test_withdraw_rejects_over_balance(wallet_service: WalletService) -> None:
    """Withdrawing more than the cash balance raises ValueError (HTTP 400)."""
    wallet_service.deposit("acct", 200.0, _card(), save_card=False)
    with pytest.raises(ValueError):
        wallet_service.withdraw("acct", 500.0)


def test_withdraw_debits_cash(wallet_service: WalletService) -> None:
    """A valid withdrawal debits cash by the withdrawn amount."""
    wallet_service.deposit("acct", 400.0, _card(), save_card=False)
    wallet, txn = wallet_service.withdraw("acct", 150.0)
    assert wallet.cash_balance == pytest.approx(250.0)
    assert txn.type == "withdrawal"


def test_balances_reconcile_after_invest(
    wallet_service: WalletService, portfolio_service: PortfolioService
) -> None:
    """cash + invested == total holds before and after investing."""
    wallet_service.deposit("acct", 1000.0, _card(), save_card=False)
    portfolio_service.invest(
        "acct",
        [AllocationItem(symbol="AAPL", amount=300.0)],
    )
    wallet = wallet_service.get_wallet("acct")
    assert wallet.total_value == pytest.approx(
        wallet.cash_balance + wallet.invested_value, abs=0.02
    )
    # Cash was reduced by exactly the invested dollars.
    assert wallet.cash_balance == pytest.approx(700.0, abs=0.01)


# ---------------------------------------------------------------------------
# Portfolio service: invest split, over-spend, cost basis, sell P&L
# ---------------------------------------------------------------------------


def test_invest_splits_cash_across_symbols(
    wallet_service: WalletService, portfolio_service: PortfolioService
) -> None:
    """Investing splits cash across multiple symbols, opening one position each."""
    wallet_service.deposit("acct", 1000.0, _card(), save_card=False)
    state = portfolio_service.invest(
        "acct",
        [AllocationItem(symbol=s, amount=200.0) for s in SAMPLE_SYMBOLS],
    )
    held = {p.symbol for p in state.positions}
    assert held == set(SAMPLE_SYMBOLS)
    # Total cost basis equals the spent dollars (3 x 200).
    assert state.total_cost == pytest.approx(600.0, abs=0.01)
    # Cash dropped by the same amount.
    assert state.wallet.cash_balance == pytest.approx(400.0, abs=0.01)


def test_invest_rejects_over_spend(
    wallet_service: WalletService, portfolio_service: PortfolioService
) -> None:
    """An order whose total exceeds available cash is rejected (all-or-nothing)."""
    wallet_service.deposit("acct", 100.0, _card(), save_card=False)
    with pytest.raises(ValueError):
        portfolio_service.invest(
            "acct",
            [AllocationItem(symbol="AAPL", amount=250.0)],
        )
    # Nothing was applied: cash untouched, no positions opened.
    state = portfolio_service.get_state("acct")
    assert state.wallet.cash_balance == pytest.approx(100.0)
    assert state.positions == []


def test_invest_unknown_symbol_raises_keyerror(
    wallet_service: WalletService, portfolio_service: PortfolioService
) -> None:
    """Investing in an unknown symbol raises KeyError (HTTP 404)."""
    wallet_service.deposit("acct", 100.0, _card(), save_card=False)
    with pytest.raises(KeyError):
        portfolio_service.invest(
            "acct",
            [AllocationItem(symbol="ZZZZ_NOPE", amount=50.0)],
        )


def test_avg_cost_basis_blends_on_reinvest(
    wallet_service: WalletService, portfolio_service: PortfolioService
) -> None:
    """Re-investing in a held symbol accumulates units and cost basis."""
    wallet_service.deposit("acct", 1000.0, _card(), save_card=False)
    portfolio_service.invest("acct", [AllocationItem(symbol="AAPL", amount=200.0)])
    state = portfolio_service.invest(
        "acct", [AllocationItem(symbol="AAPL", amount=100.0)]
    )
    pos = next(p for p in state.positions if p.symbol == "AAPL")
    # Cost basis is the total dollars invested across both buys.
    assert pos.cost_basis == pytest.approx(300.0, abs=0.01)
    # avg_price == cost_basis / units exactly.
    assert pos.avg_price == pytest.approx(pos.cost_basis / pos.units, rel=1e-4)


def test_sell_realizes_pnl_and_returns_cash(
    wallet_service: WalletService, portfolio_service: PortfolioService
) -> None:
    """Selling the whole position credits cash with the proceeds and closes it."""
    wallet_service.deposit("acct", 1000.0, _card(), save_card=False)
    portfolio_service.invest("acct", [AllocationItem(symbol="AAPL", amount=300.0)])
    cash_before_sell = portfolio_service.get_state("acct").wallet.cash_balance

    # Liquidate the whole position.
    state = portfolio_service.sell("acct", "AAPL", sell_all=True)
    assert all(p.symbol != "AAPL" for p in state.positions)

    # Cash increased by the proceeds; total P&L is finite (no NaN/inf).
    assert state.wallet.cash_balance > cash_before_sell
    assert state.total_value == pytest.approx(0.0, abs=0.01)  # no positions left

    # A sell transaction was recorded.
    txns = wallet_service.list_transactions("acct")
    assert any(t.type == "sell" and t.symbol == "AAPL" for t in txns)


def test_sell_unknown_position_raises_keyerror(
    portfolio_service: PortfolioService,
) -> None:
    """Selling a symbol with no open position raises KeyError (HTTP 404)."""
    with pytest.raises(KeyError):
        portfolio_service.sell("acct", "AAPL", sell_all=True)


# ---------------------------------------------------------------------------
# Risk policy (post-buy loss controls): stub-price provider + service tests
# ---------------------------------------------------------------------------


class StubPriceProvider(MarketDataProvider):
    """A market provider whose per-symbol last price is settable in tests.

    Delegates everything except :meth:`latest_price` to a wrapped real
    (deterministic) provider so asset snapshots stay valid, while letting a test
    push a held position's mark up or down to trigger the risk rules.
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
def stub_provider() -> StubPriceProvider:
    """A price-overridable provider wrapping the real deterministic one."""
    return StubPriceProvider(get_provider())


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


def test_risk_policy_defaults_off(risk_service: PortfolioService) -> None:
    """A fresh account has an all-OFF risk policy (no rules enabled)."""
    policy = risk_service.get_risk_policy("acct")
    assert policy.stop_loss_pct is None
    assert policy.trailing_stop_pct is None
    assert policy.take_profit_pct is None
    assert policy.max_drawdown_pct is None


def test_risk_policy_rejects_non_positive(risk_service: PortfolioService) -> None:
    """Setting a non-positive threshold raises ValueError (HTTP 400)."""
    with pytest.raises(ValueError):
        risk_service.set_risk_policy("acct", RiskPolicy(stop_loss_pct=0.0))
    with pytest.raises(ValueError):
        risk_service.set_risk_policy("acct", RiskPolicy(trailing_stop_pct=-5.0))


def test_risk_policy_set_and_get_round_trip(
    risk_service: PortfolioService,
) -> None:
    """A valid policy round-trips through set/get."""
    saved = risk_service.set_risk_policy(
        "acct",
        RiskPolicy(stop_loss_pct=10.0, take_profit_pct=40.0),
    )
    assert saved.stop_loss_pct == pytest.approx(10.0)
    assert saved.take_profit_pct == pytest.approx(40.0)
    # Trailing/drawdown stay OFF.
    assert saved.trailing_stop_pct is None
    assert saved.max_drawdown_pct is None
    fetched = risk_service.get_risk_policy("acct")
    assert fetched.stop_loss_pct == pytest.approx(10.0)


def test_apply_noop_when_policy_off(
    wallet_service: WalletService,
    store: AccountStore,
    risk_service: PortfolioService,
) -> None:
    """With the default OFF policy, apply triggers no actions and holds state."""
    wallet_service.deposit("acct", 1000.0, _card(), save_card=False)
    risk_service.invest("acct", [AllocationItem(symbol="AAPL", amount=300.0)])
    result = risk_service.evaluate_risk("acct")
    assert result.triggered is False
    assert result.actions == []
    assert any(p.symbol == "AAPL" for p in result.state.positions)


def test_stop_loss_auto_sells_and_returns_cash(
    wallet_service: WalletService,
    store: AccountStore,
    stub_provider: StubPriceProvider,
    risk_service: PortfolioService,
) -> None:
    """A position pushed below the stop-loss is auto-sold and cash returns."""
    wallet_service.deposit("acct", 1000.0, _card(), save_card=False)
    risk_service.invest("acct", [AllocationItem(symbol="AAPL", amount=300.0)])
    cash_after_buy = risk_service.get_state("acct").wallet.cash_balance

    risk_service.set_risk_policy("acct", RiskPolicy(stop_loss_pct=10.0))
    entry = _entry_price(risk_service, "acct", "AAPL")
    # Push the mark 20% below entry — well past the 10% stop.
    stub_provider.set_price("AAPL", entry * 0.80)

    result = risk_service.evaluate_risk("acct")
    assert result.triggered is True
    assert len(result.actions) == 1
    action = result.actions[0]
    assert action.symbol == "AAPL"
    assert action.action == "stop_loss"
    assert action.realized_pnl < 0  # sold at a loss
    # Position is gone and the (reduced) proceeds returned to cash.
    assert all(p.symbol != "AAPL" for p in result.state.positions)
    assert result.state.wallet.cash_balance > cash_after_buy
    # A sell transaction was recorded.
    txns = wallet_service.list_transactions("acct")
    assert any(t.type == "sell" and t.symbol == "AAPL" for t in txns)


def test_take_profit_auto_sells_on_gain(
    wallet_service: WalletService,
    stub_provider: StubPriceProvider,
    risk_service: PortfolioService,
) -> None:
    """A position pushed above the take-profit is auto-sold at a gain."""
    wallet_service.deposit("acct", 1000.0, _card(), save_card=False)
    risk_service.invest("acct", [AllocationItem(symbol="MSFT", amount=200.0)])
    risk_service.set_risk_policy("acct", RiskPolicy(take_profit_pct=30.0))
    entry = _entry_price(risk_service, "acct", "MSFT")
    stub_provider.set_price("MSFT", entry * 1.50)  # +50%, past the 30% target

    result = risk_service.evaluate_risk("acct")
    assert result.triggered is True
    assert result.actions[0].action == "take_profit"
    assert result.actions[0].realized_pnl > 0
    assert all(p.symbol != "MSFT" for p in result.state.positions)


def test_trailing_stop_sells_after_peak_pullback(
    wallet_service: WalletService,
    stub_provider: StubPriceProvider,
    risk_service: PortfolioService,
) -> None:
    """A pullback from the high-water mark beyond the trailing stop auto-sells."""
    wallet_service.deposit("acct", 1000.0, _card(), save_card=False)
    risk_service.invest("acct", [AllocationItem(symbol="JPM", amount=200.0)])
    risk_service.set_risk_policy("acct", RiskPolicy(trailing_stop_pct=15.0))
    entry = _entry_price(risk_service, "acct", "JPM")

    # Rally to a new peak (recorded as the high-water mark on the marked read).
    stub_provider.set_price("JPM", entry * 1.40)
    risk_service.get_state("acct")  # ratchets the HWM up to 1.40 * entry

    # Pull back 20% from the peak (still above entry, but past the 15% trail).
    stub_provider.set_price("JPM", entry * 1.40 * 0.80)
    result = risk_service.evaluate_risk("acct")
    assert result.triggered is True
    assert result.actions[0].action == "trailing_stop"
    assert all(p.symbol != "JPM" for p in result.state.positions)


def test_max_drawdown_reduces_exposure(
    wallet_service: WalletService,
    stub_provider: StubPriceProvider,
    risk_service: PortfolioService,
) -> None:
    """A portfolio drawdown breach sells worst-first down to a cash floor.

    Three positions, fully invested (no spare cash), so a crash is a real
    portfolio drawdown. The circuit-breaker reduces exposure to a defensive
    cash floor (selling the worst first), not a full liquidation — at least one
    position is expected to remain.
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
    # Establish the peak at the entry marks.
    risk_service.get_state("acct")
    risk_service.set_risk_policy("acct", RiskPolicy(max_drawdown_pct=20.0))

    aapl_entry = _entry_price(risk_service, "acct", "AAPL")
    msft_entry = _entry_price(risk_service, "acct", "MSFT")
    jpm_entry = _entry_price(risk_service, "acct", "JPM")
    # Crash AAPL hard (the worst), nick MSFT/JPM slightly. Total value falls
    # well past the 20% drawdown limit.
    stub_provider.set_price("AAPL", aapl_entry * 0.30)
    stub_provider.set_price("MSFT", msft_entry * 0.95)
    stub_provider.set_price("JPM", jpm_entry * 0.95)

    cash_before = risk_service.get_state("acct").wallet.cash_balance
    result = risk_service.evaluate_risk("acct")
    assert result.triggered is True
    # The worst position (AAPL) was the first sold to reduce exposure.
    drawdown_actions = [a for a in result.actions if a.action == "drawdown"]
    assert drawdown_actions
    assert drawdown_actions[0].symbol == "AAPL"
    assert all(p.symbol != "AAPL" for p in result.state.positions)
    # De-risk, not full liquidation: at least one position survives the floor.
    assert result.state.positions, (
        "drawdown breaker should keep a cash floor, not liquidate all"
    )
    # Exposure reduced: cash rose.
    assert result.state.wallet.cash_balance > cash_before


# ---------------------------------------------------------------------------
# History service: total + per-position series of correct length and shape
# ---------------------------------------------------------------------------


def test_history_returns_total_and_position_series(
    wallet_service: WalletService, portfolio_service: PortfolioService, store: AccountStore
) -> None:
    """History returns a total series of the requested length plus per-position
    series of matching length and shape."""
    from app.invest.history import PortfolioHistoryService

    wallet_service.deposit("acct", 1000.0, _card(), save_card=False)
    portfolio_service.invest(
        "acct",
        [AllocationItem(symbol=s, amount=200.0) for s in SAMPLE_SYMBOLS],
    )

    history = PortfolioHistoryService(store, get_provider()).portfolio_history(
        "acct", points=30
    )
    # Total series has exactly the requested number of points.
    assert len(history.total) == 30
    first = history.total[0]
    assert first.total_value == pytest.approx(first.cash + first.invested, abs=0.02)

    # One per-position series per held symbol, each the same length.
    assert {p.symbol for p in history.positions} == set(SAMPLE_SYMBOLS)
    for series in history.positions:
        assert len(series.points) == 30
        pt = series.points[0]
        # Shape: t (ms), value, pnl, pnlPct — all present and finite.
        assert pt.t > 0
        for value in (pt.value, pt.pnl, pt.pnl_pct):
            assert value == value  # not NaN
            assert value not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# Advisor: weights sum ~1, amounts ~= request, exactly 5 horizons
# ---------------------------------------------------------------------------


def test_advisor_weights_amounts_and_horizons() -> None:
    """Advisor reconciles risky legs + cash to ~1, amounts to the request, 5 horizons.

    Uses the *conservative* profile (only 4 picks) to keep the analysis cheap.
    The C3 fix means the advisor is no longer blindly 100% invested: a cash
    sleeve holds the undeployed remainder, so the per-item risky weights plus
    ``cash_weight`` sum to ~1 (and the item dollars plus ``cash_amount`` sum to
    the requested amount).
    """
    from app.api.recommendations import get_engine
    from app.invest.advisor import AllocationAdvisor

    amount = 1000.0
    advice = AllocationAdvisor(get_engine(), get_provider()).advise(
        amount, "conservative"
    )

    assert advice.items, "advisor should return at least one allocation leg"
    assert len(advice.items) <= 4  # conservative pick cap

    # Risky legs + cash sleeve reconcile to the full book (no longer 100% risky).
    weight_sum = sum(it.weight for it in advice.items)
    assert 0.0 < weight_sum <= 1.0 + 1e-9
    assert weight_sum + advice.cash_weight == pytest.approx(1.0, abs=1e-2)

    amount_sum = sum(it.amount for it in advice.items)
    assert amount_sum + advice.cash_amount == pytest.approx(amount, abs=1.0)

    # No $0.00 dust legs (C5): every emitted leg carries a positive amount.
    for it in advice.items:
        assert it.amount > 0.0
        assert it.weight > 0.0

    # Exactly five horizons in the canonical set.
    assert len(advice.horizons) == 5
    assert {h.horizon for h in advice.horizons} == {"1D", "1W", "1M", "1Y", "5Y"}

    # Echoes the request, and every blended stat is finite.
    assert advice.amount == pytest.approx(amount)
    assert advice.risk_tolerance == "conservative"
    for value in (advice.expected_return, advice.expected_vol, advice.sharpe):
        assert value == value  # not NaN
        assert value not in (float("inf"), float("-inf"))


# ---------------------------------------------------------------------------
# API smoke: every new route, camelCase wire, mounted under /api
# ---------------------------------------------------------------------------


def test_api_get_wallet(client: TestClient) -> None:
    """GET /api/wallet returns 200 with a reconciled camelCase Wallet."""
    resp = client.get("/api/wallet", headers=_headers())
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "accountId",
        "cashBalance",
        "investedValue",
        "totalValue",
        "currency",
        "savedCards",
    ):
        assert key in body
    assert body["totalValue"] == pytest.approx(
        body["cashBalance"] + body["investedValue"], abs=0.02
    )


def test_api_deposit_then_balance(client: TestClient) -> None:
    """POST /api/wallet/deposit credits cash and returns wallet + transaction."""
    headers = _headers()
    payload = {
        "amount": 500.0,
        "card": {
            "number": VALID_VISA,
            "expMonth": EXP_MONTH,
            "expYear": EXP_YEAR,
            "cvc": "123",
            "holder": "Ada Lovelace",
        },
        "saveCard": True,
    }
    resp = client.post("/api/wallet/deposit", json=payload, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "wallet" in body and "transaction" in body
    assert body["wallet"]["cashBalance"] == pytest.approx(500.0)
    assert body["transaction"]["type"] == "deposit"
    # The saved card is masked (camelCase keys, no raw PAN).
    saved = body["wallet"]["savedCards"]
    assert len(saved) == 1
    assert saved[0]["last4"] == "1111"
    assert VALID_VISA not in resp.text


def test_api_deposit_invalid_card_400(client: TestClient) -> None:
    """A deposit with a Luhn-invalid card returns 400."""
    payload = {
        "amount": 100.0,
        "card": {
            "number": INVALID_PAN,
            "expMonth": EXP_MONTH,
            "expYear": EXP_YEAR,
            "cvc": "123",
            "holder": "Ada Lovelace",
        },
        "saveCard": False,
    }
    resp = client.post("/api/wallet/deposit", json=payload, headers=_headers())
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_api_withdraw_over_balance_400(client: TestClient) -> None:
    """Withdrawing more than the cash balance returns 400."""
    headers = _headers()
    deposit = {
        "amount": 100.0,
        "card": {
            "number": VALID_VISA,
            "expMonth": EXP_MONTH,
            "expYear": EXP_YEAR,
            "cvc": "123",
            "holder": "Ada Lovelace",
        },
        "saveCard": False,
    }
    client.post("/api/wallet/deposit", json=deposit, headers=headers)
    resp = client.post(
        "/api/wallet/withdraw", json={"amount": 500.0}, headers=headers
    )
    assert resp.status_code == 400


def test_api_withdraw_ok(client: TestClient) -> None:
    """A valid withdrawal returns 200 with wallet + transaction."""
    headers = _headers()
    deposit = {
        "amount": 300.0,
        "card": {
            "number": VALID_VISA,
            "expMonth": EXP_MONTH,
            "expYear": EXP_YEAR,
            "cvc": "123",
            "holder": "Ada Lovelace",
        },
        "saveCard": False,
    }
    client.post("/api/wallet/deposit", json=deposit, headers=headers)
    resp = client.post(
        "/api/wallet/withdraw",
        json={"amount": 100.0, "destination": "bank"},
        headers=headers,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["wallet"]["cashBalance"] == pytest.approx(200.0)
    assert body["transaction"]["type"] == "withdrawal"


def test_api_cards_list_and_delete(client: TestClient) -> None:
    """GET /api/wallet/cards lists saved cards; DELETE removes one (404 if absent)."""
    headers = _headers()
    deposit = {
        "amount": 100.0,
        "card": {
            "number": VALID_VISA,
            "expMonth": EXP_MONTH,
            "expYear": EXP_YEAR,
            "cvc": "123",
            "holder": "Ada Lovelace",
        },
        "saveCard": True,
    }
    client.post("/api/wallet/deposit", json=deposit, headers=headers)

    listed = client.get("/api/wallet/cards", headers=headers)
    assert listed.status_code == 200
    cards = listed.json()
    assert len(cards) == 1
    card_id = cards[0]["id"]
    assert cards[0]["last4"] == "1111"

    ok = client.delete(f"/api/wallet/cards/{card_id}", headers=headers)
    assert ok.status_code == 200
    assert ok.json()["ok"] is True

    # Deleting an unknown card returns 404.
    missing = client.delete("/api/wallet/cards/does-not-exist", headers=headers)
    assert missing.status_code == 404


def test_api_transactions_newest_first(client: TestClient) -> None:
    """GET /api/wallet/transactions returns camelCase ledger entries, newest first."""
    headers = _headers()
    deposit = {
        "amount": 200.0,
        "card": {
            "number": VALID_VISA,
            "expMonth": EXP_MONTH,
            "expYear": EXP_YEAR,
            "cvc": "123",
            "holder": "Ada Lovelace",
        },
        "saveCard": False,
    }
    client.post("/api/wallet/deposit", json=deposit, headers=headers)
    client.post("/api/wallet/withdraw", json={"amount": 50.0}, headers=headers)

    resp = client.get("/api/wallet/transactions", headers=headers)
    assert resp.status_code == 200
    txns = resp.json()
    assert len(txns) == 2
    # Newest first: the withdrawal (recorded second) leads.
    assert txns[0]["type"] == "withdrawal"
    assert txns[1]["type"] == "deposit"
    for key in ("id", "type", "amount", "status", "createdAt", "ref", "note"):
        assert key in txns[0]


def test_api_get_portfolio(client: TestClient) -> None:
    """GET /api/portfolio returns 200 with a camelCase PortfolioState."""
    resp = client.get("/api/portfolio", headers=_headers())
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "wallet",
        "positions",
        "totalCost",
        "totalValue",
        "totalPnl",
        "totalPnlPct",
    ):
        assert key in body
    assert isinstance(body["positions"], list)


def test_api_invest_and_sell_round_trip(client: TestClient) -> None:
    """POST /api/portfolio/invest then /sell return camelCase PortfolioState."""
    headers = _headers()
    deposit = {
        "amount": 1000.0,
        "card": {
            "number": VALID_VISA,
            "expMonth": EXP_MONTH,
            "expYear": EXP_YEAR,
            "cvc": "123",
            "holder": "Ada Lovelace",
        },
        "saveCard": False,
    }
    client.post("/api/wallet/deposit", json=deposit, headers=headers)

    invest_body = {
        "allocations": [{"symbol": s, "amount": 200.0} for s in SAMPLE_SYMBOLS]
    }
    invested = client.post(
        "/api/portfolio/invest", json=invest_body, headers=headers
    )
    assert invested.status_code == 200
    state = invested.json()
    held = {p["symbol"] for p in state["positions"]}
    assert held == set(SAMPLE_SYMBOLS)
    assert state["totalCost"] == pytest.approx(600.0, abs=0.01)
    # Each position carries the camelCase shape.
    pos = state["positions"][0]
    for key in (
        "symbol",
        "asset",
        "units",
        "costBasis",
        "avgPrice",
        "currentPrice",
        "marketValue",
        "unrealizedPnl",
        "unrealizedPnlPct",
        "allocationPct",
        "realizedPnl",
        "openedAt",
    ):
        assert key in pos

    sold = client.post(
        "/api/portfolio/sell",
        json={"symbol": "AAPL", "amount": None, "all": True},
        headers=headers,
    )
    assert sold.status_code == 200
    assert all(p["symbol"] != "AAPL" for p in sold.json()["positions"])


def test_api_invest_over_spend_400(client: TestClient) -> None:
    """An invest order exceeding available cash returns 400."""
    headers = _headers()
    resp = client.post(
        "/api/portfolio/invest",
        json={"allocations": [{"symbol": "AAPL", "amount": 5000.0}]},
        headers=headers,
    )
    assert resp.status_code == 400


def test_api_invest_unknown_symbol_404(client: TestClient) -> None:
    """Investing in an unknown symbol returns 404."""
    headers = _headers()
    deposit = {
        "amount": 100.0,
        "card": {
            "number": VALID_VISA,
            "expMonth": EXP_MONTH,
            "expYear": EXP_YEAR,
            "cvc": "123",
            "holder": "Ada Lovelace",
        },
        "saveCard": False,
    }
    client.post("/api/wallet/deposit", json=deposit, headers=headers)
    resp = client.post(
        "/api/portfolio/invest",
        json={"allocations": [{"symbol": "ZZZZ_NOPE", "amount": 10.0}]},
        headers=headers,
    )
    assert resp.status_code == 404


def test_api_sell_unknown_position_404(client: TestClient) -> None:
    """Selling a symbol with no open position returns 404."""
    headers = _headers()
    resp = client.post(
        "/api/portfolio/sell",
        json={"symbol": "AAPL", "amount": None, "all": True},
        headers=headers,
    )
    assert resp.status_code == 404


def test_api_portfolio_history(client: TestClient) -> None:
    """GET /api/portfolio/history returns the camelCase total + position curves."""
    headers = _headers()
    deposit = {
        "amount": 1000.0,
        "card": {
            "number": VALID_VISA,
            "expMonth": EXP_MONTH,
            "expYear": EXP_YEAR,
            "cvc": "123",
            "holder": "Ada Lovelace",
        },
        "saveCard": False,
    }
    client.post("/api/wallet/deposit", json=deposit, headers=headers)
    client.post(
        "/api/portfolio/invest",
        json={"allocations": [{"symbol": "AAPL", "amount": 200.0}]},
        headers=headers,
    )

    resp = client.get(
        "/api/portfolio/history", params={"points": 20}, headers=headers
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["total"]) == 20
    tp = body["total"][0]
    for key in ("t", "totalValue", "invested", "cash"):
        assert key in tp
    assert body["positions"]
    series = body["positions"][0]
    assert series["symbol"] == "AAPL"
    assert len(series["points"]) == 20
    for key in ("t", "value", "pnl", "pnlPct"):
        assert key in series["points"][0]


def test_api_risk_policy_get_default_off(client: TestClient) -> None:
    """GET /api/portfolio/risk returns an all-OFF policy by default (camelCase)."""
    resp = client.get("/api/portfolio/risk", headers=_headers())
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "stopLossPct",
        "trailingStopPct",
        "takeProfitPct",
        "maxDrawdownPct",
    ):
        assert key in body
        assert body[key] is None


def test_api_risk_policy_put_round_trip(client: TestClient) -> None:
    """PUT /api/portfolio/risk stores the policy; GET reads it back."""
    headers = _headers()
    put = client.put(
        "/api/portfolio/risk",
        json={"stopLossPct": 10.0, "takeProfitPct": 40.0},
        headers=headers,
    )
    assert put.status_code == 200
    assert put.json()["stopLossPct"] == pytest.approx(10.0)
    assert put.json()["takeProfitPct"] == pytest.approx(40.0)
    assert put.json()["trailingStopPct"] is None

    got = client.get("/api/portfolio/risk", headers=headers)
    assert got.json()["stopLossPct"] == pytest.approx(10.0)


def test_api_risk_policy_put_invalid_400(client: TestClient) -> None:
    """A non-positive threshold on PUT /api/portfolio/risk returns 400."""
    resp = client.put(
        "/api/portfolio/risk",
        json={"stopLossPct": 0.0},
        headers=_headers(),
    )
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_api_risk_apply_noop_returns_state(client: TestClient) -> None:
    """POST /api/portfolio/risk/apply with an OFF policy returns an empty result."""
    headers = _headers()
    deposit = {
        "amount": 1000.0,
        "card": {
            "number": VALID_VISA,
            "expMonth": EXP_MONTH,
            "expYear": EXP_YEAR,
            "cvc": "123",
            "holder": "Ada Lovelace",
        },
        "saveCard": False,
    }
    client.post("/api/wallet/deposit", json=deposit, headers=headers)
    client.post(
        "/api/portfolio/invest",
        json={"allocations": [{"symbol": "AAPL", "amount": 200.0}]},
        headers=headers,
    )

    resp = client.post("/api/portfolio/risk/apply", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    for key in ("actions", "policy", "state", "triggered", "disclaimer"):
        assert key in body
    assert body["triggered"] is False
    assert body["actions"] == []
    # The position is untouched and the state shape is the camelCase PortfolioState.
    held = {p["symbol"] for p in body["state"]["positions"]}
    assert "AAPL" in held
    assert "totalValue" in body["state"]


def test_api_advisor_allocate(client: TestClient) -> None:
    """POST /api/advisor/allocate returns 200 with a camelCase AllocationAdvice.

    Uses the conservative profile (4 picks) to keep the analysis cheap.
    """
    resp = client.post(
        "/api/advisor/allocate",
        json={"amount": 1000.0, "riskTolerance": "conservative"},
        headers=_headers(),
    )
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "items",
        "expectedReturn",
        "expectedVol",
        "sharpe",
        "horizons",
        "riskTolerance",
        "amount",
    ):
        assert key in body
    assert body["amount"] == pytest.approx(1000.0)
    assert body["riskTolerance"] == "conservative"
    assert len(body["horizons"]) == 5
    assert body["items"]
    # Per-item camelCase keys (note the explicit expectedReturn1YPct alias).
    item = body["items"][0]
    for key in (
        "asset",
        "weight",
        "amount",
        "compositeScore",
        "expectedReturn1YPct",
        "rationale",
    ):
        assert key in item
    # Risky legs + cash sleeve reconcile to the full book (C3: not 100% invested).
    weight_sum = sum(it["weight"] for it in body["items"])
    assert 0.0 < weight_sum <= 1.0 + 1e-9
    assert weight_sum + body["cashWeight"] == pytest.approx(1.0, abs=1e-2)
    amount_sum = sum(it["amount"] for it in body["items"])
    assert amount_sum + body["cashAmount"] == pytest.approx(1000.0, abs=1.0)
    # Honesty flags: results are synthetic; no infeasible target was requested.
    assert body["syntheticData"] is True
    assert body["targetWarning"] is None


def test_api_advisor_flags_infeasible_target(client: TestClient) -> None:
    """A 100x-in-two-months ask is flagged with a targetWarning (still 200)."""
    resp = client.post(
        "/api/advisor/allocate",
        json={
            "amount": 100.0,
            "riskTolerance": "conservative",
            "targetAmount": 10_000.0,
            "horizonDays": 61,
        },
        headers=_headers(),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["syntheticData"] is True
    assert body["targetWarning"]  # non-empty warning string
    assert "every day" in body["targetWarning"].lower()


def test_api_advisor_amount_non_positive_400(client: TestClient) -> None:
    """A non-positive advisor amount returns 400."""
    resp = client.post(
        "/api/advisor/allocate",
        json={"amount": 0.0, "riskTolerance": "balanced"},
        headers=_headers(),
    )
    assert resp.status_code == 400


def test_invest_router_mounted_under_api(client: TestClient) -> None:
    """The invest router is mounted under /api (its routes resolve, not 404-by-path)."""
    paths = {route.path for route in app.routes}
    for path in (
        "/api/wallet",
        "/api/wallet/deposit",
        "/api/wallet/withdraw",
        "/api/wallet/cards",
        "/api/wallet/cards/{card_id}",
        "/api/wallet/transactions",
        "/api/portfolio/invest",
        "/api/portfolio/sell",
        "/api/portfolio/history",
        "/api/portfolio/risk",
        "/api/portfolio/risk/apply",
        "/api/advisor/allocate",
    ):
        assert path in paths, f"expected {path} to be mounted"
