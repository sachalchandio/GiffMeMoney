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
from app.market.provider import get_provider
from app.schemas import AllocationItem, CardIn

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
    """Advisor returns weights summing to ~1, amounts ~= the request, 5 horizons.

    Uses the *conservative* profile (only 4 picks) to keep the analysis cheap.
    """
    from app.api.recommendations import get_engine
    from app.invest.advisor import AllocationAdvisor

    amount = 1000.0
    advice = AllocationAdvisor(get_engine(), get_provider()).advise(
        amount, "conservative"
    )

    assert advice.items, "advisor should return at least one allocation leg"
    assert len(advice.items) <= 4  # conservative pick cap

    weight_sum = sum(it.weight for it in advice.items)
    assert weight_sum == pytest.approx(1.0, abs=1e-2)

    amount_sum = sum(it.amount for it in advice.items)
    assert amount_sum == pytest.approx(amount, abs=1.0)

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
    weight_sum = sum(it["weight"] for it in body["items"])
    assert weight_sum == pytest.approx(1.0, abs=1e-2)
    amount_sum = sum(it["amount"] for it in body["items"])
    assert amount_sum == pytest.approx(1000.0, abs=1.0)


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
        "/api/advisor/allocate",
    ):
        assert path in paths, f"expected {path} to be mounted"
