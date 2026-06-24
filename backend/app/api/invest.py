"""``/api`` — the simulated brokerage: wallet, portfolio, history and advisor.

This router is the HTTP front for the in-memory paper-trading sandbox documented
in ``docs/INVEST.md``. It is a thin, defensive adapter over the invest services:

    * :class:`~app.invest.wallet.WalletService` — cash, cards, ledger.
    * :class:`~app.invest.portfolio_service.PortfolioService` — buy/sell + the
      mark-to-market portfolio view.
    * :class:`~app.invest.history.PortfolioHistoryService` — backfilled value/P&L
      curves for seeding the charts.
    * :class:`~app.invest.advisor.AllocationAdvisor` — "where to invest now",
      ranking via the shared :class:`~app.strategies.engine.AnalysisEngine` and
      sizing via Markowitz.

Money-handling stance: **simulated / sandbox** only. No real money moves; cards
are validated and stored masked. The account id is resolved by the shared
:func:`app.auth.deps.account_id` dependency: a valid ``Authorization: Bearer``
token maps to the caller's own account (``user:<id>``); otherwise it falls back
to the optional ``X-Account-Id`` header, then to ``'demo'``. This gives each
logged-in user an isolated wallet while keeping anonymous/sandbox access working.

Error mapping (per the contract): service ``ValueError`` → HTTP 400 (bad input /
insufficient funds / invalid card / amount ≤ 0); service ``KeyError`` → HTTP 404
(unknown symbol / saved card). Response models are the camelCase invest DTOs.

The existing ``POST /api/portfolio/optimize`` (efficient-frontier analytics) lives
in :mod:`app.api.portfolio` and is intentionally untouched here.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Path, Query
from pydantic import BaseModel, Field

from app.api.recommendations import get_engine
from app.auth.deps import account_id as account_id_dep
from app.invest.advisor import AllocationAdvisor
from app.invest.history import PortfolioHistoryService
from app.invest.payments import PaymentProvider, get_payment_provider
from app.invest.portfolio_service import PortfolioService
from app.invest.store import AccountStore, get_store
from app.invest.wallet import WalletService
from app.market.provider import MarketDataProvider, get_provider
from app.schemas import (
    AdviceRequest,
    AllocationAdvice,
    DepositRequest,
    InvestRequest,
    PortfolioHistory,
    PortfolioState,
    SavedCard,
    SellRequest,
    Transaction,
    Wallet,
    WithdrawRequest,
)

__all__ = ["router"]

router = APIRouter(prefix="/api", tags=["invest"])


# ---------------------------------------------------------------------------
# Documentation-only account header
# ---------------------------------------------------------------------------
#
# Every invest/wallet/advisor route resolves its account id via the shared
# :func:`app.auth.deps.account_id` dependency, which reads a Bearer token first
# and falls back to the ``X-Account-Id`` header (then to ``'demo'``). That
# dependency is what actually governs resolution; the alias below exists purely
# so the header surfaces in the OpenAPI/Scalar docs as a documented, optional
# parameter on each route. It is intentionally unused by the handler bodies —
# changing it does not affect resolution or validation behavior.
AccountHeader = Annotated[
    Optional[str],
    Header(
        alias="X-Account-Id",
        description=(
            "Account id; a valid Bearer token overrides it with the user id. "
            "Anonymous/sandbox callers may pass any opaque id to get an isolated "
            "wallet, or omit it to use the shared 'demo' account."
        ),
        examples=["demo", "user:9f3c1a2b"],
    ),
]


# ---------------------------------------------------------------------------
# Response envelopes (deposit / withdraw return wallet + transaction together)
# ---------------------------------------------------------------------------


class _CamelEnvelope(BaseModel):
    """Base envelope serializing to camelCase (matches the invest DTOs)."""

    model_config = {"populate_by_name": True}


class WalletTxnResponse(_CamelEnvelope):
    """Response for deposit / withdraw: the updated wallet plus its transaction.

    Fields:
        wallet: The reconciled :class:`~app.schemas.Wallet` after the operation.
        transaction: The completed :class:`~app.schemas.Transaction` recorded.
    """

    wallet: Wallet
    transaction: Transaction


class OkResponse(_CamelEnvelope):
    """Trivial ``{ "ok": true }`` acknowledgement (e.g. after deleting a card)."""

    ok: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Service wiring (built per-request from the process-wide singletons)
# ---------------------------------------------------------------------------


def _store() -> AccountStore:
    """Return the process-wide :class:`~app.invest.store.AccountStore`."""
    return get_store()


def _provider() -> MarketDataProvider:
    """Return the process-wide :class:`~app.market.provider.MarketDataProvider`."""
    return get_provider()


def _payments() -> PaymentProvider:
    """Return the process-wide :class:`~app.invest.payments.PaymentProvider`."""
    return get_payment_provider()


def _wallet_service() -> WalletService:
    """Build a :class:`~app.invest.wallet.WalletService` over the singletons."""
    return WalletService(_store(), _payments(), _provider())


def _portfolio_service() -> PortfolioService:
    """Build a :class:`~app.invest.portfolio_service.PortfolioService`."""
    return PortfolioService(_store(), _provider())


def _history_service() -> PortfolioHistoryService:
    """Build a :class:`~app.invest.history.PortfolioHistoryService`."""
    return PortfolioHistoryService(_store(), _provider())


def _advisor() -> AllocationAdvisor:
    """Build an :class:`~app.invest.advisor.AllocationAdvisor`.

    Reuses the shared :func:`~app.api.recommendations.get_engine` singleton so the
    advisor benefits from the engine's warm per-symbol analysis cache.
    """
    return AllocationAdvisor(get_engine(), _provider())


# ---------------------------------------------------------------------------
# Wallet routes
# ---------------------------------------------------------------------------


@router.get(
    "/wallet",
    response_model=Wallet,
    summary="Get the account wallet",
    tags=["wallet"],
    description=(
        "Return the reconciled wallet snapshot for the resolved account: cash "
        "balance, mark-to-market invested value, total value (`cash + "
        "invested`), currency, and any masked saved cards.\n\n"
        "The account is selected by the `X-Account-Id` header, overridden by a "
        "valid Bearer token (which maps to the caller's own `user:<id>` "
        "account), and otherwise defaults to `demo`."
    ),
    responses={200: {"description": "The reconciled wallet snapshot."}},
)
def get_wallet(
    x_account_id: AccountHeader = "demo",
    account_id: str = Depends(account_id_dep),
) -> Wallet:
    """Return the reconciled wallet snapshot for the account.

    Args:
        x_account_id: Documentation-only echo of the ``X-Account-Id`` header;
            the shared dependency performs the real account resolution.
        account_id: The resolved account id (``user:<id>`` for a logged-in
            caller, else the ``X-Account-Id`` header, else ``'demo'``).

    Returns:
        The current :class:`~app.schemas.Wallet` (cash + invested == total).
    """
    return _wallet_service().get_wallet(account_id)


@router.post(
    "/wallet/deposit",
    response_model=WalletTxnResponse,
    summary="Deposit funds via a (validated, never-stored-raw) card",
    tags=["wallet"],
    description=(
        "Fund the account with a simulated card charge and return the updated "
        "wallet together with the recorded transaction.\n\n"
        "The card is validated (Luhn check, brand detection, future expiry) but "
        "**never stored raw** — set `saveCard: true` to keep a masked token for "
        "reuse, or pass an existing `savedCardId` to charge a card already on "
        "file. `amount` is in account currency (USD) and must be `> 0`.\n\n"
        "This is a sandbox wallet: no real money moves.\n\n"
        "**Status codes**\n"
        "- `200` — funds deposited; wallet + transaction returned.\n"
        "- `400` — invalid card or non-positive amount."
    ),
    responses={
        200: {"description": "Funds deposited; updated wallet + transaction."},
        400: {"description": "Invalid card or non-positive amount."},
    },
)
def deposit(
    body: DepositRequest,
    x_account_id: AccountHeader = "demo",
    account_id: str = Depends(account_id_dep),
) -> WalletTxnResponse:
    """Fund the account (simulated charge) and return wallet + transaction.

    Args:
        body: The :class:`~app.schemas.DepositRequest` (amount, card, saveCard).
        x_account_id: Documentation-only echo of the ``X-Account-Id`` header.
        account_id: The resolved account id (token / header / ``'demo'``).

    Returns:
        A :class:`WalletTxnResponse` with the updated wallet and the deposit txn.

    Raises:
        HTTPException: ``400`` for an invalid card or amount.
    """
    try:
        wallet, txn = _wallet_service().deposit(
            account_id,
            body.amount,
            body.card,
            save_card=body.save_card,
            saved_card_id=body.saved_card_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WalletTxnResponse(wallet=wallet, transaction=txn)


@router.post(
    "/wallet/withdraw",
    response_model=WalletTxnResponse,
    summary="Withdraw cash to an external destination (simulated)",
    tags=["wallet"],
    description=(
        "Pay cash out of the account (simulated) and return the updated wallet "
        "plus the payout transaction.\n\n"
        "`amount` must be `> 0` and no greater than the available cash balance. "
        "`destination` is an optional free-text payout label recorded on the "
        "transaction note.\n\n"
        "**Status codes**\n"
        "- `200` — cash paid out; wallet + transaction returned.\n"
        "- `400` — non-positive amount or insufficient funds."
    ),
    responses={
        200: {"description": "Cash paid out; updated wallet + transaction."},
        400: {"description": "Non-positive amount or insufficient funds."},
    },
)
def withdraw(
    body: WithdrawRequest,
    x_account_id: AccountHeader = "demo",
    account_id: str = Depends(account_id_dep),
) -> WalletTxnResponse:
    """Pay cash out of the account and return wallet + transaction.

    Args:
        body: The :class:`~app.schemas.WithdrawRequest` (amount, destination).
        x_account_id: Documentation-only echo of the ``X-Account-Id`` header.
        account_id: The resolved account id (token / header / ``'demo'``).

    Returns:
        A :class:`WalletTxnResponse` with the updated wallet and the payout txn.

    Raises:
        HTTPException: ``400`` for a non-positive amount or insufficient funds.
    """
    try:
        wallet, txn = _wallet_service().withdraw(
            account_id,
            body.amount,
            destination=body.destination,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WalletTxnResponse(wallet=wallet, transaction=txn)


@router.get(
    "/wallet/cards",
    response_model=list[SavedCard],
    summary="List the masked saved cards on file",
    tags=["wallet"],
    description=(
        "List the account's saved cards. Each entry is fully masked — only the "
        "brand, last four digits, expiry and holder name keyed by an opaque "
        "token id. Raw PAN/CVC are never stored or returned."
    ),
    responses={200: {"description": "The account's masked saved cards."}},
)
def list_cards(
    x_account_id: AccountHeader = "demo",
    account_id: str = Depends(account_id_dep),
) -> list[SavedCard]:
    """Return the account's masked saved cards (never raw PAN/CVC).

    Args:
        x_account_id: Documentation-only echo of the ``X-Account-Id`` header.
        account_id: The resolved account id (token / header / ``'demo'``).

    Returns:
        A list of :class:`~app.schemas.SavedCard`.
    """
    return _wallet_service().list_cards(account_id)


@router.delete(
    "/wallet/cards/{card_id}",
    response_model=OkResponse,
    summary="Delete a saved card",
    tags=["wallet"],
    description=(
        "Remove a saved card from the account by its opaque token id (the `id` "
        "field returned by `GET /api/wallet/cards`). Returns `{ \"ok\": true }` "
        "on success.\n\n"
        "**Status codes**\n"
        "- `200` — card removed.\n"
        "- `404` — no saved card has that id."
    ),
    responses={
        200: {"description": "Card removed."},
        404: {"description": "No saved card has that id."},
    },
)
def delete_card(
    card_id: str = Path(
        ...,
        description="The saved card's opaque token id.",
        examples=["card_123"],
    ),
    x_account_id: AccountHeader = "demo",
    account_id: str = Depends(account_id_dep),
) -> OkResponse:
    """Remove a saved card from the account.

    Args:
        card_id: The opaque token id of the card to remove.
        x_account_id: Documentation-only echo of the ``X-Account-Id`` header.
        account_id: The resolved account id (token / header / ``'demo'``).

    Returns:
        ``{ "ok": true }`` on success.

    Raises:
        HTTPException: ``404`` if no saved card has that id.
    """
    try:
        _wallet_service().delete_card(account_id, card_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return OkResponse(ok=True)


@router.get(
    "/wallet/transactions",
    response_model=list[Transaction],
    summary="List the account ledger (newest first)",
    tags=["wallet"],
    description=(
        "List the account's full transaction ledger, newest first. Entries "
        "cover cash movements (`deposit`/`withdrawal`) and trades "
        "(`buy`/`sell`); `buy`/`sell` entries carry a `symbol`."
    ),
    responses={200: {"description": "The account ledger, newest first."}},
)
def list_transactions(
    x_account_id: AccountHeader = "demo",
    account_id: str = Depends(account_id_dep),
) -> list[Transaction]:
    """Return the account's transactions, newest first.

    Args:
        x_account_id: Documentation-only echo of the ``X-Account-Id`` header.
        account_id: The resolved account id (token / header / ``'demo'``).

    Returns:
        A list of :class:`~app.schemas.Transaction` ordered most-recent first.
    """
    return _wallet_service().list_transactions(account_id)


# ---------------------------------------------------------------------------
# Portfolio routes
# ---------------------------------------------------------------------------


@router.get(
    "/portfolio",
    response_model=PortfolioState,
    summary="Get the mark-to-market portfolio state",
    tags=["portfolio"],
    description=(
        "Return the full mark-to-market portfolio view: the embedded wallet, "
        "every open position (units, cost basis, current mark, unrealized P&L "
        "and allocation share), and portfolio-level cost/value/P&L totals.\n\n"
        "Note this `GET /api/portfolio` (the simulated brokerage view) is "
        "distinct from `POST /api/portfolio/optimize` (the Markowitz analytics "
        "endpoint)."
    ),
    responses={200: {"description": "The mark-to-market portfolio state."}},
)
def get_portfolio(
    x_account_id: AccountHeader = "demo",
    account_id: str = Depends(account_id_dep),
) -> PortfolioState:
    """Return the full mark-to-market portfolio view for the account.

    Args:
        x_account_id: Documentation-only echo of the ``X-Account-Id`` header.
        account_id: The resolved account id (token / header / ``'demo'``).

    Returns:
        The current :class:`~app.schemas.PortfolioState`.
    """
    return _portfolio_service().get_state(account_id)


@router.post(
    "/portfolio/invest",
    response_model=PortfolioState,
    summary="Spend cash across one or more symbols",
    tags=["portfolio"],
    description=(
        "Spend cash across one or more symbols, opening new positions or adding "
        "to existing ones, and return the updated portfolio state.\n\n"
        "Each `allocations` leg spends `amount` dollars on `symbol`. The whole "
        "order is validated up front and applied **all-or-nothing**: if any leg "
        "is invalid (unknown symbol, non-positive amount) or total cash is "
        "insufficient, nothing is purchased.\n\n"
        "**Status codes**\n"
        "- `200` — order filled; updated portfolio returned.\n"
        "- `400` — empty order, a non-positive leg, or insufficient funds.\n"
        "- `404` — an unknown symbol in the order."
    ),
    responses={
        200: {"description": "Order filled; updated portfolio state."},
        400: {"description": "Empty order, non-positive leg, or insufficient funds."},
        404: {"description": "Unknown symbol in the order."},
    },
)
def invest(
    body: InvestRequest,
    x_account_id: AccountHeader = "demo",
    account_id: str = Depends(account_id_dep),
) -> PortfolioState:
    """Split cash across the requested symbols, opening/adding positions.

    The whole order is validated before any mutation (all-or-nothing).

    Args:
        body: The :class:`~app.schemas.InvestRequest` (allocation legs).
        x_account_id: Documentation-only echo of the ``X-Account-Id`` header.
        account_id: The resolved account id (token / header / ``'demo'``).

    Returns:
        The updated :class:`~app.schemas.PortfolioState`.

    Raises:
        HTTPException: ``400`` for an empty order, a non-positive leg, or
            insufficient funds; ``404`` for an unknown symbol.
    """
    try:
        return _portfolio_service().invest(account_id, body.allocations)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/portfolio/sell",
    response_model=PortfolioState,
    summary="Reduce or liquidate a position",
    tags=["portfolio"],
    description=(
        "Sell part or all of a held position, realizing P&L and crediting the "
        "proceeds back to cash; returns the updated portfolio state.\n\n"
        "Set `all: true` to liquidate the whole position (`amount` is then "
        "ignored), or pass a positive dollar `amount` to partially reduce it.\n\n"
        "**Status codes**\n"
        "- `200` — sale executed; updated portfolio returned.\n"
        "- `400` — missing or non-positive `amount` when `all` is not set.\n"
        "- `404` — no open position for the symbol."
    ),
    responses={
        200: {"description": "Sale executed; updated portfolio state."},
        400: {"description": "Missing or non-positive amount when not selling all."},
        404: {"description": "No open position for the symbol."},
    },
)
def sell(
    body: SellRequest,
    x_account_id: AccountHeader = "demo",
    account_id: str = Depends(account_id_dep),
) -> PortfolioState:
    """Sell part or all of a position, realizing P&L and crediting cash.

    Args:
        body: The :class:`~app.schemas.SellRequest` (symbol, amount or all).
        x_account_id: Documentation-only echo of the ``X-Account-Id`` header.
        account_id: The resolved account id (token / header / ``'demo'``).

    Returns:
        The updated :class:`~app.schemas.PortfolioState`.

    Raises:
        HTTPException: ``400`` for a missing/non-positive amount when not selling
            all; ``404`` if there is no open position for the symbol.
    """
    try:
        return _portfolio_service().sell(
            account_id,
            body.symbol,
            amount=body.amount,
            sell_all=body.all,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/portfolio/history",
    response_model=PortfolioHistory,
    summary="Backfilled portfolio value / P&L time series",
    tags=["portfolio"],
    description=(
        "Return a backfilled recent-window time series for seeding the "
        "portfolio charts: a total value/invested/cash curve plus one "
        "value/P&L curve per held, priceable position.\n\n"
        "`points` controls how many trailing daily steps to reconstruct "
        "(1..2000, default 120). Timestamps `t` are unix milliseconds."
    ),
    responses={
        200: {"description": "Backfilled total + per-position value/P&L curves."}
    },
)
def portfolio_history(
    points: int = Query(
        default=120,
        ge=1,
        le=2000,
        description="Number of trailing daily points to backfill (1..2000).",
        examples=[120],
    ),
    x_account_id: AccountHeader = "demo",
    account_id: str = Depends(account_id_dep),
) -> PortfolioHistory:
    """Return the backfilled total + per-position value/P&L curves.

    Args:
        points: Number of trailing daily points to reconstruct (1..2000).
        x_account_id: Documentation-only echo of the ``X-Account-Id`` header.
        account_id: The resolved account id (token / header / ``'demo'``).

    Returns:
        A :class:`~app.schemas.PortfolioHistory` with a length-``points`` total
        series and one per-position series per held, priceable symbol.
    """
    return _history_service().portfolio_history(account_id, points=points)


# ---------------------------------------------------------------------------
# Advisor route
# ---------------------------------------------------------------------------


@router.post(
    "/advisor/allocate",
    response_model=AllocationAdvice,
    summary="Recommend how to allocate an amount at a risk profile",
    tags=["advisor"],
    description=(
        "Recommend a Markowitz-sized basket for a given dollar `amount` and "
        "`riskTolerance` (`conservative` | `balanced` | `aggressive`), "
        "optionally filtered to specific `assetClasses` (`equity` | `crypto` | "
        "`etf`).\n\n"
        "Returns per-asset legs (weight, dollar amount, score, 1Y expected "
        "return and a short rationale) plus the blended return/volatility/"
        "Sharpe and a 5-horizon expected-return fan for the basket. The advice "
        "is account-agnostic — the account id is accepted for symmetry only.\n\n"
        "**Status codes**\n"
        "- `200` — recommended basket returned.\n"
        "- `400` — non-positive or non-finite `amount`."
    ),
    responses={
        200: {"description": "Recommended allocation basket."},
        400: {"description": "Non-positive or non-finite amount."},
    },
)
def allocate(
    body: AdviceRequest = Body(...),
    x_account_id: AccountHeader = "demo",
    account_id: str = Depends(account_id_dep),
) -> AllocationAdvice:
    """Recommend a Markowitz-sized basket for the requested amount and risk.

    The account id is accepted for symmetry but the advice is account-agnostic
    (it depends only on the amount, risk profile and asset-class filter).

    Args:
        body: The :class:`~app.schemas.AdviceRequest` (amount, riskTolerance,
            optional assetClasses).
        x_account_id: Documentation-only echo of the ``X-Account-Id`` header.
        account_id: The resolved account id (accepted for symmetry, unused).

    Returns:
        A fully-populated :class:`~app.schemas.AllocationAdvice`.

    Raises:
        HTTPException: ``400`` for a non-positive / non-finite amount.
    """
    try:
        return _advisor().advise(
            body.amount,
            body.risk_tolerance,
            asset_classes=body.asset_classes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
