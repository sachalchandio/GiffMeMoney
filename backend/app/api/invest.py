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
are validated and stored masked. The account id comes from the optional
``X-Account-Id`` header (default ``'demo'``); there is no auth.

Error mapping (per the contract): service ``ValueError`` → HTTP 400 (bad input /
insufficient funds / invalid card / amount ≤ 0); service ``KeyError`` → HTTP 404
(unknown symbol / saved card). Response models are the camelCase invest DTOs.

The existing ``POST /api/portfolio/optimize`` (efficient-frontier analytics) lives
in :mod:`app.api.portfolio` and is intentionally untouched here.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Header, HTTPException, Path, Query
from pydantic import BaseModel, Field

from app.api.recommendations import get_engine
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

#: Default account id when the ``X-Account-Id`` header is absent.
_DEFAULT_ACCOUNT = "demo"


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


def _account_id(x_account_id: str | None) -> str:
    """Resolve the effective account id from the optional header.

    Args:
        x_account_id: The raw ``X-Account-Id`` header value (may be ``None``).

    Returns:
        The trimmed account id, or ``'demo'`` when absent/blank.
    """
    return (x_account_id or "").strip() or _DEFAULT_ACCOUNT


# ---------------------------------------------------------------------------
# Wallet routes
# ---------------------------------------------------------------------------


@router.get("/wallet", response_model=Wallet, summary="Get the account wallet")
def get_wallet(
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
) -> Wallet:
    """Return the reconciled wallet snapshot for the account.

    Args:
        x_account_id: Optional ``X-Account-Id`` header (default ``'demo'``).

    Returns:
        The current :class:`~app.schemas.Wallet` (cash + invested == total).
    """
    return _wallet_service().get_wallet(_account_id(x_account_id))


@router.post(
    "/wallet/deposit",
    response_model=WalletTxnResponse,
    summary="Deposit funds via a (validated, never-stored-raw) card",
)
def deposit(
    body: DepositRequest,
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
) -> WalletTxnResponse:
    """Fund the account (simulated charge) and return wallet + transaction.

    Args:
        body: The :class:`~app.schemas.DepositRequest` (amount, card, saveCard).
        x_account_id: Optional ``X-Account-Id`` header.

    Returns:
        A :class:`WalletTxnResponse` with the updated wallet and the deposit txn.

    Raises:
        HTTPException: ``400`` for an invalid card or amount.
    """
    try:
        wallet, txn = _wallet_service().deposit(
            _account_id(x_account_id),
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
)
def withdraw(
    body: WithdrawRequest,
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
) -> WalletTxnResponse:
    """Pay cash out of the account and return wallet + transaction.

    Args:
        body: The :class:`~app.schemas.WithdrawRequest` (amount, destination).
        x_account_id: Optional ``X-Account-Id`` header.

    Returns:
        A :class:`WalletTxnResponse` with the updated wallet and the payout txn.

    Raises:
        HTTPException: ``400`` for a non-positive amount or insufficient funds.
    """
    try:
        wallet, txn = _wallet_service().withdraw(
            _account_id(x_account_id),
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
)
def list_cards(
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
) -> list[SavedCard]:
    """Return the account's masked saved cards (never raw PAN/CVC).

    Args:
        x_account_id: Optional ``X-Account-Id`` header.

    Returns:
        A list of :class:`~app.schemas.SavedCard`.
    """
    return _wallet_service().list_cards(_account_id(x_account_id))


@router.delete(
    "/wallet/cards/{card_id}",
    response_model=OkResponse,
    summary="Delete a saved card",
)
def delete_card(
    card_id: str = Path(..., description="The saved card's opaque token id."),
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
) -> OkResponse:
    """Remove a saved card from the account.

    Args:
        card_id: The opaque token id of the card to remove.
        x_account_id: Optional ``X-Account-Id`` header.

    Returns:
        ``{ "ok": true }`` on success.

    Raises:
        HTTPException: ``404`` if no saved card has that id.
    """
    try:
        _wallet_service().delete_card(_account_id(x_account_id), card_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return OkResponse(ok=True)


@router.get(
    "/wallet/transactions",
    response_model=list[Transaction],
    summary="List the account ledger (newest first)",
)
def list_transactions(
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
) -> list[Transaction]:
    """Return the account's transactions, newest first.

    Args:
        x_account_id: Optional ``X-Account-Id`` header.

    Returns:
        A list of :class:`~app.schemas.Transaction` ordered most-recent first.
    """
    return _wallet_service().list_transactions(_account_id(x_account_id))


# ---------------------------------------------------------------------------
# Portfolio routes
# ---------------------------------------------------------------------------


@router.get(
    "/portfolio",
    response_model=PortfolioState,
    summary="Get the mark-to-market portfolio state",
)
def get_portfolio(
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
) -> PortfolioState:
    """Return the full mark-to-market portfolio view for the account.

    Args:
        x_account_id: Optional ``X-Account-Id`` header.

    Returns:
        The current :class:`~app.schemas.PortfolioState`.
    """
    return _portfolio_service().get_state(_account_id(x_account_id))


@router.post(
    "/portfolio/invest",
    response_model=PortfolioState,
    summary="Spend cash across one or more symbols",
)
def invest(
    body: InvestRequest,
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
) -> PortfolioState:
    """Split cash across the requested symbols, opening/adding positions.

    The whole order is validated before any mutation (all-or-nothing).

    Args:
        body: The :class:`~app.schemas.InvestRequest` (allocation legs).
        x_account_id: Optional ``X-Account-Id`` header.

    Returns:
        The updated :class:`~app.schemas.PortfolioState`.

    Raises:
        HTTPException: ``400`` for an empty order, a non-positive leg, or
            insufficient funds; ``404`` for an unknown symbol.
    """
    try:
        return _portfolio_service().invest(
            _account_id(x_account_id), body.allocations
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/portfolio/sell",
    response_model=PortfolioState,
    summary="Reduce or liquidate a position",
)
def sell(
    body: SellRequest,
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
) -> PortfolioState:
    """Sell part or all of a position, realizing P&L and crediting cash.

    Args:
        body: The :class:`~app.schemas.SellRequest` (symbol, amount or all).
        x_account_id: Optional ``X-Account-Id`` header.

    Returns:
        The updated :class:`~app.schemas.PortfolioState`.

    Raises:
        HTTPException: ``400`` for a missing/non-positive amount when not selling
            all; ``404`` if there is no open position for the symbol.
    """
    try:
        return _portfolio_service().sell(
            _account_id(x_account_id),
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
)
def portfolio_history(
    points: int = Query(
        default=120,
        ge=1,
        le=2000,
        description="Number of trailing daily points to backfill.",
    ),
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
) -> PortfolioHistory:
    """Return the backfilled total + per-position value/P&L curves.

    Args:
        points: Number of trailing daily points to reconstruct (1..2000).
        x_account_id: Optional ``X-Account-Id`` header.

    Returns:
        A :class:`~app.schemas.PortfolioHistory` with a length-``points`` total
        series and one per-position series per held, priceable symbol.
    """
    return _history_service().portfolio_history(
        _account_id(x_account_id), points=points
    )


# ---------------------------------------------------------------------------
# Advisor route
# ---------------------------------------------------------------------------


@router.post(
    "/advisor/allocate",
    response_model=AllocationAdvice,
    summary="Recommend how to allocate an amount at a risk profile",
)
def allocate(
    body: AdviceRequest = Body(...),
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
) -> AllocationAdvice:
    """Recommend a Markowitz-sized basket for the requested amount and risk.

    The account id is accepted for symmetry but the advice is account-agnostic
    (it depends only on the amount, risk profile and asset-class filter).

    Args:
        body: The :class:`~app.schemas.AdviceRequest` (amount, riskTolerance,
            optional assetClasses).
        x_account_id: Optional ``X-Account-Id`` header (accepted, unused).

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
