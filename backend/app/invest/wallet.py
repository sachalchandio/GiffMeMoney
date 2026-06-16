"""Wallet service for the simulated brokerage: funding, payout, cards, ledger.

The :class:`WalletService` is the cash-side of the account. It mediates between
the :class:`~app.invest.store.AccountStore` (mutable state), the
:class:`~app.invest.payments.PaymentProvider` (card validation / sandbox
charges) and a :class:`~app.market.provider.MarketDataProvider` (to mark open
positions to market when computing the wallet's invested value).

Money-handling stance (critical): this is a **sandbox / paper** wallet. No real
money moves. Cards are validated and only ever stored **masked** (brand + last4
+ token). Raw PAN/CVC are never persisted or logged — they live only inside the
inbound :class:`~app.schemas.CardIn` long enough to validate and tokenize.

Invariants enforced here:
    * ``total_value == cash_balance + invested_value`` always reconciles.
    * Cash never goes negative (withdrawals over the balance are rejected with a
      :class:`ValueError`, which the API maps to HTTP 400).
    * Every mutation runs under ``store.lock`` so concurrent request / WebSocket
      threads never observe a half-applied change.
"""

from __future__ import annotations

import math

from app.invest.payments import PaymentProvider, tokenize
from app.invest.store import AccountStore
from app.market.provider import MarketDataProvider
from app.schemas import CardIn, SavedCard, Transaction, Wallet

__all__ = ["WalletService"]


class WalletService:
    """Cash, card and ledger operations for a simulated brokerage account.

    Args:
        store: The process-wide :class:`~app.invest.store.AccountStore`.
        payments: A :class:`~app.invest.payments.PaymentProvider` used to
            validate cards and produce sandbox deposit/withdrawal transactions.
        provider: A :class:`~app.market.provider.MarketDataProvider` used to mark
            open positions to the latest price when computing invested value.
    """

    def __init__(
        self,
        store: AccountStore,
        payments: PaymentProvider,
        provider: MarketDataProvider,
    ) -> None:
        """Store the collaborators (no state of its own; all state is in ``store``)."""
        self._store = store
        self._payments = payments
        self._provider = provider

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def _invested_value(self, account_id: str) -> float:
        """Compute the mark-to-market value of every open position.

        Each position is marked at ``units * provider.latest_price(symbol)``.
        Must be called while holding ``store.lock``. A symbol whose live price
        cannot be fetched is marked at zero rather than raising, so the wallet
        view never crashes on a transient provider error.

        Args:
            account_id: The account whose positions to value.

        Returns:
            The total invested market value as a finite, non-negative float.
        """
        account = self._store.get_account(account_id)
        total = 0.0
        for state in account.positions.values():
            try:
                price = float(self._provider.latest_price(state.symbol))
            except Exception:
                price = 0.0
            if not math.isfinite(price) or price < 0.0:
                price = 0.0
            value = float(state.units) * price
            if math.isfinite(value):
                total += value
        return total if math.isfinite(total) and total > 0.0 else max(total, 0.0)

    def get_wallet(self, account_id: str) -> Wallet:
        """Return a reconciled :class:`~app.schemas.Wallet` snapshot.

        ``investedValue`` is the live mark-to-market of all positions;
        ``totalValue`` is ``cashBalance + investedValue`` (the balances always
        reconcile). Saved cards are included masked.

        Args:
            account_id: The account identifier (default ``'demo'``).

        Returns:
            A populated :class:`~app.schemas.Wallet`.
        """
        with self._store.lock:
            account = self._store.get_account(account_id)
            cash = float(account.cash_balance)
            if not math.isfinite(cash):
                cash = 0.0
            invested = self._invested_value(account_id)
            return Wallet(
                account_id=account.account_id,
                cash_balance=round(cash, 2),
                invested_value=round(invested, 2),
                total_value=round(cash + invested, 2),
                currency="USD",
                saved_cards=list(account.saved_cards),
            )

    def list_cards(self, account_id: str) -> list[SavedCard]:
        """Return the masked saved cards on file for the account.

        Args:
            account_id: The account identifier.

        Returns:
            A list of :class:`~app.schemas.SavedCard` (never raw PAN/CVC).
        """
        with self._store.lock:
            account = self._store.get_account(account_id)
            return list(account.saved_cards)

    def list_transactions(self, account_id: str) -> list[Transaction]:
        """Return the account's ledger entries, newest first.

        Args:
            account_id: The account identifier.

        Returns:
            A list of :class:`~app.schemas.Transaction` ordered most-recent
            first (the store appends chronologically; this reverses a copy).
        """
        with self._store.lock:
            account = self._store.get_account(account_id)
            return list(reversed(account.transactions))

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def deposit(
        self,
        account_id: str,
        amount: float,
        card: CardIn,
        save_card: bool = False,
        saved_card_id: str | None = None,
    ) -> tuple[Wallet, Transaction]:
        """Fund the account by charging a (validated, never-stored-raw) card.

        The payment provider validates the card (Luhn + future expiry) and the
        amount (``> 0`` and ``<= MAX_CHARGE``); on success the cash balance is
        credited and the deposit transaction is appended to the ledger. When
        ``save_card`` is true the card is tokenized into a masked
        :class:`~app.schemas.SavedCard` and stored unless an identical card
        (same brand + last4 + expiry) is already on file.

        Args:
            account_id: The account to fund.
            amount: Positive dollar amount to deposit.
            card: The card to charge (raw PAN/CVC validated then discarded).
            save_card: Whether to remember the card (masked) for reuse.
            saved_card_id: Optional id of an already-saved card the caller chose;
                accepted for API symmetry — the inline ``card`` is still the one
                validated/charged in this sandbox.

        Returns:
            A ``(wallet, transaction)`` tuple: the updated wallet snapshot and
            the completed deposit transaction.

        Raises:
            ValueError: If the card or amount is invalid (mapped to HTTP 400).
        """
        with self._store.lock:
            account = self._store.get_account(account_id)
            # Validate + produce the sandbox charge (raises ValueError on bad input).
            txn = self._payments.charge(card, amount, account.account_id)

            account.cash_balance = float(account.cash_balance) + float(txn.amount)
            if not math.isfinite(account.cash_balance):
                account.cash_balance = 0.0

            if save_card:
                self._maybe_save_card(account, card)

            account.transactions.append(txn)
            wallet = self.get_wallet(account_id)
            return wallet, txn

    def _maybe_save_card(self, account, card: CardIn) -> None:
        """Tokenize and store ``card`` unless an identical one is already saved.

        Two cards are considered identical when their brand, last4 and expiry
        match (we never compare PANs because PANs are never stored). Must be
        called while holding ``store.lock``.

        Args:
            account: The :class:`~app.invest.store.Account` to mutate.
            card: The validated card input to tokenize.
        """
        saved = tokenize(card)
        for existing in account.saved_cards:
            if (
                existing.brand == saved.brand
                and existing.last4 == saved.last4
                and existing.exp_month == saved.exp_month
                and existing.exp_year == saved.exp_year
            ):
                return
        account.saved_cards.append(saved)

    def withdraw(
        self,
        account_id: str,
        amount: float,
        destination: str | None = None,
    ) -> tuple[Wallet, Transaction]:
        """Pay cash out of the account to an external destination (sandbox).

        The amount is checked against the available cash *before* the payment
        provider's bounds check so an over-withdrawal yields a clear, specific
        error. On success the cash balance is debited and a withdrawal
        transaction is appended.

        Args:
            account_id: The account to debit.
            amount: Positive dollar amount to withdraw (``<= cash``).
            destination: Optional free-text payout destination label.

        Returns:
            A ``(wallet, transaction)`` tuple: the updated wallet snapshot and
            the completed withdrawal transaction.

        Raises:
            ValueError: If the amount is non-positive, exceeds the cash balance,
                or fails the provider's sandbox bounds (mapped to HTTP 400).
        """
        with self._store.lock:
            account = self._store.get_account(account_id)
            try:
                value = float(amount)
            except (TypeError, ValueError):
                raise ValueError("Amount must be a number.") from None
            if not math.isfinite(value):
                raise ValueError("Amount must be a finite number.")
            if value <= 0.0:
                raise ValueError("Amount must be greater than zero.")

            cash = float(account.cash_balance)
            if value > cash + 1e-9:
                raise ValueError(
                    f"Insufficient funds: cannot withdraw ${value:,.2f} "
                    f"from a cash balance of ${cash:,.2f}."
                )

            # Provider enforces the positive / within-cap bounds and builds the txn.
            txn = self._payments.payout(value, account.account_id, destination)

            account.cash_balance = cash - float(txn.amount)
            # Guard against tiny negative residue from float subtraction.
            if account.cash_balance < 0.0 or not math.isfinite(account.cash_balance):
                account.cash_balance = max(0.0, round(account.cash_balance, 6))

            account.transactions.append(txn)
            wallet = self.get_wallet(account_id)
            return wallet, txn

    def delete_card(self, account_id: str, card_id: str) -> None:
        """Remove a saved card from the account.

        Args:
            account_id: The account identifier.
            card_id: The opaque token id of the saved card to remove.

        Raises:
            KeyError: If no saved card has that id (mapped to HTTP 404).
        """
        with self._store.lock:
            account = self._store.get_account(account_id)
            for index, existing in enumerate(account.saved_cards):
                if existing.id == card_id:
                    del account.saved_cards[index]
                    return
            raise KeyError(f"Unknown saved card: {card_id!r}")
