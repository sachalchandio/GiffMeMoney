"""Thread-safe in-memory account store for the simulated brokerage.

This module holds all mutable wallet state: per-account cash, open positions,
saved (masked) cards, and the transaction ledger. State lives in process memory
only and resets on restart — there is no database, no persistence, and no real
money. Every service mutation must take place under ``store.lock`` so concurrent
request and WebSocket threads never observe a half-applied change (balances must
always reconcile).

The :class:`AccountStore` is exposed as a process-wide singleton via
:func:`get_store`. Accounts are created lazily on first access, keyed by the
account id (default ``'demo'``, from the optional ``X-Account-Id`` header).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from app.schemas import SavedCard, Transaction

__all__ = [
    "PositionState",
    "Account",
    "AccountStore",
    "get_store",
]


@dataclass
class PositionState:
    """Mutable state of a single held position.

    Unlike the wire :class:`~app.schemas.Position` DTO (which is marked to the
    latest price on read), this is the raw accounting record kept in the store.

    Attributes:
        symbol: Asset ticker (canonical upper-case).
        units: Fractional units currently held (``>= 0``).
        cost_basis: Total dollars currently invested in the still-held units.
            Reduced pro-rata on partial sells.
        realized_pnl: Cumulative realized profit/loss from prior sells of this
            symbol (can be negative).
        opened_at: Unix timestamp in milliseconds when the position first
            opened.
    """

    symbol: str
    units: float
    cost_basis: float
    realized_pnl: float
    opened_at: int


@dataclass
class Account:
    """All mutable state for one simulated brokerage account.

    Attributes:
        account_id: The account identifier (default ``'demo'``).
        cash_balance: Uninvested cash available to spend or withdraw.
        positions: Open positions keyed by canonical (upper-case) symbol.
        saved_cards: Tokenized, masked cards on file (never raw PAN/CVC).
        transactions: Immutable ledger, appended in chronological order.
    """

    account_id: str
    cash_balance: float = 0.0
    positions: dict[str, PositionState] = field(default_factory=dict)
    saved_cards: list[SavedCard] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)


class AccountStore:
    """Process-wide, thread-safe registry of :class:`Account` objects.

    All reads and writes of account state should occur while holding
    :attr:`lock` (a re-entrant lock, so a service may call other store/service
    methods that also lock without deadlocking). Accounts are created on first
    access.
    """

    def __init__(self) -> None:
        """Initialize an empty store with a re-entrant lock."""
        self.lock: threading.RLock = threading.RLock()
        self._accounts: dict[str, Account] = {}

    def get_account(self, account_id: str) -> Account:
        """Return the account for ``account_id``, creating it on first use.

        The lookup and lazy creation are performed under :attr:`lock` so two
        threads racing on the same fresh id cannot create two accounts.

        Args:
            account_id: The account identifier; falsy ids fall back to
                ``'demo'`` and surrounding whitespace is stripped.

        Returns:
            The (possibly newly created) :class:`Account`.
        """
        key = (account_id or "demo").strip() or "demo"
        with self.lock:
            account = self._accounts.get(key)
            if account is None:
                account = Account(account_id=key)
                self._accounts[key] = account
            return account


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_STORE_LOCK = threading.Lock()
_STORE_INSTANCE: AccountStore | None = None


def get_store() -> AccountStore:
    """Return the process-wide :class:`AccountStore` singleton.

    Constructed lazily and memoized so every service and API call shares the
    same in-memory wallet state for the life of the process.

    Returns:
        The shared :class:`AccountStore` instance.
    """
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        with _STORE_LOCK:
            if _STORE_INSTANCE is None:
                _STORE_INSTANCE = AccountStore()
    return _STORE_INSTANCE
