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

from app.config import settings
from app.schemas import SavedCard, Transaction

__all__ = [
    "PositionState",
    "RiskPolicyState",
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
        high_water_price: Highest mark price ever observed for this position
            (the peak used by the trailing-stop rule). Seeded to the first buy's
            execution price and ratcheted up — never down — as the position is
            marked to market. ``0.0`` means "not yet observed" (treated as the
            blended entry price on first evaluation). Additive field; defaults to
            ``0.0`` so older constructions stay valid.
    """

    symbol: str
    units: float
    cost_basis: float
    realized_pnl: float
    opened_at: int
    high_water_price: float = 0.0


@dataclass
class RiskPolicyState:
    """Per-account post-buy loss controls (all optional, default OFF/``None``).

    These are protective *exit* rules applied to already-held positions by
    :meth:`app.invest.portfolio_service.PortfolioService.evaluate_risk`. They
    never block or alter buys; they only trigger protective sells / de-risking on
    demand. Every threshold is a positive percentage; ``None`` means the rule is
    disabled. Defaults are all ``None`` so the feature is OFF unless the account
    owner opts in (no behaviour change for existing accounts).

    Attributes:
        stop_loss_pct: Hard stop — sell a position once it is down more than this
            percent from its blended entry (average-cost) price.
        trailing_stop_pct: Trailing stop — sell a position once it falls more than
            this percent below its observed high-water mark price.
        take_profit_pct: Profit target — sell a position once it is up more than
            this percent above its blended entry price.
        max_drawdown_pct: Portfolio circuit-breaker — when total portfolio value
            is down more than this percent from its peak, reduce exposure (sell
            the worst-performing positions / raise cash).
    """

    stop_loss_pct: float | None = None
    trailing_stop_pct: float | None = None
    take_profit_pct: float | None = None
    max_drawdown_pct: float | None = None


@dataclass
class Account:
    """All mutable state for one simulated brokerage account.

    Attributes:
        account_id: The account identifier (default ``'demo'``).
        cash_balance: Uninvested cash available to spend or withdraw.
        positions: Open positions keyed by canonical (upper-case) symbol.
        saved_cards: Tokenized, masked cards on file (never raw PAN/CVC).
        transactions: Immutable ledger, appended in chronological order.
        risk_policy: Per-account post-buy loss controls (default all-OFF).
        peak_value: Highest total portfolio value (cash + marked positions)
            observed so far, used by the ``max_drawdown_pct`` circuit-breaker.
            ``0.0`` means "not yet observed".
    """

    account_id: str
    cash_balance: float = 0.0
    positions: dict[str, PositionState] = field(default_factory=dict)
    saved_cards: list[SavedCard] = field(default_factory=list)
    transactions: list[Transaction] = field(default_factory=list)
    risk_policy: RiskPolicyState = field(default_factory=RiskPolicyState)
    peak_value: float = 0.0


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
    """Return the process-wide account store singleton.

    Selected by :data:`app.config.settings.persist`:

    * ``'memory'`` (**default**) — the in-memory, thread-safe
      :class:`AccountStore`, constructed lazily and memoized so every service
      and API call shares the same wallet state for the life of the process.
      **This is the only path the default app and the test suite ever take, so
      default behavior is unchanged.**
    * ``'sqlite'`` (opt-in) — a SQLite-backed store implementing the same public
      surface (``lock`` + ``get_account``), returned from
      :mod:`app.db.repositories` (which initializes the database once).
      SQLAlchemy is imported lazily here so the default path never depends on it.

    Returns:
        The shared account store (an :class:`AccountStore` by default; a
        structurally compatible SQL-backed store when ``persist == 'sqlite'``).
    """
    if (settings.persist or "memory").strip().lower() == "sqlite":
        # Lazy import: keeps SQLAlchemy off the default in-memory code path.
        from app.db.repositories import (  # noqa: PLC0415 - intentional lazy import
            get_sql_account_store,
        )

        return get_sql_account_store()  # type: ignore[return-value]

    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        with _STORE_LOCK:
            if _STORE_INSTANCE is None:
                _STORE_INSTANCE = AccountStore()
    return _STORE_INSTANCE
