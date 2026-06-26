"""SQL-backed stores that mirror the in-memory auth / invest stores exactly.

:class:`SqlUserStore` and :class:`SqlAccountStore` expose the *same* public
methods (and the same ``lock`` attribute) as
:class:`app.auth.store.UserStore` and :class:`app.invest.store.AccountStore`, so
the auth / wallet / portfolio services are oblivious to which backend is active.
They are selected by the store factories only when
``settings.persist == 'sqlite'``.

How write-through works (the key design point):

The services mutate the value returned by ``get_account()`` **directly** — they
set ``account.cash_balance``, insert into ``account.positions``, and append to
``account.transactions`` / ``account.saved_cards``, always inside
``with store.lock:``. To preserve that contract unchanged, :class:`SqlAccountStore`

* returns the very same :class:`app.invest.store.Account` dataclass the
  in-memory store uses, hydrated from the database;
* hands out a *stable* dataclass instance per ``account_id`` for the duration of
  a locked section (so repeated ``get_account`` calls inside one lock see the
  same mutable object the service is editing);
* uses a custom re-entrant :class:`_FlushLock` for ``lock`` that, when the
  **outermost** ``with`` block exits, flushes every touched account back to the
  database in one transaction.

Because every service mutation already runs under ``store.lock`` (an invariant
the in-memory store also relies on), the outermost-release flush captures all of
them with no service changes.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Optional

from sqlalchemy import select

from app.auth.store import User
from app.db.models import (
    AccountRow,
    PositionRow,
    SavedCardRow,
    TransactionRow,
    UserRow,
)
from app.db.session import SessionLocal, init_db
from app.invest.store import Account, PositionState, RiskPolicyState
from app.schemas import SavedCard, Transaction

__all__ = ["SqlUserStore", "SqlAccountStore"]


# ---------------------------------------------------------------------------
# Flush-on-release re-entrant lock
# ---------------------------------------------------------------------------


class _FlushLock:
    """A re-entrant lock that flushes dirty accounts on outermost release.

    Drop-in replacement for the in-memory store's ``threading.RLock``: it
    supports ``with`` (and bare ``acquire``/``release``) and re-entrancy. The
    only addition is that when the *last* (outermost) holder releases the lock,
    it calls the owning store's flush hook so every mutation made under the lock
    is persisted atomically.
    """

    def __init__(self, store: "SqlAccountStore") -> None:
        """Bind the lock to its store and track per-thread re-entry depth."""
        self._store = store
        self._lock = threading.RLock()
        self._depth = threading.local()

    def _get_depth(self) -> int:
        return getattr(self._depth, "value", 0)

    def _set_depth(self, value: int) -> None:
        self._depth.value = value

    def acquire(self, *args: object, **kwargs: object) -> bool:
        """Acquire the underlying lock and increment this thread's depth."""
        acquired = self._lock.acquire(*args, **kwargs)  # type: ignore[arg-type]
        if acquired:
            self._set_depth(self._get_depth() + 1)
        return acquired

    def release(self) -> None:
        """Decrement depth; on the outermost release, flush then unlock.

        The flush runs while the lock is still held so no other thread can
        observe a half-applied state; only after a successful (or failed) flush
        is the underlying lock released.
        """
        depth = self._get_depth()
        if depth == 1:
            try:
                self._store._flush()
            finally:
                self._set_depth(0)
                self._lock.release()
        else:
            self._set_depth(depth - 1)
            self._lock.release()

    def __enter__(self) -> "_FlushLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


# ---------------------------------------------------------------------------
# User store
# ---------------------------------------------------------------------------


class SqlUserStore:
    """SQLite-backed user registry mirroring :class:`app.auth.store.UserStore`.

    Public surface (identical to the in-memory store):
        * ``lock`` — a re-entrant lock.
        * ``get_by_email(email) -> User | None``
        * ``get_by_id(user_id) -> User | None``
        * ``add(email, name, password_hash) -> User``

    Reads return fresh :class:`app.auth.store.User` dataclasses; writes are
    committed immediately (each ``add`` is one transaction), so no flush-on-
    release machinery is needed here.
    """

    def __init__(self) -> None:
        """Initialize with a plain re-entrant lock (writes commit immediately)."""
        self.lock: threading.RLock = threading.RLock()

    @staticmethod
    def _norm_email(email: str) -> str:
        """Trim + lowercase an email for indexing/lookups."""
        return (email or "").strip().lower()

    @staticmethod
    def _to_user(row: UserRow) -> User:
        """Project a :class:`UserRow` onto the :class:`app.auth.store.User` dataclass."""
        return User(
            id=row.id,
            email=row.email,
            name=row.name,
            password_hash=row.password_hash,
            created_at=int(row.created_at),
        )

    def get_by_email(self, email: str) -> User | None:
        """Return the user with the given email, or ``None``."""
        key = self._norm_email(email)
        if not key:
            return None
        with self.lock, SessionLocal() as session:
            row = session.scalar(select(UserRow).where(UserRow.email == key))
            return self._to_user(row) if row is not None else None

    def get_by_id(self, user_id: str) -> User | None:
        """Return the user with the given id, or ``None``."""
        if not user_id:
            return None
        with self.lock, SessionLocal() as session:
            row = session.get(UserRow, user_id)
            return self._to_user(row) if row is not None else None

    def add(self, email: str, name: str, password_hash: str) -> User:
        """Create and persist a new user.

        Args:
            email: The user's email (normalized before storing).
            name: Display name.
            password_hash: Pre-computed PBKDF2 hash string.

        Returns:
            The newly created :class:`app.auth.store.User`.

        Raises:
            ValueError: If a user with the same normalized email already exists.
        """
        key = self._norm_email(email)
        with self.lock, SessionLocal() as session:
            exists = session.scalar(
                select(UserRow.id).where(UserRow.email == key)
            )
            if exists is not None:
                raise ValueError("Email already registered")
            row = UserRow(
                id=uuid.uuid4().hex,
                email=key,
                name=name,
                password_hash=password_hash,
                created_at=int(time.time() * 1000),
            )
            session.add(row)
            session.commit()
            return self._to_user(row)


# ---------------------------------------------------------------------------
# Account store
# ---------------------------------------------------------------------------


class SqlAccountStore:
    """SQLite-backed account store mirroring :class:`app.invest.store.AccountStore`.

    Public surface (identical to the in-memory store):
        * ``lock`` — a re-entrant, flush-on-release lock.
        * ``get_account(account_id) -> Account``

    ``get_account`` returns a live :class:`app.invest.store.Account` dataclass
    hydrated from the database. Services mutate it in place under ``lock``; the
    outermost release of ``lock`` persists every touched account in one
    transaction (see :class:`_FlushLock`).
    """

    def __init__(self) -> None:
        """Initialize the flush-lock and the per-lock-section account cache."""
        self.lock: _FlushLock = _FlushLock(self)
        # Accounts handed out during the *current* locked section, so repeated
        # get_account() calls return the same mutable object the service edits.
        # Cleared after each outermost flush. Guarded by ``lock``.
        self._dirty: dict[str, Account] = {}

    # -- hydration ------------------------------------------------------

    @staticmethod
    def _row_to_account(row: AccountRow) -> Account:
        """Build a live :class:`app.invest.store.Account` from an :class:`AccountRow`."""
        positions: dict[str, PositionState] = {}
        for pos in row.positions:
            positions[pos.symbol] = PositionState(
                symbol=pos.symbol,
                units=float(pos.units),
                cost_basis=float(pos.cost_basis),
                realized_pnl=float(pos.realized_pnl),
                opened_at=int(pos.opened_at),
                high_water_price=float(pos.high_water_price or 0.0),
            )
        saved_cards = [
            SavedCard(
                id=c.id,
                brand=c.brand,
                last4=c.last4,
                exp_month=int(c.exp_month),
                exp_year=int(c.exp_year),
                holder=c.holder,
            )
            for c in row.saved_cards
        ]
        transactions = [
            Transaction(
                id=t.id,
                type=t.type,  # type: ignore[arg-type]
                amount=float(t.amount),
                symbol=t.symbol,
                status=t.status,  # type: ignore[arg-type]
                created_at=int(t.created_at),
                ref=t.ref,
                note=t.note,
            )
            for t in row.transactions
        ]
        risk_policy = RiskPolicyState(
            stop_loss_pct=(
                float(row.stop_loss_pct) if row.stop_loss_pct is not None else None
            ),
            trailing_stop_pct=(
                float(row.trailing_stop_pct)
                if row.trailing_stop_pct is not None
                else None
            ),
            take_profit_pct=(
                float(row.take_profit_pct)
                if row.take_profit_pct is not None
                else None
            ),
            max_drawdown_pct=(
                float(row.max_drawdown_pct)
                if row.max_drawdown_pct is not None
                else None
            ),
        )
        return Account(
            account_id=row.account_id,
            cash_balance=float(row.cash_balance),
            positions=positions,
            saved_cards=saved_cards,
            transactions=transactions,
            risk_policy=risk_policy,
            peak_value=float(row.peak_value or 0.0),
        )

    def get_account(self, account_id: str) -> Account:
        """Return the account for ``account_id``, creating it on first use.

        The first access within a locked section hydrates the account from the
        database (creating an empty row if none exists) and caches the live
        dataclass; subsequent accesses in the same section return that same
        object so a service's mutations accumulate on one instance.

        Args:
            account_id: The account identifier; falsy ids fall back to ``'demo'``.

        Returns:
            The (possibly newly created) :class:`app.invest.store.Account`.
        """
        key = (account_id or "demo").strip() or "demo"
        with self.lock:
            cached = self._dirty.get(key)
            if cached is not None:
                return cached
            with SessionLocal() as session:
                row = session.get(AccountRow, key)
                if row is None:
                    account = Account(account_id=key)
                else:
                    account = self._row_to_account(row)
            self._dirty[key] = account
            return account

    # -- persistence ----------------------------------------------------

    def _flush(self) -> None:
        """Persist every account touched in the just-finished locked section.

        Replaces each dirty account's child rows (positions / cards / ledger)
        with the current in-memory state and upserts the account row. Called by
        :class:`_FlushLock` on outermost release while the lock is still held.
        Any error propagates (and the cache is still cleared) so a failed write
        never leaves stale half-state cached.
        """
        if not self._dirty:
            return
        dirty = self._dirty
        self._dirty = {}
        with SessionLocal() as session:
            for account in dirty.values():
                self._persist_account(session, account)
            session.commit()

    @staticmethod
    def _persist_account(session, account: Account) -> None:
        """Upsert one account and rewrite its child collections.

        Args:
            session: An open SQLAlchemy session (committed by the caller).
            account: The live in-memory account to persist.
        """
        row = session.get(AccountRow, account.account_id)
        if row is None:
            row = AccountRow(account_id=account.account_id)
            session.add(row)
        row.cash_balance = float(account.cash_balance)
        row.peak_value = float(account.peak_value)
        policy = account.risk_policy
        row.stop_loss_pct = policy.stop_loss_pct
        row.trailing_stop_pct = policy.trailing_stop_pct
        row.take_profit_pct = policy.take_profit_pct
        row.max_drawdown_pct = policy.max_drawdown_pct

        # Positions: rewrite from the in-memory dict (delete-orphan handles
        # removed positions).
        row.positions = [
            PositionRow(
                account_id=account.account_id,
                symbol=state.symbol,
                units=float(state.units),
                cost_basis=float(state.cost_basis),
                realized_pnl=float(state.realized_pnl),
                opened_at=int(state.opened_at),
                high_water_price=float(state.high_water_price),
            )
            for state in account.positions.values()
        ]

        # Saved cards: rewrite in order (token ids carry over; ``ord`` preserves
        # the list order on reload).
        row.saved_cards = [
            SavedCardRow(
                id=card.id,
                ord=index,
                account_id=account.account_id,
                brand=card.brand,
                last4=card.last4,
                exp_month=int(card.exp_month),
                exp_year=int(card.exp_year),
                holder=card.holder,
            )
            for index, card in enumerate(account.saved_cards)
        ]

        # Transactions: rewrite the ordered ledger (uuid ids carry over; ``ord``
        # preserves the chronological order on reload).
        row.transactions = [
            TransactionRow(
                id=txn.id,
                ord=index,
                account_id=account.account_id,
                type=txn.type,
                amount=float(txn.amount),
                symbol=txn.symbol,
                status=txn.status,
                created_at=int(txn.created_at),
                ref=txn.ref,
                note=txn.note,
            )
            for index, txn in enumerate(account.transactions)
        ]


# ---------------------------------------------------------------------------
# Singletons (lazy; created only when the SQL backend is selected)
# ---------------------------------------------------------------------------

_USER_STORE_LOCK = threading.Lock()
_USER_STORE: Optional[SqlUserStore] = None
_ACCOUNT_STORE_LOCK = threading.Lock()
_ACCOUNT_STORE: Optional[SqlAccountStore] = None


def get_sql_user_store() -> SqlUserStore:
    """Return the process-wide :class:`SqlUserStore`, initializing the DB once.

    Calls :func:`app.db.session.init_db` (idempotent ``create_all``) before the
    store is first handed out and seeds the demo user so the SQLite-backed app
    is usable immediately, mirroring the in-memory store.

    Returns:
        The shared :class:`SqlUserStore`.
    """
    global _USER_STORE
    if _USER_STORE is None:
        with _USER_STORE_LOCK:
            if _USER_STORE is None:
                init_db()
                store = SqlUserStore()
                _seed_demo_user(store)
                _USER_STORE = store
    return _USER_STORE


def get_sql_account_store() -> SqlAccountStore:
    """Return the process-wide :class:`SqlAccountStore`, initializing the DB once.

    Returns:
        The shared :class:`SqlAccountStore`.
    """
    global _ACCOUNT_STORE
    if _ACCOUNT_STORE is None:
        with _ACCOUNT_STORE_LOCK:
            if _ACCOUNT_STORE is None:
                init_db()
                _ACCOUNT_STORE = SqlAccountStore()
    return _ACCOUNT_STORE


def _seed_demo_user(store: SqlUserStore) -> None:
    """Seed the documented demo account into a fresh SQL store (idempotent).

    Mirrors :func:`app.auth.store._seed_demo_user` so a fresh SQLite database is
    immediately usable with the demo credentials.

    Args:
        store: The :class:`SqlUserStore` to seed.
    """
    # Imported lazily to avoid a hard import cost on the default in-memory path.
    from app.auth.security import hash_password
    from app.auth.store import DEMO_EMAIL, DEMO_NAME, DEMO_PASSWORD

    if store.get_by_email(DEMO_EMAIL) is None:
        store.add(
            email=DEMO_EMAIL,
            name=DEMO_NAME,
            password_hash=hash_password(DEMO_PASSWORD),
        )
