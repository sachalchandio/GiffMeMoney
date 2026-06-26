"""SQLAlchemy 2.0 ORM models for the optional SQLite persistence layer.

These rows mirror the in-memory dataclasses used by the default stores:

* :class:`UserRow`        ↔ :class:`app.auth.store.User`
* :class:`AccountRow`     ↔ :class:`app.invest.store.Account`
* :class:`PositionRow`    ↔ :class:`app.invest.store.PositionState`
* :class:`TransactionRow` ↔ :class:`app.schemas.Transaction`
* :class:`SavedCardRow`   ↔ :class:`app.schemas.SavedCard`
* :class:`BotRunRow`      — a durable record of a simulated auto-trader run
  (forward-looking; not wired to a store yet, but part of the schema so a
  ``sqlite`` deployment can persist bot runs without a later migration).

This module is imported **only** when ``settings.persist == 'sqlite'`` (the SQL
stores import it lazily), so the default in-memory path has no dependency on
SQLAlchemy.

Mapping notes:
    * All monetary amounts are stored as ``Float`` (the in-memory model is also
      ``float``); this is a sandbox app, not a real ledger.
    * Timestamps follow the existing convention: user/account/transaction times
      are unix **milliseconds** (``BigInteger``), matching the DTOs.
    * Card data stored here is the already-masked :class:`SavedCardRow` (brand +
      last4 + expiry + holder + token) — **never** a raw PAN/CVC, exactly as the
      in-memory store guarantees.
"""

from __future__ import annotations

from sqlalchemy import BigInteger, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

__all__ = [
    "Base",
    "UserRow",
    "AccountRow",
    "PositionRow",
    "TransactionRow",
    "SavedCardRow",
    "BotRunRow",
]


class Base(DeclarativeBase):
    """Declarative base for every GiffMeMoney persistence model."""


class UserRow(Base):
    """A registered user (durable mirror of :class:`app.auth.store.User`).

    Columns:
        id: Opaque user id (uuid hex) — primary key.
        email: Lowercased, unique email address (indexed for login lookups).
        name: Display name.
        password_hash: PBKDF2 hash string (never the raw password).
        created_at: Unix timestamp in **milliseconds**.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[str] = mapped_column(
        String(320), unique=True, index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)


class AccountRow(Base):
    """A simulated brokerage account (mirror of :class:`app.invest.store.Account`).

    Columns:
        account_id: The account identifier (default ``'demo'``) — primary key.
        cash_balance: Uninvested cash available to spend or withdraw.
        peak_value: Highest total portfolio value observed (drawdown breaker).
        stop_loss_pct: Per-account stop-loss threshold (``NULL`` = OFF).
        trailing_stop_pct: Per-account trailing-stop threshold (``NULL`` = OFF).
        take_profit_pct: Per-account take-profit threshold (``NULL`` = OFF).
        max_drawdown_pct: Per-account drawdown circuit-breaker (``NULL`` = OFF).

    Relationships:
        positions: Open positions keyed by symbol.
        transactions: The chronological ledger.
        saved_cards: Tokenized, masked cards on file.
    """

    __tablename__ = "accounts"

    account_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    cash_balance: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    peak_value: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stop_loss_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    trailing_stop_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_drawdown_pct: Mapped[float | None] = mapped_column(Float, nullable=True)

    positions: Mapped[list["PositionRow"]] = relationship(
        back_populates="account",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    transactions: Mapped[list["TransactionRow"]] = relationship(
        back_populates="account",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="TransactionRow.ord",
    )
    saved_cards: Mapped[list["SavedCardRow"]] = relationship(
        back_populates="account",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="SavedCardRow.ord",
    )


class PositionRow(Base):
    """A single open position (mirror of :class:`app.invest.store.PositionState`).

    Columns:
        id: Surrogate auto-increment primary key.
        account_id: Owning account (FK).
        symbol: Canonical upper-case ticker (unique per account).
        units: Fractional units currently held (``>= 0``).
        cost_basis: Dollars currently invested in the still-held units.
        realized_pnl: Cumulative realized P&L from prior sells of this symbol.
        opened_at: Unix timestamp in **milliseconds** the position opened.
        high_water_price: Highest mark price ever observed (trailing-stop peak).
    """

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    account_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("accounts.account_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    units: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cost_basis: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    realized_pnl: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    opened_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    high_water_price: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )

    account: Mapped["AccountRow"] = relationship(back_populates="positions")


class TransactionRow(Base):
    """A ledger entry (mirror of :class:`app.schemas.Transaction`).

    Columns:
        id: The transaction's own uuid (primary key, matches the DTO ``id``).
        ord: Per-account ordinal preserving the ledger's insertion order on
            reload (assigned by the store on persist; the in-memory ledger is an
            ordered list).
        account_id: Owning account (FK).
        type: One of ``deposit | withdrawal | buy | sell``.
        amount: Positive dollar magnitude.
        symbol: Asset symbol for buys/sells; ``NULL`` for cash movements.
        status: ``completed`` or ``failed``.
        created_at: Unix timestamp in **milliseconds**.
        ref: Human-facing reference string.
        note: Short human-readable description.
    """

    __tablename__ = "transactions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ord: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    account_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("accounts.account_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    type: Mapped[str] = mapped_column(String(16), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    ref: Mapped[str] = mapped_column(String(64), nullable=False)
    note: Mapped[str] = mapped_column(Text, nullable=False)

    account: Mapped["AccountRow"] = relationship(back_populates="transactions")


class SavedCardRow(Base):
    """A tokenized, masked card on file (mirror of :class:`app.schemas.SavedCard`).

    Carries **no** sensitive data — only brand, last4, expiry and holder, keyed
    by an opaque token id, exactly like the in-memory store.

    Columns:
        id: Opaque token id (uuid, primary key, matches the DTO ``id``).
        ord: Per-account ordinal preserving insertion order on reload (assigned
            by the store on persist).
        account_id: Owning account (FK).
        brand: Detected network (visa / mastercard / amex / discover / unknown).
        last4: The final four digits of the PAN.
        exp_month: Expiry month in ``[1, 12]``.
        exp_year: Four-digit expiry year.
        holder: Cardholder name.
    """

    __tablename__ = "saved_cards"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ord: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    account_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("accounts.account_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    brand: Mapped[str] = mapped_column(String(32), nullable=False)
    last4: Mapped[str] = mapped_column(String(4), nullable=False)
    exp_month: Mapped[int] = mapped_column(Integer, nullable=False)
    exp_year: Mapped[int] = mapped_column(Integer, nullable=False)
    holder: Mapped[str] = mapped_column(String(255), nullable=False)

    account: Mapped["AccountRow"] = relationship(back_populates="saved_cards")


class BotRunRow(Base):
    """A durable record of one simulated auto-trader run (forward-looking).

    The bot engine does not persist runs today; this table exists so a
    ``sqlite`` deployment can store run summaries without a future migration.
    The full :class:`app.schemas.BotRunResult` is kept as a JSON blob in
    ``result_json`` (camelCase wire form), with the headline fields promoted to
    columns for cheap listing/filtering.

    Columns:
        id: Surrogate uuid primary key for the run.
        account_id: Owning account (free text; not FK-constrained so anonymous /
            ad-hoc runs are storable). ``NULL`` for unattributed runs.
        mode: The :data:`app.schemas.BotModeId` that was run.
        amount: Starting paper capital.
        final_value: The run's final paper value.
        total_return_pct: Total return over the run, in percent.
        created_at: Unix timestamp in **milliseconds**.
        result_json: The full serialized :class:`app.schemas.BotRunResult`.
    """

    __tablename__ = "bot_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str | None] = mapped_column(
        String(64), index=True, nullable=True
    )
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    final_value: Mapped[float] = mapped_column(Float, nullable=False)
    total_return_pct: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[int] = mapped_column(BigInteger, nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
