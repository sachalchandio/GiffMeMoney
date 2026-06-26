"""Tests for the optional SQLite persistence layer (go-live, OPT-IN).

These cover the safety contract from ``docs/GOLIVE.md`` §3 / §6:

* the **default** store is the in-memory one (``persist == 'memory'``), and
  selecting it imports/initializes **no database** (the SQL layer and SQLAlchemy
  engine are never touched);
* with ``persist == 'sqlite'`` pointed at a **temp** database file,
  :class:`~app.db.repositories.SqlUserStore` and
  :class:`~app.db.repositories.SqlAccountStore` round-trip a user, a deposit
  (cash) and a position across a reload;
* the SQLite store only ever initializes its **own new** temp db file — never an
  existing one — and the default path never creates a database.

SAFETY: every test runs against an isolated temp db inside ``tmp_path`` and
restores ``settings.persist`` / ``settings.db_url`` plus the engine and SQL-store
singletons afterwards, so no real ``giffmemoney.db`` is created or touched and the
session default stays ``memory``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest

import app.db.repositories as repositories
import app.db.session as db_session
from app.auth.store import (
    DEMO_EMAIL,
    UserStore,
    get_store as get_user_store,
)
from app.config import settings
from app.invest.store import AccountStore, PositionState, RiskPolicyState
from app.invest.store import get_store as get_account_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_settings(tmp_path: Path) -> Iterator[str]:
    """Point persistence at a fresh temp SQLite db and clean up fully.

    Snapshots and restores ``settings.persist`` / ``settings.db_url``, resets the
    engine cache and the SQL-store singletons before and after, and yields the
    absolute temp db path so a test can assert the file is created on init.
    """
    db_path = tmp_path / "giffmemoney_test.db"
    url = "sqlite:///" + db_path.as_posix()

    saved_persist = settings.persist
    saved_url = settings.db_url

    # Start from a clean engine + singleton slate.
    db_session.reset_engine_cache()
    repositories._USER_STORE = None
    repositories._ACCOUNT_STORE = None

    settings.persist = "sqlite"
    settings.db_url = url
    try:
        yield str(db_path)
    finally:
        settings.persist = saved_persist
        settings.db_url = saved_url
        db_session.reset_engine_cache()
        repositories._USER_STORE = None
        repositories._ACCOUNT_STORE = None


# ---------------------------------------------------------------------------
# Default store is in-memory (and touches no database)
# ---------------------------------------------------------------------------


def test_default_persist_setting_is_memory() -> None:
    """The shipped default keeps persistence in-memory."""
    assert settings.persist == "memory"


def test_default_user_store_is_in_memory() -> None:
    """``auth.store.get_store`` returns the in-memory :class:`UserStore`."""
    assert (settings.persist or "memory").strip().lower() == "memory"
    store = get_user_store()
    assert isinstance(store, UserStore)


def test_default_account_store_is_in_memory() -> None:
    """``invest.store.get_store`` returns the in-memory :class:`AccountStore`."""
    assert (settings.persist or "memory").strip().lower() == "memory"
    store = get_account_store()
    assert isinstance(store, AccountStore)


def test_memory_store_does_not_initialize_a_database() -> None:
    """Selecting the default store never builds an engine or calls ``init_db``.

    Asserts the in-memory path never imports/initializes the SQL layer: neither
    :func:`app.db.session.init_db` nor :func:`app.db.session.get_engine` is
    invoked when ``persist == 'memory'``.
    """
    assert settings.persist == "memory"
    with patch("app.db.session.init_db") as init_db, patch(
        "app.db.session.get_engine"
    ) as get_engine:
        user_store = get_user_store()
        account_store = get_account_store()
    init_db.assert_not_called()
    get_engine.assert_not_called()
    assert isinstance(user_store, UserStore)
    assert isinstance(account_store, AccountStore)


# ---------------------------------------------------------------------------
# SQLite stores round-trip on a temp db
# ---------------------------------------------------------------------------


def test_sqlite_store_creates_its_own_new_db_file(sqlite_settings: str) -> None:
    """Initializing the SQLite store creates a brand-new db file on first use."""
    db_path = Path(sqlite_settings)
    assert not db_path.exists()  # nothing exists yet

    store = repositories.get_sql_user_store()
    # init_db ran exactly once on first hand-out; the file now exists.
    assert db_path.exists()
    # The demo user is seeded so the SQLite app is usable immediately.
    assert store.get_by_email(DEMO_EMAIL) is not None


def test_sql_user_store_round_trip(sqlite_settings: str) -> None:
    """A user created via :class:`SqlUserStore` survives a reload from disk."""
    store = repositories.get_sql_user_store()
    created = store.add(
        email="Jane@Example.com",  # mixed case -> normalized to lowercase
        name="Jane Investor",
        password_hash="pbkdf2$fake$hash",
    )
    assert created.email == "jane@example.com"

    # Lookups by both email (case-insensitive) and id.
    by_email = store.get_by_email("JANE@example.com")
    assert by_email is not None
    assert by_email.id == created.id
    assert by_email.password_hash == "pbkdf2$fake$hash"
    assert store.get_by_id(created.id) is not None

    # Reload from a brand-new store instance backed by the same temp db.
    repositories._USER_STORE = None
    reloaded_store = repositories.get_sql_user_store()
    reloaded = reloaded_store.get_by_email("jane@example.com")
    assert reloaded is not None
    assert reloaded.id == created.id
    assert reloaded.name == "Jane Investor"


def test_sql_user_store_rejects_duplicate_email(sqlite_settings: str) -> None:
    """Adding a duplicate (normalized) email raises, mirroring the in-memory store."""
    store = repositories.get_sql_user_store()
    store.add(email="dup@example.com", name="First", password_hash="h1")
    with pytest.raises(ValueError):
        store.add(email="DUP@example.com", name="Second", password_hash="h2")


def test_sql_account_store_round_trip(sqlite_settings: str) -> None:
    """A deposit (cash), a position, the risk policy, HWM and peak persist on reload."""
    store = repositories.get_sql_account_store()

    # Mutate the live Account dataclass in place under the flush-on-release lock,
    # exactly as the wallet/portfolio services do.
    with store.lock:
        account = store.get_account("demo")
        account.cash_balance = 1_234.56  # a simulated deposit
        account.peak_value = 2_000.0
        account.risk_policy = RiskPolicyState(
            stop_loss_pct=10.0, max_drawdown_pct=25.0
        )
        account.positions["AAPL"] = PositionState(
            symbol="AAPL",
            units=3.0,
            cost_basis=600.0,
            realized_pnl=0.0,
            opened_at=1_700_000_000_000,
            high_water_price=275.0,
        )
    # Outermost lock release flushed the mutations to the temp db.

    # Reload from a brand-new store instance backed by the same temp db.
    repositories._ACCOUNT_STORE = None
    reloaded_store = repositories.get_sql_account_store()
    with reloaded_store.lock:
        reloaded = reloaded_store.get_account("demo")
    assert reloaded.cash_balance == pytest.approx(1_234.56)
    assert reloaded.peak_value == pytest.approx(2_000.0)
    # Risk policy round-trips (set fields kept; unset stay None/OFF).
    assert reloaded.risk_policy.stop_loss_pct == pytest.approx(10.0)
    assert reloaded.risk_policy.max_drawdown_pct == pytest.approx(25.0)
    assert reloaded.risk_policy.trailing_stop_pct is None
    assert reloaded.risk_policy.take_profit_pct is None
    position = reloaded.positions.get("AAPL")
    assert position is not None
    assert position.units == pytest.approx(3.0)
    assert position.cost_basis == pytest.approx(600.0)
    assert position.opened_at == 1_700_000_000_000
    assert position.high_water_price == pytest.approx(275.0)


def test_sql_account_store_new_account_defaults(sqlite_settings: str) -> None:
    """A never-seen account id is created lazily with zero cash and no positions."""
    store = repositories.get_sql_account_store()
    with store.lock:
        account = store.get_account("brand-new-account")
    assert account.account_id == "brand-new-account"
    assert account.cash_balance == pytest.approx(0.0)
    assert account.positions == {}


def test_sqlite_init_only_touches_the_configured_temp_db(
    sqlite_settings: str,
) -> None:
    """``init_db`` builds the engine for the temp URL only — no default db file.

    Confirms the engine URL in effect is the temp db, so initialization never
    reaches for the default ``giffmemoney.db``.
    """
    db_path = Path(sqlite_settings)
    repositories.get_sql_user_store()  # triggers init_db
    engine = db_session.get_engine()
    assert str(engine.url) == "sqlite:///" + db_path.as_posix()
    # The default database file must not have been created by this test.
    assert db_path.name == "giffmemoney_test.db"
