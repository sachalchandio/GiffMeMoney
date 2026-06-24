"""Engine, session factory and one-time schema init for the SQLite backend.

This module owns the SQLAlchemy :class:`~sqlalchemy.engine.Engine` and the
:func:`~sqlalchemy.orm.sessionmaker`-built :data:`SessionLocal`, both derived
from :data:`app.config.settings.db_url`.

Safety contract (``docs/GOLIVE.md`` §3):

* Nothing here runs unless a caller explicitly asks for it. The engine is built
  lazily on first use, and the schema is created **only** by
  :func:`init_db`, which the store factory invokes exactly once when
  ``settings.persist == 'sqlite'``.
* :func:`init_db` calls ``Base.metadata.create_all()``, which is a no-op for any
  table that already exists — it **never** drops, alters, or migrates an
  existing schema. The app thus initializes its *own* new database file without
  ever mutating a pre-existing one.
"""

from __future__ import annotations

import threading
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.db.models import Base

__all__ = [
    "get_engine",
    "get_sessionmaker",
    "SessionLocal",
    "init_db",
    "reset_engine_cache",
]

_LOCK = threading.Lock()
_ENGINE: Optional[Engine] = None
_SESSIONMAKER: Optional[sessionmaker[Session]] = None
_INITIALIZED = False


def _build_engine(db_url: str) -> Engine:
    """Create a SQLAlchemy engine for ``db_url``.

    For SQLite a connection is created per-thread by default; we pass
    ``check_same_thread=False`` so the engine can be shared across the request /
    WebSocket threads (every store access still serializes through the store's
    own re-entrant lock, mirroring the in-memory store's threading model).

    Args:
        db_url: A SQLAlchemy URL (e.g. ``sqlite:///giffmemoney.db``).

    Returns:
        A configured :class:`~sqlalchemy.engine.Engine`.
    """
    connect_args: dict[str, object] = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(db_url, future=True, connect_args=connect_args)


def get_engine() -> Engine:
    """Return the process-wide engine, building it lazily on first use.

    The URL comes from :data:`app.config.settings.db_url`. The engine is
    memoized so every store and session shares one connection pool.

    Returns:
        The shared :class:`~sqlalchemy.engine.Engine`.
    """
    global _ENGINE, _SESSIONMAKER
    if _ENGINE is None:
        with _LOCK:
            if _ENGINE is None:
                _ENGINE = _build_engine(settings.db_url)
                _SESSIONMAKER = sessionmaker(
                    bind=_ENGINE, expire_on_commit=False, future=True
                )
    return _ENGINE


def get_sessionmaker() -> sessionmaker[Session]:
    """Return the process-wide session factory (building the engine if needed).

    Returns:
        A :func:`~sqlalchemy.orm.sessionmaker` bound to the shared engine, with
        ``expire_on_commit=False`` so committed objects stay usable.
    """
    if _SESSIONMAKER is None:
        get_engine()
    assert _SESSIONMAKER is not None  # set by get_engine()
    return _SESSIONMAKER


def SessionLocal() -> Session:
    """Open a new :class:`~sqlalchemy.orm.Session` from the shared factory.

    Returns:
        A fresh session the caller is responsible for closing (use as a context
        manager).
    """
    return get_sessionmaker()()


def init_db() -> None:
    """Create the schema for the configured database (idempotent).

    Calls ``Base.metadata.create_all()`` on the shared engine, which creates any
    missing tables and leaves existing ones untouched (no drop / alter / migrate).
    Safe to call repeatedly; the actual ``create_all`` runs at most once per
    process.

    This is invoked **only** by the store factory when
    ``settings.persist == 'sqlite'``; it is never called on the default
    in-memory path.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    # Build the engine *outside* ``_LOCK`` — ``get_engine`` takes ``_LOCK``
    # itself, and ``_LOCK`` is a plain (non-reentrant) lock, so acquiring it
    # here first would deadlock.
    engine = get_engine()
    with _LOCK:
        if _INITIALIZED:
            return
        Base.metadata.create_all(engine)
        _INITIALIZED = True


def reset_engine_cache() -> None:
    """Drop the cached engine / sessionmaker / init flag (test helper).

    Lets a test point the engine at a fresh ``db_url`` (e.g. a temp file).
    Disposes the old engine if one exists. Not used in production code paths.
    """
    global _ENGINE, _SESSIONMAKER, _INITIALIZED
    with _LOCK:
        if _ENGINE is not None:
            _ENGINE.dispose()
        _ENGINE = None
        _SESSIONMAKER = None
        _INITIALIZED = False
