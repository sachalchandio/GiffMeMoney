"""Optional SQLite persistence layer for GiffMeMoney (go-live, OPT-IN).

This package is the *opt-in* durable backing for the auth and invest stores. It
is only ever activated when :data:`app.config.settings.persist` is ``'sqlite'``;
the default (``'memory'``) path never touches this package and never imports
SQLAlchemy, so the running app and the full test suite are unaffected.

Safety contract (see ``docs/GOLIVE.md`` §3):

* When ``persist == 'sqlite'`` the app creates / opens its **own** database file
  (from ``settings.db_url`` / the ``DB_URL`` env var, default
  ``sqlite:///giffmemoney.db``) and calls :func:`app.db.session.init_db` exactly
  once to ``create_all()`` the tables.
  It performs **no** auto schema-sync, drop, or migration beyond that first
  create — it never mutates a pre-existing schema.
* The SQL-backed stores (:class:`app.db.repositories.SqlUserStore`,
  :class:`app.db.repositories.SqlAccountStore`) implement the **same** public
  surface as the in-memory ``auth.store`` / ``invest.store``, so services are
  oblivious to which backend is active.

Modules:
    * :mod:`app.db.models`        — SQLAlchemy 2.0 ORM row models.
    * :mod:`app.db.session`       — engine + sessionmaker + ``init_db``.
    * :mod:`app.db.repositories`  — the SQL-backed store implementations.
"""

from __future__ import annotations

__all__: list[str] = []
