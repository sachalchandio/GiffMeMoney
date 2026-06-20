"""Authentication package for the GiffMeMoney backend.

Sandbox/demo auth (see ``docs/AUTH.md``): real PBKDF2-hashed passwords and
HS256-signed JWTs, but no email verification, no rate-limiting, and a dev
signing secret by default — **not production-hardened**.

Modules:
    * :mod:`app.auth.security` — PBKDF2 password hashing + JWT create/decode.
    * :mod:`app.auth.store`    — the in-memory, thread-safe ``UserStore`` (seeds
      the demo user) and its ``get_store`` singleton.
    * :mod:`app.auth.service`  — ``AuthService`` (signup / login / get_user).
    * :mod:`app.auth.deps`     — FastAPI dependencies: ``optional_user``,
      ``current_user`` and the shared ``account_id`` resolver wiring auth into
      the invest routes.
"""

from __future__ import annotations

from app.auth.security import (
    create_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.auth.service import AuthService
from app.auth.store import User, UserStore, get_store

__all__ = [
    "hash_password",
    "verify_password",
    "create_token",
    "decode_token",
    "User",
    "UserStore",
    "get_store",
    "AuthService",
]
