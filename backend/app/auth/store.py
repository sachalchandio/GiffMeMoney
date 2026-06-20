"""In-memory, thread-safe user store for GiffMeMoney auth (see docs/AUTH.md).

Holds all user records (id, email, name, password hash, created-at) in process
memory only — no database, no persistence; state resets on restart. The store
is indexed by both id and (lowercased) email and guarded by a re-entrant lock so
concurrent request threads never observe a half-applied write.

The :func:`get_store` singleton **seeds a demo user** on first construction
(``demo@giffmemoney.app`` / ``demo1234``, "Demo Investor") so the sandbox app is
usable immediately. Password hashes are stored, never the raw passwords.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

from app.auth.security import hash_password
from app.schemas import UserDTO

__all__ = [
    "User",
    "UserStore",
    "get_store",
    "DEMO_EMAIL",
    "DEMO_PASSWORD",
    "DEMO_NAME",
]

#: Seeded demo account credentials (documented; sandbox only).
DEMO_EMAIL = "demo@giffmemoney.app"
DEMO_PASSWORD = "demo1234"
DEMO_NAME = "Demo Investor"


@dataclass
class User:
    """A single registered user.

    Attributes:
        id: Opaque user id (uuid4 hex/string).
        email: The user's email address, stored lowercased.
        name: The user's display name.
        password_hash: The PBKDF2 hash string (never the raw password).
        created_at: Unix timestamp in milliseconds when the user was created.
    """

    id: str
    email: str
    name: str
    password_hash: str
    created_at: int

    def to_dto(self) -> UserDTO:
        """Project this record onto the public-facing :class:`~app.schemas.UserDTO`.

        The password hash is intentionally omitted — it is never exposed on the
        wire.

        Returns:
            A :class:`~app.schemas.UserDTO` carrying only public fields.
        """
        return UserDTO(
            id=self.id,
            email=self.email,
            name=self.name,
            created_at=self.created_at,
        )


class UserStore:
    """Process-wide, thread-safe registry of :class:`User` records.

    Users are indexed by id and by lowercased email. All reads and writes occur
    under :attr:`lock` (re-entrant), so a fresh duplicate-email signup race
    cannot create two accounts for the same address.
    """

    def __init__(self) -> None:
        """Initialize an empty store with a re-entrant lock."""
        self.lock: threading.RLock = threading.RLock()
        self._by_id: dict[str, User] = {}
        self._by_email: dict[str, User] = {}

    @staticmethod
    def _norm_email(email: str) -> str:
        """Normalize an email for indexing/lookups (trim + lowercase).

        Args:
            email: A raw email string (possibly ``None``-like / mixed case).

        Returns:
            The trimmed, lowercased email (empty string if input is falsy).
        """
        return (email or "").strip().lower()

    def get_by_email(self, email: str) -> User | None:
        """Return the user with the given email, or ``None`` if absent.

        Args:
            email: The email to look up (case-insensitive, whitespace-trimmed).

        Returns:
            The matching :class:`User`, or ``None``.
        """
        key = self._norm_email(email)
        if not key:
            return None
        with self.lock:
            return self._by_email.get(key)

    def get_by_id(self, user_id: str) -> User | None:
        """Return the user with the given id, or ``None`` if absent.

        Args:
            user_id: The opaque user id.

        Returns:
            The matching :class:`User`, or ``None``.
        """
        if not user_id:
            return None
        with self.lock:
            return self._by_id.get(user_id)

    def add(self, email: str, name: str, password_hash: str) -> User:
        """Create and register a new user.

        The email is normalized (trimmed + lowercased) before indexing. A fresh
        uuid id and a millisecond ``created_at`` are assigned.

        Args:
            email: The user's email (will be normalized).
            name: The user's display name.
            password_hash: A pre-computed PBKDF2 hash string.

        Returns:
            The newly created :class:`User`.

        Raises:
            ValueError: If a user with the same (normalized) email already
                exists.
        """
        key = self._norm_email(email)
        with self.lock:
            if key in self._by_email:
                raise ValueError("Email already registered")
            user = User(
                id=uuid.uuid4().hex,
                email=key,
                name=name,
                password_hash=password_hash,
                created_at=int(time.time() * 1000),
            )
            self._by_id[user.id] = user
            self._by_email[key] = user
            return user


# ---------------------------------------------------------------------------
# Singleton (seeds the demo user on first use)
# ---------------------------------------------------------------------------

_STORE_LOCK = threading.Lock()
_STORE_INSTANCE: UserStore | None = None


def _seed_demo_user(store: UserStore) -> None:
    """Seed the documented demo account into a fresh store.

    Idempotent: if the demo email is already present (e.g. the store was seeded
    earlier) nothing happens.

    Args:
        store: The :class:`UserStore` to seed.
    """
    if store.get_by_email(DEMO_EMAIL) is None:
        store.add(
            email=DEMO_EMAIL,
            name=DEMO_NAME,
            password_hash=hash_password(DEMO_PASSWORD),
        )


def get_store() -> UserStore:
    """Return the process-wide :class:`UserStore` singleton.

    Constructed lazily and memoized so every auth service and request shares the
    same in-memory user table. The demo user is seeded exactly once, on first
    construction.

    Returns:
        The shared :class:`UserStore` instance.
    """
    global _STORE_INSTANCE
    if _STORE_INSTANCE is None:
        with _STORE_LOCK:
            if _STORE_INSTANCE is None:
                store = UserStore()
                _seed_demo_user(store)
                _STORE_INSTANCE = store
    return _STORE_INSTANCE
