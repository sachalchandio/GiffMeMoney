"""Auth use-cases (signup / login / get_user) over the user store.

This is the thin business layer between the HTTP router and the store. It owns
input validation (email shape, password length), the duplicate-email rule, and
password verification. Failures are surfaced as :class:`ValueError` with a
short, user-safe message; the router maps these to HTTP 400/401.

Security notes (sandbox/demo): passwords are PBKDF2-hashed before storage and
never logged; ``login`` returns the **same** error message for an unknown email
and a wrong password so an attacker cannot enumerate registered addresses.
"""

from __future__ import annotations

import re

from app.auth.security import hash_password, verify_password
from app.auth.store import User, UserStore

__all__ = ["AuthService", "MIN_PASSWORD_LENGTH"]

#: Minimum acceptable password length (sandbox policy; >= 6 per the contract).
MIN_PASSWORD_LENGTH = 6

#: Pragmatic email shape check (not RFC-complete; good enough for a sandbox).
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

#: Identical message for both login failure modes (no account enumeration).
_INVALID_LOGIN = "Invalid email or password"


class AuthService:
    """Signup / login / lookup operations backed by a :class:`UserStore`."""

    def __init__(self, store: UserStore) -> None:
        """Bind the service to a user store.

        Args:
            store: The :class:`~app.auth.store.UserStore` holding user records.
        """
        self._store = store

    @staticmethod
    def _validate_email(email: str) -> str:
        """Validate and normalize an email address.

        Args:
            email: The raw email string.

        Returns:
            The trimmed, lowercased email.

        Raises:
            ValueError: If the email is missing or not a plausible address.
        """
        normalized = (email or "").strip().lower()
        if not normalized or not _EMAIL_RE.match(normalized):
            raise ValueError("A valid email address is required")
        return normalized

    @staticmethod
    def _validate_password(password: str) -> str:
        """Validate a plaintext password meets the minimum-length policy.

        Args:
            password: The raw plaintext password.

        Returns:
            The password unchanged (so the caller can hash it).

        Raises:
            ValueError: If the password is missing or too short.
        """
        if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
            raise ValueError(
                f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
            )
        return password

    @staticmethod
    def _validate_name(name: str) -> str:
        """Validate and trim a display name.

        Args:
            name: The raw display name.

        Returns:
            The trimmed name.

        Raises:
            ValueError: If the name is missing/blank.
        """
        cleaned = (name or "").strip()
        if not cleaned:
            raise ValueError("A name is required")
        return cleaned

    def signup(self, email: str, password: str, name: str) -> User:
        """Register a new user after validating the inputs.

        Args:
            email: The email to register (validated + lowercased).
            password: The plaintext password (length >= 6); hashed before
                storage, never persisted/logged raw.
            name: The user's display name.

        Returns:
            The newly created :class:`~app.auth.store.User`.

        Raises:
            ValueError: On invalid input (bad email, short password, blank name)
                or if the email is already registered.
        """
        normalized_email = self._validate_email(email)
        self._validate_password(password)
        cleaned_name = self._validate_name(name)
        # ``store.add`` re-checks under its lock and raises on a duplicate, so
        # the email-uniqueness rule is enforced atomically there.
        return self._store.add(
            email=normalized_email,
            name=cleaned_name,
            password_hash=hash_password(password),
        )

    def login(self, email: str, password: str) -> User:
        """Authenticate a user by email + password.

        Returns the **same** error for an unknown email and a wrong password to
        avoid leaking which addresses are registered.

        Args:
            email: The registered email (case-insensitive).
            password: The plaintext password to verify.

        Returns:
            The matching :class:`~app.auth.store.User`.

        Raises:
            ValueError: ``"Invalid email or password"`` on any miss/mismatch.
        """
        normalized_email = (email or "").strip().lower()
        user = self._store.get_by_email(normalized_email)
        if user is None or not verify_password(password, user.password_hash):
            raise ValueError(_INVALID_LOGIN)
        return user

    def get_user(self, user_id: str) -> User | None:
        """Look up a user by id.

        Args:
            user_id: The opaque user id.

        Returns:
            The matching :class:`~app.auth.store.User`, or ``None``.
        """
        return self._store.get_by_id(user_id)
