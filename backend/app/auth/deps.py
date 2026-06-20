"""FastAPI dependencies for auth (see docs/AUTH.md).

Three reusable dependencies:

* :func:`optional_user` — parse an ``Authorization: Bearer <token>`` header,
  decode it, and return the :class:`~app.auth.store.User` (or ``None``). Never
  raises, so anonymous access keeps working.
* :func:`current_user` — like :func:`optional_user` but raises HTTP 401 when the
  token is missing, malformed, expired, or names a user that no longer exists.
* :func:`account_id` — the shared resolver that wires auth into the invest API:
  a valid Bearer token maps to the per-user account id ``user:<id>``; otherwise
  it falls back to the optional ``X-Account-Id`` header, and finally to
  ``"demo"``. This gives logged-in users isolated wallets while leaving the
  existing anonymous/sandbox tests (no auth) on the shared ``demo`` account.
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from app.auth.security import decode_token
from app.auth.service import AuthService
from app.auth.store import User, get_store

__all__ = [
    "get_auth_service",
    "optional_user",
    "current_user",
    "account_id",
    "DEFAULT_ACCOUNT",
]

#: Account id used for anonymous / sandbox access (no token, no header).
DEFAULT_ACCOUNT = "demo"


def get_auth_service() -> AuthService:
    """Return an :class:`~app.auth.service.AuthService` over the user-store singleton.

    Returns:
        A freshly-bound service sharing the process-wide
        :class:`~app.auth.store.UserStore`.
    """
    return AuthService(get_store())


def _bearer_token(authorization: str | None) -> str | None:
    """Extract the raw token from an ``Authorization`` header value.

    Accepts a case-insensitive ``Bearer`` scheme with a single token argument.
    Anything else (missing header, wrong scheme, empty token) yields ``None``.

    Args:
        authorization: The raw ``Authorization`` header (may be ``None``).

    Returns:
        The bare token string, or ``None`` if no usable Bearer token is present.
    """
    if not authorization:
        return None
    parts = authorization.strip().split(None, 1)
    if len(parts) != 2:
        return None
    scheme, token = parts
    if scheme.lower() != "bearer":
        return None
    token = token.strip()
    return token or None


def _user_from_authorization(
    authorization: str | None, service: AuthService
) -> User | None:
    """Resolve the authenticated user from an ``Authorization`` header, if any.

    Pure helper (no FastAPI raising) shared by :func:`optional_user` and
    :func:`account_id`. Returns ``None`` for a missing/invalid/expired token or
    a token whose ``sub`` no longer maps to a live user.

    Args:
        authorization: The raw ``Authorization`` header value.
        service: The auth service used to look up the user by id.

    Returns:
        The matching :class:`~app.auth.store.User`, or ``None``.
    """
    token = _bearer_token(authorization)
    if token is None:
        return None
    claims = decode_token(token)
    if not claims:
        return None
    user_id = claims.get("sub")
    if not isinstance(user_id, str) or not user_id:
        return None
    return service.get_user(user_id)


def optional_user(
    authorization: str | None = Header(default=None),
    service: AuthService = Depends(get_auth_service),
) -> User | None:
    """FastAPI dep: the current user from a Bearer token, or ``None``.

    Never raises — used by routes that work for both authenticated and
    anonymous callers.

    Args:
        authorization: The ``Authorization`` request header (injected).
        service: The auth service (injected).

    Returns:
        The authenticated :class:`~app.auth.store.User`, or ``None``.
    """
    return _user_from_authorization(authorization, service)


def current_user(
    authorization: str | None = Header(default=None),
    service: AuthService = Depends(get_auth_service),
) -> User:
    """FastAPI dep: the current user, or HTTP 401.

    Args:
        authorization: The ``Authorization`` request header (injected).
        service: The auth service (injected).

    Returns:
        The authenticated :class:`~app.auth.store.User`.

    Raises:
        HTTPException: ``401`` if the token is missing, invalid, expired, or its
            subject is unknown.
    """
    user = _user_from_authorization(authorization, service)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def account_id(
    authorization: str | None = Header(default=None),
    x_account_id: str | None = Header(default=None, alias="X-Account-Id"),
    service: AuthService = Depends(get_auth_service),
) -> str:
    """FastAPI dep: resolve the effective invest account id.

    Resolution order (per docs/AUTH.md):

    1. A valid Bearer token → ``"user:<userId>"`` (the caller's own wallet).
    2. Otherwise the trimmed ``X-Account-Id`` header, if non-blank.
    3. Otherwise :data:`DEFAULT_ACCOUNT` (``"demo"``).

    This keeps anonymous/sandbox callers on the shared ``demo`` account (so the
    pre-auth invest tests still pass) while giving each logged-in user an
    isolated account namespace.

    Args:
        authorization: The ``Authorization`` request header (injected).
        x_account_id: The optional ``X-Account-Id`` header (injected).
        service: The auth service (injected).

    Returns:
        The resolved account id string (never empty).
    """
    user = _user_from_authorization(authorization, service)
    if user is not None:
        return f"user:{user.id}"
    header = (x_account_id or "").strip()
    return header or DEFAULT_ACCOUNT
