"""Password hashing and JWT helpers for GiffMeMoney auth (see docs/AUTH.md).

Two independent concerns live here, both dependency-light:

* **Password hashing** uses the standard library only (``hashlib.pbkdf2_hmac``)
  — no ``bcrypt``/``passlib`` dependency. Hashes are stored in the self-describing
  format ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>`` so the verifier can
  read back the iteration count and salt. Verification uses a constant-time
  comparison (``hmac.compare_digest``) to avoid timing leaks. Raw passwords are
  never logged or persisted — only the derived hash string is kept.
* **Tokens** are JWTs signed with HS256 using :data:`app.config.settings.jwt_secret`.
  :func:`create_token` stamps ``sub`` (user id), ``email`` and an ``exp`` of
  ``now + settings.jwt_expire_days``. :func:`decode_token` verifies the signature
  and expiry and returns the claims, or ``None`` for any invalid/expired/tampered
  token (it never raises).

Stance: sandbox/demo. 200k PBKDF2 iterations is reasonable, but there is no
email verification, no rate-limiting, and the default JWT secret is a dev value.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import jwt

from app.config import settings

__all__ = [
    "hash_password",
    "verify_password",
    "create_token",
    "decode_token",
]

# PBKDF2 parameters. ``_ALGORITHM`` prefixes every stored hash so the format is
# self-describing and future algorithm bumps remain backward-readable.
_ALGORITHM = "pbkdf2_sha256"
_HASH_NAME = "sha256"
_ITERATIONS = 200_000
_SALT_BYTES = 16

#: JWT signing algorithm (HMAC-SHA256). Symmetric secret from settings.
_JWT_ALG = "HS256"


def hash_password(password: str) -> str:
    """Hash a plaintext password with salted PBKDF2-HMAC-SHA256.

    A fresh 16-byte random salt is generated per call, so hashing the same
    password twice yields different strings. The returned value is
    self-describing: ``pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>``.

    Args:
        password: The plaintext password. Must be a non-empty string.

    Returns:
        The encoded hash string, safe to persist.

    Raises:
        ValueError: If ``password`` is not a non-empty string.
    """
    if not isinstance(password, str) or password == "":
        raise ValueError("password must be a non-empty string")
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        _HASH_NAME, password.encode("utf-8"), salt, _ITERATIONS
    )
    return f"{_ALGORITHM}${_ITERATIONS}${salt.hex()}${derived.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2 hash.

    The iteration count and salt are read back from ``stored`` and the candidate
    is re-derived and compared in constant time. Any malformed/unknown stored
    value returns ``False`` rather than raising, so a corrupt record can never
    authenticate and never crashes the caller.

    Args:
        password: The plaintext password to check.
        stored: A hash previously produced by :func:`hash_password`.

    Returns:
        ``True`` iff ``password`` matches ``stored``; ``False`` otherwise.
    """
    if not isinstance(password, str) or not isinstance(stored, str):
        return False
    parts = stored.split("$")
    if len(parts) != 4:
        return False
    algorithm, iterations_str, salt_hex, hash_hex = parts
    if algorithm != _ALGORITHM:
        return False
    try:
        iterations = int(iterations_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except (ValueError, TypeError):
        return False
    if iterations <= 0:
        return False
    candidate = hashlib.pbkdf2_hmac(
        _HASH_NAME, password.encode("utf-8"), salt, iterations
    )
    return hmac.compare_digest(candidate, expected)


def create_token(user_id: str, email: str) -> str:
    """Create a signed JWT (HS256) for an authenticated user.

    The token carries ``sub`` (the user id), ``email``, an issued-at ``iat`` and
    an expiry ``exp`` of ``now + settings.jwt_expire_days``. It is signed with
    :data:`app.config.settings.jwt_secret`.

    Args:
        user_id: The user's opaque id (becomes the ``sub`` claim).
        email: The user's email (echoed in the ``email`` claim).

    Returns:
        The encoded JWT as a ``str``.
    """
    now = datetime.now(timezone.utc)
    expire_days = max(int(settings.jwt_expire_days), 0)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + timedelta(days=expire_days),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=_JWT_ALG)
    # PyJWT >= 2 returns ``str``; guard older byte-returning behavior defensively.
    if isinstance(token, bytes):  # pragma: no cover - PyJWT 2.x returns str
        return token.decode("utf-8")
    return token


def decode_token(token: str) -> Optional[dict[str, Any]]:
    """Decode and verify a JWT, returning its claims or ``None``.

    Verifies both the HS256 signature and the ``exp`` expiry. Any failure —
    bad/empty input, tampered signature, expired token, malformed payload — is
    swallowed and reported as ``None`` so callers never have to catch.

    Args:
        token: The encoded JWT string.

    Returns:
        The decoded claims dict on success, or ``None`` if the token is
        missing, malformed, expired, or has an invalid signature.
    """
    if not isinstance(token, str) or not token:
        return None
    try:
        claims = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[_JWT_ALG],
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError:
        return None
    if not isinstance(claims, dict):
        return None
    return claims
