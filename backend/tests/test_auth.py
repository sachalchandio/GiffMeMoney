"""Tests for the email/password + JWT auth extension (docs/AUTH.md).

These cover the whole sandbox auth surface, additively, without breaking the
pre-auth suite (anonymous access must keep hitting the shared ``demo`` account):

* **Security primitives** — :func:`~app.auth.security.hash_password` round-trips
  through :func:`~app.auth.security.verify_password`, a wrong password fails, the
  stored hash is never the plaintext, and :func:`~app.auth.security.decode_token`
  rejects tampered / expired / wrong-secret tokens (returning ``None``, never
  raising).
* **Signup** — creates a user and returns a token + public user DTO (201),
  lowercases the email, rejects a duplicate email (400) and a too-short
  password (400).
* **Login** — good credentials return a token + user (200); a wrong password or
  unknown email returns 401 with the *same* (non-enumerating) message.
* **/me** — a valid Bearer token returns the user; a missing, malformed, or
  tampered token returns 401.
* **Demo user** — the documented ``demo@giffmemoney.app`` / ``demo1234`` account
  is seeded and loginable.
* **Per-user isolation** — two different tokens see two different wallets, while
  an anonymous (no-token) caller still resolves to the shared ``demo`` account.

Speed / isolation: API tests run against a single module-scoped
:class:`~fastapi.testclient.TestClient` (one lifespan / tick-loop) and register
**unique random emails** per test, so they never collide with the seeded demo
user, with each other, or with the rest of the suite. No full-universe analysis
sweep is performed. JWT helpers are exercised directly for the crypto-level
assertions (no HTTP needed).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient

from app.auth.security import (
    create_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.auth.service import MIN_PASSWORD_LENGTH
from app.auth.store import DEMO_EMAIL, DEMO_NAME, DEMO_PASSWORD
from app.config import settings
from app.main import app

# A Luhn-valid Visa test PAN and a far-future expiry for the isolation deposit.
VALID_VISA = "4111111111111111"
EXP_MONTH = 12
EXP_YEAR = 2030


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """A module-scoped TestClient that runs the app lifespan (tick loop)."""
    with TestClient(app) as test_client:
        yield test_client


def _email() -> str:
    """Return a unique, never-before-seen email so signups never collide."""
    return f"user-{uuid.uuid4().hex[:12]}@giffmemoney.test"


def _card() -> dict[str, object]:
    """Return a camelCase, Luhn-valid card payload for a simulated deposit."""
    return {
        "number": VALID_VISA,
        "expMonth": EXP_MONTH,
        "expYear": EXP_YEAR,
        "cvc": "123",
        "holder": "Ada Lovelace",
    }


def _signup(client: TestClient, email: str, password: str = "secret123") -> dict:
    """Sign up a fresh user and return the parsed ``AuthResponse`` body."""
    resp = client.post(
        "/api/auth/signup",
        json={"email": email, "password": password, "name": "Test User"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _auth(token: str) -> dict[str, str]:
    """Return an ``Authorization: Bearer <token>`` header dict."""
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Password hashing (security.py)
# ---------------------------------------------------------------------------


def test_hash_password_round_trips() -> None:
    """A hashed password verifies against the original plaintext."""
    stored = hash_password("hunter2!")
    assert verify_password("hunter2!", stored) is True


def test_verify_rejects_wrong_password() -> None:
    """The wrong password does not verify against a stored hash."""
    stored = hash_password("correct horse battery staple")
    assert verify_password("wrong password", stored) is False


def test_hash_is_not_plaintext() -> None:
    """The stored hash never contains (and is not equal to) the raw password."""
    password = "s3cr3t-passphrase"
    stored = hash_password(password)
    assert stored != password
    assert password not in stored
    # Self-describing PBKDF2 format: algorithm$iterations$salt$hash.
    assert stored.startswith("pbkdf2_sha256$")
    assert len(stored.split("$")) == 4


def test_hash_is_salted_unique_per_call() -> None:
    """Hashing the same password twice yields different (salted) strings."""
    first = hash_password("same-password")
    second = hash_password("same-password")
    assert first != second
    # ...yet both still verify against the original plaintext.
    assert verify_password("same-password", first) is True
    assert verify_password("same-password", second) is True


# ---------------------------------------------------------------------------
# JWT create/decode (security.py)
# ---------------------------------------------------------------------------


def test_token_round_trips_claims() -> None:
    """A freshly created token decodes back to its subject + email claims."""
    token = create_token("user-123", "person@example.com")
    claims = decode_token(token)
    assert claims is not None
    assert claims["sub"] == "user-123"
    assert claims["email"] == "person@example.com"
    assert "exp" in claims


def test_decode_rejects_tampered_token() -> None:
    """A token with a flipped trailing character (broken signature) → None."""
    token = create_token("user-123", "person@example.com")
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    assert tampered != token
    assert decode_token(tampered) is None


def test_decode_rejects_garbage_and_empty() -> None:
    """Malformed / empty tokens decode to None rather than raising."""
    assert decode_token("not-a-jwt") is None
    assert decode_token("") is None


def test_decode_rejects_expired_token() -> None:
    """A correctly-signed but expired token is rejected (exp enforced)."""
    expired = jwt.encode(
        {
            "sub": "user-123",
            "email": "person@example.com",
            "exp": datetime.now(timezone.utc) - timedelta(days=1),
        },
        settings.jwt_secret,
        algorithm="HS256",
    )
    assert decode_token(expired) is None


def test_decode_rejects_wrong_secret() -> None:
    """A token signed with a different secret fails signature verification."""
    forged = jwt.encode(
        {
            "sub": "user-123",
            "email": "person@example.com",
            "exp": datetime.now(timezone.utc) + timedelta(days=1),
        },
        settings.jwt_secret + "-not-the-real-secret",
        algorithm="HS256",
    )
    assert decode_token(forged) is None


# ---------------------------------------------------------------------------
# Signup
# ---------------------------------------------------------------------------


def test_signup_creates_user_and_returns_token(client: TestClient) -> None:
    """POST /api/auth/signup → 201 with a token and the public user DTO."""
    email = _email()
    resp = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "secret123", "name": "Grace Hopper"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert set(body.keys()) == {"token", "user"}
    assert isinstance(body["token"], str) and body["token"]
    user = body["user"]
    # camelCase public DTO; never carries the password hash.
    assert set(user.keys()) == {"id", "email", "name", "createdAt"}
    assert user["email"] == email
    assert user["name"] == "Grace Hopper"
    assert "password" not in user and "passwordHash" not in user
    # The issued token names this user and round-trips.
    claims = decode_token(body["token"])
    assert claims is not None and claims["sub"] == user["id"]


def test_signup_lowercases_email(client: TestClient) -> None:
    """A mixed-case email is normalized to lowercase on signup."""
    local = uuid.uuid4().hex[:12]
    resp = client.post(
        "/api/auth/signup",
        json={
            "email": f"Mixed.{local}@Giffmemoney.TEST",
            "password": "secret123",
            "name": "Casey",
        },
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["user"]["email"] == f"mixed.{local}@giffmemoney.test"


def test_signup_duplicate_email_returns_400(client: TestClient) -> None:
    """Re-registering an existing email → 400 (no second account)."""
    email = _email()
    first = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "secret123", "name": "First"},
    )
    assert first.status_code == 201, first.text
    dup = client.post(
        "/api/auth/signup",
        json={"email": email, "password": "secret123", "name": "Second"},
    )
    assert dup.status_code == 400


def test_signup_short_password_returns_400(client: TestClient) -> None:
    """A password shorter than the minimum length → 400."""
    short = "x" * (MIN_PASSWORD_LENGTH - 1)
    resp = client.post(
        "/api/auth/signup",
        json={"email": _email(), "password": short, "name": "Shorty"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


def test_login_good_credentials_returns_token_and_user(client: TestClient) -> None:
    """Correct email + password → 200 with a token and the matching user."""
    email = _email()
    _signup(client, email, password="secret123")
    resp = client.post(
        "/api/auth/login", json={"email": email, "password": "secret123"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert set(body.keys()) == {"token", "user"}
    assert body["user"]["email"] == email
    assert isinstance(body["token"], str) and body["token"]


def test_login_is_case_insensitive_on_email(client: TestClient) -> None:
    """Login succeeds regardless of the email's case."""
    email = _email()
    _signup(client, email, password="secret123")
    resp = client.post(
        "/api/auth/login",
        json={"email": email.upper(), "password": "secret123"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["user"]["email"] == email


def test_login_wrong_password_returns_401(client: TestClient) -> None:
    """A registered email with a wrong password → 401."""
    email = _email()
    _signup(client, email, password="secret123")
    resp = client.post(
        "/api/auth/login", json={"email": email, "password": "wrong-password"}
    )
    assert resp.status_code == 401


def test_login_unknown_email_returns_401_same_message(client: TestClient) -> None:
    """An unknown email and a wrong password return the *same* 401 message.

    The identical detail prevents account enumeration (no leak of which emails
    are registered).
    """
    email = _email()
    _signup(client, email, password="secret123")
    bad_password = client.post(
        "/api/auth/login", json={"email": email, "password": "nope-nope"}
    )
    unknown = client.post(
        "/api/auth/login",
        json={"email": _email(), "password": "secret123"},
    )
    assert bad_password.status_code == 401
    assert unknown.status_code == 401
    assert bad_password.json()["detail"] == unknown.json()["detail"]


# ---------------------------------------------------------------------------
# /me (current user probe)
# ---------------------------------------------------------------------------


def test_me_with_valid_token_returns_user(client: TestClient) -> None:
    """GET /api/auth/me with a valid Bearer token returns the user DTO."""
    email = _email()
    body = _signup(client, email)
    resp = client.get("/api/auth/me", headers=_auth(body["token"]))
    assert resp.status_code == 200, resp.text
    me = resp.json()
    assert set(me.keys()) == {"id", "email", "name", "createdAt"}
    assert me["email"] == email
    assert me["id"] == body["user"]["id"]
    assert "passwordHash" not in me


def test_me_without_token_returns_401(client: TestClient) -> None:
    """GET /api/auth/me without an Authorization header → 401."""
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_me_with_invalid_token_returns_401(client: TestClient) -> None:
    """GET /api/auth/me with a malformed Bearer token → 401."""
    resp = client.get("/api/auth/me", headers=_auth("not.a.real.jwt"))
    assert resp.status_code == 401


def test_me_with_tampered_token_returns_401(client: TestClient) -> None:
    """A token whose signature has been tampered with → 401."""
    body = _signup(client, _email())
    token = body["token"]
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")
    resp = client.get("/api/auth/me", headers=_auth(tampered))
    assert resp.status_code == 401


def test_me_with_expired_token_returns_401(client: TestClient) -> None:
    """A correctly-signed but expired token → 401 at /me."""
    expired = jwt.encode(
        {
            "sub": uuid.uuid4().hex,
            "email": _email(),
            "exp": datetime.now(timezone.utc) - timedelta(days=1),
        },
        settings.jwt_secret,
        algorithm="HS256",
    )
    resp = client.get("/api/auth/me", headers=_auth(expired))
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Seeded demo user
# ---------------------------------------------------------------------------


def test_demo_user_is_seeded_and_loginable(client: TestClient) -> None:
    """The documented demo account logs in with the published credentials."""
    resp = client.post(
        "/api/auth/login",
        json={"email": DEMO_EMAIL, "password": DEMO_PASSWORD},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["user"]["email"] == DEMO_EMAIL
    assert body["user"]["name"] == DEMO_NAME
    # The demo token works against the protected /me probe.
    me = client.get("/api/auth/me", headers=_auth(body["token"]))
    assert me.status_code == 200
    assert me.json()["email"] == DEMO_EMAIL


# ---------------------------------------------------------------------------
# Per-user account isolation (auth wired into the invest API)
# ---------------------------------------------------------------------------


def test_two_tokens_see_two_isolated_wallets(client: TestClient) -> None:
    """Each logged-in user gets a private ``user:<id>`` wallet.

    A deposit on user A's token must not appear on user B's wallet, and the two
    accounts must carry distinct ``accountId`` namespaces.
    """
    token_a = _signup(client, _email())["token"]
    token_b = _signup(client, _email())["token"]
    headers_a = _auth(token_a)
    headers_b = _auth(token_b)

    deposit = client.post(
        "/api/wallet/deposit",
        json={"amount": 321.0, "card": _card()},
        headers=headers_a,
    )
    assert deposit.status_code == 200, deposit.text

    wallet_a = client.get("/api/wallet", headers=headers_a).json()
    wallet_b = client.get("/api/wallet", headers=headers_b).json()

    assert wallet_a["accountId"] != wallet_b["accountId"]
    assert wallet_a["accountId"].startswith("user:")
    assert wallet_b["accountId"].startswith("user:")
    assert wallet_a["cashBalance"] == pytest.approx(321.0)
    # User B never saw user A's deposit.
    assert wallet_b["cashBalance"] == pytest.approx(0.0)


def test_anonymous_still_resolves_to_demo_account(client: TestClient) -> None:
    """A request with no token (and no header) falls back to the ``demo`` account.

    This preserves the pre-auth behavior the existing invest tests rely on.
    """
    wallet = client.get("/api/wallet").json()
    assert wallet["accountId"] == "demo"


def test_token_account_differs_from_anonymous(client: TestClient) -> None:
    """A logged-in caller's account id is namespaced away from ``demo``."""
    token = _signup(client, _email())["token"]
    wallet = client.get("/api/wallet", headers=_auth(token)).json()
    assert wallet["accountId"] != "demo"
    assert wallet["accountId"].startswith("user:")
