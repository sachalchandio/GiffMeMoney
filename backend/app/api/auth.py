"""``/api/auth`` — email/password signup, login and the current-user probe.

Sandbox/demo auth (see ``docs/AUTH.md``): real PBKDF2-hashed passwords and
HS256-signed JWTs, but no email verification and no rate-limiting. The router is
a thin, defensive adapter over :class:`~app.auth.service.AuthService`.

Error mapping (per the contract):
    * ``POST /signup`` → ``201`` on success; ``400`` on invalid input or a
      duplicate email (service ``ValueError``).
    * ``POST /login``  → ``200`` on success; ``401`` on bad credentials.
    * ``GET  /me``     → ``200`` with the user; ``401`` if the Bearer token is
      missing/invalid/expired (enforced by the ``current_user`` dependency).

Responses never include the password hash — only the public
:class:`~app.schemas.UserDTO` is returned.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.deps import current_user, get_auth_service
from app.auth.service import AuthService
from app.auth.security import create_token
from app.auth.store import User
from app.schemas import AuthResponse, LoginRequest, SignupRequest, UserDTO

__all__ = ["router"]

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _auth_response(user: User) -> AuthResponse:
    """Build the ``{token, user}`` envelope for a freshly authenticated user.

    Args:
        user: The authenticated :class:`~app.auth.store.User`.

    Returns:
        An :class:`~app.schemas.AuthResponse` with a signed token and the
        public user DTO.
    """
    token = create_token(user.id, user.email)
    return AuthResponse(token=token, user=user.to_dto())


@router.post(
    "/signup",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new account and return a signed token",
)
def signup(
    body: SignupRequest,
    service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    """Create a new user and return a token + the public user record.

    Args:
        body: The :class:`~app.schemas.SignupRequest` (email, password, name).
        service: The auth service (injected).

    Returns:
        An :class:`~app.schemas.AuthResponse` (HTTP 201).

    Raises:
        HTTPException: ``400`` for invalid input (bad email / short password /
            blank name) or a duplicate email.
    """
    try:
        user = service.signup(body.email, body.password, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return _auth_response(user)


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Authenticate with email + password and return a signed token",
)
def login(
    body: LoginRequest,
    service: AuthService = Depends(get_auth_service),
) -> AuthResponse:
    """Verify credentials and return a token + the public user record.

    Args:
        body: The :class:`~app.schemas.LoginRequest` (email, password).
        service: The auth service (injected).

    Returns:
        An :class:`~app.schemas.AuthResponse`.

    Raises:
        HTTPException: ``401`` for an unknown email or a wrong password (same
            message either way — no account enumeration).
    """
    try:
        user = service.login(body.email, body.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    return _auth_response(user)


@router.get(
    "/me",
    response_model=UserDTO,
    summary="Return the current authenticated user",
)
def me(user: User = Depends(current_user)) -> UserDTO:
    """Return the public record for the Bearer-authenticated user.

    Args:
        user: The current user, resolved from the Bearer token (injected); the
            dependency raises ``401`` when the token is missing/invalid.

    Returns:
        The :class:`~app.schemas.UserDTO` for the authenticated account.
    """
    return user.to_dto()
