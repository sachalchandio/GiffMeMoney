# GiffMeMoney — Authentication Contract (Addendum)

> Frozen contract for email/password auth + per-user accounts. Same conventions (camelCase
> wire, Pydantic v2, additive — must not break the 211 passing tests). Built AFTER the
> projection-rigor workflow finishes (it edits backend files).

## Stance
Real auth (hashed passwords + signed tokens), but still a **sandbox/demo** app: no email
verification, no rate-limiting, a dev JWT secret by default. Document this; not production-hardened.

## Backend — `backend/app/auth/`
```
auth/
├─ __init__.py
├─ security.py   # PBKDF2 hashing + JWT (PyJWT)
├─ store.py      # in-memory UserStore (thread-safe), seeds the demo user
├─ service.py    # signup / login / get_user
└─ deps.py       # FastAPI deps: current_user (required) + optional_user (from Bearer)
```
New router `backend/app/api/auth.py` (`router = APIRouter(prefix="/api/auth")`), mounted in main.py via the existing `_mount` pattern.

### security.py
- `hash_password(pw) -> str` — `pbkdf2_hmac('sha256', pw, salt, 200_000)`, stored as `pbkdf2_sha256$200000$<salt_hex>$<hash_hex>` (stdlib only, no bcrypt dep).
- `verify_password(pw, stored) -> bool` (constant-time compare).
- `create_token(user_id, email) -> str` — JWT HS256, `sub=user_id`, `email`, `exp` = now + `settings.jwt_expire_days` (7), secret `settings.jwt_secret`.
- `decode_token(token) -> dict | None` — returns claims or None on invalid/expired.
- Add `pyjwt>=2.8,<3.0` to requirements.txt. Add to config.Settings: `jwt_secret: str = "dev-secret-change-me"`, `jwt_expire_days: int = 7`.

### store.py / service.py
- `User` dataclass: `id` (uuid), `email` (lowercased), `name`, `password_hash`, `created_at` (unix ms).
- `UserStore`: dicts by id + by email, RLock; `get_by_email`, `get_by_id`, `add`. `get_store()` singleton **seeds a demo user** on first use: `demo@giffmemoney.app` / `demo1234`, name "Demo Investor".
- `AuthService(store)`:
  - `signup(email, password, name) -> User` — validate email format + password length ≥ 6; `ValueError` if email already exists or invalid input.
  - `login(email, password) -> User` — `ValueError("Invalid email or password")` on miss/bad password (same message either way).

### deps.py
- `optional_user(authorization: str | None = Header(None)) -> User | None` — parse `Bearer <token>`, decode, return user or None (never raises).
- `current_user(...) -> User` — like optional but raises 401 if missing/invalid.

## DTOs (`schemas.py`, additive, camelCase)
```ts
export interface UserDTO { id: string; email: string; name: string; createdAt: number; }
export interface SignupRequest { email: string; password: string; name: string; }
export interface LoginRequest { email: string; password: string; }
export interface AuthResponse { token: string; user: UserDTO; }
```

## REST (prefix `/api/auth`)
| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/api/auth/signup` | `SignupRequest` | `AuthResponse` (201) |
| POST | `/api/auth/login` | `LoginRequest` | `AuthResponse` |
| GET | `/api/auth/me` | — (Bearer) | `UserDTO` (401 if no/invalid token) |
Errors: 400 invalid signup / duplicate email; 401 bad login / bad token.

## Per-user accounts (wire auth into Invest)
The invest API currently derives the account id from header `X-Account-Id` (default `"demo"`).
Change the account dependency to: **if a valid Bearer token is present → account id = `user:<userId>`;
else fall back to `X-Account-Id` header or `"demo"`.** This gives each logged-in user their own
wallet/positions while keeping anonymous/sandbox access working — so the existing 211 tests (no auth)
still pass unchanged. Implement as a small shared dependency reused by all invest routes.

## Tests (`tests/test_auth.py`)
- Password hash round-trips; wrong password fails; hash is not the plaintext.
- Signup creates a user + returns a token; duplicate email → 400; short password → 400.
- Login good → token+user; login bad → 401; `/me` with token → user, without → 401.
- Demo user is seeded and can log in with the documented credentials.
- Per-user isolation: two different tokens see two different wallets; anonymous still hits `demo`.
- Token decode rejects tampered/expired tokens.

## Frontend
- `lib/auth.ts` + `AuthProvider` (React context): holds `{user, token}`, persisted in `localStorage`
  (`giff_token`); `login()`, `signup()`, `logout()`.
- `api.ts`: attach `Authorization: Bearer <token>` to every request when present.
- Screens: `LoginPage` + `SignupPage` (branded, light/dark) with the demo credentials pre-fillable
  ("Use demo account" button). Protected routes redirect to `/login` when unauthenticated; the app
  shell shows the user's name + a logout control in the top bar.
- Default to the **demo account** so the user can click "Use demo account" → straight into the app.
