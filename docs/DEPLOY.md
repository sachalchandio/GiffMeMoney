# GiffMeMoney — Deployment Guide

This guide covers running GiffMeMoney locally, running it with Docker, switching
on **live market data**, connecting the broker to **Alpaca paper trading**, and —
in a deliberately separate, flagged section — the exact gates for enabling
**real-money trading**.

> **Read this first.** GiffMeMoney ships on a **safe default path**: simulated
> market data, a **simulated (paper) broker** (no real money), and in-memory
> storage. Every "go-live" capability is **opt-in via environment variables**.
> The UI always shows whether data is **live or simulated** and whether the
> broker is **paper or live**. There is intentionally **no one-click "enable
> real trading" button** — real trading is a deliberate config action, gated as
> described in the [⚠ ENABLING REAL TRADING](#-enabling-real-trading--at-your-own-risk)
> section below. This is an educational app; nothing here is financial advice
> and trading real markets risks **real loss**.

Environment variables are documented exhaustively in **[`.env.example`](../.env.example)**
at the repository root. Names map 1:1 (case-insensitively) to the settings
fields in `backend/app/config.py` — there is **no prefix** (e.g. `PROVIDER`,
`BROKER`, `ALPACA_LIVE`, `BROKER_ACK`, `PERSIST`, `DB_URL`, `JWT_SECRET`,
`CORS_ORIGINS`).

---

## 1. Run locally (no Docker)

**Prerequisites:** Python 3.11, Node 22.

### Backend

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

`run.py` starts a hot-reloading uvicorn server on `http://0.0.0.0:8000`:

- API docs (Scalar): `http://localhost:8000/docs`
- Swagger UI: `http://localhost:8000/swagger`
- OpenAPI JSON: `http://localhost:8000/openapi.json`
- WebSocket price feed: `ws://localhost:8000/ws`

No environment variables are required — it runs on the built-in market
simulator and the simulated paper broker.

### Frontend

```bash
cd frontend
npm ci
npm run dev
```

The Vite dev server serves the app at `http://localhost:5173` and talks to the
backend at the URL in `frontend/.env` (`VITE_API_URL`, default
`http://127.0.0.1:8000`). On Windows prefer `127.0.0.1` over `localhost` so the
IPv4 backend is reached without an IPv6 (`::1`) detour.

To verify a production build without a dev server:

```bash
cd frontend
npm run build      # tsc -b && vite build -> dist/
npm run test       # vitest run
npm run preview    # serves dist/ at http://localhost:4173
```

---

## 2. Run with Docker Compose

**Prerequisites:** Docker with the Compose plugin.

From the repository root:

```bash
cp .env.example .env      # required: compose loads .env into the backend
docker compose up --build
```

| Service  | URL                     | Notes                                  |
| -------- | ----------------------- | -------------------------------------- |
| backend  | http://localhost:8000   | API, `/docs`, `/ws`                    |
| frontend | http://localhost:8080   | the web app (nginx-served static SPA)  |

What the compose stack does:

- **backend** — builds `backend/Dockerfile` (python:3.11-slim → uvicorn), loads
  every backend variable from `.env`, and publishes port `8000`. A `./data`
  volume is mounted at `/app/data` so an opted-in SQLite DB can persist across
  restarts (see §6).
- **frontend** — builds `frontend/Dockerfile` (multi-stage: node:22-alpine
  build → nginx:alpine serve), publishes port `8080`, and bakes the
  `VITE_API_URL` build arg into the bundle (default `http://localhost:8000`, so
  the browser calls the backend directly).

> The `.env` file **must exist** for `docker compose up` (compose errors on a
> missing `env_file`). An empty `.env` is fine — the app then runs on pure
> defaults (simulated / paper / in-memory).

### Single-origin mode (proxy /api + /ws through nginx)

If you would rather expose only the frontend and have nginx proxy the API,
build the frontend with an empty `VITE_API_URL`:

```bash
VITE_API_URL= docker compose up --build
```

The SPA then calls `/api` and `/ws` on its own origin (`:8080`), and
[`frontend/nginx.conf`](../frontend/nginx.conf) proxies those to the `backend`
service. (In this mode you can stop publishing the backend's `8000` port if you
don't need direct access.)

### CORS

When the browser calls the backend **directly** (the default), the backend must
allow the frontend's origin. `CORS_ORIGINS` in `.env.example` already includes
`http://localhost:8080`. Add any other origin you serve the app from. In
single-origin mode CORS is moot (same origin), but keeping the entry is
harmless.

---

## 3. Switch to LIVE market data

Live market data is **read-only** — it only changes where prices/candles come
from; it never trades. Pick a provider, set its key, and restart the backend.

**Free-tier friendly options:**

- **Finnhub** (equities + fundamentals) — free key at
  <https://finnhub.io/register>.
- **CoinGecko** (crypto) — works without a key on the public tier; a Demo key
  from <https://www.coingecko.com/en/api> raises rate limits.

Set in `.env`:

```ini
PROVIDER=finnhub
FINNHUB_API_KEY=your_key_here
# or:
# PROVIDER=coingecko
# COINGECKO_API_KEY=your_optional_demo_key
```

Then restart (`docker compose up -d --build backend`, or restart `run.py`).

**Honest note on coverage.** Real feeds drive prices, candles, and
latest-price-based strategies. Data that free feeds don't provide cleanly
(e.g. factor series like SMB/HML/risk-free, or missing fundamentals) **falls
back to the deterministic simulator** so the quant engine keeps working — so a
live run is "real prices + simulated-where-unavailable factors", not a
fully-real dataset. Robustness is built in: per-call timeouts, caching to respect
free-tier limits, and a **graceful fallback to the simulator** if a key is
missing or a call fails (the app logs once and never crashes). The UI labels
whether data is **live or simulated** so you always know which you're seeing.

---

## 4. Connect the broker to Alpaca **paper** trading

Alpaca **paper** trading is a sandbox: real order plumbing, **fake money**. This
is the recommended way to exercise the broker layer end-to-end without any risk.

1. Create a free Alpaca account: <https://alpaca.markets>.
2. In the dashboard, switch to **Paper Trading** and generate **paper** API
   keys (an API Key ID and a Secret Key).
3. Set in `.env`:

   ```ini
   BROKER=alpaca
   ALPACA_API_KEY=your_paper_key_id
   ALPACA_SECRET_KEY=your_paper_secret
   ALPACA_BASE_URL=https://paper-api.alpaca.markets   # the PAPER host (default)
   # Leave the live gates OFF (their defaults):
   ALPACA_LIVE=0
   BROKER_ACK=
   ```

4. Restart the backend.

Verify via the API (`GET /api/broker/status`) or the broker screen in the UI:
the mode reads **`paper`**, `paper: true`, and `liveEnabled: false`. Orders
placed through `POST /api/broker/order` go to Alpaca's paper sandbox; every
broker payload carries `paper: true` and the standard disclaimer.

If the keys are missing or Alpaca is unreachable, the broker **fails safe** back
to the built-in simulated paper broker (logged once) — the app still boots and
keeps working.

> Run on paper for a good while before you even consider §5.

---

## ⚠ ENABLING REAL TRADING — AT YOUR OWN RISK

> **STOP AND READ.** Everything below this line concerns placing **real orders
> with real money** against the **live** Alpaca endpoint. If you enable it, the
> app can buy and sell real securities on your behalf and **you can lose real
> money**. This is **not financial advice**. The auto-trader and all projections
> are simulations and **do not predict real results**. Enabling live trading is
> **entirely your deliberate decision and your own risk.**
>
> There is **no UI button** for this and there never will be. Real trading is a
> manual, multi-gate **configuration** action, by design.

**Validate on paper FIRST.** Before going anywhere near live, run the broker on
Alpaca **paper** (§4) for **several weeks**, across different market conditions,
and confirm the orders, positions, P&L, and risk behaviour all match your
expectations. Start with amounts you are fully prepared to lose.

### The exact gates (ALL must hold, together)

Live trading stays **OFF** unless **every one** of these is true at once. If any
single gate is missing, the broker stays on paper and any live order is
**refused with HTTP 403** — by design, so a real order is never placed by
accident.

1. `BROKER=alpaca`
2. `ALPACA_LIVE=1`
3. `ALPACA_BASE_URL=https://api.alpaca.markets`  ← the **LIVE** host (not the
   paper host)
4. Real **live** `ALPACA_API_KEY` + `ALPACA_SECRET_KEY` (your live keys, not
   paper keys)
5. `BROKER_ACK` set to the **exact** phrase, character-for-character:

   ```
   I understand this places real orders
   ```

Even with all five satisfied, the broker requires the **same exact
acknowledgement phrase to be repeated on every individual order** (the per-order
`brokerAck` on `POST /api/broker/order`). A live-armed broker with a missing or
mismatched per-order ack still refuses the order with HTTP 403.

Example `.env` (only when you have read all of the above and accept the risk):

```ini
BROKER=alpaca
ALPACA_LIVE=1
ALPACA_BASE_URL=https://api.alpaca.markets
ALPACA_API_KEY=your_LIVE_key_id
ALPACA_SECRET_KEY=your_LIVE_secret
BROKER_ACK=I understand this places real orders
```

When live is fully armed, the backend logs a **loud startup warning** and
`GET /api/broker/status` reports `mode: "live"`, `paper: false`,
`liveEnabled: true` — the UI surfaces this prominently so a real broker is never
mistaken for a paper one.

**Loss-risk warning (read again):** live orders move real money and can result
in real, total loss. Markets gap, fills slip, and software has bugs. You are
solely responsible for any orders this app places under your configuration and
keys. If in doubt, **stay on paper.**

---

## 5. Configuration reference & operations

### Persistence (optional)

By default the app stores users/accounts/positions **in memory** and resets on
restart. To persist to a **new** SQLite database:

```ini
PERSIST=sqlite
# Local (run.py):
DB_URL=sqlite:///giffmemoney.db
# Docker compose (survives container restarts via the ./data volume):
# DB_URL=sqlite:///data/giffmemoney.db
```

SQLite creates its **own new** database file on first use and **never touches a
pre-existing database** — it only `create_all`s missing tables (no drop / alter
/ migrate). Under Docker, point `DB_URL` at `/app/data/...` so the file lands in
the mounted `./data` volume and survives `docker compose down`/`up`.

### Security checklist for any real deployment

- **`JWT_SECRET`** — change it from the dev default to a long random value, e.g.
  `openssl rand -hex 32`.
- **`CORS_ORIGINS`** — restrict to the exact origins that serve your frontend.
- Put the stack behind HTTPS (a reverse proxy / load balancer terminating TLS)
  before exposing it publicly.
- Treat broker and provider API keys as secrets — never commit `.env`
  (`.env` is git-ignored).

### Health & logs

- Backend liveness: `GET /api/health`. The backend image also defines a Docker
  `HEALTHCHECK` against this route.
- Logs stream to stdout (`docker compose logs -f backend`). Provider/broker
  fallbacks and any live-trading arming are logged there.

---

## File map

| File                          | Purpose                                              |
| ----------------------------- | ---------------------------------------------------- |
| `backend/Dockerfile`          | Backend image (python:3.11-slim → uvicorn).          |
| `frontend/Dockerfile`         | Frontend image (node build → nginx serve).           |
| `frontend/nginx.conf`         | SPA fallback + optional `/api` & `/ws` proxy.        |
| `docker-compose.yml`          | Two-service local/self-host stack.                   |
| `.env.example`                | Every env var, documented, with live-trading warnings.|
| `docs/GOLIVE.md`              | The go-live contract addendum (architecture & safety).|
