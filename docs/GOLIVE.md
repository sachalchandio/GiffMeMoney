# GiffMeMoney — Go-Live Infrastructure (Contract Addendum)

> Turns the simulator into a system that *can* trade real markets — pluggable live data, a broker
> execution layer (paper-first), persistence, and deployment. **Every new capability is OPT-IN via
> env and DEFAULTS to the safe path** (simulated data · simulated/paper broker · in-memory storage),
> so the 255+ backend tests and the running app are unaffected. Additive only.

## ⚠️ Safety (non-negotiable, enforced in code)
- **Broker defaults to `simulated`.** The real-broker adapter (Alpaca) defaults to Alpaca's **paper**
  endpoint (`https://paper-api.alpaca.markets` — sandbox, no real money). **LIVE trading is hard-gated**:
  it requires BOTH `BROKER=alpaca`, real keys, `ALPACA_LIVE=1`, AND `BROKER_ACK="I understand
  this places real orders"`. If live is configured, log a loud startup warning. No order path runs live
  without all gates. (This repo ships with live OFF.)
- **Live market data is read-only.** No write/trade capability in the data layer.
- **Persistence is opt-in** (`PERSIST=memory` default). When `sqlite`, the app creates its OWN new
  `giffmemoney.db`; it never touches any pre-existing database. (No auto schema-sync unless opted in.)
- Disclaimers stay on every projection/bot result. Not financial advice; trading risks real loss.

## 1. Live market data — `backend/app/market/providers/`
Implement the EXISTING `MarketDataProvider` interface (list_assets, get_asset, get_candles, history,
market_history, factor_history, fundamentals, latest_price) with real backends; `get_provider()` selects
by `settings.provider`:
- `simulated` (default, unchanged).
- `finnhub` (equities + fundamentals), `polygon` (equities/aggregates), `coingecko` (crypto), `binance` (crypto).
- A **HybridProvider**: real prices/candles/latest from the chosen backend for its symbol universe; for
  data real feeds don't give cleanly (SMB/HML/rf factor series, missing fundamentals) **fall back to the
  deterministic simulator values** so the quant engine keeps working. Document this approximation honestly
  (price-driven strategies use real data; factor/fundamental models use real-where-available + simulated).
- Robustness: per-call timeout, on-disk/in-memory cache (respect free-tier rate limits), and **graceful
  fallback to `simulated` when a key is missing or a call fails** (never crash; log once). Keys via env
  (`FINNHUB_API_KEY`, `POLYGON_API_KEY`, `COINGECKO_API_KEY`, `BINANCE_API_KEY`) already in config.
- Add `httpx` calls (already a dep). No new heavy deps.

## 2. Broker execution layer — `backend/app/broker/`
- `base.py`: `BrokerProvider` ABC — `get_account()->BrokerAccount{cash,equity,buyingPower,mode}`,
  `get_positions()->BrokerPosition[]`, `submit_order(symbol, side, notional|qty, type='market')->BrokerOrder`,
  `list_orders()`, `cancel_order(id)`, `is_paper:bool`.
- `simulated.py`: `SimulatedBroker` (default) — paper fills against the current provider price; wraps the
  existing invest engine so today's behavior is preserved.
- `alpaca.py`: `AlpacaBroker` — REST against Alpaca; **base URL defaults to the paper endpoint**; live only
  under the hard-gate above. Maps account/positions/orders to the DTOs.
- `__init__.py`: `get_broker()` factory by `settings.broker` (default `simulated`), with the live-gate check.
- API `backend/app/api/broker.py` (router `/api/broker`): GET `/status` (mode, isPaper, connected),
  GET `/account`, GET `/positions`, GET `/orders`, POST `/order` (place a PAPER order; 403 + clear message
  if live is not fully acknowledged). Mount in main.py. Every response carries a `paper:true/false` + disclaimer.
- Bot/invest can optionally route through the broker when configured; default path unchanged.

## 3. Persistence — `backend/app/db/`
- `models.py` (SQLAlchemy 2.0): UserRow, AccountRow, PositionRow, TransactionRow, SavedCardRow, BotRunRow.
- `session.py`: engine + sessionmaker for `sqlite:///giffmemoney.db` (path from `DB_URL`); `init_db()`
  calls `create_all()` — invoked ONLY when `settings.persist=='sqlite'`.
- `repositories.py`: SqlUserStore + SqlAccountStore implementing the SAME interfaces as the in-memory
  `auth/store.py` + `invest/store.py`.
- Wire `auth.store.get_store()` / `invest.store.get_store()` to return the SQL-backed store when
  `settings.persist=='sqlite'`, else the in-memory one (default). Tests stay on memory.
- Add `sqlalchemy>=2.0,<3.0` to requirements.

## 4. Deployment — repo root + `docs/DEPLOY.md`
- `backend/Dockerfile` (python:3.11-slim → install requirements → `uvicorn app.main:app`).
- `frontend/Dockerfile` (node build → static serve via `nginx:alpine`), `frontend/nginx.conf`.
- `docker-compose.yml` (backend + frontend; env-file wired; ports 8000/5173→80).
- `.env.example` (root) documenting EVERY env var (provider + keys, broker mode + keys + live gates,
  persist + db url, jwt secret, cors).
- `docs/DEPLOY.md`: local run, Docker run, switching to live data, switching the broker to Alpaca **paper**,
  and a clearly-marked "ENABLING REAL TRADING (do at your own risk)" section listing the exact gates — framed
  as the user's deliberate action, with the loss-risk warning.

## 5. Config (`backend/app/config.py`, additive)
Add: `broker: str='simulated'`, `alpaca_api_key/secret: Optional[str]`, `alpaca_base_url: str=paper`,
`alpaca_live: bool=False`, `broker_ack: Optional[str]`, `persist: str='memory'`, `db_url: str='sqlite:///giffmemoney.db'`.
(provider + data keys already exist.) All env-overridable; defaults are the safe path.

## 6. Tests
- providers: with no key → falls back to simulated (no network); a mocked httpx response maps correctly.
- broker: SimulatedBroker places a paper order + reflects it in positions; live-gate returns 403 without acks.
- db: sqlite SqlUserStore/SqlAccountStore round-trip (create user, deposit, position) in a temp DB; default
  factory returns the in-memory store.
- Full suite stays green with defaults (simulated/simulated/memory).
