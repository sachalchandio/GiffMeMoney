# Testing Guide

This is the test guide for **GiffMeMoney**, the quant investment-advisory platform.
It covers how to run both test suites, what each test file proves, and exactly what a
green run guarantees about the product.

GiffMeMoney is an **educational simulation — NOT financial advice**. No real money moves;
everything runs on a deterministic in-process simulator (no API keys). The tests exercise
the **production code paths** end to end (real `SimulatedProvider`, real FastAPI app via
`TestClient`, real React components) rather than mocks, except where a mock is needed to
isolate a unit (e.g. the frontend page/auth tests mock only the typed API client).

---

## Headline totals

| Suite | Runner | Command | Tests | Status |
| --- | --- | --- | --- | --- |
| Backend | pytest 8 | `pytest -q` | **255** | 255 passed |
| Frontend | vitest 2 | `npm run test` | **71** | 71 passed |

- **Backend: 255 tests, all green.** A cold full run takes roughly **4–5 minutes**; almost
  all of that is the single full-universe R1–R8 re-audit in `test_projection.py` (one real
  24-asset `analyze()` pass, ~30–40 s) plus the two cross-suite full-universe sweeps in
  `test_strategies*.py`. Most files are sub-second.
- **Frontend: 71 tests, all passing.** All 71 tests across the 8 files pass.

Backend test counts per file (derived from `pytest --collect-only`):

| File | Tests |
| --- | --- |
| `tests/test_models.py` | 48 |
| `tests/test_invest.py` | 37 |
| `tests/test_metrics.py` | 28 |
| `tests/test_auth.py` | 26 |
| `tests/test_indicators.py` | 25 |
| `tests/test_api.py` | 23 |
| `tests/test_strategies.py` | 23 |
| `tests/test_projection.py` | 18 |
| `tests/test_strategies_v2.py` | 16 |
| `tests/test_backtest.py` | 11 |
| **Total** | **255** |

Frontend test counts per file (vitest):

| File | Tests |
| --- | --- |
| `src/test/format.test.ts` | 23 |
| `src/test/payment.test.ts` | 16 |
| `src/test/utils.test.ts` | 15 |
| `src/test/AllocationBuilder.test.ts` | 6 |
| `src/test/auth.test.tsx` | 6 |
| `src/test/ScoreGauge.test.tsx` | 7 |
| `src/test/InvestPage.test.tsx` | 4 |
| `src/test/RecommendationsPage.test.tsx` | 3 |
| **Total** | **71** |

> The exact per-file frontend split above sums to 71. The backend split is authoritative
> (collected directly). The two backend files **not** named in the original test brief —
> `test_indicators.py` (25) and `test_backtest.py` (11) — are part of the suite and are
> documented below.

---

## Running the BACKEND suite

The backend is FastAPI + numpy/scipy/pandas (no statsmodels/sklearn — GARCH, OLS,
Holt-Winters and the Markowitz optimizer are all hand-rolled). Config lives in
`backend/pyproject.toml` under `[tool.pytest.ini_options]`: `testpaths = ["tests"]`,
`pythonpath = ["."]`, `asyncio_mode = "auto"`, and `addopts = "-q"` (so `-q` is already the
default). Deprecation warnings and a cosmetic pydantic alias-generator warning are filtered.

From the repository root (Windows PowerShell; use `127.0.0.1` not `localhost` if you also
run the server, to avoid IPv6):

```powershell
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

Expected final line:

```
255 passed, 79 warnings in ~277s
```

The ~79 warnings are benign: scipy SLSQP "values outside bounds — clipping" notes from the
optimizer, and JWT `InsecureKeyLengthWarning` from the short dev `jwt_secret` used in tests.
Neither indicates a failure.

Useful variants:

```powershell
pytest -q tests/test_models.py            # one file
pytest -q -k "garch or black_scholes"     # by keyword
pytest -q tests/test_projection.py        # the R1-R8 re-audit (slowest file)
pytest --collect-only -q                  # list/count tests without running
```

> **Tip — the slow file:** `test_projection.py` builds a module-scoped 24-asset analysis
> once and shares it across the R1–R8 audit tests, so the cost is paid a single time even
> though eight tests read from it.

---

## Running the FRONTEND suite

The frontend is React 18 + TypeScript + Vite + Tailwind, tested with **vitest** under
**jsdom** and Testing Library. Config is in `frontend/vitest.config.ts`: `environment:
'jsdom'`, `globals: true`, `setupFiles: ['./src/test/setup.ts']`, `include:
['src/**/*.{test,spec}.{ts,tsx}']`, with `clearMocks: true` and `restoreMocks: true`. The
`@` alias resolves to `src/`.

`src/test/setup.ts` wires jest-dom matchers, auto-cleans the DOM after each test, and
polyfills `matchMedia`, `ResizeObserver`, and `WebSocket` (all absent in jsdom) so the
`ThemeProvider`, recharts' `ResponsiveContainer`, and the live-market socket hook can mount
without a real browser or server.

From the repository root:

```powershell
cd frontend
npm install
npm run test          # vitest run (one-shot, CI mode)
```

Other scripts (from `package.json`):

```powershell
npm run test:watch    # vitest (watch mode)
npm run typecheck     # tsc --noEmit
npm run build         # tsc -b && vite build
```

Expected (current state):

```
Test Files  8 passed (8)
     Tests  71 passed (71)
```

---

## Backend — what each file proves

### `tests/test_metrics.py` (28) — risk/return metrics vs closed-form answers
Pins every metric in `app.quant.metrics` to a value derivable by hand from a synthetic
series, so a *formula* regression (not just "it runs") is caught. Uses population std
(`ddof=0`) to match the implementation. Proves:
- **beta** = 1 when asset == market, = 2 for a 2× asset, and falls back to the neutral 1.0
  on a zero-variance or too-short market;
- **annual volatility** = population std × √252 (the square-root-of-time rule); 0 for a
  constant or <2-point series;
- **Sharpe / Sortino** equal their known closed forms, and collapse to 0 (not ∞) on
  zero-excess-variance / no-downside inputs;
- **downside deviation** counts only sub-MAR returns over the full sample size;
- **Treynor**, **Jensen's alpha** (0 when the asset tracks CAPM, = the bonus when there's a
  constant per-day outperformance), and **information ratio** (0 vs self) match known values;
- **max drawdown** = −0.5 on a 120→60 path, 0 on a monotone climb;
- **Calmar** = annualized return ÷ |max drawdown|, 0 when there is no drawdown;
- a parametrized **defensiveness** sweep: empty / NaN / ±inf / single-point inputs always
  return finite floats and never raise.

### `tests/test_models.py` (48) — the quant model layer (contract §7)
The headline-model file. Against known values and structural invariants:
- **Black-Scholes**: an ATM 1-year call (S=K=100, r=5%, σ=20%) prices to **≈10.4506**;
  **put-call parity** `C − P = S − K·e^(−rT)`; zero-vol price is discounted intrinsic;
  call delta ∈ (0,1) and > 0.5 ATM, put delta ∈ (−1,0), gamma/vega > 0; **implied vol
  round-trips** back to the input σ.
- **CAPM**: the Security-Market-Line identities — β=1 ⇒ E[R]=Rf+premium, β=0 ⇒ Rf, and
  linear scaling in β.
- **Fama-French 3-factor OLS**: recovers known α/β loadings (and R²=1) from a noiseless
  construction; below the minimum observation count returns the zeroed default.
- **Valuation**: DCF flat-perpetuity capitalizes to **FCF/wacc**, growth strictly raises
  value, non-positive FCF ⇒ 0; **Gordon DDM** = `D0(1+g)/(r−g)`, guarded to 0 (no ∞) when
  r ≤ g, and 0 for a non-payer.
- **VaR/CVaR ordering**: across historical, parametric and Monte-Carlo VaR, all are
  finite & non-negative, **CVaR ≥ VaR**, a 99% VaR ≥ the 95% VaR, and empty input ⇒ 0.
- **Monte-Carlo GBM**: percentile **bands are monotone** (p5 ≤ p25 ≤ p50 ≤ p75 ≤ p95) at
  every step; the first band equals the starting price; the summary DTO has the contract
  shape (probPositive ∈ [0,1], CVaR ≥ VaR); `gbm_paths` returns a (sims, steps+1) array of
  finite positive prices starting at S0.
- **Markowitz optimizer**: for *every* objective (`max_sharpe`, `min_volatility`,
  `target_return`) **weights are long-only and sum to 1**; the min-variance portfolio is
  no more volatile than equal-weight; a feasible target return is achieved; the efficient
  frontier is sorted by volatility; the Capital Market Line has a constant Sharpe slope; a
  one-asset universe gets full weight.
- **GARCH(1,1) MLE**: a fitted model is **stationary** (ω>0, α,β≥0, α+β<1) on a simulated
  GARCH series and via fallback on a too-short series; the horizon forecast and EWMA vol are
  finite & non-negative.
- **Kelly**: f\* = μ/σ² inside a [−1, 3] clamp; clamped at the leverage ceiling / −1; 0 on
  zero variance.
- **Fundamentals** (real universe seeds): **Piotroski** ∈ [0,9] (MSFT ≥ 6, BTC valid but
  low), **Altman-Z** > 1.81 (safe zone) for AAPL.
- **Technical/forecast**: RSI = 100 for a strictly rising series (50 neutral default);
  Bollinger `%B` ordering; **OLS trend** recovers a constant log-drift (slope == drift,
  R²=1); Holt-Winters forecasts above the last price for an uptrend.

### `tests/test_indicators.py` (25) — V2 OHLCV chart indicators
Pins `app.quant.indicators` — the volatility/trend/oscillator/volume primitives the V2
technical signals consume: `true_range`, `atr`, `adx`, `donchian`, `supertrend`,
`ichimoku`, `williams_r`, `stochastic`, `cci`, `keltner`, `obv`, `obv_slope`. Each is
checked on a hand-computable known series (e.g. TR₀ = high−low, later TR = max of the three
Wilder ranges) plus the defensive contract: short / empty / flat inputs collapse to safe,
finite, neutral values and never raise.

### `tests/test_backtest.py` (11) — the vectorized backtesting engine
Pins the realized-performance contract of `app.quant.backtest.backtest_positions`:
- a **constant-long** position reproduces **buy & hold** exactly (the core verification
  invariant);
- a known position series yields a known CAGR / max-drawdown / total-return;
- win-rate and profit-factor are in range (all-wins ⇒ a large but finite profit factor);
- a **flat** position earns ≈0 with no trades;
- every metric is finite; a `supported=False` strategy mirrors buy & hold with zeroed
  strategy metrics.

### `tests/test_strategies.py` (23) — strategy registry + analysis engine
Runs against the real `SimulatedProvider` (no mocks). Proves the contract-level pipeline
guarantees:
- the registry catalogs **≥ 70** strategies, **ids are unique**, **every id has exactly one
  signal builder** (and vice versa), and all metadata is populated with a **valid category**
  (one of the 8: Valuation, Factor, Risk-Adjusted, Technical, Statistical, Portfolio,
  Fundamental, Derivatives) and ≥ 1 reference;
- the **20 core contract models** (capm, fama-french, dcf, ddm, markowitz, sharpe, sortino,
  momentum, mean-reversion, macd, rsi, bollinger, montecarlo, garch, black-scholes, var,
  kelly, piotroski, altman-z, trend-ols) are all registered;
- `build_signals` returns exactly one well-formed signal per strategy in catalog order
  (score ∈ [−100,100], confidence ∈ [0,1], valid stance/category, and either 0 or all 5
  horizons attached);
- `engine.analyze(sym)` returns a complete `AssetAnalysis` for **every** universe symbol:
  exactly **5 expectedReturns** (one per horizon, in order), **≥ 70 signals**, composite ∈
  [−100,100], a valid stance, finite risk metrics, a rationale and **3–5 top reasons**, and
  is **cached case-insensitively**; unknown symbols raise `KeyError` (so the API can 404);
- `engine.recommendations` is **ranked descending** with ranks 1..n, honours `limit` and the
  asset-class filter; `engine.strategy_ranking` ranks one model descending (KeyError on an
  unknown id); `engine.market_summary` reconciles breadth (advancers+decliners+unchanged ==
  universe size) and populates sectors/indices/top-movers.

### `tests/test_strategies_v2.py` (16) — the V2 strategy expansion
Additive guarantees on top of the above, kept fast by using small symbol subsets:
- the registry now has **≥ 70** strategies and builders; every id carries a populated meta
  with sources; a representative spread of new V2 ids is present (magic-formula,
  graham-defensive, qmj-quality-minus-junk, gross-profitability, cross-sectional-momentum,
  tsmom, 52w-high, dual-momentum, golden-cross, supertrend, ichimoku, connors-rsi2,
  bollinger-squeeze, williams-r, cci-reversion, low-vol-anomaly, betting-against-beta,
  seasonality, dogs-of-dow, shareholder-yield);
- **timing strategies expose vectorized position functions** for backtesting (≥ 10
  registered; golden-cross / supertrend / donchian-turtle / tsmom are callable);
- `analyze` still returns exactly 5 horizons and now ≥ 70 signals (with `strategy_count`
  matching), and never raises across the whole universe;
- **cross-sectional strategies score assets differently** — magic-formula,
  cross-sectional-momentum, low-vol-anomaly, betting-against-beta, 52w-high must read
  `ctx.universe` and produce ≥ 2 distinct scores across the subset;
- **seasonality is deterministic given an injected month** (independent of the system clock):
  November (Halloween window) > 0, July (Sell-in-May) < 0, repeatable for the same month,
  and neutral (0) for crypto; injecting `now` bypasses the analysis cache so test months
  never leak.

### `tests/test_projection.py` (18) — projection engine + the R1–R8 re-audit gate
Two layers. **(1) Fast unit tests** on `app.quant.projection.project` / `detect_regime`
over small seeded synthetic series: `project` returns exactly the 5 horizons in order;
**bull ≥ base ≥ bear** at every horizon (base == the displayed expected return); the 5/95
**confidence band widens monotonically** with horizon; `prob_positive` ∈ [0,1]; CVaR ≥ 0
and the tail outcome (−cvar) ≤ base; no NaN/inf; a single absurd bullish signal **cannot**
push the 5Y base above the asset-class cap (the James-Stein shrinkage + cap); degenerate
(empty/constant/tiny) inputs collapse to safe finite values and never raise; `detect_regime`
returns valid labels (bull/bear/neutral, low/normal/high vol) with finite trend/score in
[−1,1], and a calm uptrend classifies as bull.

**(2) The full-universe REGRESSION AUDIT** (`test_reaudit_*`) — one real 24-asset
`analyze()` pass (14 equities, 6 crypto, 4 ETFs), module-scoped so the universe is swept
exactly once (~30–40 s). It asserts the mandatory financial-rigor fixes from
`docs/STRATEGIES-V2.md` §0 actually landed and the live-audit pathologies are gone:

| Rigor | What the audit asserts |
| --- | --- |
| **R1** | No asset projects an implausible **5Y *expected* (base) return**: equities/ETF ≤ ~250%, crypto ≤ ~400%. (Kills the old +380% / +823% / +457% medians.) |
| **R2** | **Believable bands**: every horizon `high` ≤ its credible cap (1D 15% · 1W 35% · 1M 55% · 1Y 60% · 5Y 400%) and `low` ≥ −95%, all finite. (Kills the +26,000% upper bands.) |
| **R3** | **One engine**: for one asset per class (AAPL/BTC/SPY) the analysis 1Y expected return and the Monte-Carlo 1Y expected return agree within **~1.5pp** (run at 50k sims so the residual is estimation noise, not a drift/vol mismatch). |
| **R4** | **Confidence has spread**: confidence spans a > 0.3 range across the 24 assets (≥ 5 distinct rounded values) — no more flat ~0.3 on everything. |
| **R5** | **Actionable stance mix**: ≥ 5 BUY-or-better **and** ≥ 3 SELL-or-worse — not 24×HOLD. |
| **R6** | **Honest downside**: every analysis surfaces finite 1Y CVaR, max drawdown and prob-of-loss, and a **bear scenario strictly below the base** (and not itself a gain). |
| **R8** | **No NaN/inf anywhere** in any analysis (scores, confidence, risk metrics, every horizon field, every signal metric and signal horizon), and the **disclaimer is present**. |

> R7 is not a separate assertion in this file — the audit gate is named "R1–R8" but is
> implemented as the R1–R6 + R8 checks above; the structural §4/§9 guarantees R7 would cover
> (ordered fan, widening bands, prob/CVaR sanity) are pinned by the fast unit tests in the
> same file.

### `tests/test_invest.py` (37) — the simulated brokerage / Invest extension
Covers the in-memory paper-trading sandbox (`docs/INVEST.md`) end to end. Service-level
tests use a **fresh, isolated `AccountStore`** (never the shared singleton) and the API
tests give each test a **fresh `X-Account-Id`**, so no state leaks. Proves:
- **Payments** — **Luhn** validity (accepts good Visa/MC/Amex with or without spaces;
  rejects bad-checksum/short/empty); **brand detection** from IIN (visa/mastercard incl. the
  2-series range/amex/discover/unknown); **masking** reveals only the last four (`•••• 1111`);
  **tokenize drops the raw PAN/CVC** — a saved card stores masked data only;
- **SimulatedPaymentProvider** rejects a Luhn-invalid card, a non-positive amount, and an
  over-the-sandbox-cap charge, and otherwise returns a completed deposit txn;
- **Wallet** — deposit credits cash; saving a card stores a **masked** `SavedCard` with no
  raw PAN anywhere in persisted state; withdraw rejects over-balance and otherwise debits
  cash; **the wallet always reconciles `cash + invested == total`**;
- **Portfolio** — invest **splits cash across symbols** opening one position each, **rejects
  over-spend all-or-nothing** (cash untouched, no positions), `KeyError` on an unknown
  symbol; re-investing **blends average cost basis** (avg_price == cost_basis/units);
  **selling realizes P&L**, returns cash, closes the position, and records a sell txn;
  selling a non-held symbol raises `KeyError`;
- **History** — the backfilled curve returns a **total series of the requested length** plus
  one per-position series of matching length & shape (t/value/pnl/pnlPct, all finite,
  reconciling total = cash + invested);
- **Advisor** — weights sum to ~1, per-leg dollar amounts sum to ~= the request, exactly **5
  horizons**, echoes the request, finite blended stats (uses the `conservative` profile → ≤ 4
  picks to stay cheap);
- **API smoke** — every new `/api` route returns 200 with **camelCase** keys, error codes
  hold (invalid card 400, over-balance 400, over-spend 400, unknown symbol 404, unknown
  position 404, non-positive advisor amount 400), the transactions ledger is newest-first,
  and the invest router is **mounted under `/api`**.

### `tests/test_auth.py` (26) — email/password + JWT auth
Covers the whole sandbox auth surface (`docs/AUTH.md`) additively (anonymous access still
hits the shared `demo` account). API tests use unique random emails so they never collide.
Proves:
- **Security primitives** — `hash_password` round-trips through `verify_password`; the wrong
  password fails; the **stored hash is never the plaintext** and is the self-describing
  **`pbkdf2_sha256$iterations$salt$hash`** format (4 `$`-separated parts); hashing is
  **salted/unique per call** yet both verify; `decode_token` returns `None` (never raises)
  on tampered / garbage / empty / **expired** / **wrong-secret** tokens; a fresh token
  round-trips its `sub`/`email`/`exp` claims;
- **Signup** — 201 with a token + public camelCase user DTO (`{id, email, name, createdAt}`,
  **never a password hash**); **lowercases the email**; duplicate email ⇒ 400; password
  below `MIN_PASSWORD_LENGTH` ⇒ 400;
- **Login** — good credentials ⇒ 200 with token + user; **case-insensitive** on email; wrong
  password ⇒ 401; an unknown email and a wrong password return the **same non-enumerating
  401 message**;
- **/me** — a valid Bearer token returns the user DTO; missing / malformed / tampered /
  expired tokens ⇒ 401;
- **Demo user** — the documented **`demo@giffmemoney.app` / `demo1234`** account is seeded
  and loginable, and its token works against `/me`;
- **Per-user isolation** — two tokens see two private `user:<id>` wallets (a deposit on A is
  invisible to B; distinct `accountId` namespaces), while an **anonymous** (no-token) caller
  still resolves to the shared **`demo`** account, preserving pre-auth behavior.

### `tests/test_api.py` (23) — end-to-end FastAPI routes + WebSocket
Drives the real app through `TestClient` (running the ASGI app, lifespan, and WS handshake
in-process; the client is module-scoped so the background price-tick loop stays up). Proves
every contract route (§5) and the WS protocol (§6):
- `GET /api/health` ⇒ `status=ok` + numeric time + universe size;
- `GET /api/assets` (+ `assetClass` filter) ⇒ non-empty camelCase `Asset[]`;
  `GET /api/assets/{symbol}` ⇒ the snapshot (404 unknown); `/candles` ⇒ ≤ limit OHLCV with
  valid OHLC ordering; `/analysis` ⇒ a full `AssetAnalysis` (composite ∈ [−100,100], valid
  stance, **5 horizons**, **≥ 70 signals**, risk-metrics block, 3–5 top reasons); `/montecarlo`
  ⇒ monotone bands (steps+1), probPositive ∈ [0,1], finite stats — each 404s on an unknown
  symbol;
- `GET /api/recommendations` ⇒ rank-ordered, score-descending, with the class filter;
- `GET /api/strategies` ⇒ **≥ 70** catalog entries; `GET /api/strategies/{id}/rankings` ⇒
  one model ranked descending (404 unknown id);
- `POST /api/portfolio/optimize` ⇒ weights summing to ~1 (long-only, finite stats, frontier
  + CML present) for `max_sharpe` and `min_volatility`; unknown symbol ⇒ 404, empty list ⇒
  422;
- `GET /api/market/summary` ⇒ the dashboard overview with reconciled breadth and populated
  sections;
- **WebSocket `/ws`** ⇒ a `snapshot` (full universe, camelCase `PricePoint`) on connect
  followed by a `tick` (tolerating interleaved heartbeats); subscribing to a single symbol
  narrows subsequent tick payloads to that symbol.

---

## Frontend — what each file proves

### `src/test/format.test.ts` (23) — display formatters (`lib/format.ts`)
Pure string producers asserted to exact en-US output: `formatCurrency` (incl.
**NaN/Infinity/null/undefined → `$0.00`** coercion and digit overrides);
`formatCompactCurrency` (plain under 1,000, then `$1.20M`/`$3.40B`); `formatNumber` /
`formatCompact` (thousands separators, `1.2M`/`850K`); the **percent-vs-fraction**
distinction (`formatPct` treats input as already-percent, `formatFractionPct` scales 0..1,
`formatProbability` rounds to whole-number %); signed deltas (`formatSigned` with the unicode
minus); `formatPrice` (more decimals sub-$1); `formatRatio`; `formatDate` (timezone-tolerant
parts) and the `formatRelativeTime` ladder (just now / Nm ago / Nh ago / Nd ago / future →
just now); and `horizonLabel` / `stanceLabel` humanization.

### `src/test/payment.test.ts` (16) — Add-Funds card helpers (`lib/payment.ts`)
The browser-side mirror of the backend payment checks: `digitsOnly`; **`luhn`** (accepts the
demo card + Visa/MC/Amex/Discover test PANs, rejects bad checksums and implausible
lengths < 12 / > 19); **`brandFor`** IIN detection (incl. the new MC 2-series range; unknown
fallback); `expectedLength` / `cvcLength` (15/4 for Amex, 16/3 otherwise); `formatCardNumber`
(blocks of four capped at 16; Amex 4-6-5); `maskCardNumber` (`•••• 4242`); `brandLabel`;
**`expiryValid`** (accepts current month / future, rejects past and out-of-range
month/year); **`cvcValid`** (per-brand digit length).

### `src/test/utils.test.ts` (15) — UI utilities (`lib/utils.ts`)
The Tailwind-aware `cn` merge (de-dupes conflicting classes); the **stance → tone** grouping
(positive/neutral/negative); **color helpers return semantic tokens, never hardcoded hex**
(`text-success`/`text-warning`/`text-danger`, soft badge classes, and `var(--success)` etc.);
`changeTextColor` keyed off sign; **`stanceFromScore`** mapping the −100..100 score onto the
backend thresholds (≥60 STRONG_BUY, ≥20 BUY, >−20 HOLD, ≤−20 SELL, ≤−60 STRONG_SELL);
`assetClassLabel` (Stocks/Crypto/ETFs); the numeric helpers `clamp` / `round` / `sum` (sum
ignores non-finite values); `initials`; and a deterministic, palette-bounded `colorIndex`.

### `src/test/AllocationBuilder.test.ts` (6) — allocation math (`computeAllocation`)
Pins the pure arithmetic behind the invest allocation builder: funded rows sum to
`allocated` with the correct `remaining` and `fraction`; per-symbol **weights sum to ~1**;
the fill fraction **clamps to 1** and over-budget allocations are flagged (negative
remaining); non-finite / negative amounts are treated as zero; a zero/invalid cash budget
never divides by zero (and any spend over a zero budget is over-budget); a full allocation
leaves 0 remaining and is not over budget.

### `src/test/auth.test.tsx` (6) — `AuthProvider` contract
Drives the React auth context with a **mocked api client**. Proves: starts
unauthenticated with no persisted token; **hydrates the user from a persisted token on
mount** via `me()`; **drops the stale session on a 401** from `me()` (clears state +
`giff_token`); **login applies the session** / persists the token / mirrors the bearer
header; **`loginDemo` authenticates** with the seeded demo credentials; **`logout` clears
state + storage**.

### `src/test/ScoreGauge.test.tsx` (7) — `ScoreGauge` + `StanceBadge`
Renders inside a `ThemeProvider` (it reads chart colors from the theme). Proves the gauge
exposes **accessible meter semantics** (`role="meter"`, aria-valuemin/max/now, an aria-label
naming the stance); renders the **signed numeric value + derived stance label** (`+70` /
"Strong Buy"); **clamps** out-of-range scores into −100..100; honours an **explicit stance
override** over the derived one; renders a caption and can hide the center labels while
keeping the meter. `StanceBadge` renders the humanized label and applies a danger tone for
sell stances.

### `src/test/InvestPage.test.tsx` (4) — the flagship Invest page
Drives `InvestPage` with a fully mocked api client (wallet, portfolio state, history,
transactions, assets, candles) so no network is touched. Proves the **wallet header** renders
the live total value ($1,000.00 = cash 450 + invested 550) with the **sandbox disclaimer**;
the core sections render ("Portfolio value", "Build your allocation", "Where to invest now",
"Activity"); the **open position** (AAPL) and **Add Funds / Withdraw** actions render; and the
**transaction ledger** lists entries with their typed labels (e.g. the "Buy" row for AAPL).

### `src/test/RecommendationsPage.test.tsx` (3) — the Recommendations page
Drives `RecommendationsPage` through a mocked api client. Proves the **ranked picks render**
(with the tone-summary chips, e.g. "2 buy"); the **rank-1 pick is expanded by default** with
its headline and reasons; and changing the **asset-class filter re-queries** the api
(`getRecommendations(50, 'crypto')`).

---

## What "green" guarantees

A fully green run is a strong correctness gate across the whole product:

- **The math is right, not just runnable.** Every quant primitive is pinned to a closed-form
  or hand-computed value (Black-Scholes ≈ 10.4506, put-call parity, CAPM SML identities,
  Fama-French loading recovery, DCF/DDM identities, VaR ≤ CVaR ordering, Markowitz weights
  summing to 1, GARCH stationarity, Kelly clamp, OLS drift recovery), and every primitive is
  defensive — degenerate input returns finite values and never raises.
- **The analysis pipeline is complete and bounded.** For *every* asset in the universe,
  `analyze()` yields exactly 5 ordered horizons, ≥ 70 catalog signals (each with a clamped
  score, valid stance/category and either 0 or all 5 horizons), a composite ∈ [−100,100], a
  valid stance, finite risk metrics, and 3–5 reasons.
- **The projections are financially credible.** The R1–R8 audit guarantees no implausible
  5Y "expected" returns, believable per-horizon bands, analysis/Monte-Carlo consistency from
  one engine, real confidence spread, an actionable BUY/SELL mix (not all-HOLD), honest
  downside (CVaR / drawdown / a sub-base bear case), zero NaN/inf, and a present disclaimer.
- **The money paths are safe and reconciled.** Luhn/brand/mask validation holds, raw PANs and
  CVCs never reach state, the wallet always reconciles `cash + invested == total`, over-spend
  and over-withdraw are rejected all-or-nothing, and cost basis / realized P&L are exact.
- **Auth is sound and isolating.** Passwords are salted PBKDF2 (never stored in plaintext),
  JWTs reject tampered/expired/wrong-secret tokens, login is non-enumerating, and each user
  gets a private `user:<id>` wallet while anonymous callers fall back to the shared demo
  account.
- **Every API route honours its contract.** All §5 REST routes return the right
  camelCase shape and status codes (incl. the 404/422/400 error cases), and the §6 WebSocket
  sends a snapshot then ticks with working symbol subscriptions.
- **The UI logic and key pages render correctly.** Formatters, payment helpers, stance/color
  mapping, and allocation math produce exact output; the Invest and Recommendations pages and
  the ScoreGauge render their data with accessible semantics.

**The suite is fully green:** 71/71 frontend and 255/255 backend. Every path above is
covered by an automatic green check — including the full `auth.test.tsx` contract
(login/logout applies/clears the session), which is also independently validated by the
backend `test_auth.py` suite (login, `/me`, JWT, per-user isolation).
