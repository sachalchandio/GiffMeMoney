# GiffMeMoney — System Contract (Single Source of Truth)

> Every build agent MUST implement against this spec exactly. Names, paths, types,
> routes, and message shapes are **frozen**. If something is ambiguous, prefer the
> shape written here over inventing a new one. No placeholders, no `TODO`, no stubbed
> functions — every file must be complete, importable, and type-correct.

---

## 1. Product

**GiffMeMoney** is an intelligent investment-advisory platform. It tells a user
**where to invest and why**, backed by real quantitative finance. For every asset it
runs **18+ named quant models** (CAPM, Fama–French, DCF, Markowitz, Monte Carlo,
GARCH, Black–Scholes, momentum, mean-reversion, RSI/MACD, VaR/CVaR, Kelly, Piotroski,
Altman Z, etc.), blends them into a composite recommendation, and projects **expected
return across 5 horizons: 1 Day, 1 Week, 1 Month, 1 Year, 5 Years** with confidence
bands. Real-time price streaming for equities + crypto via WebSocket. Live third-party
market APIs are a **pluggable adapter** added later — the app runs fully on a built-in
realistic market **simulator** with **no API keys required**.

Aesthetic: **2026 SaaS** — clean, dense-but-breathable, rounded-2xl surfaces, soft
shadows, subtle gradients, glassy sticky header, **light + dark mode**, fully responsive
(mobile → ultrawide). Numbers use tabular figures. Motion is subtle.

---

## 2. Monorepo layout

```
GiffMeMoney/
├─ README.md
├─ DECISIONS.md
├─ docs/
│  ├─ CONTRACT.md            # this file
│  └─ TESTING.md             # test guide (what each test proves + how to run)
├─ backend/
│  ├─ requirements.txt
│  ├─ pyproject.toml         # pytest config + tooling
│  ├─ run.py                 # `python run.py` -> uvicorn dev server
│  ├─ app/
│  │  ├─ __init__.py
│  │  ├─ main.py             # FastAPI app: CORS, mounts all routers, /ws, startup
│  │  ├─ config.py           # Settings (env-driven; provider keys optional)
│  │  ├─ schemas.py          # ALL Pydantic v2 DTOs (section 4)
│  │  ├─ market/
│  │  │  ├─ __init__.py
│  │  │  ├─ universe.py      # seed asset universe (~24 assets, equities+crypto+etf)
│  │  │  ├─ simulator.py     # deterministic OHLCV history + live tick generator
│  │  │  ├─ provider.py      # MarketDataProvider ABC + SimulatedProvider + registry
│  │  │  └─ feed.py          # ConnectionManager + async tick broadcaster
│  │  ├─ quant/
│  │  │  ├─ __init__.py
│  │  │  ├─ returns.py       # price->returns, annualization, helpers
│  │  │  ├─ metrics.py       # sharpe, sortino, treynor, jensen alpha, beta, vol,
│  │  │  │                   #   max drawdown, calmar, information ratio
│  │  │  ├─ capm.py          # CAPM expected return
│  │  │  ├─ factor.py        # Fama-French 3-factor regression
│  │  │  ├─ valuation.py     # DCF intrinsic value + Gordon DDM
│  │  │  ├─ montecarlo.py    # GBM Monte Carlo paths + percentiles + VaR/CVaR
│  │  │  ├─ volatility.py    # EWMA + GARCH(1,1) MLE volatility forecast
│  │  │  ├─ options.py       # Black-Scholes price + greeks + implied vol
│  │  │  ├─ risk.py          # VaR/CVaR (historical, parametric, monte carlo)
│  │  │  ├─ technical.py     # SMA/EMA, MACD, RSI, Bollinger, momentum, mean-reversion
│  │  │  ├─ forecast.py      # OLS trend + Holt-Winters exponential smoothing
│  │  │  ├─ fundamental.py   # Piotroski F-Score, Altman Z-Score
│  │  │  ├─ kelly.py         # Kelly criterion position sizing
│  │  │  └─ portfolio.py     # Markowitz mean-variance optimizer + efficient frontier
│  │  ├─ strategies/
│  │  │  ├─ __init__.py
│  │  │  ├─ base.py          # Strategy protocol + StrategyResult builder helpers
│  │  │  ├─ registry.py      # STRATEGIES: list[StrategyMeta]; build_signal(...) per id
│  │  │  └─ engine.py        # AnalysisEngine: run all strategies -> AssetAnalysis,
│  │  │                      #   composite score, blended horizons, rationale, ranking
│  │  └─ api/
│  │     ├─ __init__.py
│  │     ├─ assets.py        # /api/assets*, candles, analysis, montecarlo
│  │     ├─ recommendations.py # /api/recommendations
│  │     ├─ strategies.py    # /api/strategies*, rankings
│  │     ├─ portfolio.py     # /api/portfolio/optimize
│  │     └─ market.py        # /api/market/summary, /api/health
│  └─ tests/
│     ├─ __init__.py
│     ├─ conftest.py         # TestClient fixture, seeded universe
│     ├─ test_metrics.py
│     ├─ test_models.py      # capm, black-scholes, montecarlo, var, portfolio, etc.
│     ├─ test_strategies.py  # engine, registry completeness (>=18), composite
│     └─ test_api.py         # every endpoint + websocket smoke
└─ frontend/
   ├─ package.json
   ├─ index.html
   ├─ vite.config.ts
   ├─ tsconfig.json
   ├─ tsconfig.node.json
   ├─ tailwind.config.ts
   ├─ postcss.config.js
   ├─ vitest.config.ts
   ├─ src/
   │  ├─ main.tsx
   │  ├─ App.tsx              # router + AppLayout
   │  ├─ index.css            # tailwind layers + CSS variables (light/dark tokens)
   │  ├─ vite-env.d.ts
   │  ├─ lib/
   │  │  ├─ types.ts          # TS mirror of section 4 DTOs (frozen)
   │  │  ├─ api.ts            # typed fetch client (section 5)
   │  │  ├─ format.ts         # currency/pct/compact number formatters
   │  │  └─ utils.ts          # cn() classnames, color-for-stance, etc.
   │  ├─ theme/
   │  │  ├─ ThemeProvider.tsx # light/dark, persisted, system-aware
   │  │  └─ tokens.ts         # shared chart color tokens read from CSS vars
   │  ├─ hooks/
   │  │  ├─ useMarketSocket.ts# WebSocket live ticks -> store
   │  │  ├─ useAssets.ts      # react-query hooks
   │  │  ├─ useAnalysis.ts
   │  │  ├─ useRecommendations.ts
   │  │  ├─ useStrategies.ts
   │  │  └─ usePortfolio.ts
   │  ├─ store/
   │  │  └─ marketStore.ts    # zustand: live prices map, connection status
   │  ├─ components/
   │  │  ├─ ui/               # Button, Card, Badge, Stat, Tabs, Sparkline, Skeleton,
   │  │  │                    #   Tooltip, Toggle, Select, ScoreGauge, ProgressBar
   │  │  ├─ layout/           # AppLayout, Sidebar, TopBar, ThemeToggle, ConnDot
   │  │  ├─ charts/           # PriceChart(candle/area), ReturnFanChart,
   │  │  │                    #   EfficientFrontierChart, DistributionChart,
   │  │  │                    #   FactorBarChart, RadarChart, MiniSpark
   │  │  └─ domain/           # RecommendationCard, SignalCard, HorizonTable,
   │  │                       #   RiskMetricGrid, StanceBadge, AssetRow, MarketTicker
   │  ├─ pages/
   │  │  ├─ DashboardPage.tsx
   │  │  ├─ RecommendationsPage.tsx
   │  │  ├─ AssetDetailPage.tsx
   │  │  ├─ StrategyLabPage.tsx
   │  │  ├─ PortfolioPage.tsx
   │  │  └─ ScreenerPage.tsx
   │  └─ test/
   │     ├─ setup.ts
   │     ├─ format.test.ts
   │     ├─ utils.test.ts
   │     ├─ ScoreGauge.test.tsx
   │     └─ RecommendationsPage.test.tsx
   └─ public/
      └─ favicon.svg
```

---

## 3. Conventions (apply everywhere)

- **Backend:** Python 3.11, FastAPI, Pydantic **v2**, numpy/pandas/scipy only for math
  (NO statsmodels/arch/sklearn — implement GARCH/Holt-Winters/OLS by hand with
  numpy/scipy). Type hints everywhere. Google-style docstrings that **state the formula**.
- **Determinism:** the simulator seeds RNG from a stable hash of the symbol
  (`int(hashlib.sha256(symbol).hexdigest(),16) % 2**32`) so history is reproducible and
  testable. Live ticks may use a per-process RNG.
- **Frontend:** React 18 + TypeScript strict, Vite, TailwindCSS (darkMode: 'class'),
  react-router-dom v6, @tanstack/react-query v5, zustand, recharts,
  lightweight-charts, lucide-react. No `any` in exported signatures.
- **Money/number formatting** lives only in `lib/format.ts`.
- **Colors** come only from CSS variables / Tailwind tokens — never hardcode hex in
  components (charts read tokens via `theme/tokens.ts`).
- All REST responses are JSON matching section 4 exactly (camelCase keys). Pydantic
  models use `alias`/`populate_by_name` or `model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)` to emit **camelCase**.

---

## 4. Shared data contract (DTOs)

Keys are **camelCase** on the wire. TS types in `frontend/src/lib/types.ts` mirror these
1:1. Pydantic models in `backend/app/schemas.py` serialize to these.

```ts
export type AssetClass = 'equity' | 'crypto' | 'etf';
export type Horizon = '1D' | '1W' | '1M' | '1Y' | '5Y';
export type Stance = 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';
export type StrategyCategory =
  | 'Valuation' | 'Factor' | 'Risk-Adjusted' | 'Technical'
  | 'Statistical' | 'Portfolio' | 'Fundamental' | 'Derivatives';

export interface Asset {
  symbol: string; name: string; assetClass: AssetClass;
  sector: string | null; currency: string;
  price: number; change24hPct: number;
  marketCap: number | null; volume24h: number | null;
}

export interface Candle { t: number; o: number; h: number; l: number; c: number; v: number; } // t = unix seconds

export interface PricePoint { symbol: string; price: number; t: number; changePct: number; } // t = unix ms

export interface ExpectedReturn {
  horizon: Horizon;
  expectedReturnPct: number;   // mean total return over the horizon, %
  low: number; high: number;   // ~5th / 95th percentile, %
  probPositive: number;        // 0..1
  annualizedVol: number;       // %
}

export interface StrategySignal {
  strategyId: string; strategyName: string; category: StrategyCategory;
  score: number;               // -100..100 (positive = bullish)
  stance: Stance;
  confidence: number;          // 0..1
  rationale: string;           // plain-English "why"
  formula: string;             // the formula used (compact, human readable)
  metrics: Record<string, number>;  // model-specific raw numbers
  horizons: ExpectedReturn[];  // may be [] for non-projecting models
}

export interface RiskMetrics {
  beta: number; annualVol: number; sharpe: number; sortino: number;
  var95: number; cvar95: number; maxDrawdown: number; calmar: number;
}

export interface AssetAnalysis {
  asset: Asset;
  compositeScore: number;      // -100..100
  recommendation: Stance;
  confidence: number;          // 0..1
  expectedReturns: ExpectedReturn[];  // blended, one per Horizon (always 5)
  riskMetrics: RiskMetrics;
  signals: StrategySignal[];   // one per strategy (>=18)
  rationaleSummary: string;    // narrative "where/why to invest"
  topReasons: string[];        // 3-5 bullet reasons
  updatedAt: number;           // unix ms
}

export interface Recommendation {
  rank: number; asset: Asset;
  compositeScore: number; recommendation: Stance; confidence: number;
  expectedReturn1YPct: number; headline: string; reasons: string[];
}

export interface StrategyMeta {
  id: string; name: string; category: StrategyCategory;
  summary: string;             // 1-2 sentence description
  formula: string;             // compact formula
  inputs: string[];            // what data it consumes
  references: string[];        // e.g. ["Sharpe (1964)"]
}

export interface StrategyRanking { strategyId: string; entries: { asset: Asset; score: number; stance: Stance; }[]; }

export interface PortfolioRequest {
  symbols: string[];
  riskFreeRate: number;        // annual, decimal (e.g. 0.04)
  objective: 'max_sharpe' | 'min_volatility' | 'target_return';
  targetReturn: number | null; // annual decimal, required iff objective='target_return'
}
export interface PortfolioPoint { volatility: number; expectedReturn: number; sharpe: number; }
export interface PortfolioResult {
  weights: { symbol: string; weight: number }[];
  expectedReturn: number; volatility: number; sharpe: number;  // decimals (annual)
  efficientFrontier: PortfolioPoint[];
  capitalMarketLine: PortfolioPoint[];
  riskFreeRate: number;
}

export interface MonteCarloBand { t: number; p5: number; p25: number; p50: number; p75: number; p95: number; } // t = step index
export interface MonteCarloResult {
  symbol: string; horizon: Horizon; sims: number; steps: number;
  bands: MonteCarloBand[];                 // price percentile bands over time
  finalDistribution: { binStart: number; binEnd: number; count: number }[];
  expectedReturnPct: number; var95Pct: number; cvar95Pct: number; probPositive: number;
}

export interface MarketSummary {
  asOf: number;
  breadth: { advancers: number; decliners: number; unchanged: number };
  topGainers: Recommendation[];   // reuse shape: rank+asset+expectedReturn1YPct
  topLosers: Recommendation[];
  sectors: { sector: string; changePct: number; count: number }[];
  indices: { name: string; level: number; changePct: number }[];
}
```

Stance thresholds (composite or signal `score` in -100..100):
`>=60 STRONG_BUY, >=20 BUY, >-20 HOLD, >-60 SELL, else STRONG_SELL`.

---

## 5. REST API (prefix `/api`, all GET unless noted)

| Method | Path | Query | Returns |
|---|---|---|---|
| GET | `/api/health` | — | `{status:"ok", time:number, universe:number}` |
| GET | `/api/assets` | `assetClass?` | `Asset[]` |
| GET | `/api/assets/{symbol}` | — | `Asset` |
| GET | `/api/assets/{symbol}/candles` | `interval=1d`, `limit=365` | `Candle[]` |
| GET | `/api/assets/{symbol}/analysis` | — | `AssetAnalysis` |
| GET | `/api/assets/{symbol}/montecarlo` | `horizon=1Y`, `sims=2000` | `MonteCarloResult` |
| GET | `/api/recommendations` | `limit=12`, `assetClass?` | `Recommendation[]` |
| GET | `/api/strategies` | — | `StrategyMeta[]` (>=18) |
| GET | `/api/strategies/{id}/rankings` | `limit=20` | `StrategyRanking` |
| POST | `/api/portfolio/optimize` | body `PortfolioRequest` | `PortfolioResult` |
| GET | `/api/market/summary` | — | `MarketSummary` |

Errors: 404 `{detail}` for unknown symbol/strategy; 422 for bad body (FastAPI default).
CORS: allow `http://localhost:5173` and `http://127.0.0.1:5173` (Vite dev) + `*` methods/headers.

---

## 6. WebSocket `/ws`

- On connect server sends `{ "type":"snapshot", "data": PricePoint[] }` for the full universe.
- Client may send `{ "action":"subscribe", "symbols":[...] }` / `{ "action":"unsubscribe", ... }`.
  Default subscription = whole universe.
- Server pushes every ~1000 ms: `{ "type":"tick", "data": PricePoint[] }` (only subscribed symbols).
- Server pushes `{ "type":"heartbeat", "t":number }` every ~15 s.
- Frontend `useMarketSocket` auto-reconnects with backoff and writes ticks into `marketStore`.

---

## 7. Quant model catalog (the strategy registry — implement ALL)

Each produces a `StrategySignal`. `id` is kebab/camel stable. Categories in parens.
Engine must register **at least these 18**:

1. `capm` — **CAPM** (Risk-Adjusted/Factor): `E[R] = Rf + β(E[Rm]−Rf)`. Signal = expected excess vs realized.
2. `fama-french` — **Fama–French 3-Factor** (Factor): regress excess returns on Mkt, SMB, HML; alpha sign drives score.
3. `dcf` — **Discounted Cash Flow** (Valuation): intrinsic = Σ FCFₜ/(1+wacc)ᵗ + TV; score from margin of safety vs price.
4. `ddm` — **Gordon Dividend Discount** (Valuation): `P = D₁/(r−g)`; score from fair-value gap.
5. `markowitz` — **Mean-Variance fit** (Portfolio): asset's contribution to tangency portfolio weight.
6. `sharpe` — **Sharpe Ratio** (Risk-Adjusted): `(R̄−Rf)/σ`; rank-normalized to score.
7. `sortino` — **Sortino Ratio** (Risk-Adjusted): `(R̄−Rf)/σ_downside`.
8. `momentum` — **12-1 Momentum** (Technical): trailing 12m return ex last month.
9. `mean-reversion` — **Mean Reversion / OU z-score** (Statistical): z = (price−μ)/σ over window; bearish when stretched up.
10. `macd` — **MACD crossover** (Technical): EMA12−EMA26 vs signal EMA9.
11. `rsi` — **RSI(14)** (Technical): 100−100/(1+RS); overbought/oversold.
12. `bollinger` — **Bollinger %B** (Technical): position within μ±2σ band.
13. `montecarlo` — **Monte Carlo GBM** (Statistical): simulate paths; score from prob(return>0) at 1Y.
14. `garch` — **GARCH(1,1) volatility regime** (Statistical): forecast vol; score from vol trend (falling vol mildly bullish, risk-adjusted).
15. `black-scholes` — **Black–Scholes risk** (Derivatives): ATM 1Y call value & implied leverage as conviction proxy.
16. `var` — **Value at Risk / CVaR** (Risk-Adjusted): 95% historical VaR; score penalizes tail risk.
17. `kelly` — **Kelly Criterion** (Risk-Adjusted): optimal fraction `f* = μ/σ²`; score from f* sign/magnitude.
18. `piotroski` — **Piotroski F-Score** (Fundamental): 9-point quality score (use simulated fundamentals).
19. `altman-z` — **Altman Z-Score** (Fundamental): bankruptcy distance; score from safe/distress zone.
20. `trend-ols` — **OLS Trend + Holt-Winters** (Statistical): regression slope & forecast drift.

> Fundamentals (earnings, FCF, dividends, book value, ratios) are generated
> deterministically per-symbol in `universe.py`/`simulator.py` (a `Fundamentals` dataclass)
> so valuation/fundamental models have real inputs. Keep them plausible.

Each signal's `horizons` (for models that project) scales drift μ and vol σ to each
horizon h (in trading days: 1D≈1, 1W≈5, 1M≈21, 1Y≈252, 5Y≈1260):
`expectedReturnPct = (exp(μ_daily·h) − 1)·100`, bands from `±1.645·σ_daily·√h`,
`probPositive = Φ(μ_daily·√h / σ_daily)`. The **engine** always outputs a blended 5-horizon
`expectedReturns` on the AssetAnalysis (confidence-weighted mean of projecting models).

Composite score = confidence-weighted average of signal scores, lightly shrunk toward 0
by disagreement (std of signals). Recommendation from stance thresholds. `rationaleSummary`
and `topReasons` are generated from the strongest contributing signals.

---

## 8. Frontend design system

**Tokens** (CSS variables in `index.css`, both `:root` light and `.dark`):
Brand = emerald/“money” primary with an indigo-violet accent; slate neutrals.

```
Light:  --bg #f7f8fa  --surface #ffffff  --surface-2 #f1f3f7  --border #e6e8ee
        --text #0b1220  --text-muted #5b6472  --primary #0f9e6e  --primary-press #0c875e
        --accent #6d5efc  --success #16a34a  --danger #e5484d  --warning #f59e0b
        --ring rgba(15,158,110,.35)  --shadow rgba(16,24,40,.08)
Dark:   --bg #0a0d14  --surface #11151f  --surface-2 #161b27  --border #232a39
        --text #e8edf5  --text-muted #93a0b4  --primary #2bd49b  --primary-press #25b486
        --accent #8b7dff  --success #34d399  --danger #ff6b6f  --warning #fbbf24
        --ring rgba(43,212,155,.30)  --shadow rgba(0,0,0,.5)
Chart colors (tokens.ts reads these): up=success, down=danger, primary, accent,
plus a categorical palette of 8 for multi-series.
```

Tailwind maps tokens to semantic classes: `bg-bg`, `bg-surface`, `bg-surface-2`,
`border-border`, `text-text`, `text-muted`, `text-primary`, `bg-primary`, etc. (extend
`colors` in tailwind config to `var(--…)`). `darkMode: 'class'`. Radius scale: cards
`rounded-2xl`, controls `rounded-xl`. Font: Inter (CDN in index.html) + system fallback;
`font-variant-numeric: tabular-nums` on numbers via a `.tnum` utility.

**Layout:** persistent left **Sidebar** (collapsible on mobile to a top drawer), sticky
glassy **TopBar** with search, live connection dot, theme toggle. Content max-width
container with comfortable gutters (NOT excessive — `px-4 md:px-6`, content up to ~1400px).
Everything responsive with CSS grid (`grid-cols-1 lg:grid-cols-3`, etc.). Loading uses
skeletons. Stance colors: BUY/STRONG_BUY → success, HOLD → muted/warning, SELL/STRONG_SELL → danger.

**Pages:**
- **Dashboard** — hero "where to invest now" composite leaders, live MarketTicker strip,
  market breadth + sector heatmap, top movers, a featured Monte Carlo / return-fan chart,
  portfolio snapshot CTA.
- **Recommendations** — ranked cards/table of all assets by composite score, filter by
  asset class, each row expands to reasons + 1Y expected return + sparkline.
- **Asset Detail** (`/asset/:symbol`) — live PriceChart (candle/area toggle), composite
  ScoreGauge, HorizonTable (5 horizons w/ bands), RiskMetricGrid, every StrategySignal as
  a card with its formula + rationale, Monte Carlo fan chart, factor/radar viz.
- **Strategy Lab** — gallery of all 18+ models (formula + summary + references), click one
  to see its cross-asset rankings (bar chart + list).
- **Portfolio Optimizer** — pick assets + objective + risk-free rate → efficient frontier
  chart with tangency portfolio, weight allocation donut/bars, expected return/vol/Sharpe.
- **Screener** — sortable/filterable table across the universe (price, change, composite,
  Sharpe, vol, expected 1Y), click → asset detail.

---

## 9. Run & test (targets the docs must describe)

- Backend dev: `cd backend && python run.py` (serves `http://localhost:8000`, docs at `/docs`, ws at `/ws`).
- Backend tests: `cd backend && pytest -q`.
- Frontend dev: `cd frontend && npm install && npm run dev` (Vite at `http://localhost:5173`).
  Frontend reads API base from `import.meta.env.VITE_API_URL` (default `http://localhost:8000`).
- Frontend build: `npm run build`. Frontend tests: `npm run test` (vitest).
