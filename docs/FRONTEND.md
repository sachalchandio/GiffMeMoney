# GiffMeMoney — Frontend Build Contract

> The React app. Builds against the FROZEN backend contracts: `docs/CONTRACT.md` §8 (design system +
> pages), `docs/INVEST.md` (Invest page), `docs/AUTH.md` (login), `docs/STRATEGIES-V2.md` §10 (backtests,
> scenarios, regime, leaderboard). Stack: React 18 + TypeScript (strict) + Vite + TailwindCSS (darkMode
> 'class') + react-router-dom v6 + @tanstack/react-query v5 + zustand + recharts + lightweight-charts +
> lucide-react; tests Vitest + @testing-library/react + jsdom. **No `any` in exported signatures.**

## Config
- API base: `import.meta.env.VITE_API_URL` (default `http://localhost:8000`). WS base derived by swapping
  http→ws + `/ws`. A `frontend/.env` may set `VITE_API_URL` for the live demo.
- `tailwind.config.ts` maps semantic colors to CSS vars from `index.css` (`bg`, `surface`, `surface-2`,
  `border`, `text`, `muted`, `primary`, `accent`, `success`, `danger`, `warning`). Both `:root` (light) and
  `.dark` token blocks per CONTRACT §8. Radius cards `rounded-2xl`, controls `rounded-xl`. `.tnum` utility.
- Dense layout (user requirement): gutters `px-3 sm:px-4 lg:px-6`, card `p-4`, grid gaps `gap-3 lg:gap-4`,
  content max-width ~1440px. Fully responsive (mobile drawer sidebar → desktop fixed). No wasted margins.

## File tree (`frontend/src/`)
```
main.tsx ; App.tsx (router + ProtectedRoute + AppLayout) ; index.css ; vite-env.d.ts
lib/      types.ts (ALL DTOs) · api.ts (typed client) · auth.tsx (AuthProvider) · format.ts · utils.ts · payment.ts
theme/    ThemeProvider.tsx · tokens.ts (chart colors from CSS vars)
store/    marketStore.ts (zustand: live prices map, conn status)
hooks/    useMarketSocket.ts · useAssets · useAnalysis · useRecommendations · useStrategies ·
          useBacktest · usePortfolioOpt · useWallet · usePortfolioState · usePortfolioHistory · useAdvisor
components/ ui/* · layout/* · charts/* · domain/*
pages/    LoginPage · SignupPage · DashboardPage · RecommendationsPage · AssetDetailPage ·
          StrategyLabPage · PortfolioPage · InvestPage · ScreenerPage
test/     setup.ts + *.test.ts(x)
```

## `lib/types.ts` — mirror every backend DTO (camelCase), incl. the V2/invest/auth additions
Asset, Candle, PricePoint, ExpectedReturn (+ bullPct/basePct/bearPct/cvarPct), StrategySignal (+ backtest?),
RiskMetrics, RegimeInfo, AssetAnalysis (+ regime?, strategyCount, disclaimer), Recommendation, StrategyMeta,
StrategyRanking, StrategyLeaderboard, BacktestMetricsDTO, BacktestResultDTO, MonteCarloResult, MarketSummary,
PortfolioRequest/Result; Wallet, SavedCard, CardIn, Transaction, Position, PortfolioState, PortfolioHistory,
AllocationItem/InvestRequest/SellRequest, AdviceRequest/AllocationAdvice/AdviceItem, RiskTolerance;
UserDTO, AuthResponse, Signup/LoginRequest. Plus the literal unions (AssetClass, Horizon, Stance, etc.).

## `lib/api.ts` — typed methods (attach `Authorization: Bearer <token>` when present; throw on !ok)
auth: `signup`, `login`, `me`.
assets: `listAssets(assetClass?)`, `getAsset(s)`, `getCandles(s,interval,limit)`, `getAnalysis(s)`,
`getMonteCarlo(s,horizon,sims)`, `getBacktest(s,strategyId)`.
recommendations: `getRecommendations(limit?,assetClass?)`. market: `getMarketSummary()`, `getHealth()`.
strategies: `listStrategies()`, `getStrategyRankings(id,limit?)`, `getStrategyBacktest(id,s)`, `getLeaderboard(s,limit?)`.
portfolio(analytical): `optimizePortfolio(req)`.
invest: `getWallet()`, `deposit(req)`, `withdraw(req)`, `getCards()`, `deleteCard(id)`, `getTransactions()`,
`getPortfolioState()`, `invest(req)`, `sell(req)`, `getPortfolioHistory(points?)`, `advise(req)`.

## Auth flow (`lib/auth.tsx`)
AuthProvider holds `{user, token}`, persists token in `localStorage('giff_token')`; `login/signup/logout`;
on mount, if token present call `me()` to hydrate (logout on 401). `ProtectedRoute` redirects to `/login`
when unauthenticated. Login/Signup pages are branded (light/dark) with a **"Use demo account"** button that
logs in with `demo@giffmemoney.app` / `demo1234`. Top bar shows the user's name + logout.

## Real-time (`hooks/useMarketSocket.ts` + `store/marketStore.ts`)
Connect to `${WS}`, handle `snapshot` / `tick` (PricePoint[]) / `heartbeat`; write live prices into the
store; auto-reconnect with backoff. Invest & dashboard derive live P&L from store prices × position units.

## Charts (`components/charts/`, props frozen so pages + chart agents agree)
- `PriceChart({ candles, type:'candor'|'area' })` — lightweight-charts.
- `ScenarioFanChart({ expectedReturns })` — bull/base/bear bands across the 5 horizons (recharts area).
- `DistributionChart({ result: MonteCarloResult })` — final-distribution histogram + percentile bands.
- `EfficientFrontierChart({ result: PortfolioResult })` — frontier + CML + tangency.
- `EquityCurveChart({ result: BacktestResultDTO })` — strategy vs buy&hold.
- `PnlChart({ history: PortfolioHistory, live })` — total value over time + per-position lines.
- `AllocationDonut({ positions })`, `ScoreGauge({ score })`, `MiniSpark({ points })`, `FactorRadar`.
All read colors from `theme/tokens.ts` (no hardcoded hex); responsive (ResponsiveContainer).

## Pages (data via the hooks above; skeletons while loading; light/dark; responsive)
- **Login / Signup** — per AUTH.md.
- **Dashboard** — market breadth + sector heatmap (`getMarketSummary`), top opportunities (`getRecommendations`),
  live ticker strip (store), a featured ScenarioFan/MonteCarlo, portfolio snapshot CTA, regime badges.
- **Recommendations** — ranked table/cards of all assets; filter by class; expand → reasons + 1Y + sparkline.
- **Asset Detail** (`/asset/:symbol`) — live PriceChart, ScoreGauge (composite + recommendation + regime),
  HorizonTable (5 horizons w/ bull/base/bear + P(up) + CVaR), RiskMetricGrid, every StrategySignal card
  (score+formula+rationale), MonteCarlo DistributionChart, ScenarioFanChart, disclaimer line.
- **Strategy Lab** — all 73 metas grouped by category w/ sources; click → cross-asset rankings + EquityCurveChart
  + per-asset leaderboard (`getLeaderboard`).
- **Portfolio Optimizer** — pick assets + objective + risk-free → EfficientFrontierChart + weights donut/bars + stats.
- **Invest** — per INVEST.md: wallet header (live total/P&L), Add Funds modal (debit card + Luhn + brand + save
  toggle + $20 chip + "demo/sandbox" tag), Withdraw, AllocationBuilder (+ "Suggest for me" risk control → advisor),
  positions w/ Sell, real-time PnlChart + AllocationDonut, "where to invest now" panel, transactions list.
- **Screener** — sortable/filterable table across the universe (price, change, composite, Sharpe, vol, 1Y); row → asset.

## Tests (Vitest)
`format.test.ts`, `payment.test.ts` (Luhn/brand), `utils.test.ts`, a `ScoreGauge`/`StanceBadge` render test,
an `AllocationBuilder` math test, a `RecommendationsPage` render test with a mocked api client, an auth-context test.

## Verify
`npm install` → `npm run build` (tsc + vite build must pass) → `npm run test` (vitest). Fix until green.
