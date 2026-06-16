# Decisions I made for you (you said: don't prompt me, just build)

You asked for the best, most complete intelligent finance app — Python + React,
real-time, 15+ ways to invest with real economic/statistical math, expected returns over
1d/1w/1m/1y/5y, live stock + crypto charts, tests + a test guide, a README, 2026 SaaS
design, responsive, light + dark mode — and to make the calls myself. Here they are.

## Product
- **GiffMeMoney** = a quant investment advisor. For each asset it runs **20 named
  quantitative models** (exceeds your "at least 15") and blends them into one composite
  recommendation with a plain-English "why", plus expected return + confidence bands
  across all 5 horizons you asked for (1D/1W/1M/1Y/5Y).
- Six product surfaces: Dashboard, Recommendations, Asset Detail, Strategy Lab,
  Portfolio Optimizer, Screener.

## The math (the heart of it)
CAPM · Fama–French 3-factor · DCF · Gordon DDM · Markowitz mean-variance + efficient
frontier · Sharpe · Sortino · 12-1 Momentum · Mean-reversion (OU z-score) · MACD ·
RSI · Bollinger %B · Monte Carlo (GBM) · GARCH(1,1) MLE · Black–Scholes · VaR/CVaR ·
Kelly criterion · Piotroski F-Score · Altman Z-Score · OLS trend + Holt-Winters.
Implemented from scratch on numpy/scipy (no heavyweight stats libs) so the formulas are
real and the dependency surface stays small and reliable.

## Live data = pluggable, keys later (as you said)
- The app ships with a **deterministic market simulator** that generates multi-year OHLCV
  history + streams live ticks over WebSocket for ~24 assets (equities, crypto, ETFs).
  **No API keys needed** to run the whole thing end to end.
- A `MarketDataProvider` interface means real providers (Finnhub/Polygon/CoinGecko/Binance)
  drop in later behind the same contract — just add the adapter + key, no app rewrite.

## Stack
- **Backend:** FastAPI + Pydantic v2 + numpy/pandas/scipy, native WebSocket, pytest.
- **Frontend:** React 18 + TypeScript (strict) + Vite, TailwindCSS (class dark mode),
  react-query, zustand, recharts + lightweight-charts, lucide-react, vitest + RTL.
- **Design:** 2026 SaaS — emerald "money" primary + indigo accent, rounded-2xl surfaces,
  glassy sticky header, full light/dark, responsive mobile→ultrawide, tabular figures,
  tight-but-breathable gutters (no wasted margins).

## How it was built
Orchestrated with multi-agent workflows (ultracode) against a single frozen contract in
`docs/CONTRACT.md`, in phases (foundation → quant engine + API → frontend → tests + docs),
with a build/typecheck/test verification gate after each phase.

See `README.md` to run it and `docs/TESTING.md` for what each test proves.
