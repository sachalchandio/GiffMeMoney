# GiffMeMoney — Strategy Expansion & Projection Overhaul (Contract Addendum V2)

> Addendum to `docs/CONTRACT.md`. Implements the research catalog in
> `docs/research/strategy-catalog.json` (53 new strategies + 8 projection methods +
> 14 backtest metrics). Same conventions (camelCase wire, Pydantic v2, numpy/pandas/scipy
> only, no placeholders). **The existing 20 strategies and all 122 passing tests MUST keep
> working** — every change here is additive or backward-compatible.

## Goals
1. Grow the registry from **20 → ~73 strategies** by adding the 53 cited, proven strategies.
2. Add a **backtesting engine** so rules-based strategies report *realized* historical
   performance (CAGR, Sharpe, Sortino, Calmar, maxDD, Ulcer, win-rate, …) — "proven", not guessed.
3. **Overhaul the projection engine**: ensemble drift + regime-aware + GARCH vol + bull/base/bear
   scenarios + fat-tailed/bootstrapped confidence bands + CVaR + shrinkage, across all 5 horizons.

## 0. FINANCIAL RIGOR — MANDATORY (from a live output audit; this is a finance app)

A live audit of the current engine found the projection/scoring layer is over-optimistic,
inconsistent, and uninformative. The overhaul MUST fix the ROOT CAUSES below (not just clamp).
A re-audit must pass before this work is considered done.

**R1 — Credible drift (kill the +800% projections).** Ensemble the projecting strategies' daily
drifts, then **shrink hard toward a sane prior**: prior = CAPM `rf + β·ERP` with ERP≈4.5%. Use
James–Stein style shrinkage so noisy single-asset drift is pulled to the prior. **Cap annualized
expected drift** to a realistic band: equities/ETFs `[-25%, +35%]`, crypto `[-40%, +60%]`. No asset
may project a 5Y *expected* (median) total return above ~ (cap compounded) — e.g. BTC 5Y expected
should land in a believable ~50–250% range, NOT +823%.

**R2 — Believable bands.** Confidence bands come from the SAME distribution used for Monte Carlo
(fat-tailed/Student-t or bootstrap), but the **displayed** 5th/95th percentiles must be capped to
credible bounds: low floored at −95% (long-only can't lose >100%), high capped so it never prints
absurd numbers (cap 95th-pct total return per horizon, e.g. ≤ +60% (1Y) / ≤ +400% (5Y) even for
crypto). Also expose **CAGR** for multi-year horizons alongside total return (a +150% 5Y total = ~20%/yr —
show both so the user isn't misled by the big total number).

**R3 — One projection engine (consistency).** `montecarlo()` and the blended `expectedReturns` MUST
derive from the SAME drift+vol+distribution in `quant/projection.py`. After the fix, AAPL 1Y expected
return from `/analysis` and from `/montecarlo` must agree within ~1 percentage point. Unify them.

**R4 — Meaningful, differentiated confidence (no more flat 0.3).** Confidence must spread across ≈
[0.2, 0.9] and be driven by: (a) signal **consensus** (agreement direction/magnitude across strategies),
(b) **data quality** (history length, whether real fundamentals exist for the asset class),
(c) **strategy reliability** (backtest Sharpe/hit-rate of the contributing strategies), (d) regime
clarity. Two assets with clearly different signal agreement must show clearly different confidence.

**R5 — Calibrated, actionable composite (not everything HOLD).** Recalibrate so scores spread across
the full BUY/HOLD/SELL range and the universe yields a realistic MIX of stances (some STRONG_BUY/BUY,
some SELL/STRONG_SELL), not 22×HOLD. Replace the over-aggressive disagreement shrink with a
**reliability-weighted** mean (weight each signal by confidence × its backtest quality) and a milder
dispersion penalty. Validate: across the 24-asset universe at least ~5 names are BUY+ and ~3 are SELL−.

**R6 — Surface downside honestly.** Every analysis must include: probability of loss, expected
shortfall (CVaR) at the 1Y horizon, max drawdown, and an explicit **bear scenario**. Do not let the
UI/engine present only the rosy case. `prob_positive` must use the fat-tailed dist (no inflated 0.9+).

**R7 — Consistent units & labels.** Every risk/return figure carries an unambiguous unit/horizon
(daily vs annual vs horizon VaR). The schema/labels must distinguish `var95Daily` vs `var95_1Y`.

**R8 — Guardrails & disclaimer.** Nothing may emit NaN/inf/None numerics on the wire. Add a standard
disclaimer surfaced by the API (`/api/health` or analysis meta) and shown in the UI:
"Educational simulation on synthetic market data — not financial advice; projections are model
estimates, not guarantees." A `disclaimer` string field on AssetAnalysis (and a UI banner) is required.

**Re-audit gate:** after implementation, re-run the audit harness (analyze 8 assets + recommendations
+ montecarlo) and assert: no implausible projections (5Y expected within caps), confidence spread > 0.3
range, a realistic stance mix, analysis-vs-MC 1Y agreement within ~1pp, and zero non-finite values.

## Source of truth for each strategy
`docs/research/strategy-catalog.json` → `strategies[]`. Each entry has: `id, name, category,
summary, rules, parameters, computeSignal, computeProjection, assetClasses, sources, priority`.
**Implement `computeSignal` exactly as written** (it maps AVAILABLE DATA → score in [-100,100] +
confidence in [0,1]). Carry `summary/rules/sources` into the StrategyMeta. Skip nothing P1/P2;
implement P3 too (token budget is not a constraint). If a strategy needs cross-sectional ranks,
read them from `ctx.universe` (below).

---

## 1. Backward-compatible context extension

`app/strategies/engine.py` defines `AnalysisContext`. **Add one field** (do not change existing
fields or the builder signature `(ctx) -> StrategySignal`):

```python
@dataclass
class UniverseStats:
    """Cross-sectional metrics across the whole universe, computed once per engine pass.
    All dicts keyed by upper-cased symbol. Percentile helpers return 0..1 (1 = best/highest)."""
    symbols: list[str]
    asset_class: dict[str, str]
    sector: dict[str, str]
    # raw per-symbol metrics used by cross-sectional strategies:
    earnings_yield: dict[str, float]      # ebit/EV  (EV = mktcap + net debt; fallback mktcap)
    roic: dict[str, float]                # ebit / invested capital
    gross_profitability: dict[str, float] # (sales - cogs proxy)/total_assets  (proxy via net_margin*sales/TA if cogs n/a)
    momentum_12_1: dict[str, float]
    momentum_6m: dict[str, float]
    annual_vol: dict[str, float]
    beta: dict[str, float]
    dividend_yield: dict[str, float]      # dividend/price
    shareholder_yield: dict[str, float]   # (dividend + net buyback proxy)/price; proxy ok
    fcf_yield: dict[str, float]
    pe: dict[str, float]; pb: dict[str, float]; peg: dict[str, float]
    ret_52w: dict[str, float]             # price/52w-high - 1 (<=0)
    def percentile(self, metric: str, symbol: str) -> float: ...   # 0..1 rank of symbol within its... universe
    def rank(self, metric: str, symbol: str, ascending: bool=False) -> int: ...
```

Add to `AnalysisContext`: `universe: UniverseStats` and `all_symbols: list[str]`. The engine
builds **one** `UniverseStats` per pass (cache it; invalidate with the analysis cache) and injects
it into every per-asset `AnalysisContext`. Existing builders ignore it; new cross-sectional builders
(`magic-formula`, `low-vol-anomaly`, `betting-against-beta`, `cross-sectional-momentum`, `qmj-quality-minus-junk`,
`gross-profitability`, `dogs-of-dow`, `dual-momentum`, `relative-strength-rotation`, `52w-high`, …) use it.

---

## 2. New indicators — `app/quant/indicators.py` (new file)

Add vectorized + latest-value indicators not already in `technical.py` (do NOT duplicate
sma/ema/macd/rsi/bollinger/zscore/momentum_12_1 — import those from technical.py). Implement:
```
true_range(high, low, close)->np.ndarray ; atr(high, low, close, n=14)->np.ndarray
adx(high, low, close, n=14)->float            # +DI/-DI/ADX; return latest ADX (0..100)
donchian(high, low, n=20)->tuple[float,float] # (upper, lower) latest
supertrend(high, low, close, n=10, mult=3.0)->tuple[float,int]  # (level, direction +1/-1)
ichimoku(high, low, close)->dict              # tenkan, kijun, senkou_a, senkou_b, cloud_pos(+1/0/-1)
williams_r(high, low, close, n=14)->float     # -100..0
stochastic(high, low, close, k=14, d=3)->tuple[float,float]  # (%K, %D)
cci(high, low, close, n=20)->float
keltner(close, high, low, n=20, mult=2.0)->tuple[float,float,float]  # (mid, upper, lower)
obv(close, volume)->np.ndarray ; obv_slope(close, volume, n=20)->float
```
OHLCV comes from `provider.get_candles(symbol, limit)` (each candle dict has o/h/l/c/v) or the
simulator. The engine should make recent OHLC arrays available on the context (add `highs/lows/volumes`
to `AnalysisContext` alongside `closes`) so indicator-based builders don't re-fetch. Be numerically
defensive: short arrays → neutral values, never raise.

---

## 3. Backtesting engine — `app/quant/backtest.py` (new file)

```python
@dataclass
class BacktestMetrics:   # all the 14 catalog metrics (decimals; pct where noted in schema)
    cagr, total_return, ann_vol, sharpe, sortino, calmar, max_drawdown, ulcer_index,
    win_rate, profit_factor, exposure, turnover, cvar95, beta, information_ratio: float

@dataclass
class BacktestResult:
    symbol: str; strategy_id: str
    metrics: BacktestMetrics
    benchmark: BacktestMetrics            # buy & hold same asset
    equity_curve: list[dict]              # [{t, strategy, benchmark}] (downsampled ~120 pts)
    trades: int

def backtest_positions(closes, positions, rf_daily, benchmark='bh', highs=None, lows=None)->BacktestResult-ish
    # positions: np.ndarray in [-1,1] (or {0,1}) aligned to closes; strategy daily return =
    # position[t-1]*asset_return[t] minus turnover cost (e.g. 5 bps * |Δposition|). Compute the
    # 14 metrics from the strategy equity curve and the buy&hold curve. Vectorized.
```
Backtestable strategies expose a **vectorized position series** (`positions(closes, highs, lows,
volumes, params) -> np.ndarray`) — primarily the timing strategies (golden-cross, dual-ma-crossover,
faber-taa, donchian-turtle, supertrend, ichimoku, adx-trend, tsmom, absolute-momentum-overlay,
connors-rsi2, bollinger-squeeze, macd, rsi, etc.). Provide these position functions in the relevant
builder modules and register them in a `POSITION_FUNCS: dict[str, Callable]` so the API/engine can
backtest any supported strategy. Snapshot/fundamental strategies (graham, magic-formula, …) are NOT
time-backtestable per-bar → for those return a buy&hold-only result flagged `supported=False` (add a
`supported: bool` to BacktestResult). Implement metric formulas exactly per the catalog `backtestMetrics`.

---

## 4. Projection engine — `app/quant/projection.py` (new file)

Replaces the weak per-model lognormal blend. Implement the 8 catalog `projectionMethods`:
```python
@dataclass
class HorizonProjection:   # one per horizon; superset of ExpectedReturn
    horizon: str; expected_return_pct: float; low: float; high: float        # fat-tailed CI (Student-t/bootstrap)
    prob_positive: float; annualized_vol: float
    bull_pct: float; base_pct: float; bear_pct: float                        # scenario fan
    cvar_pct: float                                                          # expected shortfall (downside)

def detect_regime(closes)->dict           # {regime:'bull'|'bear'|'neutral', trend, vol_regime, score}
def ensemble_drift(signal_drifts: list[tuple[float,float]])->float  # confidence-weighted, shrunk (James-Stein toward CAPM/0 prior)
def project(closes, returns, signal_drifts, rf_daily, beta=None, capm_drift=None)->list[HorizonProjection]
    # 1) drift = shrinkage(ensemble of projecting strategies' daily drifts, prior=capm_drift)
    #    adjusted by regime (bull boosts, bear damps).
    # 2) vol = GARCH(1,1) forecast (fallback EWMA) per horizon (sqrt-time with vol term structure).
    # 3) CI: fat-tailed — Student-t quantiles (df~5) or block-bootstrap of historical returns, scaled to horizon.
    # 4) scenarios: bull=base+z*vol, bear=base-z*vol (z per horizon), base=drift projection.
    # 5) prob_positive via the fat-tailed dist; cvar_pct = expected shortfall at 95%.
```
The engine uses `project(...)` to produce the blended 5-horizon `expectedReturns` AND a new
`scenarios` block. Existing `ExpectedReturn` fields stay; new scenario/cvar fields are additive.

---

## 5. Strategy builder modules — `app/strategies/builders_*.py` (new files)

To avoid 4 agents editing `registry.py` at once, each category group is its OWN module exposing a
dict. Builder signature unchanged: `Callable[[AnalysisContext], StrategySignal]`.
```
builders_value_quality.py    -> BUILDERS: dict[str,(StrategyMeta, fn)]  + POSITION_FUNCS: dict[str,fn]
builders_momentum_trend.py   -> BUILDERS, POSITION_FUNCS
builders_meanrev_technical.py-> BUILDERS, POSITION_FUNCS
builders_allocation_div_anom.py -> BUILDERS, POSITION_FUNCS
```
Each module: for its assigned ids (see split below), build `StrategyMeta` (id, name, category mapped
to the existing `StrategyCategory` literal — map research categories: value/quality/growth→Valuation
or Fundamental, momentum/trend→Technical, mean-reversion→Statistical or Technical, allocation/risk→Portfolio,
factor→Factor, anomaly→Statistical, dividend→Fundamental, technical→Technical) and a builder fn that
implements `computeSignal` from the catalog, sets rationale (plain English w/ the numbers), formula,
metrics, confidence, and `horizons` via the projection helper where the strategy implies a drift.

**Category split (read catalog for each id's computeSignal):**
- `builders_value_quality.py`: graham-defensive, graham-number, net-net-ncav, magic-formula,
  acquirers-multiple, owner-earnings-yield, buffett-quality-fair-price, qmj-quality-minus-junk,
  gross-profitability, return-on-capital-compounder, fama-french-5, gordon-reverse-implied-growth,
  peg-lynch, canslim  (value/quality/growth/factor — 14)
- `builders_momentum_trend.py`: dual-momentum, tsmom, cross-sectional-momentum, 52w-high,
  relative-strength-rotation, frog-in-the-pan-momentum, faber-taa, donchian-turtle, golden-cross,
  dual-ma-crossover, supertrend, ichimoku, adx-trend-strength, ma-ribbon, absolute-momentum-overlay (15)
- `builders_meanrev_technical.py`: connors-rsi2, connors-cumulative-rsi2, zscore-reversion,
  pairs-trading, bollinger-squeeze, stochastic-oscillator, williams-r, cci-reversion, keltner-reversion,
  obv-volume-trend (10)
- `builders_allocation_div_anom.py`: all-weather-risk-parity, vol-target, risk-parity-inverse-vol,
  min-variance, permanent-portfolio, low-vol-anomaly, betting-against-beta, seasonality, chowder-rule,
  dividend-safety, dividend-growth-aristocrats, shareholder-yield, dogs-of-dow, small-dogs-of-dow (14)

Allocation strategies that are portfolio-level (risk-parity, min-variance, permanent-portfolio,
all-weather) produce a *per-asset* signal = that asset's suitability/weight in the corresponding
portfolio (e.g. inverse-vol weight percentile, or "is this a good diversifier") so they fit the
per-asset signal model. `pairs-trading` scores an asset by its spread z-score vs its most-cointegrated
universe peer (use `ctx.universe`/closes). `seasonality` uses the current month (pass `now` in via the
engine — the engine may read the system clock; tests inject a fixed month).

---

## 6. Registry + engine integration (one integration agent)

- `registry.py`: import the 4 BUILDERS dicts + merge with the existing 20 into `STRATEGY_META`
  (now ~73, ordered by category then priority) and `SIGNAL_BUILDERS`; merge POSITION_FUNCS into a
  `POSITION_FUNCS` export. Keep the existing 20 intact. `build_signals(ctx)` runs all in META order,
  each guarded (failure → neutral HOLD, never raises).
- `engine.py`: build `UniverseStats` once per pass; extend `AnalysisContext` (universe, all_symbols,
  highs, lows, volumes); use `projection.project(...)` for blended horizons + scenarios; attach a
  lightweight `backtest` summary to backtestable signals if cheap (else expose via API only). Keep
  `analyze()` returning exactly 5 blended `expectedReturns` and now ~73 signals. Keep recommendations/
  rankings/market_summary working. Performance: cache UniverseStats + per-symbol analyses; vectorize.

---

## 7. Schema additions (`app/schemas.py`, additive)
```ts
// extend ExpectedReturn with optional scenario fields (keep existing required ones):
ExpectedReturn += { bullPct?: number; basePct?: number; bearPct?: number; cvarPct?: number }
RegimeInfo { regime: 'bull'|'bear'|'neutral'; trend: number; volRegime: 'low'|'normal'|'high'; score: number }
BacktestMetricsDTO { cagr, totalReturn, annVol, sharpe, sortino, calmar, maxDrawdown, ulcerIndex,
  winRate, profitFactor, exposure, turnover, cvar95, beta, informationRatio: number }
BacktestResultDTO { symbol, strategyId, supported, trades, metrics, benchmark: BacktestMetricsDTO,
  equityCurve: { t:number; strategy:number; benchmark:number }[] }
StrategySignal += { backtest?: BacktestMetricsDTO | null }          // optional summary
AssetAnalysis += { regime?: RegimeInfo; strategyCount: number }     // additive
```

## 8. API additions (`app/api/`, additive — keep existing routes)
| Method | Path | Returns |
|---|---|---|
| GET | `/api/assets/{symbol}/backtest?strategy=<id>` | `BacktestResultDTO` |
| GET | `/api/strategies/{id}/backtest?symbol=<sym>` | `BacktestResultDTO` |
| GET | `/api/strategies/leaderboard?symbol=<sym>&limit=20` | strategies ranked by backtest Sharpe/CAGR for that asset |

`/api/strategies` now returns ~73 metas. 404 for unknown strategy/symbol; `supported:false` (not error)
for non-backtestable strategies.

## 9. Tests (`backend/tests/`, additive)
- `test_indicators.py`: ATR/ADX/Donchian/Supertrend/Ichimoku/Williams%R/Stochastic/CCI/Keltner/OBV on
  known series.
- `test_backtest.py`: a constant-long position reproduces buy&hold; a known position series → known
  CAGR/Sharpe/maxDD; profit_factor/win_rate sane; flat position → ~0 return; metrics finite.
- `test_projection.py`: 5 horizons; bull >= base >= bear each horizon; CI widens with horizon;
  prob_positive in [0,1]; cvar <= base; regime detection returns valid label.
- `test_strategies_v2.py`: registry now has >= 70 strategies; every id has a builder + meta + sources;
  engine.analyze() still returns exactly 5 expectedReturns and now >= 70 signals; never raises for any
  symbol; cross-sectional strategies differ across assets; seasonality is deterministic given injected month.
- Existing `test_strategies.py`/`test_api.py` must still pass (update the ">=18" / "20 signals" counts
  to the new totals where they assert exact equality — change `== 20` to `>= 70`, keep `== 5` horizons).

## 10. Frontend implications (for the later frontend build — not this phase)
Strategy Lab gets ~73 strategies grouped by family with sources; each strategy page shows its backtest
equity curve vs buy&hold + the 14 metrics; Asset Detail shows the regime badge + bull/base/bear scenario
fan chart + CVaR; a "Strategy Leaderboard" per asset ranks strategies by realized Sharpe/CAGR.
