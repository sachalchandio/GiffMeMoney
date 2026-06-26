# Strategy & Risk Review — GiffMeMoney

**Status:** evidence-led review for the record.
**Scope:** the allocation advisor, the Markowitz optimizer, the invest/portfolio
service, and the auto-trader bot backtest.
**Standing caveat (read first):** GiffMeMoney is a **SIMULATION on synthetic
(made-up) market data**. There is no real market edge in any number this app
produces. Nothing below — and nothing the app emits — is a forecast, a promise,
or financial advice. This document records what was claimed, what was actually
true in the code, and what was changed.

All file:line references are to this branch at review time. Verified by running
both suites once: **backend 350 passed**, **frontend 87 passed** (the baseline
283 / 79 grew because new-behaviour tests were added alongside the fixes; no
test was deleted to hide a regression — old behaviour tests were *updated*).

---

## 1. Bottom line

The headline ask that motivated this review — **turn $20 into $2,000 in two
months** — implies roughly **+11.6% growth every single day, compounded** (a
100x multiple over ~42 trading days). That is not achievable by any real or
simulated strategy and never was. It is **unsupported by design**, and the code
now says so out loud:

- The feasibility guard (`backend/app/strategies/feasibility.py:46`) flags any
  target implying ≥ 3%/day compounding as a "fantasy, not a plan", and the
  advisor surfaces that warning verbatim (`backend/app/invest/advisor.py:230`).
- Every advice and every backtest carries a mandatory synthetic-data /
  paper-trading disclaimer (`SYNTHETIC_DATA_NOTE`,
  `feasibility.py:33`; `BOT_DISCLAIMER`, `backend/app/schemas.py:1118`).

The review found that the *original* engine, while numerically careful, had real
strategy and risk flaws: it deployed 100% of every dollar, applied no loss
controls after a buy, ignored tail risk in the optimizer, emitted $0.00 "dust"
legs, and — most seriously — used a **look-ahead full-history score to pick
names at past rebalances** in the bot backtest, which inflates simulated
performance. Those are the flaws fixed in this change. Even with them fixed, the
results remain a synthetic-data simulation with **no real alpha** — the honesty
labels are load-bearing, not decorative.

---

## 2. Verdict table — external claims C1–C8

| Claim | One-line summary | Verdict | Evidence (file:line) |
|------|------------------|---------|----------------------|
| **C1** | Advisor is always ~100% invested; no cash sleeve. | **Confirmed** (now fixed) | Optimizer enforces long-only, fully-invested weights (`quant/portfolio.py:36-37`, `_normalize_weights` sums to 1 at `:130-153`); advisor consumed them directly with no cash carve-out. |
| **C2** | No post-buy loss controls (stop / trailing / drawdown / take-profit). | **Confirmed** (now fixed) | The original `invest` path (`invest/portfolio_service.py:265-363`) only validated funds and recorded buys — there was no stop, trailing stop, take-profit, or drawdown breaker anywhere on the positions side. |
| **C3** | Advisor optimize is bounded 0..1 with sum(w)=1 — i.e. structurally 100% invested. | **Confirmed** | Box constraints `0 ≤ w_i ≤ cap` and the budget equality `sum(w)=1` (`quant/portfolio.py:264-265`); the equal-weight fallback also sums to 1 (`:148`). The optimizer correctly never holds cash — the cash decision belongs one layer up, in the advisor. |
| **C4** | `_blend_horizons` drops bull/base/bear/cvar, so basket CVaR is always `None`. | **Confirmed** (now fixed) | The blend now carries the downside fan with per-field weights so a `None` on one pick doesn't poison the mean, and emits a finite `cvarPct` whenever any pick supplies one (`invest/advisor.py:666-781`, esp. the `bull/base/bear/cvar` accumulators at `:728-740` and resolution at `:748-755`). The flaw it fixes was real. |
| **C5** | `_build_items` emits $0.00 dust legs at small amounts. | **Confirmed** (now fixed) | Legs below both a tiny weight and a tiny notional are dropped and survivors renormalized (`invest/advisor.py:554-629`, dust test at `:592-605`, rounding-zero guard at `:613-617`). |
| **C6** | (No-alpha / synthetic-data honesty.) | **Confirmed** | Results are computed on `app/market/simulator.py` synthetic series; every surface is labelled (`feasibility.py:33`, `schemas.py:1118`, advisor sets `synthetic_data=True` at `invest/advisor.py:249,304`). |
| **C7** | `indicators.py` ADX divides without `np.errstate`, unlike its 7 sibling sites. | **Confirmed** (now fixed) | Both ADX divides are now wrapped in `np.errstate(divide="ignore", invalid="ignore")` like the siblings (`quant/indicators.py:322-329` and `:382-388`); output is unchanged, the `RuntimeWarning` is gone. |
| **C8** | (Reserved / not substantiated on this branch.) | **Unverified** | No corresponding defect was located in the code under review. Recorded as not confirmed rather than asserted either way. |

**Reading the table:** C1–C5 and C7 are confirmed flaws in the original design.
C3 is "confirmed" in the narrow, factual sense — the optimizer *is* fully
invested — but that is correct behaviour for a Markowitz solver; the real fix is
a cash sleeve **above** it (see §4), not changing the solver's contract. C6 is
confirmed as an honest self-description. C8 had no matching evidence and is left
unverified rather than padded into a false positive.

---

## 3. Additional flaws found during the review

Beyond the external list, the review surfaced these, in roughly descending
severity:

1. **Look-ahead in bot *selection* (most serious).**
   The auto-trader chose which names to hold at every *past* rebalance by
   ranking on a **full-history composite score** that was computed once over the
   entire window — i.e. it could "see" prices that came *after* the rebalance
   date. Picking past winners with future knowledge systematically inflates a
   backtest. Cited originally at `bot/engine.py` around the candidate-score
   freeze and its reuse inside `_rebalance`.
   **Nuance worth recording:** the *per-bar math* was already point-in-time —
   regime detection used `index_closes[:t+1]` and the base-weight estimator used
   a trailing window ending at `t` (`bot/engine.py:362`, `:692-695`). The leak
   was specifically in **name selection**, not in the sizing/risk math. The fix
   ranks selection by a trailing, point-in-time momentum score from
   `prices[:t+1]` (`bot/engine.py:_pit_scores`, `:610-650`), using the frozen
   composite only as a deterministic tie-breaker at `t==0`
   (`:502-518`). The module docstring now states this plainly (`:13-19`).

2. **Synthetic, no-alpha, no survivorship realism.**
   The backtest universe is a fixed, hand-built synthetic set — there is no real
   price-generating process and no survivorship modelling (no names enter or
   leave, none go to zero). A clean equal-weight buy-and-hold benchmark is
   computed (`bot/engine.py:347-349`) and the bot is reported *relative* to it,
   which is the honest framing, but it must not be read as evidence of edge.

3. **No CVaR / tail objective and no per-name cap in the optimizer.**
   Mean-variance alone penalises a symmetric variance and is blind to the
   deep-loss tail, and nothing stopped a single name from dominating a basket.
   Both are now addressed (`min_cvar` objective + `max_weight` cap; see §4).

4. **Horizon mismatch.**
   The advisor sized every basket on **annual (1Y)** `mu`/`cov` regardless of
   the actual ask, so a 2-month request was sized on year-long risk. The
   estimator now scales `mu`/`cov` to the requested `horizon_days`
   (`invest/advisor.py:362-431`, `mu_i = exp(mean(log r)·h)-1`,
   `S = Cov·h`).

5. **No out-of-sample (OOS) split.**
   Parameters and the candidate set are chosen on the same window the
   performance is measured over. There is no train/test or walk-forward split,
   so even the (now point-in-time) numbers are in-sample. This is a known,
   documented limitation rather than a fixed item — see §6.

---

## 4. Fixes applied in this change

Each fix is paired with the flaw it closes and where to read it.

| # | Fix | Closes | Where |
|---|-----|--------|-------|
| 1 | **Cash sleeve.** Advisor deploys a *risky fraction* in `[0,1]` driven by risk profile + regime + conviction; the rest is parked as cash. Reported as `cashWeight` / `cashAmount` with the invariant `Σ item.weight + cashWeight ≈ 1`. | C1, C3 | `invest/advisor.py:264-306`, `_risky_fraction` `:473-548`; DTO `schemas.py:1057-1103` |
| 2 | **CVaR objective + per-name cap in the optimizer.** New `min_cvar` (historical expected shortfall) objective and a `max_weight` box-cap, floored to `1/n` for feasibility. Conservative profiles use `min_cvar`; all profiles cap concentration. | additional #3 | `quant/portfolio.py:portfolio_cvar` `:287-330`, `optimize` `:333-464`, `_resolve_cap` `:208-234`; advisor wiring `invest/advisor.py:69-82,433-467` |
| 3 | **Invest-path stop-loss / trailing-stop / take-profit / drawdown breaker.** Opt-in `RiskPolicy` (all rules OFF by default) evaluated against held positions; protective sells reuse the normal `sell` accounting. Position high-water mark ratchets on every mark-to-market. | C2 | `invest/portfolio_service.py:evaluate_risk` `:503-609`, drawdown breaker `:615-672`, peak ratchet `:196-203,246-250` |
| 4 | **Basket CVaR / downside fan.** Horizon blend carries `bull/base/bear/cvar` with per-field weights, so the basket reports a finite `cvarPct` instead of always `None`. | C4 | `invest/advisor.py:666-781` |
| 5 | **Dust removal.** Drop legs below both a tiny weight and a tiny notional; renormalise survivors back to the intended risky fraction (dust is redistributed, not silently leaked to cash). | C5 | `invest/advisor.py:554-629` |
| 6 | **Point-in-time bot selection.** Rank names per rebalance by a trailing momentum score from `prices[:t+1]`; frozen composite is a tie-breaker only. | additional #1 | `bot/engine.py:_pit_scores` `:610-650`, `_rebalance` `:502-518` |
| 7 | **Feasibility guard + honesty labels.** Impossible-target warning surfaced (never alters the basket); synthetic-data / paper-trading disclaimers on every advice + backtest. | C6, §1 | `strategies/feasibility.py:33,46`; advisor `:230-234,249,304`; `schemas.py:1118` |
| 8 | **`np.errstate`-wrapped ADX divides.** | C7 | `quant/indicators.py:322-329,382-388` |
| 9 | **Horizon-scaled sizing.** `mu`/`cov` scaled to the requested horizon. | additional #4 | `invest/advisor.py:362-431` |

What was **deliberately not** done: the optimizer's long-only / fully-invested
contract was kept (the cash decision lives in the advisor, fix #1); no attempt
was made to manufacture alpha or to add an OOS split (recorded as a limitation,
§6); risk rules remain **OFF by default** so existing behaviour is unchanged for
callers who don't opt in.

---

## 5. Verification

- Backend: `PYTHONPATH=backend backend/.venv/Scripts/python.exe -m pytest backend/tests`
  → **350 passed**.
- Frontend: `npm test -- --run` in `frontend/` → **87 passed** (11 files).
- New-behaviour coverage was added (not substituted): cash-sleeve, `min_cvar` /
  cap, invest-path stop-loss / trailing / take-profit / drawdown, basket CVaR,
  dust removal, point-in-time selection, and the feasibility guard each have
  tests across `test_invest.py`, `test_bot.py`, `test_feasibility.py`,
  `test_metrics.py`, `test_projection.py`, and `test_indicators.py`.

---

## 6. Honest residual limitations

These are **not** fixed by this change and should not be claimed as such:

- **No real alpha.** Synthetic data; the simulation cannot demonstrate a
  real-world edge, by construction.
- **No out-of-sample / walk-forward split.** Selection and measurement share a
  window; numbers are in-sample even after the look-ahead fix.
- **No survivorship / delisting realism.** The universe is fixed; nothing fails.
- **Trading frictions are stylised.** A flat 5 bps cost
  (`bot/engine.py:_COST_RATE`, `:92`); no slippage, spread, borrow, or market
  impact.
- **Estimation error is unmodelled.** `mu`/`cov` are point estimates from a
  short history; the optimizer treats them as truth.

The single most important sentence in this review: **a +11.6%/day,
$20→$2,000-in-two-months outcome is impossible, the app now refuses to imply
it, and the fixes above make the simulation safer and more honest — not
profitable.**
