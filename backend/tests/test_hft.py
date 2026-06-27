"""Tests for the High-Frequency Simulation Lab (paper-only, synthetic data).

HONESTY / SAFETY (this is a finance tool). These tests pin the lab's whole
*reason to exist* — that it tells the truth about fast trading:

* **costs create drag, and more turnover means more drag** — with zero costs net
  equals gross exactly; with real costs net is strictly below gross, and across
  the turnover sweep the cost drag rises with turnover (the toll is linear);
* **the net-of-cost optimum trades far less than the hyperactive setting** — the
  every-bar point is the highest-turnover, and the net-return optimum sits at
  lower turnover (the honest headline: trading faster feeds the spread);
* **signals never look ahead** — a signal at bar ``t`` depends only on
  ``prices[: t + 1]``; mutating the future cannot change it;
* **risk gates actually fire** — on a crafted crash the trailing stop / drawdown
  breaker flatten the book, so the strategy loses far less than buy-&-hold;
* the engine is **deterministic** and **never raises** on degenerate configs, and
  every payload carries the mandatory :data:`~app.schemas.HFT_DISCLAIMER` and the
  ``syntheticData`` honesty flag — nothing implies microsecond trading or a real
  edge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.hft.costs import COST_PRESETS, get_cost_model
from app.hft.execution import SimSpec, run_sim
from app.hft.intraday import IntradaySeries, generate_intraday
from app.hft.lab import default_intervals, run_sweep
from app.hft.signals import raw_exposure
from app.schemas import HFT_DISCLAIMER


def _finite(*xs: float) -> bool:
    """True iff every value is a finite float."""
    return all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in xs)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def test_cost_preset_round_trip_math() -> None:
    """Round-trip bps equals twice the per-side spread + fee (zero participation)."""
    crypto = COST_PRESETS["retail-crypto"]
    assert math.isclose(crypto.round_trip_bps(), 2.0 * (4.0 + 10.0), rel_tol=1e-9)
    equity = COST_PRESETS["retail-equity"]
    assert math.isclose(equity.round_trip_bps(), 2.0 * 1.0, rel_tol=1e-9)
    # Zero preset is genuinely free; the expensive one is the worst.
    assert COST_PRESETS["zero"].round_trip_bps() == 0.0
    assert COST_PRESETS["retail-crypto-expensive"].round_trip_bps() > crypto.round_trip_bps()
    # Slippage adds cost as participation rises.
    assert crypto.per_side_fraction(0.5) > crypto.per_side_fraction(0.0)


def test_unknown_cost_preset_falls_back() -> None:
    """An unknown preset id resolves to the realistic default, never raises."""
    assert get_cost_model("nope").key == get_cost_model(None).key


# ---------------------------------------------------------------------------
# Costs create drag; zero costs do not
# ---------------------------------------------------------------------------


def test_zero_cost_means_net_equals_gross() -> None:
    """With the frictionless preset the net and gross paths are identical."""
    spec = SimSpec(symbol="BTC", amount=20.0, days=20, rebalance_interval=1, cost_preset="zero")
    res = run_sim(spec)
    m = res.metrics
    # It still trades (turnover > 0) but pays nothing, so there is no drag.
    assert m.turnover > 0.0
    assert abs(m.cost_drag_pct) < 1e-6
    assert math.isclose(m.gross_return_pct, m.net_return_pct, abs_tol=1e-6)


def test_real_costs_create_positive_drag() -> None:
    """With real costs the net return is strictly below gross (drag > 0)."""
    spec = SimSpec(symbol="BTC", amount=20.0, days=20, rebalance_interval=1, cost_preset="retail-crypto")
    res = run_sim(spec)
    m = res.metrics
    assert m.turnover > 0.0
    assert m.cost_drag_pct > 0.0
    assert m.net_return_pct < m.gross_return_pct + 1e-9
    assert m.final_net_value >= 0.0
    assert _finite(m.sharpe_net, m.sharpe_gross, m.max_drawdown_pct, m.hit_rate_pct)


def test_more_turnover_means_more_cost_drag() -> None:
    """Across the turnover sweep, cost drag rises with turnover (a linear toll).

    The mechanical truth the lab demonstrates: every trade pays a fixed fraction,
    so total drag is essentially turnover times that fraction. We assert a strong
    positive turnover→drag relationship and that the hyperactive (every-bar)
    setting carries the most turnover.
    """
    base = SimSpec(symbol="BTC", amount=20.0, days=30, signal="meanrev", cost_preset="retail-crypto")
    sweep = run_sweep(base)
    assert len(sweep.points) >= 4

    turnover = np.array([p.turnover for p in sweep.points], dtype=float)
    drag = np.array([p.cost_drag_pct for p in sweep.points], dtype=float)

    # Strong positive correlation between turnover and the cost it incurs.
    corr = float(np.corrcoef(turnover, drag)[0, 1])
    assert corr > 0.85, f"turnover↔drag correlation too weak: {corr:.3f}"

    # The naive 'every bar' point has the highest turnover and a large drag.
    assert sweep.naive_fast is not None
    assert sweep.naive_fast.interval == 1
    assert math.isclose(sweep.naive_fast.turnover, float(np.max(turnover)), rel_tol=1e-6)
    assert sweep.naive_fast.cost_drag_pct >= float(np.median(drag))


def test_net_optimum_trades_less_than_hyperactive() -> None:
    """The net-of-cost optimum trades no more than the re-decide-every-bar setting.

    This is the lab's honest headline: hammering the market every bar is the
    worst net outcome; the best after-cost setting sits at lower turnover.
    """
    base = SimSpec(symbol="BTC", amount=20.0, days=30, signal="meanrev", cost_preset="retail-crypto")
    sweep = run_sweep(base)
    assert sweep.optimum_by_net_return is not None and sweep.naive_fast is not None
    assert sweep.optimum_by_net_return.turnover <= sweep.naive_fast.turnover + 1e-9
    # And the optimum's net beats the hyperactive net (less bleed).
    assert sweep.optimum_by_net_return.net_return_pct >= sweep.naive_fast.net_return_pct - 1e-9
    # The verdict names the mechanism honestly.
    v = sweep.verdict.lower()
    assert "turnover" in v or "spread" in v
    assert "portions" in v or "lively" in v


# ---------------------------------------------------------------------------
# No look-ahead
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("signal", ["meanrev", "momentum"])
def test_signal_is_point_in_time(signal: str) -> None:
    """A signal at bar ``t`` ignores everything after ``t`` (no look-ahead)."""
    rng = np.random.default_rng(4)
    n = 200
    prices = 100.0 * np.cumprod(1.0 + (0.0003 + rng.normal(0.0, 0.01, size=n)))
    t = 120

    before = raw_exposure(signal, prices, t, lookback=30)
    future = prices.copy()
    future[t + 1 :] *= 4.0  # violently corrupt the future
    after = raw_exposure(signal, future, t, lookback=30)

    assert math.isclose(before, after, abs_tol=1e-12)
    assert -1.0 <= before <= 1.0


def test_long_only_signal_never_goes_short() -> None:
    """With shorting disallowed, exposure stays in [0, 1]."""
    rng = np.random.default_rng(9)
    n = 150
    prices = 100.0 * np.cumprod(1.0 + (-0.001 + rng.normal(0.0, 0.012, size=n)))
    for t in range(40, n):
        e = raw_exposure("meanrev", prices, t, lookback=20, allow_short=False)
        assert 0.0 <= e <= 1.0


# ---------------------------------------------------------------------------
# Risk gates fire on a crash
# ---------------------------------------------------------------------------


def test_risk_gates_cut_losses_on_a_crash() -> None:
    """On a crafted rally-then-crash, the stop / breaker beat buy-&-hold.

    The path climbs steadily (momentum goes long), then crashes ~40%. With the
    trailing stop and drawdown breaker on, the strategy flattens near the top of
    the fall, so it loses far less than buy-&-hold and its drawdown is bounded.
    """
    up = np.linspace(100.0, 130.0, 80)          # steady climb (momentum gets long)
    crash = np.linspace(130.0, 78.0, 30)        # ~40% crash
    tail = np.full(15, 78.0)                     # flat aftermath
    prices = np.concatenate([up, crash, tail]).astype(np.float64)
    series = IntradaySeries(
        symbol="CRASH", prices=prices, bars_per_day=25, days=5, bar_seconds=60, annual_vol=0.5
    )
    spec = SimSpec(
        symbol="CRASH", amount=1000.0, signal="momentum", lookback=10,
        rebalance_interval=1, target_vol=0.0, max_exposure=1.0,
        stop_loss_pct=4.0, max_drawdown_pct=8.0, cooldown_bars=10,
        cost_preset="retail-equity", deadband=0.02,
    )
    res = run_sim(spec, series=series)
    m = res.metrics
    # Buy-&-hold eats the whole crash (~-22% from 100→78); the gated strategy
    # loses much less and its drawdown is bounded well above the crash depth.
    assert m.net_return_pct > m.buy_hold_return_pct + 1e-9
    assert m.max_drawdown_pct > -20.0
    assert _finite(m.net_return_pct, m.max_drawdown_pct, m.final_net_value)


# ---------------------------------------------------------------------------
# Determinism + defensiveness
# ---------------------------------------------------------------------------


def test_simulation_is_deterministic() -> None:
    """The same spec yields byte-identical metrics across runs."""
    spec = SimSpec(symbol="ETH", amount=20.0, days=20, signal="meanrev")
    a = run_sim(spec).metrics
    b = run_sim(spec).metrics
    assert a == b


def test_intraday_path_is_deterministic_and_positive() -> None:
    """The synthetic path is reproducible and strictly positive."""
    s1 = generate_intraday("BTC", days=10, bars_per_day=40)
    s2 = generate_intraday("BTC", days=10, bars_per_day=40)
    assert np.array_equal(s1.prices, s2.prices)
    assert np.all(s1.prices > 0.0)
    assert s1.prices.size == 10 * 40 + 1


@pytest.mark.parametrize(
    "spec",
    [
        SimSpec(amount=0.0),                       # nothing to invest
        SimSpec(amount=-50.0),                     # negative capital
        SimSpec(amount=float("nan")),              # non-finite amount
        SimSpec(amount=20.0, days=0),              # no days (clamped)
        SimSpec(amount=20.0, rebalance_interval=0),  # bad cadence
        SimSpec(amount=20.0, lookback=0, deadband=-1.0, target_vol=-1.0),  # absurd params
    ],
)
def test_run_sim_never_raises(spec: SimSpec) -> None:
    """A degenerate / adversarial spec degrades to a finite result, never raises."""
    res = run_sim(spec)
    m = res.metrics
    assert _finite(m.net_return_pct, m.gross_return_pct, m.cost_drag_pct, m.final_net_value)
    assert len(res.net_curve) >= 1


def test_default_intervals_span_fast_to_slow() -> None:
    """The default turnover grid is sorted, unique and starts at 1 (every bar)."""
    grid = default_intervals(78)
    assert grid == sorted(set(grid))
    assert grid[0] == 1
    assert grid[-1] > 78


# ---------------------------------------------------------------------------
# API routes + disclaimer
# ---------------------------------------------------------------------------


def test_disclaimer_is_honest_about_microseconds_and_costs() -> None:
    """The disclaimer denies microsecond trading and states it is synthetic / costed."""
    text = HFT_DISCLAIMER.lower()
    assert "simulation" in text
    assert "synthetic data" in text
    assert "not financial advice" in text
    assert "microsecond" in text
    assert "no real funds" in text
    assert "guarantee" not in text


def test_cost_presets_route(client: TestClient) -> None:
    """GET /api/hft/cost-presets returns the presets with round-trip costs."""
    resp = client.get("/api/hft/cost-presets")
    assert resp.status_code == 200
    body = resp.json()
    keys = {p["key"] for p in body}
    assert {"zero", "retail-equity", "retail-crypto"} <= keys
    crypto = next(p for p in body if p["key"] == "retail-crypto")
    assert math.isclose(crypto["roundTripBps"], 28.0, rel_tol=1e-6)


def test_simulate_route_returns_curves_and_disclaimer(client: TestClient) -> None:
    """POST /api/hft/simulate returns aligned curves, finite metrics, disclaimer."""
    resp = client.post(
        "/api/hft/simulate",
        json={"symbol": "BTC", "amount": 20, "days": 20, "signal": "meanrev", "costPreset": "retail-crypto"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["disclaimer"] == HFT_DISCLAIMER
    assert body["syntheticData"] is True
    m = body["metrics"]
    assert _finite(m["netReturnPct"], m["grossReturnPct"], m["costDragPct"], m["finalNetValue"])
    # Net is never above gross (costs only subtract).
    assert m["netReturnPct"] <= m["grossReturnPct"] + 1e-9
    assert len(body["netCurve"]) >= 2
    assert len(body["grossCurve"]) == len(body["netCurve"])


def test_sweep_route_returns_optimum_and_verdict(client: TestClient) -> None:
    """POST /api/hft/sweep returns the curve, an optimum and a verdict."""
    resp = client.post(
        "/api/hft/sweep",
        json={"base": {"symbol": "BTC", "amount": 20, "days": 30, "signal": "meanrev"}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["disclaimer"] == HFT_DISCLAIMER
    assert len(body["points"]) >= 4
    assert body["optimumByNetReturn"] is not None
    assert body["naiveFast"]["interval"] == 1
    # The optimum trades no more than the hyperactive setting.
    assert body["optimumByNetReturn"]["turnover"] <= body["naiveFast"]["turnover"] + 1e-9
    assert body["verdict"]


def test_hft_routes_carry_disclaimer_in_openapi(client: TestClient) -> None:
    """Every /api/hft route description carries the simulation disclaimer."""
    schema = client.get("/openapi.json").json()
    hft_paths = {p: v for p, v in schema["paths"].items() if p.startswith("/api/hft")}
    assert hft_paths, "no /api/hft routes registered"
    for path, methods in hft_paths.items():
        for method, op in methods.items():
            assert HFT_DISCLAIMER in op.get("description", ""), (
                f"{method.upper()} {path} description is missing the disclaimer"
            )
