"""Tests for the Real-Time mode (live-feeling, multi-venue PAPER simulation).

HONESTY / SAFETY (this is a finance tool). These tests pin the properties that
keep the Real-Time mode honest and safe:

* it spreads capital wider as equity grows but **never past 80 venues** (and
  never below the diversification floor);
* the self-updating predictor genuinely **learns** (weights move) yet stays
  **humble** — outputs are probabilities in ``[0, 1]`` near a coin flip;
* returns stay **realistic** — a long run cannot fabricate a $20→$4k moonshot;
* the engine is **defensive** (degenerate configs never raise) and every payload
  carries the simulation **disclaimer** with the ``syntheticData`` flag.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.livesim.engine import (
    LIVESIM_MAX_VENUES,
    LIVESIM_MIN_VENUES,
    create_session,
    tick,
)
from app.livesim.predictor import FEATURE_NAMES, OnlinePredictor, features_from_path
from app.livesim.store import LiveSimStore
from app.schemas import LIVESIM_DISCLAIMER


def _run(session, ticks: int, steps: int = 4):
    """Advance a session ``ticks`` times and return it."""
    for _ in range(ticks):
        tick(session, steps)
    return session


# ---------------------------------------------------------------------------
# Session lifecycle + spreading rule
# ---------------------------------------------------------------------------


def test_start_creates_book_at_starting_equity() -> None:
    """A new session holds the requested venues and starts fully in cash."""
    s = create_session("t1", amount=20.0, max_venues=20)
    assert len(s.venues) == 20
    assert math.isclose(s.equity(), 20.0, rel_tol=1e-9)
    assert s.cash == 20.0
    assert s.step == 0


def test_venue_count_is_capped_at_80() -> None:
    """Requesting more than 80 venues is clamped to the hard cap."""
    s = create_session("t2", amount=1000.0, max_venues=500)
    assert len(s.venues) == LIVESIM_MAX_VENUES
    assert s.max_venues == LIVESIM_MAX_VENUES


def test_venue_count_has_a_floor() -> None:
    """Requesting too few venues is raised to the diversification floor."""
    s = create_session("t3", amount=20.0, max_venues=1)
    assert len(s.venues) == LIVESIM_MIN_VENUES


def test_capital_spreads_wider_as_equity_grows() -> None:
    """The spread width scales with equity / dollars-per-venue, within the cap.

    This is the user's mental model: $20 ≈ 20 venues, more money ≈ more venues,
    hard-capped at 80.
    """
    from app.api.livesim import _state_dto

    small = _state_dto(create_session("s-small", amount=10.0, dollars_per_venue=1.0, max_venues=80))
    assert small.venues_target == 10  # 10 / $1, above the floor, under the cap

    big = _state_dto(create_session("s-big", amount=250.0, dollars_per_venue=1.0, max_venues=80))
    assert big.venues_target == LIVESIM_MAX_VENUES  # capped at 80 even though 250/$1 = 250


# ---------------------------------------------------------------------------
# Ticking: advances, trades, stays finite + realistic
# ---------------------------------------------------------------------------


def test_tick_advances_and_stays_finite() -> None:
    """Ticking advances the clock, logs trades, and keeps the book finite."""
    s = create_session("t4", amount=20.0, max_venues=20)
    _run(s, ticks=60, steps=4)
    assert s.step >= 200
    assert s.day >= 1
    assert math.isfinite(s.equity()) and s.equity() >= 0.0
    assert s.cash >= 0.0
    assert len(s.trades) > 0           # it actually traded
    assert len(s.daily_pnl) >= 1       # daily P&L accrued
    assert len(s.equity_curve) >= 10


def test_returns_stay_realistic_never_a_moonshot() -> None:
    """A long run cannot fabricate a $20→$4k moonshot — honesty by construction."""
    s = create_session("t5", amount=20.0, max_venues=40, signal="momentum")
    _run(s, ticks=300, steps=4)
    total_pct = (s.equity() / 20.0 - 1.0) * 100.0
    # Plausible band: it can lose a lot or gain modestly, but never +19,900%.
    assert -95.0 < total_pct < 300.0, f"unrealistic return: {total_pct:.1f}%"


def test_engine_never_raises_on_degenerate_config() -> None:
    """Degenerate configs degrade to a sane session and tick without raising."""
    for amt in (0.0, -50.0, float("nan")):
        s = create_session("deg", amount=amt, max_venues=0, rebalance_every=0, steps_per_tick=0)
        _run(s, ticks=5, steps=3)
        assert math.isfinite(s.equity())
        assert len(s.venues) >= LIVESIM_MIN_VENUES


# ---------------------------------------------------------------------------
# The self-updating predictor: learns, stays humble
# ---------------------------------------------------------------------------


def test_predictor_outputs_probability_and_learns() -> None:
    """Predictions stay in [0, 1]; consistent up-moves push the model to predict up."""
    p = OnlinePredictor()
    x = np.array([0.02, 0.03, 0.4, 0.02, 0.01], dtype=np.float64)
    before = p.w.copy()
    for _ in range(80):
        proba = p.predict_proba(x)
        assert 0.0 <= proba <= 1.0
        p.update(x, 1.0)  # it always went up
    assert not np.allclose(before, p.w)        # it learned (weights moved)
    assert p.predict_proba(x) > 0.5            # toward "up"
    assert np.all(np.isfinite(p.w))


def test_predictor_features_are_point_in_time() -> None:
    """Features at bar t depend only on prices[:t+1] (no look-ahead)."""
    rng = np.random.default_rng(2)
    prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, size=120))
    t = 60
    a = features_from_path(prices, t)
    future = prices.copy()
    future[t + 1 :] *= 3.0
    b = features_from_path(future, t)
    assert np.allclose(a, b)
    assert a.size == len(FEATURE_NAMES)


def test_live_predictions_stay_humble() -> None:
    """Across a live run, venue predictions stay valid and not over-confident."""
    s = create_session("t6", amount=20.0, max_venues=20)
    _run(s, ticks=80, steps=4)
    probs = [v.pred_up for v in s.venues]
    assert all(0.0 <= p <= 1.0 for p in probs)
    # On edge-free data the average confidence stays modest (near a coin flip).
    avg_conf = float(np.mean([abs(p - 0.5) * 2.0 for p in probs]))
    assert avg_conf < 0.6, f"suspiciously confident on noise: {avg_conf:.2f}"


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def test_store_start_tick_stop() -> None:
    """The store creates, advances, finds and removes sessions."""
    store = LiveSimStore()
    s = store.start(amount=20.0, max_venues=10)
    assert store.get(s.id) is not None
    advanced = store.tick(s.id, 4)
    assert advanced is not None and advanced.step >= 1
    assert store.tick("nope-not-real", 4) is None
    assert store.stop(s.id) is True
    assert store.get(s.id) is None


def test_store_evicts_oldest_over_cap() -> None:
    """The store never grows past its cap (oldest sessions are evicted)."""
    store = LiveSimStore(max_sessions=8)
    for _ in range(20):
        store.start(amount=20.0, max_venues=6)
    assert store.count() <= 8


# ---------------------------------------------------------------------------
# API + disclaimer
# ---------------------------------------------------------------------------


def test_disclaimer_is_honest() -> None:
    """The disclaimer denies real money and over-promised returns."""
    text = LIVESIM_DISCLAIMER.lower()
    assert "simulation" in text
    assert "no real money" in text or "$0 real" in text
    assert "not financial advice" in text
    assert "guarantee" not in text


def test_start_tick_routes(client: TestClient) -> None:
    """POST /start then /tick advance a session and carry the disclaimer."""
    r = client.post("/api/livesim/start", json={"amount": 20, "maxVenues": 20})
    assert r.status_code == 200
    body = r.json()
    assert body["disclaimer"] == LIVESIM_DISCLAIMER
    assert body["syntheticData"] is True
    assert body["venuesMax"] == 20
    sid = body["sessionId"]
    assert len(body["venues"]) == 20

    r2 = client.post("/api/livesim/tick", json={"sessionId": sid, "steps": 8})
    assert r2.status_code == 200
    t = r2.json()
    assert t["step"] >= 1
    assert math.isfinite(t["equity"])
    assert -95.0 < t["totalPnlPct"] < 300.0  # realistic

    # Read-only state, then stop.
    assert client.get(f"/api/livesim/state/{sid}").status_code == 200
    assert client.post(f"/api/livesim/stop/{sid}").json()["ok"] is True


def test_tick_unknown_session_404(client: TestClient) -> None:
    """Ticking an unknown session is a clean 404, never a 500."""
    r = client.post("/api/livesim/tick", json={"sessionId": "does-not-exist", "steps": 4})
    assert r.status_code == 404


def test_livesim_routes_carry_disclaimer_in_openapi(client: TestClient) -> None:
    """The state-bearing /api/livesim routes disclose the simulation in their docs."""
    schema = client.get("/openapi.json").json()
    for path in ("/api/livesim/start", "/api/livesim/tick", "/api/livesim/state/{session_id}"):
        ops = schema["paths"].get(path, {})
        assert ops, f"missing route {path}"
        for op in ops.values():
            assert LIVESIM_DISCLAIMER in op.get("description", ""), f"{path} missing disclaimer"
