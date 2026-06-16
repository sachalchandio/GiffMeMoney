"""End-to-end API tests over the FastAPI app (Tests set 2).

These tests drive the real application through :class:`fastapi.testclient.TestClient`
(which runs the ASGI app, lifespan and WebSocket handshake in-process), proving
every contract route (section 5) and the WebSocket protocol (section 6):

* ``GET /api/health`` reports ``status=ok`` and the universe size;
* ``GET /api/assets`` (and the ``assetClass`` filter) returns a non-empty
  ``Asset[]``;
* ``GET /api/assets/{symbol}`` + ``/candles`` + ``/analysis`` + ``/montecarlo``
  return 200 with the right shape; an unknown symbol returns 404;
* ``GET /api/recommendations`` is ranked; ``GET /api/strategies`` lists >= 18
  models; ``GET /api/strategies/{id}/rankings`` ranks one model (404 for an
  unknown id);
* ``POST /api/portfolio/optimize`` returns weights that sum to ~1;
* ``GET /api/market/summary`` returns the dashboard summary;
* the ``/ws`` WebSocket sends a ``snapshot`` on connect followed by a ``tick``.

The client is module-scoped and context-managed so the application lifespan
(which starts the background price-tick loop used by the WebSocket test) runs for
the whole module.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.market.universe import symbols as universe_symbols

# Valid discrete stances from the contract (section 4).
VALID_STANCES = {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}

# Canonical horizon labels (section 4).
HORIZON_LABELS = {"1D", "1W", "1M", "1Y", "5Y"}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """A module-scoped TestClient that runs the app lifespan (tick loop)."""
    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_ok(client: TestClient) -> None:
    """/api/health returns status=ok with a numeric time and universe size."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["time"], int)
    assert isinstance(body["universe"], int)
    assert body["universe"] == len(universe_symbols())


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


def test_list_assets_non_empty(client: TestClient) -> None:
    """/api/assets returns a non-empty list of camelCase Asset objects."""
    resp = client.get("/api/assets")
    assert resp.status_code == 200
    assets = resp.json()
    assert isinstance(assets, list)
    assert len(assets) == len(universe_symbols())
    first = assets[0]
    # Wire keys are camelCase (section 4).
    for key in ("symbol", "name", "assetClass", "currency", "price", "change24hPct"):
        assert key in first
    assert first["assetClass"] in {"equity", "crypto", "etf"}


def test_list_assets_filter_by_class(client: TestClient) -> None:
    """The ``assetClass`` query filters the asset list."""
    resp = client.get("/api/assets", params={"assetClass": "crypto"})
    assert resp.status_code == 200
    assets = resp.json()
    assert assets
    assert all(a["assetClass"] == "crypto" for a in assets)


def test_get_asset_detail(client: TestClient) -> None:
    """/api/assets/{symbol} returns the single asset snapshot."""
    resp = client.get("/api/assets/AAPL")
    assert resp.status_code == 200
    asset = resp.json()
    assert asset["symbol"] == "AAPL"
    assert asset["price"] > 0


def test_get_asset_unknown_symbol_404(client: TestClient) -> None:
    """An unknown symbol returns 404 with a ``detail`` body."""
    resp = client.get("/api/assets/ZZZZ_NOPE")
    assert resp.status_code == 404
    assert "detail" in resp.json()


def test_get_candles_shape(client: TestClient) -> None:
    """/api/assets/{symbol}/candles returns up to ``limit`` OHLCV candles."""
    resp = client.get("/api/assets/AAPL/candles", params={"limit": 60})
    assert resp.status_code == 200
    candles = resp.json()
    assert isinstance(candles, list)
    assert 0 < len(candles) <= 60
    c = candles[0]
    for key in ("t", "o", "h", "l", "c", "v"):
        assert key in c
    # OHLC ordering holds for a well-formed candle.
    assert c["h"] >= c["l"]
    assert c["h"] >= c["o"] and c["h"] >= c["c"]
    assert c["l"] <= c["o"] and c["l"] <= c["c"]


def test_get_candles_unknown_symbol_404(client: TestClient) -> None:
    """Candles for an unknown symbol return 404."""
    resp = client.get("/api/assets/ZZZZ_NOPE/candles")
    assert resp.status_code == 404


def test_get_analysis_shape(client: TestClient) -> None:
    """/api/assets/{symbol}/analysis returns a full AssetAnalysis."""
    resp = client.get("/api/assets/MSFT/analysis")
    assert resp.status_code == 200
    body = resp.json()
    assert body["asset"]["symbol"] == "MSFT"
    assert -100.0 <= body["compositeScore"] <= 100.0
    assert body["recommendation"] in VALID_STANCES
    assert 0.0 <= body["confidence"] <= 1.0
    # Exactly five horizons, one per label.
    horizons = body["expectedReturns"]
    assert len(horizons) == 5
    assert {h["horizon"] for h in horizons} == HORIZON_LABELS
    # At least 18 strategy signals.
    assert len(body["signals"]) >= 18
    sig = body["signals"][0]
    for key in ("strategyId", "strategyName", "category", "score", "stance"):
        assert key in sig
    assert sig["stance"] in VALID_STANCES
    # Risk metrics block present.
    rm = body["riskMetrics"]
    for key in (
        "beta",
        "annualVol",
        "sharpe",
        "sortino",
        "var95",
        "cvar95",
        "maxDrawdown",
        "calmar",
    ):
        assert key in rm
    assert 3 <= len(body["topReasons"]) <= 5


def test_get_analysis_unknown_symbol_404(client: TestClient) -> None:
    """Analysis for an unknown symbol returns 404."""
    resp = client.get("/api/assets/ZZZZ_NOPE/analysis")
    assert resp.status_code == 404


def test_get_montecarlo_shape(client: TestClient) -> None:
    """/api/assets/{symbol}/montecarlo returns a MonteCarloResult."""
    resp = client.get(
        "/api/assets/AAPL/montecarlo", params={"horizon": "1M", "sims": 300}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "AAPL"
    assert body["horizon"] in HORIZON_LABELS
    assert body["sims"] == 300
    assert body["steps"] > 0
    assert len(body["bands"]) == body["steps"] + 1
    band = body["bands"][0]
    # Percentile bands are monotonically ordered.
    assert band["p5"] <= band["p25"] <= band["p50"] <= band["p75"] <= band["p95"]
    assert body["finalDistribution"]
    assert 0.0 <= body["probPositive"] <= 1.0
    for key in ("expectedReturnPct", "var95Pct", "cvar95Pct"):
        assert math.isfinite(body[key])


def test_get_montecarlo_unknown_symbol_404(client: TestClient) -> None:
    """Monte Carlo for an unknown symbol returns 404."""
    resp = client.get("/api/assets/ZZZZ_NOPE/montecarlo")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def test_recommendations_ranked(client: TestClient) -> None:
    """/api/recommendations returns rank-ordered, score-descending results."""
    resp = client.get("/api/recommendations", params={"limit": 8})
    assert resp.status_code == 200
    recs = resp.json()
    assert isinstance(recs, list)
    assert 0 < len(recs) <= 8
    scores = [r["compositeScore"] for r in recs]
    assert scores == sorted(scores, reverse=True)
    assert [r["rank"] for r in recs] == list(range(1, len(recs) + 1))
    first = recs[0]
    assert first["recommendation"] in VALID_STANCES
    assert "expectedReturn1YPct" in first


def test_recommendations_filter(client: TestClient) -> None:
    """The ``assetClass`` filter narrows recommendations to one class."""
    resp = client.get(
        "/api/recommendations", params={"limit": 50, "assetClass": "etf"}
    )
    assert resp.status_code == 200
    recs = resp.json()
    assert recs
    assert all(r["asset"]["assetClass"] == "etf" for r in recs)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def test_list_strategies_at_least_18(client: TestClient) -> None:
    """/api/strategies lists at least 18 catalog entries."""
    resp = client.get("/api/strategies")
    assert resp.status_code == 200
    metas = resp.json()
    assert isinstance(metas, list)
    assert len(metas) >= 18
    m = metas[0]
    for key in ("id", "name", "category", "summary", "formula", "inputs", "references"):
        assert key in m


def test_strategy_rankings_known_id(client: TestClient) -> None:
    """/api/strategies/{id}/rankings ranks every asset by that model's score."""
    resp = client.get("/api/strategies/momentum/rankings", params={"limit": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategyId"] == "momentum"
    entries = body["entries"]
    assert entries
    assert len(entries) <= 10
    scores = [e["score"] for e in entries]
    assert scores == sorted(scores, reverse=True)
    assert all(e["stance"] in VALID_STANCES for e in entries)


def test_strategy_rankings_unknown_id_404(client: TestClient) -> None:
    """An unknown strategy id returns 404."""
    resp = client.get("/api/strategies/not-a-strategy/rankings")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


def test_portfolio_optimize_weights_sum_to_one(client: TestClient) -> None:
    """POST /api/portfolio/optimize returns weights that sum to ~1."""
    payload = {
        "symbols": ["AAPL", "MSFT", "JPM", "BTC"],
        "riskFreeRate": 0.04,
        "objective": "max_sharpe",
        "targetReturn": None,
    }
    resp = client.post("/api/portfolio/optimize", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    weights = body["weights"]
    assert {w["symbol"] for w in weights} == set(payload["symbols"])
    total = sum(w["weight"] for w in weights)
    assert total == pytest.approx(1.0, abs=1e-3)
    # Long-only: weights are non-negative (small numerical slack tolerated).
    assert all(w["weight"] >= -1e-6 for w in weights)
    assert math.isfinite(body["expectedReturn"])
    assert math.isfinite(body["volatility"])
    assert math.isfinite(body["sharpe"])
    assert body["efficientFrontier"]
    assert body["capitalMarketLine"]
    assert body["riskFreeRate"] == pytest.approx(0.04)


def test_portfolio_optimize_min_volatility(client: TestClient) -> None:
    """The min-volatility objective also returns a valid simplex of weights."""
    payload = {
        "symbols": ["SPY", "QQQ", "GLD"],
        "riskFreeRate": 0.04,
        "objective": "min_volatility",
        "targetReturn": None,
    }
    resp = client.post("/api/portfolio/optimize", json=payload)
    assert resp.status_code == 200
    total = sum(w["weight"] for w in resp.json()["weights"])
    assert total == pytest.approx(1.0, abs=1e-3)


def test_portfolio_optimize_unknown_symbol_404(client: TestClient) -> None:
    """An unknown symbol in the request returns 404."""
    payload = {
        "symbols": ["AAPL", "ZZZZ_NOPE"],
        "riskFreeRate": 0.04,
        "objective": "max_sharpe",
        "targetReturn": None,
    }
    resp = client.post("/api/portfolio/optimize", json=payload)
    assert resp.status_code == 404


def test_portfolio_optimize_empty_symbols_422(client: TestClient) -> None:
    """An empty symbol list returns 422."""
    payload = {
        "symbols": [],
        "riskFreeRate": 0.04,
        "objective": "max_sharpe",
        "targetReturn": None,
    }
    resp = client.post("/api/portfolio/optimize", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Market summary
# ---------------------------------------------------------------------------


def test_market_summary(client: TestClient) -> None:
    """/api/market/summary returns the dashboard overview with all sections."""
    resp = client.get("/api/market/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "asOf" in body
    breadth = body["breadth"]
    for key in ("advancers", "decliners", "unchanged"):
        assert key in breadth
    assert (
        breadth["advancers"] + breadth["decliners"] + breadth["unchanged"]
        == len(universe_symbols())
    )
    assert body["sectors"]
    assert body["indices"]
    assert isinstance(body["topGainers"], list)
    assert isinstance(body["topLosers"], list)


# ---------------------------------------------------------------------------
# WebSocket smoke test (section 6)
# ---------------------------------------------------------------------------


def test_websocket_snapshot_then_tick(client: TestClient) -> None:
    """The /ws socket sends a snapshot on connect, then a tick message."""
    with client.websocket_connect("/ws") as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"
        assert isinstance(snapshot["data"], list)
        assert snapshot["data"]  # full-universe snapshot is non-empty
        point = snapshot["data"][0]
        # PricePoint wire shape (camelCase): symbol, price, t (ms), changePct.
        for key in ("symbol", "price", "t", "changePct"):
            assert key in point

        # The next non-heartbeat message should be a tick from the loop. Tolerate
        # an interleaved heartbeat without failing.
        message = ws.receive_json()
        while message.get("type") == "heartbeat":
            message = ws.receive_json()
        assert message["type"] == "tick"
        assert isinstance(message["data"], list)


def test_websocket_subscribe_filters_ticks(client: TestClient) -> None:
    """Subscribing to a single symbol narrows subsequent tick payloads."""
    with client.websocket_connect("/ws") as ws:
        # Drain the initial snapshot.
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"

        ws.send_json({"action": "subscribe", "symbols": ["AAPL"]})

        # Collect a tick after the subscription takes effect.
        tick = ws.receive_json()
        for _ in range(5):
            if tick.get("type") == "tick":
                break
            tick = ws.receive_json()
        assert tick["type"] == "tick"
        symbols = {p["symbol"] for p in tick["data"]}
        # Once the subscription is applied, only AAPL should appear.
        assert symbols <= {"AAPL"}
