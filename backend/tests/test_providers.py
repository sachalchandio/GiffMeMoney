"""Tests for the live market-data provider layer (go-live, OPT-IN).

These cover the three contract guarantees from ``docs/GOLIVE.md`` §1 / §6:

* ``get_provider()`` with **no API keys** falls back to the deterministic
  :class:`~app.market.provider.SimulatedProvider` and touches **no network**;
* a **mocked** httpx response from a real backend maps cleanly onto the frozen
  :class:`~app.schemas.Asset` / :class:`~app.schemas.Candle` DTOs;
* a :class:`~app.market.providers.hybrid.HybridProvider` delegates the
  ``factor_history`` / ``fundamentals`` (and ``market_history``) calls — which
  real feeds don't publish cleanly — to the simulator.

No test here hits a real network: the only "real" backend exercised is mocked at
its HTTP boundary (``RealPriceBackend._get_json``), and the fallback paths assert
that the simulated provider is selected when no key is configured.

SAFETY: the default ``provider`` is ``'simulated'``; these tests never change the
process-wide default permanently (every ``settings`` mutation is restored and the
provider singleton cache is reset).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional
from unittest.mock import patch

import numpy as np
import pytest

from app.config import settings
from app.market.provider import (
    SimulatedProvider,
    get_provider,
    reset_provider_cache,
)
from app.market.providers.base import BackendError, RealPriceBackend
from app.market.providers.finnhub import FinnhubBackend
from app.market.providers.hybrid import HybridProvider
from app.schemas import Asset, Candle


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_provider() -> Iterator[None]:
    """Snapshot/restore ``settings.provider`` + keys and the provider cache.

    Ensures a test that flips the configured provider (or sets a fake key) never
    leaks state into the rest of the session, and that the next ``get_provider``
    re-selects from the (restored) default.
    """
    saved = {
        "provider": settings.provider,
        "finnhub_api_key": settings.finnhub_api_key,
        "polygon_api_key": settings.polygon_api_key,
        "coingecko_api_key": settings.coingecko_api_key,
        "binance_api_key": settings.binance_api_key,
    }
    reset_provider_cache()
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(settings, key, value)
        reset_provider_cache()


def _fake_finnhub_get_json(
    self: RealPriceBackend,
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 6.0,
) -> Any:
    """Stand in for ``RealPriceBackend._get_json`` with canned Finnhub payloads.

    Returns a deterministic quote for ``/quote`` and a 5-bar daily candle series
    for ``/stock/candle`` so the mapped :class:`Asset` / :class:`Candle` values
    are predictable. No network is involved.
    """
    if url.endswith("/quote"):
        # ``c`` = current price, ``pc`` = previous close.
        return {"c": 192.5, "pc": 190.0}
    if url.endswith("/stock/candle"):
        n = 5
        now = int(time.time())
        ts = [now - 86400 * (n - 1 - i) for i in range(n)]
        return {
            "s": "ok",
            "t": ts,
            "o": [189.0, 190.0, 191.0, 191.5, 192.0],
            "h": [191.0, 192.0, 193.0, 193.0, 193.5],
            "l": [188.0, 189.0, 190.0, 190.5, 191.0],
            "c": [190.0, 191.0, 192.0, 192.2, 192.5],
            "v": [1_000.0, 1_100.0, 1_200.0, 1_050.0, 1_300.0],
        }
    raise BackendError(f"unexpected url in test stub: {url}")


# ---------------------------------------------------------------------------
# get_provider() fallback — no key -> simulated, no network
# ---------------------------------------------------------------------------


def test_get_provider_defaults_to_simulated(restore_provider: None) -> None:
    """With the default settings, ``get_provider`` returns the simulator."""
    settings.provider = "simulated"
    reset_provider_cache()
    provider = get_provider()
    assert isinstance(provider, SimulatedProvider)


def test_get_provider_real_without_key_falls_back_to_simulated(
    restore_provider: None,
) -> None:
    """Selecting a real provider with NO API key degrades to the simulator.

    Crucially this must happen **without any network call**: the backend reports
    itself unavailable (missing key) and the factory raises, so ``get_provider``
    never constructs a hybrid and never issues an HTTP request. We assert no
    ``httpx.Client`` is ever instantiated during selection.
    """
    settings.provider = "finnhub"
    settings.finnhub_api_key = None  # no credentials
    reset_provider_cache()

    with patch("httpx.Client") as http_client:
        provider = get_provider()

    assert isinstance(provider, SimulatedProvider)
    http_client.assert_not_called()


def test_get_provider_unknown_key_falls_back_to_simulated(
    restore_provider: None,
) -> None:
    """An unknown ``provider`` key is fail-safe: it returns the simulator."""
    settings.provider = "definitely-not-a-real-provider"
    reset_provider_cache()
    provider = get_provider()
    assert isinstance(provider, SimulatedProvider)


def test_simulated_provider_is_network_free() -> None:
    """A direct ``SimulatedProvider`` issues no HTTP calls for core reads."""
    provider = SimulatedProvider()
    with patch("httpx.Client") as http_client:
        asset = provider.get_asset("AAPL")
        candles = provider.get_candles("AAPL", 10)
        price = provider.latest_price("AAPL")
    http_client.assert_not_called()
    assert isinstance(asset, Asset)
    assert all(isinstance(c, Candle) for c in candles)
    assert price > 0.0


# ---------------------------------------------------------------------------
# Mocked httpx response maps to Asset / Candle
# ---------------------------------------------------------------------------


def test_mocked_backend_maps_to_asset() -> None:
    """A mocked Finnhub quote/candle maps onto a well-formed :class:`Asset`."""
    with patch.object(FinnhubBackend, "_get_json", _fake_finnhub_get_json):
        backend = FinnhubBackend(api_key="test-key")
        assert backend.available() is True
        assert backend.supports("AAPL") is True

        hybrid = HybridProvider(backend, SimulatedProvider())
        asset = hybrid.get_asset("AAPL")

    assert isinstance(asset, Asset)
    assert asset.symbol == "AAPL"
    # Price comes straight from the mocked quote's ``c`` field.
    assert asset.price == pytest.approx(192.5)
    # change24hPct derived from the last two mocked closes (192.2 -> 192.5).
    expected_change = (192.5 / 192.2 - 1.0) * 100.0
    assert asset.change24h_pct == pytest.approx(expected_change, abs=1e-4)
    # Identity is taken from the universe seed, not invented by the backend.
    assert asset.name
    assert asset.currency


def test_mocked_backend_maps_to_candles() -> None:
    """A mocked Finnhub candle series maps onto :class:`Candle` DTOs."""
    with patch.object(FinnhubBackend, "_get_json", _fake_finnhub_get_json):
        backend = FinnhubBackend(api_key="test-key")
        hybrid = HybridProvider(backend, SimulatedProvider())
        candles = hybrid.get_candles("AAPL", 5)

    assert len(candles) == 5
    assert all(isinstance(c, Candle) for c in candles)
    last = candles[-1]
    assert last.c == pytest.approx(192.5)
    assert last.o == pytest.approx(192.0)
    assert last.h == pytest.approx(193.5)
    assert last.l == pytest.approx(191.0)
    assert last.v == pytest.approx(1_300.0)
    # Timestamps are unix seconds, strictly increasing oldest -> newest.
    ts = [c.t for c in candles]
    assert ts == sorted(ts)


def test_mocked_backend_latest_price() -> None:
    """The hybrid latest price uses the mocked real quote when available."""
    with patch.object(FinnhubBackend, "_get_json", _fake_finnhub_get_json):
        backend = FinnhubBackend(api_key="test-key")
        hybrid = HybridProvider(backend, SimulatedProvider())
        price = hybrid.latest_price("AAPL")
    assert price == pytest.approx(192.5)


# ---------------------------------------------------------------------------
# Hybrid delegates factor/fundamental data to the simulator
# ---------------------------------------------------------------------------


def test_hybrid_factor_history_delegates_to_simulator() -> None:
    """``factor_history`` always comes from the simulator, not the backend."""
    sim = SimulatedProvider()
    backend = FinnhubBackend(api_key="test-key")
    hybrid = HybridProvider(backend, sim)

    with patch.object(FinnhubBackend, "_get_json", _fake_finnhub_get_json):
        factors = hybrid.factor_history(120)

    expected = sim.factor_history(120)
    assert set(factors.keys()) == {"mkt", "smb", "hml", "rf"}
    for key in ("mkt", "smb", "hml", "rf"):
        np.testing.assert_array_equal(factors[key], expected[key])


def test_hybrid_fundamentals_delegates_to_simulator() -> None:
    """``fundamentals`` always comes from the simulator (real feeds vary)."""
    sim = SimulatedProvider()
    backend = FinnhubBackend(api_key="test-key")
    hybrid = HybridProvider(backend, sim)

    with patch.object(FinnhubBackend, "_get_json", _fake_finnhub_get_json):
        fundamentals = hybrid.fundamentals("AAPL")

    assert fundamentals == sim.fundamentals("AAPL")


def test_hybrid_market_history_delegates_to_simulator() -> None:
    """``market_history`` (the synthetic index) is always simulated."""
    sim = SimulatedProvider()
    backend = FinnhubBackend(api_key="test-key")
    hybrid = HybridProvider(backend, sim)
    np.testing.assert_array_equal(
        hybrid.market_history(200), sim.market_history(200)
    )


def test_hybrid_falls_back_to_sim_when_backend_errors() -> None:
    """A backend error on a price call never crashes — the sim answers instead."""

    class _BrokenBackend(RealPriceBackend):
        name = "broken"

        def available(self) -> bool:
            return True

        def supports(self, symbol: str) -> bool:
            return True

        def closes(self, symbol: str, days: int) -> np.ndarray:
            raise BackendError("boom")

        def candles(self, symbol: str, limit: int) -> List[Dict[str, float]]:
            raise BackendError("boom")

        def latest(self, symbol: str) -> float:
            raise BackendError("boom")

    sim = SimulatedProvider()
    hybrid = HybridProvider(_BrokenBackend(), sim)

    # Every price-driven call degrades to the deterministic simulated value.
    assert hybrid.latest_price("AAPL") == pytest.approx(sim.latest_price("AAPL"))
    assert hybrid.get_asset("AAPL").price == pytest.approx(
        sim.get_asset("AAPL").price
    )
    assert len(hybrid.get_candles("AAPL", 10)) == len(sim.get_candles("AAPL", 10))


def test_hybrid_uses_sim_for_unsupported_symbol() -> None:
    """A symbol the backend doesn't cover is served from the simulator."""
    sim = SimulatedProvider()
    # Finnhub only supports equities/ETFs; a crypto symbol falls through.
    backend = FinnhubBackend(api_key="test-key")
    assert backend.supports("BTC") is False
    hybrid = HybridProvider(backend, sim)
    # No HTTP should be attempted for an unsupported symbol.
    with patch.object(FinnhubBackend, "_get_json") as get_json:
        price = hybrid.latest_price("BTC")
    get_json.assert_not_called()
    assert price == pytest.approx(sim.latest_price("BTC"))
