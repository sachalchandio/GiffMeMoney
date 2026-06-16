"""Shared pytest fixtures for the GiffMeMoney backend test suite.

Provides:

* a session-scoped FastAPI :class:`~fastapi.testclient.TestClient` bound to the
  real ASGI app (``app.main:app``) — used by the API / WebSocket smoke tests;
* small deterministic helpers and seeds reused across the quant tests
  (a constant-excess return series, a known-drawdown price path, the simulated
  market data provider, and a couple of universe symbols).

The TestClient is entered as a context manager so the app's ``lifespan``
(startup/shutdown of the background price-tick loop) runs exactly once per
session and is cleanly torn down afterwards.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.market.provider import MarketDataProvider, get_provider
from app.market.universe import UNIVERSE, symbols


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """Yield a TestClient bound to the real app with its lifespan active.

    Entering the client as a context manager triggers the FastAPI ``lifespan``
    handler (which starts the background tick loop) on setup and stops it on
    teardown, so the WebSocket snapshot/tick protocol is exercisable.

    Yields:
        A ready-to-use :class:`fastapi.testclient.TestClient`.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def provider() -> MarketDataProvider:
    """Return the process-wide simulated market-data provider singleton.

    Returns:
        The :class:`~app.market.provider.MarketDataProvider` chosen by settings
        (the deterministic :class:`~app.market.provider.SimulatedProvider`).
    """
    return get_provider()


@pytest.fixture(scope="session")
def universe_symbols() -> list[str]:
    """Return every symbol in the seed universe, in declaration order.

    Returns:
        A list of ticker strings (e.g. ``["AAPL", "MSFT", ...]``).
    """
    return symbols()


@pytest.fixture(scope="session")
def sample_symbols() -> list[str]:
    """Return a small, stable subset of real universe symbols for portfolio tests.

    Returns:
        A list of three known equity symbols present in the universe.
    """
    return ["AAPL", "MSFT", "JPM"]


@pytest.fixture
def constant_excess_returns() -> np.ndarray:
    """A return series with constant *excess* return over a zero risk-free rate.

    With every daily return equal to a fixed positive constant ``c`` and
    ``rf_daily = 0``, the excess series ``r - rf`` is constant, so its standard
    deviation is zero. This is the canonical input for verifying the Sharpe /
    Sortino zero-volatility guards (they must return ``0.0`` rather than
    diverging).

    Returns:
        A length-252 ``float64`` array of the constant ``0.001`` daily return.
    """
    return np.full(252, 0.001, dtype=np.float64)


@pytest.fixture
def known_drawdown_prices() -> np.ndarray:
    """A price path whose maximum drawdown is exactly -50%.

    The path rises 100 -> 120 (new peak), then falls 120 -> 60 (a 50% decline
    from the peak), then partially recovers to 90. The peak-to-trough drawdown is
    ``60 / 120 - 1 = -0.5``.

    Returns:
        A ``float64`` array ``[100, 110, 120, 90, 60, 75, 90]``.
    """
    return np.array([100.0, 110.0, 120.0, 90.0, 60.0, 75.0, 90.0], dtype=np.float64)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded numpy random generator for reproducible synthetic series.

    Returns:
        A :class:`numpy.random.Generator` seeded with a fixed value.
    """
    return np.random.default_rng(12345)
