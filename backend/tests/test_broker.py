"""Tests for the broker execution layer (go-live, OPT-IN; ships live OFF).

These cover the safety contract from ``docs/GOLIVE.md`` §2 / §6:

* the default ``get_broker()`` is the :class:`~app.broker.simulated.SimulatedBroker`
  (``is_paper`` True, ``live_enabled`` False) and needs no keys/network;
* a paper order fills immediately and is reflected in ``get_positions()``;
* the LIVE hard-gate **refuses** without the full acknowledgement — and, most
  importantly, **no live path is reachable by default** (default settings can
  never produce a live-enabled Alpaca broker, and the simulated broker never
  places a real order regardless of any ``broker_ack``).

SAFETY: the default ``broker`` is ``'simulated'``. No test enables live; the one
test that constructs a *live-enabled* Alpaca instance does so only to prove the
order is still refused without the exact per-order ack, and it never issues a
network request (the refusal happens before any HTTP I/O). All ``settings``
mutations are restored and the broker cache is reset.
"""

from __future__ import annotations

from typing import Iterator
from unittest.mock import patch

import pytest

from app.broker import (
    LIVE_ACK_PHRASE,
    SimulatedBroker,
    get_broker,
    reset_broker_cache,
)
from app.broker.alpaca import AlpacaBroker, LIVE_BASE_URL, PAPER_BASE_URL
from app.broker.base import BrokerError, LiveTradingNotEnabledError
from app.config import settings
from app.market.provider import SimulatedProvider
from app.schemas import BrokerAccount, BrokerOrder, BrokerPosition, BrokerStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_broker() -> Iterator[None]:
    """Snapshot/restore broker-related settings and reset the broker cache.

    Guarantees a test that flips ``settings.broker`` (or the Alpaca live knobs)
    cannot leak the live configuration into the rest of the session.
    """
    saved = {
        "broker": settings.broker,
        "alpaca_api_key": settings.alpaca_api_key,
        "alpaca_secret_key": settings.alpaca_secret_key,
        "alpaca_base_url": settings.alpaca_base_url,
        "alpaca_live": settings.alpaca_live,
        "broker_ack": settings.broker_ack,
    }
    reset_broker_cache()
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(settings, key, value)
        reset_broker_cache()


@pytest.fixture
def sim_broker() -> SimulatedBroker:
    """A fresh simulated paper broker over the deterministic provider."""
    return SimulatedBroker(SimulatedProvider())


# ---------------------------------------------------------------------------
# Default broker is the simulated paper broker
# ---------------------------------------------------------------------------


def test_default_broker_is_simulated(restore_broker: None) -> None:
    """With default settings, ``get_broker`` returns a paper SimulatedBroker."""
    settings.broker = "simulated"
    reset_broker_cache()
    broker = get_broker()
    assert isinstance(broker, SimulatedBroker)
    assert broker.is_paper is True


def test_default_broker_status_is_paper_and_not_live(restore_broker: None) -> None:
    """The default broker advertises simulated/paper, live disabled."""
    settings.broker = "simulated"
    reset_broker_cache()
    status = get_broker().status()
    assert isinstance(status, BrokerStatus)
    assert status.broker == "simulated"
    assert status.mode == "simulated"
    assert status.paper is True
    assert status.live_enabled is False
    assert status.disclaimer  # disclaimer always present


def test_default_broker_account_is_paper(sim_broker: SimulatedBroker) -> None:
    """The simulated account reports paper mode and finite balances."""
    account = sim_broker.get_account()
    assert isinstance(account, BrokerAccount)
    assert account.paper is True
    assert account.mode == "simulated"
    assert account.cash > 0.0
    assert account.equity == pytest.approx(account.cash, abs=1e-6)


# ---------------------------------------------------------------------------
# A paper order fills and appears in positions
# ---------------------------------------------------------------------------


def test_paper_buy_appears_in_positions(sim_broker: SimulatedBroker) -> None:
    """A simulated buy fills immediately and shows up as a paper position."""
    order = sim_broker.submit_order("AAPL", "buy", notional=1_000.0)

    assert isinstance(order, BrokerOrder)
    assert order.paper is True
    assert order.status == "filled"
    assert order.side == "buy"
    assert order.symbol == "AAPL"
    assert order.filled_qty > 0.0
    assert order.filled_avg_price > 0.0

    positions = sim_broker.get_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert isinstance(pos, BrokerPosition)
    assert pos.symbol == "AAPL"
    assert pos.qty == pytest.approx(order.filled_qty, abs=1e-8)
    assert pos.paper is True
    assert pos.cost_basis == pytest.approx(1_000.0, rel=1e-6)


def test_paper_buy_reduces_cash(sim_broker: SimulatedBroker) -> None:
    """Buying reduces paper cash by the filled notional."""
    before = sim_broker.get_account().cash
    sim_broker.submit_order("MSFT", "buy", notional=500.0)
    after = sim_broker.get_account().cash
    assert after == pytest.approx(before - 500.0, abs=1e-2)


def test_paper_sell_closes_position(sim_broker: SimulatedBroker) -> None:
    """Selling the full position removes it from ``get_positions``."""
    buy = sim_broker.submit_order("AAPL", "buy", notional=750.0)
    sell = sim_broker.submit_order("AAPL", "sell", qty=buy.filled_qty)
    assert sell.status == "filled"
    assert sell.side == "sell"
    assert sim_broker.get_positions() == []


def test_paper_order_recorded_in_list_orders(sim_broker: SimulatedBroker) -> None:
    """Filled paper orders are recorded and listed newest-first."""
    sim_broker.submit_order("AAPL", "buy", notional=100.0)
    sim_broker.submit_order("MSFT", "buy", notional=100.0)
    orders = sim_broker.list_orders()
    assert len(orders) == 2
    # Newest first.
    assert orders[0].symbol == "MSFT"
    assert all(o.paper is True for o in orders)


def test_simulated_broker_ignores_broker_ack(sim_broker: SimulatedBroker) -> None:
    """Even passing the live ack to the SIM broker never makes it live.

    The simulated broker is always paper: a ``broker_ack`` is accepted but
    ignored, and the resulting order is still a paper fill.
    """
    order = sim_broker.submit_order(
        "AAPL", "buy", notional=100.0, broker_ack=LIVE_ACK_PHRASE
    )
    assert order.paper is True
    assert sim_broker.is_paper is True


def test_simulated_broker_rejects_bad_input(sim_broker: SimulatedBroker) -> None:
    """Invalid sizing / unknown symbol raise a clear :class:`BrokerError`."""
    with pytest.raises(BrokerError):
        sim_broker.submit_order("AAPL", "buy")  # neither notional nor qty
    with pytest.raises(BrokerError):
        sim_broker.submit_order("NOPE_NOT_A_SYMBOL", "buy", notional=100.0)
    with pytest.raises(BrokerError):
        sim_broker.submit_order("AAPL", "sell", qty=1.0)  # nothing held


# ---------------------------------------------------------------------------
# The LIVE gate refuses — no live path reachable by default
# ---------------------------------------------------------------------------


def test_alpaca_paper_host_is_never_live(restore_broker: None) -> None:
    """Alpaca on the PAPER host stays paper even with keys + ack + live flag."""
    settings.broker = "alpaca"
    broker = AlpacaBroker(
        api_key="key",
        secret_key="secret",
        base_url=PAPER_BASE_URL,
        live_flag=True,
        broker_ack=LIVE_ACK_PHRASE,
    )
    assert broker.live_enabled is False
    assert broker.is_paper is True
    assert broker.status().paper is True


def test_alpaca_not_selected_is_never_live(restore_broker: None) -> None:
    """Live host + keys + ack but ``broker != 'alpaca'`` is still gated off."""
    settings.broker = "simulated"  # the default
    broker = AlpacaBroker(
        api_key="key",
        secret_key="secret",
        base_url=LIVE_BASE_URL,
        live_flag=True,
        broker_ack=LIVE_ACK_PHRASE,
    )
    assert broker.live_enabled is False
    assert broker.is_paper is True


def test_alpaca_without_keys_is_never_live(restore_broker: None) -> None:
    """Missing API keys keep Alpaca on the paper path, never live."""
    settings.broker = "alpaca"
    broker = AlpacaBroker(
        api_key=None,
        secret_key=None,
        base_url=LIVE_BASE_URL,
        live_flag=True,
        broker_ack=LIVE_ACK_PHRASE,
    )
    assert broker.live_enabled is False
    assert broker.is_paper is True


def test_alpaca_wrong_ack_is_never_live(restore_broker: None) -> None:
    """An incorrect ``broker_ack`` phrase keeps Alpaca paper-only."""
    settings.broker = "alpaca"
    broker = AlpacaBroker(
        api_key="key",
        secret_key="secret",
        base_url=LIVE_BASE_URL,
        live_flag=True,
        broker_ack="i understand",  # not the exact phrase
    )
    assert broker.live_enabled is False
    assert broker.is_paper is True


def test_live_enabled_broker_refuses_order_without_per_order_ack(
    restore_broker: None,
) -> None:
    """Even a fully live-enabled broker refuses an order lacking the exact ack.

    This is the strongest gate: the broker instance satisfies the full hard-gate
    (broker=='alpaca', live host, keys, live flag, configured ack), yet a
    ``submit_order`` without the matching per-order ``broker_ack`` raises
    :class:`LiveTradingNotEnabledError` (HTTP 403) and issues NO network request.
    """
    settings.broker = "alpaca"
    broker = AlpacaBroker(
        api_key="key",
        secret_key="secret",
        base_url=LIVE_BASE_URL,
        live_flag=True,
        broker_ack=LIVE_ACK_PHRASE,
    )
    assert broker.live_enabled is True  # the gate is configured open...

    # ...but the per-order ack is still required, and it is checked BEFORE any
    # HTTP call. Patch the request layer to prove it is never reached.
    with patch.object(AlpacaBroker, "_request") as request:
        with pytest.raises(LiveTradingNotEnabledError):
            broker.submit_order("AAPL", "buy", notional=100.0, broker_ack=None)
        with pytest.raises(LiveTradingNotEnabledError):
            broker.submit_order(
                "AAPL", "buy", notional=100.0, broker_ack="not the phrase"
            )
    request.assert_not_called()


def test_get_broker_alpaca_without_keys_falls_back_to_simulated(
    restore_broker: None,
) -> None:
    """Selecting Alpaca with NO keys fails safe to the simulated paper broker."""
    settings.broker = "alpaca"
    settings.alpaca_api_key = None
    settings.alpaca_secret_key = None
    reset_broker_cache()
    broker = get_broker()
    assert isinstance(broker, SimulatedBroker)
    assert broker.is_paper is True


def test_no_live_broker_reachable_by_default() -> None:
    """The default-configured broker can never be live (the headline guarantee).

    Using the *current* process defaults (broker='simulated'), the selected
    broker is a paper SimulatedBroker, ``is_paper`` is True, and its status
    reports ``live_enabled`` False. This asserts the shipped configuration places
    no real orders.
    """
    reset_broker_cache()
    broker = get_broker()
    assert isinstance(broker, SimulatedBroker)
    assert broker.is_paper is True
    assert broker.status().live_enabled is False
    # And the defaults themselves are the safe path.
    assert settings.broker == "simulated"
    assert settings.alpaca_live is False
    assert settings.alpaca_base_url == PAPER_BASE_URL
