"""Broker execution layer for GiffMeMoney (go-live, OPT-IN; ships safe).

The application places orders through the :class:`~app.broker.base.BrokerProvider`
interface; :func:`get_broker` selects the concrete backend by
:data:`app.config.settings.broker`.

SAFETY (non-negotiable, enforced in code):

* The default is :class:`~app.broker.simulated.SimulatedBroker` — paper fills at
  the market provider price, no real money, no network.
* :class:`~app.broker.alpaca.AlpacaBroker` defaults to Alpaca's **PAPER**
  (sandbox) endpoint. **LIVE trading is hard-gated**: it is only ever attempted
  when ``settings.broker == 'alpaca'`` AND ``settings.alpaca_live`` AND the live
  host is configured AND real keys are present AND ``settings.broker_ack`` (plus
  the caller's per-order ack) exactly equals the live phrase. Otherwise the
  broker stays paper / refuses live.
* Selection is **fail-safe**: an unknown ``broker`` key, missing keys, or *any*
  construction error falls back to the simulated broker (logged once) so the app
  always boots and default behaviour is never disrupted. This repo ships live OFF.
"""

from __future__ import annotations

import logging
import threading

from app.config import settings
from app.market.provider import get_provider

from app.broker.base import (
    LIVE_ACK_PHRASE,
    BrokerError,
    BrokerProvider,
    LiveTradingNotEnabledError,
)
from app.broker.simulated import SimulatedBroker

__all__ = [
    "BrokerProvider",
    "BrokerError",
    "LiveTradingNotEnabledError",
    "SimulatedBroker",
    "LIVE_ACK_PHRASE",
    "get_broker",
    "reset_broker_cache",
]

logger = logging.getLogger("app.broker")

_BROKER_LOCK = threading.Lock()
_BROKER_INSTANCE: BrokerProvider | None = None


def _build_simulated() -> BrokerProvider:
    """Build the default simulated (paper) broker over the market provider."""
    return SimulatedBroker(get_provider())


def _build_alpaca() -> BrokerProvider:
    """Build the Alpaca adapter (PAPER by default; live hard-gated).

    Imported lazily so the simulated default never imports httpx-backed code.

    Returns:
        An :class:`~app.broker.alpaca.AlpacaBroker`.

    Raises:
        BrokerError: If Alpaca is selected without API keys (the factory then
            falls back to the simulated broker in :func:`get_broker`).
    """
    from app.broker.alpaca import AlpacaBroker  # noqa: PLC0415 - lazy import

    broker = AlpacaBroker()
    if not (settings.alpaca_api_key and settings.alpaca_secret_key):
        raise BrokerError(
            "broker 'alpaca' selected but API keys are not configured"
        )
    return broker


def get_broker() -> BrokerProvider:
    """Return the process-wide :class:`~app.broker.base.BrokerProvider` singleton.

    The concrete broker is selected by :data:`app.config.settings.broker`
    (default ``'simulated'``). Selection is **fail-safe**: an unknown key,
    missing API keys, or any construction error falls back to the simulated
    paper broker (logged once at WARNING).

    Returns:
        The shared broker instance (a :class:`~app.broker.simulated.SimulatedBroker`
        by default).
    """
    global _BROKER_INSTANCE
    if _BROKER_INSTANCE is None:
        with _BROKER_LOCK:
            if _BROKER_INSTANCE is None:
                _BROKER_INSTANCE = _select_broker()
    return _BROKER_INSTANCE


def _select_broker() -> BrokerProvider:
    """Construct the configured broker, falling back to simulated on any error.

    Returns:
        The selected broker, or a :class:`~app.broker.simulated.SimulatedBroker`
        when the configured one is unknown / unconfigured / errored.
    """
    key = (settings.broker or "simulated").strip().lower()
    if key == "simulated":
        return _build_simulated()
    if key == "alpaca":
        try:
            broker = _build_alpaca()
        except Exception as exc:  # noqa: BLE001 - never crash; degrade safely
            logger.warning(
                "Broker 'alpaca' unavailable (%s); falling back to the "
                "simulated paper broker.",
                exc,
            )
            return _build_simulated()
        mode = "LIVE" if getattr(broker, "live_enabled", False) else "paper"
        logger.info("Broker: using Alpaca adapter (%s).", mode)
        return broker
    logger.warning(
        "Unknown broker %r; falling back to the simulated paper broker.", key
    )
    return _build_simulated()


def reset_broker_cache() -> None:
    """Drop the cached broker singleton (test helper).

    The next :func:`get_broker` call re-selects based on the current
    :data:`app.config.settings.broker`. Not used in production code paths.
    """
    global _BROKER_INSTANCE
    with _BROKER_LOCK:
        _BROKER_INSTANCE = None
