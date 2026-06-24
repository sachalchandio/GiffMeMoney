"""``/api/broker`` — the broker execution layer (go-live, OPT-IN; ships safe).

This router is the HTTP front for the pluggable broker (see ``docs/GOLIVE.md``
§2). It is a thin, defensive adapter over the
:class:`~app.broker.base.BrokerProvider` selected by
:func:`app.broker.get_broker` (default
:class:`~app.broker.simulated.SimulatedBroker`).

SAFETY (non-negotiable, enforced in code):

* The broker ships **simulated** (paper fills at the market provider price; no
  real money). The Alpaca adapter defaults to Alpaca's PAPER endpoint.
* ``POST /order`` places a **PAPER** order in every default configuration. A
  LIVE order is **refused with HTTP 403** unless the full hard-gate is
  satisfied (``broker == 'alpaca'`` AND ``alpaca_live`` AND real keys AND the
  exact ``broker_ack`` both in config and on the request).
* Every response carries ``paper: true/false`` and the standard disclaimer.

Error mapping: :class:`~app.broker.base.LiveTradingNotEnabledError` → HTTP 403;
:class:`~app.broker.base.BrokerError` → HTTP 400. Reads that fail upstream
surface as HTTP 502.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException

from app.broker import get_broker
from app.broker.base import BrokerError, LiveTradingNotEnabledError
from app.schemas import (
    BrokerAccount,
    BrokerOrder,
    BrokerOrderRequest,
    BrokerPosition,
    BrokerStatus,
)

__all__ = ["router"]

router = APIRouter(prefix="/api/broker", tags=["broker"])


@router.get(
    "/status",
    response_model=BrokerStatus,
    summary="Broker mode, paper flag and connectivity",
    description=(
        "Return the active broker backend, execution mode (`simulated` / "
        "`paper` / `live`), whether it is paper, whether it is connected, and "
        "whether the full live-trading hard-gate is satisfied.\n\n"
        "This app ships with the **simulated** paper broker and live trading "
        "**OFF**; `paper` is `true` and `liveEnabled` is `false` by default. "
        "Every payload carries the standard simulation disclaimer."
    ),
    responses={200: {"description": "The broker mode / connectivity snapshot."}},
)
def status() -> BrokerStatus:
    """Return the broker connectivity / mode snapshot.

    Returns:
        A :class:`~app.schemas.BrokerStatus` (paper + disclaimer included).
    """
    return get_broker().status()


@router.get(
    "/account",
    response_model=BrokerAccount,
    summary="Broker account: cash, equity and buying power",
    description=(
        "Return the broker account summary (cash, equity, buying power) for the "
        "active broker. For the default simulated broker this is a paper account "
        "starting with simulated cash; no real money is involved.\n\n"
        "**Status codes**\n"
        "- `200` — the account summary.\n"
        "- `502` — the upstream broker could not be reached."
    ),
    responses={
        200: {"description": "The broker account summary."},
        502: {"description": "Upstream broker error."},
    },
)
def account() -> BrokerAccount:
    """Return the broker account summary.

    Returns:
        A :class:`~app.schemas.BrokerAccount` (paper + disclaimer included).

    Raises:
        HTTPException: ``502`` if the upstream broker call fails.
    """
    try:
        return get_broker().get_account()
    except BrokerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get(
    "/positions",
    response_model=list[BrokerPosition],
    summary="Open broker positions marked to market",
    description=(
        "Return every open position held by the active broker, marked to the "
        "latest price. For the simulated broker these are paper positions filled "
        "at the market provider price.\n\n"
        "**Status codes**\n"
        "- `200` — the open positions.\n"
        "- `502` — the upstream broker could not be reached."
    ),
    responses={
        200: {"description": "The open broker positions."},
        502: {"description": "Upstream broker error."},
    },
)
def positions() -> list[BrokerPosition]:
    """Return open broker positions marked to the latest price.

    Returns:
        A list of :class:`~app.schemas.BrokerPosition`.

    Raises:
        HTTPException: ``502`` if the upstream broker call fails.
    """
    try:
        return get_broker().get_positions()
    except BrokerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get(
    "/orders",
    response_model=list[BrokerOrder],
    summary="Recorded broker orders (newest first)",
    description=(
        "Return the broker's known orders, newest first. For the simulated "
        "broker, market orders fill instantly at the provider price; each entry "
        "carries `paper: true` and the standard disclaimer.\n\n"
        "**Status codes**\n"
        "- `200` — the recorded orders.\n"
        "- `502` — the upstream broker could not be reached."
    ),
    responses={
        200: {"description": "The recorded broker orders."},
        502: {"description": "Upstream broker error."},
    },
)
def orders() -> list[BrokerOrder]:
    """Return the broker's recorded orders, newest first.

    Returns:
        A list of :class:`~app.schemas.BrokerOrder`.

    Raises:
        HTTPException: ``502`` if the upstream broker call fails.
    """
    try:
        return get_broker().list_orders()
    except BrokerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post(
    "/order",
    response_model=BrokerOrder,
    summary="Place a (PAPER) market order",
    description=(
        "Place a market order through the active broker. Size the order by "
        "`notional` (dollars) or `qty` (units) — `notional` wins if both are "
        "given.\n\n"
        "In every default configuration this places a **PAPER** order (no real "
        "money). A LIVE order is **refused with HTTP 403** unless the full "
        "hard-gate is satisfied: `broker == 'alpaca'`, `alpaca_live` enabled, "
        "real keys configured, and the exact `brokerAck` "
        '(`"I understand this places real orders"`) supplied **both** in server '
        "config and on this request. This app ships with live trading OFF.\n\n"
        "**Status codes**\n"
        "- `200` — order placed (paper unless live is fully enabled).\n"
        "- `400` — invalid order (bad symbol/side, non-positive size, no funds).\n"
        "- `403` — a live order was requested without the full acknowledgement."
    ),
    responses={
        200: {"description": "Order placed; paper flag + disclaimer included."},
        400: {"description": "Invalid order parameters."},
        403: {"description": "Live order requested without the full acknowledgement."},
    },
)
def place_order(body: BrokerOrderRequest = Body(...)) -> BrokerOrder:
    """Place a market order (paper unless live is fully, deliberately enabled).

    Args:
        body: The :class:`~app.schemas.BrokerOrderRequest` (symbol, side, and a
            ``notional`` or ``qty``; optional ``brokerAck`` for live).

    Returns:
        The placed :class:`~app.schemas.BrokerOrder`.

    Raises:
        HTTPException: ``403`` if a live order is requested without the full
            acknowledgement; ``400`` for invalid order parameters.
    """
    try:
        return get_broker().submit_order(
            body.symbol,
            body.side,
            notional=body.notional,
            qty=body.qty,
            type=body.type,
            broker_ack=body.broker_ack,
        )
    except LiveTradingNotEnabledError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except BrokerError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
