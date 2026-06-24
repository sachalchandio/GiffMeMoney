"""Broker execution abstraction (see docs/GOLIVE.md §2).

The application places orders exclusively through the :class:`BrokerProvider`
interface, so the default in-memory paper broker and a real REST broker (Alpaca)
are interchangeable behind one contract. A process-wide singleton is selected by
:data:`app.config.settings.broker` via :func:`app.broker.get_broker`.

SAFETY (non-negotiable, enforced in code): the broker ships **simulated** (paper
fills against the market provider price; no real money). The real adapter
defaults to Alpaca's PAPER (sandbox) endpoint. **LIVE trading is hard-gated**:
it is only ever attempted when ``settings.broker == 'alpaca'`` AND
``settings.alpaca_live`` AND the order's ``broker_ack`` exactly equals
``"I understand this places real orders"`` AND real keys are present. Otherwise
any order path refuses with a clear error. This repo ships with live OFF.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from app.schemas import (
    BrokerAccount,
    BrokerOrder,
    BrokerOrderSide,
    BrokerPosition,
    BrokerStatus,
)

__all__ = [
    "BrokerError",
    "LiveTradingNotEnabledError",
    "LIVE_ACK_PHRASE",
    "BrokerProvider",
]

#: The exact acknowledgement phrase a caller must supply (and that must match
#: ``settings.broker_ack``) before any live order is ever attempted. Anything
#: other than this exact string keeps the broker on the paper path.
LIVE_ACK_PHRASE: str = "I understand this places real orders"


class BrokerError(RuntimeError):
    """Raised by a broker when an order or query cannot be completed.

    The API layer maps this to a clear client error (HTTP 400/502) rather than
    leaking a raw network/parse error.
    """


class LiveTradingNotEnabledError(BrokerError):
    """Raised when a LIVE order is requested without the full hard-gate.

    The API layer maps this to **HTTP 403** with a clear message. Live trading
    requires ``settings.broker == 'alpaca'`` AND ``settings.alpaca_live`` AND a
    matching ``broker_ack`` AND real keys; any missing piece refuses here so no
    real order is ever placed by accident.
    """


class BrokerProvider(ABC):
    """Abstract interface every broker backend must implement.

    All return types are the frozen wire DTOs from :mod:`app.schemas`. Every
    response carries ``paper: True/False`` so a caller can never mistake a
    simulated/paper fill for a real one. Implementations must be safe to call
    concurrently from request threads.

    Attributes:
        is_paper: ``True`` whenever orders are simulated or routed to a paper
            sandbox; ``False`` only when real live trading is fully enabled. The
            default simulated broker is always paper.
    """

    #: Whether this broker is in paper/simulated mode (never real money). The
    #: live adapter only flips this to ``False`` when the full hard-gate passes.
    is_paper: bool = True

    @abstractmethod
    def status(self) -> BrokerStatus:
        """Return the broker connectivity / mode snapshot.

        Returns:
            A :class:`~app.schemas.BrokerStatus` describing the backend, mode,
            paper flag, connectivity and whether live is enabled.
        """
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> BrokerAccount:
        """Return the broker account summary (cash, equity, buying power).

        Returns:
            A :class:`~app.schemas.BrokerAccount`.

        Raises:
            BrokerError: If the account cannot be fetched.
        """
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        """Return all open positions marked to the latest price.

        Returns:
            A list of :class:`~app.schemas.BrokerPosition`.

        Raises:
            BrokerError: If positions cannot be fetched.
        """
        raise NotImplementedError

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        side: BrokerOrderSide,
        *,
        notional: Optional[float] = None,
        qty: Optional[float] = None,
        type: str = "market",
        broker_ack: Optional[str] = None,
    ) -> BrokerOrder:
        """Place an order and return the resulting :class:`~app.schemas.BrokerOrder`.

        Exactly one of ``notional`` or ``qty`` must size the order (``notional``
        wins when both are given). The simulated broker fills immediately at the
        market provider price; the real adapter submits to Alpaca (PAPER by
        default).

        Args:
            symbol: Asset ticker (case-insensitive).
            side: ``'buy'`` or ``'sell'``.
            notional: Dollar amount to trade (sizes by dollars).
            qty: Units to trade (used when ``notional`` is omitted).
            type: Order type (only ``'market'`` is supported).
            broker_ack: Live-trading acknowledgement. Ignored in paper modes;
                required (and must match :data:`LIVE_ACK_PHRASE`) to attempt a
                live order.

        Returns:
            The placed :class:`~app.schemas.BrokerOrder`.

        Raises:
            BrokerError: On invalid input or an execution failure (HTTP 400/502).
            LiveTradingNotEnabledError: If a live order is requested without the
                full hard-gate (HTTP 403).
        """
        raise NotImplementedError

    @abstractmethod
    def list_orders(self) -> List[BrokerOrder]:
        """Return the recorded/known orders, newest first.

        Returns:
            A list of :class:`~app.schemas.BrokerOrder`.

        Raises:
            BrokerError: If orders cannot be fetched.
        """
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> BrokerOrder:
        """Cancel an order by id and return its updated state.

        Args:
            order_id: The opaque order id to cancel.

        Returns:
            The updated :class:`~app.schemas.BrokerOrder`.

        Raises:
            BrokerError: If the order is unknown or cannot be canceled.
        """
        raise NotImplementedError
