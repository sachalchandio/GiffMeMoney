"""The default simulated (paper) broker — no real money, ever.

:class:`SimulatedBroker` is the safe default selected by
:func:`app.broker.get_broker`. It fills market orders **immediately** at the
current market provider's latest price and tracks the resulting cash, positions
and order history in process memory (thread-safe, mirroring the in-memory stores
in :mod:`app.invest.store` / :mod:`app.auth.store`). State resets on restart.

``is_paper`` is always ``True`` and every DTO it returns carries ``paper: True``
plus the standard disclaimer, so a caller can never mistake a simulated fill for
a real one. No network is ever touched.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.market.provider import MarketDataProvider
from app.schemas import (
    BROKER_DISCLAIMER,
    BrokerAccount,
    BrokerOrder,
    BrokerOrderSide,
    BrokerPosition,
    BrokerStatus,
)

from app.broker.base import BrokerError, BrokerProvider

__all__ = ["SimulatedBroker"]

#: Units below this magnitude are treated as a fully closed position. Set a hair
#: above the 8-dp rounding used on the wire so selling a client-supplied
#: (rounded) ``qty`` sweeps any sub-unit dust instead of leaving a phantom
#: position.
_QTY_EPS: float = 1e-7

#: Starting paper cash for a fresh simulated broker account ($100k).
_DEFAULT_PAPER_CASH: float = 100_000.0


def _now_ms() -> int:
    """Return the current unix time in milliseconds."""
    return int(time.time() * 1000)


@dataclass
class _Holding:
    """Mutable accounting record for one simulated paper position.

    Attributes:
        symbol: Asset ticker (canonical upper-case).
        qty: Units currently held (``>= 0``).
        cost_basis: Total dollars invested in the still-held units.
    """

    symbol: str
    qty: float = 0.0
    cost_basis: float = 0.0


@dataclass
class _Account:
    """All mutable state for the single simulated paper account.

    Attributes:
        cash: Uninvested paper cash.
        holdings: Open positions keyed by canonical (upper-case) symbol.
        orders: Recorded orders, appended in chronological order.
    """

    cash: float = _DEFAULT_PAPER_CASH
    holdings: Dict[str, _Holding] = field(default_factory=dict)
    orders: List[BrokerOrder] = field(default_factory=list)


class SimulatedBroker(BrokerProvider):
    """A self-contained paper broker that fills at the provider price.

    Requires no network and no API keys. Market orders fill instantly at the
    market provider's latest price; positions are marked to that same price.

    Args:
        provider: A :class:`~app.market.provider.MarketDataProvider` used to
            price fills and to mark positions to market.
        starting_cash: Initial paper cash balance.
    """

    is_paper = True

    def __init__(
        self,
        provider: MarketDataProvider,
        starting_cash: float = _DEFAULT_PAPER_CASH,
    ) -> None:
        """Initialise an empty paper account with a re-entrant lock."""
        self._provider = provider
        self._lock = threading.RLock()
        self._account = _Account(cash=float(starting_cash))
        self._account_id = "sim-paper"

    # ------------------------------------------------------------------
    # Pricing helpers
    # ------------------------------------------------------------------

    def _price(self, symbol: str) -> float:
        """Return a strictly-positive latest price for ``symbol``.

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            The latest price as a positive float.

        Raises:
            BrokerError: If the symbol is unknown or has no valid price.
        """
        try:
            price = float(self._provider.latest_price(symbol))
        except KeyError as exc:
            raise BrokerError(f"Unknown symbol {symbol!r}.") from exc
        if not math.isfinite(price) or price <= 0.0:
            raise BrokerError(f"No valid market price available for {symbol!r}.")
        return price

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def status(self) -> BrokerStatus:
        """Return the simulated-broker status (always paper, always connected)."""
        return BrokerStatus(
            broker="simulated",
            mode="simulated",
            paper=True,
            connected=True,
            live_enabled=False,
            base_url=None,
            message="Simulated paper broker (fills at the market provider price).",
            disclaimer=BROKER_DISCLAIMER,
        )

    def _invested_value(self) -> float:
        """Return the marked-to-market value of all open holdings.

        Must be called while holding the lock.
        """
        total = 0.0
        for holding in self._account.holdings.values():
            try:
                price = self._price(holding.symbol)
            except BrokerError:
                price = 0.0
            total += float(holding.qty) * price
        return total if math.isfinite(total) else 0.0

    def get_account(self) -> BrokerAccount:
        """Return the paper account summary (cash, equity, buying power)."""
        with self._lock:
            cash = float(self._account.cash)
            if not math.isfinite(cash):
                cash = 0.0
            equity = cash + self._invested_value()
            return BrokerAccount(
                account_id=self._account_id,
                cash=round(cash, 2),
                equity=round(equity, 2),
                buying_power=round(max(0.0, cash), 2),
                currency="USD",
                mode="simulated",
                paper=True,
                disclaimer=BROKER_DISCLAIMER,
            )

    def get_positions(self) -> List[BrokerPosition]:
        """Return all open paper positions marked to the latest price."""
        with self._lock:
            out: List[BrokerPosition] = []
            for holding in self._account.holdings.values():
                qty = float(holding.qty)
                if qty <= _QTY_EPS:
                    continue
                try:
                    price = self._price(holding.symbol)
                except BrokerError:
                    price = 0.0
                cost_basis = float(holding.cost_basis)
                market_value = qty * price
                avg_entry = cost_basis / qty if qty > _QTY_EPS else 0.0
                unrealized = market_value - cost_basis
                unrealized_pct = (
                    unrealized / cost_basis * 100.0 if cost_basis > 0.0 else 0.0
                )
                out.append(
                    BrokerPosition(
                        symbol=holding.symbol,
                        qty=round(qty, 8),
                        avg_entry_price=round(_finite(avg_entry), 6),
                        current_price=round(_finite(price), 6),
                        market_value=round(_finite(market_value), 2),
                        cost_basis=round(_finite(cost_basis), 2),
                        unrealized_pnl=round(_finite(unrealized), 2),
                        unrealized_pnl_pct=round(_finite(unrealized_pct), 4),
                        paper=True,
                    )
                )
            out.sort(key=lambda p: p.market_value, reverse=True)
            return out

    def list_orders(self) -> List[BrokerOrder]:
        """Return the recorded paper orders, newest first."""
        with self._lock:
            return list(reversed(self._account.orders))

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

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
        """Fill a paper market order immediately at the provider price.

        ``broker_ack`` is intentionally ignored — the simulated broker is always
        paper and never places a real order. Sizing is by ``notional`` (dollars)
        when present, else by ``qty`` (units).

        Args:
            symbol: Asset ticker (case-insensitive).
            side: ``'buy'`` or ``'sell'``.
            notional: Dollar amount to trade (sizes by dollars).
            qty: Units to trade (used when ``notional`` is omitted).
            type: Order type (only ``'market'`` is supported).
            broker_ack: Ignored (simulated broker is always paper).

        Returns:
            A filled :class:`~app.schemas.BrokerOrder` (``paper=True``).

        Raises:
            BrokerError: On invalid input, an unknown symbol, insufficient cash,
                or selling more than is held.
        """
        sym = str(symbol).strip().upper()
        if not sym:
            raise BrokerError("Order symbol must not be empty.")
        if side not in ("buy", "sell"):
            raise BrokerError("Order side must be 'buy' or 'sell'.")
        if str(type).strip().lower() != "market":
            raise BrokerError("Only 'market' orders are supported by the paper broker.")

        with self._lock:
            price = self._price(sym)

            # Resolve order quantity from notional or qty.
            order_qty = self._resolve_qty(notional, qty, price)

            if side == "buy":
                cost = order_qty * price
                if cost > float(self._account.cash) + 1e-9:
                    raise BrokerError(
                        f"Insufficient paper cash: order needs ${cost:,.2f} "
                        f"but only ${float(self._account.cash):,.2f} is available."
                    )
                holding = self._account.holdings.get(sym)
                if holding is None:
                    holding = _Holding(symbol=sym)
                    self._account.holdings[sym] = holding
                holding.qty = float(holding.qty) + order_qty
                holding.cost_basis = float(holding.cost_basis) + cost
                self._account.cash = float(self._account.cash) - cost
            else:  # sell
                holding = self._account.holdings.get(sym)
                held = float(holding.qty) if holding is not None else 0.0
                if holding is None or held <= _QTY_EPS:
                    raise BrokerError(f"No open position to sell for {sym!r}.")
                order_qty = min(order_qty, held)
                if order_qty <= _QTY_EPS:
                    raise BrokerError("Sell quantity is too small to execute.")
                fraction = order_qty / held if held > 0.0 else 1.0
                fraction = min(1.0, max(0.0, fraction))
                proceeds = order_qty * price
                holding.qty = held - order_qty
                holding.cost_basis = float(holding.cost_basis) * (1.0 - fraction)
                self._account.cash = float(self._account.cash) + proceeds
                if holding.qty <= _QTY_EPS:
                    del self._account.holdings[sym]

            order = BrokerOrder(
                id=f"sim_{uuid.uuid4().hex}",
                symbol=sym,
                side=side,
                type="market",
                qty=round(order_qty, 8),
                notional=round(float(notional), 2) if notional is not None else None,
                filled_qty=round(order_qty, 8),
                filled_avg_price=round(price, 6),
                status="filled",
                created_at=_now_ms(),
                paper=True,
                disclaimer=BROKER_DISCLAIMER,
            )
            self._account.orders.append(order)
            return order

    def cancel_order(self, order_id: str) -> BrokerOrder:
        """Cancel an order by id.

        Simulated market orders fill instantly, so there is nothing to cancel;
        a known order is returned unchanged (already ``filled``).

        Args:
            order_id: The opaque order id.

        Returns:
            The :class:`~app.schemas.BrokerOrder` for ``order_id``.

        Raises:
            BrokerError: If the order id is unknown.
        """
        with self._lock:
            for order in self._account.orders:
                if order.id == order_id:
                    return order
        raise BrokerError(f"Unknown order id {order_id!r}.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_qty(
        notional: Optional[float], qty: Optional[float], price: float
    ) -> float:
        """Resolve a strictly-positive order quantity from notional or qty.

        ``notional`` (dollars) wins when both are supplied.

        Args:
            notional: Dollar amount to trade, or ``None``.
            qty: Units to trade, or ``None``.
            price: The fill price per unit (``> 0``).

        Returns:
            A strictly-positive order quantity in units.

        Raises:
            BrokerError: If neither sizing is a positive finite number.
        """
        if notional is not None:
            try:
                dollars = float(notional)
            except (TypeError, ValueError):
                raise BrokerError("Order notional must be a number.") from None
            if not math.isfinite(dollars) or dollars <= 0.0:
                raise BrokerError("Order notional must be greater than zero.")
            return dollars / price
        if qty is not None:
            try:
                units = float(qty)
            except (TypeError, ValueError):
                raise BrokerError("Order qty must be a number.") from None
            if not math.isfinite(units) or units <= 0.0:
                raise BrokerError("Order qty must be greater than zero.")
            return units
        raise BrokerError("An order must specify a positive 'notional' or 'qty'.")


def _finite(value: float, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, else ``default``.

    Args:
        value: Candidate number.
        default: Substitute for NaN / +-inf / non-numeric input.

    Returns:
        A finite float.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default
