"""WebSocket connection management and the live price-tick broadcaster.

This module implements the server side of the ``/ws`` protocol (section 6 of the
contract):

* on connect the server sends a ``{"type":"snapshot","data":PricePoint[]}`` for
  the whole universe;
* clients may ``subscribe`` / ``unsubscribe`` to a subset of symbols (default =
  whole universe);
* every ``settings.tick_interval_ms`` the server pushes
  ``{"type":"tick","data":PricePoint[]}`` containing only each connection's
  subscribed symbols;
* every ~15 s the server pushes ``{"type":"heartbeat","t":<unix ms>}``.

Live prices are kept in an in-memory map seeded from the provider's latest
closes and nudged each tick by a small Gaussian random walk (per-process RNG —
live ticks need not be reproducible). ``change_pct`` on each :class:`PricePoint`
is measured against the day's anchor price (the provider's latest close), so it
mirrors an intraday "% change on the day" figure.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import time
from typing import Dict, Iterable, List, Optional, Set

from app.config import settings
from app.market.provider import MarketDataProvider
from app.schemas import PricePoint

__all__ = [
    "ConnectionManager",
    "LivePriceBook",
    "price_tick_loop",
]

# Heartbeat cadence in seconds (contract: ~15 s).
_HEARTBEAT_SECONDS: float = 15.0

# Per-tick random-walk volatility (fraction of price) for the live nudge. Small
# so the live feed looks like realistic intraday drift, not noise.
_TICK_SIGMA: float = 0.0008

# Clamp how far the live price may drift from the day's anchor (±25%), so a long
# session can't wander into absurd territory.
_MAX_DRIFT: float = 0.25


def _now_ms() -> int:
    """Return the current unix time in **milliseconds**."""
    return int(time.time() * 1000)


class LivePriceBook:
    """Mutable in-memory map of live prices keyed by symbol.

    Each symbol tracks a current ``price`` and an ``anchor`` (the day's opening /
    latest-close reference) used to compute ``change_pct``. The book is seeded
    from the provider and then evolved by :meth:`nudge` on every tick.

    Attributes:
        prices: Symbol → current live price.
        anchors: Symbol → reference price for percent-change.
    """

    def __init__(self, provider: MarketDataProvider) -> None:
        """Seed the book from the provider's latest closes.

        Args:
            provider: The market-data provider to read initial prices from.
        """
        self._provider = provider
        self._rng = random.Random()
        self.prices: Dict[str, float] = {}
        self.anchors: Dict[str, float] = {}
        self._seed()

    def _seed(self) -> None:
        """Populate ``prices`` and ``anchors`` from the provider snapshot."""
        for asset in self._provider.list_assets():
            price = float(asset.price)
            self.prices[asset.symbol] = price
            self.anchors[asset.symbol] = price if price > 0 else 1.0

    def symbols(self) -> List[str]:
        """Return all symbols currently tracked, in insertion order."""
        return list(self.prices.keys())

    def nudge(self) -> None:
        """Advance every live price by one random-walk step.

        Formula:
            P <- clamp(P * (1 + N(0, _TICK_SIGMA)), anchor*(1-_MAX_DRIFT),
                                                      anchor*(1+_MAX_DRIFT))

        The drift is clamped around the day's anchor so prices stay realistic.
        """
        for sym, price in self.prices.items():
            anchor = self.anchors.get(sym, price) or price
            shock = self._rng.gauss(0.0, _TICK_SIGMA)
            new_price = price * (1.0 + shock)
            lo = anchor * (1.0 - _MAX_DRIFT)
            hi = anchor * (1.0 + _MAX_DRIFT)
            new_price = max(lo, min(hi, new_price))
            if not (new_price > 0) or new_price != new_price:  # NaN-safe
                new_price = anchor
            self.prices[sym] = new_price

    def point(self, symbol: str) -> Optional[PricePoint]:
        """Build a :class:`PricePoint` for one symbol from current state.

        Formula:
            changePct = (price / anchor - 1) * 100

        Args:
            symbol: The symbol to snapshot.

        Returns:
            A :class:`PricePoint` (``t`` in unix ms), or ``None`` if the symbol
            is not tracked.
        """
        price = self.prices.get(symbol)
        if price is None:
            return None
        anchor = self.anchors.get(symbol, price) or price
        change_pct = (price / anchor - 1.0) * 100.0 if anchor > 0 else 0.0
        if change_pct != change_pct:  # NaN guard
            change_pct = 0.0
        return PricePoint(
            symbol=symbol,
            price=round(float(price), 6),
            t=_now_ms(),
            change_pct=round(float(change_pct), 4),
        )

    def points(self, symbols: Optional[Iterable[str]] = None) -> List[PricePoint]:
        """Build :class:`PricePoint` snapshots for a set of symbols.

        Args:
            symbols: Symbols to include; ``None`` means every tracked symbol.

        Returns:
            A list of :class:`PricePoint` objects (unknown symbols skipped).
        """
        syms = list(symbols) if symbols is not None else self.symbols()
        out: List[PricePoint] = []
        for sym in syms:
            p = self.point(sym)
            if p is not None:
                out.append(p)
        return out


class ConnectionManager:
    """Tracks active WebSocket connections and their per-connection subscriptions.

    Each connection defaults to receiving the entire universe; clients may narrow
    this with ``subscribe`` / ``unsubscribe`` actions. ``None`` as a connection's
    subscription set is the sentinel for "all symbols".

    Attributes:
        active: The set of currently-connected WebSockets.
    """

    def __init__(self) -> None:
        """Initialise an empty manager with an asyncio lock for mutation."""
        self.active: Set[object] = set()
        # ws -> set of subscribed symbols, or None for "all".
        self._subscriptions: Dict[object, Optional[Set[str]]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, ws: object) -> None:
        """Accept a WebSocket and register it with a default (all) subscription.

        Args:
            ws: A Starlette/FastAPI ``WebSocket`` (accepts via ``ws.accept()``).
        """
        accept = getattr(ws, "accept", None)
        if accept is not None:
            await accept()
        async with self._lock:
            self.active.add(ws)
            self._subscriptions[ws] = None  # None => subscribed to everything

    async def disconnect(self, ws: object) -> None:
        """Remove a WebSocket and its subscription state.

        Safe to call multiple times for the same connection.

        Args:
            ws: The WebSocket to drop.
        """
        async with self._lock:
            self.active.discard(ws)
            self._subscriptions.pop(ws, None)

    async def set_subscription(self, ws: object, symbols: Optional[Iterable[str]]) -> None:
        """Replace a connection's subscription set.

        Args:
            ws: The connection to update.
            symbols: An iterable of symbols to subscribe to, or ``None`` to
                subscribe to the whole universe. Symbols are upper-cased.
        """
        async with self._lock:
            if ws not in self.active:
                return
            if symbols is None:
                self._subscriptions[ws] = None
            else:
                self._subscriptions[ws] = {s.strip().upper() for s in symbols if s}

    async def subscribe(self, ws: object, symbols: Iterable[str]) -> None:
        """Add symbols to a connection's subscription.

        If the connection was subscribed to "all" (``None``) this narrows it to
        exactly the provided set.

        Args:
            ws: The connection to update.
            symbols: Symbols to add.
        """
        async with self._lock:
            if ws not in self.active:
                return
            current = self._subscriptions.get(ws)
            add = {s.strip().upper() for s in symbols if s}
            if current is None:
                self._subscriptions[ws] = add
            else:
                current.update(add)
                self._subscriptions[ws] = current

    async def unsubscribe(self, ws: object, symbols: Iterable[str]) -> None:
        """Remove symbols from a connection's subscription.

        If the connection was subscribed to "all", it is first materialised — but
        since we don't know the universe here, "all minus X" is represented by
        leaving it as "all" and relying on the empty-removal being a no-op; to
        keep semantics simple, unsubscribing from an "all" subscription is
        treated as a no-op (clients should subscribe to an explicit set instead).

        Args:
            ws: The connection to update.
            symbols: Symbols to remove.
        """
        async with self._lock:
            if ws not in self.active:
                return
            current = self._subscriptions.get(ws)
            if current is None:
                return
            for s in symbols:
                current.discard(s.strip().upper())
            self._subscriptions[ws] = current

    def subscription_for(self, ws: object) -> Optional[Set[str]]:
        """Return the subscription set for a connection (``None`` = all).

        Args:
            ws: The connection to query.

        Returns:
            The subscribed symbol set, or ``None`` if subscribed to everything.
        """
        return self._subscriptions.get(ws)

    async def send_json(self, ws: object, message: dict) -> bool:
        """Send a JSON message to one connection, dropping it on failure.

        Args:
            ws: The target connection.
            message: A JSON-serialisable dict.

        Returns:
            ``True`` if the send succeeded, ``False`` if the connection errored
            (and was therefore removed).
        """
        try:
            await ws.send_json(message)  # type: ignore[attr-defined]
            return True
        except Exception:
            await self.disconnect(ws)
            return False

    async def broadcast(self, message: dict) -> None:
        """Broadcast a message to every active connection.

        The message is sent verbatim to all connections regardless of their
        subscriptions (used for heartbeats and the full snapshot). Connections
        that error are dropped.

        Args:
            message: A JSON-serialisable dict.
        """
        async with self._lock:
            targets = list(self.active)
        for ws in targets:
            await self.send_json(ws, message)

    async def broadcast_points(self, book: "LivePriceBook", message_type: str) -> None:
        """Broadcast per-connection-filtered price points.

        Each connection receives only the :class:`PricePoint` items for the
        symbols it is subscribed to (or all symbols when subscribed to "all").

        Args:
            book: The live price book to read current points from.
            message_type: The ``type`` field of the emitted message
                (``"tick"`` or ``"snapshot"``).
        """
        async with self._lock:
            targets = [(ws, self._subscriptions.get(ws)) for ws in self.active]
        for ws, subs in targets:
            if subs is None:
                points = book.points()
            else:
                points = book.points(subs)
            payload = {
                "type": message_type,
                "data": [p.model_dump(by_alias=True) for p in points],
            }
            await self.send_json(ws, payload)


async def price_tick_loop(
    manager: ConnectionManager,
    provider: MarketDataProvider,
    stop_event: asyncio.Event,
) -> None:
    """Continuously nudge live prices and broadcast ticks until stopped.

    On each iteration (every ``settings.tick_interval_ms``):

    * the in-memory :class:`LivePriceBook` is advanced one random-walk step;
    * a ``{"type":"tick","data":PricePoint[]}`` is sent to each connection,
      filtered to its subscribed symbols;
    * roughly every 15 s a ``{"type":"heartbeat","t":<unix ms>}`` is also sent.

    The loop exits promptly when ``stop_event`` is set (used on app shutdown).

    Args:
        manager: The :class:`ConnectionManager` to broadcast through.
        provider: The provider used to seed the live price book.
        stop_event: An :class:`asyncio.Event`; setting it stops the loop.
    """
    book = LivePriceBook(provider)
    interval = max(float(settings.tick_interval_ms) / 1000.0, 0.05)
    last_heartbeat = time.monotonic()

    while not stop_event.is_set():
        # Advance live prices and broadcast a tick to subscribers.
        book.nudge()
        try:
            await manager.broadcast_points(book, "tick")
        except Exception:
            # Never let a broadcast error kill the loop.
            pass

        now = time.monotonic()
        if now - last_heartbeat >= _HEARTBEAT_SECONDS:
            last_heartbeat = now
            with contextlib.suppress(Exception):
                await manager.broadcast({"type": "heartbeat", "t": _now_ms()})

        # Sleep for the tick interval, but wake immediately if asked to stop.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
