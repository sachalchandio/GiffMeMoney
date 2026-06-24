"""Polygon.io equities/ETF price backend.

Implements :class:`~app.market.providers.base.RealPriceBackend` against Polygon's
REST API (https://polygon.io). It serves **equities and ETFs** whose Polygon
tickers match our symbols 1:1 (e.g. ``AAPL``, ``SPY``); crypto is left to the
crypto backends / simulator via the hybrid wrapper.

Endpoints used (read-only):
    * ``/v2/aggs/ticker/{sym}/range/1/day/{from}/{to}`` — daily OHLCV aggregates.
    * ``/v2/aggs/ticker/{sym}/prev`` — previous-day close (used for ``latest``).

The API key is sent as the ``apiKey`` query parameter. Honest limitation:
Polygon's free tier is delayed/limited; on any failure :class:`BackendError` is
raised and the :class:`~app.market.providers.hybrid.HybridProvider` falls back to
the simulator. Fundamentals/factors are served from the simulator (documented in
the hybrid wrapper), not here.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from app.config import settings
from app.market.providers.base import BackendError, RealPriceBackend, TTLCache

__all__ = ["PolygonBackend"]

_BASE_URL = "https://api.polygon.io"

_EQUITY_ETF_SYMBOLS = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "BAC", "V", "JNJ", "PFE",
    "XOM", "CVX", "AMZN", "KO", "CAT", "SPY", "QQQ", "VTI", "GLD",
}


class PolygonBackend(RealPriceBackend):
    """Real equity/ETF price backend backed by Polygon.io.

    Attributes:
        name: Backend name used in logs.
    """

    name = "polygon"

    def __init__(self, api_key: str | None = None, ttl: float = 60.0) -> None:
        """Initialise the backend.

        Args:
            api_key: Polygon API key; defaults to ``settings.polygon_api_key``.
            ttl: Cache lifetime in seconds for price/aggregate responses.
        """
        self._api_key = api_key if api_key is not None else settings.polygon_api_key
        self._cache = TTLCache(ttl=ttl)

    def available(self) -> bool:
        """Return whether a Polygon API key is configured."""
        return bool(self._api_key)

    def supports(self, symbol: str) -> bool:
        """Return whether ``symbol`` is an equity/ETF Polygon can serve."""
        return symbol.strip().upper() in _EQUITY_ETF_SYMBOLS

    def _aggs_raw(self, symbol: str, days: int) -> List[Dict[str, float]]:
        """Fetch and normalise daily aggregates, with caching.

        Args:
            symbol: Asset ticker.
            days: Number of trailing calendar days of history to request.

        Returns:
            A list of ``{'t','o','h','l','c','v'}`` dicts oldest → newest
            (``t`` in unix **seconds**).

        Raises:
            BackendError: On HTTP/parse failure or an empty result.
        """
        sym = symbol.strip().upper()
        span_days = max(int(days) + 10, 10)
        cache_key = ("aggs", sym, span_days)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        now = int(time.time())
        frm = time.strftime("%Y-%m-%d", time.gmtime(now - span_days * 86400))
        to = time.strftime("%Y-%m-%d", time.gmtime(now))
        url = f"{_BASE_URL}/v2/aggs/ticker/{sym}/range/1/day/{frm}/{to}"
        data = self._get_json(
            url, params={"adjusted": "true", "sort": "asc", "apiKey": self._api_key}
        )
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            raise BackendError(f"polygon: no aggregate data for {sym!r}")
        try:
            out = [
                {
                    # Polygon timestamps are unix **milliseconds** → seconds.
                    "t": int(int(bar["t"]) / 1000),
                    "o": float(bar["o"]),
                    "h": float(bar["h"]),
                    "l": float(bar["l"]),
                    "c": float(bar["c"]),
                    "v": float(bar.get("v", 0.0)),
                }
                for bar in results
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"polygon: malformed aggregate for {sym!r}: {exc}") from exc
        self._cache.set(cache_key, out)
        return out

    def closes(self, symbol: str, days: int) -> np.ndarray:
        """Return up to ``days`` trailing daily closes (see base contract)."""
        bars = self._aggs_raw(symbol, days)
        closes = np.asarray([b["c"] for b in bars], dtype=np.float64)
        if closes.size > days:
            closes = closes[-days:]
        return closes

    def candles(self, symbol: str, limit: int) -> List[Dict[str, float]]:
        """Return up to ``limit`` recent OHLCV candles (see base contract)."""
        bars = self._aggs_raw(symbol, limit)
        return bars[-limit:] if len(bars) > limit else bars

    def latest(self, symbol: str) -> float:
        """Return the previous-day close via ``/prev`` (see base contract)."""
        sym = symbol.strip().upper()
        cache_key = ("latest", sym)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        url = f"{_BASE_URL}/v2/aggs/ticker/{sym}/prev"
        data = self._get_json(url, params={"adjusted": "true", "apiKey": self._api_key})
        results = data.get("results") if isinstance(data, dict) else None
        if not results:
            raise BackendError(f"polygon: no prev close for {sym!r}")
        try:
            price = float(results[0]["c"])
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise BackendError(f"polygon: malformed prev close for {sym!r}: {exc}") from exc
        if not price > 0:
            raise BackendError(f"polygon: non-positive prev close for {sym!r}")
        self._cache.set(cache_key, price)
        return price
