"""Binance crypto price backend.

Implements :class:`~app.market.providers.base.RealPriceBackend` against Binance's
public market-data REST API (https://api.binance.com). It serves the **crypto**
symbols in our universe by mapping each ticker to its ``...USDT`` trading pair
(``BTC`` → ``BTCUSDT``, etc.). Equities/ETFs are left to the equity backends /
simulator via the hybrid wrapper.

Endpoints used (read-only, no key required for public market data):
    * ``/api/v3/klines`` — daily OHLCV candles + close history.
    * ``/api/v3/ticker/price`` — latest price.

Binance public market data needs no API key (the configured key, if any, is not
required for these read-only endpoints), so this backend is always
:meth:`available`. On any failure :class:`BackendError` is raised and the
:class:`~app.market.providers.hybrid.HybridProvider` falls back to the simulator.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from app.config import settings
from app.market.providers.base import BackendError, RealPriceBackend, TTLCache

__all__ = ["BinanceBackend"]

_BASE_URL = "https://api.binance.com"

#: Map our crypto tickers to Binance USDT trading pairs.
_PAIRS: Dict[str, str] = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "ADA": "ADAUSDT",
    "XRP": "XRPUSDT",
    "DOGE": "DOGEUSDT",
}

#: Binance caps klines at 1000 rows per request.
_MAX_KLINES = 1000


class BinanceBackend(RealPriceBackend):
    """Real crypto price backend backed by Binance public market data.

    Note:
        Read-only market-data endpoints require **no API key**, so this backend
        reports itself :meth:`available` unconditionally.

    Attributes:
        name: Backend name used in logs.
    """

    name = "binance"

    def __init__(self, api_key: str | None = None, ttl: float = 60.0) -> None:
        """Initialise the backend.

        Args:
            api_key: Optional Binance key (not needed for public market data);
                defaults to ``settings.binance_api_key``.
            ttl: Cache lifetime in seconds for price/candle responses.
        """
        self._api_key = api_key if api_key is not None else settings.binance_api_key
        self._cache = TTLCache(ttl=ttl)

    def available(self) -> bool:
        """Return ``True`` — Binance public market data needs no API key."""
        return True

    def supports(self, symbol: str) -> bool:
        """Return whether ``symbol`` is a crypto Binance can serve."""
        return symbol.strip().upper() in _PAIRS

    def _pair(self, symbol: str) -> str:
        """Map a ticker to its Binance pair or raise :class:`BackendError`."""
        pair = _PAIRS.get(symbol.strip().upper())
        if pair is None:
            raise BackendError(f"binance: unsupported symbol {symbol!r}")
        return pair

    def _klines_raw(self, symbol: str, limit: int) -> List[Dict[str, float]]:
        """Fetch and normalise daily klines, with caching.

        Args:
            symbol: Asset ticker.
            limit: Number of trailing daily candles to request (capped at 1000).

        Returns:
            A list of ``{'t','o','h','l','c','v'}`` dicts oldest → newest
            (``t`` in unix **seconds**).

        Raises:
            BackendError: On HTTP/parse failure or an empty result.
        """
        pair = self._pair(symbol)
        n = min(max(int(limit), 1), _MAX_KLINES)
        cache_key = ("klines", pair, n)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        data = self._get_json(
            f"{_BASE_URL}/api/v3/klines",
            params={"symbol": pair, "interval": "1d", "limit": n},
        )
        if not isinstance(data, list) or not data:
            raise BackendError(f"binance: no klines for {symbol!r}")
        try:
            # Kline row: [openTime, open, high, low, close, volume, closeTime, ...]
            out = [
                {
                    "t": int(int(row[0]) / 1000),
                    "o": float(row[1]),
                    "h": float(row[2]),
                    "l": float(row[3]),
                    "c": float(row[4]),
                    "v": float(row[5]),
                }
                for row in data
            ]
        except (TypeError, ValueError, IndexError) as exc:
            raise BackendError(f"binance: malformed klines for {symbol!r}: {exc}") from exc
        self._cache.set(cache_key, out)
        return out

    def closes(self, symbol: str, days: int) -> np.ndarray:
        """Return up to ``days`` trailing daily closes (see base contract)."""
        rows = self._klines_raw(symbol, days)
        closes = np.asarray([r["c"] for r in rows], dtype=np.float64)
        if closes.size > days:
            closes = closes[-days:]
        return closes

    def candles(self, symbol: str, limit: int) -> List[Dict[str, float]]:
        """Return up to ``limit`` recent OHLCV candles (see base contract)."""
        rows = self._klines_raw(symbol, limit)
        return rows[-limit:] if len(rows) > limit else rows

    def latest(self, symbol: str) -> float:
        """Return the latest price via ``/ticker/price`` (see base contract)."""
        pair = self._pair(symbol)
        cache_key = ("latest", pair)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        data = self._get_json(
            f"{_BASE_URL}/api/v3/ticker/price", params={"symbol": pair}
        )
        try:
            price = float(data["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"binance: malformed price for {symbol!r}: {exc}") from exc
        if not price > 0:
            raise BackendError(f"binance: non-positive price for {symbol!r}")
        self._cache.set(cache_key, price)
        return price
