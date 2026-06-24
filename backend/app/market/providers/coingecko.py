"""CoinGecko crypto price backend.

Implements :class:`~app.market.providers.base.RealPriceBackend` against the
CoinGecko REST API (https://www.coingecko.com/en/api). It serves the **crypto**
symbols in our universe (``BTC``, ``ETH``, ``SOL``, ``ADA``, ``XRP``, ``DOGE``),
mapping each ticker to its CoinGecko coin id. Equities/ETFs are left to the
equity backends / simulator via the hybrid wrapper.

Endpoints used (read-only, all USD vs):
    * ``/simple/price`` — latest price.
    * ``/coins/{id}/market_chart`` — historical daily prices.
    * ``/coins/{id}/ohlc`` — OHLC candles (volume is not provided by this
      endpoint, so candle ``v`` is filled with 0.0 — documented honestly).

CoinGecko's public (demo) tier needs no key but is rate-limited; a key, if set,
is sent via the ``x-cg-demo-api-key`` header. On any failure
:class:`BackendError` is raised and the
:class:`~app.market.providers.hybrid.HybridProvider` falls back to the simulator.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from app.config import settings
from app.market.providers.base import BackendError, RealPriceBackend, TTLCache

__all__ = ["CoinGeckoBackend"]

_BASE_URL = "https://api.coingecko.com/api/v3"

#: Map our crypto tickers to CoinGecko coin ids.
_COIN_IDS: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "ADA": "cardano",
    "XRP": "ripple",
    "DOGE": "dogecoin",
}


class CoinGeckoBackend(RealPriceBackend):
    """Real crypto price backend backed by CoinGecko.

    Note:
        CoinGecko's public tier works **without** a key (rate-limited), so this
        backend reports itself :meth:`available` even when no key is set. A demo
        key, if present, raises the rate limit.

    Attributes:
        name: Backend name used in logs.
    """

    name = "coingecko"

    def __init__(self, api_key: str | None = None, ttl: float = 60.0) -> None:
        """Initialise the backend.

        Args:
            api_key: Optional CoinGecko demo key; defaults to
                ``settings.coingecko_api_key`` (public tier works without one).
            ttl: Cache lifetime in seconds for price/candle responses.
        """
        self._api_key = api_key if api_key is not None else settings.coingecko_api_key
        self._cache = TTLCache(ttl=ttl)

    def _headers(self) -> Optional[Dict[str, str]]:
        """Return the demo-key header when a key is configured, else ``None``."""
        if self._api_key:
            return {"x-cg-demo-api-key": self._api_key}
        return None

    def available(self) -> bool:
        """Return ``True`` — CoinGecko's public tier needs no API key."""
        return True

    def supports(self, symbol: str) -> bool:
        """Return whether ``symbol`` is a crypto CoinGecko can serve."""
        return symbol.strip().upper() in _COIN_IDS

    def _coin_id(self, symbol: str) -> str:
        """Map a ticker to its CoinGecko coin id or raise :class:`BackendError`."""
        coin = _COIN_IDS.get(symbol.strip().upper())
        if coin is None:
            raise BackendError(f"coingecko: unsupported symbol {symbol!r}")
        return coin

    def closes(self, symbol: str, days: int) -> np.ndarray:
        """Return up to ``days`` trailing daily closes (see base contract)."""
        coin = self._coin_id(symbol)
        span = max(int(days), 1)
        cache_key = ("closes", coin, span)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        data = self._get_json(
            f"{_BASE_URL}/coins/{coin}/market_chart",
            params={"vs_currency": "usd", "days": span, "interval": "daily"},
            headers=self._headers(),
        )
        prices = data.get("prices") if isinstance(data, dict) else None
        if not prices:
            raise BackendError(f"coingecko: no price history for {symbol!r}")
        try:
            closes = np.asarray([float(p[1]) for p in prices], dtype=np.float64)
        except (TypeError, ValueError, IndexError) as exc:
            raise BackendError(f"coingecko: malformed history for {symbol!r}: {exc}") from exc
        if closes.size > span:
            closes = closes[-span:]
        self._cache.set(cache_key, closes)
        return closes

    def candles(self, symbol: str, limit: int) -> List[Dict[str, float]]:
        """Return up to ``limit`` recent OHLC candles (see base contract).

        Note:
            CoinGecko's OHLC endpoint does not return volume, so each candle's
            ``v`` is ``0.0``. Volume-dependent strategies should treat this as
            unavailable for crypto on this backend.
        """
        coin = self._coin_id(symbol)
        n = max(int(limit), 1)
        # OHLC endpoint accepts a fixed set of day windows; pick the smallest
        # that covers the request to limit payload size.
        for window in (1, 7, 14, 30, 90, 180, 365):
            if window >= n:
                days_param: object = window
                break
        else:
            days_param = "max"
        cache_key = ("ohlc", coin, str(days_param))
        cached = self._cache.get(cache_key)
        if cached is None:
            data = self._get_json(
                f"{_BASE_URL}/coins/{coin}/ohlc",
                params={"vs_currency": "usd", "days": days_param},
                headers=self._headers(),
            )
            if not isinstance(data, list) or not data:
                raise BackendError(f"coingecko: no OHLC for {symbol!r}")
            try:
                cached = [
                    {
                        # CoinGecko OHLC timestamps are unix **ms** → seconds.
                        "t": int(int(row[0]) / 1000),
                        "o": float(row[1]),
                        "h": float(row[2]),
                        "l": float(row[3]),
                        "c": float(row[4]),
                        "v": 0.0,
                    }
                    for row in data
                ]
            except (TypeError, ValueError, IndexError) as exc:
                raise BackendError(f"coingecko: malformed OHLC for {symbol!r}: {exc}") from exc
            self._cache.set(cache_key, cached)
        return cached[-n:] if len(cached) > n else cached

    def latest(self, symbol: str) -> float:
        """Return the latest USD price via ``/simple/price`` (see base contract)."""
        coin = self._coin_id(symbol)
        cache_key = ("latest", coin)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        data = self._get_json(
            f"{_BASE_URL}/simple/price",
            params={"ids": coin, "vs_currencies": "usd"},
            headers=self._headers(),
        )
        try:
            price = float(data[coin]["usd"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"coingecko: malformed price for {symbol!r}: {exc}") from exc
        if not price > 0:
            raise BackendError(f"coingecko: non-positive price for {symbol!r}")
        self._cache.set(cache_key, price)
        return price
