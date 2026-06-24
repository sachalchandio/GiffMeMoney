"""Finnhub equities price backend.

Implements :class:`~app.market.providers.base.RealPriceBackend` against Finnhub's
REST API (https://finnhub.io). It serves **equities and ETFs** (Finnhub tickers
match our symbols 1:1, e.g. ``AAPL``, ``SPY``); crypto in our universe is left to
the crypto backends / simulator via the hybrid wrapper.

Endpoints used (read-only):
    * ``/quote`` — latest price (``c`` = current, ``pc`` = previous close).
    * ``/stock/candle`` — daily OHLCV history (resolution ``D``).

Honest limitations: Finnhub's free tier rate-limits candles and may not return
history for every ticker; on any failure :class:`BackendError` is raised and the
:class:`~app.market.providers.hybrid.HybridProvider` falls back to the simulator.
Finnhub does expose fundamentals, but the quant engine needs the full uniform
:class:`app.market.universe.Fundamentals` record, so fundamentals are served
from the simulator (documented in the hybrid wrapper), not here.
"""

from __future__ import annotations

import time
from typing import Dict, List

import numpy as np

from app.config import settings
from app.market.providers.base import BackendError, RealPriceBackend, TTLCache

__all__ = ["FinnhubBackend"]

_BASE_URL = "https://finnhub.io/api/v1"

#: Symbols this backend will attempt (equities + ETFs in our universe). Crypto is
#: handled elsewhere. Kept explicit so an unknown ticker delegates to the sim.
_EQUITY_ETF_SYMBOLS = {
    "AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "BAC", "V", "JNJ", "PFE",
    "XOM", "CVX", "AMZN", "KO", "CAT", "SPY", "QQQ", "VTI", "GLD",
}


class FinnhubBackend(RealPriceBackend):
    """Real equity/ETF price backend backed by Finnhub.

    Attributes:
        name: Backend name used in logs.
    """

    name = "finnhub"

    def __init__(self, api_key: str | None = None, ttl: float = 60.0) -> None:
        """Initialise the backend.

        Args:
            api_key: Finnhub API key; defaults to ``settings.finnhub_api_key``.
            ttl: Cache lifetime in seconds for price/candle responses.
        """
        self._api_key = api_key if api_key is not None else settings.finnhub_api_key
        self._cache = TTLCache(ttl=ttl)

    def available(self) -> bool:
        """Return whether a Finnhub API key is configured."""
        return bool(self._api_key)

    def supports(self, symbol: str) -> bool:
        """Return whether ``symbol`` is an equity/ETF Finnhub can serve."""
        return symbol.strip().upper() in _EQUITY_ETF_SYMBOLS

    def _candles_raw(self, symbol: str, days: int) -> List[Dict[str, float]]:
        """Fetch and normalise daily candles, with caching.

        Args:
            symbol: Asset ticker.
            days: Number of trailing calendar days of history to request.

        Returns:
            A list of ``{'t','o','h','l','c','v'}`` dicts oldest → newest.

        Raises:
            BackendError: On HTTP/parse failure or a non-``ok`` response.
        """
        sym = symbol.strip().upper()
        # Pad the window so we still get ~``days`` trading bars despite weekends.
        span_days = max(int(days) + 10, 10)
        cache_key = ("candles", sym, span_days)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        now = int(time.time())
        params = {
            "symbol": sym,
            "resolution": "D",
            "from": now - span_days * 86400,
            "to": now,
            "token": self._api_key,
        }
        data = self._get_json(f"{_BASE_URL}/stock/candle", params=params)
        if not isinstance(data, dict) or data.get("s") != "ok":
            raise BackendError(f"finnhub: no candle data for {sym!r}")
        try:
            ts = data["t"]
            opens, highs, lows, closes, vols = (
                data["o"], data["h"], data["l"], data["c"], data["v"]
            )
            out = [
                {
                    "t": int(ts[i]),
                    "o": float(opens[i]),
                    "h": float(highs[i]),
                    "l": float(lows[i]),
                    "c": float(closes[i]),
                    "v": float(vols[i]),
                }
                for i in range(len(ts))
            ]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise BackendError(f"finnhub: malformed candle data for {sym!r}: {exc}") from exc
        if not out:
            raise BackendError(f"finnhub: empty candle data for {sym!r}")
        self._cache.set(cache_key, out)
        return out

    def closes(self, symbol: str, days: int) -> np.ndarray:
        """Return up to ``days`` trailing daily closes (see base contract)."""
        candles = self._candles_raw(symbol, days)
        closes = np.asarray([c["c"] for c in candles], dtype=np.float64)
        if closes.size > days:
            closes = closes[-days:]
        return closes

    def candles(self, symbol: str, limit: int) -> List[Dict[str, float]]:
        """Return up to ``limit`` recent OHLCV candles (see base contract)."""
        out = self._candles_raw(symbol, limit)
        return out[-limit:] if len(out) > limit else out

    def latest(self, symbol: str) -> float:
        """Return the latest quote price via ``/quote`` (see base contract)."""
        sym = symbol.strip().upper()
        cache_key = ("latest", sym)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        data = self._get_json(
            f"{_BASE_URL}/quote", params={"symbol": sym, "token": self._api_key}
        )
        try:
            price = float(data["c"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError(f"finnhub: malformed quote for {sym!r}: {exc}") from exc
        if not price > 0:
            raise BackendError(f"finnhub: non-positive quote for {sym!r}")
        self._cache.set(cache_key, price)
        return price
