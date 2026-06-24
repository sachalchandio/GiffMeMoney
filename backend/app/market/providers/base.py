"""Shared plumbing for the real market-data backends.

This module defines:

* :class:`TTLCache` — a tiny thread-safe time-to-live cache so repeated calls
  (and free-tier rate limits) are respected without a heavy dependency;
* :class:`BackendError` — the single error type real backends raise so the
  :class:`~app.market.providers.hybrid.HybridProvider` can cleanly fall back to
  the simulator;
* :class:`RealPriceBackend` — the abstract contract a real **price** backend
  implements (prices/candles/latest over its own symbol universe). Everything
  the quant engine needs beyond prices (factors, fundamentals) is supplied by
  the simulator via :class:`HybridProvider`, so backends only deal with prices.

A backend never raises raw network/parse errors at the call site: it converts
them to :class:`BackendError`, which the hybrid wrapper catches and translates
into a simulated fallback (logged once). Live data is strictly read-only.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

__all__ = [
    "BackendError",
    "TTLCache",
    "RealPriceBackend",
    "DEFAULT_TIMEOUT",
]

#: Per-call HTTP timeout (seconds) for every real backend request. Short enough
#: that a slow/blocked feed degrades to the simulator quickly rather than
#: hanging a request or WebSocket thread.
DEFAULT_TIMEOUT: float = 6.0


class BackendError(RuntimeError):
    """Raised by a real backend when data cannot be fetched or parsed.

    The :class:`~app.market.providers.hybrid.HybridProvider` catches this and
    falls back to the deterministic simulator, so a single failing symbol or a
    flaky network never crashes a request.
    """


class TTLCache:
    """A minimal thread-safe time-to-live cache.

    Values live for ``ttl`` seconds; expired entries are recomputed on next
    access. Used to coalesce repeated provider calls and stay inside free-tier
    rate limits without pulling in an external caching dependency.

    Attributes:
        ttl: Entry lifetime in seconds.
    """

    def __init__(self, ttl: float = 60.0) -> None:
        """Initialise an empty cache.

        Args:
            ttl: Entry lifetime in seconds (values older than this are stale).
        """
        self.ttl: float = float(ttl)
        self._store: Dict[Any, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        """Return a cached value if present and not expired, else ``None``.

        Args:
            key: The cache key (must be hashable).

        Returns:
            The stored value, or ``None`` if absent or stale.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            stamp, value = entry
            if now - stamp > self.ttl:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: Any, value: Any) -> None:
        """Store ``value`` under ``key`` stamped with the current time.

        Args:
            key: The cache key (must be hashable).
            value: The value to cache.
        """
        with self._lock:
            self._store[key] = (time.monotonic(), value)

    def clear(self) -> None:
        """Drop every cached entry (used by tests)."""
        with self._lock:
            self._store.clear()


class RealPriceBackend(ABC):
    """Abstract contract for a network-backed **price** source.

    A backend only deals with prices — closing-price histories, OHLCV candles,
    and the latest price — for the symbols in :meth:`supports`. Factor series and
    fundamentals are intentionally *not* part of this contract: the
    :class:`~app.market.providers.hybrid.HybridProvider` supplies those (and any
    symbol the backend does not cover) from the deterministic simulator.

    Implementations must:
        * apply a per-call HTTP timeout (use :data:`DEFAULT_TIMEOUT`);
        * cache responses (use a :class:`TTLCache`) to respect rate limits;
        * raise :class:`BackendError` (never a raw network/parse error) on any
          failure so the hybrid wrapper can fall back to the simulator;
        * be safe to call from concurrent request / WebSocket threads.
    """

    #: Human-readable backend name (used in log messages).
    name: str = "real"

    @abstractmethod
    def available(self) -> bool:
        """Return whether this backend is configured (e.g. has its API key).

        Returns:
            ``True`` if the backend can attempt real calls; ``False`` if a
            required key is missing (the hybrid wrapper then stays on the
            simulator entirely).
        """
        raise NotImplementedError

    @abstractmethod
    def supports(self, symbol: str) -> bool:
        """Return whether this backend covers ``symbol``.

        Symbols a backend does not cover are served from the simulator by the
        hybrid wrapper, so the universe stays whole.

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            ``True`` if the backend has a real mapping for the symbol.
        """
        raise NotImplementedError

    @abstractmethod
    def closes(self, symbol: str, days: int) -> np.ndarray:
        """Return up to ``days`` trailing daily closes for ``symbol``.

        Args:
            symbol: Asset ticker (case-insensitive).
            days: Number of trailing daily closes desired.

        Returns:
            A ``float64`` array of closing prices ordered oldest → newest.

        Raises:
            BackendError: On any network/parse failure or unknown symbol.
        """
        raise NotImplementedError

    @abstractmethod
    def candles(self, symbol: str, limit: int) -> List[Dict[str, float]]:
        """Return up to ``limit`` recent OHLCV candles for ``symbol``.

        Args:
            symbol: Asset ticker (case-insensitive).
            limit: Maximum number of candles (most recent), oldest → newest.

        Returns:
            A list of dicts ``{'t','o','h','l','c','v'}`` where ``t`` is unix
            **seconds** (matching :class:`app.schemas.Candle`).

        Raises:
            BackendError: On any network/parse failure or unknown symbol.
        """
        raise NotImplementedError

    @abstractmethod
    def latest(self, symbol: str) -> float:
        """Return the most recent price for ``symbol``.

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            The latest trade/close price as a float.

        Raises:
            BackendError: On any network/parse failure or unknown symbol.
        """
        raise NotImplementedError

    # -- shared HTTP helper -------------------------------------------------

    def _get_json(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> Any:
        """Perform a GET and return the decoded JSON, or raise :class:`BackendError`.

        Centralises the timeout, status check, and JSON decode so every backend
        converts *all* failure modes (connect/read timeout, non-2xx, malformed
        body) into a single :class:`BackendError` the hybrid wrapper handles.

        Args:
            url: Absolute request URL.
            params: Optional query parameters.
            headers: Optional request headers (e.g. API-key headers).
            timeout: Per-call timeout in seconds.

        Returns:
            The decoded JSON payload (dict or list).

        Raises:
            BackendError: On any HTTP error, timeout, or JSON decode failure.
        """
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - normalise every failure
            raise BackendError(f"{self.name}: GET {url} failed: {exc}") from exc
