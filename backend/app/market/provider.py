"""Market-data provider abstraction and the built-in simulated implementation.

The rest of the application talks to market data exclusively through the
:class:`MarketDataProvider` interface, so a real third-party adapter (Finnhub,
Polygon, …) can be dropped in later without touching the quant or API layers.
The only implementation shipped here is :class:`SimulatedProvider`, which wraps
:mod:`app.market.universe` (static seeds) and :mod:`app.market.simulator`
(deterministic dynamics).

A process-wide singleton is chosen by :data:`app.config.settings.provider` via
:func:`get_provider`.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import numpy as np

from app.config import settings
from app.market import simulator
from app.market.universe import (
    AssetSeed,
    Fundamentals,
    UNIVERSE,
    get_seed,
)
from app.schemas import Asset, Candle

__all__ = [
    "MarketDataProvider",
    "SimulatedProvider",
    "get_provider",
]

logger = logging.getLogger("app.market.provider")


class MarketDataProvider(ABC):
    """Abstract interface every market-data source must implement.

    All return types are the frozen wire DTOs (:class:`app.schemas.Asset`,
    :class:`app.schemas.Candle`) or plain numpy arrays for the quant layer.
    Implementations must be safe to call concurrently from request and
    WebSocket threads.
    """

    @abstractmethod
    def list_assets(self) -> List[Asset]:
        """Return every asset in the universe as an :class:`Asset` snapshot."""
        raise NotImplementedError

    @abstractmethod
    def get_asset(self, symbol: str) -> Asset:
        """Return one :class:`Asset` snapshot.

        Raises:
            KeyError: If the symbol is unknown.
        """
        raise NotImplementedError

    @abstractmethod
    def get_candles(self, symbol: str, limit: int) -> List[Candle]:
        """Return up to ``limit`` recent OHLCV candles for a symbol."""
        raise NotImplementedError

    @abstractmethod
    def history(self, symbol: str, days: int) -> np.ndarray:
        """Return a symbol's daily closing prices as a numpy array."""
        raise NotImplementedError

    @abstractmethod
    def market_history(self, days: int) -> np.ndarray:
        """Return the shared market-index daily closing prices."""
        raise NotImplementedError

    @abstractmethod
    def factor_history(self, days: int) -> Dict[str, np.ndarray]:
        """Return the shared factor daily returns ``{'mkt','smb','hml','rf'}``."""
        raise NotImplementedError

    @abstractmethod
    def fundamentals(self, symbol: str) -> Fundamentals:
        """Return a symbol's :class:`Fundamentals` record.

        Raises:
            KeyError: If the symbol is unknown.
        """
        raise NotImplementedError

    @abstractmethod
    def latest_price(self, symbol: str) -> float:
        """Return a symbol's most recent close price."""
        raise NotImplementedError


def _change24h_pct(closes: np.ndarray) -> float:
    """Compute the last-day percentage change from a close series.

    Formula:
        change24hPct = (P_last / P_prev - 1) * 100

    Args:
        closes: Daily closing prices (length ``>= 1``).

    Returns:
        The day-over-day change in percent, or ``0.0`` when there are fewer than
        two valid closes or the previous close is non-positive.
    """
    if closes.size < 2:
        return 0.0
    prev = float(closes[-2])
    last = float(closes[-1])
    if not np.isfinite(prev) or not np.isfinite(last) or prev <= 0.0:
        return 0.0
    pct = (last / prev - 1.0) * 100.0
    return float(pct) if np.isfinite(pct) else 0.0


class SimulatedProvider(MarketDataProvider):
    """A fully self-contained provider backed by the deterministic simulator.

    Requires no network access and no API keys: histories come from
    :mod:`app.market.simulator` and static identity / fundamentals from
    :mod:`app.market.universe`. :class:`Asset` snapshots use the latest
    simulated close as the price and the last-two-closes delta as
    ``change24hPct``.
    """

    def _seed_to_asset(self, seed: AssetSeed) -> Asset:
        """Build an :class:`Asset` DTO from a seed and its simulated history.

        Args:
            seed: The asset's static :class:`AssetSeed`.

        Returns:
            A populated :class:`Asset`. For crypto assets ``volume24h`` and
            ``market_cap`` come from the seed; equities/ETFs likewise.
        """
        closes = simulator.daily_closes(seed.symbol)
        price = float(closes[-1]) if closes.size else float(seed.base_price)
        change = _change24h_pct(closes)
        return Asset(
            symbol=seed.symbol,
            name=seed.name,
            asset_class=seed.asset_class,  # type: ignore[arg-type]
            sector=seed.sector,
            currency=seed.currency,
            price=round(price, 6),
            change24h_pct=round(change, 4),
            market_cap=float(seed.market_cap) if seed.market_cap else None,
            volume24h=float(seed.volume24h) if seed.volume24h else None,
        )

    def list_assets(self) -> List[Asset]:
        """Return an :class:`Asset` snapshot for the whole universe.

        Returns:
            A list of :class:`Asset` objects in universe declaration order.
        """
        return [self._seed_to_asset(seed) for seed in UNIVERSE]

    def get_asset(self, symbol: str) -> Asset:
        """Return a single :class:`Asset` snapshot.

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            The :class:`Asset` snapshot.

        Raises:
            KeyError: If the symbol is unknown.
        """
        return self._seed_to_asset(get_seed(symbol))

    def get_candles(self, symbol: str, limit: int = 365) -> List[Candle]:
        """Return up to ``limit`` recent OHLCV candles for a symbol.

        Args:
            symbol: Asset ticker (case-insensitive).
            limit: Maximum number of candles (most recent).

        Returns:
            A list of :class:`Candle` objects ordered oldest → newest.

        Raises:
            KeyError: If the symbol is unknown.
        """
        # Validate symbol (raises KeyError if unknown) before generating.
        get_seed(symbol)
        raw = simulator.generate_candles(symbol, limit=limit)
        return [Candle(**c) for c in raw]

    def history(self, symbol: str, days: int = simulator._DEFAULT_DAYS) -> np.ndarray:
        """Return a symbol's daily closing prices.

        Args:
            symbol: Asset ticker (case-insensitive).
            days: Number of daily returns (result length is ``days + 1``).

        Returns:
            A ``float64`` array of closing prices.

        Raises:
            KeyError: If the symbol is unknown.
        """
        get_seed(symbol)
        return simulator.daily_closes(symbol, days=days)

    def market_history(self, days: int = simulator._DEFAULT_DAYS) -> np.ndarray:
        """Return the shared synthetic market-index closing prices.

        Args:
            days: Number of daily returns (result length is ``days + 1``).

        Returns:
            A ``float64`` array of index levels.
        """
        return simulator.market_closes(days)

    def factor_history(self, days: int = simulator._DEFAULT_DAYS) -> Dict[str, np.ndarray]:
        """Return the shared factor daily-return series.

        Args:
            days: Number of trailing daily returns.

        Returns:
            A dict ``{'mkt','smb','hml','rf'}`` of equal-length arrays.
        """
        return simulator.factor_returns(days)

    def fundamentals(self, symbol: str) -> Fundamentals:
        """Return a symbol's :class:`Fundamentals` record.

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            The :class:`Fundamentals` from the universe seed.

        Raises:
            KeyError: If the symbol is unknown.
        """
        return get_seed(symbol).fundamentals

    def latest_price(self, symbol: str) -> float:
        """Return a symbol's most recent simulated close.

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            The latest close as a float.

        Raises:
            KeyError: If the symbol is unknown.
        """
        get_seed(symbol)
        return simulator.latest_price(symbol)


# ---------------------------------------------------------------------------
# Provider registry / singleton
# ---------------------------------------------------------------------------


def _build_simulated() -> MarketDataProvider:
    """Build the pure deterministic simulated provider (the safe default)."""
    return SimulatedProvider()


def _build_hybrid(backend_name: str) -> Callable[[], MarketDataProvider]:
    """Return a factory that wraps a named real price backend in a Hybrid.

    The real adapters live in :mod:`app.market.providers`; they are imported
    lazily inside the factory so the base provider module has **no import-time
    dependency** on httpx-backed code (and the default simulated path never
    touches the network). If a backend reports itself unavailable (e.g. a
    missing API key) or anything goes wrong constructing it, the factory raises,
    and :func:`get_provider` catches that and falls back to the simulator.

    Args:
        backend_name: One of ``'finnhub' | 'polygon' | 'coingecko' | 'binance'``.

    Returns:
        A zero-arg factory producing a :class:`MarketDataProvider`.
    """

    def factory() -> MarketDataProvider:
        # Lazy imports keep the simulated default network-free and import-light.
        from app.market.providers import (  # noqa: PLC0415 - intentional lazy import
            BinanceBackend,
            CoinGeckoBackend,
            FinnhubBackend,
            HybridProvider,
            PolygonBackend,
        )

        backends: Dict[str, type] = {
            "finnhub": FinnhubBackend,
            "polygon": PolygonBackend,
            "coingecko": CoinGeckoBackend,
            "binance": BinanceBackend,
        }
        backend_cls = backends[backend_name]
        backend = backend_cls()
        if not backend.available():
            # No usable credentials → signal a fallback to the simulator.
            raise RuntimeError(
                f"provider '{backend_name}' is not configured (missing API key)"
            )
        return HybridProvider(backend, SimulatedProvider())

    return factory


# Maps the settings ``provider`` key to a zero-arg factory. New adapters
# register here without changing call sites. Every real adapter is a
# HybridProvider so the quant engine always has factor/fundamental data.
_PROVIDERS: Dict[str, Callable[[], MarketDataProvider]] = {
    "simulated": _build_simulated,
    "finnhub": _build_hybrid("finnhub"),
    "polygon": _build_hybrid("polygon"),
    "coingecko": _build_hybrid("coingecko"),
    "binance": _build_hybrid("binance"),
}

_PROVIDER_LOCK = threading.Lock()
_PROVIDER_INSTANCE: MarketDataProvider | None = None


def get_provider() -> MarketDataProvider:
    """Return the process-wide :class:`MarketDataProvider` singleton.

    The concrete provider is selected by :data:`app.config.settings.provider`.
    Selection is **fail-safe**: an unknown key, a missing API key, or *any*
    error while constructing a real adapter falls back to the pure
    :class:`SimulatedProvider` (logged once at WARNING) so the app always boots
    and the default behaviour is never disrupted. Live data is read-only.

    Returns:
        The shared provider instance (a :class:`SimulatedProvider` by default,
        or a hybrid real-backed provider when a real ``provider`` is configured
        and usable).
    """
    global _PROVIDER_INSTANCE
    if _PROVIDER_INSTANCE is None:
        with _PROVIDER_LOCK:
            if _PROVIDER_INSTANCE is None:
                _PROVIDER_INSTANCE = _select_provider()
    return _PROVIDER_INSTANCE


def _select_provider() -> MarketDataProvider:
    """Construct the configured provider, falling back to simulated on any error.

    Returns:
        The selected provider, or a :class:`SimulatedProvider` if the configured
        one is unknown/unconfigured/errored.
    """
    key = (settings.provider or "simulated").strip().lower()
    factory = _PROVIDERS.get(key)
    if factory is None:
        logger.warning(
            "Unknown provider %r; falling back to the simulated provider.", key
        )
        return SimulatedProvider()
    if key == "simulated":
        return factory()
    try:
        provider = factory()
        logger.info("Market-data provider: using real backend %r (hybrid).", key)
        return provider
    except Exception as exc:  # noqa: BLE001 - never crash; always degrade safely
        logger.warning(
            "Provider %r unavailable (%s); falling back to the simulated provider.",
            key,
            exc,
        )
        return SimulatedProvider()


def reset_provider_cache() -> None:
    """Drop the cached provider singleton (test helper).

    The next :func:`get_provider` call re-selects based on the current
    :data:`app.config.settings.provider`. Not used in production code paths.
    """
    global _PROVIDER_INSTANCE
    with _PROVIDER_LOCK:
        _PROVIDER_INSTANCE = None
