"""Real (network-backed) market-data adapters for GiffMeMoney.

Every adapter in this package implements the existing
:class:`app.market.provider.MarketDataProvider` interface so it can be selected
by :data:`app.config.settings.provider` without touching the quant or API layers.

Design (honest about approximations)
------------------------------------
Real feeds give clean **prices, candles, and latest quotes** for the symbols in
their universe, but they do *not* expose the academic factor series (SMB / HML /
rf) the quant engine regresses against, nor a uniform fundamentals record for
every ticker. So each real adapter is wrapped by a :class:`HybridProvider`:

* **price-driven** methods (``list_assets``, ``get_asset``, ``get_candles``,
  ``history``, ``latest_price``) use the **real** backend for symbols it covers;
* **factor / fundamental / market-index** methods (``factor_history``,
  ``fundamentals``, ``market_history``) and **any symbol the backend lacks**
  delegate to the deterministic :class:`SimulatedProvider`.

This means: price/technical strategies run on real data, while factor and
fundamental models run on real-where-available + simulated values. The
approximation is documented per-method.

Safety / robustness
-------------------
* Every real call has a per-call HTTP timeout and a small TTL cache (so free
  API tiers are respected).
* A missing key or **any** error never crashes: the selecting
  :func:`app.market.provider.get_provider` falls back to the pure
  :class:`SimulatedProvider` and logs once. Live data is strictly **read-only**.
"""

from __future__ import annotations

from app.market.providers.base import RealPriceBackend, TTLCache
from app.market.providers.binance import BinanceBackend
from app.market.providers.coingecko import CoinGeckoBackend
from app.market.providers.finnhub import FinnhubBackend
from app.market.providers.hybrid import HybridProvider
from app.market.providers.polygon import PolygonBackend

__all__ = [
    "RealPriceBackend",
    "TTLCache",
    "FinnhubBackend",
    "PolygonBackend",
    "CoinGeckoBackend",
    "BinanceBackend",
    "HybridProvider",
]
