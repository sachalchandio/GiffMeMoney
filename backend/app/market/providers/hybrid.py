"""Hybrid provider: real prices + simulated factors/fundamentals.

:class:`HybridProvider` is the bridge that lets the quant engine keep working on
real market data. It implements the full
:class:`app.market.provider.MarketDataProvider` interface by combining:

* a :class:`~app.market.providers.base.RealPriceBackend` (Finnhub / Polygon /
  CoinGecko / Binance) for **prices, candles, and latest quotes**, and
* the deterministic :class:`app.market.provider.SimulatedProvider` for
  everything else.

Approximation (documented honestly)
-----------------------------------
Real price feeds do **not** publish the academic factor series the engine
regresses against (the market / SMB / HML / risk-free returns) nor a uniform
fundamentals record for every ticker. So:

* ``factor_history``, ``market_history`` and ``fundamentals`` **always** come
  from the simulator;
* ``list_assets``, ``get_asset``, ``get_candles``, ``history`` and
  ``latest_price`` use the **real** backend for symbols it ``supports`` and that
  succeed, and **fall back to the simulator** for any symbol the backend lacks
  or any call that errors.

Net effect: price/technical strategies run on real data; factor and fundamental
models run on real-where-available prices plus simulated factor/fundamental
inputs. Every failure is non-fatal and logged once at WARNING.

The asset *identity* (name, class, sector, currency, market cap) always comes
from the static :mod:`app.market.universe` seed; only the live ``price`` and
``change24hPct`` are overlaid from the real backend when available.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from app.market.provider import MarketDataProvider, SimulatedProvider
from app.market.providers.base import BackendError, RealPriceBackend
from app.market.universe import Fundamentals, UNIVERSE, get_seed
from app.schemas import Asset, Candle

__all__ = ["HybridProvider"]

logger = logging.getLogger("app.market.providers.hybrid")


class HybridProvider(MarketDataProvider):
    """A provider that overlays a real price backend on the simulator.

    Args:
        backend: The real price backend to use for price-driven calls.
        simulated: The simulator used for factors, fundamentals, and as the
            fallback for any unsupported symbol or failed call. A fresh
            :class:`SimulatedProvider` is created if not supplied.

    Attributes:
        backend: The wrapped real price backend.
        sim: The simulated provider used for delegation/fallback.
    """

    def __init__(
        self,
        backend: RealPriceBackend,
        simulated: SimulatedProvider | None = None,
    ) -> None:
        self.backend: RealPriceBackend = backend
        self.sim: SimulatedProvider = simulated or SimulatedProvider()
        # Latch so the "using real backend / falling back" decision is logged at
        # most once per provider instance rather than per call.
        self._logged_fallback = False

    # -- internal helpers ---------------------------------------------------

    def _warn_once(self, message: str) -> None:
        """Log ``message`` at WARNING exactly once per provider instance."""
        if not self._logged_fallback:
            logger.warning("%s", message)
            self._logged_fallback = True

    def _use_real(self, symbol: str) -> bool:
        """Return whether the real backend should be tried for ``symbol``."""
        try:
            return self.backend.available() and self.backend.supports(symbol)
        except Exception:  # noqa: BLE001 - a broken backend never breaks us
            return False

    # -- Asset snapshots ----------------------------------------------------

    def _asset_with_real_price(self, symbol: str) -> Asset:
        """Build an :class:`Asset` using the seed identity + real price overlay.

        The static identity (name/class/sector/currency/market cap/volume) comes
        from the universe seed; ``price`` is the real latest price and
        ``change24hPct`` is derived from the real backend's last two closes when
        available. Any failure falls back to the simulator's snapshot.

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            A populated :class:`Asset`.
        """
        sym = symbol.strip().upper()
        seed = get_seed(sym)  # raises KeyError for unknown symbols (intended)
        try:
            price = self.backend.latest(sym)
            change_pct = 0.0
            try:
                closes = self.backend.closes(sym, 2)
                if closes.size >= 2 and float(closes[-2]) > 0:
                    change_pct = (float(closes[-1]) / float(closes[-2]) - 1.0) * 100.0
            except BackendError:
                # Price is enough for a snapshot; leave change at 0 if history
                # is unavailable on this tier.
                change_pct = 0.0
            return Asset(
                symbol=seed.symbol,
                name=seed.name,
                asset_class=seed.asset_class,  # type: ignore[arg-type]
                sector=seed.sector,
                currency=seed.currency,
                price=round(float(price), 6),
                change24h_pct=round(float(change_pct), 4),
                market_cap=float(seed.market_cap) if seed.market_cap else None,
                volume24h=float(seed.volume24h) if seed.volume24h else None,
            )
        except BackendError as exc:
            self._warn_once(
                f"hybrid: real backend '{self.backend.name}' failed for {sym!r} "
                f"({exc}); falling back to simulated for unavailable data."
            )
            return self.sim.get_asset(sym)

    def list_assets(self) -> List[Asset]:
        """Return an :class:`Asset` snapshot for the whole universe.

        Symbols the real backend covers use real prices; the rest use the
        simulator. Order matches the universe declaration order.

        Returns:
            A list of :class:`Asset` snapshots.
        """
        out: List[Asset] = []
        for seed in UNIVERSE:
            if self._use_real(seed.symbol):
                out.append(self._asset_with_real_price(seed.symbol))
            else:
                out.append(self.sim.get_asset(seed.symbol))
        return out

    def get_asset(self, symbol: str) -> Asset:
        """Return a single :class:`Asset` snapshot (real price when available).

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            The :class:`Asset` snapshot.

        Raises:
            KeyError: If the symbol is unknown to the universe.
        """
        if self._use_real(symbol):
            return self._asset_with_real_price(symbol)
        return self.sim.get_asset(symbol)

    # -- Candles / histories (real with fallback) ---------------------------

    def get_candles(self, symbol: str, limit: int = 365) -> List[Candle]:
        """Return up to ``limit`` recent candles (real backend, else simulator).

        Args:
            symbol: Asset ticker (case-insensitive).
            limit: Maximum number of candles (most recent).

        Returns:
            A list of :class:`Candle` objects oldest → newest.

        Raises:
            KeyError: If the symbol is unknown to the universe.
        """
        get_seed(symbol)  # validate symbol
        if self._use_real(symbol):
            try:
                raw = self.backend.candles(symbol, limit)
                if raw:
                    return [Candle(**c) for c in raw]
            except BackendError as exc:
                self._warn_once(
                    f"hybrid: '{self.backend.name}' candles failed for {symbol!r} "
                    f"({exc}); using simulated candles."
                )
        return self.sim.get_candles(symbol, limit)

    def history(self, symbol: str, days: int) -> np.ndarray:
        """Return a symbol's daily closing prices (real backend, else simulator).

        Args:
            symbol: Asset ticker (case-insensitive).
            days: Number of trailing daily closes desired.

        Returns:
            A ``float64`` array of closing prices.

        Raises:
            KeyError: If the symbol is unknown to the universe.
        """
        get_seed(symbol)
        if self._use_real(symbol):
            try:
                closes = self.backend.closes(symbol, days)
                if closes.size:
                    return closes
            except BackendError as exc:
                self._warn_once(
                    f"hybrid: '{self.backend.name}' history failed for {symbol!r} "
                    f"({exc}); using simulated history."
                )
        return self.sim.history(symbol, days)

    def latest_price(self, symbol: str) -> float:
        """Return the latest price (real backend, else simulator).

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            The latest price as a float.

        Raises:
            KeyError: If the symbol is unknown to the universe.
        """
        get_seed(symbol)
        if self._use_real(symbol):
            try:
                return float(self.backend.latest(symbol))
            except BackendError as exc:
                self._warn_once(
                    f"hybrid: '{self.backend.name}' latest failed for {symbol!r} "
                    f"({exc}); using simulated price."
                )
        return self.sim.latest_price(symbol)

    # -- Always-simulated (real feeds don't provide these cleanly) ----------

    def market_history(self, days: int) -> np.ndarray:
        """Return the shared market-index closes — always simulated.

        Real feeds expose individual index ETFs, not the engine's synthetic
        market-index level series, so this delegates to the simulator.
        """
        return self.sim.market_history(days)

    def factor_history(self, days: int) -> Dict[str, np.ndarray]:
        """Return the SMB/HML/mkt/rf factor series — always simulated.

        The Fama-French factor returns are not published by these price feeds,
        so the deterministic simulator's factor history is used (this is the
        documented approximation: factor models use simulated factors).
        """
        return self.sim.factor_history(days)

    def fundamentals(self, symbol: str) -> Fundamentals:
        """Return a symbol's fundamentals — always simulated.

        A uniform :class:`Fundamentals` record (the full Altman/Piotroski/DCF
        input set) is not available cleanly across these feeds, so the
        simulator's deterministic fundamentals are used.

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            The :class:`Fundamentals` record from the universe seed.

        Raises:
            KeyError: If the symbol is unknown to the universe.
        """
        return self.sim.fundamentals(symbol)
