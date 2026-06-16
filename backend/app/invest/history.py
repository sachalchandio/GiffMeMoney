"""Backfilled portfolio value / P&L time series for seeding the live chart.

The :class:`PortfolioHistory` service reconstructs a recent-window value curve
for the account from each held asset's daily closing prices. This is a
*backfill*: it answers "what would this exact basket of units have been worth
over the last ``points`` trading days?" so the frontend has a curve to render
immediately, then appends live socket ticks for true real-time.

Per held position, over the trailing ``points`` daily closes::

    value[i]  = units * close[i]
    pnl[i]    = value[i] - cost_basis           (current cost basis, held flat)
    pnlPct[i] = pnl[i] / cost_basis * 100        (0 when there is no cost basis)

The total curve holds cash constant across the backfilled window (cash is a
"now" quantity; we do not have a cash history), so::

    total[i] = cash + Σ_position value[i]
    invested[i] = Σ_position value[i]

Timestamps are evenly spaced one trading day apart, ending at "now". With no
open positions the result is a flat cash line of length ``points``.

The service is read-only and defensive: a symbol whose history is unavailable is
simply skipped from the per-position series (and contributes nothing to the
total), so the curve never crashes.
"""

from __future__ import annotations

import math
import time

import numpy as np

from app.invest.store import AccountStore
from app.market.provider import MarketDataProvider
from app.schemas import (
    PortfolioHistory,
    PortfolioHistoryPoint,
    PositionHistory,
    PositionHistoryPoint,
)

__all__ = ["PortfolioHistory", "PortfolioHistoryService"]

# Default number of backfilled points (≈ 6 trading months).
_DEFAULT_POINTS: int = 120

# Milliseconds in one (calendar) day — the spacing between backfilled steps.
_DAY_MS: int = 86_400_000


class PortfolioHistoryService:
    """Reconstruct a recent-window value/P&L curve for an account.

    Args:
        store: The process-wide :class:`~app.invest.store.AccountStore`.
        provider: A :class:`~app.market.provider.MarketDataProvider` supplying
            each asset's daily closing-price history.
    """

    def __init__(self, store: AccountStore, provider: MarketDataProvider) -> None:
        """Store the collaborators (read-only; no state of its own)."""
        self._store = store
        self._provider = provider

    def portfolio_history(
        self, account_id: str, points: int = _DEFAULT_POINTS
    ) -> PortfolioHistory:
        """Build the backfilled total + per-position value/P&L series.

        Args:
            account_id: The account identifier.
            points: Number of trailing daily points to backfill (clamped to
                ``>= 1``; default 120).

        Returns:
            A :class:`~app.schemas.PortfolioHistory` whose ``total`` series has
            exactly ``n`` points and whose ``positions`` list has one
            :class:`~app.schemas.PositionHistory` (also ``n`` points) per held,
            priceable symbol. With no positions the ``total`` series is a flat
            cash line and ``positions`` is empty.
        """
        n = max(1, int(points)) if points else _DEFAULT_POINTS

        with self._store.lock:
            account = self._store.get_account(account_id)
            cash = float(account.cash_balance)
            if not math.isfinite(cash):
                cash = 0.0
            # Snapshot the position accounting under the lock; price-fetching and
            # series math happen on the copy so the lock is held briefly.
            holdings = [
                (s.symbol, float(s.units), float(s.cost_basis))
                for s in account.positions.values()
                if float(s.units) > 0.0
            ]

        timestamps = self._timestamps(n)

        position_series: list[PositionHistory] = []
        # Running sum of every position's value at each step (for the total curve).
        invested_per_step = np.zeros(n, dtype=np.float64)

        for symbol, units, cost_basis in holdings:
            closes = self._aligned_closes(symbol, n)
            if closes is None:
                continue
            values = closes * units
            pnl = values - cost_basis
            points_out: list[PositionHistoryPoint] = []
            for i in range(n):
                value = float(values[i])
                pnl_i = float(pnl[i])
                pnl_pct = pnl_i / cost_basis * 100.0 if cost_basis > 0.0 else 0.0
                points_out.append(
                    PositionHistoryPoint(
                        t=timestamps[i],
                        value=round(self._finite(value), 2),
                        pnl=round(self._finite(pnl_i), 2),
                        pnl_pct=round(self._finite(pnl_pct), 4),
                    )
                )
            position_series.append(
                PositionHistory(symbol=symbol, points=points_out)
            )
            invested_per_step += np.nan_to_num(
                values, nan=0.0, posinf=0.0, neginf=0.0
            )

        total_points: list[PortfolioHistoryPoint] = []
        for i in range(n):
            invested = float(invested_per_step[i])
            total_points.append(
                PortfolioHistoryPoint(
                    t=timestamps[i],
                    total_value=round(self._finite(cash + invested), 2),
                    invested=round(self._finite(invested), 2),
                    cash=round(self._finite(cash), 2),
                )
            )

        return PortfolioHistory(total=total_points, positions=position_series)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _aligned_closes(self, symbol: str, n: int) -> np.ndarray | None:
        """Return the last ``n`` daily closes for ``symbol``, padded if short.

        Pulls at least ``n`` days of history from the provider and keeps the most
        recent ``n`` closes. If the provider returns fewer than ``n`` valid
        closes the series is left-padded with the earliest available close so the
        output length is always exactly ``n``.

        Args:
            symbol: Asset ticker.
            n: Desired number of trailing closes.

        Returns:
            A length-``n`` ``float64`` array of closes, or ``None`` if no valid
            history is available for the symbol.
        """
        try:
            raw = self._provider.history(symbol, days=max(n, n + 1))
        except Exception:
            return None
        closes = np.asarray(raw, dtype=np.float64).ravel()
        closes = closes[np.isfinite(closes) & (closes > 0.0)]
        if closes.size == 0:
            return None
        if closes.size >= n:
            return closes[-n:].astype(np.float64)
        pad = np.full(n - closes.size, float(closes[0]), dtype=np.float64)
        return np.concatenate([pad, closes]).astype(np.float64)

    def _timestamps(self, n: int) -> list[int]:
        """Build ``n`` evenly spaced unix-ms timestamps ending at now.

        Steps are one (calendar) day apart; the last entry is the current time
        and the first is ``(n - 1)`` days earlier — a recent trailing window.

        Args:
            n: Number of timestamps.

        Returns:
            An ascending list of ``n`` unix-millisecond timestamps.
        """
        now = int(time.time() * 1000)
        return [now - (n - 1 - i) * _DAY_MS for i in range(n)]

    @staticmethod
    def _finite(value: float, default: float = 0.0) -> float:
        """Return ``value`` as a finite float, falling back to ``default``.

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
