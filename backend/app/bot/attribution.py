"""Per-sleeve performance attribution for an auto-trader run (SIMULATION only).

The engine tracks, for each sleeve (a held symbol / strategy), its accumulated
realized + marked dollar P&L, the number of trades it generated, and how many of
its rebalance legs were profitable. This module turns those raw accumulators
into ranked :class:`~app.schemas.SleeveAttribution` rows:

    * ``contribution_pct`` — the sleeve's share of the run's TOTAL absolute P&L
      (signed), so the magnitudes sum sensibly and a single sleeve's importance
      is comparable across runs;
    * ``verdict`` — ``'best'`` for the top positive contributor, ``'worst'`` for
      the most negative, ``'neutral'`` for everything in between.

Everything is defensive: empty / degenerate inputs yield an empty list, and all
emitted numbers are finite. These are SIMULATED attributions on synthetic data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.schemas import SleeveAttribution

__all__ = ["SleeveStat", "build_attribution"]


@dataclass
class SleeveStat:
    """Mutable per-sleeve accumulator the engine fills during a backtest.

    Attributes:
        key: The sleeve key (symbol or strategy name).
        realized_pnl: Cumulative realized + marked dollar P&L of the sleeve.
        trades: Number of trades the sleeve generated.
        wins: Number of profitable rebalance legs.
        legs: Total number of completed rebalance legs (for the win rate).
    """

    key: str
    realized_pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    legs: int = 0

    def win_rate(self) -> float:
        """Fraction of profitable legs in ``[0, 1]`` (0 when no legs)."""
        if self.legs <= 0:
            return 0.0
        wr = self.wins / float(self.legs)
        if not math.isfinite(wr):
            return 0.0
        return min(1.0, max(0.0, wr))


def _finite(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def build_attribution(stats: list[SleeveStat]) -> list[SleeveAttribution]:
    """Rank sleeves best → worst by realized P&L and mark best/worst verdicts.

    The contribution percentage is each sleeve's P&L as a share of the run's
    total *absolute* P&L (so winners and losers are comparable in magnitude and a
    flat run yields zero contributions rather than dividing by zero). The single
    most positive contributor is flagged ``'best'`` and the single most negative
    ``'worst'`` (only when their P&L actually has that sign); all others are
    ``'neutral'``.

    Args:
        stats: The per-sleeve accumulators (any order).

    Returns:
        A list of :class:`~app.schemas.SleeveAttribution`, sorted by realized
        P&L descending (best first). Empty when ``stats`` is empty.
    """
    rows = [s for s in (stats or []) if s is not None]
    if not rows:
        return []

    total_abs = sum(abs(_finite(s.realized_pnl)) for s in rows)
    ranked = sorted(rows, key=lambda s: _finite(s.realized_pnl), reverse=True)

    best_key = ranked[0].key if _finite(ranked[0].realized_pnl) > 0.0 else None
    worst_key = ranked[-1].key if _finite(ranked[-1].realized_pnl) < 0.0 else None

    out: list[SleeveAttribution] = []
    for s in ranked:
        pnl = _finite(s.realized_pnl)
        contribution = (pnl / total_abs * 100.0) if total_abs > 1e-12 else 0.0
        if s.key == best_key:
            verdict = "best"
        elif s.key == worst_key:
            verdict = "worst"
        else:
            verdict = "neutral"
        out.append(
            SleeveAttribution(
                key=str(s.key),
                realized_pnl=round(pnl, 2),
                contribution_pct=round(_finite(contribution), 2),
                win_rate=round(s.win_rate(), 4),
                trades=int(max(0, s.trades)),
                verdict=verdict,  # type: ignore[arg-type]
            )
        )
    return out
