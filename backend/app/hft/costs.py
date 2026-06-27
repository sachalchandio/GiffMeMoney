"""Transaction-cost model for the High-Frequency Simulation Lab.

The single most important honest input to short-horizon trading is the cost of
*doing* a trade. Every time you cross the market you pay:

    * **half the bid-ask spread** — you buy at the ask, sell at the bid, so each
      side gives up ~half the quoted spread;
    * **fees** — exchange / taker fees (crypto) or the implicit price you give up
      via payment-for-order-flow (commission-free equity);
    * **slippage / market impact** — your own order pushes the price against you,
      scaled by how big you are relative to the bar's liquidity (*participation*
      = order notional / bar dollar-volume).

The per-side cost as a fraction of notional is::

    cost_side(participation) = (half_spread_bps + fee_bps) / 1e4
                             + impact_coef * participation

A round trip (in then out) pays roughly twice the per-side cost. This is the toll
that makes turnover bleed money linearly — the whole point the lab demonstrates.

All values are basis points (1 bp = 0.01%) except ``impact_coef`` which is a
fraction-per-unit-participation. Everything is finite and defensive: bad inputs
collapse to 0 cost rather than raising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["CostModel", "COST_PRESETS", "get_cost_model", "DEFAULT_COST_PRESET"]

# 1 basis point as a fraction.
_BPS: float = 1e-4


def _finite(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite, non-negative float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v) or v < 0.0:
        return default
    return v


@dataclass(frozen=True)
class CostModel:
    """A realistic per-trade cost stack (spread + fee + slippage).

    Attributes:
        key: Stable preset id (e.g. ``"retail-crypto"``).
        name: Human-readable label.
        half_spread_bps: Half the bid-ask spread paid on each side, in bps.
        fee_bps: Exchange / taker fee (or PFOF-implied cost) per side, in bps.
        impact_coef: Slippage as a *fraction of notional* per unit of
            participation (order notional / bar dollar-volume). Small accounts
            have ~zero participation, so for a $20 book this term is negligible
            and the spread + fee dominate — exactly as in real life.
        note: One-line plain-English description of who this models.
    """

    key: str
    name: str
    half_spread_bps: float
    fee_bps: float
    impact_coef: float
    note: str

    def per_side_fraction(self, participation: float = 0.0) -> float:
        """Cost of one side (buy *or* sell) as a fraction of the traded notional.

        Args:
            participation: Order notional / bar dollar-volume in ``[0, 1]``
                (clamped). Drives the slippage term; ~0 for a tiny account.

        Returns:
            A finite, non-negative fraction (e.g. ``0.0015`` = 15 bps).
        """
        part = min(max(_finite(participation), 0.0), 1.0)
        spread_fee = (_finite(self.half_spread_bps) + _finite(self.fee_bps)) * _BPS
        impact = _finite(self.impact_coef) * part
        return spread_fee + impact

    def round_trip_bps(self, participation: float = 0.0) -> float:
        """Approximate round-trip (in + out) cost in basis points.

        Args:
            participation: See :meth:`per_side_fraction`.

        Returns:
            The round-trip cost in bps (``2 * per_side`` expressed in bps).
        """
        return 2.0 * self.per_side_fraction(participation) / _BPS

    def cost_of(self, trade_notional: float, participation: float = 0.0) -> float:
        """Dollar cost charged for trading ``|trade_notional|`` on one side.

        Args:
            trade_notional: Signed dollar value transacted (sign ignored).
            participation: See :meth:`per_side_fraction`.

        Returns:
            A finite, non-negative dollar cost.
        """
        notional = abs(_finite(trade_notional))
        return notional * self.per_side_fraction(participation)


#: The built-in cost presets, from frictionless illustration to a brutal taker.
#: ``zero`` exists ONLY to measure the *gross* (pre-cost) path; it is never a
#: real-world option. The retail presets are deliberately realistic.
COST_PRESETS: dict[str, CostModel] = {
    "zero": CostModel(
        key="zero",
        name="Frictionless (illustration only)",
        half_spread_bps=0.0,
        fee_bps=0.0,
        impact_coef=0.0,
        note="No costs at all. Not real — used only to show the pre-cost path.",
    ),
    "retail-equity": CostModel(
        key="retail-equity",
        name="Retail equity (commission-free)",
        half_spread_bps=1.0,
        fee_bps=0.0,
        impact_coef=0.0015,
        note="Liquid US stock, $0 commission but you still cross the spread / PFOF (~2 bps round trip).",
    ),
    "retail-crypto": CostModel(
        key="retail-crypto",
        name="Retail crypto (taker)",
        half_spread_bps=4.0,
        fee_bps=10.0,
        impact_coef=0.003,
        note="Typical taker fee + spread on a major exchange (~28 bps round trip).",
    ),
    "retail-crypto-expensive": CostModel(
        key="retail-crypto-expensive",
        name="Retail crypto (high-fee venue)",
        half_spread_bps=10.0,
        fee_bps=40.0,
        impact_coef=0.006,
        note="A pricey app / illiquid pair (~100 bps round trip) — the turnover killer.",
    ),
}

#: Default preset when a request does not name one (a realistic, not flattering, choice).
DEFAULT_COST_PRESET: str = "retail-crypto"


def get_cost_model(key: str | None) -> CostModel:
    """Resolve a cost-preset id to its :class:`CostModel` (defensive default).

    Args:
        key: A preset id (case-insensitive). ``None`` / unknown falls back to
            :data:`DEFAULT_COST_PRESET`.

    Returns:
        The matching :class:`CostModel`, or the default preset.
    """
    norm = str(key or "").strip().lower()
    return COST_PRESETS.get(norm, COST_PRESETS[DEFAULT_COST_PRESET])
