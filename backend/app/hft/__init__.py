"""High-Frequency Simulation Lab — honest, paper-only short-horizon trading.

HONESTY / SAFETY (this is a finance tool). Everything in this package is a
**SIMULATION on synthetic data**. It exists to answer one question truthfully:
*does trading faster / in smaller portions make more money, or less?* — by
modelling the things that actually decide that (the bid-ask spread, fees,
slippage, and noise) instead of pretending they don't exist.

Three honest facts this package is built around:

1. **You cannot trade in microseconds from a web app.** Real HFT co-locates
   hardware inside the exchange; a broker REST round-trip is ~50-300 *milli*
   seconds — a million times slower. So this lab simulates *bars*, and is
   explicit that it is not, and cannot be, microsecond trading.
2. **Every trade pays a fixed toll** (spread + fee + slippage), so turnover
   bleeds money *linearly*. The :mod:`~app.hft.lab` turnover sweep measures
   exactly where net-of-cost return peaks — almost always at *low* turnover.
3. **Synthetic data has no free alpha.** Any "edge" the signals show gross is
   noise; the honest result is what survives *after* costs.

No real money ever moves and no live broker is ever contacted.
"""

from __future__ import annotations

__all__: list[str] = []
