"""Real-Time mode — a live-feeling, multi-venue PAPER trading simulation.

HONESTY / SAFETY (this is a finance tool). Everything in this package is a
**SIMULATION on (accelerated) synthetic data**. It mimics what a real-time,
spread-across-many-venues bot would *feel* like — ticking forward, scoring
venues, rotating capital into the ones doing well, showing daily profit/loss —
so you can learn the mechanics with **zero real money at risk**.

It is deliberately honest, not flattering:

* **No real money moves.** No bank, no live broker, no withdrawals. $0 real.
* **Costs are charged on every trade** (spread + fee), so churn bleeds — just
  like real life.
* **The "prediction model" updates itself** as ticks arrive, but on data with no
  real edge it hovers near a coin flip, and it reports its own uncertainty.
* **Returns stay realistic.** Drift and volatility are plausible, so you will not
  see $20 become $4,000 — because nothing legitimate does that.

The capital-spreading mechanic matches the mental model: with more equity the
book spreads across more venues (hard-capped at 80), rotating toward winners.
"""

from __future__ import annotations

__all__: list[str] = []
