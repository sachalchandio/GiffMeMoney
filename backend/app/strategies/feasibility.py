"""Honesty helpers: the synthetic-data note + an impossible-target guard.

HONESTY / SAFETY (this is a finance tool). Everything the app produces is a
**SIMULATION on synthetic data** — there is no real market edge to harvest and
nothing here predicts real-world results. Two small, dependency-free helpers let
every advice / backtest surface that plainly:

    * :data:`SYNTHETIC_DATA_NOTE` — the one-line note the API/UI shows so a
      reader can never mistake a simulated result for a real, tradable forecast.
    * :func:`feasibility_warning` — flags a goal whose implied compounding is
      physically extreme (e.g. "turn $100 into $10,000 in two months"). It
      returns a plain-English warning string for such asks and ``None`` for
      sane ones, so the caller can refuse to imply the impossible is achievable.

The guard is deliberately conservative and never raises: any non-finite or
nonsensical input simply yields ``None`` (no warning) rather than blowing up a
request.
"""

from __future__ import annotations

import math

__all__ = [
    "SYNTHETIC_DATA_NOTE",
    "DAILY_COMPOUNDING_WARN_PCT",
    "feasibility_warning",
]

#: The mandatory plain-English honesty note surfaced alongside any advice or
#: backtest. It states the results are simulated on synthetic data and are not a
#: real forecast — never implying guaranteed or even achievable real profit.
SYNTHETIC_DATA_NOTE: str = (
    "Results are a simulation on synthetic (made-up) market data — not a real "
    "forecast and not financial advice. There is no real-world edge here; past "
    "simulated performance does not predict real results."
)

#: Implied per-day compounding rate (in percent) above which a target is treated
#: as physically extreme / infeasible. ~3%/day compounds to roughly 9,500x over
#: a single year — far beyond anything a real (or this simulated) strategy does,
#: so anything at/above this is flagged as a fantasy target.
DAILY_COMPOUNDING_WARN_PCT: float = 3.0


def feasibility_warning(
    amount: float,
    target_amount: float,
    horizon_days: float,
) -> str | None:
    """Warn when a savings/return goal implies physically extreme compounding.

    Computes the constant daily growth rate that would be required to turn
    ``amount`` into ``target_amount`` over ``horizon_days``::

        daily_rate = (target_amount / amount) ** (1 / horizon_days) - 1

    and returns a plain-English warning when that rate is at or above
    :data:`DAILY_COMPOUNDING_WARN_PCT` percent per day (an ask no real or
    simulated strategy can deliver). Sane goals — and any malformed / non-finite
    input — return ``None``.

    Args:
        amount: The starting amount in dollars (must be ``> 0`` to assess a
            multiple).
        target_amount: The desired ending amount in dollars.
        horizon_days: The number of (calendar/trading) days to reach the target.

    Returns:
        A warning string when the implied daily compounding is extreme, else
        ``None``.
    """
    try:
        a = float(amount)
        target = float(target_amount)
        days = float(horizon_days)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(a) and math.isfinite(target) and math.isfinite(days)):
        return None
    # Nothing to assess: non-positive start, non-positive horizon, or a target
    # that is not an increase over the start.
    if a <= 0.0 or days <= 0.0 or target <= a:
        return None

    multiple = target / a
    try:
        daily_rate = multiple ** (1.0 / days) - 1.0
    except (OverflowError, ValueError):  # pragma: no cover - defensive
        return None
    if not math.isfinite(daily_rate):
        return None

    daily_pct = daily_rate * 100.0
    if daily_pct < DAILY_COMPOUNDING_WARN_PCT:
        return None

    return (
        f"Turning ${a:,.0f} into ${target:,.0f} in {days:,.0f} day(s) implies "
        f"~{daily_pct:,.1f}% growth EVERY day ({multiple:,.0f}x overall) — that "
        "is not achievable by any real or simulated strategy. Treat this target "
        "as a fantasy, not a plan."
    )
