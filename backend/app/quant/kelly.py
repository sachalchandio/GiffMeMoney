"""Kelly-criterion position sizing for a continuous (Gaussian) return process.

For a normally-distributed excess return with daily drift ``mu`` and daily
volatility ``sigma`` (so variance ``sigma^2``), the growth-optimal fraction of
capital to allocate is the continuous Kelly fraction ``f* = mu / sigma^2``.
"""

from __future__ import annotations

import math

__all__ = ["kelly_fraction"]

# Smallest variance treated as non-zero; below this the optimal fraction is
# undefined (infinite leverage), so we collapse to 0.
_VAR_FLOOR: float = 1e-12

# Clamp the Kelly fraction to a sane leverage band: never short more than the
# whole book, never lever beyond 3x. Matches the contract's [-1, 3] range.
_KELLY_MIN: float = -1.0
_KELLY_MAX: float = 3.0


def kelly_fraction(mu_daily: float, sigma_daily: float) -> float:
    """Compute the growth-optimal (continuous) Kelly fraction.

    Formula:
        f* = mu_daily / sigma_daily^2

    This is the continuous-time Kelly result for a Gaussian return process: the
    capital fraction that maximizes the expected logarithmic growth rate. A
    positive drift gives a long fraction, a negative drift a short fraction, and
    higher variance shrinks the position.

    Args:
        mu_daily: Mean daily (excess) return / drift.
        sigma_daily: Daily return volatility (standard deviation). Its square is
            the variance used in the denominator.

    Returns:
        The Kelly fraction, clamped to ``[-1, 3]`` (no more than a full short,
        no more than 3x leverage). Returns ``0.0`` for non-finite inputs or when
        the variance is effectively zero (the fraction would diverge).
    """
    mu = float(mu_daily) if math.isfinite(mu_daily) else 0.0
    sigma = float(sigma_daily) if math.isfinite(sigma_daily) else 0.0

    variance = sigma * sigma
    if variance < _VAR_FLOOR:
        return 0.0

    f = mu / variance
    if not math.isfinite(f):
        return 0.0
    return max(_KELLY_MIN, min(_KELLY_MAX, f))
