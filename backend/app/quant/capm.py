"""Capital Asset Pricing Model (CAPM) expected-return helper.

Implements the Security Market Line: the expected return on an asset equals the
risk-free rate plus its market beta times the equity market risk premium.
"""

from __future__ import annotations

import math

__all__ = ["capm_expected_return"]

# Clamp the output to a sane band so a pathological beta / premium can never
# emit absurd expected returns into downstream signals.
_RET_CLAMP: float = 100.0


def capm_expected_return(
    beta: float,
    rf_annual: float,
    market_premium_annual: float,
) -> float:
    """Compute the CAPM (Security Market Line) expected annual return.

    Formula:
        E[R] = Rf + beta * (E[Rm] - Rf)
             = rf_annual + beta * market_premium_annual

    where ``market_premium_annual = E[Rm] - Rf`` is the equity market risk
    premium. With ``beta == 1`` the identity collapses to
    ``E[R] = rf_annual + market_premium_annual`` (the expected market return),
    and with ``beta == 0`` to ``E[R] = rf_annual``.

    Args:
        beta: The asset's market beta (sensitivity to the market factor).
        rf_annual: Annual risk-free rate as a decimal (e.g. ``0.04``).
        market_premium_annual: Annual market risk premium ``E[Rm] - Rf`` as a
            decimal (e.g. ``0.06``).

    Returns:
        The CAPM expected annual return as a decimal. Non-finite inputs are
        treated as ``0.0`` and the result is clamped to ``[-100, 100]`` so it
        stays finite and sane.
    """
    b = float(beta) if math.isfinite(beta) else 0.0
    rf = float(rf_annual) if math.isfinite(rf_annual) else 0.0
    premium = float(market_premium_annual) if math.isfinite(market_premium_annual) else 0.0

    result = rf + b * premium
    if not math.isfinite(result):
        return 0.0
    return max(-_RET_CLAMP, min(_RET_CLAMP, result))
