"""Intrinsic-value valuation models: two-stage DCF and the Gordon DDM.

Both functions return a per-share intrinsic value (in the asset's reporting
currency) and are numerically defensive: degenerate inputs (non-positive cash
flows, ``wacc <= growth``, ``r <= g``) collapse to a safe finite value rather
than raising or returning ``inf``.
"""

from __future__ import annotations

import math

__all__ = ["dcf_intrinsic_value", "gordon_ddm"]

# Upper bound on a per-share intrinsic value so a near-singular denominator
# (discount rate barely above the growth rate) cannot emit absurd numbers.
_VALUE_CLAMP: float = 1.0e9

# Minimum spread between discount rate and growth required for a finite,
# well-behaved perpetuity / terminal value.
_MIN_SPREAD: float = 1e-6


def dcf_intrinsic_value(
    fcf_per_share: float,
    growth: float,
    wacc: float,
    terminal_growth: float = 0.025,
    years: int = 10,
) -> float:
    """Two-stage discounted-cash-flow intrinsic value per share.

    Stage 1 grows the most recent free cash flow per share at ``growth`` for
    ``years`` periods and discounts each at the weighted-average cost of capital.
    Stage 2 capitalizes the year-``years`` cash flow into a Gordon terminal value
    and discounts it back:

        FCF_t = fcf_per_share * (1 + growth)^t                 for t = 1..N
        PV(stage 1) = sum_{t=1}^{N} FCF_t / (1 + wacc)^t
        TV_N = FCF_N * (1 + terminal_growth) / (wacc - terminal_growth)
        PV(TV) = TV_N / (1 + wacc)^N
        intrinsic = PV(stage 1) + PV(TV)

    For a flat cash flow (``growth == 0``, ``terminal_growth == 0``) the result
    is the standard growing-annuity-plus-perpetuity value
    ``FCF * [1 - (1+wacc)^-N]/wacc + (FCF/wacc)/(1+wacc)^N = FCF / wacc``,
    i.e. a flat perpetuity equals ``FCF / wacc``.

    Args:
        fcf_per_share: Trailing free cash flow per share (the stage-1 base, year
            0). Non-positive values yield ``0.0``.
        growth: Stage-1 annual FCF growth rate (decimal).
        wacc: Weighted-average cost of capital / discount rate (decimal). Must
            exceed ``terminal_growth`` for a finite terminal value; otherwise the
            terminal value is dropped (stage-1 PV only) to stay finite.
        terminal_growth: Perpetual growth rate after the explicit horizon.
            Defaults to ``0.025``.
        years: Number of explicit forecast years (stage 1). Defaults to ``10``.
            Non-positive values yield ``0.0``.

    Returns:
        The per-share intrinsic value, clamped to ``[0, 1e9]`` and guaranteed
        finite.
    """
    fcf0 = float(fcf_per_share) if math.isfinite(fcf_per_share) else 0.0
    g = float(growth) if math.isfinite(growth) else 0.0
    r = float(wacc) if math.isfinite(wacc) else 0.0
    tg = float(terminal_growth) if math.isfinite(terminal_growth) else 0.0
    n = int(years)

    if fcf0 <= 0.0 or n <= 0:
        return 0.0
    # A non-positive discount rate makes the present-value math meaningless.
    if r <= 0.0:
        return min(_VALUE_CLAMP, max(0.0, fcf0 * n))

    discount = 1.0 + r
    pv_stage1 = 0.0
    fcf_t = fcf0
    for t in range(1, n + 1):
        fcf_t = fcf0 * (1.0 + g) ** t
        pv_stage1 += fcf_t / discount**t

    # Stage-2 terminal value (Gordon growth on the final explicit cash flow).
    # ``fcf_t`` now holds FCF in year N.
    pv_terminal = 0.0
    spread = r - tg
    if spread > _MIN_SPREAD:
        terminal_value = fcf_t * (1.0 + tg) / spread
        pv_terminal = terminal_value / discount**n

    intrinsic = pv_stage1 + pv_terminal
    if not math.isfinite(intrinsic):
        return _VALUE_CLAMP
    return max(0.0, min(_VALUE_CLAMP, intrinsic))


def gordon_ddm(dividend: float, required_return: float, growth: float) -> float:
    """Gordon dividend-discount-model (constant-growth) fair value per share.

    Formula:
        P = D_1 / (r - g) = dividend * (1 + g) / (required_return - growth)

    where ``dividend`` is the most recent (year-0) dividend, so the next-period
    dividend is ``D_1 = dividend * (1 + g)``.

    Args:
        dividend: Most recent annual dividend per share (year 0). Non-positive
            values yield ``0.0`` (a non-dividend payer has no DDM value).
        required_return: Investor's required annual rate of return ``r``
            (decimal).
        growth: Perpetual dividend growth rate ``g`` (decimal).

    Returns:
        The Gordon fair value per share, clamped to ``[0, 1e9]`` and guaranteed
        finite. Returns ``0.0`` when ``required_return <= growth`` (the model is
        undefined / divergent there).
    """
    d0 = float(dividend) if math.isfinite(dividend) else 0.0
    r = float(required_return) if math.isfinite(required_return) else 0.0
    g = float(growth) if math.isfinite(growth) else 0.0

    if d0 <= 0.0:
        return 0.0
    spread = r - g
    if spread <= _MIN_SPREAD:
        # r <= g: the perpetuity diverges; guard rather than return inf.
        return 0.0

    price = d0 * (1.0 + g) / spread
    if not math.isfinite(price):
        return _VALUE_CLAMP
    return max(0.0, min(_VALUE_CLAMP, price))
