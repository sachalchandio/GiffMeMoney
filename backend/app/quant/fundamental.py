"""Fundamental-quality scoring: Piotroski F-Score and Altman Z-Score.

These two models back the ``piotroski`` and ``altman-z`` fundamental strategies.
They consume the deterministic :class:`~app.market.universe.Fundamentals` record
attached to every asset seed, so they always have plausible inputs.

* :func:`piotroski_score` returns an integer ``0..9`` quality score built from
  nine sign-based accounting criteria (profitability, leverage/liquidity, and
  operating efficiency). Because the simulator exposes a single-period snapshot
  rather than a two-year panel, the year-over-year ("improvement") criteria are
  evaluated against sensible levels of the same snapshot (e.g. positive revenue
  growth, manageable leverage, healthy current ratio).
* :func:`altman_z` returns the classic manufacturing Z-Score, a weighted sum of
  five balance-sheet/market ratios where higher is safer.

Both functions are numerically defensive: zero/None/non-finite balance-sheet
denominators never raise — ratios degrade to ``0.0`` and outputs are clamped to
sane ranges. For crypto/ETF seeds (mostly-zero fundamentals) they return
neutral, non-distress values rather than crashing.
"""

from __future__ import annotations

import math

from app.market.universe import Fundamentals

__all__ = [
    "piotroski_score",
    "altman_z",
]


def _safe(value: float, default: float = 0.0) -> float:
    """Return ``value`` as a finite float, falling back to ``default``.

    Args:
        value: Candidate number (may be None-like / non-finite).
        default: Value substituted when ``value`` is NaN / +-inf.

    Returns:
        ``float(value)`` if finite, else ``default``.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Divide guarding against a zero / non-finite denominator.

    Formula:
        ratio = numerator / denominator   (when |denominator| > 0)

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Returned when the denominator is zero or the result is
            non-finite.

    Returns:
        The finite ratio, or ``default``.
    """
    n = _safe(numerator)
    d = _safe(denominator)
    if d == 0.0:
        return default
    r = n / d
    return r if math.isfinite(r) else default


def piotroski_score(f: Fundamentals) -> int:
    """Piotroski F-Score: a 0..9 integer fundamental-quality score.

    Awards one point for each of nine binary criteria. With a single-period
    snapshot the original year-over-year tests are approximated by level/sign
    tests on the same data:

    Profitability (4 points):
        1. ROA > 0                       (positive return on assets)
        2. Operating cash flow proxy > 0 (FCF per share > 0)
        3. ROA "improving"               (ROA exceeds a small positive hurdle)
        4. Accruals quality              (FCF per share >= EPS, i.e. earnings are
           backed by cash, not accruals)

    Leverage, liquidity & dilution (3 points):
        5. Lower leverage                (debt/equity < 1.0)
        6. Higher current ratio          (current ratio > 1.0)
        7. No dilution                   (revenue growth >= 0, growing the base
           rather than shrinking — a proxy for not issuing equity into decline)

    Operating efficiency (2 points):
        8. Higher gross/net margin       (net margin > 0)
        9. Higher asset turnover         (sales / total assets > 0)

    Args:
        f: A :class:`Fundamentals` record.

    Returns:
        The integer F-Score in ``[0, 9]``. Mostly-zero records (crypto/ETF)
        score low but never raise.
    """
    score = 0

    roa = _safe(f.roa)
    fcf_ps = _safe(f.fcf_per_share)
    eps = _safe(f.eps)
    de = _safe(f.debt_to_equity)
    current_ratio = _safe(f.current_ratio)
    rev_growth = _safe(f.revenue_growth)
    net_margin = _safe(f.net_margin)
    asset_turnover = _ratio(f.sales, f.total_assets)

    # --- Profitability ---
    if roa > 0.0:
        score += 1
    if fcf_ps > 0.0:
        score += 1
    if roa > 0.02:  # ROA comfortably positive ~ improving profitability
        score += 1
    if fcf_ps >= eps:  # cash backs earnings (quality of accruals)
        score += 1

    # --- Leverage / liquidity / dilution ---
    if 0.0 <= de < 1.0:
        score += 1
    if current_ratio > 1.0:
        score += 1
    if rev_growth >= 0.0:
        score += 1

    # --- Operating efficiency ---
    if net_margin > 0.0:
        score += 1
    if asset_turnover > 0.0:
        score += 1

    return int(min(9, max(0, score)))


def altman_z(f: Fundamentals, market_cap: float) -> float:
    """Altman Z-Score for bankruptcy/distress distance (manufacturing model).

    Formula:
        Z = 1.2 * (WC / TA)
          + 1.4 * (RE / TA)
          + 3.3 * (EBIT / TA)
          + 0.6 * (MktCap / TL)
          + 1.0 * (Sales / TA)

    where WC = working capital, TA = total assets, RE = retained earnings,
    EBIT = earnings before interest & taxes, TL = total liabilities, MktCap =
    market capitalisation. Interpretation: ``Z > 2.99`` = safe zone,
    ``1.81 <= Z <= 2.99`` = grey zone, ``Z < 1.81`` = distress zone.

    Args:
        f: A :class:`Fundamentals` record supplying WC, RE, EBIT, TA, TL, Sales.
        market_cap: Market capitalisation (equity market value) for the
            ``0.6 * MktCap / TL`` term.

    Returns:
        The Z-Score as a finite float, clamped to ``[-50, 50]`` to keep
        pathological tiny-denominator inputs bounded. When liabilities are zero
        (debt-free / crypto/ETF seeds) the leverage term contributes ``0.0``
        rather than blowing up, yielding a safe-zone-style score.
    """
    ta = _safe(f.total_assets)
    # Total assets is the dominant denominator; if it is non-positive the model
    # is undefined, so fall back to a neutral safe-zone value.
    if ta <= 0.0:
        return 3.0

    wc_ta = _ratio(f.working_capital, ta)
    re_ta = _ratio(f.retained_earnings, ta)
    ebit_ta = _ratio(f.ebit, ta)
    sales_ta = _ratio(f.sales, ta)
    # Market value of equity over total liabilities; 0 when debt-free.
    mktcap_tl = _ratio(market_cap, f.total_liabilities, default=0.0)

    z = (
        1.2 * wc_ta
        + 1.4 * re_ta
        + 3.3 * ebit_ta
        + 0.6 * mktcap_tl
        + 1.0 * sales_ta
    )
    z = _safe(z, 0.0)
    return min(50.0, max(-50.0, z))
