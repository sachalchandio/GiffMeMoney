"""Fama-French three-factor model estimated by ordinary least squares.

Regresses an asset's *excess* daily returns on the three Fama-French factors
(market excess return, SMB, HML) and reports the annualized intercept (alpha),
the three factor loadings, and the regression R-squared.

The OLS solution is computed directly via :func:`numpy.linalg.lstsq` (the
normal-equation least-squares solver) — no statsmodels dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.quant.returns import annualize_return

__all__ = ["FF3Result", "fama_french_3factor"]

# Minimum observations required to identify 4 parameters (intercept + 3 betas)
# with at least one residual degree of freedom.
_MIN_OBS: int = 5


@dataclass(frozen=True)
class FF3Result:
    """Result of a Fama-French three-factor OLS regression.

    Attributes:
        alpha_annual: Annualized intercept (Jensen-style alpha), geometrically
            compounded from the daily intercept via
            :func:`app.quant.returns.annualize_return`. A decimal (e.g. ``0.03``
            for 3%).
        beta_mkt: Loading on the market excess-return factor (Mkt-Rf).
        beta_smb: Loading on the size factor (small-minus-big).
        beta_hml: Loading on the value factor (high-minus-low).
        r2: Coefficient of determination of the regression, in ``[0, 1]``.
    """

    alpha_annual: float
    beta_mkt: float
    beta_smb: float
    beta_hml: float
    r2: float


def _clean(arr: np.ndarray | list[float]) -> np.ndarray:
    """Coerce an input to a 1-D float array (non-finite entries kept as-is here).

    Args:
        arr: Sequence of numbers.

    Returns:
        A 1-D ``float64`` array.
    """
    return np.asarray(arr, dtype=np.float64).ravel()


def fama_french_3factor(
    asset_excess_ret: np.ndarray | list[float],
    mkt: np.ndarray | list[float],
    smb: np.ndarray | list[float],
    hml: np.ndarray | list[float],
) -> FF3Result:
    """Estimate the Fama-French three-factor model by OLS.

    Model:
        r_excess_t = alpha + b_mkt * Mkt_t + b_smb * SMB_t + b_hml * HML_t + e_t

    The coefficients ``[alpha, b_mkt, b_smb, b_hml]`` are obtained as the
    least-squares solution of ``X @ coef = y`` with design matrix
    ``X = [1, Mkt, SMB, HML]`` via :func:`numpy.linalg.lstsq`. The R-squared is::

        R^2 = 1 - SS_res / SS_tot
        SS_res = sum( (y - X@coef)^2 )
        SS_tot = sum( (y - mean(y))^2 )

    The daily intercept is annualized with geometric compounding
    (``exp(alpha_daily * 252) - 1``).

    All four series are trailing-aligned to their common length and rows with any
    non-finite value are dropped before fitting.

    Args:
        asset_excess_ret: Asset returns in excess of the risk-free rate (daily).
        mkt: Market excess-return factor series (Mkt-Rf, daily).
        smb: Size factor series (small-minus-big, daily).
        hml: Value factor series (high-minus-low, daily).

    Returns:
        An :class:`FF3Result`. If there are too few aligned observations, the
        design is rank-deficient, or the solve fails, a safe default is returned
        (``alpha_annual=0``, all betas ``0`` except a market-neutral
        ``beta_mkt=1`` is *not* assumed here — every loading defaults to ``0`` —
        and ``r2=0``).
    """
    y = _clean(asset_excess_ret)
    x1 = _clean(mkt)
    x2 = _clean(smb)
    x3 = _clean(hml)

    n = min(y.size, x1.size, x2.size, x3.size)
    if n < _MIN_OBS:
        return FF3Result(0.0, 0.0, 0.0, 0.0, 0.0)

    # Trailing-align so the most recent observations match across series.
    y = y[-n:]
    x1 = x1[-n:]
    x2 = x2[-n:]
    x3 = x3[-n:]

    # Keep only rows where every value is finite.
    mask = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2) & np.isfinite(x3)
    y = y[mask]
    x1 = x1[mask]
    x2 = x2[mask]
    x3 = x3[mask]
    if y.size < _MIN_OBS:
        return FF3Result(0.0, 0.0, 0.0, 0.0, 0.0)

    design = np.column_stack([np.ones_like(y), x1, x2, x3])

    try:
        coef, _residuals, rank, _sv = np.linalg.lstsq(design, y, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        return FF3Result(0.0, 0.0, 0.0, 0.0, 0.0)

    if rank < design.shape[1] or coef.size != 4 or not np.all(np.isfinite(coef)):
        # Rank-deficient (e.g. collinear / constant factors): coefficients are
        # not uniquely identified, so fall back to a safe default.
        return FF3Result(0.0, 0.0, 0.0, 0.0, 0.0)

    alpha_daily = float(coef[0])
    b_mkt = float(coef[1])
    b_smb = float(coef[2])
    b_hml = float(coef[3])

    fitted = design @ coef
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot < 1e-18:
        # y is (near) constant: any variance explained is meaningless.
        r2 = 0.0
    else:
        r2 = 1.0 - ss_res / ss_tot
    if not math.isfinite(r2):
        r2 = 0.0
    r2 = min(1.0, max(0.0, r2))

    alpha_annual = annualize_return(alpha_daily) if math.isfinite(alpha_daily) else 0.0

    return FF3Result(
        alpha_annual=alpha_annual if math.isfinite(alpha_annual) else 0.0,
        beta_mkt=b_mkt if math.isfinite(b_mkt) else 0.0,
        beta_smb=b_smb if math.isfinite(b_smb) else 0.0,
        beta_hml=b_hml if math.isfinite(b_hml) else 0.0,
        r2=r2,
    )
