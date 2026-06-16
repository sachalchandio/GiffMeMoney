"""Value-at-Risk and Conditional VaR estimators.

Provides three VaR families plus CVaR, all expressed as **positive loss
fractions** (e.g. ``0.05`` means a 5% loss):

* historical (empirical quantile of realized returns),
* parametric (Normal / variance-covariance),
* Monte Carlo (Normal-return simulation),
* CVaR / Expected Shortfall (mean loss beyond the VaR threshold).

All functions are numerically defensive: empty/short inputs, zero variance, and
non-finite values never raise — they return safe finite defaults and losses are
floored at zero.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

__all__ = [
    "historical_var",
    "parametric_var",
    "cvar",
    "monte_carlo_var",
]

# Smallest volatility treated as non-zero.
_VOL_FLOOR: float = 1e-12

# Hard clamp: a single-period loss fraction never exceeds 100% (total loss) for
# the risk figures we report on a return series.
_LOSS_CLAMP: float = 1.0


def _clean_returns(returns: np.ndarray | list[float]) -> np.ndarray:
    """Coerce a returns input to a clean 1-D finite float array.

    Args:
        returns: Sequence of period returns.

    Returns:
        A 1-D ``float64`` array containing only finite entries (possibly empty).
    """
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def _clamp_loss(loss: float) -> float:
    """Floor a loss fraction at zero and clamp it to a sane maximum.

    Args:
        loss: Candidate loss fraction (positive = loss).

    Returns:
        ``loss`` clamped to ``[0, _LOSS_CLAMP]`` and guaranteed finite.
    """
    if not math.isfinite(loss):
        return 0.0
    return max(0.0, min(_LOSS_CLAMP, float(loss)))


def _alpha(conf: float) -> float:
    """Convert a confidence level to a lower-tail probability ``alpha``.

    Args:
        conf: Confidence level in ``(0, 1)`` (e.g. ``0.95``). Values outside the
            open interval are clamped to a safe band.

    Returns:
        ``alpha = 1 - conf`` clamped to ``[1e-6, 0.5]``.
    """
    c = float(conf)
    if not math.isfinite(c):
        c = 0.95
    c = min(0.999999, max(0.5, c))
    return 1.0 - c


def historical_var(
    returns: np.ndarray | list[float], conf: float = 0.95
) -> float:
    """Historical (empirical) Value at Risk as a positive loss fraction.

    Formula:
        VaR_conf = -Quantile_{1-conf}(returns)

    The lower-tail empirical quantile of the realized return distribution; the
    sign is flipped so a typical loss is reported as a positive number.

    Args:
        returns: Sequence of period returns.
        conf: Confidence level (default ``0.95`` -> 5% tail).

    Returns:
        The VaR as a positive loss fraction, floored at ``0.0``. Returns ``0.0``
        for empty input.
    """
    arr = _clean_returns(returns)
    if arr.size == 0:
        return 0.0
    alpha = _alpha(conf)
    q = float(np.quantile(arr, alpha))
    return _clamp_loss(-q)


def parametric_var(
    returns: np.ndarray | list[float], conf: float = 0.95
) -> float:
    """Parametric (Normal / variance-covariance) Value at Risk.

    Assumes returns are Normal with sample mean ``mu`` and std ``sigma``:

        VaR_conf = -(mu + z_{1-conf} * sigma) = -(mu - z_conf * sigma)

    where ``z_conf = Phi^{-1}(conf)`` (e.g. 1.645 for 95%). Reported as a
    positive loss fraction.

    Args:
        returns: Sequence of period returns.
        conf: Confidence level (default ``0.95``).

    Returns:
        The parametric VaR as a positive loss fraction, floored at ``0.0``.
        Returns ``0.0`` for empty input.
    """
    arr = _clean_returns(returns)
    if arr.size == 0:
        return 0.0
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    if not math.isfinite(sigma) or sigma < _VOL_FLOOR:
        # No dispersion: VaR is just the (possibly negative) mean loss.
        return _clamp_loss(-mu)
    z = float(norm.ppf(1.0 - _alpha(conf)))  # = z_conf, positive
    var = -(mu - z * sigma)
    return _clamp_loss(var)


def cvar(returns: np.ndarray | list[float], conf: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) as a positive loss fraction.

    Formula:
        threshold = Quantile_{1-conf}(returns)
        CVaR      = -E[ r | r <= threshold ]

    The mean of returns in the lower ``(1 - conf)`` tail, sign-flipped to a
    positive loss. By construction ``CVaR >= VaR``.

    Args:
        returns: Sequence of period returns.
        conf: Confidence level (default ``0.95``).

    Returns:
        The CVaR as a positive loss fraction, floored at ``0.0`` and never below
        the corresponding historical VaR. Returns ``0.0`` for empty input.
    """
    arr = _clean_returns(returns)
    if arr.size == 0:
        return 0.0
    alpha = _alpha(conf)
    threshold = float(np.quantile(arr, alpha))
    tail = arr[arr <= threshold]
    if tail.size == 0:
        es = -threshold
    else:
        es = -float(np.mean(tail))
    var = -threshold
    return _clamp_loss(max(es, var))


def monte_carlo_var(
    mu_daily: float,
    sigma_daily: float,
    conf: float = 0.95,
    sims: int = 10000,
    seed: int = 0,
) -> float:
    """Monte Carlo Value at Risk under a Normal one-period return model.

    Simulates ``sims`` one-period returns ``r ~ N(mu_daily, sigma_daily^2)`` and
    reports the empirical lower-tail quantile:

        VaR_conf = -Quantile_{1-conf}(simulated returns)

    Args:
        mu_daily: Mean daily return.
        sigma_daily: Daily volatility (floored to a tiny positive value).
        conf: Confidence level (default ``0.95``).
        sims: Number of simulated draws (coerced to >= 1).
        seed: RNG seed for reproducibility.

    Returns:
        The simulated VaR as a positive loss fraction, floored at ``0.0``.
    """
    mu = float(mu_daily) if math.isfinite(mu_daily) else 0.0
    sigma = float(sigma_daily) if math.isfinite(sigma_daily) else 0.0
    sigma = max(_VOL_FLOOR, sigma)
    n = max(1, int(sims))

    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    draws = rng.normal(loc=mu, scale=sigma, size=n)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return 0.0
    q = float(np.quantile(draws, _alpha(conf)))
    return _clamp_loss(-q)
