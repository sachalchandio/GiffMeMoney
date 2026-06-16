"""Black-Scholes-Merton option pricing, Greeks, and implied volatility.

Implements the standard closed-form European option price and its first-order
sensitivities (delta, gamma, vega, theta, rho), plus an implied-volatility solver
via Brent's method (with a robust bisection fallback). No dividend yield is
modeled (q = 0).

Reference value for self-check: ``black_scholes(100, 100, 1, 0.05, 0.2, 'call')``
≈ ``10.4506``.

All functions are numerically defensive: zero/negative time to expiry, zero
volatility, and non-finite inputs never raise — they collapse to the
no-uncertainty intrinsic value or safe finite defaults.
"""

from __future__ import annotations

import math

from scipy.optimize import brentq
from scipy.stats import norm

__all__ = [
    "black_scholes",
    "bs_greeks",
    "implied_vol",
]

# Smallest time / volatility treated as non-zero.
_T_FLOOR: float = 1e-12
_SIGMA_FLOOR: float = 1e-12

# Implied-vol search bounds (annualized) and tolerance.
_IV_LOW: float = 1e-6
_IV_HIGH: float = 5.0
_IV_TOL: float = 1e-8


def _norm(value: str) -> str:
    """Normalize an option-kind string to ``'call'`` or ``'put'``.

    Args:
        value: Raw kind string (case-insensitive; ``'c'``/``'p'`` accepted).

    Returns:
        ``'put'`` if the input clearly denotes a put, otherwise ``'call'``.
    """
    v = str(value).strip().lower()
    if v in ("put", "p"):
        return "put"
    return "call"


def _intrinsic(S: float, K: float, kind: str) -> float:
    """Undiscounted intrinsic value of the option (used when T or sigma -> 0).

    Args:
        S: Spot price.
        K: Strike price.
        kind: ``'call'`` or ``'put'``.

    Returns:
        ``max(S - K, 0)`` for a call, ``max(K - S, 0)`` for a put.
    """
    if kind == "put":
        return max(K - S, 0.0)
    return max(S - K, 0.0)


def _d1_d2(
    S: float, K: float, T: float, r: float, sigma: float
) -> tuple[float, float]:
    """Compute the Black-Scholes ``d1`` and ``d2`` terms.

    Formula:
        d1 = (ln(S/K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Args:
        S: Spot price (positive).
        K: Strike price (positive).
        T: Time to expiry in years (positive).
        r: Continuously-compounded risk-free rate (annual).
        sigma: Volatility (annualized, positive).

    Returns:
        ``(d1, d2)`` as floats.
    """
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def black_scholes(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
) -> float:
    """Price a European option with the Black-Scholes-Merton formula.

    Formula (no dividends, q = 0):
        Call = S * Phi(d1) - K * exp(-r T) * Phi(d2)
        Put  = K * exp(-r T) * Phi(-d2) - S * Phi(-d1)
        d1   = (ln(S/K) + (r + sigma^2 / 2) T) / (sigma sqrt(T))
        d2   = d1 - sigma sqrt(T)

    When ``T <= 0`` or ``sigma <= 0`` (no uncertainty) the price collapses to the
    discounted intrinsic value.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Continuously-compounded annual risk-free rate.
        sigma: Annualized volatility.
        kind: ``'call'`` (default) or ``'put'``.

    Returns:
        The non-negative option premium. Non-finite inputs / degenerate cases
        return the (discounted) intrinsic value or ``0.0``.
    """
    k = _norm(kind)
    Sf = float(S)
    Kf = float(K)
    Tf = float(T)
    rf = float(r)
    sig = float(sigma)

    if not all(math.isfinite(x) for x in (Sf, Kf, Tf, rf, sig)):
        return max(_intrinsic(Sf if math.isfinite(Sf) else 0.0,
                              Kf if math.isfinite(Kf) else 0.0, k), 0.0)

    if Sf <= 0.0:
        # A worthless underlying: call -> 0, put -> discounted strike.
        if k == "put" and Kf > 0.0 and Tf > 0.0 and math.isfinite(rf):
            return Kf * math.exp(-rf * Tf)
        return max(_intrinsic(max(Sf, 0.0), Kf, k), 0.0)
    if Kf <= 0.0:
        # Zero strike: call worth the spot, put worthless.
        return Sf if k == "call" else 0.0

    if Tf <= _T_FLOOR or sig <= _SIGMA_FLOOR:
        # No time value left; discount the intrinsic payoff.
        disc = math.exp(-rf * max(Tf, 0.0)) if math.isfinite(rf) else 1.0
        if k == "call":
            return max(Sf - Kf * disc, 0.0)
        return max(Kf * disc - Sf, 0.0)

    d1, d2 = _d1_d2(Sf, Kf, Tf, rf, sig)
    disc = math.exp(-rf * Tf)
    if k == "call":
        price = Sf * norm.cdf(d1) - Kf * disc * norm.cdf(d2)
    else:
        price = Kf * disc * norm.cdf(-d2) - Sf * norm.cdf(-d1)

    if not math.isfinite(price):
        return max(_intrinsic(Sf, Kf, k), 0.0)
    return max(0.0, float(price))


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    kind: str = "call",
) -> dict:
    """First-order Black-Scholes Greeks (no dividends).

    Formulas (with ``phi`` = standard-normal pdf, ``Phi`` = cdf):
        delta_call =  Phi(d1)            delta_put = Phi(d1) - 1
        gamma      =  phi(d1) / (S sigma sqrt(T))
        vega       =  S phi(d1) sqrt(T)                       (per 1.00 vol)
        theta_call = -S phi(d1) sigma / (2 sqrt(T)) - r K e^{-rT} Phi(d2)
        theta_put  = -S phi(d1) sigma / (2 sqrt(T)) + r K e^{-rT} Phi(-d2)
        rho_call   =  K T e^{-rT} Phi(d2)
        rho_put    = -K T e^{-rT} Phi(-d2)

    ``vega`` is per +1.00 (100 vol-points) change in sigma and ``theta`` is the
    per-year time decay (annualized). Degenerate inputs (``T <= 0`` /
    ``sigma <= 0``) yield zeroed sensitivities with a step-function delta.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Continuously-compounded annual risk-free rate.
        sigma: Annualized volatility.
        kind: ``'call'`` (default) or ``'put'``.

    Returns:
        A dict ``{'delta', 'gamma', 'vega', 'theta', 'rho'}`` of finite floats.
    """
    k = _norm(kind)
    Sf = float(S)
    Kf = float(K)
    Tf = float(T)
    rf = float(r)
    sig = float(sigma)

    zero = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    if not all(math.isfinite(x) for x in (Sf, Kf, Tf, rf, sig)):
        return dict(zero)
    if Sf <= 0.0 or Kf <= 0.0 or Tf <= _T_FLOOR or sig <= _SIGMA_FLOOR:
        # No optionality: delta is a step function on intrinsic; rest are ~0.
        if k == "call":
            delta = 1.0 if Sf > Kf else 0.0
        else:
            delta = -1.0 if Sf < Kf else 0.0
        out = dict(zero)
        out["delta"] = delta
        return out

    d1, d2 = _d1_d2(Sf, Kf, Tf, rf, sig)
    pdf_d1 = float(norm.pdf(d1))
    sqrt_t = math.sqrt(Tf)
    disc = math.exp(-rf * Tf)

    gamma = pdf_d1 / (Sf * sig * sqrt_t)
    vega = Sf * pdf_d1 * sqrt_t

    if k == "call":
        delta = float(norm.cdf(d1))
        theta = (
            -(Sf * pdf_d1 * sig) / (2.0 * sqrt_t)
            - rf * Kf * disc * float(norm.cdf(d2))
        )
        rho = Kf * Tf * disc * float(norm.cdf(d2))
    else:
        delta = float(norm.cdf(d1)) - 1.0
        theta = (
            -(Sf * pdf_d1 * sig) / (2.0 * sqrt_t)
            + rf * Kf * disc * float(norm.cdf(-d2))
        )
        rho = -Kf * Tf * disc * float(norm.cdf(-d2))

    out = {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }
    return {key: (float(val) if math.isfinite(val) else 0.0) for key, val in out.items()}


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    kind: str = "call",
) -> float:
    """Solve for Black-Scholes implied volatility from a market price.

    Inverts the pricing formula by finding the ``sigma`` that makes the model
    price equal the observed ``price``. Uses Brent's method (``scipy.optimize.brentq``)
    on the root function ``f(sigma) = BS(sigma) - price`` over
    ``[_IV_LOW, _IV_HIGH]``; if the price lies outside the bracketed range or
    Brent fails, falls back to bisection / boundary clamping.

    Args:
        price: Observed option premium.
        S: Spot price.
        K: Strike price.
        T: Time to expiry in years.
        r: Continuously-compounded annual risk-free rate.
        kind: ``'call'`` (default) or ``'put'``.

    Returns:
        The implied annualized volatility, clamped to ``[_IV_LOW, _IV_HIGH]``.
        Returns ``0.0`` when no positive time value is present (price at/below
        intrinsic) or inputs are non-finite/degenerate.
    """
    k = _norm(kind)
    p = float(price)
    Sf = float(S)
    Kf = float(K)
    Tf = float(T)
    rf = float(r)

    if not all(math.isfinite(x) for x in (p, Sf, Kf, Tf, rf)):
        return 0.0
    if Sf <= 0.0 or Kf <= 0.0 or Tf <= _T_FLOOR or p <= 0.0:
        return 0.0

    # No-arbitrage bounds: price must exceed (discounted) intrinsic and be below
    # the underlying (call) / strike (put). Outside that, clamp.
    disc = math.exp(-rf * Tf)
    if k == "call":
        lower_bound = max(Sf - Kf * disc, 0.0)
        upper_bound = Sf
    else:
        lower_bound = max(Kf * disc - Sf, 0.0)
        upper_bound = Kf * disc

    if p <= lower_bound + 1e-12:
        # At/below intrinsic -> zero (or near-zero) implied vol.
        return _IV_LOW if p > lower_bound else 0.0
    if p >= upper_bound:
        return _IV_HIGH

    def _objective(sig: float) -> float:
        return black_scholes(Sf, Kf, Tf, rf, sig, k) - p

    f_low = _objective(_IV_LOW)
    f_high = _objective(_IV_HIGH)

    # If the root is not bracketed, return the nearer boundary.
    if f_low * f_high > 0.0:
        return _IV_LOW if abs(f_low) < abs(f_high) else _IV_HIGH

    try:
        iv = brentq(_objective, _IV_LOW, _IV_HIGH, xtol=_IV_TOL, maxiter=200)
        if math.isfinite(iv):
            return max(_IV_LOW, min(_IV_HIGH, float(iv)))
    except Exception:
        pass

    # Bisection fallback.
    lo, hi = _IV_LOW, _IV_HIGH
    flo = _objective(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = _objective(mid)
        if not math.isfinite(fmid):
            break
        if abs(fmid) < _IV_TOL or (hi - lo) < _IV_TOL:
            return max(_IV_LOW, min(_IV_HIGH, mid))
        if flo * fmid <= 0.0:
            hi = mid
        else:
            lo = mid
            flo = fmid
    return max(_IV_LOW, min(_IV_HIGH, 0.5 * (lo + hi)))
