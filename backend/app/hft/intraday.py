"""Deterministic intraday bar generator for the simulation lab.

Short-horizon trading lives or dies on intraday dynamics — clustered volatility
(calm stretches punctuated by bursts), fat tails (occasional violent bars), and
near-zero drift (intraday price is mostly a random walk plus microstructure
noise). This module synthesises exactly that, deterministically, so a backtest is
reproducible across runs and processes.

Model (per bar ``t``)::

    sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}   (GARCH(1,1))
    eps_t    = sigma_t * z_t,   z_t ~ standardised Student-t(df)   (fat tails)
    r_t      = mu_bar + eps_t
    P_t      = P_{t-1} * (1 + clip(r_t))

The GARCH(1,1) recursion produces realistic **volatility clustering**; the
Student-t shocks give **fat tails**; ``mu_bar`` is a deliberately tiny per-bar
drift (intraday is ~driftless — an honest stance, not a rigged uptrend). The
whole path is seeded from the instrument name so it is identical every run.

HONESTY: this is **synthetic data with no real edge**. It mimics the *statistical
texture* of intraday markets so cost/turnover effects are realistic, but it does
not contain any predictable pattern a strategy could legitimately exploit — which
is itself the honest lesson.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

__all__ = ["IntradaySeries", "generate_intraday", "DEFAULT_BARS_PER_DAY"]

# Trading days per year (matches app.quant.returns.TRADING_DAYS without importing
# the heavier module — kept local so the lab has no quant dependency).
_TRADING_DAYS: int = 252

#: Default number of bars per trading day (~6.5h session in ~5-minute bars ≈ 78;
#: we use 78 so "bar" reads as a few-minute candle, the fastest a web app could
#: realistically act on — honestly far from microseconds).
DEFAULT_BARS_PER_DAY: int = 78

# GARCH(1,1) parameters: high persistence (alpha + beta ≈ 0.98) gives realistic
# volatility clustering without exploding.
_GARCH_ALPHA: float = 0.08
_GARCH_BETA: float = 0.90

# Degrees of freedom for the Student-t innovations (lower ⇒ fatter tails).
_T_DOF: float = 5.0

# Per-bar simple-return clip so a single fat-tailed shock can't break the path.
_RET_CLIP: float = 0.25

# Smallest price ever emitted.
_PRICE_FLOOR: float = 1e-6


def _seed_for(name: str, offset: int = 0) -> int:
    """Return a stable 32-bit RNG seed derived from an instrument name."""
    key = f"{name.strip().upper()}::hft".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return (int(digest, 16) + int(offset)) % (2**32)


def _annual_vol_for(symbol: str) -> float:
    """Pick a plausible annual volatility for an instrument.

    If the symbol is a known universe asset its seeded idiosyncratic vol nudges
    the figure (crypto-like names end up more volatile); unknown names get a
    sensible mid default. Always returns a finite value in a sane band.

    Args:
        symbol: Instrument name (case-insensitive).

    Returns:
        An annualised volatility (decimal) in ``[0.12, 1.20]``.
    """
    base = 0.40
    try:  # Reuse the universe seed when the symbol is real (kept optional).
        from app.market.universe import get_seed

        seed = get_seed(symbol)
        idio = float(getattr(seed, "idio_vol", 0.0))
        beta = float(getattr(seed, "market_beta", 1.0))
        base = 0.16 * max(beta, 0.1) + 1.2 * max(idio, 0.0) + 0.06
    except Exception:
        base = 0.40
    if not math.isfinite(base):
        base = 0.40
    return float(min(max(base, 0.12), 1.20))


@dataclass(frozen=True)
class IntradaySeries:
    """A deterministic intraday price path with realistic microstructure.

    Attributes:
        symbol: The instrument name the path was generated for.
        prices: Strictly-positive ``float64`` price array of length
            ``days * bars_per_day + 1`` (the leading entry is the start price).
        bars_per_day: Number of bars per trading day.
        days: Number of trading days spanned.
        bar_seconds: Nominal wall-clock seconds represented by one bar (for
            display only — the sim is bar-discrete, not literally real-time).
        annual_vol: The annualised volatility the path was calibrated to.
    """

    symbol: str
    prices: np.ndarray
    bars_per_day: int
    days: int
    bar_seconds: int
    annual_vol: float

    @property
    def bars_per_year(self) -> int:
        """Number of bars in a trading year (for annualising bar metrics)."""
        return int(_TRADING_DAYS * max(1, self.bars_per_day))

    @property
    def n_bars(self) -> int:
        """Number of return steps (``len(prices) - 1``)."""
        return int(self.prices.size - 1)


def generate_intraday(
    symbol: str,
    days: int = 30,
    bars_per_day: int = DEFAULT_BARS_PER_DAY,
    base_price: float = 100.0,
    annual_drift: float | None = None,
    annual_vol: float | None = None,
    seed_offset: int = 0,
) -> IntradaySeries:
    """Generate a deterministic intraday price path for an instrument.

    Args:
        symbol: Instrument name (drives the seed and the default vol).
        days: Trading days to span (clamped to ``[1, 250]``).
        bars_per_day: Bars per day (clamped to ``[1, 390]``).
        base_price: Strictly-positive starting price.
        annual_drift: Optional annual drift override (decimal). Defaults to a
            tiny positive drift — intraday is ~driftless by design.
        annual_vol: Optional annual vol override; defaults to a per-symbol value.
        seed_offset: Offset mixed into the seed (for drawing independent paths).

    Returns:
        A populated :class:`IntradaySeries` (strictly positive, finite prices).
    """
    d = int(min(max(int(days), 1), 250))
    bpd = int(min(max(int(bars_per_day), 1), 390))
    n = d * bpd
    bars_per_year = _TRADING_DAYS * bpd

    start = float(base_price) if math.isfinite(base_price) and base_price > 0 else 100.0
    ann_vol = float(annual_vol) if annual_vol is not None else _annual_vol_for(symbol)
    ann_vol = float(min(max(ann_vol, 0.05), 2.0)) if math.isfinite(ann_vol) else 0.40
    ann_drift = 0.08 if annual_drift is None else float(annual_drift)
    if not math.isfinite(ann_drift):
        ann_drift = 0.0

    # Per-bar moments. Drift is damped hard: intraday edge is ~0 by design.
    mu_bar = ann_drift / float(bars_per_year)
    sigma_bar = ann_vol / math.sqrt(float(bars_per_year))
    base_var = sigma_bar * sigma_bar

    rng = np.random.default_rng(_seed_for(symbol, seed_offset))

    # Standardised Student-t innovations (unit variance) for fat tails.
    dof = _T_DOF
    raw_t = rng.standard_t(dof, size=n)
    # Variance of a t with dof>2 is dof/(dof-2); rescale to unit variance.
    scale = math.sqrt((dof - 2.0) / dof) if dof > 2.0 else 1.0
    z = raw_t * scale

    # GARCH(1,1) recursion for clustered volatility.
    omega = base_var * (1.0 - _GARCH_ALPHA - _GARCH_BETA)
    omega = max(omega, base_var * 1e-3)  # keep strictly positive
    sigma2 = np.empty(n, dtype=np.float64)
    eps = np.empty(n, dtype=np.float64)
    prev_sigma2 = base_var
    prev_eps2 = base_var
    for i in range(n):
        s2 = omega + _GARCH_ALPHA * prev_eps2 + _GARCH_BETA * prev_sigma2
        if not math.isfinite(s2) or s2 <= 0.0:
            s2 = base_var
        sigma2[i] = s2
        e = math.sqrt(s2) * float(z[i])
        eps[i] = e
        prev_sigma2 = s2
        prev_eps2 = e * e

    rets = mu_bar + eps
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
    rets = np.clip(rets, -_RET_CLIP, _RET_CLIP)

    growth = np.concatenate(([1.0], np.cumprod(1.0 + rets)))
    prices = start * growth
    prices = np.nan_to_num(prices, nan=start, posinf=start, neginf=_PRICE_FLOOR)
    prices = np.maximum(prices, _PRICE_FLOOR)

    return IntradaySeries(
        symbol=str(symbol).strip().upper() or "SYNTH",
        prices=prices,
        bars_per_day=bpd,
        days=d,
        bar_seconds=int(max(1, round(6.5 * 3600 / bpd))),
        annual_vol=ann_vol,
    )
