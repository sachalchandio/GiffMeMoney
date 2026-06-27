"""Short-horizon trading signals — strictly point-in-time (no look-ahead).

Each signal maps the price history *up to and including* bar ``t`` to a desired
**raw exposure** in ``[-1, 1]`` (or ``[0, 1]`` when shorting is disallowed):

    +1  = fully long,   0 = flat / in cash,   -1 = fully short.

The defining safety property is **no look-ahead**: a signal at bar ``t`` reads
only ``prices[: t + 1]``. Mutating any future bar can never change the decision
at ``t`` (pinned by tests). This is what separates an honest backtest from a
fantasy that secretly peeks at tomorrow.

Two classic families are implemented:

    * **mean-reversion** (`zscore_meanrev`) — fade stretches: when price is far
      *above* its recent average (high z-score) lean short / to cash; far *below*,
      lean long. A dead-band around zero avoids churning on noise.
    * **momentum / breakout** (`momentum`) — ride trends: go long when the
      trailing return (or channel breakout) is positive, flat/short otherwise.

Neither has a real edge on the synthetic data — that is the point. They exist so
the lab can measure what *any* active style costs once the spread and fees are
charged.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["raw_exposure", "SIGNALS"]

_EPS: float = 1e-12

#: The signal ids the lab understands.
SIGNALS: tuple[str, ...] = ("meanrev", "momentum", "buyhold")


def _finite(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _zscore_meanrev(
    prices: np.ndarray,
    t: int,
    lookback: int,
    entry_z: float,
    exit_z: float,
    allow_short: bool,
) -> float:
    """Mean-reversion exposure from the trailing z-score of log-price.

    Computes ``z = (logP_t - mean) / std`` over the trailing ``lookback`` bars
    (ending AT ``t``). A high positive z (price stretched up) maps to negative
    exposure (fade), a low negative z to positive exposure. Inside ``±exit_z``
    the desired exposure is 0 (sit out the noise).

    Args:
        prices: The full price path (strictly positive).
        t: Current bar index (only ``prices[: t + 1]`` is read).
        lookback: Trailing window length in bars.
        entry_z: |z| at which exposure reaches full magnitude.
        exit_z: |z| below which exposure is 0 (dead-band).
        allow_short: If ``False``, negative exposure is clamped to 0 (long/flat).

    Returns:
        A finite exposure in ``[-1, 1]`` (``[0, 1]`` if ``allow_short`` is False).
    """
    lb = max(5, int(lookback))
    if t < lb:
        return 0.0
    seg = prices[t - lb + 1 : t + 1]
    seg = seg[seg > 0.0]
    if seg.size < 5:
        return 0.0
    logp = np.log(seg)
    mean = float(np.mean(logp))
    std = float(np.std(logp))
    if std <= _EPS:
        return 0.0
    z = (float(logp[-1]) - mean) / std
    az = abs(z)
    ez = max(0.0, _finite(exit_z))
    nz = max(ez + 1e-6, _finite(entry_z, 1.5))
    if az <= ez:
        return 0.0
    # Linearly ramp magnitude from 0 at exit_z to 1 at entry_z, then saturate.
    mag = min(1.0, (az - ez) / (nz - ez))
    expo = -math.copysign(mag, z)  # fade the move
    if not allow_short and expo < 0.0:
        return 0.0
    return float(max(-1.0, min(1.0, expo)))


def _momentum(
    prices: np.ndarray,
    t: int,
    lookback: int,
    allow_short: bool,
) -> float:
    """Trend / breakout exposure from the trailing return over ``lookback`` bars.

    Long when the trailing return is positive and the price is at/above the top
    of its trailing channel; flat (or short) when negative. Magnitude scales with
    a ``tanh`` of the trailing return so a stronger trend means a fuller position.

    Args:
        prices: The full price path (strictly positive).
        t: Current bar index (only ``prices[: t + 1]`` is read).
        lookback: Trailing window length in bars.
        allow_short: If ``False``, negative exposure is clamped to 0.

    Returns:
        A finite exposure in ``[-1, 1]`` (``[0, 1]`` if ``allow_short`` is False).
    """
    lb = max(5, int(lookback))
    if t < lb:
        return 0.0
    seg = prices[t - lb + 1 : t + 1]
    if seg.size < 5 or seg[0] <= 0.0:
        return 0.0
    trail_ret = float(seg[-1]) / float(seg[0]) - 1.0
    # Scale by the window's own volatility so the tanh argument is unit-free.
    with np.errstate(divide="ignore", invalid="ignore"):
        steps = seg[1:] / seg[:-1] - 1.0
    steps = np.nan_to_num(steps, nan=0.0, posinf=0.0, neginf=0.0)
    vol = float(np.std(steps)) * math.sqrt(max(1, lb))
    if vol <= _EPS:
        return 0.0
    expo = math.tanh(trail_ret / vol)
    if not allow_short and expo < 0.0:
        return 0.0
    return float(max(-1.0, min(1.0, expo)))


def raw_exposure(
    signal: str,
    prices: np.ndarray,
    t: int,
    *,
    lookback: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.4,
    allow_short: bool = False,
) -> float:
    """Dispatch to a named signal and return its point-in-time raw exposure.

    Args:
        signal: One of :data:`SIGNALS` (``"meanrev"`` / ``"momentum"`` /
            ``"buyhold"``). Unknown ids fall back to flat (0).
        prices: The full price path.
        t: Current bar index (only ``prices[: t + 1]`` is read).
        lookback: Trailing window for the signal.
        entry_z: Mean-reversion entry threshold (|z| for full magnitude).
        exit_z: Mean-reversion dead-band half-width.
        allow_short: Whether negative (short) exposure is permitted.

    Returns:
        A finite exposure in ``[-1, 1]`` (``[0, 1]`` if ``allow_short`` is False).
    """
    key = str(signal or "").strip().lower()
    if key == "buyhold":
        return 1.0
    if key == "momentum":
        return _momentum(prices, t, lookback, allow_short)
    # default / "meanrev"
    return _zscore_meanrev(prices, t, lookback, entry_z, exit_z, allow_short)
