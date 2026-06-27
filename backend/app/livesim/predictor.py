"""A self-updating online predictor for the Real-Time mode.

Each venue carries one :class:`OnlinePredictor` that, on every tick, (1) emits a
probability that the next move is *up*, then (2) learns from what actually
happened by nudging its weights. This is genuine online learning — the model
"updates itself with the market" as the user asked.

It is a small **online logistic regression** over a handful of momentum /
mean-reversion / volatility features, with running feature standardisation so it
stays numerically stable. Honesty matters here: on data with no real edge the
output sits near 0.5 (a coin flip), and the model surfaces that as low
confidence rather than pretending to see the future.

Everything is defensive: non-finite inputs are scrubbed, weights and outputs are
clipped, and nothing ever raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["OnlinePredictor", "FEATURE_NAMES", "features_from_path"]

#: Human-readable names of the features the predictor consumes (in order).
FEATURE_NAMES: tuple[str, ...] = (
    "ret_fast",   # short trailing return
    "ret_slow",   # longer trailing return
    "rsi_gap",    # RSI distance from 50 (overbought/oversold)
    "ewma_gap",   # distance of price from its EWMA (mean-reversion pull)
    "vol",        # trailing volatility
)

_EPS = 1e-9


def _finite(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def features_from_path(prices: np.ndarray, t: int) -> np.ndarray:
    """Build the point-in-time feature vector for bar ``t`` from ``prices[:t+1]``.

    Args:
        prices: The venue's price path (strictly positive).
        t: Current bar index (only data up to and including ``t`` is read).

    Returns:
        A length-``len(FEATURE_NAMES)`` finite feature vector.
    """
    n = len(FEATURE_NAMES)
    if t < 2:
        return np.zeros(n, dtype=np.float64)
    p = np.asarray(prices, dtype=np.float64)
    cur = float(p[t]) if p[t] > 0 else _EPS

    def trailing_return(win: int) -> float:
        j = max(0, t - win)
        base = float(p[j]) if p[j] > 0 else _EPS
        return cur / base - 1.0

    ret_fast = trailing_return(5)
    ret_slow = trailing_return(20)

    # RSI-ish: share of up-moves over a window, mapped to [-1, 1].
    w = min(14, t)
    seg = p[t - w : t + 1]
    diffs = np.diff(seg)
    ups = float(np.sum(diffs > 0))
    rsi = (ups / max(1, diffs.size)) * 2.0 - 1.0

    # EWMA gap: how far above/below the recent EWMA we are.
    span = min(20, t)
    alpha = 2.0 / (span + 1.0)
    ewma = float(p[max(0, t - span)])
    for k in range(max(1, t - span) + 1, t + 1):
        ewma = alpha * float(p[k]) + (1.0 - alpha) * ewma
    ewma_gap = cur / (ewma if ewma > 0 else _EPS) - 1.0

    # Trailing volatility of returns.
    rseg = seg[1:] / np.where(seg[:-1] > 0, seg[:-1], _EPS) - 1.0
    vol = float(np.std(np.nan_to_num(rseg, nan=0.0, posinf=0.0, neginf=0.0)))

    out = np.array([ret_fast, ret_slow, rsi, ewma_gap, vol], dtype=np.float64)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class OnlinePredictor:
    """An online logistic regression with running feature standardisation.

    Attributes:
        n_features: Number of input features.
        lr: SGD learning rate.
        w: Weight vector (length ``n_features``).
        b: Bias term.
        mean: Running feature means (for standardisation).
        m2: Running feature second moments (Welford) for variance.
        count: Number of updates seen.
    """

    n_features: int = len(FEATURE_NAMES)
    lr: float = 0.05
    w: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURE_NAMES)))
    b: float = 0.0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURE_NAMES)))
    m2: np.ndarray = field(default_factory=lambda: np.ones(len(FEATURE_NAMES)))
    count: int = 0

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        """Standardise features by the running mean/variance (clipped)."""
        if self.count < 2:
            std = np.ones(self.n_features)
        else:
            std = np.sqrt(np.maximum(self.m2 / max(1, self.count), 1e-6))
        z = (x - self.mean) / std
        return np.clip(np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0), -5.0, 5.0)

    def _observe(self, x: np.ndarray) -> None:
        """Update the running feature mean/variance with a new observation (Welford)."""
        self.count += 1
        delta = x - self.mean
        self.mean = self.mean + delta / self.count
        self.m2 = self.m2 + delta * (x - self.mean)

    def predict_proba(self, x: np.ndarray) -> float:
        """Return P(next move is up) in ``[0, 1]`` for feature vector ``x``."""
        z = self._standardize(np.asarray(x, dtype=np.float64))
        logit = float(np.dot(self.w, z) + self.b)
        logit = max(-30.0, min(30.0, logit))
        p = 1.0 / (1.0 + math.exp(-logit))
        return _finite(p, 0.5)

    def update(self, x: np.ndarray, outcome: float) -> None:
        """Learn from a realised outcome (1.0 = went up, 0.0 = went down).

        Performs one SGD step of logistic-regression loss, then folds the new
        observation into the running normalisation. Weights are clipped so a
        pathological streak can never blow them up.

        Args:
            x: The feature vector that produced the prediction.
            outcome: The realised label in ``{0.0, 1.0}``.
        """
        xv = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        z = self._standardize(xv)
        p = self.predict_proba(xv)
        y = 1.0 if outcome >= 0.5 else 0.0
        grad = p - y  # dLoss/dlogit
        self.w = np.clip(self.w - self.lr * grad * z, -10.0, 10.0)
        self.b = float(np.clip(self.b - self.lr * grad, -10.0, 10.0))
        self._observe(xv)

    def confidence(self, p: float) -> float:
        """Map an up-probability to a 0..1 confidence (distance from a coin flip)."""
        return _finite(abs(p - 0.5) * 2.0, 0.0)
