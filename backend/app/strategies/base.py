"""Shared helpers for building strategy signals.

This module provides the small, dependency-free toolkit that every strategy
builder in :mod:`app.strategies.registry` relies on:

    * :func:`stance_from_score` — map a ``[-100, 100]`` score to a discrete
      :data:`~app.schemas.Stance` using the frozen contract thresholds.
    * :func:`clamp` — clamp a float to ``[lo, hi]`` (NaN/inf-safe).
    * :func:`squash` — squash an unbounded raw indicator to a ``[-100, 100]``
      score via a scaled hyperbolic tangent (``100 * tanh(x / scale)``).
    * :func:`linear_score` — map a value within an expected ``[lo, hi]`` band to
      ``[-100, 100]`` linearly (with optional inversion for "lower is better").
    * :func:`make_signal` — assemble a fully-validated
      :class:`~app.schemas.StrategySignal` with clamped score/confidence and a
      stance derived from the (clamped) score.

Every helper is numerically defensive: non-finite inputs collapse to safe
neutral values rather than propagating NaN/inf into the wire DTOs.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

from app.schemas import (
    ExpectedReturn,
    Stance,
    StrategyCategory,
    StrategySignal,
)

__all__ = [
    "stance_from_score",
    "clamp",
    "squash",
    "linear_score",
    "make_signal",
    "SCORE_MIN",
    "SCORE_MAX",
]

#: Inclusive bounds for any signal / composite score.
SCORE_MIN: float = -100.0
SCORE_MAX: float = 100.0


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp ``x`` to the inclusive range ``[lo, hi]``.

    Formula:
        clamp(x) = min(hi, max(lo, x))

    Args:
        x: Value to clamp. Non-finite values (NaN / +-inf) collapse to the
            midpoint ``(lo + hi) / 2`` so they never escape the range.
        lo: Lower bound.
        hi: Upper bound. If ``hi < lo`` the bounds are swapped.

    Returns:
        ``x`` constrained to ``[lo, hi]`` as a finite float.
    """
    low = float(lo)
    high = float(hi)
    if high < low:
        low, high = high, low
    xf = float(x)
    if not math.isfinite(xf):
        return 0.5 * (low + high)
    return max(low, min(high, xf))


def stance_from_score(score: float) -> Stance:
    """Map a numeric score to a discrete stance via the contract thresholds.

    Thresholds (score in ``[-100, 100]``)::

        score >= 60   -> STRONG_BUY
        score >= 20   -> BUY
        score >  -20  -> HOLD
        score >  -60  -> SELL
        else          -> STRONG_SELL

    Args:
        score: Bullishness score (positive = bullish). Non-finite values are
            treated as ``0.0`` (``HOLD``).

    Returns:
        The corresponding :data:`~app.schemas.Stance`.
    """
    s = float(score) if math.isfinite(score) else 0.0
    if s >= 60.0:
        return "STRONG_BUY"
    if s >= 20.0:
        return "BUY"
    if s > -20.0:
        return "HOLD"
    if s > -60.0:
        return "SELL"
    return "STRONG_SELL"


def squash(raw: float, scale: float = 1.0) -> float:
    """Squash an unbounded indicator into a ``[-100, 100]`` score with ``tanh``.

    Formula:
        score = 100 * tanh(raw / scale)

    The hyperbolic tangent maps the whole real line into ``(-1, 1)`` smoothly and
    monotonically, so a larger ``scale`` makes the mapping gentler (it takes a
    bigger ``raw`` to saturate toward +-100). This is the canonical way the
    registry turns a model's natural quantity (a Sharpe ratio, a z-score, a
    margin-of-safety, …) into a comparable bullishness score.

    Args:
        raw: The raw indicator value (any finite real number).
        scale: Positive scaling constant controlling sensitivity. Non-finite or
            non-positive values fall back to ``1.0``.

    Returns:
        A score in ``[-100, 100]`` (finite; ``0.0`` for a non-finite ``raw``).
    """
    r = float(raw)
    if not math.isfinite(r):
        return 0.0
    sc = float(scale)
    if not math.isfinite(sc) or sc <= 0.0:
        sc = 1.0
    val = 100.0 * math.tanh(r / sc)
    return clamp(val, SCORE_MIN, SCORE_MAX)


def linear_score(
    value: float,
    lo: float,
    hi: float,
    invert: bool = False,
) -> float:
    """Linearly map ``value`` within ``[lo, hi]`` onto ``[-100, 100]``.

    Formula (for ``invert == False``):
        t     = (value - lo) / (hi - lo)        # 0 at lo, 1 at hi
        score = (2 * t - 1) * 100               # -100 at lo, +100 at hi

    With ``invert == True`` the mapping is reflected (``hi`` -> -100, ``lo`` ->
    +100), used when *lower* values are bullish (e.g. tail-risk VaR, a stretched
    z-score). The result is clamped so values outside ``[lo, hi]`` saturate at
    the appropriate extreme.

    Args:
        value: The quantity to score. Non-finite values yield ``0.0``.
        lo: Value mapped to ``-100`` (or ``+100`` when inverted).
        hi: Value mapped to ``+100`` (or ``-100`` when inverted).
        invert: If ``True``, reverse the direction (lower is bullish).

    Returns:
        A score in ``[-100, 100]``.
    """
    v = float(value)
    if not math.isfinite(v):
        return 0.0
    low = float(lo)
    high = float(hi)
    if not math.isfinite(low) or not math.isfinite(high) or high == low:
        return 0.0
    t = (v - low) / (high - low)
    score = (2.0 * t - 1.0) * 100.0
    if invert:
        score = -score
    return clamp(score, SCORE_MIN, SCORE_MAX)


def _clean_metrics(metrics: Mapping[str, float] | None) -> dict[str, float]:
    """Coerce a metrics mapping to finite floats keyed by string.

    Non-finite metric values are replaced with ``0.0`` so the serialized DTO
    never carries NaN/inf.

    Args:
        metrics: Optional mapping of metric name to value.

    Returns:
        A new ``dict[str, float]`` with finite values.
    """
    if not metrics:
        return {}
    out: dict[str, float] = {}
    for key, val in metrics.items():
        try:
            fv = float(val)
        except (TypeError, ValueError):
            fv = 0.0
        out[str(key)] = fv if math.isfinite(fv) else 0.0
    return out


def _to_expected_returns(
    horizons: Iterable[Mapping[str, float] | ExpectedReturn] | None,
) -> list[ExpectedReturn]:
    """Normalize horizon dicts / models into validated ``ExpectedReturn`` objects.

    Accepts both the camelCase dicts produced by
    :func:`app.quant.returns.project_horizons` and already-built
    :class:`~app.schemas.ExpectedReturn` instances.

    Args:
        horizons: Iterable of horizon projections (dicts or models), or ``None``.

    Returns:
        A list of :class:`~app.schemas.ExpectedReturn`; empty if ``horizons`` is
        falsy. Individual malformed entries are skipped rather than raising.
    """
    if not horizons:
        return []
    out: list[ExpectedReturn] = []
    for h in horizons:
        if isinstance(h, ExpectedReturn):
            out.append(h)
            continue
        try:
            # ``project_horizons`` emits camelCase keys; CamelModel accepts them
            # via its alias generator (populate_by_name keeps snake_case too).
            out.append(ExpectedReturn(**dict(h)))
        except Exception:
            # A single bad projection must never sink the whole signal.
            continue
    return out


def make_signal(
    strategy_id: str,
    name: str,
    category: StrategyCategory,
    score: float,
    confidence: float,
    rationale: str,
    formula: str,
    metrics: Mapping[str, float] | None = None,
    horizons: Iterable[Mapping[str, float] | ExpectedReturn] | None = None,
) -> StrategySignal:
    """Assemble a fully-validated :class:`~app.schemas.StrategySignal`.

    The ``score`` is clamped to ``[-100, 100]`` and the ``confidence`` to
    ``[0, 1]``; the ``stance`` is derived from the **clamped** score via
    :func:`stance_from_score` so it is always internally consistent. Metric
    values are sanitized to finite floats and ``horizons`` (which may be empty
    for non-projecting models) are normalized into
    :class:`~app.schemas.ExpectedReturn` objects.

    Args:
        strategy_id: Stable strategy id (matches the registry / catalog).
        name: Human-readable strategy name.
        category: One of the eight :data:`~app.schemas.StrategyCategory` labels.
        score: Raw bullishness score (clamped to ``[-100, 100]``).
        confidence: Model confidence (clamped to ``[0, 1]``).
        rationale: Plain-English explanation referencing the numbers.
        formula: Compact human-readable formula used by the model.
        metrics: Optional model-specific raw numbers.
        horizons: Optional per-horizon projections (dicts or models).

    Returns:
        A populated :class:`~app.schemas.StrategySignal`.
    """
    clamped_score = clamp(score, SCORE_MIN, SCORE_MAX)
    clamped_conf = clamp(confidence, 0.0, 1.0)
    stance = stance_from_score(clamped_score)
    return StrategySignal(
        strategy_id=str(strategy_id),
        strategy_name=str(name),
        category=category,
        score=clamped_score,
        stance=stance,
        confidence=clamped_conf,
        rationale=str(rationale),
        formula=str(formula),
        metrics=_clean_metrics(metrics),
        horizons=_to_expected_returns(horizons),
    )
