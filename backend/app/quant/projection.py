"""Financially-credible multi-horizon projection engine (STRATEGIES-V2 §4).

This module replaces the old per-model lognormal blend (``returns.project_horizons``)
that produced over-optimistic, internally-inconsistent forecasts (e.g. +800% 5Y
"expected" returns with +26,000% upper bands, flat confidence, everything HOLD).
It implements the mandatory financial-rigor fixes from STRATEGIES-V2 §0:

    * **R1 — Credible drift.** Strategy daily drifts are ensembled by confidence,
      then shrunk hard (James–Stein) toward a CAPM prior, regime-tilted, and the
      **annualized** expected drift is capped to a realistic band per asset class
      (equities/ETF ``[-25%, +35%]``, crypto ``[-40%, +60%]``) before being
      converted back to a daily drift. No asset can project an absurd median.
    * **R2 — Believable bands.** Confidence bands come from a *fat-tailed*
      distribution (Student-t, df≈5) scaled per horizon (or a block-bootstrap of
      the historical returns when enough history exists). The displayed 5th/95th
      percentiles are floored at −95% (a long-only position cannot lose >100%) and
      the 95th percentile is capped to credible per-horizon bounds (≤ +60% at 1Y,
      ≤ +400% at 5Y even for crypto). ``cagr_pct`` is exposed for multi-year
      horizons so a big total return is never read as an annual rate.
    * **R3 — One engine.** :func:`mc_summary` runs the Monte Carlo from the SAME
      drift + vol that :func:`project` uses, so the engine's ``montecarlo()`` and
      the blended ``expectedReturns`` agree within ~1 percentage point.
    * **R6 — Honest downside.** Every horizon carries ``prob_positive`` from the
      fat-tailed dist (never an inflated 0.9+), an explicit bull/base/bear fan,
      and ``cvar_pct`` — the 95% expected shortfall (a positive loss percentage).

Everything is numerically defensive: short / empty / constant / non-finite inputs
never raise and never emit NaN/inf — they collapse to safe, finite, capped values.

The public surface is::

    HorizonProjection                       # one per horizon (superset of ExpectedReturn)
    detect_regime(closes) -> dict
    ensemble_drift(signal_drifts, prior) -> float
    forward_vol(returns, horizon_days) -> float
    project(closes, returns, signal_drifts, rf_daily, beta, capm_drift_daily,
            asset_class) -> (list[HorizonProjection], regime_dict)
    mc_summary(s0, drift_daily, vol_daily, horizon, sims, seed) -> dict
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, t as student_t

from app.quant import montecarlo
from app.quant.returns import HORIZON_DAYS, TRADING_DAYS
from app.quant.volatility import ewma_vol, garch11_fit, garch11_forecast

__all__ = [
    "HorizonProjection",
    "detect_regime",
    "ensemble_drift",
    "forward_vol",
    "forward_vol_term_structure",
    "project",
    "mc_summary",
    "annual_to_daily_drift",
    "daily_to_annual_drift",
]


# ---------------------------------------------------------------------------
# Constants & tunables
# ---------------------------------------------------------------------------

# Smallest daily volatility treated as non-zero (constant series otherwise).
_VOL_FLOOR: float = 1e-6

# Degrees of freedom for the Student-t confidence bands (fat tails). df≈5 gives
# noticeably heavier tails than a Normal while keeping a finite variance.
_T_DF: float = 5.0

# Equity-risk premium used to build the CAPM sanity prior when no prior given.
_ERP_ANNUAL: float = 0.045

# Default annual risk-free rate (decimal) when none can be inferred.
_DEFAULT_RF_ANNUAL: float = 0.04

# R1 — annualized expected-drift caps per asset class (simple-return decimals).
# Equities / ETFs are pulled to a believable single-digit-to-low-double-digit
# band; crypto is allowed a wider but still bounded band.
_DRIFT_CAP_EQUITY: tuple[float, float] = (-0.25, 0.35)
_DRIFT_CAP_CRYPTO: tuple[float, float] = (-0.40, 0.60)

# James–Stein shrinkage controls. ``_SHRINK_K`` (a pseudo-count of "prior
# observations") sets how hard a low-confidence ensemble is pulled to the prior;
# the realized shrink weight is k / (k + effective_confidence_mass).
_SHRINK_K: float = 4.0

# Regime drift tilt: a clear bull adds up to this fraction of the annual cap
# width, a clear bear subtracts it. Kept inside the cap (applied pre-cap).
_REGIME_TILT_ANNUAL: float = 0.05

# R2 — per-horizon credible cap on the 95th-percentile *total* return (decimal).
# Deliberately generous (so a real fat right tail still shows) yet bounded so no
# horizon ever prints an absurd upper band, even for crypto. The caps are kept
# **monotone non-decreasing** with horizon so that capping the upper band can
# never make a longer horizon's band narrower than a shorter one (the §0 R2
# anchors are ``≤ +60% (1Y)`` and ``≤ +400% (5Y)``; the intermediate caps are
# chosen ≤ the 1Y anchor and increasing).
_HIGH_CAP_BY_HORIZON: dict[str, float] = {
    "1D": 0.15,    # ≤ +15% in a day
    "1W": 0.35,    # ≤ +35% in a week
    "1M": 0.55,    # ≤ +55% in a month   (≤ 1Y anchor, monotone)
    "1Y": 0.60,    # ≤ +60% in a year    (per §0 R2)
    "5Y": 4.00,    # ≤ +400% over 5y     (per §0 R2)
}

# R2 — a long-only position cannot lose more than ~100%; floor the low band a
# hair above −100% so the displayed worst case is credible.
_LOW_FLOOR_PCT: float = -95.0

# Hard finite clamp on any emitted percentage (last line of defence).
_PCT_CLAMP: float = 1.0e6

# Two-sided z used for the 5th/95th scenario fan when we fall back to Normal.
_Z_90: float = 1.6448536269514722  # Phi^{-1}(0.95)

# Minimum history (daily returns) before we trust a block-bootstrap over the
# parametric Student-t for the bands.
_MIN_BOOTSTRAP_OBS: int = 120

# Number of bootstrap resamples used to estimate horizon quantiles. 1500 is
# ample for stable 5th/95th-percentile and lower-tail-mean estimates while
# keeping a full-universe projection pass fast.
_N_BOOTSTRAP: int = 1500

# RNG seed for the (deterministic) bootstrap so projections are reproducible.
_BOOTSTRAP_SEED: int = 1_234_567

# Sims used inside :func:`mc_summary` by default.
_MC_SIMS_DEFAULT: int = 2000


def _safe_float(x: float, default: float = 0.0) -> float:
    """Return ``x`` as a finite float, else ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _clamp_pct(pct: float) -> float:
    """Clamp a percentage to a finite, sane range (never NaN/inf)."""
    p = _safe_float(pct, 0.0)
    return max(-_PCT_CLAMP, min(_PCT_CLAMP, p))


def annual_to_daily_drift(annual_return: float) -> float:
    """Convert an annual *simple* return to a daily continuously-compounded drift.

    Formula:
        mu_daily = ln(1 + annual_return) / TRADING_DAYS

    Args:
        annual_return: Annual simple return as a decimal (e.g. ``0.10``).

    Returns:
        The equivalent daily log-drift. ``0.0`` for non-finite or ``<= -100%``
        inputs (a total loss has no finite log-drift).
    """
    a = _safe_float(annual_return, 0.0)
    if a <= -1.0:
        return 0.0
    return math.log1p(a) / TRADING_DAYS


def daily_to_annual_drift(daily_drift: float) -> float:
    """Convert a daily log-drift to an annual *simple* return.

    Formula:
        R_annual = exp(mu_daily * TRADING_DAYS) - 1

    Args:
        daily_drift: Mean daily log return.

    Returns:
        The annualized simple return as a decimal. Non-finite inputs yield
        ``0.0``; the exponent is clamped so the result stays finite.
    """
    mu = _safe_float(daily_drift, 0.0)
    exponent = max(-50.0, min(50.0, mu * TRADING_DAYS))
    result = math.exp(exponent) - 1.0
    return result if math.isfinite(result) else 0.0


# ---------------------------------------------------------------------------
# HorizonProjection
# ---------------------------------------------------------------------------


@dataclass
class HorizonProjection:
    """A credible projection for one horizon — a superset of ``ExpectedReturn``.

    All ``*_pct`` fields are **total** returns over the horizon expressed in
    percent (e.g. ``12.0`` = +12%), except ``annualized_vol`` (annual vol in
    percent) and ``cvar_pct`` (a positive loss percentage).

    Attributes:
        horizon: One of ``'1D' | '1W' | '1M' | '1Y' | '5Y'``.
        expected_return_pct: Median / base-case total return (the drift
            projection), in percent. Equal to ``base_pct``.
        low: ~5th-percentile total return from the fat-tailed dist, floored at
            −95% (long-only), in percent.
        high: ~95th-percentile total return from the fat-tailed dist, capped to
            a credible per-horizon bound, in percent.
        prob_positive: Probability the total return is positive, in ``[0, 1]``,
            from the fat-tailed distribution (never inflated to 0.9+ artificially).
        annualized_vol: Annualized volatility forecast, in percent.
        bull_pct: Bull-scenario total return (``base + z*sigma_h``), in percent.
        base_pct: Base-scenario total return (the drift projection), in percent.
        bear_pct: Bear-scenario total return (``base - z*sigma_h``), in percent.
        cvar_pct: 95% expected shortfall (mean of the worst 5% of horizon
            outcomes) as a positive loss percentage.
    """

    horizon: str
    expected_return_pct: float
    low: float
    high: float
    prob_positive: float
    annualized_vol: float
    bull_pct: float
    base_pct: float
    bear_pct: float
    cvar_pct: float

    def cagr_pct(self) -> float:
        """Compound annual growth rate of ``expected_return_pct``, in percent.

        For multi-year horizons a large *total* return can mislead (a +150% 5Y
        total is only ~20%/yr); this helper annualizes the base-case total return
        over the horizon length so the per-year rate can be shown alongside it.

        Formula (``y`` = horizon length in years):
            cagr = ((1 + total_return) ** (1 / y) - 1) * 100

        Returns:
            The annualized growth rate in percent. For sub-year horizons the
            total return is still annualized (``y < 1`` ⇒ compounding up); a
            total loss (``<= -100%``) maps to ``-100``. Always finite.
        """
        years = HORIZON_DAYS.get(self.horizon, TRADING_DAYS) / float(TRADING_DAYS)
        if years <= 0.0:
            return _clamp_pct(self.expected_return_pct)
        total = _safe_float(self.expected_return_pct, 0.0) / 100.0
        if total <= -1.0:
            return -100.0
        try:
            cagr = (math.pow(1.0 + total, 1.0 / years) - 1.0) * 100.0
        except (ValueError, OverflowError):
            return _clamp_pct(self.expected_return_pct)
        return _clamp_pct(cagr)

    def to_dict(self) -> dict:
        """Return the camelCase ``ExpectedReturn``-shaped dict (with V2 fields).

        The keys match the :class:`~app.schemas.ExpectedReturn` aliases so the
        engine can build the DTO directly (``ExpectedReturn(**proj.to_dict())``).
        """
        return {
            "horizon": self.horizon,
            "expectedReturnPct": _clamp_pct(self.expected_return_pct),
            "low": _clamp_pct(self.low),
            "high": _clamp_pct(self.high),
            "probPositive": min(1.0, max(0.0, _safe_float(self.prob_positive, 0.5))),
            "annualizedVol": _clamp_pct(self.annualized_vol),
            "bullPct": _clamp_pct(self.bull_pct),
            "basePct": _clamp_pct(self.base_pct),
            "bearPct": _clamp_pct(self.bear_pct),
            "cvarPct": max(0.0, _clamp_pct(self.cvar_pct)),
        }


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------


def _clean_closes(closes: np.ndarray | list[float]) -> np.ndarray:
    """Coerce closes to a clean 1-D array of finite, strictly-positive prices."""
    arr = np.asarray(closes, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr) & (arr > 0.0)
    return arr[mask]


def _clean_returns(returns: np.ndarray | list[float]) -> np.ndarray:
    """Coerce returns to a clean 1-D array of finite values."""
    arr = np.asarray(returns, dtype=np.float64).ravel()
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def detect_regime(closes: np.ndarray | list[float]) -> dict:
    """Classify the market regime of a price series.

    Combines a **trend** read (price relative to its 200-day moving average plus
    the sign/strength of the long trend slope) with a **volatility regime** read
    (recent ~21-day vol relative to the long ~252-day vol):

        ma_gap   = price / MA200 - 1                      (price above/below trend)
        slope    = OLS slope of log price over the window  (normalized to /yr)
        trend    = 0.6 * tanh(ma_gap / 0.10) + 0.4 * tanh(slope_annual / 0.30)
        vol_ratio= recent_vol / long_vol                  (vol regime)
        score    = trend dampened slightly when vol is elevated

    Labels:
        regime    : 'bull' (score >  0.15) | 'bear' (score < -0.15) | 'neutral'
        vol_regime: 'low' (ratio < 0.8) | 'high' (ratio > 1.3) | 'normal'

    Args:
        closes: Daily closing prices (most recent last).

    Returns:
        A dict ``{'regime': str, 'trend': float, 'vol_regime': str,
        'score': float}`` with ``trend`` / ``score`` in roughly ``[-1, 1]`` and
        all values finite. Short / empty input yields a neutral, finite result.
    """
    arr = _clean_closes(closes)
    n = arr.size
    if n < 5:
        return {"regime": "neutral", "trend": 0.0, "vol_regime": "normal", "score": 0.0}

    price = float(arr[-1])

    # --- trend: price vs 200d MA (use the longest available window up to 200) ---
    ma_win = min(200, n)
    ma = float(np.mean(arr[-ma_win:]))
    ma_gap = (price / ma - 1.0) if ma > 0.0 else 0.0
    ma_gap = _safe_float(ma_gap, 0.0)

    # --- trend slope: OLS of log price over the long window, annualized ---
    slope_win = min(252, n)
    seg = arr[-slope_win:]
    y = np.log(seg)
    x = np.arange(seg.size, dtype=np.float64)
    if seg.size >= 2:
        xm = float(np.mean(x))
        ym = float(np.mean(y))
        dx = x - xm
        denom = float(np.dot(dx, dx))
        slope_daily = float(np.dot(dx, y - ym)) / denom if denom > 0.0 else 0.0
    else:
        slope_daily = 0.0
    slope_annual = _safe_float(slope_daily * TRADING_DAYS, 0.0)  # ~ annual log drift

    trend = 0.6 * math.tanh(ma_gap / 0.10) + 0.4 * math.tanh(slope_annual / 0.30)
    trend = _safe_float(trend, 0.0)
    trend = max(-1.0, min(1.0, trend))

    # --- volatility regime: recent vs long realized vol ---
    rets = arr[1:] / arr[:-1] - 1.0
    rets = rets[np.isfinite(rets)]
    if rets.size >= 21:
        recent = float(np.std(rets[-21:]))
        long_win = rets[-min(252, rets.size):]
        long_vol = float(np.std(long_win))
        vol_ratio = (recent / long_vol) if long_vol > _VOL_FLOOR else 1.0
    else:
        vol_ratio = 1.0
    vol_ratio = _safe_float(vol_ratio, 1.0)

    if vol_ratio < 0.8:
        vol_regime = "low"
    elif vol_ratio > 1.3:
        vol_regime = "high"
    else:
        vol_regime = "normal"

    # Elevated vol dampens trend conviction (a sharp move in a turbulent tape is
    # less reliable than the same move in a calm one).
    vol_damp = 1.0 / (1.0 + max(0.0, vol_ratio - 1.0))
    score = _safe_float(trend * vol_damp, 0.0)
    score = max(-1.0, min(1.0, score))

    if score > 0.15:
        regime = "bull"
    elif score < -0.15:
        regime = "bear"
    else:
        regime = "neutral"

    return {
        "regime": regime,
        "trend": round(trend, 6),
        "vol_regime": vol_regime,
        "score": round(score, 6),
    }


# ---------------------------------------------------------------------------
# Ensemble drift (R1)
# ---------------------------------------------------------------------------


def _drift_cap(asset_class: str | None) -> tuple[float, float]:
    """Return the annualized drift cap ``(lo, hi)`` for an asset class."""
    cls = str(asset_class or "").strip().lower()
    if cls == "crypto":
        return _DRIFT_CAP_CRYPTO
    # equities, etf, and anything unknown get the tighter equity band.
    return _DRIFT_CAP_EQUITY


def ensemble_drift(
    signal_drifts: list[tuple[float, float]],
    prior: float,
    asset_class: str | None = "equity",
    regime: dict | None = None,
) -> float:
    """Confidence-weighted, shrunk, capped, regime-tilted daily drift (R1).

    The pipeline (all on **daily** log-drifts):

        1. **Ensemble.** Take the confidence-weighted mean of the projecting
           strategies' daily drifts ``mu_i`` with weights ``c_i``:
               raw = sum(c_i * mu_i) / sum(c_i)
        2. **James–Stein shrinkage toward the prior** (the CAPM daily drift).
           A noisy single-asset ensemble is pulled toward the prior; the more
           total confidence mass, the less it is shrunk:
               w_prior = k / (k + sum(c_i))            (k = ``_SHRINK_K``)
               shrunk  = (1 - w_prior) * raw + w_prior * prior
        3. **Regime tilt.** A clear bull adds, a clear bear subtracts, a small
           annual amount (kept inside the cap, applied in annual space).
        4. **Cap.** Convert to an annual simple return, clamp to the asset-class
           band (R1: equities/ETF ``[-25%, +35%]``, crypto ``[-40%, +60%]``),
           then convert back to a daily drift.

    Args:
        signal_drifts: ``[(daily_drift, confidence), ...]`` from the projecting
            strategies. Non-finite entries and non-positive confidences are
            ignored. May be empty (then the prior carries the estimate).
        prior: The CAPM daily log-drift (``rf + β·ERP`` expressed daily). Used
            both as the shrinkage target and as the fallback when no signals
            project.
        asset_class: ``'equity' | 'crypto' | 'etf'`` (drives the cap). Unknown
            classes use the equity band.
        regime: Optional regime dict from :func:`detect_regime`; its ``score``
            tilts the annual drift within the cap.

    Returns:
        A finite, capped **daily** log-drift.
    """
    prior_daily = _safe_float(prior, 0.0)

    # 1) confidence-weighted ensemble of the daily drifts.
    weight_sum = 0.0
    weighted = 0.0
    for entry in signal_drifts or []:
        try:
            mu, conf = float(entry[0]), float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not (math.isfinite(mu) and math.isfinite(conf)) or conf <= 0.0:
            continue
        # Defensive per-signal clamp on the daily drift (≈ ±300%/yr) so a single
        # pathological strategy cannot dominate before shrinkage.
        mu = max(-0.005, min(0.005, mu))
        weighted += conf * mu
        weight_sum += conf

    if weight_sum > 0.0:
        raw = weighted / weight_sum
    else:
        # No projecting strategy → lean entirely on the prior.
        raw = prior_daily
        weight_sum = 0.0

    # 2) James–Stein shrinkage toward the CAPM prior.
    w_prior = _SHRINK_K / (_SHRINK_K + weight_sum)
    w_prior = max(0.0, min(1.0, w_prior))
    shrunk_daily = (1.0 - w_prior) * raw + w_prior * prior_daily

    # 3) regime tilt (in annual space so the magnitude is interpretable).
    annual = daily_to_annual_drift(shrunk_daily)
    score = 0.0
    if regime is not None:
        score = _safe_float(regime.get("score", 0.0), 0.0)
        score = max(-1.0, min(1.0, score))
    annual += _REGIME_TILT_ANNUAL * score

    # 4) cap the annual drift to the asset-class band, convert back to daily.
    lo, hi = _drift_cap(asset_class)
    annual = max(lo, min(hi, annual))
    return annual_to_daily_drift(annual)


# ---------------------------------------------------------------------------
# Forward volatility (GARCH(1,1) → EWMA fallback) with term structure
# ---------------------------------------------------------------------------


def forward_vol(returns: np.ndarray | list[float], horizon_days: int) -> float:
    """Forecast annualized volatility ``horizon_days`` ahead (GARCH→EWMA).

    Delegates to the GARCH(1,1) horizon forecast in
    :mod:`app.quant.volatility` (which itself falls back to EWMA on short data
    or a failed fit). The GARCH forecast already returns the average
    annualized vol over the horizon — i.e. it embeds the vol *term structure*
    (mean-reversion of conditional variance toward the long-run level), so a
    1-day forecast and a 5-year forecast differ appropriately even though both
    are reported on an annualized basis.

    Args:
        returns: Daily simple/log returns of the asset.
        horizon_days: Forecast horizon in trading days (coerced to ``>= 1``).

    Returns:
        Annualized volatility forecast as a decimal, floored at ``_VOL_FLOOR``
        and guaranteed finite (never NaN/inf).
    """
    arr = _clean_returns(returns)
    h = max(1, int(horizon_days))
    if arr.size < 2:
        return _VOL_FLOOR
    try:
        ann = garch11_forecast(arr, h)
    except Exception:  # pragma: no cover - defensive
        ann = ewma_vol(arr)
    ann = _safe_float(ann, 0.0)
    if ann < _VOL_FLOOR:
        # Final fallback: sample annualized vol.
        sd = float(np.std(arr))
        ann = max(_VOL_FLOOR, sd * math.sqrt(TRADING_DAYS))
    return ann


# Volatility floors mirrored from ``app.quant.volatility`` so the single-fit
# term-structure forecast below matches ``garch11_forecast`` numerically.
_VAR_FLOOR_PROJ: float = 1e-12
_PERSISTENCE_CAP_PROJ: float = 0.9999


def forward_vol_term_structure(
    returns: np.ndarray | list[float], horizon_days: list[int]
) -> dict[int, float]:
    """Forecast annualized vol for several horizons from ONE GARCH(1,1) fit.

    :func:`forward_vol` refits GARCH on every call, so projecting all five
    horizons re-runs the (scipy-optimized) fit five times per asset — by far the
    dominant cost of :func:`project`. This helper fits GARCH **once**, rolls the
    conditional-variance recursion through the sample to the latest variance, and
    then applies the SAME mean-reverting k-step forecast that
    :func:`app.quant.volatility.garch11_forecast` uses for each requested
    horizon. The per-horizon results are therefore numerically identical to
    calling :func:`forward_vol` five times, at ~1/5 the cost.

    Args:
        returns: Daily returns of the asset.
        horizon_days: The list of horizons (in trading days) to forecast.

    Returns:
        A ``{horizon_days: annualized_vol_decimal}`` map. Every value is finite
        and floored at ``_VOL_FLOOR``. On short data / a failed fit, falls back
        to the EWMA vol for every horizon.
    """
    arr = _clean_returns(returns)
    hzs = [max(1, int(h)) for h in horizon_days]
    if arr.size < 30:
        ann = max(_VOL_FLOOR, _safe_float(ewma_vol(arr), _VOL_FLOOR))
        return {h: ann for h in hzs}

    try:
        omega, alpha, beta = garch11_fit(arr)
    except Exception:  # pragma: no cover - defensive
        ann = max(_VOL_FLOOR, _safe_float(ewma_vol(arr), _VOL_FLOOR))
        return {h: ann for h in hzs}

    persistence = alpha + beta
    denom = 1.0 - persistence
    sample_var = float(np.var(arr))
    if denom <= _VAR_FLOOR_PROJ:
        uncond_var = sample_var
    else:
        uncond_var = omega / denom
    if not math.isfinite(uncond_var) or uncond_var < _VAR_FLOOR_PROJ:
        uncond_var = max(sample_var, _VAR_FLOOR_PROJ)

    # Roll the recursion to the latest one-step-ahead variance h_{T+1}.
    eps = arr - float(np.mean(arr))
    e2 = eps * eps
    h_var = max(sample_var, _VAR_FLOOR_PROJ)
    for t in range(arr.size):
        h_var = omega + alpha * e2[t] + beta * h_var
        if not math.isfinite(h_var) or h_var < _VAR_FLOOR_PROJ:
            h_var = _VAR_FLOOR_PROJ

    ewma_fallback = max(_VOL_FLOOR, _safe_float(ewma_vol(arr), _VOL_FLOOR))
    if not math.isfinite(h_var) or h_var < _VAR_FLOOR_PROJ:
        return {h: ewma_fallback for h in hzs}

    out: dict[int, float] = {}
    no_reversion = persistence >= _PERSISTENCE_CAP_PROJ or persistence <= 0.0
    for h in sorted(set(hzs)):
        if no_reversion:
            avg_var = h_var
        else:
            total = 0.0
            for k in range(1, h + 1):
                fc = uncond_var + (persistence ** (k - 1)) * (h_var - uncond_var)
                if not math.isfinite(fc) or fc < 0.0:
                    fc = uncond_var
                total += fc
            avg_var = total / h
        if not math.isfinite(avg_var) or avg_var < 0.0:
            out[h] = ewma_fallback
            continue
        ann = math.sqrt(max(avg_var, 0.0)) * math.sqrt(TRADING_DAYS)
        out[h] = max(_VOL_FLOOR, ann) if math.isfinite(ann) else ewma_fallback
    return {h: out[h] for h in hzs}


# ---------------------------------------------------------------------------
# Fat-tailed / bootstrap horizon quantiles (R2)
# ---------------------------------------------------------------------------


def _student_t_log_quantiles(
    drift_h: float, sigma_h: float, p_low: float, p_high: float
) -> tuple[float, float, float, float]:
    """Student-t (df≈5) low/high log-return quantiles + tail expectations.

    The horizon log return is modelled as ``drift_h + sigma_h * T`` where ``T``
    is a *standardized* Student-t (unit variance) with ``_T_DF`` degrees of
    freedom — heavier tails than a Normal but a finite variance, so ``sigma_h``
    stays the genuine horizon volatility.

    Returns:
        ``(low_log, high_log, prob_positive, es_log)`` where ``es_log`` is the
        expected log return in the lower ``(1 - 0.95)`` tail (for CVaR).
    """
    df = _T_DF
    # Standardize so the t has unit variance: var(t_df) = df/(df-2).
    std_scale = math.sqrt(df / (df - 2.0))
    # Quantiles of the standardized t.
    q_low = float(student_t.ppf(p_low, df)) / std_scale
    q_high = float(student_t.ppf(p_high, df)) / std_scale

    low_log = drift_h + sigma_h * q_low
    high_log = drift_h + sigma_h * q_high

    # P(return > 0) = P(T > -drift_h / sigma_h).
    if sigma_h > _VOL_FLOOR:
        z = -drift_h / sigma_h * std_scale
        prob_positive = float(1.0 - student_t.cdf(z, df))
    else:
        prob_positive = 1.0 if drift_h > 0 else (0.0 if drift_h < 0 else 0.5)

    # Expected shortfall of the *log* return at 95%: mean of the tail below the
    # 5% quantile. Use a fine grid of lower-tail quantiles (closed form for the
    # t-ES is messier; a quadrature over the tail is robust and simple).
    alpha = 0.05
    grid = np.linspace(1e-4, alpha, 256)
    t_q = student_t.ppf(grid, df) / std_scale
    es_std = float(np.mean(t_q))  # E[T | T <= q_alpha] (standardized)
    es_log = drift_h + sigma_h * es_std

    return low_log, high_log, prob_positive, es_log


def _bootstrap_log_quantiles(
    rets: np.ndarray,
    horizon_days: int,
    drift_h_target: float,
    sigma_h_target: float,
    p_low: float,
    p_high: float,
    seed: int,
) -> tuple[float, float, float, float]:
    """Block-bootstrap horizon log-return quantiles, re-centered & re-scaled.

    Resamples overlapping blocks of historical daily returns to build a
    horizon-return distribution that inherits the empirical *shape* (skew /
    fat tails / autocorrelation), then affine-maps it so its mean and std match
    the engine's target ``drift_h_target`` / ``sigma_h_target`` (so the bands are
    consistent with the capped drift and the GARCH vol forecast — R1/R3).

    Returns:
        ``(low_log, high_log, prob_positive, es_log)``.
    """
    n = rets.size
    h = max(1, int(horizon_days))
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

    # Log returns of the (clean) series for additive horizon aggregation.
    # rets are simple returns; convert to log defensively.
    with np.errstate(invalid="ignore", divide="ignore"):
        log_r = np.log1p(rets)
    log_r = log_r[np.isfinite(log_r)]
    if log_r.size < 2:
        return _student_t_log_quantiles(
            drift_h_target, sigma_h_target, p_low, p_high
        )

    block = max(1, min(h, 21))  # ~ up to a month-long block
    n_blocks = int(math.ceil(h / block))
    n_lr = log_r.size

    # Draw all block start indices at once: (sims, n_blocks).
    starts = rng.integers(0, n_lr, size=(_N_BOOTSTRAP, n_blocks))
    # Build an index grid for each block of length ``block`` (wrap-around).
    offsets = np.arange(block)
    sims_sum = np.zeros(_N_BOOTSTRAP, dtype=np.float64)
    taken = 0
    for b in range(n_blocks):
        take = min(block, h - taken)
        if take <= 0:
            break
        idx = (starts[:, b][:, None] + offsets[None, :take]) % n_lr
        sims_sum += np.sum(log_r[idx], axis=1)
        taken += take

    sims_sum = sims_sum[np.isfinite(sims_sum)]
    if sims_sum.size < 8:
        return _student_t_log_quantiles(
            drift_h_target, sigma_h_target, p_low, p_high
        )

    # Affine re-center / re-scale to the target moments.
    emp_mean = float(np.mean(sims_sum))
    emp_std = float(np.std(sims_sum))
    if emp_std > _VOL_FLOOR and sigma_h_target > _VOL_FLOOR:
        z = (sims_sum - emp_mean) / emp_std
        sims_adj = drift_h_target + sigma_h_target * z
    else:
        sims_adj = drift_h_target + (sims_sum - emp_mean)

    low_log = float(np.quantile(sims_adj, p_low))
    high_log = float(np.quantile(sims_adj, p_high))
    prob_positive = float(np.mean(sims_adj > 0.0))

    alpha = 0.05
    thresh = float(np.quantile(sims_adj, alpha))
    tail = sims_adj[sims_adj <= thresh]
    es_log = float(np.mean(tail)) if tail.size else thresh

    return low_log, high_log, prob_positive, es_log


def _expm1_pct(log_growth: float) -> float:
    """``(exp(log_growth) - 1) * 100`` with overflow / NaN guards."""
    g = _safe_float(log_growth, 0.0)
    g = max(-700.0, min(700.0, g))
    pct = (math.exp(g) - 1.0) * 100.0
    return _clamp_pct(pct)


# ---------------------------------------------------------------------------
# The projection (R1 + R2 + R6)
# ---------------------------------------------------------------------------


def project(
    closes: np.ndarray | list[float],
    returns: np.ndarray | list[float],
    signal_drifts: list[tuple[float, float]],
    rf_daily: float,
    beta: float | None = None,
    capm_drift_daily: float | None = None,
    asset_class: str | None = "equity",
) -> tuple[list[HorizonProjection], dict]:
    """Produce credible 5-horizon projections + a regime classification.

    Pipeline (STRATEGIES-V2 §4 + R1/R2/R6):

        1. **Regime.** :func:`detect_regime` from ``closes``.
        2. **Drift (R1).** :func:`ensemble_drift` ⇒ a capped, shrunk,
           regime-tilted daily log-drift. The shrinkage prior is the CAPM daily
           drift (``capm_drift_daily`` if given, else built from ``rf_daily`` +
           ``beta`` · ERP).
        3. **Vol.** :func:`forward_vol` (GARCH→EWMA) per horizon (the GARCH
           forecast embeds the vol term structure); the *horizon* sigma is the
           annualized vol scaled by ``sqrt(h/252)``.
        4. **Bands (R2).** Fat-tailed Student-t (df≈5) quantiles, or a
           re-centered block-bootstrap when ≥ ``_MIN_BOOTSTRAP_OBS`` returns
           exist. ``low`` floored at −95%, ``high`` capped per horizon.
        5. **Scenarios.** ``base`` = drift projection; ``bull`` = base + z·σ_h,
           ``bear`` = base − z·σ_h (z grows mildly with horizon).
        6. **prob_positive** from the fat-tailed dist; **cvar_pct** = 95%
           expected shortfall (a positive loss percentage) — R6.

    Args:
        closes: Daily closing prices (most recent last).
        returns: Daily returns of the asset (for vol / bootstrap).
        signal_drifts: ``[(daily_drift, confidence), ...]`` from projecting
            strategies (may be empty).
        rf_daily: Daily risk-free rate (decimal).
        beta: Asset market beta (default 1.0 when ``None``) — used to build the
            CAPM prior if ``capm_drift_daily`` is not supplied.
        capm_drift_daily: The CAPM daily log-drift prior; built from
            ``rf_daily``/``beta`` when ``None``.
        asset_class: ``'equity' | 'crypto' | 'etf'`` (drives the R1 drift cap and
            the per-horizon high cap leniency).

    Returns:
        ``(projections, regime)`` where ``projections`` is one
        :class:`HorizonProjection` per horizon in :data:`HORIZON_DAYS` order and
        ``regime`` is the :func:`detect_regime` dict. Never raises; all numbers
        finite.
    """
    closes_arr = _clean_closes(closes)
    rets_arr = _clean_returns(returns)

    regime = detect_regime(closes_arr)

    # --- CAPM shrinkage prior (daily log-drift) ---
    rf_d = _safe_float(rf_daily, 0.0)
    if capm_drift_daily is not None and math.isfinite(_safe_float(capm_drift_daily, float("nan"))):
        prior_daily = _safe_float(capm_drift_daily, 0.0)
    else:
        b = _safe_float(beta, 1.0) if beta is not None else 1.0
        rf_annual = daily_to_annual_drift(rf_d) if rf_d != 0.0 else _DEFAULT_RF_ANNUAL
        capm_annual = rf_annual + b * _ERP_ANNUAL
        prior_daily = annual_to_daily_drift(capm_annual)

    # Defensive cap on the prior itself (so a wild beta cannot pollute it).
    lo_cap, hi_cap = _drift_cap(asset_class)
    prior_annual = max(lo_cap, min(hi_cap, daily_to_annual_drift(prior_daily)))
    prior_daily = annual_to_daily_drift(prior_annual)

    # --- R1 drift: ensemble → shrink → tilt → cap ---
    drift_daily = ensemble_drift(
        signal_drifts, prior=prior_daily, asset_class=asset_class, regime=regime
    )
    # Expose the EXACT drift used so callers (the engine's montecarlo()) can run
    # the Monte Carlo from the same number rather than recomputing the ensemble
    # with a possibly-different prior (R3). Stored on the regime dict it returns.
    regime = dict(regime)
    regime["drift_daily"] = _safe_float(drift_daily, 0.0)

    use_bootstrap = rets_arr.size >= _MIN_BOOTSTRAP_OBS

    # Precompute per-horizon volatility. The GARCH forecast returns the *average*
    # annualized vol over the horizon, which mean-reverts toward the long-run
    # level — so the annualized rate can fall as the horizon lengthens. The
    # cumulative log-return uncertainty over the horizon is
    # ``sigma_h = ann_vol_h * sqrt(h / 252)``. Total variance must *accumulate*
    # with horizon (a longer holding period is never more certain in absolute
    # terms), so we enforce a running max on ``sigma_h``; this both reflects
    # accumulating uncertainty and guarantees the confidence band widens with the
    # horizon (§9) even when the annualized GARCH rate mean-reverts down.
    horizon_sigma: dict[str, float] = {}
    horizon_annvol: dict[str, float] = {}
    running_sigma = 0.0
    # Fit GARCH ONCE and read the per-horizon annualized vol from the single fit
    # (numerically identical to calling ``forward_vol`` per horizon, ~1/5 the
    # cost — the GARCH fit dominates ``project``).
    annvol_by_h = forward_vol_term_structure(rets_arr, list(HORIZON_DAYS.values()))
    for label, h in HORIZON_DAYS.items():
        ann_vol_h = max(_VOL_FLOOR, _safe_float(annvol_by_h.get(h), _VOL_FLOOR))
        sigma_h = max(_VOL_FLOOR, ann_vol_h * math.sqrt(h / float(TRADING_DAYS)))
        running_sigma = max(running_sigma, sigma_h)
        horizon_sigma[label] = running_sigma
        # Report the annualized vol consistent with the (monotone) horizon sigma.
        horizon_annvol[label] = running_sigma / math.sqrt(h / float(TRADING_DAYS))

    projections: list[HorizonProjection] = []
    for label, h in HORIZON_DAYS.items():
        sigma_h = horizon_sigma[label]
        ann_vol_h = horizon_annvol[label]

        # ``drift_daily`` is the mean daily log return (the GBM ``mu``). Two
        # central estimates matter and they must NOT be conflated:
        #   * the MEDIAN log path drifts at ``mu - 0.5*sigma^2`` per day, i.e.
        #     ``median_log = drift_daily*h - 0.5*sigma_h^2`` over the horizon —
        #     this is the right centre for the symmetric-in-log fat-tailed bands
        #     and the bull/bear fan (so the 5/95 band straddles the median path);
        #   * the MEAN (expected value) of the terminal return is
        #     ``E[S_T/S_0] - 1 = exp(drift_daily*h) - 1`` (the +0.5*sigma^2*h Ito
        #     term lifts the mean above the median for a lognormal).
        # The DISPLAYED ``expected_return_pct`` is the MEAN, because that is what
        # ``montecarlo_summary`` reports (mean of terminal returns) — using the
        # mean here is what makes the analysis 1Y and the Monte-Carlo 1Y expected
        # returns agree within ~1pp (R3). The bands/fan stay centred on the median.
        median_log = drift_daily * h - 0.5 * sigma_h * sigma_h
        mean_log = drift_daily * h
        high_cap_pct = _HIGH_CAP_BY_HORIZON.get(label, 4.0) * 100.0
        # Clamp BOTH central estimates into the credible band FIRST, so every value
        # derived from them (the ordered low/high and the bull/bear scenarios) is
        # guaranteed to stay within ``[-95%, high_cap]`` no matter how the raw
        # drift/quantiles land. The drift cap (R1) already keeps these sane; this
        # is the last-line guard that makes the per-horizon bounds airtight.
        median_pct = min(high_cap_pct, max(_LOW_FLOOR_PCT, _expm1_pct(median_log)))
        mean_pct = min(high_cap_pct, max(_LOW_FLOOR_PCT, _expm1_pct(mean_log)))
        # ``base_pct`` (the scenario-fan centre) is the displayed expected return
        # (the mean) so the user reads one consistent central number.
        base_pct = mean_pct

        # --- fat-tailed bands (centred on the median log path) ---
        p_low, p_high = 0.05, 0.95
        if use_bootstrap:
            low_log, high_log, prob_pos, es_log = _bootstrap_log_quantiles(
                rets_arr, h, median_log, sigma_h, p_low, p_high,
                seed=_BOOTSTRAP_SEED + h,
            )
        else:
            low_log, high_log, prob_pos, es_log = _student_t_log_quantiles(
                median_log, sigma_h, p_low, p_high
            )

        # R2 caps: floor the low at −95% (long-only), cap the high to a credible
        # per-horizon bound, and keep the band ordered around the (capped) base
        # (the mean): low ≤ base ≤ high.
        low_pct = max(_LOW_FLOOR_PCT, min(base_pct, _expm1_pct(low_log)))
        high_pct = min(high_cap_pct, max(base_pct, _expm1_pct(high_log)))

        # --- scenario fan (z grows mildly with horizon; centred on the median) ---
        z_h = _Z_90 * (1.0 + 0.10 * math.log1p(h / float(TRADING_DAYS)))
        bull_log = median_log + z_h * sigma_h
        bear_log = median_log - z_h * sigma_h
        # bull ≥ base ≥ bear, and both stay inside the credible bounds.
        bull_pct = min(high_cap_pct, max(base_pct, _expm1_pct(bull_log)))
        bear_pct = max(_LOW_FLOOR_PCT, min(base_pct, _expm1_pct(bear_log)))

        prob_pos = min(1.0, max(0.0, _safe_float(prob_pos, 0.5)))

        # --- CVaR (R6): 95% expected shortfall as a positive loss % ---
        es_pct = _expm1_pct(es_log)             # expected tail *return* (negative)
        cvar_pct = max(0.0, -es_pct)
        cvar_pct = min(100.0, cvar_pct)         # long-only: loss ≤ 100%

        ann_vol_pct = ann_vol_h * 100.0

        projections.append(
            HorizonProjection(
                horizon=label,
                expected_return_pct=base_pct,
                low=low_pct,
                high=high_pct,
                prob_positive=prob_pos,
                annualized_vol=ann_vol_pct,
                bull_pct=bull_pct,
                base_pct=base_pct,
                bear_pct=bear_pct,
                cvar_pct=cvar_pct,
            )
        )

    return projections, regime


# ---------------------------------------------------------------------------
# Monte Carlo summary that AGREES with project() (R3)
# ---------------------------------------------------------------------------


def mc_summary(
    s0: float,
    drift_daily: float,
    vol_daily: float,
    horizon: str,
    sims: int = _MC_SIMS_DEFAULT,
    seed: int = 0,
) -> dict:
    """Run a GBM Monte Carlo from the SAME drift + vol the projection uses (R3).

    Thin wrapper over :func:`app.quant.montecarlo.montecarlo_summary` so the
    engine can drive its ``/montecarlo`` endpoint from the exact drift and daily
    volatility that :func:`project` derived (the capped/shrunk daily log-drift,
    and the horizon's annualized GARCH vol converted back to a *daily* vol). This
    guarantees the analysis 1Y expected return and the Monte-Carlo 1Y expected
    return agree within ~1pp (R3), instead of disagreeing as in the live audit.

    The output dict matches the ``MonteCarloResult`` wire shape (minus
    ``symbol``, which the caller fills in).

    Args:
        s0: Starting price (floored to a tiny positive value if non-positive).
        drift_daily: Mean daily log return — pass ``drift_daily`` from
            :func:`project` (i.e. ``ensemble_drift(...)``).
        vol_daily: Daily volatility — pass the horizon's annualized
            :func:`forward_vol` divided by ``sqrt(252)`` so MC and ``project``
            share the same vol.
        horizon: One of ``'1D' | '1W' | '1M' | '1Y' | '5Y'`` (unknown → ``'1Y'``).
        sims: Number of simulated paths (coerced to ``>= 1``).
        seed: RNG seed for reproducibility.

    Returns:
        A ``MonteCarloResult``-shaped dict; all numbers finite, ``probPositive``
        in ``[0, 1]``, percentages clamped.
    """
    mu = _safe_float(drift_daily, 0.0)
    sigma = _safe_float(vol_daily, 0.0)
    if sigma <= 0.0:
        sigma = _VOL_FLOOR
    s0_f = _safe_float(s0, 1.0)
    if s0_f <= 0.0:
        s0_f = 1.0
    hz = horizon if horizon in HORIZON_DAYS else "1Y"
    return montecarlo.montecarlo_summary(
        s0=s0_f,
        mu_daily=mu,
        sigma_daily=sigma,
        horizon=hz,
        sims=max(1, int(sims)),
        seed=int(seed),
    )
