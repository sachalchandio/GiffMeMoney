"""The strategy catalog: metadata plus one signal builder per model.

This module is the single registry of the 20 quant models named in section 7 of
the contract. It exposes:

    * :data:`STRATEGY_META` — a :class:`~app.schemas.StrategyMeta` for every id,
      in catalog order (``id``, ``name``, ``category``, ``summary``, ``formula``,
      ``inputs``, ``references``).
    * :data:`SIGNAL_BUILDERS` — a ``dict[str, Callable[[AnalysisContext],
      StrategySignal]]`` mapping each id to a pure function that turns an
      :class:`~app.strategies.engine.AnalysisContext` into a
      :class:`~app.schemas.StrategySignal` using the real :mod:`app.quant`
      functions.
    * :func:`build_signals` — run every builder in catalog order, returning the
      full list of signals for one asset.

Every builder is defensive: it relies on the quant layer's own guards and wraps
its body so that a single model can never raise out of :func:`build_signals`
(the engine also guards, giving two layers of safety). A failed builder emits a
neutral ``HOLD`` signal so an :class:`~app.schemas.AssetAnalysis` always carries
one signal per registered strategy.

Score convention: **positive = bullish** everywhere. Builders that imply a
forward drift/vol attach 5-horizon projections via
:func:`app.quant.returns.project_horizons`; the others leave ``horizons`` empty
and the engine blends only the projecting ones.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import numpy as np

from app.quant import (
    capm,
    factor,
    forecast,
    fundamental,
    kelly,
    metrics,
    montecarlo,
    options,
    returns,
    risk,
    technical,
    valuation,
    volatility,
)
from app.schemas import StrategyMeta, StrategySignal
from app.strategies.base import (
    clamp,
    linear_score,
    make_signal,
    squash,
)

if TYPE_CHECKING:  # pragma: no cover - import only for static typing
    from app.strategies.engine import AnalysisContext

__all__ = [
    "STRATEGY_META",
    "SIGNAL_BUILDERS",
    "build_signals",
    "META_BY_ID",
]


# ---------------------------------------------------------------------------
# Catalog metadata (section 7 — all 20 ids, in order)
# ---------------------------------------------------------------------------

STRATEGY_META: list[StrategyMeta] = [
    StrategyMeta(
        id="capm",
        name="Capital Asset Pricing Model",
        category="Risk-Adjusted",
        summary=(
            "Prices required return from systematic risk: the higher an asset's "
            "CAPM expected return relative to its realized return, the more "
            "undervalued it looks."
        ),
        formula="E[R] = Rf + beta * (E[Rm] - Rf)",
        inputs=["asset returns", "market returns", "risk-free rate"],
        references=["Sharpe (1964)", "Lintner (1965)"],
    ),
    StrategyMeta(
        id="fama-french",
        name="Fama-French 3-Factor",
        category="Factor",
        summary=(
            "Regresses excess returns on market, size (SMB) and value (HML) "
            "factors; a positive, well-fit alpha signals factor-adjusted "
            "outperformance."
        ),
        formula="r - Rf = alpha + b_mkt*Mkt + b_smb*SMB + b_hml*HML + e",
        inputs=["asset excess returns", "Mkt-Rf", "SMB", "HML"],
        references=["Fama & French (1993)"],
    ),
    StrategyMeta(
        id="dcf",
        name="Discounted Cash Flow",
        category="Valuation",
        summary=(
            "Estimates intrinsic value from discounted free cash flows plus a "
            "terminal value; the margin of safety versus price drives the score."
        ),
        formula="V = sum_t FCF_t/(1+wacc)^t + TV/(1+wacc)^N",
        inputs=["FCF per share", "growth", "WACC", "price"],
        references=["Williams (1938)", "Damodaran"],
    ),
    StrategyMeta(
        id="ddm",
        name="Gordon Dividend Discount",
        category="Valuation",
        summary=(
            "Values a dividend payer as a growing perpetuity; the fair-value gap "
            "versus market price drives the score. Non-payers are neutral."
        ),
        formula="P = D1 / (r - g)",
        inputs=["dividend", "dividend growth", "required return", "price"],
        references=["Gordon (1959)"],
    ),
    StrategyMeta(
        id="markowitz",
        name="Mean-Variance (Markowitz)",
        category="Portfolio",
        summary=(
            "Scores an asset by its risk-adjusted excess return (return per unit "
            "variance), a proxy for its pull into the tangency portfolio."
        ),
        formula="contribution ~ (mu - rf) / variance",
        inputs=["expected return", "variance", "risk-free rate"],
        references=["Markowitz (1952)"],
    ),
    StrategyMeta(
        id="sharpe",
        name="Sharpe Ratio",
        category="Risk-Adjusted",
        summary=(
            "Total risk-adjusted return: excess return per unit of total "
            "volatility. A higher Sharpe is more bullish."
        ),
        formula="SR = (R_bar - Rf) / sigma",
        inputs=["asset returns", "risk-free rate"],
        references=["Sharpe (1966)"],
    ),
    StrategyMeta(
        id="sortino",
        name="Sortino Ratio",
        category="Risk-Adjusted",
        summary=(
            "Downside risk-adjusted return: excess return per unit of downside "
            "deviation, rewarding assets that limit losses."
        ),
        formula="Sortino = (R_bar - Rf) / sigma_downside",
        inputs=["asset returns", "risk-free rate"],
        references=["Sortino & Price (1994)"],
    ),
    StrategyMeta(
        id="momentum",
        name="12-1 Momentum",
        category="Technical",
        summary=(
            "Cross-sectional momentum: trailing 12-month return excluding the "
            "most recent month. Strong past winners tend to keep winning."
        ),
        formula="mom = P_{t-21} / P_{t-252} - 1",
        inputs=["price history"],
        references=["Jegadeesh & Titman (1993)"],
    ),
    StrategyMeta(
        id="mean-reversion",
        name="Mean Reversion (OU z-score)",
        category="Statistical",
        summary=(
            "Treats price as mean-reverting: a high positive z-score (stretched "
            "above the mean) is bearish, a depressed price is bullish."
        ),
        formula="z = (P - mu) / sigma  (signal = -z)",
        inputs=["price history"],
        references=["Ornstein-Uhlenbeck", "Poterba & Summers (1988)"],
    ),
    StrategyMeta(
        id="macd",
        name="MACD Crossover",
        category="Technical",
        summary=(
            "Trend/momentum oscillator: the MACD line versus its signal line. A "
            "positive histogram is bullish, a negative one bearish."
        ),
        formula="MACD = EMA12 - EMA26; Hist = MACD - EMA9(MACD)",
        inputs=["price history"],
        references=["Appel (1979)"],
    ),
    StrategyMeta(
        id="rsi",
        name="Relative Strength Index (14)",
        category="Technical",
        summary=(
            "Overbought/oversold oscillator over 14 periods: RSI < 30 is "
            "oversold (bullish), RSI > 70 overbought (bearish)."
        ),
        formula="RSI = 100 - 100 / (1 + avg_gain/avg_loss)",
        inputs=["price history"],
        references=["Wilder (1978)"],
    ),
    StrategyMeta(
        id="bollinger",
        name="Bollinger %B",
        category="Technical",
        summary=(
            "Position of price within its mean +-2 sigma band. Near the lower "
            "band (%B ~ 0) is bullish; near the upper band (%B ~ 1) is bearish."
        ),
        formula="%B = (P - lower) / (upper - lower)",
        inputs=["price history"],
        references=["Bollinger (1980s)"],
    ),
    StrategyMeta(
        id="montecarlo",
        name="Monte Carlo GBM",
        category="Statistical",
        summary=(
            "Simulates geometric-Brownian price paths to a 1-year horizon; the "
            "probability of a positive return drives the score."
        ),
        formula="S_{t+1} = S_t * exp((mu - sigma^2/2) + sigma*Z)",
        inputs=["drift", "volatility", "spot price"],
        references=["Boyle (1977)"],
    ),
    StrategyMeta(
        id="garch",
        name="GARCH(1,1) Volatility Regime",
        category="Statistical",
        summary=(
            "Forecasts conditional volatility; a forecast below recent realized "
            "vol (a calming regime) is mildly bullish on a risk-adjusted basis."
        ),
        formula="h_t = omega + alpha*eps_{t-1}^2 + beta*h_{t-1}",
        inputs=["asset returns"],
        references=["Bollerslev (1986)"],
    ),
    StrategyMeta(
        id="black-scholes",
        name="Black-Scholes Risk",
        category="Derivatives",
        summary=(
            "Prices an ATM 1-year call and reads its leverage (delta-implied "
            "exposure) as a conviction proxy under positive expected drift."
        ),
        formula="Call = S*Phi(d1) - K*e^{-rT}*Phi(d2)",
        inputs=["spot", "strike", "vol", "rate", "time"],
        references=["Black & Scholes (1973)", "Merton (1973)"],
    ),
    StrategyMeta(
        id="var",
        name="Value at Risk / CVaR",
        category="Risk-Adjusted",
        summary=(
            "Quantifies 95% tail risk (historical VaR and CVaR). Heavier tail "
            "losses penalize the score; contained tails are mildly supportive."
        ),
        formula="VaR_95 = -Quantile_{0.05}(returns)",
        inputs=["asset returns"],
        references=["JP Morgan RiskMetrics (1996)"],
    ),
    StrategyMeta(
        id="kelly",
        name="Kelly Criterion",
        category="Risk-Adjusted",
        summary=(
            "Growth-optimal position size for a Gaussian return process. A large "
            "positive Kelly fraction is bullish; negative implies a short."
        ),
        formula="f* = mu / sigma^2",
        inputs=["drift", "volatility"],
        references=["Kelly (1956)", "Thorp (1969)"],
    ),
    StrategyMeta(
        id="piotroski",
        name="Piotroski F-Score",
        category="Fundamental",
        summary=(
            "A 9-point accounting-quality screen (profitability, leverage, "
            "efficiency). 8-9 is strong quality; 0-2 is weak."
        ),
        formula="F = sum of 9 binary fundamental criteria (0..9)",
        inputs=["fundamentals (ROA, FCF, leverage, margins, ...)"],
        references=["Piotroski (2000)"],
    ),
    StrategyMeta(
        id="altman-z",
        name="Altman Z-Score",
        category="Fundamental",
        summary=(
            "Bankruptcy-distance score from five balance-sheet ratios. Z > 2.99 "
            "is the safe zone; Z < 1.81 is distress."
        ),
        formula="Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MV/TL + 1.0*S/TA",
        inputs=["fundamentals (WC, RE, EBIT, TA, TL, Sales)", "market cap"],
        references=["Altman (1968)"],
    ),
    StrategyMeta(
        id="trend-ols",
        name="OLS Trend + Holt-Winters",
        category="Statistical",
        summary=(
            "Fits a least-squares trend to log price and a Holt-Winters forecast; "
            "a positive, well-fit slope is bullish."
        ),
        formula="ln(P_t) = a + b*t;  forecast = level + h*trend",
        inputs=["price history"],
        references=["Holt (1957)", "Winters (1960)"],
    ),
]

#: Fast lookup of metadata by strategy id.
META_BY_ID: dict[str, StrategyMeta] = {m.id: m for m in STRATEGY_META}


# ---------------------------------------------------------------------------
# Small shared helpers used by the builders
# ---------------------------------------------------------------------------

def _market_excess(ctx: "AnalysisContext") -> np.ndarray:
    """Return the market *excess* daily return series (Mkt - rf).

    The simulator's ``mkt`` factor is a market *total* return, so the CAPM/FF
    market factor is recovered by subtracting the daily risk-free rate.

    Args:
        ctx: The analysis context.

    Returns:
        A ``float64`` array of market excess returns (possibly empty).
    """
    mkt = np.asarray(ctx.market_ret, dtype=np.float64).ravel()
    if mkt.size == 0:
        return mkt
    return mkt - float(ctx.rf_daily)


def _drift_vol(ctx: "AnalysisContext") -> tuple[float, float]:
    """Estimate the daily log-drift and daily volatility from history.

    Drift is the mean log return; volatility is the population std of log
    returns. Both are finite (NaN/inf collapse to ``0.0`` / a tiny floor).

    Args:
        ctx: The analysis context.

    Returns:
        A ``(mu_daily, sigma_daily)`` tuple of finite floats.
    """
    lr = returns.log_returns(ctx.closes)
    if lr.size == 0:
        return 0.0, 1e-4
    mu = float(np.mean(lr))
    sigma = float(np.std(lr))
    if not math.isfinite(mu):
        mu = 0.0
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = 1e-4
    return mu, sigma


def _neutral_signal(strategy_id: str, reason: str) -> StrategySignal:
    """Build a neutral ``HOLD`` signal for a strategy that could not run.

    Args:
        strategy_id: The strategy id (must exist in :data:`META_BY_ID`).
        reason: Short explanation included in the rationale.

    Returns:
        A zero-score, low-confidence :class:`~app.schemas.StrategySignal`.
    """
    meta = META_BY_ID[strategy_id]
    return make_signal(
        strategy_id=meta.id,
        name=meta.name,
        category=meta.category,
        score=0.0,
        confidence=0.1,
        rationale=f"Insufficient or degenerate data for {meta.name}: {reason}.",
        formula=meta.formula,
        metrics={},
        horizons=[],
    )


def _project(mu_daily: float, sigma_daily: float) -> list[dict]:
    """Project the 5 horizons from a daily drift/vol (thin wrapper).

    Args:
        mu_daily: Mean daily log return.
        sigma_daily: Daily volatility.

    Returns:
        The list of horizon dicts from :func:`app.quant.returns.project_horizons`.
    """
    return returns.project_horizons(mu_daily, sigma_daily)


# ---------------------------------------------------------------------------
# Signal builders (one per strategy id)
# ---------------------------------------------------------------------------

def _build_capm(ctx: "AnalysisContext") -> StrategySignal:
    """CAPM signal: CAPM-required return vs. realized return.

    Compares the asset's CAPM expected annual return
    ``E[R] = Rf + beta*(E[Rm]-Rf)`` against its realized annualized return.
    A realized return *above* the CAPM requirement is bullish (the asset has
    been beating its risk-priced hurdle). Horizons project the CAPM-implied
    drift at the asset's realized volatility.
    """
    meta = META_BY_ID["capm"]
    b = metrics.beta(ctx.returns, ctx.market_ret)
    rf_annual = float(ctx.rf_daily) * returns.TRADING_DAYS
    mkt_excess = _market_excess(ctx)
    if mkt_excess.size:
        market_premium = float(np.mean(mkt_excess)) * returns.TRADING_DAYS
    else:
        market_premium = 0.0
    capm_er = capm.capm_expected_return(b, rf_annual, market_premium)

    lr = returns.log_returns(ctx.closes)
    realized_annual = returns.annualize_return(float(np.mean(lr))) if lr.size else 0.0

    alpha = realized_annual - capm_er  # excess of realized over required
    score = squash(alpha, scale=0.15)  # ~+-15% annual alpha saturates
    confidence = clamp(0.4 + 0.4 * min(1.0, abs(b)), 0.0, 1.0)

    # Horizon drift = CAPM-implied daily log drift; vol = realized daily vol.
    _, sigma_daily = _drift_vol(ctx)
    capm_daily_drift = math.log1p(capm_er) / returns.TRADING_DAYS if capm_er > -1.0 else 0.0
    horizons = _project(capm_daily_drift, sigma_daily)

    rationale = (
        f"Beta {b:.2f} implies a CAPM hurdle of {capm_er * 100:.1f}% per year; the "
        f"asset realized {realized_annual * 100:.1f}%, an alpha of "
        f"{alpha * 100:+.1f}%."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "beta": b,
            "capmExpectedReturn": capm_er,
            "realizedReturn": realized_annual,
            "alpha": alpha,
            "marketPremium": market_premium,
        },
        horizons=horizons,
    )


def _build_fama_french(ctx: "AnalysisContext") -> StrategySignal:
    """Fama-French 3-factor signal: factor-adjusted alpha drives the score.

    Regresses the asset's excess returns on Mkt-Rf, SMB and HML; a positive
    annualized alpha (scaled by the fit's R^2 as confidence) is bullish.
    """
    meta = META_BY_ID["fama-french"]
    asset_excess = np.asarray(ctx.returns, dtype=np.float64).ravel() - float(ctx.rf_daily)
    mkt_excess = _market_excess(ctx)
    res = factor.fama_french_3factor(asset_excess, mkt_excess, ctx.smb, ctx.hml)

    score = squash(res.alpha_annual, scale=0.10)
    confidence = clamp(0.3 + 0.6 * res.r2, 0.0, 1.0)

    rationale = (
        f"Factor alpha of {res.alpha_annual * 100:+.1f}%/yr (R^2={res.r2:.2f}); "
        f"loadings Mkt {res.beta_mkt:.2f}, SMB {res.beta_smb:.2f}, "
        f"HML {res.beta_hml:.2f}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "alphaAnnual": res.alpha_annual,
            "betaMkt": res.beta_mkt,
            "betaSmb": res.beta_smb,
            "betaHml": res.beta_hml,
            "r2": res.r2,
        },
        horizons=[],
    )


def _build_dcf(ctx: "AnalysisContext") -> StrategySignal:
    """Discounted-cash-flow signal: margin of safety vs. price.

    Intrinsic value is a two-stage DCF on FCF per share discounted at a CAPM
    cost of equity. The margin of safety ``(intrinsic - price) / price`` drives
    the score. Assets without meaningful FCF (crypto/ETF) are neutral.
    """
    meta = META_BY_ID["dcf"]
    f = ctx.fundamentals
    price = float(ctx.asset.price)
    if f.fcf_per_share <= 0.0 or price <= 0.0:
        return _neutral_signal("dcf", "no positive free cash flow per share")

    b = metrics.beta(ctx.returns, ctx.market_ret)
    rf_annual = float(ctx.rf_daily) * returns.TRADING_DAYS
    mkt_excess = _market_excess(ctx)
    premium = float(np.mean(mkt_excess)) * returns.TRADING_DAYS if mkt_excess.size else 0.06
    wacc = capm.capm_expected_return(b, rf_annual, premium)
    # Keep WACC in a sensible band so the perpetuity is well behaved.
    wacc = clamp(wacc, 0.05, 0.20)
    growth = clamp(float(f.revenue_growth), -0.10, 0.25)

    intrinsic = valuation.dcf_intrinsic_value(
        fcf_per_share=float(f.fcf_per_share),
        growth=growth,
        wacc=wacc,
        terminal_growth=0.025,
        years=10,
    )
    margin = (intrinsic - price) / price if price > 0 else 0.0
    score = squash(margin, scale=0.5)  # +-50% margin of safety saturates
    confidence = clamp(0.35 + 0.35 * min(1.0, abs(margin)), 0.0, 1.0)

    rationale = (
        f"DCF intrinsic value {intrinsic:.2f} vs price {price:.2f} -> margin of "
        f"safety {margin * 100:+.1f}% (WACC {wacc * 100:.1f}%, growth "
        f"{growth * 100:.1f}%)."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "intrinsicValue": intrinsic,
            "price": price,
            "marginOfSafety": margin,
            "wacc": wacc,
            "growth": growth,
        },
        horizons=[],
    )


def _build_ddm(ctx: "AnalysisContext") -> StrategySignal:
    """Gordon dividend-discount-model signal: fair-value gap vs. price.

    Values a dividend payer as ``P = D1/(r-g)`` with a CAPM required return.
    The gap ``(fair - price)/price`` drives the score. Non-payers are neutral.
    """
    meta = META_BY_ID["ddm"]
    f = ctx.fundamentals
    price = float(ctx.asset.price)
    if f.dividend <= 0.0 or price <= 0.0:
        return _neutral_signal("ddm", "no dividend (DDM not applicable)")

    b = metrics.beta(ctx.returns, ctx.market_ret)
    rf_annual = float(ctx.rf_daily) * returns.TRADING_DAYS
    mkt_excess = _market_excess(ctx)
    premium = float(np.mean(mkt_excess)) * returns.TRADING_DAYS if mkt_excess.size else 0.06
    req = capm.capm_expected_return(b, rf_annual, premium)
    req = clamp(req, 0.05, 0.20)
    g = clamp(float(f.dividend_growth), 0.0, req - 0.01)

    fair = valuation.gordon_ddm(float(f.dividend), req, g)
    if fair <= 0.0:
        return _neutral_signal("ddm", "required return not above dividend growth")
    gap = (fair - price) / price
    score = squash(gap, scale=0.5)
    confidence = clamp(0.3 + 0.3 * min(1.0, abs(gap)), 0.0, 1.0)

    rationale = (
        f"Gordon fair value {fair:.2f} vs price {price:.2f} -> "
        f"{gap * 100:+.1f}% gap (required {req * 100:.1f}%, div growth "
        f"{g * 100:.1f}%)."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "fairValue": fair,
            "price": price,
            "gap": gap,
            "requiredReturn": req,
            "dividendGrowth": g,
        },
        horizons=[],
    )


def _build_markowitz(ctx: "AnalysisContext") -> StrategySignal:
    """Mean-variance signal: risk-adjusted excess return (return / variance).

    Scores an asset by ``(mu_annual - rf) / variance_annual`` — the quantity
    proportional to its unconstrained tangency-portfolio weight. A large
    positive value means the asset earns a lot per unit of variance and would be
    pulled heavily into the optimal portfolio.
    """
    meta = META_BY_ID["markowitz"]
    lr = returns.log_returns(ctx.closes)
    if lr.size < 2:
        return _neutral_signal("markowitz", "too few returns")
    mu_annual = returns.annualize_return(float(np.mean(lr)))
    vol_annual = metrics.annual_volatility(ctx.returns)
    rf_annual = float(ctx.rf_daily) * returns.TRADING_DAYS
    var_annual = max(vol_annual * vol_annual, 1e-6)
    contribution = (mu_annual - rf_annual) / var_annual
    score = squash(contribution, scale=3.0)
    confidence = clamp(0.4 + 0.3 * min(1.0, abs(contribution) / 3.0), 0.0, 1.0)

    rationale = (
        f"Excess return {(mu_annual - rf_annual) * 100:+.1f}%/yr over variance "
        f"{var_annual:.3f} gives a tangency pull of {contribution:.2f}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "expectedReturn": mu_annual,
            "variance": var_annual,
            "tangencyContribution": contribution,
        },
        horizons=[],
    )


def _build_sharpe(ctx: "AnalysisContext") -> StrategySignal:
    """Sharpe-ratio signal: annualized excess return per unit of total vol."""
    meta = META_BY_ID["sharpe"]
    sr = metrics.sharpe(ctx.returns, float(ctx.rf_daily))
    score = squash(sr, scale=1.0)  # SR of ~1 -> ~76 score
    confidence = clamp(0.4 + 0.4 * min(1.0, abs(sr)), 0.0, 1.0)
    rationale = f"Annualized Sharpe ratio of {sr:.2f} (excess return per unit of total volatility)."
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"sharpe": sr},
        horizons=[],
    )


def _build_sortino(ctx: "AnalysisContext") -> StrategySignal:
    """Sortino-ratio signal: excess return per unit of downside deviation."""
    meta = META_BY_ID["sortino"]
    so = metrics.sortino(ctx.returns, float(ctx.rf_daily))
    dd = metrics.downside_deviation(ctx.returns, mar=float(ctx.rf_daily))
    score = squash(so, scale=1.2)
    confidence = clamp(0.4 + 0.4 * min(1.0, abs(so) / 1.5), 0.0, 1.0)
    rationale = (
        f"Sortino ratio of {so:.2f}; downside deviation {dd * 100:.2f}% per day "
        f"(penalizes only losses below the risk-free rate)."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"sortino": so, "downsideDeviation": dd},
        horizons=[],
    )


def _build_momentum(ctx: "AnalysisContext") -> StrategySignal:
    """12-1 momentum signal: trailing 12m return ex the last month.

    A strong positive 12-1 momentum is bullish. Horizons project the momentum
    implied daily drift at realized volatility.
    """
    meta = META_BY_ID["momentum"]
    mom = technical.momentum_12_1(ctx.closes)
    score = squash(mom, scale=0.30)  # +-30% momentum saturates
    confidence = clamp(0.35 + 0.4 * min(1.0, abs(mom) / 0.5), 0.0, 1.0)

    _, sigma_daily = _drift_vol(ctx)
    # Annualize the 11-month momentum to a daily log drift (~231 trading days).
    if 1.0 + mom > 0.0:
        mom_daily = math.log1p(mom) / 231.0
    else:
        mom_daily = 0.0
    horizons = _project(clamp(mom_daily, -0.02, 0.02), sigma_daily)

    rationale = (
        f"12-1 momentum of {mom * 100:+.1f}% (trailing 12-month return excluding "
        f"the last month)."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"momentum12_1": mom},
        horizons=horizons,
    )


def _build_mean_reversion(ctx: "AnalysisContext") -> StrategySignal:
    """Mean-reversion signal: negative of the price z-score.

    A high positive z (price stretched above its mean) is bearish; a depressed
    price (negative z) is bullish, so ``score = squash(-z)``.
    """
    meta = META_BY_ID["mean-reversion"]
    z = technical.zscore(ctx.closes, n=60)
    score = squash(-z, scale=1.5)  # z of ~+-1.5 sigma -> ~+-76
    confidence = clamp(0.3 + 0.4 * min(1.0, abs(z) / 2.0), 0.0, 1.0)
    rationale = (
        f"Price z-score of {z:+.2f} vs its 60-day mean; "
        f"{'stretched high (bearish)' if z > 0 else 'depressed (bullish)'} on a "
        f"mean-reversion view."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"zscore": z},
        horizons=[],
    )


def _build_macd(ctx: "AnalysisContext") -> StrategySignal:
    """MACD signal: the latest histogram (MACD - signal), normalized by price."""
    meta = META_BY_ID["macd"]
    macd_line, signal_line, hist = technical.macd(ctx.closes)
    if hist.size == 0:
        return _neutral_signal("macd", "no price history")
    price = float(ctx.asset.price) if ctx.asset.price > 0 else 1.0
    latest_hist = float(hist[-1])
    latest_macd = float(macd_line[-1])
    latest_sig = float(signal_line[-1])
    # Normalize histogram by price so the score is scale-free across assets.
    norm_hist = latest_hist / price
    score = squash(norm_hist, scale=0.01)
    confidence = clamp(0.3 + 0.4 * min(1.0, abs(norm_hist) / 0.02), 0.0, 1.0)
    rationale = (
        f"MACD {latest_macd:.3f} vs signal {latest_sig:.3f} -> histogram "
        f"{latest_hist:+.3f} ({'bullish' if latest_hist > 0 else 'bearish'} "
        f"crossover momentum)."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "macd": latest_macd,
            "signal": latest_sig,
            "histogram": latest_hist,
        },
        horizons=[],
    )


def _build_rsi(ctx: "AnalysisContext") -> StrategySignal:
    """RSI(14) signal: oversold (<30) bullish, overbought (>70) bearish.

    The RSI is centered at 50 and mapped so 30 -> strongly bullish and 70 ->
    strongly bearish: ``score = squash((50 - RSI))``.
    """
    meta = META_BY_ID["rsi"]
    value = technical.rsi(ctx.closes, n=14)
    # Distance from neutral 50, inverted (low RSI = bullish).
    score = squash(50.0 - value, scale=20.0)
    # Confidence rises as RSI nears an extreme.
    confidence = clamp(0.3 + 0.5 * min(1.0, abs(value - 50.0) / 30.0), 0.0, 1.0)
    if value < 30.0:
        tone = "oversold (bullish)"
    elif value > 70.0:
        tone = "overbought (bearish)"
    else:
        tone = "neutral"
    rationale = f"RSI(14) at {value:.1f} -> {tone}."
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"rsi": value},
        horizons=[],
    )


def _build_bollinger(ctx: "AnalysisContext") -> StrategySignal:
    """Bollinger %B signal: near lower band bullish, near upper band bearish.

    %B is 0 at the lower band and 1 at the upper band; centered at 0.5. Score is
    ``squash(0.5 - %B)`` so a price near the lower band scores bullish.
    """
    meta = META_BY_ID["bollinger"]
    mid, upper, lower, percent_b = technical.bollinger(ctx.closes, n=20, k=2.0)
    score = squash(0.5 - percent_b, scale=0.4)
    confidence = clamp(0.3 + 0.4 * min(1.0, abs(percent_b - 0.5) / 0.5), 0.0, 1.0)
    rationale = (
        f"Bollinger %B at {percent_b:.2f} (band {lower:.2f}-{upper:.2f}, mid "
        f"{mid:.2f}); "
        f"{'near lower band (bullish)' if percent_b < 0.5 else 'near upper band (bearish)'}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "percentB": percent_b,
            "mid": mid,
            "upper": upper,
            "lower": lower,
        },
        horizons=[],
    )


def _build_montecarlo(ctx: "AnalysisContext") -> StrategySignal:
    """Monte Carlo GBM signal: probability of a positive 1-year return.

    Runs a GBM simulation to 1 year from the realized drift/vol; the score maps
    the probability of a positive return around the neutral 50%. Horizons reuse
    the realized drift/vol projection.
    """
    meta = META_BY_ID["montecarlo"]
    mu_daily, sigma_daily = _drift_vol(ctx)
    s0 = float(ctx.asset.price) if ctx.asset.price > 0 else 1.0
    summary = montecarlo.montecarlo_summary(
        s0=s0,
        mu_daily=mu_daily,
        sigma_daily=sigma_daily,
        horizon="1Y",
        sims=1500,
        seed=abs(hash(ctx.asset.symbol)) % (2**32),
    )
    prob_pos = float(summary["probPositive"])
    exp_ret = float(summary["expectedReturnPct"])
    # Map probability (0.5 neutral) to a score.
    score = squash(prob_pos - 0.5, scale=0.15)
    confidence = clamp(0.4 + 0.4 * abs(prob_pos - 0.5) * 2.0, 0.0, 1.0)
    horizons = _project(mu_daily, sigma_daily)
    rationale = (
        f"Monte Carlo (1500 GBM paths) puts the 1-year probability of a gain at "
        f"{prob_pos * 100:.0f}% with a mean return of {exp_ret:+.1f}%."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "probPositive": prob_pos,
            "expectedReturnPct": exp_ret,
            "var95Pct": float(summary["var95Pct"]),
            "cvar95Pct": float(summary["cvar95Pct"]),
        },
        horizons=horizons,
    )


def _build_garch(ctx: "AnalysisContext") -> StrategySignal:
    """GARCH(1,1) signal: a calming vol regime is mildly bullish.

    Forecasts 21-day-ahead annualized volatility and compares it to recent
    realized annualized volatility. A forecast *below* realized vol (vol falling)
    is mildly bullish on a risk-adjusted basis; rising vol is mildly bearish.
    """
    meta = META_BY_ID["garch"]
    r = np.asarray(ctx.returns, dtype=np.float64).ravel()
    if r.size < 30:
        return _neutral_signal("garch", "need >= 30 returns to fit GARCH")
    forecast_vol = volatility.garch11_forecast(r, horizon_days=21)
    realized_vol = metrics.annual_volatility(r)
    if realized_vol <= 0.0:
        return _neutral_signal("garch", "zero realized volatility")
    # Relative vol change: negative = calming (bullish).
    rel_change = (forecast_vol - realized_vol) / realized_vol
    score = squash(-rel_change, scale=0.25)
    confidence = clamp(0.3 + 0.3 * min(1.0, abs(rel_change) / 0.5), 0.0, 1.0)
    omega, alpha, beta = volatility.garch11_fit(r)
    rationale = (
        f"GARCH forecasts {forecast_vol * 100:.1f}% annualized vol vs realized "
        f"{realized_vol * 100:.1f}% ({rel_change * 100:+.1f}% change; "
        f"persistence alpha+beta={alpha + beta:.2f})."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "forecastVol": forecast_vol,
            "realizedVol": realized_vol,
            "volChange": rel_change,
            "alpha": alpha,
            "beta": beta,
        },
        horizons=[],
    )


def _build_black_scholes(ctx: "AnalysisContext") -> StrategySignal:
    """Black-Scholes signal: ATM 1Y call leverage as a conviction proxy.

    Prices an at-the-money 1-year call on the asset (strike = spot) at its
    realized annualized volatility and reads the call's delta as the option's
    bullish exposure. Combined with the realized drift sign, a high-delta call on
    a positively-drifting asset is a conviction-bullish read.
    """
    meta = META_BY_ID["black-scholes"]
    s = float(ctx.asset.price) if ctx.asset.price > 0 else 1.0
    k = s  # at the money
    sigma = metrics.annual_volatility(ctx.returns)
    sigma = clamp(sigma, 0.01, 2.0)
    r_annual = float(ctx.rf_daily) * returns.TRADING_DAYS
    call = options.black_scholes(s, k, 1.0, r_annual, sigma, "call")
    greeks = options.bs_greeks(s, k, 1.0, r_annual, sigma, "call")
    delta = float(greeks["delta"])
    # Conviction = call value as a % of spot, signed by realized drift.
    mu_daily, _ = _drift_vol(ctx)
    call_pct = call / s if s > 0 else 0.0
    signed = call_pct * (1.0 if mu_daily >= 0 else -1.0)
    score = squash(signed, scale=0.12)
    confidence = clamp(0.3 + 0.4 * delta, 0.0, 1.0)
    rationale = (
        f"ATM 1Y call worth {call:.2f} ({call_pct * 100:.1f}% of spot) with delta "
        f"{delta:.2f} at {sigma * 100:.0f}% implied vol; "
        f"{'bullish' if signed >= 0 else 'bearish'} leverage read."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "callValue": call,
            "callPctOfSpot": call_pct,
            "delta": delta,
            "impliedVol": sigma,
            "vega": float(greeks["vega"]),
        },
        horizons=[],
    )


def _build_var(ctx: "AnalysisContext") -> StrategySignal:
    """VaR/CVaR signal: tail risk penalizes the score.

    Computes the 95% historical VaR and CVaR (positive loss fractions). Higher
    tail losses are bearish (risk-adjusted); a contained tail is mildly
    supportive. Score maps daily VaR around a ~2% reference loss.
    """
    meta = META_BY_ID["var"]
    var95 = risk.historical_var(ctx.returns, conf=0.95)
    cv = risk.cvar(ctx.returns, conf=0.95)
    pvar = risk.parametric_var(ctx.returns, conf=0.95)
    # 2% daily VaR is "typical"; deeper tails score negative.
    score = linear_score(var95, lo=0.0, hi=0.05, invert=True)
    confidence = clamp(0.4 + 0.3 * min(1.0, var95 / 0.05), 0.0, 1.0)
    rationale = (
        f"95% daily VaR of {var95 * 100:.2f}% (CVaR {cv * 100:.2f}%); tail risk "
        f"{'elevated' if var95 > 0.03 else 'contained'}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "var95": var95,
            "cvar95": cv,
            "parametricVar95": pvar,
        },
        horizons=[],
    )


def _build_kelly(ctx: "AnalysisContext") -> StrategySignal:
    """Kelly-criterion signal: growth-optimal fraction f* = mu/sigma^2.

    A large positive Kelly fraction (high drift relative to variance) is
    bullish; a negative fraction implies a short. Horizons reuse the realized
    drift/vol projection.
    """
    meta = META_BY_ID["kelly"]
    mu_daily, sigma_daily = _drift_vol(ctx)
    f_star = kelly.kelly_fraction(mu_daily, sigma_daily)
    score = squash(f_star, scale=1.0)  # f* of 1 (full Kelly) -> ~76
    confidence = clamp(0.35 + 0.4 * min(1.0, abs(f_star) / 2.0), 0.0, 1.0)
    horizons = _project(mu_daily, sigma_daily)
    rationale = (
        f"Kelly fraction f* = {f_star:+.2f} (drift {mu_daily * returns.TRADING_DAYS * 100:+.1f}%/yr "
        f"over variance); "
        f"{'lever long' if f_star > 0 else 'reduce/short'}."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"kellyFraction": f_star},
        horizons=horizons,
    )


def _build_piotroski(ctx: "AnalysisContext") -> StrategySignal:
    """Piotroski F-Score signal: 0..9 accounting-quality score.

    Maps the F-Score (0-9) onto ``[-100, 100]`` centered at the neutral 4.5:
    8-9 is strong quality (bullish), 0-2 weak (bearish). Crypto/ETF seeds with
    empty fundamentals score low-quality but at reduced confidence.
    """
    meta = META_BY_ID["piotroski"]
    f = ctx.fundamentals
    fscore = fundamental.piotroski_score(f)
    # Center at 4.5: F=9 -> +100, F=0 -> -100.
    score = linear_score(float(fscore), lo=0.0, hi=9.0)
    # Lower confidence for assets without real fundamentals (e.g. crypto/ETF).
    has_fundamentals = abs(f.total_assets) > 1.0 or f.eps != 0.0
    base_conf = 0.6 if has_fundamentals else 0.2
    confidence = clamp(base_conf, 0.0, 1.0)
    rationale = (
        f"Piotroski F-Score of {fscore}/9 -> "
        f"{'strong' if fscore >= 7 else 'weak' if fscore <= 3 else 'moderate'} "
        f"fundamental quality."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"fScore": float(fscore)},
        horizons=[],
    )


def _build_altman_z(ctx: "AnalysisContext") -> StrategySignal:
    """Altman Z-Score signal: distance from financial distress.

    Z > 2.99 is the safe zone (bullish), Z < 1.81 is distress (bearish). Score
    maps Z linearly across that band centered at the 2.4 grey-zone midpoint.
    Debt-free crypto/ETF seeds yield a neutral safe-zone read at low confidence.
    """
    meta = META_BY_ID["altman-z"]
    f = ctx.fundamentals
    mc = float(ctx.market_cap) if ctx.market_cap else 0.0
    z = fundamental.altman_z(f, mc)
    # Map distress (1.0) -> bearish, safe (4.0) -> bullish; center ~2.4.
    score = linear_score(z, lo=1.0, hi=4.0)
    has_fundamentals = abs(f.total_assets) > 1.0 and f.total_liabilities > 0.0
    base_conf = 0.55 if has_fundamentals else 0.2
    confidence = clamp(base_conf, 0.0, 1.0)
    if z > 2.99:
        zone = "safe zone"
    elif z >= 1.81:
        zone = "grey zone"
    else:
        zone = "distress zone"
    rationale = f"Altman Z-Score of {z:.2f} ({zone})."
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={"zScore": z},
        horizons=[],
    )


def _build_trend_ols(ctx: "AnalysisContext") -> StrategySignal:
    """OLS-trend + Holt-Winters signal: log-price slope and forecast drift.

    Fits an OLS line to log price (slope ~ daily log drift) and a Holt-Winters
    forecast. A positive, well-fit slope is bullish; the R^2 sets confidence.
    Horizons project the OLS drift at realized volatility.
    """
    meta = META_BY_ID["trend-ols"]
    slope, intercept, r2, drift = forecast.ols_trend(ctx.closes)
    level, trend, fc_value = forecast.holt_winters(ctx.closes, horizon=21)
    price = float(ctx.asset.price) if ctx.asset.price > 0 else 1.0
    hw_change = (fc_value - price) / price if price > 0 else 0.0
    # Annualized OLS drift drives the score.
    annual_drift = drift * returns.TRADING_DAYS
    score = squash(annual_drift, scale=0.25)
    confidence = clamp(0.3 + 0.6 * r2, 0.0, 1.0)
    _, sigma_daily = _drift_vol(ctx)
    horizons = _project(drift, sigma_daily)
    rationale = (
        f"OLS log-price trend of {annual_drift * 100:+.1f}%/yr (R^2={r2:.2f}); "
        f"Holt-Winters 21-day forecast implies {hw_change * 100:+.1f}% vs spot."
    )
    return make_signal(
        meta.id, meta.name, meta.category, score, confidence, rationale, meta.formula,
        metrics={
            "slope": slope,
            "r2": r2,
            "driftDaily": drift,
            "holtWintersForecast": fc_value,
            "holtWintersChange": hw_change,
        },
        horizons=horizons,
    )


# ---------------------------------------------------------------------------
# Registry assembly
# ---------------------------------------------------------------------------

#: One builder per strategy id (positive score = bullish). Order mirrors
#: :data:`STRATEGY_META` so :func:`build_signals` runs them in catalog order.
SIGNAL_BUILDERS: dict[str, Callable[["AnalysisContext"], StrategySignal]] = {
    "capm": _build_capm,
    "fama-french": _build_fama_french,
    "dcf": _build_dcf,
    "ddm": _build_ddm,
    "markowitz": _build_markowitz,
    "sharpe": _build_sharpe,
    "sortino": _build_sortino,
    "momentum": _build_momentum,
    "mean-reversion": _build_mean_reversion,
    "macd": _build_macd,
    "rsi": _build_rsi,
    "bollinger": _build_bollinger,
    "montecarlo": _build_montecarlo,
    "garch": _build_garch,
    "black-scholes": _build_black_scholes,
    "var": _build_var,
    "kelly": _build_kelly,
    "piotroski": _build_piotroski,
    "altman-z": _build_altman_z,
    "trend-ols": _build_trend_ols,
}


def build_signals(ctx: "AnalysisContext") -> list[StrategySignal]:
    """Run every registered strategy builder for one asset, in catalog order.

    Each builder is wrapped so that any unexpected failure degrades to a neutral
    ``HOLD`` signal rather than propagating — the returned list therefore always
    contains exactly ``len(STRATEGY_META)`` signals (one per registered id).

    Args:
        ctx: The analysis context for the asset.

    Returns:
        A list of :class:`~app.schemas.StrategySignal`, one per strategy, in the
        same order as :data:`STRATEGY_META`.
    """
    out: list[StrategySignal] = []
    for meta in STRATEGY_META:
        builder = SIGNAL_BUILDERS.get(meta.id)
        if builder is None:
            out.append(_neutral_signal(meta.id, "no builder registered"))
            continue
        try:
            signal = builder(ctx)
        except Exception as exc:  # pragma: no cover - defensive catch-all
            signal = _neutral_signal(meta.id, f"builder error ({type(exc).__name__})")
        out.append(signal)
    return out
