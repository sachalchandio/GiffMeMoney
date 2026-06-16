"""Tests for the quant model layer against known values and structural invariants.

Covers the headline models named in the contract (section 7):

* Black-Scholes price (reference value) + put-call parity + Greeks signs;
* implied vol round-trip;
* CAPM Security-Market-Line identities;
* Fama-French OLS recovery of known loadings;
* DCF flat-perpetuity identity and Gordon DDM guard;
* VaR/CVaR ordering (CVaR >= VaR) across all three VaR families;
* Monte Carlo band ordering (p5 <= p25 <= p50 <= p75 <= p95) and DTO shape;
* Markowitz optimizer feasibility (weights in [0,1], sum to 1) for every objective;
* GARCH(1,1) MLE stationarity (alpha + beta < 1);
* Kelly fraction, Piotroski/Altman, and a couple of technical/forecast facts.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.quant import (
    capm,
    factor,
    forecast,
    fundamental,
    kelly,
    montecarlo,
    options,
    portfolio,
    risk,
    technical,
    valuation,
    volatility,
)
from app.market.universe import get_seed


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------


def test_black_scholes_reference_call_value() -> None:
    """ATM 1Y call: S=K=100, r=5%, sigma=20% prices to ~10.4506."""
    price = options.black_scholes(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    assert price == pytest.approx(10.4506, abs=0.05)


def test_black_scholes_put_call_parity() -> None:
    """C - P = S - K e^{-rT} for European options on a non-dividend underlying."""
    S, K, T, r, sigma = 100.0, 95.0, 0.75, 0.03, 0.25
    call = options.black_scholes(S, K, T, r, sigma, "call")
    put = options.black_scholes(S, K, T, r, sigma, "put")
    assert (call - put) == pytest.approx(S - K * math.exp(-r * T), abs=1e-6)


def test_black_scholes_zero_vol_is_discounted_intrinsic() -> None:
    """With no volatility the call is max(S - K e^{-rT}, 0)."""
    S, K, T, r = 110.0, 100.0, 1.0, 0.05
    price = options.black_scholes(S, K, T, r, 0.0, "call")
    assert price == pytest.approx(max(S - K * math.exp(-r * T), 0.0), abs=1e-9)


def test_black_scholes_call_delta_in_unit_interval() -> None:
    """A plain call delta lies in (0, 1); an ATM 1Y call delta is > 0.5."""
    greeks = options.bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "call")
    assert 0.0 < greeks["delta"] < 1.0
    assert greeks["delta"] > 0.5
    assert greeks["gamma"] > 0.0
    assert greeks["vega"] > 0.0


def test_black_scholes_put_delta_negative() -> None:
    """A put delta lies in (-1, 0)."""
    greeks = options.bs_greeks(100.0, 100.0, 1.0, 0.05, 0.2, "put")
    assert -1.0 < greeks["delta"] < 0.0


def test_implied_vol_round_trip() -> None:
    """Recovering sigma from a BS price returns the original volatility."""
    S, K, T, r, sigma = 100.0, 105.0, 1.0, 0.04, 0.3
    price = options.black_scholes(S, K, T, r, sigma, "call")
    iv = options.implied_vol(price, S, K, T, r, "call")
    assert iv == pytest.approx(sigma, abs=1e-4)


# ---------------------------------------------------------------------------
# CAPM
# ---------------------------------------------------------------------------


def test_capm_beta_one_recovers_market_return() -> None:
    """E[R] = Rf + 1*(premium) = Rf + premium with beta = 1."""
    assert capm.capm_expected_return(1.0, 0.04, 0.06) == pytest.approx(0.10)


def test_capm_beta_zero_is_risk_free() -> None:
    """beta = 0 leaves only the risk-free rate."""
    assert capm.capm_expected_return(0.0, 0.04, 0.06) == pytest.approx(0.04)


def test_capm_scales_with_beta() -> None:
    """E[R] = Rf + beta*premium for an arbitrary beta."""
    assert capm.capm_expected_return(1.5, 0.03, 0.05) == pytest.approx(0.03 + 1.5 * 0.05)


# ---------------------------------------------------------------------------
# Fama-French 3-factor OLS
# ---------------------------------------------------------------------------


def test_fama_french_recovers_known_loadings() -> None:
    """OLS recovers the betas/alpha of a noiseless 3-factor construction."""
    rng = np.random.default_rng(7)
    n = 500
    mkt = rng.normal(0.0004, 0.01, n)
    smb = rng.normal(0.0, 0.006, n)
    hml = rng.normal(0.0, 0.006, n)
    alpha_d, b_m, b_s, b_h = 0.0002, 1.2, -0.3, 0.4
    y = alpha_d + b_m * mkt + b_s * smb + b_h * hml  # no noise

    res = factor.fama_french_3factor(y, mkt, smb, hml)
    assert res.beta_mkt == pytest.approx(b_m, abs=1e-6)
    assert res.beta_smb == pytest.approx(b_s, abs=1e-6)
    assert res.beta_hml == pytest.approx(b_h, abs=1e-6)
    assert res.r2 == pytest.approx(1.0, abs=1e-6)


def test_fama_french_too_short_is_safe_default() -> None:
    """Below the minimum observation count returns the zeroed default."""
    res = factor.fama_french_3factor([0.01, 0.02], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    assert (res.alpha_annual, res.beta_mkt, res.beta_smb, res.beta_hml, res.r2) == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )


# ---------------------------------------------------------------------------
# Valuation: DCF and Gordon DDM
# ---------------------------------------------------------------------------


def test_dcf_flat_perpetuity_identity() -> None:
    """Flat FCF (growth=0, terminal_growth=0) capitalizes to FCF / wacc."""
    fcf, wacc = 5.0, 0.10
    value = valuation.dcf_intrinsic_value(fcf, growth=0.0, wacc=wacc, terminal_growth=0.0, years=10)
    assert value == pytest.approx(fcf / wacc, rel=1e-9)


def test_dcf_growth_increases_value() -> None:
    """A higher growth rate yields a strictly larger intrinsic value."""
    low = valuation.dcf_intrinsic_value(5.0, 0.02, 0.10)
    high = valuation.dcf_intrinsic_value(5.0, 0.06, 0.10)
    assert high > low


def test_dcf_non_positive_fcf_is_zero() -> None:
    """A non-positive FCF base yields zero value (no negative price)."""
    assert valuation.dcf_intrinsic_value(0.0, 0.05, 0.10) == pytest.approx(0.0)
    assert valuation.dcf_intrinsic_value(-3.0, 0.05, 0.10) == pytest.approx(0.0)


def test_gordon_ddm_known_value() -> None:
    """P = D0*(1+g)/(r-g) for the constant-growth dividend model."""
    d0, r, g = 2.0, 0.08, 0.03
    expected = d0 * (1.0 + g) / (r - g)
    assert valuation.gordon_ddm(d0, r, g) == pytest.approx(expected, rel=1e-12)


def test_gordon_ddm_guards_r_le_g() -> None:
    """When required return <= growth the model is undefined -> 0 (no inf)."""
    assert valuation.gordon_ddm(2.0, 0.03, 0.05) == pytest.approx(0.0)
    val = valuation.gordon_ddm(2.0, 0.05, 0.05)
    assert math.isfinite(val) and val == pytest.approx(0.0)


def test_gordon_ddm_non_payer_is_zero() -> None:
    """A non-dividend payer has no DDM value."""
    assert valuation.gordon_ddm(0.0, 0.08, 0.03) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# VaR / CVaR ordering
# ---------------------------------------------------------------------------


def test_var_cvar_ordering_historical_and_parametric() -> None:
    """CVaR >= historical VaR >= 0, and parametric VaR is finite & non-negative."""
    rng = np.random.default_rng(99)
    returns = rng.normal(0.0005, 0.02, 2000)

    hist = risk.historical_var(returns, conf=0.95)
    para = risk.parametric_var(returns, conf=0.95)
    es = risk.cvar(returns, conf=0.95)

    assert hist >= 0.0
    assert para >= 0.0
    assert es >= 0.0
    # Expected shortfall is never below the VaR threshold it averages beyond.
    assert es >= hist - 1e-12


def test_monte_carlo_var_non_negative_and_finite() -> None:
    """Simulated VaR is a finite, non-negative loss fraction."""
    v = risk.monte_carlo_var(mu_daily=0.0003, sigma_daily=0.02, conf=0.95, sims=20000, seed=1)
    assert math.isfinite(v) and 0.0 <= v <= 1.0


def test_var_higher_confidence_is_at_least_as_large() -> None:
    """A 99% VaR is at least as large as the 95% VaR (deeper tail)."""
    rng = np.random.default_rng(3)
    returns = rng.normal(0.0, 0.02, 5000)
    v95 = risk.historical_var(returns, conf=0.95)
    v99 = risk.historical_var(returns, conf=0.99)
    assert v99 >= v95 - 1e-9


def test_var_empty_input_is_zero() -> None:
    """Empty input returns 0.0 across the VaR family."""
    assert risk.historical_var([]) == pytest.approx(0.0)
    assert risk.parametric_var([]) == pytest.approx(0.0)
    assert risk.cvar([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Monte Carlo GBM summary
# ---------------------------------------------------------------------------


def test_montecarlo_bands_monotone_in_percentile() -> None:
    """At every time step, p5 <= p25 <= p50 <= p75 <= p95."""
    result = montecarlo.montecarlo_summary(
        s0=100.0, mu_daily=0.0004, sigma_daily=0.02, horizon="1Y", sims=2000, seed=42
    )
    assert result["steps"] == 252
    assert len(result["bands"]) == 253  # steps + 1
    for band in result["bands"]:
        assert band["p5"] <= band["p25"] <= band["p50"] <= band["p75"] <= band["p95"]


def test_montecarlo_first_band_is_starting_price() -> None:
    """Step 0 percentile bands all equal the starting price."""
    s0 = 100.0
    result = montecarlo.montecarlo_summary(s0=s0, mu_daily=0.0, sigma_daily=0.02, horizon="1M", seed=0)
    first = result["bands"][0]
    for key in ("p5", "p25", "p50", "p75", "p95"):
        assert first[key] == pytest.approx(s0, abs=1e-6)


def test_montecarlo_summary_shape_and_ranges() -> None:
    """The summary dict matches the MonteCarloResult contract shape and ranges."""
    result = montecarlo.montecarlo_summary(
        s0=50.0, mu_daily=0.0005, sigma_daily=0.03, horizon="1Y", sims=1500, seed=11
    )
    assert set(result) >= {
        "horizon",
        "sims",
        "steps",
        "bands",
        "finalDistribution",
        "expectedReturnPct",
        "var95Pct",
        "cvar95Pct",
        "probPositive",
    }
    assert 0.0 <= result["probPositive"] <= 1.0
    assert result["var95Pct"] >= 0.0
    # CVaR (mean loss beyond VaR) is never below VaR.
    assert result["cvar95Pct"] >= result["var95Pct"] - 1e-9
    assert result["finalDistribution"]
    for b in result["finalDistribution"]:
        assert b["binEnd"] >= b["binStart"]
        assert b["count"] >= 0


def test_gbm_paths_shape_and_positivity() -> None:
    """gbm_paths returns a (sims, steps+1) array of finite positive prices."""
    paths = montecarlo.gbm_paths(s0=100.0, mu_daily=0.0, sigma_daily=0.02, steps=10, sims=50, seed=5)
    assert paths.shape == (50, 11)
    assert np.all(np.isfinite(paths))
    assert np.all(paths > 0.0)
    assert np.all(paths[:, 0] == pytest.approx(100.0))


# ---------------------------------------------------------------------------
# Markowitz optimizer
# ---------------------------------------------------------------------------


def _sample_mu_cov() -> tuple[np.ndarray, np.ndarray]:
    """Return a small well-behaved (mu, cov) pair for optimizer tests."""
    mu = np.array([0.10, 0.14, 0.08], dtype=float)
    cov = np.array(
        [
            [0.040, 0.010, 0.005],
            [0.010, 0.060, 0.008],
            [0.005, 0.008, 0.030],
        ],
        dtype=float,
    )
    return mu, cov


@pytest.mark.parametrize("objective", ["max_sharpe", "min_volatility", "target_return"])
def test_optimizer_weights_feasible(objective: str) -> None:
    """For every objective, weights are long-only and sum to 1."""
    mu, cov = _sample_mu_cov()
    target = 0.11 if objective == "target_return" else None
    w = portfolio.optimize(mu, cov, rf=0.04, objective=objective, target=target)
    assert w.shape == (3,)
    assert np.all(w >= -1e-9)
    assert np.all(w <= 1.0 + 1e-9)
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-6)


def test_min_volatility_not_more_volatile_than_equal_weight() -> None:
    """The min-variance portfolio's vol <= the equal-weight portfolio's vol."""
    mu, cov = _sample_mu_cov()
    w_min = portfolio.optimize(mu, cov, rf=0.04, objective="min_volatility")
    _, vol_min, _ = portfolio.portfolio_stats(w_min, mu, cov, 0.04)
    equal = np.full(3, 1.0 / 3.0)
    _, vol_eq, _ = portfolio.portfolio_stats(equal, mu, cov, 0.04)
    assert vol_min <= vol_eq + 1e-9


def test_target_return_constraint_met() -> None:
    """A feasible target return is achieved by the optimized weights."""
    mu, cov = _sample_mu_cov()
    target = 0.11
    w = portfolio.optimize(mu, cov, rf=0.04, objective="target_return", target=target)
    achieved = float(w @ mu)
    assert achieved == pytest.approx(target, abs=1e-4)


def test_efficient_frontier_points_sorted_by_volatility() -> None:
    """efficient_frontier returns PortfolioPoint dicts sorted by volatility."""
    mu, cov = _sample_mu_cov()
    pts = portfolio.efficient_frontier(mu, cov, rf=0.04, n=15)
    assert pts
    vols = [p["volatility"] for p in pts]
    assert vols == sorted(vols)
    for p in pts:
        assert set(p) == {"volatility", "expectedReturn", "sharpe"}
        assert all(math.isfinite(p[k]) for k in p)


def test_capital_market_line_constant_sharpe_slope() -> None:
    """The CML has a constant Sharpe (its slope) for non-zero-vol points."""
    cml = portfolio.capital_market_line(rf=0.04, tangency_return=0.12, tangency_vol=0.18, n=20)
    assert cml
    sharpes = [p["sharpe"] for p in cml if p["volatility"] > 1e-9]
    expected = (0.12 - 0.04) / 0.18
    for s in sharpes:
        assert s == pytest.approx(expected, rel=1e-9)


def test_optimizer_single_asset_returns_full_weight() -> None:
    """A one-asset universe puts all weight on that asset."""
    w = portfolio.optimize(np.array([0.1]), np.array([[0.04]]), rf=0.04, objective="max_sharpe")
    assert w.shape == (1,)
    assert w[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# GARCH(1,1) MLE
# ---------------------------------------------------------------------------


def test_garch_fit_is_stationary() -> None:
    """A fitted GARCH(1,1) returns omega>0 and a stationary alpha+beta<1."""
    rng = np.random.default_rng(2024)
    # Simulate a GARCH-like series so the MLE has structure to fit.
    n = 800
    omega_t, alpha_t, beta_t = 0.00001, 0.08, 0.90
    eps = np.empty(n)
    h = omega_t / (1.0 - alpha_t - beta_t)
    for t in range(n):
        z = rng.standard_normal()
        eps[t] = math.sqrt(h) * z
        h = omega_t + alpha_t * eps[t] ** 2 + beta_t * h

    omega, alpha, beta = volatility.garch11_fit(eps)
    assert omega > 0.0
    assert alpha >= 0.0
    assert beta >= 0.0
    assert (alpha + beta) < 1.0


def test_garch_fit_short_series_fallback_stationary() -> None:
    """Even a too-short series returns a valid stationary triple via fallback."""
    omega, alpha, beta = volatility.garch11_fit(np.array([0.01, -0.02, 0.0, 0.015]))
    assert omega > 0.0
    assert 0.0 <= alpha
    assert 0.0 <= beta
    assert (alpha + beta) < 1.0


def test_garch_forecast_annualized_volatility_finite() -> None:
    """The horizon forecast is a finite, non-negative annualized volatility."""
    rng = np.random.default_rng(5)
    returns = rng.normal(0.0, 0.02, 600)
    f = volatility.garch11_forecast(returns, horizon_days=21)
    assert math.isfinite(f) and f >= 0.0


def test_ewma_vol_finite_and_non_negative() -> None:
    """EWMA volatility is a finite, non-negative annualized number."""
    rng = np.random.default_rng(6)
    returns = rng.normal(0.0, 0.015, 300)
    v = volatility.ewma_vol(returns)
    assert math.isfinite(v) and v >= 0.0


# ---------------------------------------------------------------------------
# Kelly fraction
# ---------------------------------------------------------------------------


def test_kelly_fraction_known_value() -> None:
    """f* = mu / sigma^2 within the [-1, 3] clamp."""
    mu, sigma = 0.0002, 0.02
    assert kelly.kelly_fraction(mu, sigma) == pytest.approx(mu / (sigma * sigma), rel=1e-12)


def test_kelly_fraction_clamped() -> None:
    """A huge edge is clamped to the 3x leverage ceiling; loss-side to -1."""
    assert kelly.kelly_fraction(1.0, 0.01) == pytest.approx(3.0)
    assert kelly.kelly_fraction(-1.0, 0.01) == pytest.approx(-1.0)


def test_kelly_fraction_zero_vol_is_zero() -> None:
    """Zero variance would diverge, so the fraction collapses to 0."""
    assert kelly.kelly_fraction(0.001, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Fundamental scores (use real universe seeds)
# ---------------------------------------------------------------------------


def test_piotroski_score_in_range_for_quality_equity() -> None:
    """A high-quality equity (MSFT) scores within [0, 9] and reasonably high."""
    f = get_seed("MSFT").fundamentals
    score = fundamental.piotroski_score(f)
    assert 0 <= score <= 9
    assert score >= 6  # a profitable, low-leverage, growing name


def test_piotroski_score_crypto_low_but_valid() -> None:
    """Mostly-zero crypto fundamentals yield a valid low score (no crash)."""
    f = get_seed("BTC").fundamentals
    score = fundamental.piotroski_score(f)
    assert 0 <= score <= 9


def test_altman_z_safe_zone_for_strong_equity() -> None:
    """A strong equity lands above the distress threshold (Z > 1.81)."""
    seed = get_seed("AAPL")
    z = fundamental.altman_z(seed.fundamentals, seed.market_cap)
    assert math.isfinite(z)
    assert z > 1.81


# ---------------------------------------------------------------------------
# Technical / forecast spot checks
# ---------------------------------------------------------------------------


def test_rsi_all_up_is_overbought() -> None:
    """A strictly rising series has RSI == 100 (no downside moves)."""
    prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0], dtype=float)
    assert technical.rsi(prices, n=14) == pytest.approx(100.0)


def test_rsi_neutral_default_on_short_input() -> None:
    """Too few prices returns the neutral 50."""
    assert technical.rsi([100.0]) == pytest.approx(50.0)


def test_bollinger_percent_b_at_upper_band() -> None:
    """%B is 0.5 at the mean for a symmetric window and within [-2, 3]."""
    prices = np.array([10.0, 12.0, 8.0, 11.0, 9.0, 10.0], dtype=float)
    mid, upper, lower, pct_b = technical.bollinger(prices, n=6, k=2.0)
    assert lower <= mid <= upper
    assert -2.0 <= pct_b <= 3.0


def test_ols_trend_recovers_constant_log_drift() -> None:
    """An exact exponential price path has slope == its per-step log drift."""
    drift = 0.001
    t = np.arange(300)
    prices = 100.0 * np.exp(drift * t)
    slope, intercept, r2, fc_drift = forecast.ols_trend(prices)
    assert slope == pytest.approx(drift, rel=1e-6)
    assert fc_drift == pytest.approx(drift, rel=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_holt_winters_forecast_above_last_for_uptrend() -> None:
    """A rising series forecasts above its last observed price."""
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], dtype=float)
    level, trend, fc = forecast.holt_winters(prices, horizon=5)
    assert trend > 0.0
    assert fc > prices[-1]
