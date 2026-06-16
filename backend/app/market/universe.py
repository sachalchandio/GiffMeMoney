"""Deterministic seed universe for the GiffMeMoney simulator.

This module is the single source of truth for the *static* characteristics of
every tradable asset: its identity (symbol, name, class, sector, currency), the
parameters that drive the price simulator (drift, volatility, factor loadings),
and a full deterministic :class:`Fundamentals` record used by the valuation and
fundamental quant models.

Nothing here is random — every value is hand-tuned to be plausible and
internally consistent so that downstream models (DCF, DDM, Piotroski, Altman-Z,
CAPM, Fama-French, …) receive realistic inputs. The simulator seeds its RNG from
a stable hash of each symbol, so the *dynamic* history is reproducible too.

The universe contains ~24 assets:
    * 14 equities spanning technology, financials, healthcare, energy,
      consumer, and industrial sectors;
    * 6 crypto assets (BTC, ETH, SOL, ADA, XRP, DOGE) — higher volatility,
      no dividends, varied beta;
    * 4 ETFs (SPY, QQQ, VTI, GLD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

__all__ = [
    "Fundamentals",
    "AssetSeed",
    "UNIVERSE",
    "get_seed",
    "symbols",
    "by_class",
]


@dataclass(frozen=True)
class Fundamentals:
    """Deterministic, plausible fundamental data for a single asset.

    All monetary per-share figures are in the asset's reporting currency.
    Balance-sheet aggregates (``total_assets``, ``sales``, …) are absolute and
    feed the Altman-Z ratios; per-share figures feed DCF/DDM. For crypto assets
    these are set to neutral / non-distress values (``dividend == 0``) so the
    fundamental models degrade gracefully rather than crash.

    Attributes:
        eps: Trailing earnings per share.
        fcf_per_share: Free cash flow per share (DCF input).
        dividend: Annual dividend per share (0 for non-payers / crypto).
        dividend_growth: Long-run annual dividend growth rate (decimal).
        book_value_per_share: Common equity per share.
        revenue_growth: Year-over-year revenue growth (decimal).
        net_margin: Net income / sales (decimal).
        debt_to_equity: Total debt / total equity.
        current_ratio: Current assets / current liabilities.
        roa: Return on assets = net income / total assets (decimal).
        shares_out: Diluted shares outstanding.
        working_capital: Current assets − current liabilities (Altman WC).
        retained_earnings: Cumulative retained earnings (Altman RE).
        ebit: Earnings before interest and taxes (Altman EBIT).
        total_assets: Total assets (Altman TA denominator).
        total_liabilities: Total liabilities (Altman TL).
        sales: Total revenue / sales (Altman Sales).
    """

    eps: float
    fcf_per_share: float
    dividend: float
    dividend_growth: float
    book_value_per_share: float
    revenue_growth: float
    net_margin: float
    debt_to_equity: float
    current_ratio: float
    roa: float
    shares_out: float
    working_capital: float
    retained_earnings: float
    ebit: float
    total_assets: float
    total_liabilities: float
    sales: float


@dataclass(frozen=True)
class AssetSeed:
    """Static seed describing one asset for the deterministic simulator.

    The simulator builds a daily return series from a single-index / 3-factor
    model::

        r_t = rf_d + market_beta * (mkt_t - rf_d)
                   + smb_loading * SMB_t + hml_loading * HML_t
                   + idiosyncratic noise (scaled by ``idio_vol``)

    and integrates it into a price path starting at ``base_price``.

    Attributes:
        symbol: Ticker, unique within the universe.
        name: Human-readable asset name.
        asset_class: One of ``'equity'``, ``'crypto'``, ``'etf'``.
        sector: Sector label (``'Crypto'`` / ``'Index'`` for non-equities).
        currency: ISO currency code (``'USD'`` throughout).
        base_price: Starting price of the simulated history.
        annual_drift: Expected annual log drift μ (decimal).
        annual_vol: Total annual volatility σ (decimal).
        market_beta: Sensitivity to the market factor.
        smb_loading: Size (small-minus-big) factor loading.
        hml_loading: Value (high-minus-low) factor loading.
        idio_vol: Annual idiosyncratic volatility (decimal).
        market_cap: Market capitalisation (None permitted but always set here).
        volume24h: 24-hour traded notional volume.
        fundamentals: Full :class:`Fundamentals` record.
    """

    symbol: str
    name: str
    asset_class: str
    sector: str
    currency: str
    base_price: float
    annual_drift: float
    annual_vol: float
    market_beta: float
    smb_loading: float
    hml_loading: float
    idio_vol: float
    market_cap: float
    volume24h: float
    fundamentals: Fundamentals


# ---------------------------------------------------------------------------
# Universe definition
# ---------------------------------------------------------------------------

UNIVERSE: List[AssetSeed] = [
    # ----------------------------- TECHNOLOGY -----------------------------
    AssetSeed(
        symbol="AAPL",
        name="Apple Inc.",
        asset_class="equity",
        sector="Technology",
        currency="USD",
        base_price=187.50,
        annual_drift=0.14,
        annual_vol=0.26,
        market_beta=1.20,
        smb_loading=-0.25,
        hml_loading=-0.30,
        idio_vol=0.16,
        market_cap=2_900_000_000_000.0,
        volume24h=58_000_000_000.0,
        fundamentals=Fundamentals(
            eps=6.42,
            fcf_per_share=6.90,
            dividend=0.96,
            dividend_growth=0.05,
            book_value_per_share=4.40,
            revenue_growth=0.08,
            net_margin=0.253,
            debt_to_equity=1.45,
            current_ratio=0.98,
            roa=0.276,
            shares_out=15_500_000_000.0,
            working_capital=-1_700_000_000.0,
            retained_earnings=8_200_000_000.0,
            ebit=119_000_000_000.0,
            total_assets=352_000_000_000.0,
            total_liabilities=290_000_000_000.0,
            sales=383_000_000_000.0,
        ),
    ),
    AssetSeed(
        symbol="MSFT",
        name="Microsoft Corporation",
        asset_class="equity",
        sector="Technology",
        currency="USD",
        base_price=415.00,
        annual_drift=0.16,
        annual_vol=0.25,
        market_beta=1.10,
        smb_loading=-0.30,
        hml_loading=-0.20,
        idio_vol=0.15,
        market_cap=3_080_000_000_000.0,
        volume24h=22_000_000_000.0,
        fundamentals=Fundamentals(
            eps=11.05,
            fcf_per_share=8.10,
            dividend=3.00,
            dividend_growth=0.10,
            book_value_per_share=32.50,
            revenue_growth=0.16,
            net_margin=0.362,
            debt_to_equity=0.47,
            current_ratio=1.27,
            roa=0.176,
            shares_out=7_430_000_000.0,
            working_capital=34_000_000_000.0,
            retained_earnings=118_000_000_000.0,
            ebit=109_000_000_000.0,
            total_assets=512_000_000_000.0,
            total_liabilities=234_000_000_000.0,
            sales=237_000_000_000.0,
        ),
    ),
    AssetSeed(
        symbol="NVDA",
        name="NVIDIA Corporation",
        asset_class="equity",
        sector="Technology",
        currency="USD",
        base_price=126.00,
        annual_drift=0.30,
        annual_vol=0.48,
        market_beta=1.65,
        smb_loading=-0.10,
        hml_loading=-0.55,
        idio_vol=0.30,
        market_cap=3_100_000_000_000.0,
        volume24h=42_000_000_000.0,
        fundamentals=Fundamentals(
            eps=2.95,
            fcf_per_share=2.40,
            dividend=0.04,
            dividend_growth=0.04,
            book_value_per_share=2.20,
            revenue_growth=1.26,
            net_margin=0.488,
            debt_to_equity=0.22,
            current_ratio=4.17,
            roa=0.452,
            shares_out=24_600_000_000.0,
            working_capital=46_000_000_000.0,
            retained_earnings=30_000_000_000.0,
            ebit=33_000_000_000.0,
            total_assets=66_000_000_000.0,
            total_liabilities=22_000_000_000.0,
            sales=61_000_000_000.0,
        ),
    ),
    AssetSeed(
        symbol="GOOGL",
        name="Alphabet Inc.",
        asset_class="equity",
        sector="Technology",
        currency="USD",
        base_price=176.00,
        annual_drift=0.13,
        annual_vol=0.27,
        market_beta=1.05,
        smb_loading=-0.20,
        hml_loading=-0.15,
        idio_vol=0.17,
        market_cap=2_180_000_000_000.0,
        volume24h=24_000_000_000.0,
        fundamentals=Fundamentals(
            eps=5.80,
            fcf_per_share=5.10,
            dividend=0.20,
            dividend_growth=0.06,
            book_value_per_share=22.10,
            revenue_growth=0.13,
            net_margin=0.241,
            debt_to_equity=0.10,
            current_ratio=2.10,
            roa=0.165,
            shares_out=12_300_000_000.0,
            working_capital=74_000_000_000.0,
            retained_earnings=211_000_000_000.0,
            ebit=84_000_000_000.0,
            total_assets=402_000_000_000.0,
            total_liabilities=119_000_000_000.0,
            sales=307_000_000_000.0,
        ),
    ),
    # ----------------------------- FINANCIALS -----------------------------
    AssetSeed(
        symbol="JPM",
        name="JPMorgan Chase & Co.",
        asset_class="equity",
        sector="Financials",
        currency="USD",
        base_price=198.00,
        annual_drift=0.10,
        annual_vol=0.22,
        market_beta=1.10,
        smb_loading=-0.05,
        hml_loading=0.45,
        idio_vol=0.13,
        market_cap=570_000_000_000.0,
        volume24h=8_500_000_000.0,
        fundamentals=Fundamentals(
            eps=16.20,
            fcf_per_share=18.50,
            dividend=4.60,
            dividend_growth=0.07,
            book_value_per_share=104.00,
            revenue_growth=0.09,
            net_margin=0.330,
            debt_to_equity=1.30,
            current_ratio=1.10,
            roa=0.013,
            shares_out=2_880_000_000.0,
            working_capital=120_000_000_000.0,
            retained_earnings=340_000_000_000.0,
            ebit=62_000_000_000.0,
            total_assets=3_900_000_000_000.0,
            total_liabilities=3_570_000_000_000.0,
            sales=158_000_000_000.0,
        ),
    ),
    AssetSeed(
        symbol="BAC",
        name="Bank of America Corporation",
        asset_class="equity",
        sector="Financials",
        currency="USD",
        base_price=39.50,
        annual_drift=0.08,
        annual_vol=0.25,
        market_beta=1.25,
        smb_loading=0.05,
        hml_loading=0.55,
        idio_vol=0.15,
        market_cap=305_000_000_000.0,
        volume24h=3_200_000_000.0,
        fundamentals=Fundamentals(
            eps=3.10,
            fcf_per_share=3.40,
            dividend=0.96,
            dividend_growth=0.08,
            book_value_per_share=33.50,
            revenue_growth=0.04,
            net_margin=0.270,
            debt_to_equity=1.10,
            current_ratio=1.05,
            roa=0.009,
            shares_out=7_900_000_000.0,
            working_capital=60_000_000_000.0,
            retained_earnings=190_000_000_000.0,
            ebit=30_000_000_000.0,
            total_assets=3_180_000_000_000.0,
            total_liabilities=2_900_000_000_000.0,
            sales=98_000_000_000.0,
        ),
    ),
    AssetSeed(
        symbol="V",
        name="Visa Inc.",
        asset_class="equity",
        sector="Financials",
        currency="USD",
        base_price=275.00,
        annual_drift=0.12,
        annual_vol=0.21,
        market_beta=0.95,
        smb_loading=-0.20,
        hml_loading=-0.05,
        idio_vol=0.13,
        market_cap=540_000_000_000.0,
        volume24h=6_000_000_000.0,
        fundamentals=Fundamentals(
            eps=9.10,
            fcf_per_share=9.80,
            dividend=2.08,
            dividend_growth=0.16,
            book_value_per_share=17.80,
            revenue_growth=0.10,
            net_margin=0.530,
            debt_to_equity=0.55,
            current_ratio=1.45,
            roa=0.190,
            shares_out=2_030_000_000.0,
            working_capital=14_000_000_000.0,
            retained_earnings=18_000_000_000.0,
            ebit=23_000_000_000.0,
            total_assets=93_000_000_000.0,
            total_liabilities=57_000_000_000.0,
            sales=33_000_000_000.0,
        ),
    ),
    # ----------------------------- HEALTHCARE -----------------------------
    AssetSeed(
        symbol="JNJ",
        name="Johnson & Johnson",
        asset_class="equity",
        sector="Healthcare",
        currency="USD",
        base_price=152.00,
        annual_drift=0.07,
        annual_vol=0.17,
        market_beta=0.60,
        smb_loading=-0.25,
        hml_loading=0.25,
        idio_vol=0.11,
        market_cap=366_000_000_000.0,
        volume24h=2_400_000_000.0,
        fundamentals=Fundamentals(
            eps=5.80,
            fcf_per_share=6.40,
            dividend=4.76,
            dividend_growth=0.06,
            book_value_per_share=28.50,
            revenue_growth=0.06,
            net_margin=0.180,
            debt_to_equity=0.45,
            current_ratio=1.16,
            roa=0.090,
            shares_out=2_410_000_000.0,
            working_capital=8_000_000_000.0,
            retained_earnings=120_000_000_000.0,
            ebit=24_000_000_000.0,
            total_assets=187_000_000_000.0,
            total_liabilities=110_000_000_000.0,
            sales=85_000_000_000.0,
        ),
    ),
    AssetSeed(
        symbol="PFE",
        name="Pfizer Inc.",
        asset_class="equity",
        sector="Healthcare",
        currency="USD",
        base_price=28.50,
        annual_drift=0.04,
        annual_vol=0.20,
        market_beta=0.70,
        smb_loading=-0.10,
        hml_loading=0.40,
        idio_vol=0.14,
        market_cap=161_000_000_000.0,
        volume24h=1_700_000_000.0,
        fundamentals=Fundamentals(
            eps=1.10,
            fcf_per_share=1.30,
            dividend=1.68,
            dividend_growth=0.03,
            book_value_per_share=15.80,
            revenue_growth=-0.41,
            net_margin=0.095,
            debt_to_equity=0.72,
            current_ratio=1.21,
            roa=0.035,
            shares_out=5_660_000_000.0,
            working_capital=6_000_000_000.0,
            retained_earnings=95_000_000_000.0,
            ebit=9_000_000_000.0,
            total_assets=227_000_000_000.0,
            total_liabilities=137_000_000_000.0,
            sales=58_000_000_000.0,
        ),
    ),
    # ------------------------------- ENERGY -------------------------------
    AssetSeed(
        symbol="XOM",
        name="Exxon Mobil Corporation",
        asset_class="equity",
        sector="Energy",
        currency="USD",
        base_price=114.00,
        annual_drift=0.09,
        annual_vol=0.24,
        market_beta=0.90,
        smb_loading=-0.05,
        hml_loading=0.60,
        idio_vol=0.16,
        market_cap=455_000_000_000.0,
        volume24h=3_600_000_000.0,
        fundamentals=Fundamentals(
            eps=8.90,
            fcf_per_share=9.50,
            dividend=3.80,
            dividend_growth=0.04,
            book_value_per_share=68.00,
            revenue_growth=-0.10,
            net_margin=0.105,
            debt_to_equity=0.20,
            current_ratio=1.48,
            roa=0.110,
            shares_out=3_960_000_000.0,
            working_capital=12_000_000_000.0,
            retained_earnings=440_000_000_000.0,
            ebit=48_000_000_000.0,
            total_assets=377_000_000_000.0,
            total_liabilities=163_000_000_000.0,
            sales=345_000_000_000.0,
        ),
    ),
    AssetSeed(
        symbol="CVX",
        name="Chevron Corporation",
        asset_class="equity",
        sector="Energy",
        currency="USD",
        base_price=158.00,
        annual_drift=0.08,
        annual_vol=0.23,
        market_beta=0.95,
        smb_loading=-0.05,
        hml_loading=0.55,
        idio_vol=0.15,
        market_cap=292_000_000_000.0,
        volume24h=2_100_000_000.0,
        fundamentals=Fundamentals(
            eps=11.40,
            fcf_per_share=10.80,
            dividend=6.52,
            dividend_growth=0.06,
            book_value_per_share=87.00,
            revenue_growth=-0.18,
            net_margin=0.100,
            debt_to_equity=0.15,
            current_ratio=1.30,
            roa=0.095,
            shares_out=1_870_000_000.0,
            working_capital=6_500_000_000.0,
            retained_earnings=180_000_000_000.0,
            ebit=29_000_000_000.0,
            total_assets=261_000_000_000.0,
            total_liabilities=99_000_000_000.0,
            sales=200_000_000_000.0,
        ),
    ),
    # ------------------------------ CONSUMER ------------------------------
    AssetSeed(
        symbol="AMZN",
        name="Amazon.com, Inc.",
        asset_class="equity",
        sector="Consumer",
        currency="USD",
        base_price=185.00,
        annual_drift=0.15,
        annual_vol=0.32,
        market_beta=1.30,
        smb_loading=-0.15,
        hml_loading=-0.40,
        idio_vol=0.20,
        market_cap=1_920_000_000_000.0,
        volume24h=14_000_000_000.0,
        fundamentals=Fundamentals(
            eps=3.05,
            fcf_per_share=3.20,
            dividend=0.0,
            dividend_growth=0.0,
            book_value_per_share=19.50,
            revenue_growth=0.12,
            net_margin=0.053,
            debt_to_equity=0.55,
            current_ratio=1.05,
            roa=0.052,
            shares_out=10_400_000_000.0,
            working_capital=9_000_000_000.0,
            retained_earnings=110_000_000_000.0,
            ebit=37_000_000_000.0,
            total_assets=528_000_000_000.0,
            total_liabilities=325_000_000_000.0,
            sales=575_000_000_000.0,
        ),
    ),
    AssetSeed(
        symbol="KO",
        name="The Coca-Cola Company",
        asset_class="equity",
        sector="Consumer",
        currency="USD",
        base_price=62.50,
        annual_drift=0.06,
        annual_vol=0.16,
        market_beta=0.58,
        smb_loading=-0.30,
        hml_loading=0.20,
        idio_vol=0.10,
        market_cap=269_000_000_000.0,
        volume24h=1_300_000_000.0,
        fundamentals=Fundamentals(
            eps=2.45,
            fcf_per_share=2.30,
            dividend=1.94,
            dividend_growth=0.05,
            book_value_per_share=6.40,
            revenue_growth=0.07,
            net_margin=0.234,
            debt_to_equity=1.55,
            current_ratio=1.13,
            roa=0.103,
            shares_out=4_310_000_000.0,
            working_capital=2_500_000_000.0,
            retained_earnings=72_000_000_000.0,
            ebit=12_500_000_000.0,
            total_assets=99_000_000_000.0,
            total_liabilities=71_000_000_000.0,
            sales=46_000_000_000.0,
        ),
    ),
    # ----------------------------- INDUSTRIAL -----------------------------
    AssetSeed(
        symbol="CAT",
        name="Caterpillar Inc.",
        asset_class="equity",
        sector="Industrial",
        currency="USD",
        base_price=345.00,
        annual_drift=0.11,
        annual_vol=0.27,
        market_beta=1.15,
        smb_loading=0.10,
        hml_loading=0.35,
        idio_vol=0.17,
        market_cap=168_000_000_000.0,
        volume24h=1_600_000_000.0,
        fundamentals=Fundamentals(
            eps=20.10,
            fcf_per_share=18.40,
            dividend=5.20,
            dividend_growth=0.08,
            book_value_per_share=37.00,
            revenue_growth=0.03,
            net_margin=0.165,
            debt_to_equity=2.10,
            current_ratio=1.38,
            roa=0.122,
            shares_out=490_000_000.0,
            working_capital=14_000_000_000.0,
            retained_earnings=42_000_000_000.0,
            ebit=13_500_000_000.0,
            total_assets=87_000_000_000.0,
            total_liabilities=69_000_000_000.0,
            sales=67_000_000_000.0,
        ),
    ),
    # ------------------------------- CRYPTO -------------------------------
    AssetSeed(
        symbol="BTC",
        name="Bitcoin",
        asset_class="crypto",
        sector="Crypto",
        currency="USD",
        base_price=64_500.00,
        annual_drift=0.40,
        annual_vol=0.65,
        market_beta=1.40,
        smb_loading=0.20,
        hml_loading=-0.30,
        idio_vol=0.55,
        market_cap=1_270_000_000_000.0,
        volume24h=28_000_000_000.0,
        fundamentals=Fundamentals(
            eps=0.0,
            fcf_per_share=0.0,
            dividend=0.0,
            dividend_growth=0.0,
            book_value_per_share=0.0,
            revenue_growth=0.0,
            net_margin=0.0,
            debt_to_equity=0.0,
            current_ratio=1.0,
            roa=0.0,
            shares_out=19_700_000.0,
            working_capital=0.0,
            retained_earnings=0.0,
            ebit=0.0,
            total_assets=1.0,
            total_liabilities=0.0,
            sales=0.0,
        ),
    ),
    AssetSeed(
        symbol="ETH",
        name="Ethereum",
        asset_class="crypto",
        sector="Crypto",
        currency="USD",
        base_price=3_450.00,
        annual_drift=0.38,
        annual_vol=0.72,
        market_beta=1.55,
        smb_loading=0.25,
        hml_loading=-0.35,
        idio_vol=0.62,
        market_cap=415_000_000_000.0,
        volume24h=15_000_000_000.0,
        fundamentals=Fundamentals(
            eps=0.0,
            fcf_per_share=0.0,
            dividend=0.0,
            dividend_growth=0.0,
            book_value_per_share=0.0,
            revenue_growth=0.0,
            net_margin=0.0,
            debt_to_equity=0.0,
            current_ratio=1.0,
            roa=0.0,
            shares_out=120_000_000.0,
            working_capital=0.0,
            retained_earnings=0.0,
            ebit=0.0,
            total_assets=1.0,
            total_liabilities=0.0,
            sales=0.0,
        ),
    ),
    AssetSeed(
        symbol="SOL",
        name="Solana",
        asset_class="crypto",
        sector="Crypto",
        currency="USD",
        base_price=145.00,
        annual_drift=0.45,
        annual_vol=0.95,
        market_beta=1.80,
        smb_loading=0.40,
        hml_loading=-0.45,
        idio_vol=0.85,
        market_cap=66_000_000_000.0,
        volume24h=3_200_000_000.0,
        fundamentals=Fundamentals(
            eps=0.0,
            fcf_per_share=0.0,
            dividend=0.0,
            dividend_growth=0.0,
            book_value_per_share=0.0,
            revenue_growth=0.0,
            net_margin=0.0,
            debt_to_equity=0.0,
            current_ratio=1.0,
            roa=0.0,
            shares_out=460_000_000.0,
            working_capital=0.0,
            retained_earnings=0.0,
            ebit=0.0,
            total_assets=1.0,
            total_liabilities=0.0,
            sales=0.0,
        ),
    ),
    AssetSeed(
        symbol="ADA",
        name="Cardano",
        asset_class="crypto",
        sector="Crypto",
        currency="USD",
        base_price=0.45,
        annual_drift=0.30,
        annual_vol=0.90,
        market_beta=1.70,
        smb_loading=0.45,
        hml_loading=-0.40,
        idio_vol=0.82,
        market_cap=16_000_000_000.0,
        volume24h=420_000_000.0,
        fundamentals=Fundamentals(
            eps=0.0,
            fcf_per_share=0.0,
            dividend=0.0,
            dividend_growth=0.0,
            book_value_per_share=0.0,
            revenue_growth=0.0,
            net_margin=0.0,
            debt_to_equity=0.0,
            current_ratio=1.0,
            roa=0.0,
            shares_out=35_000_000_000.0,
            working_capital=0.0,
            retained_earnings=0.0,
            ebit=0.0,
            total_assets=1.0,
            total_liabilities=0.0,
            sales=0.0,
        ),
    ),
    AssetSeed(
        symbol="XRP",
        name="XRP",
        asset_class="crypto",
        sector="Crypto",
        currency="USD",
        base_price=0.52,
        annual_drift=0.25,
        annual_vol=0.85,
        market_beta=1.50,
        smb_loading=0.35,
        hml_loading=-0.30,
        idio_vol=0.78,
        market_cap=29_000_000_000.0,
        volume24h=1_100_000_000.0,
        fundamentals=Fundamentals(
            eps=0.0,
            fcf_per_share=0.0,
            dividend=0.0,
            dividend_growth=0.0,
            book_value_per_share=0.0,
            revenue_growth=0.0,
            net_margin=0.0,
            debt_to_equity=0.0,
            current_ratio=1.0,
            roa=0.0,
            shares_out=55_000_000_000.0,
            working_capital=0.0,
            retained_earnings=0.0,
            ebit=0.0,
            total_assets=1.0,
            total_liabilities=0.0,
            sales=0.0,
        ),
    ),
    AssetSeed(
        symbol="DOGE",
        name="Dogecoin",
        asset_class="crypto",
        sector="Crypto",
        currency="USD",
        base_price=0.135,
        annual_drift=0.18,
        annual_vol=1.10,
        market_beta=1.90,
        smb_loading=0.50,
        hml_loading=-0.50,
        idio_vol=1.00,
        market_cap=19_000_000_000.0,
        volume24h=900_000_000.0,
        fundamentals=Fundamentals(
            eps=0.0,
            fcf_per_share=0.0,
            dividend=0.0,
            dividend_growth=0.0,
            book_value_per_share=0.0,
            revenue_growth=0.0,
            net_margin=0.0,
            debt_to_equity=0.0,
            current_ratio=1.0,
            roa=0.0,
            shares_out=144_000_000_000.0,
            working_capital=0.0,
            retained_earnings=0.0,
            ebit=0.0,
            total_assets=1.0,
            total_liabilities=0.0,
            sales=0.0,
        ),
    ),
    # -------------------------------- ETFs --------------------------------
    AssetSeed(
        symbol="SPY",
        name="SPDR S&P 500 ETF Trust",
        asset_class="etf",
        sector="Index",
        currency="USD",
        base_price=545.00,
        annual_drift=0.10,
        annual_vol=0.16,
        market_beta=1.00,
        smb_loading=0.0,
        hml_loading=0.0,
        idio_vol=0.02,
        market_cap=540_000_000_000.0,
        volume24h=30_000_000_000.0,
        fundamentals=Fundamentals(
            eps=22.50,
            fcf_per_share=0.0,
            dividend=6.80,
            dividend_growth=0.06,
            book_value_per_share=545.0,
            revenue_growth=0.08,
            net_margin=0.0,
            debt_to_equity=0.0,
            current_ratio=1.0,
            roa=0.0,
            shares_out=990_000_000.0,
            working_capital=0.0,
            retained_earnings=0.0,
            ebit=0.0,
            total_assets=1.0,
            total_liabilities=0.0,
            sales=0.0,
        ),
    ),
    AssetSeed(
        symbol="QQQ",
        name="Invesco QQQ Trust",
        asset_class="etf",
        sector="Index",
        currency="USD",
        base_price=470.00,
        annual_drift=0.14,
        annual_vol=0.21,
        market_beta=1.15,
        smb_loading=-0.15,
        hml_loading=-0.30,
        idio_vol=0.03,
        market_cap=290_000_000_000.0,
        volume24h=18_000_000_000.0,
        fundamentals=Fundamentals(
            eps=14.20,
            fcf_per_share=0.0,
            dividend=2.60,
            dividend_growth=0.10,
            book_value_per_share=470.0,
            revenue_growth=0.14,
            net_margin=0.0,
            debt_to_equity=0.0,
            current_ratio=1.0,
            roa=0.0,
            shares_out=620_000_000.0,
            working_capital=0.0,
            retained_earnings=0.0,
            ebit=0.0,
            total_assets=1.0,
            total_liabilities=0.0,
            sales=0.0,
        ),
    ),
    AssetSeed(
        symbol="VTI",
        name="Vanguard Total Stock Market ETF",
        asset_class="etf",
        sector="Index",
        currency="USD",
        base_price=272.00,
        annual_drift=0.10,
        annual_vol=0.16,
        market_beta=1.02,
        smb_loading=0.10,
        hml_loading=0.05,
        idio_vol=0.02,
        market_cap=420_000_000_000.0,
        volume24h=1_500_000_000.0,
        fundamentals=Fundamentals(
            eps=11.30,
            fcf_per_share=0.0,
            dividend=3.70,
            dividend_growth=0.06,
            book_value_per_share=272.0,
            revenue_growth=0.08,
            net_margin=0.0,
            debt_to_equity=0.0,
            current_ratio=1.0,
            roa=0.0,
            shares_out=1_540_000_000.0,
            working_capital=0.0,
            retained_earnings=0.0,
            ebit=0.0,
            total_assets=1.0,
            total_liabilities=0.0,
            sales=0.0,
        ),
    ),
    AssetSeed(
        symbol="GLD",
        name="SPDR Gold Shares",
        asset_class="etf",
        sector="Commodity",
        currency="USD",
        base_price=216.00,
        annual_drift=0.06,
        annual_vol=0.14,
        market_beta=0.15,
        smb_loading=0.0,
        hml_loading=0.05,
        idio_vol=0.12,
        market_cap=64_000_000_000.0,
        volume24h=1_400_000_000.0,
        fundamentals=Fundamentals(
            eps=0.0,
            fcf_per_share=0.0,
            dividend=0.0,
            dividend_growth=0.0,
            book_value_per_share=216.0,
            revenue_growth=0.0,
            net_margin=0.0,
            debt_to_equity=0.0,
            current_ratio=1.0,
            roa=0.0,
            shares_out=296_000_000.0,
            working_capital=0.0,
            retained_earnings=0.0,
            ebit=0.0,
            total_assets=1.0,
            total_liabilities=0.0,
            sales=0.0,
        ),
    ),
]


# Index by symbol for O(1) lookups. Built once at import time.
_BY_SYMBOL: Dict[str, AssetSeed] = {seed.symbol: seed for seed in UNIVERSE}


def get_seed(symbol: str) -> AssetSeed:
    """Return the :class:`AssetSeed` for ``symbol``.

    Lookup is case-insensitive on the symbol and O(1) via a prebuilt index.

    Args:
        symbol: Asset ticker, e.g. ``"AAPL"`` or ``"btc"``.

    Returns:
        The matching :class:`AssetSeed`.

    Raises:
        KeyError: If no asset with that symbol exists in the universe.
    """
    key = symbol.strip().upper()
    seed = _BY_SYMBOL.get(key)
    if seed is None:
        raise KeyError(f"Unknown symbol: {symbol!r}")
    return seed


def symbols() -> List[str]:
    """Return all symbols in the universe, in declaration order.

    Returns:
        A new list of ticker strings.
    """
    return [seed.symbol for seed in UNIVERSE]


def by_class(cls: str) -> List[AssetSeed]:
    """Return all seeds whose ``asset_class`` matches ``cls``.

    Matching is case-insensitive (``"EQUITY"`` == ``"equity"``).

    Args:
        cls: Asset class, one of ``'equity'``, ``'crypto'``, ``'etf'``.

    Returns:
        A list of matching :class:`AssetSeed` objects (possibly empty).
    """
    target = cls.strip().lower()
    return [seed for seed in UNIVERSE if seed.asset_class == target]
