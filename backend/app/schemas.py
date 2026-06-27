"""Pydantic v2 DTOs for the GiffMeMoney API (section 4 of the contract).

Every model serializes to **camelCase** on the wire and accepts both camelCase
and snake_case on input. This is achieved with a shared :class:`CamelModel`
base whose ``ConfigDict`` uses ``alias_generator=to_camel`` together with
``populate_by_name=True`` (so Python code can construct models with snake_case
field names) and ``from_attributes=True`` (so models can be built from ORM-like
objects / dataclasses).

To emit camelCase JSON, always dump with ``model_dump(by_alias=True)`` or
``model_dump_json(by_alias=True)``.

The TypeScript types in ``frontend/src/lib/types.ts`` mirror these 1:1.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# ---------------------------------------------------------------------------
# Literal type aliases (section 4)
# ---------------------------------------------------------------------------

AssetClass = Literal["equity", "crypto", "etf"]
Horizon = Literal["1D", "1W", "1M", "1Y", "5Y"]
Stance = Literal["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"]
StrategyCategory = Literal[
    "Valuation",
    "Factor",
    "Risk-Adjusted",
    "Technical",
    "Statistical",
    "Portfolio",
    "Fundamental",
    "Derivatives",
]

#: Canonical, ordered list of projection horizons.
HORIZONS: list[Horizon] = ["1D", "1W", "1M", "1Y", "5Y"]


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------


class CamelModel(BaseModel):
    """Base model that serializes to camelCase and accepts both casings.

    Configuration:
        * ``alias_generator=to_camel`` — field aliases become camelCase, so
          ``model_dump(by_alias=True)`` emits the wire format.
        * ``populate_by_name=True`` — models may be constructed using the
          original snake_case field names from Python.
        * ``from_attributes=True`` — models may be built from arbitrary objects
          (dataclasses, ORM rows) via attribute access.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


# ---------------------------------------------------------------------------
# Core market DTOs
# ---------------------------------------------------------------------------


class Asset(CamelModel):
    """A tradable instrument with its latest snapshot price."""

    symbol: str
    name: str
    asset_class: AssetClass
    sector: Optional[str] = None
    currency: str
    price: float
    change24h_pct: float = Field(alias="change24hPct")
    market_cap: Optional[float] = None
    volume24h: Optional[float] = Field(default=None, alias="volume24h")


class Candle(CamelModel):
    """A single OHLCV candle. ``t`` is a unix timestamp in **seconds**."""

    t: int
    o: float
    h: float
    l: float
    c: float
    v: float


class PricePoint(CamelModel):
    """A live price sample. ``t`` is a unix timestamp in **milliseconds**."""

    symbol: str
    price: float
    t: int
    change_pct: float


# ---------------------------------------------------------------------------
# Projection & signal DTOs
# ---------------------------------------------------------------------------


class ExpectedReturn(CamelModel):
    """Projected total return over a horizon with confidence bands.

    Fields:
        horizon: One of :data:`HORIZONS`.
        expected_return_pct: Mean total return over the horizon, in percent.
        low: ~5th percentile total return, in percent.
        high: ~95th percentile total return, in percent.
        prob_positive: Probability the return is positive, in ``[0, 1]``.
        annualized_vol: Annualized volatility, in percent.
        bull_pct: Optional bull-scenario total return, in percent (V2 fan).
        base_pct: Optional base-scenario total return, in percent (V2 fan).
        bear_pct: Optional bear-scenario total return, in percent (V2 fan).
        cvar_pct: Optional expected shortfall (CVaR) at this horizon, as a
            positive loss fraction in percent (V2 downside).

    The four V2 scenario/downside fields are additive and optional (``None``
    by default) so pre-V2 projections remain valid; populate them when the V2
    projection engine produces a scenario fan + CVaR.
    """

    horizon: Horizon
    expected_return_pct: float
    low: float
    high: float
    prob_positive: float
    annualized_vol: float
    bull_pct: Optional[float] = None
    base_pct: Optional[float] = None
    bear_pct: Optional[float] = None
    cvar_pct: Optional[float] = None


# ---------------------------------------------------------------------------
# V2 backtest & regime DTOs (additive — see docs/STRATEGIES-V2.md §7)
# ---------------------------------------------------------------------------


class RegimeInfo(CamelModel):
    """Market-regime classification for an asset (V2 projection engine).

    Fields:
        regime: Discrete regime label.
        trend: Signed trend strength (e.g. normalized slope); sign indicates
            direction, magnitude indicates conviction.
        vol_regime: Volatility regime bucket.
        score: Composite regime score (typically in ``[-1, 1]`` or similar);
            positive leans bullish, negative bearish.
    """

    regime: Literal["bull", "bear", "neutral"]
    trend: float
    vol_regime: Literal["low", "normal", "high"]
    score: float


class BacktestMetricsDTO(CamelModel):
    """The 14 realized performance metrics for a backtested strategy/asset.

    All values are decimals unless a percent is documented by the backtest
    engine. ``cvar95`` is a downside (expected-shortfall) figure. Every field
    must be finite on the wire (never NaN/inf).

    Fields:
        cagr: Compound annual growth rate (decimal).
        total_return: Total return over the backtest window (decimal).
        ann_vol: Annualized volatility (decimal).
        sharpe: Annualized Sharpe ratio.
        sortino: Annualized Sortino ratio.
        calmar: Calmar ratio (CAGR / abs(max drawdown)).
        max_drawdown: Maximum peak-to-trough drawdown (decimal, <= 0 or its
            magnitude per the engine convention).
        ulcer_index: Ulcer index (drawdown-based risk measure).
        win_rate: Fraction of winning periods, in ``[0, 1]``.
        profit_factor: Gross profit / gross loss.
        exposure: Fraction of time invested (market exposure), in ``[0, 1]``.
        turnover: Aggregate position turnover.
        cvar95: 95% conditional VaR (expected shortfall), as a loss figure.
        beta: Beta to the buy & hold benchmark.
        information_ratio: Information ratio vs the benchmark.
    """

    cagr: float
    total_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    ulcer_index: float
    win_rate: float
    profit_factor: float
    exposure: float
    turnover: float
    cvar95: float
    beta: float
    information_ratio: float


class BacktestEquityPoint(CamelModel):
    """One downsampled point on the strategy-vs-benchmark equity curve.

    Fields:
        t: Unix timestamp (seconds) of the bar.
        strategy: Strategy equity value (indexed, e.g. starting at 1.0).
        benchmark: Buy & hold benchmark equity value (indexed).
    """

    t: int
    strategy: float
    benchmark: float


class BacktestResultDTO(CamelModel):
    """Full backtest result for one strategy applied to one asset.

    Fields:
        symbol: The backtested asset.
        strategy_id: Stable id of the strategy (matches the registry).
        supported: ``False`` for snapshot/fundamental strategies that are not
            time-backtestable per-bar (a buy & hold-only result is returned).
        trades: Number of round-trip/position-change trades.
        metrics: Realized metrics for the strategy.
        benchmark: Realized metrics for buy & hold of the same asset.
        equity_curve: Downsampled strategy-vs-benchmark equity series.
    """

    symbol: str
    strategy_id: str
    supported: bool
    trades: int
    metrics: BacktestMetricsDTO
    benchmark: BacktestMetricsDTO
    equity_curve: list[BacktestEquityPoint] = Field(default_factory=list)


class StrategySignal(CamelModel):
    """The output of one quant model for one asset.

    Fields:
        strategy_id: Stable id of the strategy (matches the registry).
        strategy_name: Human-readable strategy name.
        category: Strategy category.
        score: Bullishness in ``[-100, 100]`` (positive = bullish).
        stance: Discrete stance derived from ``score`` thresholds.
        confidence: Model confidence in ``[0, 1]``.
        rationale: Plain-English explanation of the signal.
        formula: Compact, human-readable formula used by the model.
        metrics: Model-specific raw numbers keyed by name.
        horizons: Per-horizon projections (may be empty for non-projecting
            models).
        backtest: Optional lightweight realized-performance summary for
            backtestable strategies (``None`` for snapshot strategies or when
            not computed; V2 additive).
    """

    strategy_id: str
    strategy_name: str
    category: StrategyCategory
    score: float
    stance: Stance
    confidence: float
    rationale: str
    formula: str
    metrics: dict[str, float] = Field(default_factory=dict)
    horizons: list[ExpectedReturn] = Field(default_factory=list)
    backtest: Optional[BacktestMetricsDTO] = None


class RiskMetrics(CamelModel):
    """Annualized risk metrics for an asset."""

    beta: float
    annual_vol: float
    sharpe: float
    sortino: float
    var95: float
    cvar95: float
    max_drawdown: float
    calmar: float


class AssetAnalysis(CamelModel):
    """Full composite analysis for a single asset.

    Fields:
        asset: The asset being analyzed.
        composite_score: Blended score in ``[-100, 100]``.
        recommendation: Composite stance.
        confidence: Aggregate confidence in ``[0, 1]``.
        expected_returns: Blended projections, always one per :data:`HORIZONS`.
        risk_metrics: Annualized risk metrics.
        signals: One :class:`StrategySignal` per registered strategy.
        rationale_summary: Narrative "where/why to invest".
        top_reasons: 3-5 bullet reasons.
        updated_at: Unix timestamp in milliseconds.
        regime: Optional market-regime classification (V2 additive; ``None``
            when not computed).
        strategy_count: Number of strategy signals contributing to this
            analysis (V2 additive).
        disclaimer: Standard educational-use disclaimer surfaced by the API
            and shown in the UI (V2 additive; defaults to the standard text).
    """

    asset: Asset
    composite_score: float
    recommendation: Stance
    confidence: float
    expected_returns: list[ExpectedReturn] = Field(default_factory=list)
    risk_metrics: RiskMetrics
    signals: list[StrategySignal] = Field(default_factory=list)
    rationale_summary: str
    top_reasons: list[str] = Field(default_factory=list)
    updated_at: int
    regime: Optional[RegimeInfo] = None
    strategy_count: int = 0
    disclaimer: str = (
        "Educational simulation on synthetic market data — not financial "
        "advice; projections are model estimates, not guarantees."
    )


class Recommendation(CamelModel):
    """A ranked recommendation row (also reused by market movers)."""

    rank: int
    asset: Asset
    composite_score: float
    recommendation: Stance
    confidence: float
    expected_return1y_pct: float = Field(alias="expectedReturn1YPct")
    headline: str
    reasons: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Strategy catalog DTOs
# ---------------------------------------------------------------------------


class StrategyMeta(CamelModel):
    """Static metadata describing a strategy in the catalog."""

    id: str
    name: str
    category: StrategyCategory
    summary: str
    formula: str
    inputs: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class RankingEntry(CamelModel):
    """One asset's standing under a single strategy."""

    asset: Asset
    score: float
    stance: Stance


class StrategyRanking(CamelModel):
    """Cross-asset rankings produced by one strategy."""

    strategy_id: str
    entries: list[RankingEntry] = Field(default_factory=list)


class StrategyLeaderboardEntry(CamelModel):
    """One strategy's realized backtest standing for a single asset (V2).

    Fields:
        rank: 1-based rank within the leaderboard (best Sharpe first).
        strategy_id: Stable id of the strategy (matches the registry).
        strategy_name: Human-readable strategy name.
        category: Strategy category.
        supported: ``False`` for snapshot/fundamental strategies that are not
            time-backtestable per-bar (ranked at the bottom).
        sharpe: Annualized Sharpe ratio of the strategy equity curve.
        cagr: Compound annual growth rate of the strategy equity curve (decimal).
        total_return: Total return over the backtest window (decimal).
        max_drawdown: Maximum peak-to-trough drawdown (decimal).
        calmar: Calmar ratio (CAGR / abs(max drawdown)).
        win_rate: Fraction of winning periods, in ``[0, 1]``.
        trades: Number of position-change trades over the window.
    """

    rank: int
    strategy_id: str
    strategy_name: str
    category: StrategyCategory
    supported: bool
    sharpe: float
    cagr: float
    total_return: float
    max_drawdown: float
    calmar: float
    win_rate: float
    trades: int


class StrategyLeaderboard(CamelModel):
    """Per-asset leaderboard of strategies ranked by realized backtest performance.

    Fields:
        symbol: The asset the strategies were backtested on.
        benchmark: Buy & hold metrics for the asset (the bar to beat).
        entries: Strategies ranked best-first (by Sharpe, then CAGR).
    """

    symbol: str
    benchmark: BacktestMetricsDTO
    entries: list[StrategyLeaderboardEntry] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Portfolio DTOs
# ---------------------------------------------------------------------------


class PortfolioRequest(CamelModel):
    """Request body for the Markowitz optimizer endpoint.

    Fields:
        symbols: Universe of symbols to allocate across.
        risk_free_rate: Annual risk-free rate (decimal, e.g. ``0.04``).
        objective: Optimization objective.
        target_return: Annual target return (decimal); required iff
            ``objective == 'target_return'``.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "symbols": ["AAPL", "MSFT", "BTC"],
                    "riskFreeRate": 0.04,
                    "objective": "max_sharpe",
                    "targetReturn": None,
                }
            ]
        }
    )

    symbols: list[str]
    risk_free_rate: float
    objective: Literal["max_sharpe", "min_volatility", "target_return"]
    target_return: Optional[float] = None


class PortfolioPoint(CamelModel):
    """A single point on the efficient frontier / capital market line."""

    volatility: float
    expected_return: float
    sharpe: float


class PortfolioWeight(CamelModel):
    """An allocation weight for one symbol (decimal fraction)."""

    symbol: str
    weight: float


class PortfolioResult(CamelModel):
    """Result of a mean-variance optimization.

    All ``expectedReturn``/``volatility`` values are annual decimals.
    """

    weights: list[PortfolioWeight] = Field(default_factory=list)
    expected_return: float
    volatility: float
    sharpe: float
    efficient_frontier: list[PortfolioPoint] = Field(default_factory=list)
    capital_market_line: list[PortfolioPoint] = Field(default_factory=list)
    risk_free_rate: float


# ---------------------------------------------------------------------------
# Monte Carlo DTOs
# ---------------------------------------------------------------------------


class MonteCarloBand(CamelModel):
    """Price percentile band at one time step. ``t`` is the step index."""

    t: int
    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


class MonteCarloBin(CamelModel):
    """A histogram bin of the simulated final-price distribution."""

    bin_start: float
    bin_end: float
    count: int


class MonteCarloResult(CamelModel):
    """Result of a GBM Monte Carlo simulation for one asset.

    Fields:
        symbol: The simulated asset.
        horizon: Projection horizon.
        sims: Number of simulated paths.
        steps: Number of time steps (trading days in the horizon).
        bands: Price percentile bands over time.
        final_distribution: Histogram of terminal prices.
        expected_return_pct: Mean total return over the horizon, in percent.
        var95_pct: 95% Value-at-Risk, positive loss fraction in percent.
        cvar95_pct: 95% Conditional VaR, positive loss fraction in percent.
        prob_positive: Probability of a positive return, in ``[0, 1]``.
    """

    symbol: str
    horizon: Horizon
    sims: int
    steps: int
    bands: list[MonteCarloBand] = Field(default_factory=list)
    final_distribution: list[MonteCarloBin] = Field(default_factory=list)
    expected_return_pct: float
    var95_pct: float = Field(alias="var95Pct")
    cvar95_pct: float = Field(alias="cvar95Pct")
    prob_positive: float


# ---------------------------------------------------------------------------
# Market summary DTOs
# ---------------------------------------------------------------------------


class Breadth(CamelModel):
    """Advance/decline breadth counts."""

    advancers: int
    decliners: int
    unchanged: int


class SectorPerf(CamelModel):
    """Aggregated performance for one sector."""

    sector: str
    change_pct: float
    count: int


class IndexLevel(CamelModel):
    """A synthetic index level and its daily change."""

    name: str
    level: float
    change_pct: float


class MarketSummary(CamelModel):
    """Top-level market overview for the dashboard.

    Fields:
        as_of: Unix timestamp in milliseconds.
        breadth: Advance/decline breadth.
        top_gainers: Best movers (reuses :class:`Recommendation`).
        top_losers: Worst movers (reuses :class:`Recommendation`).
        sectors: Per-sector performance.
        indices: Synthetic index levels.
    """

    as_of: int
    breadth: Breadth
    top_gainers: list[Recommendation] = Field(default_factory=list)
    top_losers: list[Recommendation] = Field(default_factory=list)
    sectors: list[SectorPerf] = Field(default_factory=list)
    indices: list[IndexLevel] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Invest / wallet DTOs (simulated brokerage — see docs/INVEST.md)
# ---------------------------------------------------------------------------
#
# Money-handling stance: this is a SIMULATED / sandbox wallet. No real money
# moves. ``CardIn`` is input-only (it carries a raw PAN/CVC for validation only)
# and is NEVER persisted — services tokenize it into a masked :class:`SavedCard`
# (brand + last4 + token) and discard the sensitive fields immediately.

#: Discrete kinds of ledger entry recorded for an account.
TxnType = Literal["deposit", "withdrawal", "buy", "sell"]

#: Risk profile that drives the advisor's pick count and optimizer objective.
RiskTolerance = Literal["conservative", "balanced", "aggressive"]


class CardIn(CamelModel):
    """Input-only debit/credit card payload for a simulated deposit.

    This DTO carries the *raw* card details solely so the payment provider can
    validate them (Luhn check, brand detection, future-expiry check). It is
    never stored or logged: services immediately tokenize it into a masked
    :class:`SavedCard` and drop ``number``/``cvc``.

    Fields:
        number: The card primary account number (PAN); digits, may contain
            spaces which callers should strip.
        exp_month: Expiry month in ``[1, 12]``.
        exp_year: Four-digit expiry year (e.g. ``2027``).
        cvc: Card verification code (3-4 digits).
        holder: Cardholder name as printed on the card.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "number": "4242424242424242",
                    "expMonth": 12,
                    "expYear": 2030,
                    "cvc": "123",
                    "holder": "Demo Investor",
                }
            ]
        }
    )

    number: str
    exp_month: int
    exp_year: int
    cvc: str
    holder: str


class SavedCard(CamelModel):
    """A tokenized, masked card safe to persist and return on the wire.

    Contains no sensitive data: only the brand, last four digits, expiry, and
    holder name, keyed by an opaque token id. Raw PAN/CVC are never present.

    Fields:
        id: Opaque token id (uuid) standing in for the stored card.
        brand: Detected network (``'visa'`` / ``'mastercard'`` / ``'amex'`` /
            ``'discover'`` / ``'unknown'``).
        last4: The final four digits of the PAN.
        exp_month: Expiry month in ``[1, 12]``.
        exp_year: Four-digit expiry year.
        holder: Cardholder name.
    """

    id: str
    brand: str
    last4: str
    exp_month: int
    exp_year: int
    holder: str


class Transaction(CamelModel):
    """A single immutable ledger entry on an account.

    Fields:
        id: Unique transaction id (uuid).
        type: One of :data:`TxnType`.
        amount: Positive dollar magnitude of the entry.
        symbol: Asset symbol for ``buy``/``sell`` entries; ``None`` for
            cash movements (``deposit``/``withdrawal``).
        status: ``'completed'`` on success, ``'failed'`` otherwise.
        created_at: Unix timestamp in milliseconds.
        ref: Human-facing reference string (uuid-derived).
        note: Short human-readable description.
    """

    id: str
    type: TxnType
    amount: float
    symbol: Optional[str] = None
    status: Literal["completed", "failed"]
    created_at: int
    ref: str
    note: str


class Wallet(CamelModel):
    """A snapshot of an account's cash, invested value, and saved cards.

    Invariant: ``total_value == cash_balance + invested_value`` (balances
    always reconcile).

    Fields:
        account_id: The account identifier (default ``'demo'``).
        cash_balance: Uninvested cash available to spend or withdraw.
        invested_value: Mark-to-market value of all open positions.
        total_value: ``cash_balance + invested_value``.
        currency: ISO currency code (``'USD'``).
        saved_cards: Tokenized cards on file.
    """

    account_id: str
    cash_balance: float
    invested_value: float
    total_value: float
    currency: str = "USD"
    saved_cards: list[SavedCard] = Field(default_factory=list)


class DepositRequest(CamelModel):
    """Request body for ``POST /api/wallet/deposit``.

    Fields:
        amount: Dollar amount to deposit (must be ``> 0``).
        card: The card to charge (validated, never stored raw).
        save_card: Whether to tokenize and remember the card.
        saved_card_id: Optional id of an already-saved card to reuse; when set
            the inline ``card`` is still validated but reuse is preferred.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "amount": 20,
                    "card": {
                        "number": "4242424242424242",
                        "expMonth": 12,
                        "expYear": 2030,
                        "cvc": "123",
                        "holder": "Demo Investor",
                    },
                    "saveCard": True,
                }
            ]
        }
    )

    amount: float
    card: CardIn
    save_card: bool = False
    saved_card_id: Optional[str] = None


class WithdrawRequest(CamelModel):
    """Request body for ``POST /api/wallet/withdraw``.

    Fields:
        amount: Dollar amount to withdraw (must be ``> 0`` and ``<= cash``).
        destination: Optional free-text payout destination label.
    """

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"amount": 10}]}
    )

    amount: float
    destination: Optional[str] = None


class AllocationItem(CamelModel):
    """One leg of an invest order: spend ``amount`` dollars on ``symbol``."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"symbol": "AAPL", "amount": 10}]}
    )

    symbol: str
    amount: float


class InvestRequest(CamelModel):
    """Request body for ``POST /api/portfolio/invest``."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "allocations": [
                        {"symbol": "AAPL", "amount": 10},
                        {"symbol": "BTC", "amount": 10},
                    ]
                }
            ]
        }
    )

    allocations: list[AllocationItem] = Field(default_factory=list)


class SellRequest(CamelModel):
    """Request body for ``POST /api/portfolio/sell``.

    Fields:
        symbol: Symbol of the position to reduce.
        amount: Dollar amount to sell; ``None`` when ``all`` is true.
        all: When true, liquidate the entire position (``amount`` ignored).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"symbol": "AAPL", "amount": None, "all": True}]
        }
    )

    symbol: str
    amount: Optional[float] = None
    all: bool = False


class Position(CamelModel):
    """A held position marked to the latest price.

    Fields:
        symbol: Asset ticker.
        asset: The embedded :class:`Asset` snapshot.
        units: Fractional units currently held.
        cost_basis: Total dollars invested in the still-held units.
        avg_price: ``cost_basis / units`` (0 when units are 0).
        current_price: Latest mark price.
        market_value: ``units * current_price``.
        unrealized_pnl: ``market_value - cost_basis``.
        unrealized_pnl_pct: ``unrealized_pnl / cost_basis * 100`` (0 when no
            cost basis).
        allocation_pct: This position's share of total invested market value,
            in percent.
        realized_pnl: Cumulative realized P&L from prior sells of this symbol.
        opened_at: Unix timestamp (ms) the position was first opened.
    """

    symbol: str
    asset: Asset
    units: float
    cost_basis: float
    avg_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    allocation_pct: float
    realized_pnl: float
    opened_at: int


class PortfolioState(CamelModel):
    """Full mark-to-market portfolio view returned by the portfolio routes.

    Fields:
        wallet: The current :class:`Wallet` snapshot.
        positions: All open positions (marked to latest price).
        total_cost: Sum of every position's ``cost_basis``.
        total_value: Sum of every position's ``market_value``.
        total_pnl: ``total_value - total_cost``.
        total_pnl_pct: ``total_pnl / total_cost * 100`` (0 when no cost).
    """

    wallet: Wallet
    positions: list[Position] = Field(default_factory=list)
    total_cost: float
    total_value: float
    total_pnl: float
    total_pnl_pct: float


#: The action a protective risk rule takes on a position.
RiskActionType = Literal["stop_loss", "trailing_stop", "take_profit", "drawdown"]


class RiskPolicy(CamelModel):
    """Per-account post-buy loss controls (all optional, default OFF/``None``).

    These are protective *exit* rules evaluated against already-held positions by
    ``POST /api/portfolio/risk/apply``. They never block or change a buy — they
    only trigger protective sells / de-risking when explicitly applied. This is a
    SIMULATION on synthetic data: enabling these rules does not guarantee a
    profit or prevent loss; they are mechanical, after-the-fact exits.

    Every threshold is a **positive percent** (e.g. ``stopLossPct = 10`` means
    "exit once down 10% from entry"). ``None`` means the rule is disabled, and
    every field defaults to ``None`` so the feature is OFF unless opted into.

    Fields:
        stop_loss_pct: Sell a position once it is down more than this percent from
            its blended entry (average-cost) price.
        trailing_stop_pct: Sell a position once it falls more than this percent
            below its observed high-water-mark price.
        take_profit_pct: Sell a position once it is up more than this percent
            above its blended entry price.
        max_drawdown_pct: When total portfolio value is down more than this
            percent from its peak, reduce exposure (sell the worst positions /
            raise cash) until the drawdown is back within the limit.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "stopLossPct": 10,
                    "trailingStopPct": 15,
                    "takeProfitPct": 40,
                    "maxDrawdownPct": 25,
                }
            ]
        }
    )

    stop_loss_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None


class RiskAction(CamelModel):
    """One protective action triggered by :class:`RiskPolicy` evaluation.

    Fields:
        symbol: The position acted on.
        action: Which rule fired (:data:`RiskActionType`).
        reason: Human-readable explanation (e.g. ``"down 12.3% from entry; "
            "stop-loss is 10%"``).
        amount: Dollar proceeds of the protective sell credited back to cash.
        units_sold: Units liquidated by the action.
        price: Mark price at which the protective sell executed.
        realized_pnl: Realized P&L on the protective sell (signed).
    """

    symbol: str
    action: RiskActionType
    reason: str
    amount: float
    units_sold: float
    price: float
    realized_pnl: float


class RiskApplyResult(CamelModel):
    """Result of running ``POST /api/portfolio/risk/apply``.

    Fields:
        actions: Protective actions taken (empty when nothing breached).
        policy: Echo of the active :class:`RiskPolicy`.
        state: The updated :class:`PortfolioState` after any protective sells.
        triggered: ``True`` when at least one action fired.
        disclaimer: Standard educational-simulation disclaimer.
    """

    actions: list[RiskAction] = Field(default_factory=list)
    policy: RiskPolicy
    state: PortfolioState
    triggered: bool = False
    disclaimer: str = (
        "Educational simulation on synthetic market data — not financial "
        "advice. Risk controls are mechanical, after-the-fact exits; they do "
        "not guarantee a profit or prevent loss."
    )


class PortfolioHistoryPoint(CamelModel):
    """One step of the total-portfolio value curve. ``t`` is unix ms."""

    t: int
    total_value: float
    invested: float
    cash: float


class PositionHistoryPoint(CamelModel):
    """One step of a single position's value/P&L curve. ``t`` is unix ms."""

    t: int
    value: float
    pnl: float
    pnl_pct: float


class PositionHistory(CamelModel):
    """A single position's backfilled value/P&L series."""

    symbol: str
    points: list[PositionHistoryPoint] = Field(default_factory=list)


class PortfolioHistory(CamelModel):
    """The backfilled recent-window portfolio chart seed.

    Fields:
        total: The total-portfolio value series.
        positions: Per-position value/P&L series.
    """

    total: list[PortfolioHistoryPoint] = Field(default_factory=list)
    positions: list[PositionHistory] = Field(default_factory=list)


class AdviceRequest(CamelModel):
    """Request body for ``POST /api/advisor/allocate``.

    Fields:
        amount: Dollar amount to allocate across recommended assets.
        risk_tolerance: Risk profile driving pick count + optimizer objective.
        asset_classes: Optional filter; ``None`` means consider all classes.
        target_amount: Optional goal amount the caller hopes to reach (additive);
            used only to flag a physically extreme / infeasible target.
        horizon_days: Optional number of days the caller wants to reach
            ``target_amount`` in (additive); pairs with ``target_amount`` for the
            feasibility check.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "amount": 20,
                    "riskTolerance": "balanced",
                    "assetClasses": ["equity", "crypto", "etf"],
                }
            ]
        }
    )

    amount: float
    risk_tolerance: RiskTolerance
    asset_classes: Optional[list[AssetClass]] = None
    target_amount: Optional[float] = None
    horizon_days: Optional[float] = None


class AdviceItem(CamelModel):
    """One recommended allocation leg from the advisor.

    Fields:
        asset: The recommended :class:`Asset`.
        weight: Portfolio weight in ``[0, 1]``.
        amount: ``weight * request.amount`` dollars.
        composite_score: The engine's composite score for this asset.
        expected_return1y_pct: The asset's blended 1Y expected return (percent).
        rationale: Short plain-English reason for the pick.
    """

    asset: Asset
    weight: float
    amount: float
    composite_score: float
    expected_return1y_pct: float = Field(alias="expectedReturn1YPct")
    rationale: str


class AllocationAdvice(CamelModel):
    """The advisor's full basket recommendation.

    The advisor splits the requested ``amount`` into a **risky sleeve** (the
    ``items`` below) and a **cash sleeve** parked uninvested. The risky fraction
    is driven by the risk profile *and* the market regime / conviction:
    conservative profiles cap risky exposure and de-risk further in a bear regime
    or on a weak top composite. This keeps the advice honest — it never implies
    "always 100% invested".

    Invariants:
        * ``sum(item.weight for item in items) + cash_weight ~= 1``
        * ``sum(item.amount for item in items) + cash_amount ~= amount``

    Fields:
        items: Per-asset allocation legs (weights sum to the risky fraction,
            i.e. ``1 - cash_weight``).
        expected_return: Blended annual expected return of the **risky sleeve**
            (decimal); cash earns nothing in this simulation.
        expected_vol: Blended annual volatility of the risky sleeve (decimal).
        sharpe: Blended Sharpe ratio of the risky sleeve.
        horizons: Weight-blended 5-horizon expected returns for the risky sleeve
            (including the bull/base/bear scenario fan and the CVaR downside).
        risk_tolerance: Echo of the requested risk profile.
        amount: Echo of the requested dollar amount.
        cash_weight: Fraction of ``amount`` held as uninvested cash, in
            ``[0, 1]`` (``1 - sum of item weights``; additive).
        cash_amount: Dollars held as uninvested cash (``cash_weight * amount``;
            additive).
        synthetic_data: Always ``True`` — the advice is computed on synthetic
            (made-up) market data, not a real forecast (honesty flag; additive).
        target_warning: Optional plain-English warning when the request implies a
            physically extreme / infeasible target (``None`` for sane asks;
            additive).
    """

    items: list[AdviceItem] = Field(default_factory=list)
    expected_return: float
    expected_vol: float
    sharpe: float
    horizons: list[ExpectedReturn] = Field(default_factory=list)
    risk_tolerance: RiskTolerance
    amount: float
    cash_weight: float = 0.0
    cash_amount: float = 0.0
    synthetic_data: bool = True
    target_warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Auto-trader bot DTOs (simulated paper-trading — see app/bot/*)
# ---------------------------------------------------------------------------
#
# HONESTY / SAFETY: the auto-trader is a SIMULATION on synthetic data. It is
# paper-traded only — no real money moves and no live broker is ever contacted.
# Every result carries :data:`BOT_DISCLAIMER`. Rotation is momentum / bandit
# style (allocate MORE to recent winners, LESS to losers); the engine NEVER
# martingales (never increases a losing sleeve to "recover"). Nothing here
# implies guaranteed profit.

#: The mandatory, prominent disclaimer surfaced on every bot result + the UI.
BOT_DISCLAIMER: str = (
    "Simulated paper-trading on synthetic data — not financial advice; past "
    "simulated performance does not predict real results; no real funds are "
    "traded."
)

#: Stable id of one of the five preset bot modes.
BotModeId = Literal[
    "conservative",
    "balanced",
    "aggressive",
    "adaptive-bandit",
    "all-weather",
]

#: Discrete risk level of a bot mode (UI badge).
BotRiskLevel = Literal["low", "moderate", "high"]

#: Rotation style of a bot mode. ``momentum`` / ``bandit`` allocate MORE to
#: recent winners and LESS to losers; ``none`` is rebalance-only (no tilt).
#: There is deliberately no martingale option — the engine never chases losses.
BotRotation = Literal["none", "slow", "moderate", "fast", "bandit"]

#: A simulated bot trade side.
BotSide = Literal["buy", "sell"]

#: A sleeve's contribution verdict (best / worst by realized contribution).
BotVerdict = Literal["best", "worst", "neutral"]


class BotMode(CamelModel):
    """A preset auto-trader strategy mode (objective + rotation behaviour).

    Fields:
        id: Stable mode id (:data:`BotModeId`).
        name: Human-readable mode name.
        summary: One-line description of how the mode behaves.
        risk_level: ``'low' | 'moderate' | 'high'`` UI risk badge.
        objective: Portfolio objective driving the per-rebalance weights
            (``'min_volatility'`` / ``'max_sharpe'`` / ``'momentum'`` /
            ``'risk_parity'`` / ``'bandit'``).
        rotation: Rotation style (:data:`BotRotation`) — how aggressively the
            sleeve weights tilt toward recent winners (never toward losers).
        max_names: Maximum number of sleeves (assets) held at once.
    """

    id: BotModeId
    name: str
    summary: str
    risk_level: BotRiskLevel
    objective: str
    rotation: BotRotation
    max_names: int


class BotConfig(CamelModel):
    """Configuration for one simulated auto-trader backtest run.

    Fields:
        amount: Starting paper capital in dollars (must be ``> 0``).
        mode: The :data:`BotModeId` to run.
        asset_classes: Optional class filter for the candidate universe
            (``None`` = all classes).
        rebalance_days: Trading days between rebalances (default 21 ≈ monthly).
        stop_loss_pct: Per-sleeve stop: exit a sleeve once it is down more than
            this percent from its entry (default 25).
        max_drawdown_pct: Portfolio circuit-breaker: raise cash once total
            drawdown exceeds this percent (default 35).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "amount": 10000,
                    "mode": "balanced",
                    "assetClasses": ["equity", "etf"],
                    "rebalanceDays": 21,
                    "stopLossPct": 25,
                    "maxDrawdownPct": 35,
                }
            ]
        }
    )

    amount: float
    mode: BotModeId
    asset_classes: Optional[list[AssetClass]] = None
    rebalance_days: int = 21
    stop_loss_pct: float = 25.0
    max_drawdown_pct: float = 35.0


class BotRunRequest(CamelModel):
    """Request body for a bot backtest run.

    Fields:
        config: The :class:`BotConfig` to backtest.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"config": {"amount": 10000, "mode": "balanced"}}]
        }
    )

    config: BotConfig


class BotTrade(CamelModel):
    """One simulated paper trade recorded by the auto-trader.

    Fields:
        t: Unix timestamp (ms) of the (simulated) fill.
        symbol: The traded asset.
        side: ``'buy'`` or ``'sell'``.
        amount: Positive dollar magnitude transacted.
        strategy: The sleeve / strategy that motivated the trade.
        price: The (simulated) fill price.
    """

    t: int
    symbol: str
    side: BotSide
    amount: float
    strategy: str
    price: float


class SleeveAttribution(CamelModel):
    """Realized contribution of one sleeve (strategy or symbol) to the run.

    Fields:
        key: The sleeve key — a strategy name or a symbol.
        realized_pnl: Cumulative realized + marked dollar P&L of the sleeve.
        contribution_pct: Share of the run's total P&L attributable to the
            sleeve, in percent (signed).
        win_rate: Fraction of the sleeve's rebalance legs that were profitable,
            in ``[0, 1]``.
        trades: Number of trades the sleeve generated.
        verdict: ``'best'`` / ``'worst'`` / ``'neutral'`` ranking flag.
    """

    key: str
    realized_pnl: float
    contribution_pct: float
    win_rate: float
    trades: int
    verdict: BotVerdict


class BotEquityPoint(CamelModel):
    """One point on the bot-vs-benchmark equity curve.

    Fields:
        t: Unix timestamp (ms) of the bar.
        bot_value: The bot's total paper value (cash + marked positions).
        benchmark_value: Equal-weight buy & hold value of the same candidates.
        drawdown_pct: The bot's drawdown from its running peak, in percent
            (``<= 0``).
        regime: The detected market regime at that bar
            (``'bull' | 'bear' | 'neutral'``).
    """

    t: int
    bot_value: float
    benchmark_value: float
    drawdown_pct: float
    regime: str


class BotMetrics(CamelModel):
    """Realized performance metrics for one simulated bot run.

    Fields:
        total_return_pct: Total return over the run, in percent.
        cagr_pct: Compound annual growth rate, in percent.
        sharpe: Annualized Sharpe ratio of the bot's daily returns.
        sortino: Annualized Sortino ratio.
        max_drawdown_pct: Worst peak-to-trough drawdown, in percent (``<= 0``).
        win_rate_pct: Fraction of profitable rebalance periods, in percent.
        vs_benchmark_pct: Final-value outperformance vs the benchmark, in
            percentage points (signed).
        final_value: The bot's final total paper value.
    """

    total_return_pct: float
    cagr_pct: float
    sharpe: float
    sortino: float
    max_drawdown_pct: float
    win_rate_pct: float
    vs_benchmark_pct: float
    final_value: float


class BotRunResult(CamelModel):
    """The full result of one simulated auto-trader backtest.

    Fields:
        mode: The :class:`BotMode` that was run.
        config: Echo of the :class:`BotConfig` used.
        equity_curve: Bot-vs-benchmark equity series with regime + drawdown.
        trades: Every simulated paper trade recorded.
        attribution: Per-sleeve realized contribution (best → worst).
        metrics: Realized :class:`BotMetrics` for the run.
        best_strategy: Key of the best-contributing sleeve (``None`` if none).
        worst_strategy: Key of the worst-contributing sleeve (``None`` if none).
        regime_timeline: The regime label at each rebalance, in order.
        disclaimer: The mandatory simulation disclaimer (:data:`BOT_DISCLAIMER`).
        synthetic_data: Always ``True`` — the backtest runs on synthetic
            (made-up) market data, so results are not a real edge (honesty flag;
            additive).
        target_warning: Optional plain-English warning when the run implies a
            physically extreme / infeasible target (``None`` for sane asks;
            additive).
    """

    mode: BotMode
    config: BotConfig
    equity_curve: list[BotEquityPoint] = Field(default_factory=list)
    trades: list[BotTrade] = Field(default_factory=list)
    attribution: list[SleeveAttribution] = Field(default_factory=list)
    metrics: BotMetrics
    best_strategy: Optional[str] = None
    worst_strategy: Optional[str] = None
    regime_timeline: list[str] = Field(default_factory=list)
    disclaimer: str = BOT_DISCLAIMER
    synthetic_data: bool = True
    target_warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Broker execution DTOs (go-live, OPT-IN — see docs/GOLIVE.md §2)
# ---------------------------------------------------------------------------
#
# SAFETY / HONESTY: the broker layer ships in the **simulated** mode (paper
# fills against the market provider's price; no real money). The real Alpaca
# adapter defaults to Alpaca's PAPER (sandbox) endpoint, so even with keys set
# no real orders are placed. LIVE trading is hard-gated and OFF by default —
# every broker payload carries ``paper: true/false`` and the disclaimer below so
# a caller can never mistake a simulated/paper fill for a real one.

#: The mandatory disclaimer surfaced on every broker status/account/order
#: response (the broker is paper unless live is fully, deliberately enabled).
BROKER_DISCLAIMER: str = (
    "Simulated / paper trading — no real money moves unless live trading is "
    "deliberately enabled via the documented hard-gate. Not financial advice; "
    "trading carries the risk of real loss."
)

#: Broker execution backend keys (selected by ``settings.broker``).
BrokerName = Literal["simulated", "alpaca"]

#: A broker order side.
BrokerOrderSide = Literal["buy", "sell"]

#: A broker order type. Only ``market`` is supported by the paper broker.
BrokerOrderType = Literal["market"]

#: Lifecycle status of a broker order.
BrokerOrderStatus = Literal[
    "accepted",
    "filled",
    "partially_filled",
    "canceled",
    "rejected",
    "pending",
]


class BrokerStatus(CamelModel):
    """Connectivity / mode snapshot for the configured broker.

    Fields:
        broker: The active broker backend key (``'simulated'`` / ``'alpaca'``).
        mode: Human-facing execution mode (``'simulated'`` / ``'paper'`` /
            ``'live'``).
        paper: ``True`` whenever orders are simulated or routed to a paper
            sandbox; ``False`` only when real live trading is fully enabled.
        connected: Whether the broker is reachable / usable right now.
        live_enabled: Whether the full live-trading hard-gate is satisfied
            (``broker == 'alpaca'`` AND ``alpaca_live`` AND the exact
            ``broker_ack`` AND real keys). Ships ``False``.
        base_url: The broker REST base URL in effect (the Alpaca PAPER endpoint
            by default); ``None`` for the simulated broker.
        message: Optional human-readable note (e.g. why live is disabled).
        disclaimer: The mandatory broker disclaimer (:data:`BROKER_DISCLAIMER`).
    """

    broker: BrokerName
    mode: Literal["simulated", "paper", "live"]
    paper: bool
    connected: bool
    live_enabled: bool = False
    base_url: Optional[str] = None
    message: Optional[str] = None
    disclaimer: str = BROKER_DISCLAIMER


class BrokerAccount(CamelModel):
    """A broker account summary (cash + equity + buying power).

    Fields:
        account_id: Opaque broker account identifier.
        cash: Settled cash available.
        equity: Total account equity (cash + position market value).
        buying_power: Spendable buying power.
        currency: ISO currency code (``'USD'``).
        mode: Execution mode (``'simulated'`` / ``'paper'`` / ``'live'``).
        paper: ``True`` for simulated/paper accounts, ``False`` only when live.
        disclaimer: The mandatory broker disclaimer (:data:`BROKER_DISCLAIMER`).
    """

    account_id: str
    cash: float
    equity: float
    buying_power: float
    currency: str = "USD"
    mode: Literal["simulated", "paper", "live"]
    paper: bool
    disclaimer: str = BROKER_DISCLAIMER


class BrokerPosition(CamelModel):
    """One open broker position marked to the latest price.

    Fields:
        symbol: Asset ticker (canonical upper-case).
        qty: Units held (fractional allowed).
        avg_entry_price: Average entry price per unit (``0`` when ``qty`` is 0).
        current_price: Latest mark price per unit.
        market_value: ``qty * current_price``.
        cost_basis: Total dollars invested in the still-held units.
        unrealized_pnl: ``market_value - cost_basis``.
        unrealized_pnl_pct: ``unrealized_pnl / cost_basis * 100`` (0 if no cost).
        paper: ``True`` for simulated/paper positions, ``False`` only when live.
    """

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    paper: bool = True


class BrokerOrder(CamelModel):
    """A broker order (paper unless live is fully, deliberately enabled).

    Fields:
        id: Opaque order id.
        symbol: Asset ticker (canonical upper-case).
        side: ``'buy'`` or ``'sell'``.
        type: Order type (only ``'market'`` is supported).
        qty: Filled/ordered units (``None`` when the order was sized by
            ``notional`` and not yet filled).
        notional: Dollar amount the order was sized by (``None`` when sized by
            ``qty``).
        filled_qty: Units actually filled.
        filled_avg_price: Average fill price per unit (``0`` until filled).
        status: Order lifecycle status.
        created_at: Unix timestamp in milliseconds when the order was created.
        paper: ``True`` for simulated/paper orders, ``False`` only when live.
        disclaimer: The mandatory broker disclaimer (:data:`BROKER_DISCLAIMER`).
    """

    id: str
    symbol: str
    side: BrokerOrderSide
    type: BrokerOrderType = "market"
    qty: Optional[float] = None
    notional: Optional[float] = None
    filled_qty: float = 0.0
    filled_avg_price: float = 0.0
    status: BrokerOrderStatus
    created_at: int
    paper: bool = True
    disclaimer: str = BROKER_DISCLAIMER


class BrokerOrderRequest(CamelModel):
    """Request body for ``POST /api/broker/order`` (places a PAPER order).

    Exactly one of ``notional`` or ``qty`` should be supplied (``notional``
    wins if both are present). ``brokerAck`` is required only to attempt a LIVE
    order; it is ignored in simulated/paper modes. A live order is refused
    (HTTP 403) unless the full hard-gate is satisfied.

    Fields:
        symbol: Asset ticker (case-insensitive).
        side: ``'buy'`` or ``'sell'``.
        notional: Dollar amount to trade (sizes the order by dollars).
        qty: Units to trade (used when ``notional`` is omitted).
        type: Order type (only ``'market'`` is supported).
        broker_ack: Live-trading acknowledgement; must exactly equal
            ``"I understand this places real orders"`` to attempt a live order.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"symbol": "AAPL", "side": "buy", "notional": 100}
            ]
        }
    )

    symbol: str
    side: BrokerOrderSide
    notional: Optional[float] = None
    qty: Optional[float] = None
    type: BrokerOrderType = "market"
    broker_ack: Optional[str] = None


# ---------------------------------------------------------------------------
# Auth DTOs (email/password + JWT — see docs/AUTH.md)
# ---------------------------------------------------------------------------
#
# Sandbox/demo auth: real PBKDF2-hashed passwords + signed JWTs, but no email
# verification, no rate-limiting, and a dev signing secret by default. Password
# hashes are NEVER exposed on the wire — :class:`UserDTO` is the only user shape
# returned to clients.


class UserDTO(CamelModel):
    """A public-facing user record (never carries the password hash).

    Fields:
        id: Opaque user id (uuid).
        email: The user's lowercased email address.
        name: The user's display name.
        created_at: Unix timestamp in milliseconds when the user was created.
    """

    id: str
    email: str
    name: str
    created_at: int


class SignupRequest(CamelModel):
    """Request body for ``POST /api/auth/signup``.

    Fields:
        email: The email address to register (validated + lowercased).
        password: The plaintext password (length >= 6); never stored or logged
            raw — it is immediately PBKDF2-hashed.
        name: The user's display name.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "email": "you@example.com",
                    "password": "hunter2xx",
                    "name": "Jane Investor",
                }
            ]
        }
    )

    email: str
    password: str
    name: str


class LoginRequest(CamelModel):
    """Request body for ``POST /api/auth/login``.

    Fields:
        email: The registered email address (case-insensitive).
        password: The plaintext password to verify; never stored or logged raw.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"email": "demo@giffmemoney.app", "password": "demo1234"}
            ]
        }
    )

    email: str
    password: str


class AuthResponse(CamelModel):
    """Successful auth result: a signed token plus the public user record.

    Fields:
        token: A signed JWT (HS256) bearing the user id, email and expiry.
        user: The :class:`UserDTO` for the authenticated account.
    """

    token: str
    user: UserDTO


# ---------------------------------------------------------------------------
# High-Frequency Simulation Lab DTOs (paper-only short-horizon experiment)
# ---------------------------------------------------------------------------
#
# SAFETY / HONESTY: every payload here is a SIMULATION on synthetic data. The lab
# exists to show — truthfully — that trading faster / in smaller portions usually
# makes LESS money once spreads and fees are charged, not more. It is explicit
# that a web app cannot trade in microseconds (broker round-trips are ~100ms, a
# million times slower than real HFT), so it simulates *bars*, never microseconds.

#: Mandatory disclaimer attached to every HFT-lab payload.
HFT_DISCLAIMER: str = (
    "Educational SIMULATION on synthetic data — not financial advice and not a "
    "real trading system. This is NOT microsecond/high-frequency trading: a web "
    "app's broker round-trip is ~100ms (a million times slower than co-located "
    "HFT), so the lab simulates bars, not microseconds. There is no real edge in "
    "synthetic data; results show only how transaction costs make turnover bleed. "
    "No real funds are traded."
)

#: A short-horizon signal id understood by the lab.
HftSignal = Literal["meanrev", "momentum", "buyhold"]


class HftCostModel(CamelModel):
    """A transaction-cost preset (spread + fee + slippage)."""

    key: str
    name: str
    half_spread_bps: float
    fee_bps: float
    impact_coef: float
    round_trip_bps: float
    note: str


class HftSimRequest(CamelModel):
    """Request body for a single HFT-lab simulation.

    Fields mirror :class:`app.hft.execution.SimSpec`; all have safe defaults so a
    bare ``{}`` runs a sensible demo (mean-reversion on a $20 book, retail-crypto
    costs, re-deciding every bar).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "symbol": "BTC",
                    "amount": 20,
                    "days": 30,
                    "barsPerDay": 78,
                    "signal": "meanrev",
                    "rebalanceInterval": 1,
                    "costPreset": "retail-crypto",
                }
            ]
        }
    )

    symbol: str = "SYNTH"
    amount: float = 20.0
    days: int = 30
    bars_per_day: int = 78
    signal: HftSignal = "meanrev"
    lookback: int = 20
    rebalance_interval: int = 1
    deadband: float = 0.05
    target_vol: float = 0.25
    max_exposure: float = 1.0
    allow_short: bool = False
    stop_loss_pct: float = 3.0
    take_profit_pct: float = 0.0
    max_drawdown_pct: float = 15.0
    cooldown_bars: int = 5
    cost_preset: str = "retail-crypto"


class HftSimMetrics(CamelModel):
    """Realized metrics for one simulation (gross vs net vs buy-&-hold)."""

    gross_return_pct: float
    net_return_pct: float
    cost_drag_pct: float
    buy_hold_return_pct: float
    vs_buy_hold_pct: float
    turnover: float
    turnover_per_day: float
    trades: int
    time_in_market_pct: float
    sharpe_net: float
    sharpe_gross: float
    max_drawdown_pct: float
    hit_rate_pct: float
    final_net_value: float


class HftSimResult(CamelModel):
    """The full result of one simulation, with aligned equity curves."""

    metrics: HftSimMetrics
    net_curve: list[float] = Field(default_factory=list)
    gross_curve: list[float] = Field(default_factory=list)
    buy_hold_curve: list[float] = Field(default_factory=list)
    exposure_curve: list[float] = Field(default_factory=list)
    bars: int = 0
    bars_per_year: int = 0
    cost_model: Optional[HftCostModel] = None
    synthetic_data: bool = True
    disclaimer: str = HFT_DISCLAIMER


class HftSweepPoint(CamelModel):
    """One setting on the turnover curve."""

    interval: int
    label: str
    turnover: float
    turnover_per_day: float
    trades: int
    gross_return_pct: float
    net_return_pct: float
    cost_drag_pct: float
    sharpe_net: float
    max_drawdown_pct: float
    vs_buy_hold_pct: float


class HftSweepRequest(CamelModel):
    """Request body for a turnover sweep (same base spec as :class:`HftSimRequest`)."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"symbol": "BTC", "amount": 20, "signal": "meanrev", "costPreset": "retail-crypto"}
            ]
        }
    )

    base: HftSimRequest = Field(default_factory=HftSimRequest)
    intervals: Optional[list[int]] = None


class HftSweepResult(CamelModel):
    """The full turnover-sweep result with reference points and a verdict."""

    points: list[HftSweepPoint] = Field(default_factory=list)
    optimum_by_net_return: Optional[HftSweepPoint] = None
    optimum_by_net_sharpe: Optional[HftSweepPoint] = None
    naive_fast: Optional[HftSweepPoint] = None
    buy_hold_return_pct: float = 0.0
    verdict: str = ""
    synthetic_data: bool = True
    disclaimer: str = HFT_DISCLAIMER


__all__ = [
    # type aliases
    "AssetClass",
    "Horizon",
    "Stance",
    "StrategyCategory",
    "HORIZONS",
    # base
    "CamelModel",
    # market
    "Asset",
    "Candle",
    "PricePoint",
    # projections & signals
    "ExpectedReturn",
    "RegimeInfo",
    "BacktestMetricsDTO",
    "BacktestEquityPoint",
    "BacktestResultDTO",
    "StrategySignal",
    "RiskMetrics",
    "AssetAnalysis",
    "Recommendation",
    # strategy catalog
    "StrategyMeta",
    "RankingEntry",
    "StrategyRanking",
    "StrategyLeaderboardEntry",
    "StrategyLeaderboard",
    # portfolio
    "PortfolioRequest",
    "PortfolioPoint",
    "PortfolioWeight",
    "PortfolioResult",
    # monte carlo
    "MonteCarloBand",
    "MonteCarloBin",
    "MonteCarloResult",
    # market summary
    "Breadth",
    "SectorPerf",
    "IndexLevel",
    "MarketSummary",
    # invest / wallet type aliases
    "TxnType",
    "RiskTolerance",
    # invest / wallet
    "CardIn",
    "SavedCard",
    "Transaction",
    "Wallet",
    "DepositRequest",
    "WithdrawRequest",
    "AllocationItem",
    "InvestRequest",
    "SellRequest",
    "Position",
    "PortfolioState",
    "RiskActionType",
    "RiskPolicy",
    "RiskAction",
    "RiskApplyResult",
    "PortfolioHistoryPoint",
    "PositionHistoryPoint",
    "PositionHistory",
    "PortfolioHistory",
    "AdviceRequest",
    "AdviceItem",
    "AllocationAdvice",
    # auto-trader bot type aliases
    "BotModeId",
    "BotRiskLevel",
    "BotRotation",
    "BotSide",
    "BotVerdict",
    "BOT_DISCLAIMER",
    # auto-trader bot
    "BotMode",
    "BotConfig",
    "BotRunRequest",
    "BotTrade",
    "SleeveAttribution",
    "BotEquityPoint",
    "BotMetrics",
    "BotRunResult",
    # broker execution type aliases
    "BROKER_DISCLAIMER",
    "BrokerName",
    "BrokerOrderSide",
    "BrokerOrderType",
    "BrokerOrderStatus",
    # broker execution
    "BrokerStatus",
    "BrokerAccount",
    "BrokerPosition",
    "BrokerOrder",
    "BrokerOrderRequest",
    # auth
    "UserDTO",
    "SignupRequest",
    "LoginRequest",
    "AuthResponse",
    # HFT simulation lab
    "HFT_DISCLAIMER",
    "HftSignal",
    "HftCostModel",
    "HftSimRequest",
    "HftSimMetrics",
    "HftSimResult",
    "HftSweepPoint",
    "HftSweepRequest",
    "HftSweepResult",
]
