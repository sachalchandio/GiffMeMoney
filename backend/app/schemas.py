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

    amount: float
    destination: Optional[str] = None


class AllocationItem(CamelModel):
    """One leg of an invest order: spend ``amount`` dollars on ``symbol``."""

    symbol: str
    amount: float


class InvestRequest(CamelModel):
    """Request body for ``POST /api/portfolio/invest``."""

    allocations: list[AllocationItem] = Field(default_factory=list)


class SellRequest(CamelModel):
    """Request body for ``POST /api/portfolio/sell``.

    Fields:
        symbol: Symbol of the position to reduce.
        amount: Dollar amount to sell; ``None`` when ``all`` is true.
        all: When true, liquidate the entire position (``amount`` ignored).
    """

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
    """

    amount: float
    risk_tolerance: RiskTolerance
    asset_classes: Optional[list[AssetClass]] = None


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

    Fields:
        items: Per-asset allocation legs (weights sum to ~1).
        expected_return: Blended annual expected return (decimal).
        expected_vol: Blended annual volatility (decimal).
        sharpe: Blended Sharpe ratio of the basket.
        horizons: Weight-blended 5-horizon expected returns for the basket.
        risk_tolerance: Echo of the requested risk profile.
        amount: Echo of the requested dollar amount.
    """

    items: list[AdviceItem] = Field(default_factory=list)
    expected_return: float
    expected_vol: float
    sharpe: float
    horizons: list[ExpectedReturn] = Field(default_factory=list)
    risk_tolerance: RiskTolerance
    amount: float


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

    email: str
    password: str
    name: str


class LoginRequest(CamelModel):
    """Request body for ``POST /api/auth/login``.

    Fields:
        email: The registered email address (case-insensitive).
        password: The plaintext password to verify; never stored or logged raw.
    """

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
    "PortfolioHistoryPoint",
    "PositionHistoryPoint",
    "PositionHistory",
    "PortfolioHistory",
    "AdviceRequest",
    "AdviceItem",
    "AllocationAdvice",
    # auth
    "UserDTO",
    "SignupRequest",
    "LoginRequest",
    "AuthResponse",
]
