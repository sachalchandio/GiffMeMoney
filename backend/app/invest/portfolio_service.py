"""Portfolio service: invest cash across symbols, sell, and mark to market.

The :class:`PortfolioService` is the positions-side of the account. It turns
cash into fractional units, realizes P&L on sells, and renders the full
mark-to-market :class:`~app.schemas.PortfolioState` (every position priced at the
provider's latest close).

Accounting model (matches :class:`~app.invest.store.PositionState`):
    * ``units``        — fractional units currently held.
    * ``cost_basis``   — total dollars currently invested in the *still-held*
      units. A buy adds ``amount`` to both cash spent and cost basis. A sell of
      ``f = units_sold / units`` reduces both ``units`` and ``cost_basis`` by the
      fraction ``f``.
    * ``realized_pnl`` — cumulative realized profit/loss from prior sells:
      ``realized_pnl += proceeds - cost_basis * f``.

Buys spend cash (``units = amount / price``); sells credit cash with the
proceeds (``units_sold * price``). Balances always reconcile because the wallet's
invested value is recomputed from these positions at the live price.

Every mutation runs under ``store.lock`` so concurrent threads never observe a
half-applied invest/sell. Bad input raises :class:`ValueError` (HTTP 400);
selling a symbol with no open position raises :class:`KeyError` (HTTP 404).
"""

from __future__ import annotations

import math
import time
import uuid

from app.invest.store import AccountStore, PositionState, RiskPolicyState
from app.market.provider import MarketDataProvider
from app.schemas import (
    AllocationItem,
    Asset,
    PortfolioState,
    Position,
    RiskAction,
    RiskApplyResult,
    RiskPolicy,
    Transaction,
    Wallet,
)

__all__ = ["PortfolioService"]

# Units below this magnitude are treated as a fully closed position.
_UNITS_EPS: float = 1e-9

# Defensive cash floor the drawdown circuit-breaker reduces *to*: once total
# portfolio value breaches ``maxDrawdownPct`` from its peak, exposure is cut so
# at least this fraction of total value is held in cash (mirrors the auto-trader
# bot's ``_DRAWDOWN_CASH_FLOOR`` so both layers de-risk consistently).
_DRAWDOWN_CASH_FLOOR: float = 0.5


def _now_ms() -> int:
    """Return the current unix time in milliseconds."""
    return int(time.time() * 1000)


class PortfolioService:
    """Buy/sell positions and render the mark-to-market portfolio view.

    Args:
        store: The process-wide :class:`~app.invest.store.AccountStore`.
        provider: A :class:`~app.market.provider.MarketDataProvider` used to
            price buys/sells and to mark positions to market.
    """

    def __init__(self, store: AccountStore, provider: MarketDataProvider) -> None:
        """Store the collaborators (state lives entirely in ``store``)."""
        self._store = store
        self._provider = provider

    # ------------------------------------------------------------------
    # Pricing helpers
    # ------------------------------------------------------------------

    def _price(self, symbol: str) -> float:
        """Return a strictly-positive latest price for ``symbol``.

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            The latest close as a positive float.

        Raises:
            KeyError: If the symbol is unknown (propagated for a 404).
            ValueError: If the provider returns a non-positive / non-finite price
                (a market data fault that should not silently corrupt accounting).
        """
        price = float(self._provider.latest_price(symbol))
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError(f"No valid market price available for {symbol!r}.")
        return price

    def _asset(self, symbol: str) -> Asset:
        """Return the :class:`~app.schemas.Asset` snapshot for ``symbol``.

        Args:
            symbol: Asset ticker (case-insensitive).

        Returns:
            The :class:`~app.schemas.Asset` snapshot.

        Raises:
            KeyError: If the symbol is unknown.
        """
        return self._provider.get_asset(symbol)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def _wallet(self, account_id: str) -> Wallet:
        """Build a reconciled :class:`~app.schemas.Wallet` from store state.

        Mirrors :meth:`app.invest.wallet.WalletService.get_wallet` so the
        portfolio view is self-contained (no cross-service call) while staying
        consistent. Must be called while holding ``store.lock``.

        Args:
            account_id: The account identifier.

        Returns:
            A populated :class:`~app.schemas.Wallet`.
        """
        account = self._store.get_account(account_id)
        cash = float(account.cash_balance)
        if not math.isfinite(cash):
            cash = 0.0
        invested = 0.0
        for state in account.positions.values():
            try:
                price = self._price(state.symbol)
            except (KeyError, ValueError):
                price = 0.0
            invested += float(state.units) * price
        if not math.isfinite(invested) or invested < 0.0:
            invested = max(0.0, invested if math.isfinite(invested) else 0.0)
        return Wallet(
            account_id=account.account_id,
            cash_balance=round(cash, 2),
            invested_value=round(invested, 2),
            total_value=round(cash + invested, 2),
            currency="USD",
            saved_cards=list(account.saved_cards),
        )

    def get_state(self, account_id: str) -> PortfolioState:
        """Render the full mark-to-market :class:`~app.schemas.PortfolioState`.

        Every open position is priced at the provider's latest close. Per
        position::

            current_price      = latest_price(symbol)
            market_value       = units * current_price
            avg_price          = cost_basis / units            (0 when units ~ 0)
            unrealized_pnl     = market_value - cost_basis
            unrealized_pnl_pct = unrealized_pnl / cost_basis * 100   (0 if no cost)
            allocation_pct     = market_value / Σ market_value * 100

        Totals are ``Σ cost_basis``, ``Σ market_value`` and the derived total
        P&L / P&L %. A position whose symbol can no longer be priced is marked at
        zero rather than dropped, so the view never crashes.

        Args:
            account_id: The account identifier.

        Returns:
            A populated :class:`~app.schemas.PortfolioState`.
        """
        with self._store.lock:
            account = self._store.get_account(account_id)
            wallet = self._wallet(account_id)

            priced: list[tuple[PositionState, Asset, float, float]] = []
            total_market = 0.0
            for state in account.positions.values():
                try:
                    asset = self._asset(state.symbol)
                    price = self._price(state.symbol)
                except (KeyError, ValueError):
                    # Fall back to a snapshot-less mark of zero for a faulty symbol.
                    try:
                        asset = self._asset(state.symbol)
                    except KeyError:
                        continue
                    price = 0.0
                market_value = float(state.units) * price
                if not math.isfinite(market_value):
                    market_value = 0.0
                # Ratchet the trailing-stop high-water mark up to the latest mark
                # (never down). Reading the portfolio keeps the peak current so a
                # later evaluate_risk() trailing stop is measured from the true peak.
                if price > 0.0 and math.isfinite(price):
                    state.high_water_price = max(
                        float(state.high_water_price), price
                    )
                total_market += market_value
                priced.append((state, asset, price, market_value))

            positions: list[Position] = []
            total_cost = 0.0
            for state, asset, price, market_value in priced:
                cost_basis = float(state.cost_basis)
                units = float(state.units)
                avg_price = cost_basis / units if units > _UNITS_EPS else 0.0
                unrealized = market_value - cost_basis
                unrealized_pct = (
                    unrealized / cost_basis * 100.0 if cost_basis > 0.0 else 0.0
                )
                allocation_pct = (
                    market_value / total_market * 100.0 if total_market > 0.0 else 0.0
                )
                total_cost += cost_basis
                positions.append(
                    Position(
                        symbol=state.symbol,
                        asset=asset,
                        units=round(units, 8),
                        cost_basis=round(cost_basis, 2),
                        avg_price=round(self._finite(avg_price), 6),
                        current_price=round(self._finite(price), 6),
                        market_value=round(self._finite(market_value), 2),
                        unrealized_pnl=round(self._finite(unrealized), 2),
                        unrealized_pnl_pct=round(self._finite(unrealized_pct), 4),
                        allocation_pct=round(self._finite(allocation_pct), 4),
                        realized_pnl=round(self._finite(float(state.realized_pnl)), 2),
                        opened_at=int(state.opened_at),
                    )
                )

            # Stable ordering: largest market value first.
            positions.sort(key=lambda p: p.market_value, reverse=True)

            total_value = total_market
            total_pnl = total_value - total_cost
            total_pnl_pct = (
                total_pnl / total_cost * 100.0 if total_cost > 0.0 else 0.0
            )

            # Ratchet the account-level peak (cash + invested) so the drawdown
            # circuit-breaker is measured from the true high-water value.
            account_value = float(wallet.cash_balance) + total_market
            if math.isfinite(account_value):
                account.peak_value = max(float(account.peak_value), account_value)

            return PortfolioState(
                wallet=wallet,
                positions=positions,
                total_cost=round(self._finite(total_cost), 2),
                total_value=round(self._finite(total_value), 2),
                total_pnl=round(self._finite(total_pnl), 2),
                total_pnl_pct=round(self._finite(total_pnl_pct), 4),
            )

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def invest(
        self, account_id: str, allocations: list[AllocationItem]
    ) -> PortfolioState:
        """Spend cash across one or more symbols, opening/adding positions.

        Validates the whole order *before* mutating anything (all-or-nothing):
        every leg's amount must be ``> 0`` and finite, every symbol must be
        known and priceable, and the total spend must not exceed available cash.
        Then for each leg::

            units = amount / price
            position.units      += units
            position.cost_basis += amount      (cost basis = dollars invested)
            cash                -= amount

        A new position records ``opened_at`` at first purchase; re-investing in an
        existing symbol keeps the original ``opened_at`` and simply adds units and
        cost basis (so ``avg_price`` is the blended average cost). One ``buy``
        transaction is recorded per leg.

        Args:
            account_id: The account to invest from.
            allocations: The legs of the order (``symbol`` + dollar ``amount``).

        Returns:
            The updated :class:`~app.schemas.PortfolioState`.

        Raises:
            ValueError: If ``allocations`` is empty, any amount is ``<= 0`` /
                non-finite, or the total spend exceeds available cash (HTTP 400).
            KeyError: If any symbol is unknown (HTTP 404).
        """
        if not allocations:
            raise ValueError("At least one allocation is required.")

        with self._store.lock:
            account = self._store.get_account(account_id)

            # ---- Validate the entire order before applying any of it. ----
            resolved: list[tuple[str, float, float]] = []  # (symbol, amount, price)
            total_spend = 0.0
            for item in allocations:
                symbol = str(item.symbol).strip().upper()
                if not symbol:
                    raise ValueError("Allocation symbol must not be empty.")
                try:
                    amount = float(item.amount)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Allocation amount for {symbol!r} must be a number."
                    ) from None
                if not math.isfinite(amount) or amount <= 0.0:
                    raise ValueError(
                        f"Allocation amount for {symbol!r} must be greater than zero."
                    )
                # Validates the symbol (KeyError -> 404) and prices it.
                price = self._price(symbol)
                resolved.append((symbol, amount, price))
                total_spend += amount

            cash = float(account.cash_balance)
            if total_spend > cash + 1e-9:
                raise ValueError(
                    f"Insufficient funds: order totals ${total_spend:,.2f} "
                    f"but only ${cash:,.2f} is available."
                )

            # ---- Apply the order. ----
            now = _now_ms()
            for symbol, amount, price in resolved:
                units = amount / price
                state = account.positions.get(symbol)
                if state is None:
                    account.positions[symbol] = PositionState(
                        symbol=symbol,
                        units=units,
                        cost_basis=amount,
                        realized_pnl=0.0,
                        opened_at=now,
                        # Seed the trailing-stop peak at the execution price so the
                        # high-water mark is meaningful from the first mark-to-market.
                        high_water_price=price,
                    )
                else:
                    state.units = float(state.units) + units
                    state.cost_basis = float(state.cost_basis) + amount
                    # Ratchet the peak up to the latest buy price (never down).
                    state.high_water_price = max(
                        float(state.high_water_price), price
                    )
                account.cash_balance = float(account.cash_balance) - amount
                account.transactions.append(
                    self._buy_txn(symbol, amount, units, price)
                )

            if account.cash_balance < 0.0 or not math.isfinite(account.cash_balance):
                account.cash_balance = max(0.0, round(account.cash_balance, 6))

            return self.get_state(account_id)

    def sell(
        self,
        account_id: str,
        symbol: str,
        amount: float | None = None,
        sell_all: bool = False,
    ) -> PortfolioState:
        """Reduce or liquidate a position, realizing P&L and crediting cash.

        ``units_sold`` is the full holding when ``sell_all`` is true, otherwise
        ``amount / price`` (capped at the held units). Then::

            f               = units_sold / units            (fraction sold)
            proceeds        = units_sold * price
            realized_pnl   += proceeds - cost_basis * f
            units          -= units_sold
            cost_basis     -= cost_basis * f                 (pro-rata reduction)
            cash           += proceeds

        When the remaining units fall to ~0 the position is dropped (its
        ``realized_pnl`` is folded into the realized total that the wallet/ledger
        already reflect). One ``sell`` transaction is recorded.

        Args:
            account_id: The account holding the position.
            symbol: The symbol to sell.
            amount: Dollar amount to sell; ignored when ``sell_all`` is true.
                Required (``> 0``) otherwise.
            sell_all: When true, liquidate the entire position.

        Returns:
            The updated :class:`~app.schemas.PortfolioState`.

        Raises:
            ValueError: If ``amount`` is missing/``<= 0`` when not selling all
                (HTTP 400).
            KeyError: If there is no open position for ``symbol`` (HTTP 404).
        """
        sym = str(symbol).strip().upper()
        with self._store.lock:
            account = self._store.get_account(account_id)
            state = account.positions.get(sym)
            if state is None or float(state.units) <= _UNITS_EPS:
                raise KeyError(f"No open position for {sym!r}.")

            price = self._price(sym)
            units = float(state.units)

            if sell_all:
                units_sold = units
            else:
                if amount is None:
                    raise ValueError(
                        "A sell amount is required unless selling the whole position."
                    )
                try:
                    dollars = float(amount)
                except (TypeError, ValueError):
                    raise ValueError("Sell amount must be a number.") from None
                if not math.isfinite(dollars) or dollars <= 0.0:
                    raise ValueError("Sell amount must be greater than zero.")
                units_sold = min(units, dollars / price)

            if units_sold <= _UNITS_EPS:
                raise ValueError("Sell amount is too small to execute.")

            fraction = units_sold / units if units > 0.0 else 1.0
            fraction = min(1.0, max(0.0, fraction))
            cost_basis = float(state.cost_basis)
            proceeds = units_sold * price
            realized = proceeds - cost_basis * fraction

            state.realized_pnl = float(state.realized_pnl) + realized
            state.units = units - units_sold
            state.cost_basis = cost_basis - cost_basis * fraction
            account.cash_balance = float(account.cash_balance) + proceeds
            if not math.isfinite(account.cash_balance):
                account.cash_balance = round(cost_basis, 6)

            account.transactions.append(
                self._sell_txn(sym, proceeds, units_sold, price, realized)
            )

            # Drop the position once it is effectively closed.
            if state.units <= _UNITS_EPS:
                del account.positions[sym]

            return self.get_state(account_id)

    # ------------------------------------------------------------------
    # Risk policy (post-buy loss controls)
    # ------------------------------------------------------------------

    def get_risk_policy(self, account_id: str) -> RiskPolicy:
        """Return the account's current :class:`~app.schemas.RiskPolicy`.

        Args:
            account_id: The account identifier.

        Returns:
            The stored policy (all fields ``None`` / OFF by default).
        """
        with self._store.lock:
            account = self._store.get_account(account_id)
            return self._policy_to_dto(account.risk_policy)

    def set_risk_policy(self, account_id: str, policy: RiskPolicy) -> RiskPolicy:
        """Validate and store the account's post-buy loss-control policy.

        Every supplied threshold must be a finite number ``> 0`` (a percent);
        omit a field or pass ``null`` to disable that rule. The policy never
        affects buys — it is only consulted by :meth:`evaluate_risk`.

        Args:
            account_id: The account identifier.
            policy: The desired :class:`~app.schemas.RiskPolicy`.

        Returns:
            The stored policy (echoes the validated input).

        Raises:
            ValueError: If any supplied threshold is ``<= 0`` or non-finite
                (HTTP 400).
        """
        stop = self._validate_pct(policy.stop_loss_pct, "stopLossPct")
        trail = self._validate_pct(policy.trailing_stop_pct, "trailingStopPct")
        take = self._validate_pct(policy.take_profit_pct, "takeProfitPct")
        draw = self._validate_pct(policy.max_drawdown_pct, "maxDrawdownPct")
        with self._store.lock:
            account = self._store.get_account(account_id)
            account.risk_policy = RiskPolicyState(
                stop_loss_pct=stop,
                trailing_stop_pct=trail,
                take_profit_pct=take,
                max_drawdown_pct=draw,
            )
            return self._policy_to_dto(account.risk_policy)

    def evaluate_risk(self, account_id: str) -> RiskApplyResult:
        """Apply the account's :class:`~app.schemas.RiskPolicy` to held positions.

        Marks every position to the current provider price, then, per the active
        policy:

        * **Stop-loss** — a position down more than ``stopLossPct`` from its
          blended entry (avg-cost) price is fully sold.
        * **Trailing stop** — a position down more than ``trailingStopPct`` from
          its observed high-water-mark price is fully sold.
        * **Take-profit** — a position up more than ``takeProfitPct`` above its
          blended entry price is fully sold.
        * **Max drawdown** — if total portfolio value is down more than
          ``maxDrawdownPct`` from its peak, the worst-performing positions are
          sold (worst unrealized-% first) until the drawdown is back within the
          limit (raising cash / reducing exposure).

        Each protective sell reuses :meth:`sell` (so accounting, the ledger and
        cash all stay consistent) and is recorded as a :class:`~app.schemas.RiskAction`.
        With an all-OFF policy (the default) nothing ever triggers.

        Args:
            account_id: The account identifier.

        Returns:
            A :class:`~app.schemas.RiskApplyResult` listing every protective
            action taken plus the post-evaluation portfolio state.
        """
        with self._store.lock:
            account = self._store.get_account(account_id)
            policy = self._policy_to_dto(account.risk_policy)
            actions: list[RiskAction] = []

            # Refresh marks + ratchet the per-position / account peaks.
            self.get_state(account_id)

            # ---- Per-position rules (stop-loss / trailing / take-profit). ----
            # Snapshot the symbols up front: selling mutates the positions dict.
            for symbol in list(account.positions.keys()):
                state = account.positions.get(symbol)
                if state is None or float(state.units) <= _UNITS_EPS:
                    continue
                try:
                    price = self._price(symbol)
                except (KeyError, ValueError):
                    continue
                units = float(state.units)
                cost_basis = float(state.cost_basis)
                entry = cost_basis / units if units > _UNITS_EPS else 0.0
                peak = max(float(state.high_water_price), price, entry)

                action_type = None
                reason = ""
                if entry > 0.0:
                    change_pct = (price / entry - 1.0) * 100.0
                    drop_pct = (1.0 - price / peak) * 100.0 if peak > 0.0 else 0.0
                    if (
                        policy.stop_loss_pct is not None
                        and change_pct <= -policy.stop_loss_pct
                    ):
                        action_type = "stop_loss"
                        reason = (
                            f"{symbol} is down {abs(change_pct):.2f}% from entry "
                            f"(${entry:,.2f}); stop-loss is "
                            f"{policy.stop_loss_pct:.2f}%."
                        )
                    elif (
                        policy.take_profit_pct is not None
                        and change_pct >= policy.take_profit_pct
                    ):
                        action_type = "take_profit"
                        reason = (
                            f"{symbol} is up {change_pct:.2f}% from entry "
                            f"(${entry:,.2f}); take-profit is "
                            f"{policy.take_profit_pct:.2f}%."
                        )
                    elif (
                        policy.trailing_stop_pct is not None
                        and drop_pct >= policy.trailing_stop_pct
                    ):
                        action_type = "trailing_stop"
                        reason = (
                            f"{symbol} is down {drop_pct:.2f}% from its peak "
                            f"(${peak:,.2f}); trailing stop is "
                            f"{policy.trailing_stop_pct:.2f}%."
                        )

                if action_type is not None:
                    actions.append(
                        self._protective_sell(
                            account_id, symbol, action_type, reason
                        )
                    )

            # ---- Portfolio-level drawdown circuit-breaker. ----
            if policy.max_drawdown_pct is not None:
                actions.extend(
                    self._reduce_on_drawdown(account_id, policy.max_drawdown_pct)
                )

            state = self.get_state(account_id)
            return RiskApplyResult(
                actions=actions,
                policy=policy,
                state=state,
                triggered=bool(actions),
            )

    # ------------------------------------------------------------------
    # Risk-policy helpers
    # ------------------------------------------------------------------

    def _reduce_on_drawdown(
        self, account_id: str, max_drawdown_pct: float
    ) -> list[RiskAction]:
        """Reduce exposure to a defensive cash floor when drawdown is breached.

        A protective *sell* converts marked position value into an equal amount
        of cash at the same price, so it does **not** change total account value
        — the paper loss is already locked into the peak-vs-value drawdown and
        cannot be "sold away". A drawdown circuit-breaker therefore de-risks
        rather than tries to recover the number: when total portfolio value is
        more than ``max_drawdown_pct`` below its peak, this sells the worst-
        performing positions (worst unrealized-% first) until invested exposure
        is at or below :data:`_DRAWDOWN_CASH_FLOOR` of total value (i.e. at least
        a defensive cash floor is held), mirroring the auto-trader bot's
        circuit-breaker. With no breach nothing is sold.

        Args:
            account_id: The account identifier.
            max_drawdown_pct: The drawdown circuit-breaker threshold (percent).

        Returns:
            The protective :class:`~app.schemas.RiskAction` list (possibly empty).
        """
        actions: list[RiskAction] = []
        account = self._store.get_account(account_id)

        value, peak = self._account_value_and_peak(account_id)
        if peak <= 0.0:
            return actions
        drawdown_pct = (1.0 - value / peak) * 100.0
        if drawdown_pct <= max_drawdown_pct:
            return actions  # within the limit — no de-risking needed.

        # Target invested exposure after de-risking: hold at least the defensive
        # cash floor of total value in cash.
        target_invested = value * (1.0 - _DRAWDOWN_CASH_FLOOR)

        # Sell worst-first until invested value is at/below the target (or no
        # priceable positions remain). One sell per held position at most.
        for _ in range(len(account.positions) + 1):
            invested = self._invested_value(account_id)
            if invested <= target_invested + _UNITS_EPS:
                break
            worst = self._worst_position(account_id)
            if worst is None:
                break
            symbol, change_pct = worst
            reason = (
                f"Portfolio is down {drawdown_pct:.2f}% from its peak "
                f"(${peak:,.2f}); max-drawdown limit is "
                f"{max_drawdown_pct:.2f}%. Reducing exposure to a defensive "
                f"cash floor by selling {symbol} (worst position, "
                f"{change_pct:+.2f}% vs entry)."
            )
            actions.append(
                self._protective_sell(account_id, symbol, "drawdown", reason)
            )
        return actions

    def _invested_value(self, account_id: str) -> float:
        """Return the current marked-to-market invested value (Σ market_value).

        Args:
            account_id: The account identifier.

        Returns:
            The total invested value as a finite, non-negative float.
        """
        account = self._store.get_account(account_id)
        invested = 0.0
        for state in account.positions.values():
            try:
                price = self._price(state.symbol)
            except (KeyError, ValueError):
                continue
            mv = float(state.units) * price
            if math.isfinite(mv):
                invested += mv
        return invested if math.isfinite(invested) and invested > 0.0 else max(
            0.0, invested if math.isfinite(invested) else 0.0
        )

    def _account_value_and_peak(self, account_id: str) -> tuple[float, float]:
        """Return ``(account_value, peak_value)`` for the drawdown check.

        ``account_value`` is ``cash + Σ market_value``; ``peak_value`` is the
        account's ratcheted high-water value.

        Args:
            account_id: The account identifier.

        Returns:
            A ``(value, peak)`` tuple of finite floats.
        """
        account = self._store.get_account(account_id)
        cash = float(account.cash_balance)
        invested = 0.0
        for state in account.positions.values():
            try:
                price = self._price(state.symbol)
            except (KeyError, ValueError):
                continue
            mv = float(state.units) * price
            if math.isfinite(mv):
                invested += mv
        value = cash + invested
        value = value if math.isfinite(value) else 0.0
        peak = float(account.peak_value)
        peak = peak if math.isfinite(peak) and peak > 0.0 else value
        return value, peak

    def _worst_position(self, account_id: str) -> tuple[str, float] | None:
        """Return the worst-performing held position by unrealized return %.

        Args:
            account_id: The account identifier.

        Returns:
            A ``(symbol, change_pct)`` tuple for the position with the lowest
            unrealized return vs entry, or ``None`` if there are no priceable
            positions.
        """
        account = self._store.get_account(account_id)
        worst: tuple[str, float] | None = None
        for state in account.positions.values():
            units = float(state.units)
            if units <= _UNITS_EPS:
                continue
            try:
                price = self._price(state.symbol)
            except (KeyError, ValueError):
                continue
            cost_basis = float(state.cost_basis)
            entry = cost_basis / units if units > _UNITS_EPS else 0.0
            change_pct = (price / entry - 1.0) * 100.0 if entry > 0.0 else 0.0
            if worst is None or change_pct < worst[1]:
                worst = (state.symbol, change_pct)
        return worst

    def _protective_sell(
        self, account_id: str, symbol: str, action: str, reason: str
    ) -> RiskAction:
        """Liquidate ``symbol`` and build the corresponding :class:`RiskAction`.

        Args:
            account_id: The account identifier.
            symbol: The position to sell in full.
            action: The :data:`~app.schemas.RiskActionType` that fired.
            reason: Human-readable explanation for the action.

        Returns:
            A populated :class:`~app.schemas.RiskAction`.
        """
        account = self._store.get_account(account_id)
        state = account.positions.get(symbol)
        units = float(state.units) if state is not None else 0.0
        cost_basis = float(state.cost_basis) if state is not None else 0.0
        price = self._price(symbol)
        # Realized P&L mirrors PortfolioService.sell's accounting for a full exit.
        realized = units * price - cost_basis
        self.sell(account_id, symbol, sell_all=True)
        return RiskAction(
            symbol=symbol,
            action=action,  # type: ignore[arg-type]
            reason=reason,
            amount=round(self._finite(units * price), 2),
            units_sold=round(self._finite(units), 8),
            price=round(self._finite(price), 6),
            realized_pnl=round(self._finite(realized), 2),
        )

    @staticmethod
    def _policy_to_dto(state: RiskPolicyState) -> RiskPolicy:
        """Project a stored :class:`RiskPolicyState` onto the wire DTO.

        Args:
            state: The stored risk-policy state.

        Returns:
            The equivalent :class:`~app.schemas.RiskPolicy`.
        """
        return RiskPolicy(
            stop_loss_pct=state.stop_loss_pct,
            trailing_stop_pct=state.trailing_stop_pct,
            take_profit_pct=state.take_profit_pct,
            max_drawdown_pct=state.max_drawdown_pct,
        )

    @staticmethod
    def _validate_pct(value: float | None, field: str) -> float | None:
        """Validate one optional percentage threshold.

        Args:
            value: The supplied percent (or ``None`` to disable the rule).
            field: The camelCase field name (for the error message).

        Returns:
            The validated positive percent, or ``None`` when disabled.

        Raises:
            ValueError: If ``value`` is not ``None`` and is ``<= 0`` or
                non-finite.
        """
        if value is None:
            return None
        try:
            pct = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be a number.") from None
        if not math.isfinite(pct) or pct <= 0.0:
            raise ValueError(f"{field} must be a positive percentage.")
        return pct

    # ------------------------------------------------------------------
    # Transaction builders
    # ------------------------------------------------------------------

    def _buy_txn(
        self, symbol: str, amount: float, units: float, price: float
    ) -> Transaction:
        """Build a completed ``buy`` ledger entry.

        Args:
            symbol: The purchased symbol.
            amount: Dollars spent.
            units: Units acquired.
            price: Execution price per unit.

        Returns:
            A completed :class:`~app.schemas.Transaction` of type ``buy``.
        """
        txn_id = str(uuid.uuid4())
        return Transaction(
            id=txn_id,
            type="buy",
            amount=round(float(amount), 2),
            symbol=symbol,
            status="completed",
            created_at=_now_ms(),
            ref=f"buy_{txn_id[:8]}",
            note=f"Bought {units:.6f} {symbol} @ ${price:,.2f}",
        )

    def _sell_txn(
        self,
        symbol: str,
        proceeds: float,
        units: float,
        price: float,
        realized: float,
    ) -> Transaction:
        """Build a completed ``sell`` ledger entry.

        Args:
            symbol: The sold symbol.
            proceeds: Cash credited from the sale.
            units: Units sold.
            price: Execution price per unit.
            realized: Realized P&L on this sale (for the human-readable note).

        Returns:
            A completed :class:`~app.schemas.Transaction` of type ``sell``.
        """
        txn_id = str(uuid.uuid4())
        sign = "+" if realized >= 0 else "-"
        return Transaction(
            id=txn_id,
            type="sell",
            amount=round(float(proceeds), 2),
            symbol=symbol,
            status="completed",
            created_at=_now_ms(),
            ref=f"sell_{txn_id[:8]}",
            note=(
                f"Sold {units:.6f} {symbol} @ ${price:,.2f} "
                f"(realized {sign}${abs(realized):,.2f})"
            ),
        )

    @staticmethod
    def _finite(value: float, default: float = 0.0) -> float:
        """Return ``value`` as a finite float, falling back to ``default``.

        Args:
            value: Candidate number.
            default: Substitute for NaN / +-inf / non-numeric input.

        Returns:
            A finite float.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default
