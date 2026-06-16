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

from app.invest.store import AccountStore, PositionState
from app.market.provider import MarketDataProvider
from app.schemas import (
    AllocationItem,
    Asset,
    PortfolioState,
    Position,
    Transaction,
    Wallet,
)

__all__ = ["PortfolioService"]

# Units below this magnitude are treated as a fully closed position.
_UNITS_EPS: float = 1e-9


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
                    )
                else:
                    state.units = float(state.units) + units
                    state.cost_basis = float(state.cost_basis) + amount
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
