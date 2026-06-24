"""Alpaca broker adapter (REST over httpx) — PAPER by default, live HARD-GATED.

Implements :class:`~app.broker.base.BrokerProvider` against the Alpaca Trading
API (https://alpaca.markets). It maps Alpaca's account / positions / orders onto
the frozen broker DTOs in :mod:`app.schemas`.

SAFETY (non-negotiable, enforced in code):

* The base URL **defaults to Alpaca's PAPER (sandbox) endpoint**
  (``https://paper-api.alpaca.markets``), so even with real keys set, no real
  money moves. ``is_paper`` is ``True`` unless the *full* live hard-gate passes.
* **LIVE trading is hard-gated.** :attr:`live_enabled` is ``True`` only when
  ALL of these hold:
    1. ``settings.broker == 'alpaca'``;
    2. ``settings.alpaca_live`` is truthy;
    3. ``settings.alpaca_base_url`` points at the live host (not the paper host);
    4. real API keys are configured;
    5. ``settings.broker_ack`` exactly equals :data:`LIVE_ACK_PHRASE`.
  Even then, :meth:`submit_order` additionally requires the *caller's* per-order
  ``broker_ack`` to match exactly. Any missing piece → the order is refused with
  :class:`~app.broker.base.LiveTradingNotEnabledError` (HTTP 403). This repo
  ships with live OFF.
* If live is configured, a loud WARNING is logged at construction. Market data
  pulled here is read-only.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.schemas import (
    BROKER_DISCLAIMER,
    BrokerAccount,
    BrokerOrder,
    BrokerOrderSide,
    BrokerOrderStatus,
    BrokerPosition,
    BrokerStatus,
)

from app.broker.base import (
    LIVE_ACK_PHRASE,
    BrokerError,
    BrokerProvider,
    LiveTradingNotEnabledError,
)

__all__ = ["AlpacaBroker", "PAPER_BASE_URL", "LIVE_BASE_URL"]

logger = logging.getLogger("app.broker.alpaca")

#: Alpaca's PAPER (sandbox) trading host — the safe default. No real money.
PAPER_BASE_URL = "https://paper-api.alpaca.markets"

#: Alpaca's LIVE trading host. Reaching this requires the full hard-gate.
LIVE_BASE_URL = "https://api.alpaca.markets"

#: Per-call HTTP timeout (seconds).
_TIMEOUT: float = 8.0

#: Map Alpaca order statuses onto our :data:`~app.schemas.BrokerOrderStatus`.
_STATUS_MAP: Dict[str, BrokerOrderStatus] = {
    "new": "accepted",
    "accepted": "accepted",
    "pending_new": "pending",
    "accepted_for_bidding": "accepted",
    "filled": "filled",
    "partially_filled": "partially_filled",
    "done_for_day": "accepted",
    "canceled": "canceled",
    "cancelled": "canceled",
    "expired": "canceled",
    "replaced": "accepted",
    "pending_cancel": "pending",
    "pending_replace": "pending",
    "rejected": "rejected",
    "suspended": "pending",
    "calculated": "pending",
    "stopped": "accepted",
}


def _is_paper_host(base_url: str) -> bool:
    """Return whether ``base_url`` points at the Alpaca PAPER host.

    Args:
        base_url: The configured Alpaca REST base URL.

    Returns:
        ``True`` unless the URL clearly targets the live host. Defaults to the
        safe answer (paper) for an unrecognised URL.
    """
    url = (base_url or "").strip().rstrip("/").lower()
    if not url:
        return True
    if url == LIVE_BASE_URL:
        return False
    # Treat the canonical paper host (and anything that isn't the live host) as
    # paper, so an unknown/misconfigured URL never silently goes live.
    return True


class AlpacaBroker(BrokerProvider):
    """Alpaca REST broker adapter (paper by default; live is hard-gated).

    The constructor reads keys / base URL / live flags from
    :data:`app.config.settings` (overridable for tests). It performs **no**
    network I/O at construction; it only computes :attr:`is_paper` /
    :attr:`live_enabled` from the hard-gate so the API can advertise the mode
    without hitting Alpaca.

    Args:
        api_key: Alpaca API key id; defaults to ``settings.alpaca_api_key``.
        secret_key: Alpaca API secret; defaults to ``settings.alpaca_secret_key``.
        base_url: Alpaca REST base URL; defaults to ``settings.alpaca_base_url``
            (the PAPER host).
        live_flag: The master live flag; defaults to ``settings.alpaca_live``.
        broker_ack: The configured ack; defaults to ``settings.broker_ack``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        live_flag: Optional[bool] = None,
        broker_ack: Optional[str] = None,
    ) -> None:
        """Compute the mode from the hard-gate (no network I/O here)."""
        self._api_key = api_key if api_key is not None else settings.alpaca_api_key
        self._secret_key = (
            secret_key if secret_key is not None else settings.alpaca_secret_key
        )
        self._base_url = (
            base_url if base_url is not None else settings.alpaca_base_url
        ) or PAPER_BASE_URL
        self._base_url = self._base_url.rstrip("/")
        self._live_flag = (
            bool(settings.alpaca_live) if live_flag is None else bool(live_flag)
        )
        self._configured_ack = (
            settings.broker_ack if broker_ack is None else broker_ack
        )

        # The full live hard-gate: every piece must hold or we stay on paper.
        self.live_enabled: bool = (
            (settings.broker or "").strip().lower() == "alpaca"
            and self._live_flag
            and not _is_paper_host(self._base_url)
            and self._has_keys()
            and self._configured_ack == LIVE_ACK_PHRASE
        )
        # ``is_paper`` is True unless the full gate passes. This drives the
        # ``paper`` flag on every DTO so a paper fill is never mislabelled.
        self.is_paper: bool = not self.live_enabled

        if self.live_enabled:
            logger.warning(
                "ALPACA LIVE TRADING IS ENABLED — real orders may be placed "
                "against %s. This is a deliberate, hard-gated configuration.",
                self._base_url,
            )

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------

    def _has_keys(self) -> bool:
        """Return whether both Alpaca API credentials are present."""
        return bool(self._api_key) and bool(self._secret_key)

    @property
    def _mode(self) -> str:
        """Return the human-facing mode string for DTOs/logs."""
        if self.live_enabled:
            return "live"
        return "paper"

    def _headers(self) -> Dict[str, str]:
        """Return the Alpaca auth headers."""
        return {
            "APCA-API-KEY-ID": self._api_key or "",
            "APCA-API-SECRET-KEY": self._secret_key or "",
        }

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Perform an authenticated request and return decoded JSON.

        Args:
            method: HTTP method (``GET`` / ``POST`` / ``DELETE``).
            path: API path beginning with ``/`` (e.g. ``/v2/account``).
            json: Optional JSON body.
            params: Optional query parameters.

        Returns:
            The decoded JSON payload.

        Raises:
            BrokerError: On any HTTP error, timeout, or JSON decode failure.
        """
        if not self._has_keys():
            raise BrokerError("Alpaca API keys are not configured.")
        url = f"{self._base_url}{path}"
        try:
            with httpx.Client(timeout=_TIMEOUT) as client:
                resp = client.request(
                    method, url, headers=self._headers(), json=json, params=params
                )
            resp.raise_for_status()
            if resp.status_code == 204 or not resp.content:
                return {}
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - normalise every failure
            raise BrokerError(f"Alpaca {method} {path} failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def status(self) -> BrokerStatus:
        """Return the Alpaca broker status without hitting the network.

        Connectivity is reported as "configured" (keys present) rather than a
        live ping, to keep ``/status`` cheap and side-effect free.
        """
        connected = self._has_keys()
        if not connected:
            message = "Alpaca selected but API keys are not configured."
        elif self.live_enabled:
            message = "LIVE trading enabled (hard-gate satisfied)."
        else:
            message = (
                "Paper mode (Alpaca PAPER endpoint). Live trading is disabled: "
                "the full hard-gate (live host + alpaca_live + matching "
                "broker_ack + keys) is not satisfied."
            )
        return BrokerStatus(
            broker="alpaca",
            mode=self._mode,  # type: ignore[arg-type]
            paper=self.is_paper,
            connected=connected,
            live_enabled=self.live_enabled,
            base_url=self._base_url,
            message=message,
            disclaimer=BROKER_DISCLAIMER,
        )

    def get_account(self) -> BrokerAccount:
        """Return the Alpaca account summary mapped to :class:`BrokerAccount`."""
        data = self._request("GET", "/v2/account")
        return BrokerAccount(
            account_id=str(data.get("account_number") or data.get("id") or "alpaca"),
            cash=_to_float(data.get("cash")),
            equity=_to_float(data.get("equity")),
            buying_power=_to_float(data.get("buying_power")),
            currency=str(data.get("currency") or "USD"),
            mode=self._mode,  # type: ignore[arg-type]
            paper=self.is_paper,
            disclaimer=BROKER_DISCLAIMER,
        )

    def get_positions(self) -> List[BrokerPosition]:
        """Return open Alpaca positions mapped to :class:`BrokerPosition`."""
        data = self._request("GET", "/v2/positions")
        if not isinstance(data, list):
            return []
        out: List[BrokerPosition] = []
        for p in data:
            qty = _to_float(p.get("qty"))
            cost_basis = _to_float(p.get("cost_basis"))
            out.append(
                BrokerPosition(
                    symbol=str(p.get("symbol", "")).upper(),
                    qty=qty,
                    avg_entry_price=_to_float(p.get("avg_entry_price")),
                    current_price=_to_float(p.get("current_price")),
                    market_value=_to_float(p.get("market_value")),
                    cost_basis=cost_basis,
                    unrealized_pnl=_to_float(p.get("unrealized_pl")),
                    unrealized_pnl_pct=_to_float(p.get("unrealized_plpc")) * 100.0,
                    paper=self.is_paper,
                )
            )
        return out

    def list_orders(self) -> List[BrokerOrder]:
        """Return recent Alpaca orders mapped to :class:`BrokerOrder`."""
        data = self._request(
            "GET", "/v2/orders", params={"status": "all", "limit": 100}
        )
        if not isinstance(data, list):
            return []
        return [self._map_order(o) for o in data]

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        side: BrokerOrderSide,
        *,
        notional: Optional[float] = None,
        qty: Optional[float] = None,
        type: str = "market",
        broker_ack: Optional[str] = None,
    ) -> BrokerOrder:
        """Submit a market order to Alpaca (PAPER unless the hard-gate passes).

        A LIVE order is **refused** unless ALL of the following hold: the broker
        instance is live-enabled (the configured hard-gate) AND the caller's
        per-order ``broker_ack`` exactly matches :data:`LIVE_ACK_PHRASE`.

        Args:
            symbol: Asset ticker (case-insensitive).
            side: ``'buy'`` or ``'sell'``.
            notional: Dollar amount to trade (sizes by dollars).
            qty: Units to trade (used when ``notional`` is omitted).
            type: Order type (only ``'market'`` is supported).
            broker_ack: Per-order live acknowledgement (required only for live).

        Returns:
            The submitted :class:`~app.schemas.BrokerOrder`.

        Raises:
            LiveTradingNotEnabledError: If this is a live broker but the caller's
                ``broker_ack`` does not match (HTTP 403).
            BrokerError: On invalid input or a submission failure.
        """
        sym = str(symbol).strip().upper()
        if not sym:
            raise BrokerError("Order symbol must not be empty.")
        if side not in ("buy", "sell"):
            raise BrokerError("Order side must be 'buy' or 'sell'.")
        if str(type).strip().lower() != "market":
            raise BrokerError("Only 'market' orders are supported.")

        # Hard-gate the LIVE path one more time at the call site: even a
        # live-enabled broker still demands the caller's exact ack per order.
        if self.live_enabled and broker_ack != LIVE_ACK_PHRASE:
            raise LiveTradingNotEnabledError(
                "Live trading requires the exact acknowledgement "
                f'"{LIVE_ACK_PHRASE}" on the order. Refusing to place a real order.'
            )

        body: Dict[str, Any] = {
            "symbol": sym,
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        if notional is not None:
            n = _require_positive(notional, "notional")
            body["notional"] = n
        elif qty is not None:
            q = _require_positive(qty, "qty")
            body["qty"] = q
        else:
            raise BrokerError("An order must specify a positive 'notional' or 'qty'.")

        data = self._request("POST", "/v2/orders", json=body)
        return self._map_order(data)

    def cancel_order(self, order_id: str) -> BrokerOrder:
        """Cancel an Alpaca order by id and return its updated state.

        Args:
            order_id: The Alpaca order id.

        Returns:
            The updated :class:`~app.schemas.BrokerOrder`.

        Raises:
            BrokerError: If the cancel or the follow-up fetch fails.
        """
        oid = str(order_id).strip()
        if not oid:
            raise BrokerError("Order id must not be empty.")
        self._request("DELETE", f"/v2/orders/{oid}")
        data = self._request("GET", f"/v2/orders/{oid}")
        return self._map_order(data)

    # ------------------------------------------------------------------
    # Mapping
    # ------------------------------------------------------------------

    def _map_order(self, data: Dict[str, Any]) -> BrokerOrder:
        """Map an Alpaca order payload onto :class:`~app.schemas.BrokerOrder`.

        Args:
            data: The decoded Alpaca order object.

        Returns:
            A populated :class:`~app.schemas.BrokerOrder` carrying the broker's
            paper flag and the standard disclaimer.
        """
        raw_status = str(data.get("status", "")).lower()
        status: BrokerOrderStatus = _STATUS_MAP.get(raw_status, "pending")
        return BrokerOrder(
            id=str(data.get("id", "")),
            symbol=str(data.get("symbol", "")).upper(),
            side=str(data.get("side", "buy")),  # type: ignore[arg-type]
            type="market",
            qty=_to_optional_float(data.get("qty")),
            notional=_to_optional_float(data.get("notional")),
            filled_qty=_to_float(data.get("filled_qty")),
            filled_avg_price=_to_float(data.get("filled_avg_price")),
            status=status,
            created_at=_iso_to_ms(data.get("created_at")),
            paper=self.is_paper,
            disclaimer=BROKER_DISCLAIMER,
        )


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _to_float(value: Any, default: float = 0.0) -> float:
    """Coerce an Alpaca numeric field (often a string) to a finite float."""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    import math

    return f if math.isfinite(f) else default


def _to_optional_float(value: Any) -> Optional[float]:
    """Coerce an Alpaca numeric field to a float, or ``None`` when absent."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_positive(value: Any, name: str) -> float:
    """Return ``value`` as a strictly-positive finite float or raise.

    Args:
        value: Candidate numeric.
        name: Field name for the error message.

    Returns:
        The positive float.

    Raises:
        BrokerError: If ``value`` is not a positive finite number.
    """
    import math

    try:
        f = float(value)
    except (TypeError, ValueError):
        raise BrokerError(f"Order {name} must be a number.") from None
    if not math.isfinite(f) or f <= 0.0:
        raise BrokerError(f"Order {name} must be greater than zero.")
    return f


def _iso_to_ms(value: Any) -> int:
    """Convert an ISO-8601 timestamp string to unix milliseconds.

    Args:
        value: An ISO-8601 datetime string (Alpaca's ``created_at``), or ``None``.

    Returns:
        Unix milliseconds; falls back to the current time on parse failure.
    """
    import time
    from datetime import datetime

    if not value:
        return int(time.time() * 1000)
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError):
        return int(time.time() * 1000)
