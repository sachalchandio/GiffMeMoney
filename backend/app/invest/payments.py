"""Payment provider abstraction for the simulated brokerage.

Money-handling stance (critical): this is a **sandbox / paper** wallet. No real
money moves and no real payment network is contacted. Cards are *validated*
(Luhn check, brand detection, future-expiry, amount bounds) but never charged,
and only **masked** data (brand + last4 + an opaque token) is ever retained.
Raw PAN/CVC are validated in memory and discarded immediately — never persisted
and never logged.

The :class:`PaymentProvider` ABC isolates this so a real PSP (Stripe, Plaid, …)
can drop in later behind the same contract, exactly like the market-data
provider. The only implementation shipped is :class:`SimulatedPaymentProvider`,
selected by :func:`get_payment_provider`.

Public helpers:
    * :func:`luhn_valid` — Luhn (mod-10) checksum validation.
    * :func:`brand_for` — card-network brand detection from the PAN prefix.
    * :func:`mask` — render a masked ``'•••• 1234'`` display string.
    * :func:`tokenize` — turn a :class:`~app.schemas.CardIn` into a masked
      :class:`~app.schemas.SavedCard` (drops PAN/CVC).
"""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod

from app.schemas import CardIn, SavedCard, Transaction

__all__ = [
    "luhn_valid",
    "brand_for",
    "mask",
    "tokenize",
    "PaymentProvider",
    "SimulatedPaymentProvider",
    "get_payment_provider",
    "MAX_CHARGE",
]

#: Maximum amount a single simulated charge / payout may move (sandbox guard).
MAX_CHARGE: float = 10_000.0


def _digits(number: str) -> str:
    """Strip everything except digits from a card number string.

    Args:
        number: Raw card number, possibly containing spaces or dashes.

    Returns:
        The digit-only string (may be empty).
    """
    return "".join(ch for ch in (number or "") if ch.isdigit())


def luhn_valid(number: str) -> bool:
    """Validate a card number with the Luhn (mod-10) checksum.

    The Luhn algorithm doubles every second digit from the right, subtracting 9
    from any product over 9, sums all digits, and accepts iff the total is a
    multiple of 10.

    Args:
        number: Card number (spaces / dashes allowed; non-digits ignored).

    Returns:
        ``True`` if the number is non-empty (13-19 digits) and passes the Luhn
        checksum, else ``False``.
    """
    digits = _digits(number)
    if len(digits) < 13 or len(digits) > 19:
        return False
    total = 0
    # Process right-to-left; double every second digit.
    for index, ch in enumerate(reversed(digits)):
        value = int(ch)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def brand_for(number: str) -> str:
    """Detect the card network brand from the number's leading digits.

    Recognizes the common test ranges:
        * ``visa`` — starts with ``4``.
        * ``mastercard`` — ``51``-``55`` or the ``2221``-``2720`` range.
        * ``amex`` — ``34`` or ``37``.
        * ``discover`` — ``6011``, ``65``, or ``644``-``649``.

    Args:
        number: Card number (non-digits ignored).

    Returns:
        One of ``'visa'``, ``'mastercard'``, ``'amex'``, ``'discover'``, or
        ``'unknown'``.
    """
    digits = _digits(number)
    if not digits:
        return "unknown"
    if digits[0] == "4":
        return "visa"
    if digits[:2] in {"34", "37"}:
        return "amex"
    two = int(digits[:2]) if len(digits) >= 2 else -1
    four = int(digits[:4]) if len(digits) >= 4 else -1
    if 51 <= two <= 55 or 2221 <= four <= 2720:
        return "mastercard"
    three = int(digits[:3]) if len(digits) >= 3 else -1
    if digits[:4] == "6011" or two == 65 or 644 <= three <= 649:
        return "discover"
    return "unknown"


def mask(number: str) -> str:
    """Render a masked display string for a card number.

    Args:
        number: Card number (non-digits ignored).

    Returns:
        ``'•••• 1234'`` using the last four digits, or ``'•••• ••••'`` when
        fewer than four digits are present.
    """
    digits = _digits(number)
    if len(digits) < 4:
        return "•••• ••••"
    return f"•••• {digits[-4:]}"


def _expiry_is_future(exp_month: int, exp_year: int) -> bool:
    """Return whether a card's expiry month is still in the future.

    A card expires at the end of its expiry month, so it is valid through the
    last day of ``exp_month``/``exp_year``. Comparison uses the current local
    year/month from the system clock.

    Args:
        exp_month: Expiry month in ``[1, 12]``.
        exp_year: Four-digit expiry year.

    Returns:
        ``True`` if (year, month) is at or after the current (year, month) and
        the month is in range, else ``False``.
    """
    if not (1 <= exp_month <= 12):
        return False
    now = time.localtime()
    cur_year, cur_month = now.tm_year, now.tm_mon
    if exp_year > cur_year:
        return True
    if exp_year < cur_year:
        return False
    return exp_month >= cur_month


def tokenize(card: CardIn) -> SavedCard:
    """Turn a raw :class:`~app.schemas.CardIn` into a masked saved card.

    Only the brand, last four digits, expiry and holder are retained, keyed by
    a freshly minted opaque token id. The PAN and CVC are read here solely to
    derive ``last4``/``brand`` and are then discarded — they are never returned,
    stored, or logged.

    Args:
        card: The validated card input.

    Returns:
        A :class:`~app.schemas.SavedCard` containing no sensitive data.
    """
    digits = _digits(card.number)
    last4 = digits[-4:] if len(digits) >= 4 else digits
    return SavedCard(
        id=str(uuid.uuid4()),
        brand=brand_for(card.number),
        last4=last4,
        exp_month=card.exp_month,
        exp_year=card.exp_year,
        holder=card.holder,
    )


def _now_ms() -> int:
    """Return the current unix time in milliseconds."""
    return int(time.time() * 1000)


class PaymentProvider(ABC):
    """Abstract payment interface isolating funding from the rest of the app.

    A real PSP adapter can implement this same contract later; until then
    :class:`SimulatedPaymentProvider` fulfils it without moving real money.
    """

    @abstractmethod
    def charge(self, card: CardIn, amount: float, account: str) -> Transaction:
        """Charge ``amount`` to ``card`` for ``account`` (a deposit).

        Args:
            card: The card to validate/charge (never stored raw).
            amount: Positive dollar amount to charge.
            account: The account id being funded.

        Returns:
            A completed ``deposit`` :class:`~app.schemas.Transaction`.

        Raises:
            ValueError: If the card or amount is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def payout(
        self, amount: float, account: str, destination: str | None
    ) -> Transaction:
        """Pay ``amount`` out of ``account`` to ``destination`` (a withdrawal).

        Args:
            amount: Positive dollar amount to pay out.
            account: The account id being debited.
            destination: Optional free-text payout destination label.

        Returns:
            A completed ``withdrawal`` :class:`~app.schemas.Transaction`.

        Raises:
            ValueError: If the amount is invalid.
        """
        raise NotImplementedError


class SimulatedPaymentProvider(PaymentProvider):
    """A sandbox payment provider that never contacts a real network.

    Charges always *succeed* for a valid card and a sane amount, and always
    *raise* a clear :class:`ValueError` otherwise. Validation rules for a
    charge:

        * the card number must pass the Luhn checksum;
        * the expiry month/year must not be in the past;
        * the amount must be ``> 0`` and ``<= MAX_CHARGE`` (sandbox cap).

    Payouts validate only the amount bounds (the cash-vs-balance check lives in
    the wallet service, which knows the account balance).
    """

    def _validate_amount(self, amount: float) -> None:
        """Validate a money amount is positive, finite and within the cap.

        Args:
            amount: The dollar amount to validate.

        Raises:
            ValueError: If the amount is not a finite number, ``<= 0``, or
                ``> MAX_CHARGE``.
        """
        try:
            value = float(amount)
        except (TypeError, ValueError):
            raise ValueError("Amount must be a number.") from None
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Amount must be a finite number.")
        if value <= 0:
            raise ValueError("Amount must be greater than zero.")
        if value > MAX_CHARGE:
            raise ValueError(
                f"Amount exceeds the sandbox limit of ${MAX_CHARGE:,.2f}."
            )

    def charge(self, card: CardIn, amount: float, account: str) -> Transaction:
        """Validate the card + amount and return a completed deposit txn.

        See :meth:`PaymentProvider.charge`. No money actually moves.

        Raises:
            ValueError: On a failed Luhn check, an expired card, or an invalid
                amount (with a clear, user-facing message).
        """
        self._validate_amount(amount)
        if not luhn_valid(card.number):
            raise ValueError("Invalid card number (failed Luhn check).")
        if not _expiry_is_future(card.exp_month, card.exp_year):
            raise ValueError("Card is expired or has an invalid expiry date.")
        txn_id = str(uuid.uuid4())
        return Transaction(
            id=txn_id,
            type="deposit",
            amount=round(float(amount), 2),
            symbol=None,
            status="completed",
            created_at=_now_ms(),
            ref=f"dep_{txn_id[:8]}",
            note=f"Deposit via {brand_for(card.number)} {mask(card.number)}",
        )

    def payout(
        self, amount: float, account: str, destination: str | None
    ) -> Transaction:
        """Validate the amount and return a completed withdrawal txn.

        See :meth:`PaymentProvider.payout`. The caller (wallet service) is
        responsible for rejecting amounts greater than the available cash; this
        method only enforces the positive / within-cap amount bounds.

        Raises:
            ValueError: If the amount is invalid.
        """
        self._validate_amount(amount)
        dest = (destination or "").strip() or "bank account"
        txn_id = str(uuid.uuid4())
        return Transaction(
            id=txn_id,
            type="withdrawal",
            amount=round(float(amount), 2),
            symbol=None,
            status="completed",
            created_at=_now_ms(),
            ref=f"wd_{txn_id[:8]}",
            note=f"Withdrawal to {dest}",
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_PROVIDER_LOCK = threading.Lock()
_PROVIDER_INSTANCE: PaymentProvider | None = None


def get_payment_provider() -> PaymentProvider:
    """Return the process-wide :class:`PaymentProvider` singleton.

    Currently always a :class:`SimulatedPaymentProvider`; a real PSP can be
    selected here from configuration later without changing call sites.

    Returns:
        The shared :class:`PaymentProvider` instance.
    """
    global _PROVIDER_INSTANCE
    if _PROVIDER_INSTANCE is None:
        with _PROVIDER_LOCK:
            if _PROVIDER_INSTANCE is None:
                _PROVIDER_INSTANCE = SimulatedPaymentProvider()
    return _PROVIDER_INSTANCE
