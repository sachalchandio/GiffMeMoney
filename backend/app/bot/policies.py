"""Preset auto-trader modes and their rotation policies (SIMULATION only).

This module defines the five user-selectable :class:`~app.schemas.BotMode`
presets and the numerical :class:`RotationPolicy` each one uses. A mode bundles:

    * a **portfolio objective** that turns the per-rebalance candidate set into
      base weights (delegated to :mod:`app.quant.portfolio` or a simple
      risk-parity / momentum rule), and
    * a **rotation policy** that *tilts* those base weights toward sleeves with
      strong trailing realized performance via a softmax over a bounded reward —
      MORE to recent winners, LESS to recent losers.

HONESTY / SAFETY (this is a finance tool). Everything here is a SIMULATION on
synthetic data: paper-traded, no real money, no live broker. The rotation is
deliberately **momentum / bandit** style and is hard-capped so a single sleeve
can never dominate; it is **never martingale** — the engine never increases a
losing sleeve's weight to "recover". There is no configuration that makes the
bot chase losses, and nothing here implies guaranteed profit.

The five modes:

    * **Conservative** — minimum-variance, slow rotation, 4 names (low risk).
    * **Balanced** — maximum-Sharpe, moderate rotation, 6 names.
    * **Aggressive** — momentum-weighted, fast rotation, 8 names (high risk).
    * **Adaptive Bandit** — softmax / Thompson-style allocation over the strategy
      sleeves by trailing reward (the most adaptive rotation).
    * **All-Weather** — risk-parity, rebalance-only (no momentum tilt).
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas import BotMode, BotModeId

__all__ = [
    "RotationPolicy",
    "ModePolicy",
    "BOT_MODES",
    "MODE_POLICIES",
    "get_mode",
    "get_policy",
]


@dataclass(frozen=True)
class RotationPolicy:
    """Numerical controls for the momentum / bandit weight tilt.

    The rotation takes each sleeve's bounded trailing reward ``r_i`` (a realized
    risk-adjusted return over ``lookback_days``, clipped to ``[-1, 1]``) and
    forms a softmax tilt ``exp(temperature * r_i)``. The base portfolio weights
    are multiplied by that tilt and renormalized, then each weight is clamped to
    ``[0, max_weight]`` and renormalized again so no single sleeve dominates.

    Because the tilt is monotone increasing in the reward, a winner's weight can
    only rise and a loser's can only fall relative to the base — momentum, never
    martingale. ``temperature = 0`` disables the tilt entirely (rebalance-only).

    Attributes:
        lookback_days: Trailing window (trading days) for the sleeve reward.
        temperature: Softmax sharpness; higher ⇒ more aggressive tilt toward
            winners. ``0`` disables rotation.
        max_weight: Hard cap on any single sleeve weight after the tilt, in
            ``(0, 1]`` (concentration guard).
        bandit: When ``True`` the reward also blends a Thompson-style optimistic
            exploration bonus (uncertainty-scaled) so under-sampled sleeves keep
            a chance — the Adaptive Bandit mode.
    """

    lookback_days: int
    temperature: float
    max_weight: float
    bandit: bool = False


@dataclass(frozen=True)
class ModePolicy:
    """A complete bot mode: its display metadata, objective and rotation policy.

    Attributes:
        mode: The wire :class:`~app.schemas.BotMode` (display metadata).
        objective: Internal objective key the engine dispatches on
            (``'min_volatility'`` / ``'max_sharpe'`` / ``'momentum'`` /
            ``'risk_parity'`` / ``'bandit'``).
        rotation: The :class:`RotationPolicy` controlling the weight tilt.
    """

    mode: BotMode
    objective: str
    rotation: RotationPolicy


# ---------------------------------------------------------------------------
# The five presets.
# ---------------------------------------------------------------------------

MODE_POLICIES: dict[BotModeId, ModePolicy] = {
    "conservative": ModePolicy(
        mode=BotMode(
            id="conservative",
            name="Conservative",
            summary=(
                "Minimum-variance sleeve held with slow rotation across a tight "
                "4-name book — prioritizes stability over reach."
            ),
            risk_level="low",
            objective="min_volatility",
            rotation="slow",
            max_names=4,
        ),
        objective="min_volatility",
        # Slow rotation: long lookback, gentle tilt, generous per-name cap (a
        # 4-name min-var book is inherently concentrated).
        rotation=RotationPolicy(lookback_days=126, temperature=1.0, max_weight=0.40),
    ),
    "balanced": ModePolicy(
        mode=BotMode(
            id="balanced",
            name="Balanced",
            summary=(
                "Maximum-Sharpe sleeve with moderate rotation across 6 names — a "
                "risk-adjusted middle ground."
            ),
            risk_level="moderate",
            objective="max_sharpe",
            rotation="moderate",
            max_names=6,
        ),
        objective="max_sharpe",
        rotation=RotationPolicy(lookback_days=63, temperature=2.0, max_weight=0.35),
    ),
    "aggressive": ModePolicy(
        mode=BotMode(
            id="aggressive",
            name="Aggressive",
            summary=(
                "Momentum-weighted across 8 names with fast rotation — leans hard "
                "into recent winners (and out of laggards)."
            ),
            risk_level="high",
            objective="momentum",
            rotation="fast",
            max_names=8,
        ),
        objective="momentum",
        # Fast rotation: short lookback, sharp tilt, lower per-name cap so the
        # book stays diversified even while chasing momentum.
        rotation=RotationPolicy(lookback_days=42, temperature=3.5, max_weight=0.30),
    ),
    "adaptive-bandit": ModePolicy(
        mode=BotMode(
            id="adaptive-bandit",
            name="Adaptive Bandit",
            summary=(
                "Softmax / Thompson-style allocation over the strategy sleeves by "
                "trailing reward — explores then exploits the best performers."
            ),
            risk_level="high",
            objective="bandit",
            rotation="bandit",
            max_names=6,
        ),
        objective="bandit",
        # Bandit rotation: medium lookback, sharp exploit tilt + an exploration
        # bonus for under-sampled sleeves.
        rotation=RotationPolicy(
            lookback_days=63, temperature=3.0, max_weight=0.35, bandit=True
        ),
    ),
    "all-weather": ModePolicy(
        mode=BotMode(
            id="all-weather",
            name="All-Weather",
            summary=(
                "Risk-parity across the candidate book, rebalance-only — equalizes "
                "risk contribution with no momentum tilt."
            ),
            risk_level="low",
            objective="risk_parity",
            rotation="none",
            max_names=6,
        ),
        objective="risk_parity",
        # Rebalance-only: temperature 0 disables the tilt entirely.
        rotation=RotationPolicy(lookback_days=126, temperature=0.0, max_weight=0.40),
    ),
}

#: The ordered list of wire :class:`~app.schemas.BotMode` presets (for the API).
BOT_MODES: list[BotMode] = [mp.mode for mp in MODE_POLICIES.values()]


def get_mode(mode_id: str) -> BotMode:
    """Return the :class:`~app.schemas.BotMode` for ``mode_id`` (default balanced).

    Args:
        mode_id: A :data:`~app.schemas.BotModeId` (case-insensitive). Unknown
            ids fall back to ``'balanced'`` so the bot always runs.

    Returns:
        The matching :class:`~app.schemas.BotMode`.
    """
    return get_policy(mode_id).mode


def get_policy(mode_id: str) -> ModePolicy:
    """Return the :class:`ModePolicy` for ``mode_id`` (default balanced).

    Args:
        mode_id: A :data:`~app.schemas.BotModeId` (case-insensitive). Unknown
            ids fall back to ``'balanced'``.

    Returns:
        The matching :class:`ModePolicy`.
    """
    key = str(mode_id or "").strip().lower()
    return MODE_POLICIES.get(key, MODE_POLICIES["balanced"])  # type: ignore[arg-type]
