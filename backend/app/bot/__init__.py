"""Simulated auto-trader (paper-trading bot) package for GiffMeMoney.

HONESTY / SAFETY (this is a finance tool). Everything in this package is a
**SIMULATION on synthetic data**: the bot paper-trades a starting cash balance
over a deterministic historical window — no real money moves and no live broker
is ever contacted. Rotation is **momentum / bandit** style (allocate MORE to
recent winners, LESS to losers) and is hard-capped; the engine **never
martingales** — it never increases a losing sleeve's weight to recover. Every
result carries :data:`~app.schemas.BOT_DISCLAIMER` and nothing here implies
guaranteed profit.

Public surface:

    * :data:`~app.bot.policies.BOT_MODES` — the five preset bot modes.
    * :class:`~app.bot.engine.AutoTraderEngine` — runs a backtest from a
      :class:`~app.schemas.BotConfig` into a :class:`~app.schemas.BotRunResult`.
    * :func:`~app.bot.attribution.build_attribution` — rank sleeves best→worst.
"""

from __future__ import annotations

from app.bot.attribution import SleeveStat, build_attribution
from app.bot.engine import AutoTraderEngine
from app.bot.policies import (
    BOT_MODES,
    MODE_POLICIES,
    ModePolicy,
    RotationPolicy,
    get_mode,
    get_policy,
)

__all__ = [
    "AutoTraderEngine",
    "BOT_MODES",
    "MODE_POLICIES",
    "ModePolicy",
    "RotationPolicy",
    "SleeveStat",
    "build_attribution",
    "get_mode",
    "get_policy",
]
