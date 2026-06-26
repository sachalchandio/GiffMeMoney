"""Tests for the simulated auto-trader (paper-trading bot).

HONESTY / SAFETY (this is a finance tool). These tests pin the additive
auto-trader guarantees:

* the five preset :class:`~app.schemas.BotMode` ids exist with valid metadata,
  and rotation is never martingale (no ``"martingale"`` rotation option exists);
* a backtest returns **finite** metrics, a non-empty equity curve (bot vs
  benchmark, with per-bar drawdown + regime), a trade blotter and per-sleeve
  attribution that surfaces a single **best** and a single **worst** sleeve;
* the **anti-martingale** invariant holds on a crafted winner/loser case — the
  rotation never increases a losing sleeve's target weight; the loser's tilted
  weight is strictly below its base weight (winners up, losers DOWN);
* the engine **never raises** for degenerate / adversarial configs (zero amount,
  empty class filter, unknown mode, absurd risk params), always returning a
  finite result carrying the disclaimer;
* the mandatory simulation **disclaimer** is present on the engine result, the
  :class:`~app.schemas.BotRunResult` default, and every ``/api/bot`` route
  description, and there is nothing that implies guaranteed profit.

The suite reuses a single module-scoped :class:`~app.bot.engine.AutoTraderEngine`
whose shared analysis cache is warmed exactly once (the first backtest), so every
subsequent run is sub-second — no full-universe sweeps are repeated (anti-stall).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.bot.attribution import SleeveStat, build_attribution
from app.bot.engine import AutoTraderEngine
from app.bot.policies import BOT_MODES, MODE_POLICIES, get_policy
from app.schemas import (
    BOT_DISCLAIMER,
    BotConfig,
    BotMode,
    BotRunResult,
)

# The five preset mode ids the auto-trader must expose.
EXPECTED_MODE_IDS = {
    "conservative",
    "balanced",
    "aggressive",
    "adaptive-bandit",
    "all-weather",
}

VALID_RISK_LEVELS = {"low", "moderate", "high"}
# Rotation styles — note there is deliberately NO "martingale" option.
VALID_ROTATIONS = {"none", "slow", "moderate", "fast", "bandit"}


@pytest.fixture(scope="module")
def engine() -> AutoTraderEngine:
    """A single auto-trader engine reused across the module (cache warmed once)."""
    return AutoTraderEngine()


@pytest.fixture(scope="module")
def run_result(engine: AutoTraderEngine) -> BotRunResult:
    """One representative backtest with both winning and losing sleeves.

    The ``aggressive`` mode over the full (unfiltered) candidate universe holds a
    wide book, so the realized run reliably contains both a positive (best) and a
    negative (worst) sleeve — letting the best/worst attribution be asserted. This
    is the first backtest, so it warms the shared analysis cache for the module.
    """
    return engine.backtest(BotConfig(amount=10_000.0, mode="aggressive"))


def _all_finite(*values: float) -> bool:
    """True iff every value is a finite float."""
    return all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in values)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


def test_five_preset_modes_exist() -> None:
    """The five preset modes are registered with the expected ids."""
    assert {m.id for m in BOT_MODES} == EXPECTED_MODE_IDS
    assert set(MODE_POLICIES.keys()) == EXPECTED_MODE_IDS
    # Each policy's wire mode id matches its registry key.
    for mode_id, policy in MODE_POLICIES.items():
        assert policy.mode.id == mode_id


def test_mode_metadata_is_valid() -> None:
    """Every mode carries a valid risk level, rotation style and name count."""
    for mode in BOT_MODES:
        assert isinstance(mode, BotMode)
        assert mode.name
        assert mode.summary
        assert mode.risk_level in VALID_RISK_LEVELS
        assert mode.rotation in VALID_ROTATIONS
        assert mode.max_names >= 1


def test_no_martingale_rotation_option_anywhere() -> None:
    """No mode (or rotation policy) chases losses — there is no martingale knob."""
    for mode in BOT_MODES:
        assert "martingale" not in str(mode.rotation).lower()
    # The rotation softmax temperature is finite and non-negative (a negative
    # temperature would invert the tilt and reward losers — never allowed).
    for policy in MODE_POLICIES.values():
        assert policy.rotation.temperature >= 0.0
        assert math.isfinite(policy.rotation.temperature)
        assert 0.0 < policy.rotation.max_weight <= 1.0


# ---------------------------------------------------------------------------
# Backtest output shape: finite metrics + equity curve + attribution
# ---------------------------------------------------------------------------


def test_backtest_metrics_are_finite(run_result: BotRunResult) -> None:
    """Every realized metric on the run is a finite number."""
    m = run_result.metrics
    assert _all_finite(
        m.total_return_pct,
        m.cagr_pct,
        m.sharpe,
        m.sortino,
        m.max_drawdown_pct,
        m.win_rate_pct,
        m.vs_benchmark_pct,
        m.final_value,
    )
    # Sanity bounds: max drawdown is non-positive, win rate is a percentage.
    assert m.max_drawdown_pct <= 0.0 + 1e-9
    assert 0.0 <= m.win_rate_pct <= 100.0
    assert m.final_value >= 0.0


def test_backtest_equity_curve_is_finite_and_ordered(run_result: BotRunResult) -> None:
    """The equity curve is non-empty, time-ordered and finite throughout."""
    curve = run_result.equity_curve
    assert len(curve) >= 2
    ts = [p.t for p in curve]
    assert ts == sorted(ts)  # strictly chronological
    for p in curve:
        assert _all_finite(p.bot_value, p.benchmark_value, p.drawdown_pct)
        assert p.bot_value >= 0.0
        assert p.benchmark_value >= 0.0
        assert p.drawdown_pct <= 0.0 + 1e-9  # drawdown from peak is <= 0
        assert p.regime in {"bull", "bear", "neutral"}


def test_backtest_has_best_and_worst_sleeve(run_result: BotRunResult) -> None:
    """Attribution surfaces exactly one best and one worst sleeve."""
    attribution = run_result.attribution
    assert len(attribution) >= 2

    verdicts = [a.verdict for a in attribution]
    assert verdicts.count("best") == 1
    assert verdicts.count("worst") == 1

    best = next(a for a in attribution if a.verdict == "best")
    worst = next(a for a in attribution if a.verdict == "worst")
    # The best sleeve made money; the worst lost money.
    assert best.realized_pnl > 0.0
    assert worst.realized_pnl < 0.0
    # Attribution is sorted best (highest P&L) → worst (lowest P&L).
    pnls = [a.realized_pnl for a in attribution]
    assert pnls == sorted(pnls, reverse=True)
    # The result's best/worst pointers agree with the verdicts.
    assert run_result.best_strategy == best.key
    assert run_result.worst_strategy == worst.key
    # All attribution figures are finite.
    for a in attribution:
        assert _all_finite(a.realized_pnl, a.contribution_pct, a.win_rate)
        assert 0.0 <= a.win_rate <= 1.0


def test_backtest_records_trades_and_regime_timeline(run_result: BotRunResult) -> None:
    """The run logs simulated paper trades and a per-rebalance regime timeline."""
    assert len(run_result.trades) > 0
    for tr in run_result.trades:
        assert tr.side in {"buy", "sell"}
        assert tr.amount > 0.0
        assert _all_finite(tr.amount, tr.price)
    assert len(run_result.regime_timeline) > 0
    assert all(r in {"bull", "bear", "neutral"} for r in run_result.regime_timeline)


def test_all_modes_produce_finite_results(engine: AutoTraderEngine) -> None:
    """Each preset mode backtests to a finite result (cache already warm → fast)."""
    for mode_id in EXPECTED_MODE_IDS:
        res = engine.backtest(BotConfig(amount=10_000.0, mode=mode_id))  # type: ignore[arg-type]
        assert res.mode.id == mode_id
        m = res.metrics
        assert _all_finite(
            m.total_return_pct, m.cagr_pct, m.sharpe, m.sortino, m.final_value
        )
        assert len(res.equity_curve) >= 1
        assert res.disclaimer == BOT_DISCLAIMER


# ---------------------------------------------------------------------------
# Anti-martingale invariant (the core safety property) — crafted case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode_id", ["conservative", "balanced", "aggressive", "adaptive-bandit"]
)
def test_rotation_never_increases_a_losing_sleeve(
    engine: AutoTraderEngine, mode_id: str
) -> None:
    """Rotation tilts AWAY from a loser — its weight never rises above base.

    Crafted case: five sleeves over a trailing window. Four trend upward with
    decreasing strength; the fifth (index 4) trends DOWN. Each sleeve carries
    realistic noise so its trailing volatility (and thus the Sharpe-like reward)
    is live. Starting from equal base weights, a martingale would *raise* the
    loser's weight to "recover"; the engine must instead push it strictly BELOW
    its base weight (and it should be the smallest weight in the book).
    """
    rng = np.random.default_rng(7)
    n = 80
    t = n - 1
    drifts = [0.010, 0.007, 0.004, 0.002, -0.012]  # last sleeve is the loser
    cols = []
    for d in drifts:
        rets = d + rng.normal(0.0, 0.004, size=n)
        cols.append(100.0 * np.cumprod(1.0 + rets))
    prices = np.column_stack(cols)

    selected = [0, 1, 2, 3, 4]
    base = np.full(5, 0.2)
    policy = get_policy(mode_id)

    tilted = engine._rotate(selected, base, prices, t, sleeve_legs=None, policy=policy)

    loser = 4
    # The loser's weight is NOT increased after its loss (anti-martingale).
    assert tilted[loser] < base[loser], (
        f"{mode_id}: losing sleeve weight rose from {base[loser]} to "
        f"{tilted[loser]} — that would be martingale behaviour."
    )
    # The loser ends up as the smallest weight in the book, and the strongest
    # winner outweighs it (winners up, losers down).
    assert math.isclose(float(tilted[loser]), float(np.min(tilted)))
    assert tilted[0] > tilted[loser]
    # Weights remain a valid, finite distribution.
    assert math.isclose(float(np.sum(tilted)), 1.0, abs_tol=1e-9)
    assert np.all(tilted >= -1e-12)
    assert np.all(np.isfinite(tilted))


def test_rebalance_only_mode_applies_no_loss_chasing_tilt(
    engine: AutoTraderEngine,
) -> None:
    """The rebalance-only mode (All-Weather) never tilts toward a loser either."""
    rng = np.random.default_rng(11)
    n = 80
    t = n - 1
    drifts = [0.008, 0.005, 0.003, 0.001, -0.010]
    cols = [
        100.0 * np.cumprod(1.0 + (d + rng.normal(0.0, 0.004, size=n))) for d in drifts
    ]
    prices = np.column_stack(cols)
    base = np.full(5, 0.2)
    policy = get_policy("all-weather")  # temperature 0 → rebalance only

    tilted = engine._rotate([0, 1, 2, 3, 4], base, prices, t, None, policy)
    # No tilt: the loser keeps (does not exceed) its base weight.
    assert tilted[4] <= base[4] + 1e-9
    assert math.isclose(float(np.sum(tilted)), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# No look-ahead: per-rebalance selection is point-in-time
# ---------------------------------------------------------------------------


def test_pit_scores_use_only_data_up_to_t(engine: AutoTraderEngine) -> None:
    """``_pit_scores`` at bar t depends only on prices[:t+1] (no look-ahead).

    Mutating prices strictly AFTER bar ``t`` must not change the point-in-time
    score computed at ``t`` — the defining property of a no-look-ahead signal.
    """
    rng = np.random.default_rng(3)
    n = 120
    cols = [
        100.0 * np.cumprod(1.0 + (d + rng.normal(0.0, 0.005, size=n)))
        for d in (0.004, 0.001, -0.003)
    ]
    prices = np.column_stack(cols)
    t = 60

    before = engine._pit_scores(prices, t)

    # Corrupt every bar AFTER t with a wild spike; the as-of-t score is unchanged.
    future = prices.copy()
    future[t + 1 :] *= 5.0
    after = engine._pit_scores(future, t)

    assert np.allclose(before, after)
    assert np.all(np.isfinite(before))


def test_pit_ranking_differs_from_full_history_ranking(
    engine: AutoTraderEngine,
) -> None:
    """The point-in-time ranking is genuinely different from the full-history one.

    A crafted set where the eventual full-history winner is FALLING through the
    early window (it only rallies AFTER bar ``t``) proves the new selection ranks
    by what was known at the early rebalance — not by the future-peeking
    full-history order the old engine froze and reused at every past rebalance.
    """
    rng = np.random.default_rng(5)
    n = 120
    t = 50  # an early rebalance, strictly before A's late rally

    def _series(daily: np.ndarray) -> np.ndarray:
        return 100.0 * np.cumprod(1.0 + daily)

    # Sleeve A: DOWN through the early window (negative drift), then a violent late
    # rally so it wins the full-history race — but is the worst as-of t.
    a_daily = np.concatenate(
        [
            -0.004 + rng.normal(0.0, 0.003, size=t + 1),  # falling up to & incl. t
            0.05 + rng.normal(0.0, 0.003, size=n - t - 1),  # explosive after t
        ]
    )
    # Sleeve B: steady, strong UP climb through the early window (early leader).
    b_daily = 0.006 + rng.normal(0.0, 0.003, size=n)
    # Sleeve C: gentle positive drift throughout (a stable middle).
    c_daily = 0.002 + rng.normal(0.0, 0.003, size=n)
    prices = np.column_stack([_series(a_daily), _series(b_daily), _series(c_daily)])

    # Point-in-time order as-of the early rebalance t (uses only prices[:t+1]).
    pit = engine._pit_scores(prices, t)
    pit_order = sorted(range(3), key=lambda j: pit[j], reverse=True)

    # The OLD engine ranked by a full-history score; the full-window trailing
    # return is a faithful stand-in — A wins it purely because of its late rally.
    full_hist = prices[-1] / prices[0] - 1.0
    full_order = sorted(range(3), key=lambda j: full_hist[j], reverse=True)

    assert full_order[0] == 0  # A is the full-history (look-ahead) winner
    # As-of t, A has been FALLING, so it must rank last point-in-time — the two
    # rankings genuinely disagree, which is exactly the look-ahead being removed.
    assert pit_order[0] != 0
    assert pit_order[-1] == 0
    assert pit_order != full_order


def test_attribution_builder_marks_best_and_worst() -> None:
    """The attribution builder flags the top winner best and bottom loser worst."""
    stats = [
        SleeveStat(key="WIN", realized_pnl=120.0, trades=4, wins=3, legs=4),
        SleeveStat(key="MID", realized_pnl=5.0, trades=2, wins=1, legs=2),
        SleeveStat(key="LOSE", realized_pnl=-80.0, trades=3, wins=0, legs=3),
    ]
    rows = build_attribution(stats)
    assert [r.key for r in rows] == ["WIN", "MID", "LOSE"]  # best → worst
    assert rows[0].verdict == "best"
    assert rows[-1].verdict == "worst"
    assert rows[1].verdict == "neutral"
    # Contribution shares are finite and the winner's is positive.
    assert _all_finite(*[r.contribution_pct for r in rows])
    assert rows[0].contribution_pct > 0.0
    assert rows[-1].contribution_pct < 0.0


# ---------------------------------------------------------------------------
# The engine never raises (defensive) — degenerate / adversarial configs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config",
    [
        BotConfig(amount=0.0, mode="balanced"),  # nothing to invest
        BotConfig(amount=-100.0, mode="balanced"),  # negative capital
        BotConfig(amount=10_000.0, mode="balanced", asset_classes=[]),  # empty filter
        BotConfig(amount=10_000.0, mode="balanced", rebalance_days=0),  # bad cadence
        BotConfig(
            amount=10_000.0, mode="balanced", stop_loss_pct=-5.0, max_drawdown_pct=999.0
        ),  # absurd risk params
        BotConfig(amount=float("nan"), mode="balanced"),  # non-finite amount
    ],
)
def test_engine_never_raises_on_degenerate_config(
    engine: AutoTraderEngine, config: BotConfig
) -> None:
    """A degenerate / adversarial config degrades to a finite result, never raises."""
    res = engine.backtest(config)
    assert isinstance(res, BotRunResult)
    m = res.metrics
    assert _all_finite(m.total_return_pct, m.cagr_pct, m.sharpe, m.final_value)
    assert len(res.equity_curve) >= 1
    assert res.disclaimer == BOT_DISCLAIMER


def test_get_policy_falls_back_for_unknown_mode() -> None:
    """An unknown mode id resolves to the balanced policy (defensive default)."""
    assert get_policy("totally-made-up").mode.id == "balanced"
    assert get_policy("").mode.id == "balanced"


def test_engine_handles_forced_unknown_mode_without_raising(
    engine: AutoTraderEngine,
) -> None:
    """Even a config whose mode bypassed validation degrades gracefully.

    ``BotConfig.mode`` is a strict ``Literal`` so the API/Pydantic reject an
    unknown id up front (a 422). This pins the engine's own defensive contract:
    given a config carrying a bogus mode (constructed without validation via
    ``model_construct``), it still resolves a fallback policy and returns a finite
    result rather than raising.
    """
    bogus = BotConfig.model_construct(
        amount=10_000.0,
        mode="totally-made-up",  # type: ignore[arg-type]
        asset_classes=["equity"],
        rebalance_days=21,
        stop_loss_pct=25.0,
        max_drawdown_pct=35.0,
    )
    res = engine.backtest(bogus)
    assert isinstance(res, BotRunResult)
    assert res.disclaimer == BOT_DISCLAIMER


# ---------------------------------------------------------------------------
# Disclaimer presence (engine, schema default, API routes)
# ---------------------------------------------------------------------------


def test_disclaimer_text_is_honest_and_specific() -> None:
    """The disclaimer states it is simulated, on synthetic data, with no real funds."""
    text = BOT_DISCLAIMER.lower()
    assert "simulated" in text
    assert "synthetic data" in text
    assert "not financial advice" in text
    assert "no real funds" in text
    # And it never implies guaranteed profit.
    assert "guarantee" not in text
    assert "guaranteed profit" not in text


def test_backtest_marks_results_as_synthetic(run_result: BotRunResult) -> None:
    """Every run honestly flags itself as synthetic data with no implied target."""
    assert run_result.synthetic_data is True
    # No target was requested, so there is no infeasible-target warning.
    assert run_result.target_warning is None


def test_run_result_carries_disclaimer_by_default() -> None:
    """A bare BotRunResult defaults its disclaimer to the mandatory text."""
    from app.schemas import BotMetrics

    built = BotRunResult(
        mode=MODE_POLICIES["balanced"].mode,
        config=BotConfig(amount=10.0, mode="balanced"),
        metrics=BotMetrics(
            total_return_pct=0.0,
            cagr_pct=0.0,
            sharpe=0.0,
            sortino=0.0,
            max_drawdown_pct=0.0,
            win_rate_pct=0.0,
            vs_benchmark_pct=0.0,
            final_value=10.0,
        ),
    )
    assert built.disclaimer == BOT_DISCLAIMER


def test_bot_modes_route_exposes_disclaimer(client: TestClient) -> None:
    """GET /api/bot/modes lists the five modes; the route description discloses sim."""
    resp = client.get("/api/bot/modes")
    assert resp.status_code == 200
    body = resp.json()
    assert {m["id"] for m in body} == EXPECTED_MODE_IDS
    for m in body:
        assert m["riskLevel"] in VALID_RISK_LEVELS
        assert m["rotation"] in VALID_ROTATIONS


def test_bot_route_descriptions_carry_disclaimer(client: TestClient) -> None:
    """The OpenAPI descriptions of every /api/bot route carry the disclaimer."""
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    bot_paths = {p: v for p, v in paths.items() if p.startswith("/api/bot")}
    assert bot_paths, "no /api/bot routes registered"
    for path, methods in bot_paths.items():
        for method, op in methods.items():
            desc = op.get("description", "")
            assert BOT_DISCLAIMER in desc, (
                f"{method.upper()} {path} description is missing the disclaimer"
            )


def test_bot_backtest_route_returns_disclaimer(client: TestClient) -> None:
    """POST /api/bot/backtest returns a finite result carrying the disclaimer."""
    resp = client.post(
        "/api/bot/backtest",
        json={"config": {"amount": 10000, "mode": "balanced", "assetClasses": ["equity"]}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["disclaimer"] == BOT_DISCLAIMER
    assert len(body["equityCurve"]) >= 1
    m = body["metrics"]
    assert _all_finite(
        m["totalReturnPct"], m["cagrPct"], m["sharpe"], m["finalValue"]
    )


def test_bot_backtest_route_rejects_unknown_mode(client: TestClient) -> None:
    """POST /api/bot/backtest rejects an unknown mode id with a client error.

    ``BotConfig.mode`` is a strict ``Literal``, so an unknown id is rejected by
    request validation (422) before the handler's defensive 400 guard runs. Either
    way the run never proceeds — the response is a 4xx client error, never a 5xx.
    """
    resp = client.post(
        "/api/bot/backtest",
        json={"config": {"amount": 10000, "mode": "nope-not-a-mode"}},
    )
    assert resp.status_code in (400, 422)
    assert 400 <= resp.status_code < 500
