/**
 * AutoTraderPage — the simulated auto-trader (paper-trading bot) experience.
 *
 * HONESTY / SAFETY (this is a finance tool). Everything on this page is a
 * SIMULATION on synthetic data: the bot paper-trades a starting balance over a
 * deterministic historical window — no real money moves and no live broker is
 * ever contacted. Rotation is momentum / bandit style (allocate MORE to recent
 * winners, LESS to losers) and is hard-capped; the engine never martingales. A
 * prominent disclaimer banner makes this explicit and nothing here implies
 * guaranteed profit.
 *
 * Wiring: a mode picker (the five presets, each with a risk badge), a starting
 * amount input (default 20), asset-class toggles, two risk sliders (per-sleeve
 * stop-loss + portfolio max-drawdown circuit-breaker), a Run button →
 * `runBotBacktest`, and a "Compare modes" action → `compareBotModes`. Results
 * render a bot-vs-benchmark equity curve, realized metrics, per-sleeve
 * attribution and a regime timeline. Loading shows skeletons. Strict TS, no any.
 */

import { useMemo, useState } from 'react';
import {
  AlertTriangle,
  Bot,
  GitCompareArrows,
  Layers,
  Play,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge, type BadgeTone } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Skeleton } from '@/components/ui/Skeleton';
import { StatCard } from '@/components/ui/StatCard';
import { ChartEmpty } from '@/components/charts/ChartEmpty';
import { ChartTooltip, type TooltipRow } from '@/components/charts/ChartTooltip';
import { useBotModes, useBotRun, useBotCompare } from '@/hooks/useBot';
import { useChartTokens } from '@/theme/tokens';
import type {
  AssetClass,
  BotConfig,
  BotMode,
  BotRiskLevel,
  BotRunResult,
  SleeveAttribution,
} from '@/lib/types';
import { BOT_DISCLAIMER } from '@/lib/types';
import { formatCurrency, formatDateTime, formatPct, formatRatio } from '@/lib/format';
import { assetClassLabel, changeTextColor, cn } from '@/lib/utils';

/* ------------------------------------------------------------------ */
/* Static option metadata                                              */
/* ------------------------------------------------------------------ */

const ASSET_CLASSES: readonly AssetClass[] = ['equity', 'crypto', 'etf'] as const;

/** Map a mode's risk level to a badge tone. */
function riskTone(level: BotRiskLevel): BadgeTone {
  switch (level) {
    case 'low':
      return 'success';
    case 'moderate':
      return 'warning';
    case 'high':
      return 'danger';
    default:
      return 'neutral';
  }
}

/** Title-case a market regime label for display. */
function regimeLabel(regime: string): string {
  return regime.charAt(0).toUpperCase() + regime.slice(1);
}

/** Badge tone for a market regime. */
function regimeTone(regime: string): BadgeTone {
  if (regime === 'bull') return 'success';
  if (regime === 'bear') return 'danger';
  return 'neutral';
}

/* ------------------------------------------------------------------ */
/* Sub-components                                                      */
/* ------------------------------------------------------------------ */

/** A selectable preset-mode tile with its risk badge + rotation summary. */
function ModeTile({
  mode,
  selected,
  onSelect,
}: {
  mode: BotMode;
  selected: boolean;
  onSelect: () => void;
}): JSX.Element {
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={selected}
      className={cn(
        'flex h-full flex-col gap-2 rounded-xl border p-3 text-left transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
        selected
          ? 'border-primary bg-primary/8 shadow-soft'
          : 'border-border bg-surface hover:border-primary/40 hover:bg-surface-2/50',
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-semibold tracking-tight text-text">{mode.name}</span>
        <Badge tone={riskTone(mode.riskLevel)} variant="soft" size="sm">
          {mode.riskLevel} risk
        </Badge>
      </div>
      <p className="line-clamp-3 text-[11px] leading-relaxed text-muted">{mode.summary}</p>
      <div className="mt-auto flex flex-wrap items-center gap-1 pt-1">
        <Badge tone="neutral" variant="outline" size="sm">
          {mode.objective.replace(/_/g, ' ')}
        </Badge>
        <Badge tone="neutral" variant="outline" size="sm">
          {mode.rotation} rotation
        </Badge>
        <Badge tone="neutral" variant="outline" size="sm">
          {mode.maxNames} names
        </Badge>
      </div>
    </button>
  );
}

/** A labelled range slider with a live percent readout. */
function RiskSlider({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}): JSX.Element {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-text">{label}</span>
        <span className="text-xs font-semibold tnum text-primary">{formatPct(value, { digits: 0 })}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-surface-2 accent-primary"
        aria-label={label}
      />
      <p className="text-[11px] leading-snug text-muted">{hint}</p>
    </div>
  );
}

/** The bot-vs-benchmark equity curve (growth of the starting amount). */
function BotEquityChart({ result, height = 300 }: { result: BotRunResult; height?: number }): JSX.Element {
  const tokens = useChartTokens();
  const rows = result.equityCurve.map((p) => ({
    t: p.t,
    bot: p.botValue,
    benchmark: p.benchmarkValue,
  }));

  if (rows.length === 0) {
    return <ChartEmpty height={height} label="No equity curve available" />;
  }

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -6 }}>
          <defs>
            <linearGradient id="bot-equity-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={tokens.primary} stopOpacity={0.22} />
              <stop offset="100%" stopColor={tokens.primary} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={tokens.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="t"
            tickFormatter={(v: number) => formatDateTime(v)}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: tokens.grid }}
            minTickGap={56}
          />
          <YAxis
            tickFormatter={(v: number) => formatCurrency(v, { maximumFractionDigits: 0, minimumFractionDigits: 0 })}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={56}
            domain={['auto', 'auto']}
          />
          <RechartsTooltip
            cursor={{ stroke: tokens.muted, strokeDasharray: '3 3' }}
            content={({ active, payload, label }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0]?.payload as { bot: number; benchmark: number } | undefined;
              if (!row) return null;
              const rowsOut: TooltipRow[] = [
                { label: 'Bot', value: formatCurrency(row.bot), color: tokens.primary },
                { label: 'Buy & Hold', value: formatCurrency(row.benchmark), color: tokens.muted },
              ];
              return <ChartTooltip title={formatDateTime(Number(label))} rows={rowsOut} />;
            }}
          />
          <Legend
            verticalAlign="top"
            height={24}
            iconType="plainline"
            wrapperStyle={{ fontSize: 11, color: tokens.muted }}
          />
          <Area
            type="monotone"
            name="Bot (simulated)"
            dataKey="bot"
            stroke={tokens.primary}
            strokeWidth={2}
            fill="url(#bot-equity-fill)"
            isAnimationActive={false}
            dot={false}
          />
          <Line
            type="monotone"
            name="Buy & Hold"
            dataKey="benchmark"
            stroke={tokens.muted}
            strokeWidth={1.5}
            strokeDasharray="5 4"
            isAnimationActive={false}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Per-sleeve realized attribution (best → worst), with a verdict chip. */
function AttributionRow({ row }: { row: SleeveAttribution }): JSX.Element {
  const tone: BadgeTone = row.verdict === 'best' ? 'success' : row.verdict === 'worst' ? 'danger' : 'neutral';
  return (
    <li className="flex items-center gap-3 rounded-xl border border-border bg-surface px-3 py-2">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-semibold tracking-tight text-text">{row.key}</span>
          {row.verdict !== 'neutral' && (
            <Badge tone={tone} variant="soft" size="sm">
              {row.verdict}
            </Badge>
          )}
        </div>
        <span className="text-[11px] text-muted">
          {row.trades} trades · {formatPct(row.winRate * 100, { digits: 0 })} win rate
        </span>
      </div>
      <span className={cn('w-20 shrink-0 text-right text-xs font-medium tnum', changeTextColor(row.realizedPnl))}>
        {formatCurrency(row.realizedPnl)}
      </span>
      <span className={cn('w-14 shrink-0 text-right text-xs tnum', changeTextColor(row.contributionPct))}>
        {formatPct(row.contributionPct, { digits: 0, sign: true })}
      </span>
    </li>
  );
}

/** The full result panel for one simulated run. */
function RunResultPanel({ result }: { result: BotRunResult }): JSX.Element {
  const m = result.metrics;
  return (
    <div className="flex flex-col gap-4">
      {/* Headline metrics */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          label="Final value"
          value={formatCurrency(m.finalValue)}
          deltaPct={m.totalReturnPct}
          icon={<TrendingUp className="h-4 w-4" />}
          hint={`from ${formatCurrency(result.config.amount)} start`}
        />
        <StatCard
          label="CAGR"
          value={formatPct(m.cagrPct)}
          hint="annualized (simulated)"
        />
        <StatCard
          label="Sharpe"
          value={formatRatio(m.sharpe)}
          hint={`Sortino ${formatRatio(m.sortino)}`}
        />
        <StatCard
          label="vs Buy & Hold"
          value={formatPct(m.vsBenchmarkPct, { sign: true })}
          deltaLabel={`Max DD ${formatPct(m.maxDrawdownPct)}`}
          hint={`Win rate ${formatPct(m.winRatePct, { digits: 0 })}`}
        />
      </div>

      {/* Equity curve */}
      <Card className="flex flex-col gap-3">
        <CardHeader>
          <CardTitle icon={<TrendingUp className="h-4 w-4" />}>Bot vs Buy &amp; Hold</CardTitle>
          <Badge tone="neutral" variant="soft" size="sm">
            Simulated
          </Badge>
        </CardHeader>
        <BotEquityChart result={result} />
      </Card>

      {/* Attribution + regime timeline */}
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 lg:gap-4">
        <Card className="flex flex-col gap-3">
          <CardHeader>
            <CardTitle icon={<Layers className="h-4 w-4" />}>Sleeve attribution</CardTitle>
            {result.attribution.length > 0 && (
              <span className="text-[11px] text-muted">{result.attribution.length}</span>
            )}
          </CardHeader>
          {result.attribution.length === 0 ? (
            <div className="rounded-xl border border-dashed border-border py-8 text-center text-xs text-muted">
              No sleeve attribution for this run.
            </div>
          ) : (
            <ul className="flex max-h-[360px] flex-col gap-2 overflow-y-auto pr-0.5">
              {result.attribution.map((row) => (
                <AttributionRow key={row.key} row={row} />
              ))}
            </ul>
          )}
          <p className="text-[11px] leading-snug text-muted">
            Rotation tilts toward recent winners (momentum / bandit) and away from losers — never increasing a
            losing sleeve to chase losses.
          </p>
        </Card>

        <Card className="flex flex-col gap-3">
          <CardHeader>
            <CardTitle icon={<GitCompareArrows className="h-4 w-4" />}>Regime timeline</CardTitle>
          </CardHeader>
          {result.regimeTimeline.length === 0 ? (
            <div className="rounded-xl border border-dashed border-border py-8 text-center text-xs text-muted">
              No regime timeline for this run.
            </div>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {result.regimeTimeline.map((regime, i) => (
                <Badge key={`${regime}-${i}`} tone={regimeTone(regime)} variant="soft" size="sm">
                  {regimeLabel(regime)}
                </Badge>
              ))}
            </div>
          )}
          <div className="mt-1 grid grid-cols-2 gap-2 text-xs">
            <div className="rounded-lg bg-surface-2/60 px-3 py-2">
              <span className="block text-[11px] text-muted">Best sleeve</span>
              <span className="font-semibold text-success">{result.bestStrategy ?? '—'}</span>
            </div>
            <div className="rounded-lg bg-surface-2/60 px-3 py-2">
              <span className="block text-[11px] text-muted">Worst sleeve</span>
              <span className="font-semibold text-danger">{result.worstStrategy ?? '—'}</span>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

/** One compact column in the "Compare modes" results grid. */
function CompareColumn({ result }: { result: BotRunResult }): JSX.Element {
  const m = result.metrics;
  return (
    <Card className="flex flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-semibold tracking-tight text-text">{result.mode.name}</span>
        <Badge tone={riskTone(result.mode.riskLevel)} variant="soft" size="sm">
          {result.mode.riskLevel}
        </Badge>
      </div>
      <BotEquityChart result={result} height={160} />
      <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs">
        <CompareStat label="Final" value={formatCurrency(m.finalValue)} />
        <CompareStat label="Total" value={formatPct(m.totalReturnPct, { sign: true })} tone={m.totalReturnPct} />
        <CompareStat label="Sharpe" value={formatRatio(m.sharpe)} />
        <CompareStat label="Max DD" value={formatPct(m.maxDrawdownPct)} tone={m.maxDrawdownPct} />
        <CompareStat label="vs B&H" value={formatPct(m.vsBenchmarkPct, { sign: true })} tone={m.vsBenchmarkPct} />
        <CompareStat label="Win" value={formatPct(m.winRatePct, { digits: 0 })} />
      </dl>
    </Card>
  );
}

function CompareStat({ label, value, tone }: { label: string; value: string; tone?: number }): JSX.Element {
  return (
    <div className="flex items-center justify-between gap-2">
      <dt className="text-muted">{label}</dt>
      <dd className={cn('font-medium tnum', tone !== undefined ? changeTextColor(tone) : 'text-text')}>{value}</dd>
    </div>
  );
}

/** Skeleton shown while a run/compare is in flight. */
function ResultSkeleton(): JSX.Element {
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-24 w-full" />
        ))}
      </div>
      <Skeleton className="h-[300px] w-full" />
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-64 w-full" />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function AutoTraderPage(): JSX.Element {
  const modesQuery = useBotModes();
  const run = useBotRun();
  const compare = useBotCompare();

  const [modeId, setModeId] = useState<BotConfig['mode']>('balanced');
  const [amount, setAmount] = useState<number>(20);
  const [classes, setClasses] = useState<AssetClass[]>([]);
  const [stopLossPct, setStopLossPct] = useState<number>(25);
  const [maxDrawdownPct, setMaxDrawdownPct] = useState<number>(35);

  const modes = useMemo<BotMode[]>(() => modesQuery.data ?? [], [modesQuery.data]);

  const toggleClass = (cls: AssetClass): void => {
    setClasses((prev) => (prev.includes(cls) ? prev.filter((c) => c !== cls) : [...prev, cls]));
  };

  const buildConfig = (): BotConfig => ({
    amount: Number.isFinite(amount) && amount > 0 ? amount : 20,
    mode: modeId,
    assetClasses: classes.length > 0 ? classes : null,
    rebalanceDays: 21,
    stopLossPct,
    maxDrawdownPct,
  });

  const onRun = (): void => {
    compare.reset();
    run.mutate({ config: buildConfig() });
  };

  const onCompare = (): void => {
    run.reset();
    // Empty modes => backend runs all five presets side-by-side.
    compare.mutate({ modes: [], config: buildConfig() });
  };

  const busy = run.isPending || compare.isPending;
  const error = run.error ?? compare.error;

  return (
    <div className="flex flex-col gap-4">
      {/* Page heading */}
      <div className="flex flex-col gap-1">
        <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-text">
          <Bot className="h-5 w-5 text-primary" aria-hidden />
          Auto-Trader
        </h1>
        <p className="text-sm text-muted">
          Backtest a simulated paper-trading bot across preset strategy modes — momentum / bandit rotation, never
          martingale.
        </p>
      </div>

      {/* PROMINENT simulation disclaimer */}
      <Card className="flex items-start gap-3 border-warning/40 bg-warning/8" role="note">
        <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-warning" aria-hidden />
        <div className="flex flex-col gap-0.5">
          <p className="text-sm font-semibold text-text">Simulation only — no real funds are traded</p>
          <p className="text-xs leading-relaxed text-muted">{BOT_DISCLAIMER}</p>
        </div>
      </Card>

      {/* Configuration */}
      <Card className="flex flex-col gap-4">
        <CardHeader>
          <CardTitle icon={<Bot className="h-4 w-4" />}>Configure your run</CardTitle>
        </CardHeader>

        {/* Mode picker */}
        <div className="flex flex-col gap-2">
          <span className="text-xs font-medium text-muted">Strategy mode</span>
          {modesQuery.isPending ? (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-28 w-full" />
              ))}
            </div>
          ) : modesQuery.isError ? (
            <div className="flex items-center justify-between gap-3 rounded-xl border border-danger/30 bg-danger/8 px-3 py-2">
              <span className="text-xs text-danger">Couldn&apos;t load the bot modes.</span>
              <Button variant="secondary" size="sm" onClick={() => void modesQuery.refetch()}>
                Retry
              </Button>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
              {modes.map((mode) => (
                <ModeTile
                  key={mode.id}
                  mode={mode}
                  selected={mode.id === modeId}
                  onSelect={() => setModeId(mode.id)}
                />
              ))}
            </div>
          )}
        </div>

        {/* Amount + asset classes + sliders */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="flex flex-col gap-4">
            {/* Amount */}
            <div className="flex flex-col gap-1.5">
              <label htmlFor="bot-amount" className="text-xs font-medium text-muted">
                Starting amount (simulated)
              </label>
              <div className="relative w-full sm:w-48">
                <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-sm text-muted">$</span>
                <input
                  id="bot-amount"
                  type="number"
                  inputMode="decimal"
                  min={1}
                  value={amount === 0 ? '' : amount}
                  onChange={(e) => {
                    const next = Number(e.target.value);
                    setAmount(Number.isFinite(next) ? next : 0);
                  }}
                  placeholder="20"
                  className="h-9 w-full rounded-xl border border-border bg-surface-2 pl-6 pr-3 text-sm tnum text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT"
                />
              </div>
            </div>

            {/* Asset-class toggles */}
            <div className="flex flex-col gap-1.5">
              <span className="text-xs font-medium text-muted">Asset classes</span>
              <div className="flex flex-wrap gap-2">
                {ASSET_CLASSES.map((cls) => {
                  const on = classes.includes(cls);
                  return (
                    <button
                      key={cls}
                      type="button"
                      aria-pressed={on}
                      onClick={() => toggleClass(cls)}
                      className={cn(
                        'inline-flex h-8 items-center rounded-xl border px-3 text-xs font-medium tracking-tight transition-colors',
                        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
                        on
                          ? 'border-primary bg-primary/12 text-primary'
                          : 'border-border bg-surface text-muted hover:bg-surface-2 hover:text-text',
                      )}
                    >
                      {assetClassLabel(cls)}
                    </button>
                  );
                })}
              </div>
              <p className="text-[11px] text-muted">
                {classes.length === 0 ? 'All classes (no filter).' : `${classes.length} selected.`}
              </p>
            </div>
          </div>

          {/* Risk sliders */}
          <div className="flex flex-col gap-4 rounded-xl border border-border bg-surface-2/40 p-3">
            <div className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-muted" aria-hidden />
              <span className="text-xs font-medium text-text">Risk controls</span>
            </div>
            <RiskSlider
              label="Per-sleeve stop-loss"
              hint="Exit a sleeve once it falls more than this from entry."
              value={stopLossPct}
              min={5}
              max={50}
              step={1}
              onChange={setStopLossPct}
            />
            <RiskSlider
              label="Portfolio max-drawdown"
              hint="Raise cash (circuit-breaker) once total drawdown exceeds this."
              value={maxDrawdownPct}
              min={10}
              max={60}
              step={1}
              onChange={setMaxDrawdownPct}
            />
          </div>
        </div>

        {/* Actions */}
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <Button
            variant="primary"
            size="lg"
            onClick={onRun}
            loading={run.isPending}
            disabled={busy || modesQuery.isError}
            leftIcon={!run.isPending ? <Play className="h-4 w-4" /> : undefined}
            className="sm:w-auto"
            fullWidth
          >
            Run backtest
          </Button>
          <Button
            variant="secondary"
            size="lg"
            onClick={onCompare}
            loading={compare.isPending}
            disabled={busy || modesQuery.isError}
            leftIcon={!compare.isPending ? <GitCompareArrows className="h-4 w-4" /> : undefined}
            className="sm:w-auto"
            fullWidth
          >
            Compare modes
          </Button>
        </div>

        {error && (
          <p className="rounded-lg bg-danger/10 px-3 py-2 text-xs font-medium text-danger" role="alert">
            {error instanceof Error ? error.message : 'The simulated run failed. Please try again.'}
          </p>
        )}
      </Card>

      {/* Results */}
      {busy ? (
        <ResultSkeleton />
      ) : compare.data && compare.data.length > 0 ? (
        <div className="flex flex-col gap-3">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-semibold tracking-tight text-text">Mode comparison</h2>
            <Badge tone="neutral" variant="soft" size="sm">
              {compare.data.length} modes · simulated
            </Badge>
          </div>
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
            {compare.data.map((res) => (
              <CompareColumn key={res.mode.id} result={res} />
            ))}
          </div>
        </div>
      ) : run.data ? (
        <RunResultPanel result={run.data} />
      ) : (
        <Card className="flex flex-col items-center justify-center gap-2 py-12 text-center">
          <TrendingDown className="h-6 w-6 text-muted opacity-60" aria-hidden />
          <p className="text-sm font-medium text-text">No simulation yet</p>
          <p className="max-w-md text-xs text-muted">
            Pick a mode, set your starting amount and risk controls, then run a backtest or compare all five modes
            side-by-side. Everything is simulated on synthetic data.
          </p>
        </Card>
      )}

      {/* Footer disclaimer */}
      <div className="flex items-start gap-2 rounded-xl border border-border bg-surface-2/40 px-3 py-2">
        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted" aria-hidden />
        <p className="text-[11px] leading-relaxed text-muted">{BOT_DISCLAIMER}</p>
      </div>
    </div>
  );
}
