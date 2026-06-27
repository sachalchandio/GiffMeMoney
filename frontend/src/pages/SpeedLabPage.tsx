/**
 * SpeedLabPage — the Trade-Speed Lab.
 *
 * Answers one question honestly: *does trading faster / in smaller portions make
 * more money?* It runs a paper simulation that charges real transaction costs and
 * sweeps how often the strategy trades, then shows where after-cost return
 * actually peaks. The honest result on edge-free synthetic data: past a low
 * turnover, every extra trade just feeds the spread.
 *
 * Two lenses (Easy / Expert):
 *  - Easy   — the big answer, three plain scenarios, and a one-line "why".
 *  - Expert — full controls, the turnover curve, the sweep table, the gross-vs-
 *    net-vs-buy&hold equity chart, and the cost model.
 *
 * Everything here is a SIMULATION on synthetic data — no real money, and (stated
 * plainly) not microsecond trading: a web app is ~a million times too slow.
 */

import { useEffect, useMemo, useState } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Gauge,
  Rocket,
  ShieldCheck,
  TimerReset,
  TrendingDown,
  TrendingUp,
  Wallet,
  Zap,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Select } from '@/components/ui/Select';
import { Skeleton } from '@/components/ui/Skeleton';
import { ExpertOnly, Explain, WhatThisMeans } from '@/components/ui/ModeView';
import { ChartTooltip, type TooltipRow } from '@/components/charts/ChartTooltip';
import { ChartEmpty } from '@/components/charts/ChartEmpty';
import { useHftSweep, useHftSim } from '@/hooks/useHft';
import { useUiMode } from '@/theme/UiModeProvider';
import { useChartTokens } from '@/theme/tokens';
import { cn, changeTextColor } from '@/lib/utils';
import { formatPct, formatNumber } from '@/lib/format';
import type { HftSignal, HftSweepPoint, HftSimResult } from '@/lib/types';

/* ------------------------------------------------------------------ */
/* Controls                                                            */
/* ------------------------------------------------------------------ */

const SYMBOLS = ['BTC', 'ETH', 'AAPL', 'NVDA', 'SOL', 'SYNTH'] as const;

const SIGNALS: { value: HftSignal; label: string }[] = [
  { value: 'meanrev', label: 'Mean reversion (fade the move)' },
  { value: 'momentum', label: 'Momentum (ride the trend)' },
  { value: 'buyhold', label: 'Buy & hold (no trading)' },
];

const COST_OPTS = [
  { value: 'retail-crypto', label: 'Retail crypto (~28 bps round trip)' },
  { value: 'retail-equity', label: 'Retail stock (~2 bps round trip)' },
  { value: 'retail-crypto-expensive', label: 'High-fee venue (~100 bps)' },
  { value: 'zero', label: 'Frictionless (illustration only)' },
] as const;

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function SpeedLabPage(): JSX.Element {
  const { isEasy } = useUiMode();
  const [symbol, setSymbol] = useState<string>('BTC');
  const [signal, setSignal] = useState<HftSignal>('meanrev');
  const [costPreset, setCostPreset] = useState<string>('retail-crypto');

  const sweep = useHftSweep();
  const sim = useHftSim();

  const runExperiment = (): void => {
    const base = { symbol, signal, costPreset, amount: 20, days: 30 };
    sweep.mutate({ base });
    // A single hyperactive run powers the gross-vs-net equity chart (Expert).
    sim.mutate({ ...base, rebalanceInterval: 1 });
  };

  // Auto-run so the page is never empty. Data-driven (not a fire-once ref) so it
  // reliably lands even under React StrictMode's mount/remount in dev, and never
  // loops (it stops once there's a result, a request in flight, or an error).
  useEffect(() => {
    if (!sweep.data && !sweep.isPending && !sweep.isError) {
      runExperiment();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sweep.data, sweep.isPending, sweep.isError]);

  const data = sweep.data;
  const loading = sweep.isPending;

  return (
    <div className="space-y-5">
      {/* ---- Hero ---- */}
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="accent" variant="soft" size="sm" icon={<Zap className="h-[0.875rem] w-[0.875rem]" />}>
            Trade-Speed Lab
          </Badge>
          <Badge tone="neutral" variant="outline" size="sm">
            Simulation · synthetic data
          </Badge>
        </div>
        <h1 className="text-balance text-2xl font-bold tracking-tight text-text lg:text-3xl">
          Does trading faster make you more money?
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted">
          {isEasy ? (
            <>
              Splitting your money into more pieces and trading more often <em>feels</em> smart. Let&apos;s
              test it honestly — with the real cost of every trade included — and see what actually happens
              to your money.
            </>
          ) : (
            <>
              We hold a short-horizon strategy fixed and vary only <em>how often it trades</em>, charging a
              realistic spread + fee on every trade. The curve below is the after-cost truth.
            </>
          )}
        </p>
      </div>

      {/* ---- Controls ---- */}
      <Card className="flex flex-col gap-3">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <Select
            label="Market"
            value={symbol}
            onChange={setSymbol}
            options={SYMBOLS.map((s) => ({ value: s, label: s }))}
            fullWidth
          />
          <ExpertOnly>
            <Select
              label="Strategy"
              value={signal}
              onChange={(v) => setSignal(v as HftSignal)}
              options={SIGNALS}
              fullWidth
            />
          </ExpertOnly>
          <Select
            label={isEasy ? 'Where you trade' : 'Cost model'}
            value={costPreset}
            onChange={setCostPreset}
            options={COST_OPTS.map((o) => ({ value: o.value, label: o.label }))}
            fullWidth
          />
          <div className="flex items-end">
            <Button
              variant="primary"
              fullWidth
              loading={loading || sim.isPending}
              onClick={runExperiment}
              leftIcon={<TimerReset className="h-4 w-4" />}
            >
              {isEasy ? 'Run the experiment' : 'Run sweep'}
            </Button>
          </div>
        </div>
        <p className="text-[0.6875rem] leading-relaxed text-muted">
          Starts with a simulated <strong>$20</strong> over 30 days of made-up price data. Not microsecond
          trading — a web app is roughly a million times too slow for that, so this trades in minutes-long
          bars. No real money moves.
        </p>
      </Card>

      {/* ---- The answer ---- */}
      {loading || !data ? (
        <Skeleton className="h-[18rem] w-full rounded-2xl" />
      ) : (
        <Answer data={data} />
      )}

      {/* ---- The curve ---- */}
      {!loading && data && data.points.length > 0 && (
        <Card className="flex flex-col gap-3">
          <CardHeader>
            <div>
              <CardTitle icon={<Gauge className="h-4 w-4" />}>
                {isEasy ? 'How much you keep, by how often you trade' : 'Net return vs turnover'}
              </CardTitle>
              <CardDescription>
                {isEasy
                  ? 'Left = trading constantly. Right = trading rarely. Higher line = more money kept.'
                  : 'Gross (no costs) vs net (after costs). The gap is the cost drag; the dot is the net-of-cost optimum.'}
              </CardDescription>
            </div>
          </CardHeader>
          <TurnoverChart data={data} />
        </Card>
      )}

      {/* ---- Expert detail: sweep table + equity chart + cost model ---- */}
      <ExpertOnly>
        {!loading && data && data.points.length > 0 && (
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <Card className="flex flex-col gap-3" flush>
              <div className="p-4 pb-0">
                <CardTitle icon={<TrendingUp className="h-4 w-4" />}>Every setting, side by side</CardTitle>
              </div>
              <SweepTable points={data.points} optimumInterval={data.optimumByNetReturn?.interval} />
            </Card>
            <Card className="flex flex-col gap-3">
              <CardHeader>
                <div>
                  <CardTitle icon={<TrendingDown className="h-4 w-4 text-danger" />}>
                    Cost drag, live (trading every bar)
                  </CardTitle>
                  <CardDescription>Same decisions, with vs without costs — the gap is what you lose.</CardDescription>
                </div>
              </CardHeader>
              {sim.data ? <EquityChart result={sim.data} /> : <ChartEmpty height={260} label="Run the experiment" />}
              {sim.data?.costModel && (
                <p className="text-xs text-muted">
                  Cost model: <strong>{sim.data.costModel.name}</strong> · ~
                  {formatNumber(sim.data.costModel.roundTripBps, 0)} bps round trip. {sim.data.costModel.note}
                </p>
              )}
            </Card>
          </div>
        )}
      </ExpertOnly>

      {/* ---- Why / honesty ---- */}
      <WhyCard />

      <p className="text-[0.6875rem] leading-relaxed text-muted">
        {data?.disclaimer}
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The answer block (big scenarios + verdict)                          */
/* ------------------------------------------------------------------ */

function Answer({ data }: { data: NonNullable<ReturnType<typeof useHftSweep>['data']> }): JSX.Element {
  const fast = data.naiveFast;
  const opt = data.optimumByNetReturn;
  const bh = data.buyHoldReturnPct;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <ScenarioCard
          icon={<Rocket className="h-4 w-4" />}
          tone="danger"
          title="Trade constantly"
          plain="Re-decide every few minutes"
          net={fast?.netReturnPct ?? 0}
          footer={fast ? `${formatNumber(fast.turnover, 0)}× turnover · ${formatPct(fast.costDragPct)} lost to costs` : ''}
        />
        <ScenarioCard
          icon={<ShieldCheck className="h-4 w-4" />}
          tone="primary"
          title="Trade in moderation"
          plain={opt ? `Best after-cost setting (${opt.label})` : 'Best after-cost setting'}
          net={opt?.netReturnPct ?? 0}
          footer={opt ? `${formatNumber(opt.turnover, 0)}× turnover · only ${formatPct(opt.costDragPct)} lost` : ''}
          highlight
        />
        <ScenarioCard
          icon={<Wallet className="h-4 w-4" />}
          tone="neutral"
          title="Just buy & wait"
          plain="One trade, then hold"
          net={bh}
          footer="~0 costs · the patient option"
        />
      </div>

      <WhatThisMeans tone="warn">
        {data.verdict}
      </WhatThisMeans>
    </div>
  );
}

function ScenarioCard({
  icon,
  tone,
  title,
  plain,
  net,
  footer,
  highlight = false,
}: {
  icon: JSX.Element;
  tone: 'danger' | 'primary' | 'neutral';
  title: string;
  plain: string;
  net: number;
  footer: string;
  highlight?: boolean;
}): JSX.Element {
  const ring =
    tone === 'primary'
      ? 'border-primary/40 bg-primary/[0.06]'
      : tone === 'danger'
        ? 'border-danger/30'
        : 'border-border';
  const chip =
    tone === 'primary' ? 'bg-primary/15 text-primary' : tone === 'danger' ? 'bg-danger/12 text-danger' : 'bg-surface-2 text-muted';
  return (
    <Card className={cn('flex flex-col gap-2', ring, highlight && 'shadow-card')}>
      <div className="flex items-center gap-2">
        <span className={cn('flex h-7 w-7 items-center justify-center rounded-xl', chip)}>{icon}</span>
        <div className="min-w-0">
          <div className="text-sm font-semibold tracking-tight text-text">{title}</div>
          <div className="truncate text-[0.6875rem] text-muted">{plain}</div>
        </div>
      </div>
      <div className={cn('text-3xl font-bold tracking-tight tnum', changeTextColor(net))}>
        {formatPct(net, { sign: true, digits: 1 })}
      </div>
      <div className="mt-auto text-[0.6875rem] leading-relaxed text-muted">{footer}</div>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Turnover curve                                                      */
/* ------------------------------------------------------------------ */

interface CurveRow {
  turnover: number;
  net: number;
  gross: number;
  drag: number;
  label: string;
}

function TurnoverChart({ data }: { data: NonNullable<ReturnType<typeof useHftSweep>['data']> }): JSX.Element {
  const tokens = useChartTokens();
  const rows: CurveRow[] = useMemo(
    () =>
      [...data.points]
        .sort((a, b) => a.turnover - b.turnover)
        .map((p) => ({
          turnover: p.turnover,
          net: p.netReturnPct,
          gross: p.grossReturnPct,
          drag: p.costDragPct,
          label: p.label,
        })),
    [data.points],
  );
  const opt = data.optimumByNetReturn;

  return (
    <div style={{ width: '100%', height: '20rem' }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 18, left: -6 }}>
          <defs>
            <linearGradient id="drag-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={tokens.down} stopOpacity={0.18} />
              <stop offset="100%" stopColor={tokens.down} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={tokens.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="turnover"
            type="number"
            scale="sqrt"
            domain={['auto', 'auto']}
            tickFormatter={(v: number) => `${formatNumber(v, 0)}×`}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: tokens.grid }}
            label={{ value: 'Turnover (how much you trade) →', position: 'insideBottom', offset: -10, fill: tokens.muted, fontSize: 11 }}
          />
          <YAxis
            tickFormatter={(v: number) => `${formatNumber(v, 0)}%`}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={48}
          />
          <Tooltip
            cursor={{ stroke: tokens.muted, strokeDasharray: '3 3' }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0]?.payload as CurveRow | undefined;
              if (!row) return null;
              const out: TooltipRow[] = [
                { label: 'Net (after costs)', value: formatPct(row.net, { sign: true }), color: tokens.primary },
                { label: 'Gross (no costs)', value: formatPct(row.gross, { sign: true }), color: tokens.accent },
                { label: 'Cost drag', value: formatPct(row.drag), color: tokens.down },
              ];
              return <ChartTooltip title={row.label} rows={out} />;
            }}
          />
          <Legend verticalAlign="top" height={24} iconType="plainline" wrapperStyle={{ fontSize: 11, color: tokens.muted }} />
          <ReferenceLine
            y={data.buyHoldReturnPct}
            stroke={tokens.muted}
            strokeDasharray="5 4"
            label={{ value: 'buy & hold', position: 'right', fill: tokens.muted, fontSize: 10 }}
          />
          <Area
            type="monotone"
            name="Cost drag"
            dataKey="drag"
            stroke={tokens.down}
            strokeWidth={1}
            fill="url(#drag-fill)"
            isAnimationActive={false}
            dot={false}
          />
          <Line type="monotone" name="Gross (no costs)" dataKey="gross" stroke={tokens.accent} strokeWidth={1.5} strokeDasharray="5 4" isAnimationActive={false} dot={false} />
          <Line type="monotone" name="Net (after costs)" dataKey="net" stroke={tokens.primary} strokeWidth={2.5} isAnimationActive={false} dot={{ r: 2 }} />
          {opt && (
            <ReferenceDot
              x={opt.turnover}
              y={opt.netReturnPct}
              r={5}
              fill={tokens.primary}
              stroke={tokens.surface}
              strokeWidth={2}
              isFront
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Sweep table (expert)                                                */
/* ------------------------------------------------------------------ */

function SweepTable({ points, optimumInterval }: { points: HftSweepPoint[]; optimumInterval?: number }): JSX.Element {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-[0.625rem] uppercase tracking-wide text-muted">
            <th className="px-4 py-2 font-semibold">Cadence</th>
            <th className="px-3 py-2 text-right font-semibold">Turnover</th>
            <th className="px-3 py-2 text-right font-semibold">Trades</th>
            <th className="px-3 py-2 text-right font-semibold">Gross</th>
            <th className="px-3 py-2 text-right font-semibold">Net</th>
            <th className="px-4 py-2 text-right font-semibold">Drag</th>
          </tr>
        </thead>
        <tbody>
          {points.map((p) => {
            const best = p.interval === optimumInterval;
            return (
              <tr
                key={p.interval}
                className={cn('border-b border-border/60 last:border-0', best && 'bg-primary/[0.06]')}
              >
                <td className="px-4 py-2">
                  <span className="flex items-center gap-1.5 font-medium text-text">
                    {p.label}
                    {best && <Badge tone="primary" variant="soft" size="sm">best net</Badge>}
                  </span>
                </td>
                <td className="px-3 py-2 text-right tnum text-muted">{formatNumber(p.turnover, 0)}×</td>
                <td className="px-3 py-2 text-right tnum text-muted">{formatNumber(p.trades, 0)}</td>
                <td className="px-3 py-2 text-right tnum text-muted">{formatPct(p.grossReturnPct, { sign: true })}</td>
                <td className={cn('px-3 py-2 text-right font-semibold tnum', changeTextColor(p.netReturnPct))}>
                  {formatPct(p.netReturnPct, { sign: true })}
                </td>
                <td className="px-4 py-2 text-right tnum text-danger">−{formatPct(p.costDragPct)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Gross vs net vs buy&hold equity (expert)                            */
/* ------------------------------------------------------------------ */

interface EqRow {
  i: number;
  gross: number;
  net: number;
  bh: number;
}

function EquityChart({ result }: { result: HftSimResult }): JSX.Element {
  const tokens = useChartTokens();
  const n = Math.min(result.netCurve.length, result.grossCurve.length, result.buyHoldCurve.length);
  if (n < 2) return <ChartEmpty height={260} label="No curve" />;
  const rows: EqRow[] = Array.from({ length: n }, (_, i) => ({
    i,
    gross: result.grossCurve[i] ?? 0,
    net: result.netCurve[i] ?? 0,
    bh: result.buyHoldCurve[i] ?? 0,
  }));
  return (
    <div style={{ width: '100%', height: '16rem' }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -6 }}>
          <CartesianGrid stroke={tokens.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="i" hide />
          <YAxis
            tickFormatter={(v: number) => `$${formatNumber(v, 0)}`}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={48}
            domain={['auto', 'auto']}
          />
          <Tooltip
            cursor={{ stroke: tokens.muted, strokeDasharray: '3 3' }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0]?.payload as EqRow | undefined;
              if (!row) return null;
              const out: TooltipRow[] = [
                { label: 'Gross (no costs)', value: `$${formatNumber(row.gross, 2)}`, color: tokens.accent },
                { label: 'Net (after costs)', value: `$${formatNumber(row.net, 2)}`, color: tokens.primary },
                { label: 'Buy & hold', value: `$${formatNumber(row.bh, 2)}`, color: tokens.muted },
              ];
              return <ChartTooltip title="Account value" rows={out} />;
            }}
          />
          <Legend verticalAlign="top" height={24} iconType="plainline" wrapperStyle={{ fontSize: 11, color: tokens.muted }} />
          <Line type="monotone" name="Gross" dataKey="gross" stroke={tokens.accent} strokeWidth={1.5} strokeDasharray="5 4" isAnimationActive={false} dot={false} />
          <Line type="monotone" name="Net" dataKey="net" stroke={tokens.primary} strokeWidth={2} isAnimationActive={false} dot={false} />
          <Line type="monotone" name="Buy & hold" dataKey="bh" stroke={tokens.muted} strokeWidth={1.25} isAnimationActive={false} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Why card                                                            */
/* ------------------------------------------------------------------ */

function WhyCard(): JSX.Element {
  return (
    <Card className="flex flex-col gap-3">
      <CardTitle icon={<ShieldCheck className="h-4 w-4 text-primary" />}>Why fast trading usually loses</CardTitle>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Reason
          n="1"
          title="Every trade pays a toll"
          body={
            <>
              You cross the <Explain term="bid-ask spread" plain="gap between buy and sell prices" /> and pay
              fees on <em>each</em> trade. Trade twice as often, pay twice as much.
            </>
          }
        />
        <Reason
          n="2"
          title="Speed you don't have"
          body={
            <>
              Real high-frequency firms sit <em>inside</em> the exchange. A web app is ~a million times slower,
              so you're the slow money the fast money trades against.
            </>
          }
        />
        <Reason
          n="3"
          title="Noise isn't signal"
          body={
            <>
              The faster you trade, the more you're reacting to random wiggles. More chances to be wrong, at
              full cost each time.
            </>
          }
        />
      </div>
    </Card>
  );
}

function Reason({ n, title, body }: { n: string; title: string; body: JSX.Element }): JSX.Element {
  return (
    <div className="rounded-xl border border-border bg-surface-2/40 p-3">
      <div className="mb-1 flex items-center gap-2">
        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-primary/15 text-[0.6875rem] font-bold text-primary">
          {n}
        </span>
        <span className="text-sm font-semibold tracking-tight text-text">{title}</span>
      </div>
      <p className="text-[0.8125rem] leading-relaxed text-muted">{body}</p>
    </div>
  );
}
