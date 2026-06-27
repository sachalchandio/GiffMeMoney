/**
 * RealTimePage — the Real-Time mode: a live-feeling, multi-venue PAPER book.
 *
 * It ticks forward every ~1.5s, spreading the (paper) balance across more venues
 * as it grows, rotating into the ones scoring best, and showing today's profit or
 * loss. Two lenses: Easy (your balance, today's move, where the money is, in plain
 * words) and Expert (full stat grid, equity + daily-P&L charts, the venue board,
 * recent trades, the cost model).
 *
 * HONESTY: this is an accelerated SIMULATION — $0 real money, costs charged,
 * returns kept realistic. A loud banner says so; it never pretends to be live
 * real-money trading.
 */

import { useState } from 'react';
import {
  Area,
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  ComposedChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Activity,
  AlertTriangle,
  Brain,
  Pause,
  Play,
  RotateCcw,
  Radio,
  TrendingUp,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Select } from '@/components/ui/Select';
import { StatCard } from '@/components/ui/StatCard';
import { EasyOnly, ExpertOnly, Explain, WhatThisMeans } from '@/components/ui/ModeView';
import { ChartTooltip, type TooltipRow } from '@/components/charts/ChartTooltip';
import { ChartEmpty } from '@/components/charts/ChartEmpty';
import { useLiveSim } from '@/hooks/useLiveSim';
import { useUiMode } from '@/theme/UiModeProvider';
import { useChartTokens } from '@/theme/tokens';
import { cn, changeTextColor } from '@/lib/utils';
import { formatCurrency, formatPct, formatNumber } from '@/lib/format';
import type { LiveSimSignal, LiveSimState, LiveSimVenue } from '@/lib/types';

const AMOUNTS = ['20', '50', '100', '500'] as const;
const MAX_VENUES = ['20', '40', '80'] as const;
const SIGNALS: { value: LiveSimSignal; label: string }[] = [
  { value: 'momentum', label: 'Momentum (ride winners)' },
  { value: 'meanrev', label: 'Mean reversion (buy dips)' },
];
const COST_OPTS = [
  { value: 'retail-crypto', label: 'Retail crypto (~28 bps)' },
  { value: 'retail-equity', label: 'Retail stock (~2 bps)' },
  { value: 'retail-crypto-expensive', label: 'High-fee venue (~100 bps)' },
] as const;

export default function RealTimePage(): JSX.Element {
  const { isEasy } = useUiMode();
  const sim = useLiveSim();
  const [amount, setAmount] = useState<string>('20');
  const [maxVenues, setMaxVenues] = useState<string>('20');
  const [signal, setSignal] = useState<LiveSimSignal>('momentum');
  const [costPreset, setCostPreset] = useState<string>('retail-crypto');

  const started = sim.state !== null;

  const onStart = (): void => {
    void sim.start({
      amount: Number(amount),
      maxVenues: Number(maxVenues),
      signal,
      costPreset,
    });
  };

  return (
    <div className="space-y-5">
      {/* Always-on honesty banner */}
      <div className="flex items-start gap-2 rounded-2xl border border-warning/40 bg-warning/[0.08] px-4 py-3 text-warning">
        <AlertTriangle className="mt-0.5 h-[1rem] w-[1rem] shrink-0" aria-hidden />
        <p className="text-[0.8125rem] leading-relaxed">
          <b>Practice mode — $0 real money.</b> This is an accelerated simulation so you can watch how
          spreading and rotation work. Costs are charged and returns are realistic, so it won&apos;t make
          you rich — nothing legitimate turns $20 into thousands.
        </p>
      </div>

      {/* Hero */}
      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="accent" variant="soft" size="sm" icon={<Radio className="h-[0.875rem] w-[0.875rem]" />}>
            Real-Time mode
          </Badge>
          {sim.running && (
            <span className="inline-flex items-center gap-1.5 text-[0.75rem] font-semibold text-success">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-success/70" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-success" />
              </span>
              LIVE · ticking
            </span>
          )}
        </div>
        <h1 className="text-balance text-2xl font-bold tracking-tight text-text lg:text-3xl">
          {isEasy ? 'Watch your money work, live' : 'Real-time multi-venue rotation'}
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted">
          {isEasy ? (
            <>
              Press start. The bot spreads your practice balance across many venues, keeps the ones doing
              well, drops the ones that aren&apos;t, and shows your profit as it goes — all in fast-forward.
            </>
          ) : (
            <>
              A self-updating predictor scores every venue each tick; the book spreads wider as equity grows
              (capped at 80), rotates into winners, and charges realistic costs. Accelerated synthetic data.
            </>
          )}
        </p>
      </div>

      {/* Controls */}
      <Card className="flex flex-col gap-3">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
          <Select
            label={isEasy ? 'Practice money' : 'Starting capital'}
            value={amount}
            onChange={setAmount}
            options={AMOUNTS.map((a) => ({ value: a, label: `$${a}` }))}
            disabled={started && !sim.state?.finished}
            fullWidth
          />
          <Select
            label={isEasy ? 'Spread up to' : 'Max venues'}
            value={maxVenues}
            onChange={setMaxVenues}
            options={MAX_VENUES.map((m) => ({ value: m, label: `${m} venues` }))}
            disabled={started && !sim.state?.finished}
            fullWidth
          />
          <ExpertOnly>
            <Select
              label="Strategy"
              value={signal}
              onChange={(v) => setSignal(v as LiveSimSignal)}
              options={SIGNALS}
              disabled={started && !sim.state?.finished}
              fullWidth
            />
            <Select
              label="Cost model"
              value={costPreset}
              onChange={setCostPreset}
              options={COST_OPTS.map((o) => ({ value: o.value, label: o.label }))}
              disabled={started && !sim.state?.finished}
              fullWidth
            />
          </ExpertOnly>
          <div className="flex items-end gap-2">
            {!started || sim.state?.finished ? (
              <Button variant="primary" fullWidth loading={sim.starting} onClick={onStart} leftIcon={<Play className="h-4 w-4" />}>
                Start
              </Button>
            ) : (
              <>
                {sim.running ? (
                  <Button variant="secondary" fullWidth onClick={sim.pause} leftIcon={<Pause className="h-4 w-4" />}>
                    Pause
                  </Button>
                ) : (
                  <Button variant="primary" fullWidth onClick={sim.resume} leftIcon={<Play className="h-4 w-4" />}>
                    Resume
                  </Button>
                )}
                <Button variant="ghost" size="icon" onClick={sim.reset} aria-label="Reset" title="Reset">
                  <RotateCcw className="h-4 w-4" />
                </Button>
              </>
            )}
          </div>
        </div>
        {sim.error && <p className="text-xs text-danger">{sim.error}</p>}
      </Card>

      {/* Empty state */}
      {!sim.state ? (
        <Card className="flex flex-col items-center justify-center gap-2 py-12 text-center">
          <Brain className="h-8 w-8 text-muted" aria-hidden />
          <p className="text-sm font-medium text-text">Press start to run the live simulation.</p>
          <p className="max-w-md text-xs text-muted">
            You&apos;ll see your practice balance spread across venues and update every second or so.
          </p>
        </Card>
      ) : (
        <LiveDashboard state={sim.state} />
      )}

      <p className="text-[0.6875rem] leading-relaxed text-muted">{sim.state?.disclaimer}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Live dashboard                                                      */
/* ------------------------------------------------------------------ */

function LiveDashboard({ state }: { state: LiveSimState }): JSX.Element {
  const { isEasy } = useUiMode();

  return (
    <div className="space-y-4">
      {/* Headline stats */}
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          label={isEasy ? 'Practice balance' : 'Equity'}
          value={formatCurrency(state.equity)}
          deltaPct={state.totalPnlPct}
          icon={<Activity className="h-4 w-4" />}
        />
        <StatCard
          label="Today's profit"
          value={formatPct(state.dayPnlPct, { sign: true, digits: 2 })}
          hint={`Sim day ${formatNumber(state.day, 0)}`}
        />
        <StatCard
          label={isEasy ? 'Spread across' : 'Venues held'}
          value={`${state.venuesActive}`}
          hint={`of up to ${state.venuesTarget} (cap ${state.venuesMax})`}
        />
        <StatCard
          label={isEasy ? 'Worst dip so far' : 'Max drawdown'}
          value={formatPct(state.maxDrawdownPct, { digits: 1 })}
          hint={`Cash: ${formatCurrency(state.cash)}`}
        />
      </div>

      <EasyOnly>
        <WhatThisMeans tone={state.totalPnlPct >= 0 ? 'good' : 'warn'}>
          Your <b>{formatCurrency(state.startEquity)}</b> practice balance is now{' '}
          <b>{formatCurrency(state.equity)}</b> ({formatPct(state.totalPnlPct, { sign: true, digits: 1 })}),
          spread across <b>{state.venuesActive}</b> venues. The bot keeps moving money toward what&apos;s
          working — but costs and randomness mean it drifts. This is how real investing *feels*, safely.
        </WhatThisMeans>
      </EasyOnly>

      {/* Equity curve */}
      <Card className="flex flex-col gap-3">
        <CardHeader>
          <div>
            <CardTitle icon={<TrendingUp className="h-4 w-4" />}>
              {isEasy ? 'Your balance over time' : 'Equity curve'}
            </CardTitle>
            <CardDescription>Updates live as the simulation ticks forward.</CardDescription>
          </div>
        </CardHeader>
        <EquityArea curve={state.equityCurve} start={state.startEquity} />
      </Card>

      {/* Venue board + daily P&L */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <Card className="flex flex-col gap-3 xl:col-span-2">
          <CardHeader>
            <div>
              <CardTitle icon={<Activity className="h-4 w-4" />}>
                {isEasy ? 'Where your money is' : 'Venue board (by score)'}
              </CardTitle>
              <CardDescription>
                {isEasy ? 'The venues the bot likes most right now.' : 'Live score, prediction, weight and P&L per venue.'}
              </CardDescription>
            </div>
          </CardHeader>
          <VenueBoard venues={state.venues} easy={isEasy} />
        </Card>

        <Card className="flex flex-col gap-3">
          <CardHeader>
            <CardTitle icon={<TrendingUp className="h-4 w-4" />}>Daily profit</CardTitle>
          </CardHeader>
          <DailyPnl daily={state.dailyPnl} />
        </Card>
      </div>

      {/* Recent trades (expert) */}
      <ExpertOnly>
        <Card className="flex flex-col gap-3">
          <CardTitle icon={<Activity className="h-4 w-4" />}>Recent trades</CardTitle>
          {state.recentTrades.length === 0 ? (
            <p className="text-xs text-muted">No trades yet — the book reallocates on a cadence.</p>
          ) : (
            <div className="flex flex-wrap gap-1.5">
              {state.recentTrades.slice(0, 24).map((t, i) => (
                <span
                  key={`${t.symbol}-${t.t}-${i}`}
                  className={cn(
                    'inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-[0.6875rem] tnum',
                    t.side === 'buy'
                      ? 'border-success/30 bg-success/8 text-success'
                      : 'border-danger/30 bg-danger/8 text-danger',
                  )}
                >
                  <span className="font-semibold">{t.side === 'buy' ? 'BUY' : 'SELL'}</span>
                  {t.symbol} · {formatCurrency(t.amount)}
                </span>
              ))}
            </div>
          )}
          {state.costModel && (
            <p className="text-xs text-muted">
              Cost model: <strong>{state.costModel.name}</strong> · ~
              {formatNumber(state.costModel.roundTripBps, 0)} bps round trip — charged on every trade.
            </p>
          )}
        </Card>
      </ExpertOnly>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Charts                                                              */
/* ------------------------------------------------------------------ */

function EquityArea({ curve, start }: { curve: number[]; start: number }): JSX.Element {
  const tokens = useChartTokens();
  if (curve.length < 2) return <ChartEmpty height={220} label="Warming up…" />;
  const rows = curve.map((v, i) => ({ i, v }));
  const up = (curve[curve.length - 1] ?? start) >= start;
  const color = up ? tokens.up : tokens.down;
  return (
    <div style={{ width: '100%', height: '16rem' }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -6 }}>
          <defs>
            <linearGradient id="ls-eq" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.24} />
              <stop offset="100%" stopColor={color} stopOpacity={0.02} />
            </linearGradient>
          </defs>
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
              const row = payload[0]?.payload as { v: number } | undefined;
              if (!row) return null;
              const out: TooltipRow[] = [{ label: 'Balance', value: formatCurrency(row.v), color }];
              return <ChartTooltip title="Practice balance" rows={out} />;
            }}
          />
          <Area type="monotone" dataKey="v" stroke={color} strokeWidth={2} fill="url(#ls-eq)" isAnimationActive={false} dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function DailyPnl({ daily }: { daily: LiveSimState['dailyPnl'] }): JSX.Element {
  const tokens = useChartTokens();
  if (daily.length === 0) return <ChartEmpty height={200} label="No full day yet" />;
  const rows = daily.slice(-30).map((d) => ({ day: d.day, pnl: d.pnlPct }));
  return (
    <div style={{ width: '100%', height: '14rem' }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 8, right: 8, bottom: 4, left: -10 }}>
          <CartesianGrid stroke={tokens.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="day" tick={{ fill: tokens.muted, fontSize: 10 }} tickLine={false} axisLine={{ stroke: tokens.grid }} minTickGap={16} />
          <YAxis tickFormatter={(v: number) => `${formatNumber(v, 0)}%`} tick={{ fill: tokens.muted, fontSize: 11 }} tickLine={false} axisLine={false} width={40} />
          <Tooltip
            cursor={{ fill: tokens.surface2, opacity: 0.5 }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0]?.payload as { day: number; pnl: number } | undefined;
              if (!row) return null;
              const out: TooltipRow[] = [
                { label: `Day ${row.day}`, value: formatPct(row.pnl, { sign: true }), color: row.pnl >= 0 ? tokens.up : tokens.down },
              ];
              return <ChartTooltip title="Daily P&L" rows={out} />;
            }}
          />
          <Bar dataKey="pnl" radius={[3, 3, 0, 0]} isAnimationActive={false}>
            {rows.map((r, i) => (
              <Cell key={i} fill={r.pnl >= 0 ? tokens.up : tokens.down} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Venue board                                                         */
/* ------------------------------------------------------------------ */

function VenueBoard({ venues, easy }: { venues: LiveSimVenue[]; easy: boolean }): JSX.Element {
  const held = venues.filter((v) => v.held);
  const list = easy ? (held.length ? held : venues).slice(0, 6) : venues.slice(0, 40);

  if (easy) {
    return (
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {list.map((v) => (
          <div key={v.symbol} className="flex items-center justify-between gap-2 rounded-xl border border-border bg-surface-2/40 px-3 py-2">
            <div className="min-w-0">
              <div className="text-sm font-semibold tracking-tight text-text">{v.label}</div>
              <div className="text-[0.6875rem] text-muted">{v.held ? `${v.weightPct.toFixed(0)}% of your money` : 'watching'}</div>
            </div>
            <div className="text-right">
              <div className="text-sm tnum text-text">{formatCurrency(v.positionValue)}</div>
              <div className={cn('text-[0.6875rem] font-medium tnum', changeTextColor(v.pnlPct))}>
                {formatPct(v.pnlPct, { sign: true })}
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border text-left text-[0.625rem] uppercase tracking-wide text-muted">
            <th className="px-3 py-2 font-semibold">Venue</th>
            <th className="px-3 py-2 text-right font-semibold">Price</th>
            <th className="px-3 py-2 text-right font-semibold">
              <Explain term="Pred ↑" plain="chance up" />
            </th>
            <th className="px-3 py-2 text-right font-semibold">Score</th>
            <th className="px-3 py-2 text-right font-semibold">Weight</th>
            <th className="px-3 py-2 text-right font-semibold">P&L</th>
          </tr>
        </thead>
        <tbody>
          {list.map((v) => (
            <tr key={v.symbol} className={cn('border-b border-border/60 last:border-0', v.held && 'bg-primary/[0.05]')}>
              <td className="px-3 py-1.5">
                <span className="flex items-center gap-1.5">
                  <span className="font-medium text-text">{v.symbol}</span>
                  {v.held && <span className="h-1.5 w-1.5 rounded-full bg-primary" aria-label="held" />}
                </span>
              </td>
              <td className="px-3 py-1.5 text-right tnum text-muted">{formatCurrency(v.price)}</td>
              <td className="px-3 py-1.5 text-right tnum text-muted">{(v.predUpProb * 100).toFixed(0)}%</td>
              <td className={cn('px-3 py-1.5 text-right tnum', v.score > 0 ? 'text-success' : 'text-muted')}>
                {v.score.toFixed(2)}
              </td>
              <td className="px-3 py-1.5 text-right tnum text-text">{v.weightPct > 0 ? `${v.weightPct.toFixed(0)}%` : '—'}</td>
              <td className={cn('px-3 py-1.5 text-right tnum', changeTextColor(v.pnlPct))}>
                {v.held ? formatPct(v.pnlPct, { sign: true }) : '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
