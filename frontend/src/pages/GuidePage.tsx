/**
 * GuidePage — "Start here", the friendly on-ramp for someone brand new.
 *
 * Built to be *played with*, not just read: a short click-through tour, a
 * hands-on compounding playground (drag the sliders, watch the money grow), flip
 * cards that bust the get-rich-quick myths, a jargon buster, and "pick your path"
 * cards that send you to your first real action. Plain language throughout; no
 * finance background assumed.
 *
 * Pure client-side — no backend, no real money. Honest by design: the playground
 * uses a realistic ~8%/year average and shows that small + regular + patient is
 * the path, never $20→$4k.
 */

import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Area,
  AreaChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  ArrowRight,
  Brain,
  ChevronLeft,
  ChevronRight,
  Coins,
  Compass,
  Leaf,
  Radio,
  Rocket,
  Shield,
  Sparkles,
  SlidersHorizontal,
  TrendingUp,
  Wallet,
  Zap,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { useChartTokens } from '@/theme/tokens';
import { cn } from '@/lib/utils';
import { formatCurrency, formatNumber } from '@/lib/format';

/* ================================================================== */
/* 1. The click-through tour                                           */
/* ================================================================== */

interface Step {
  icon: JSX.Element;
  eyebrow: string;
  title: string;
  body: JSX.Element;
}

const STEPS: Step[] = [
  {
    icon: <Sparkles className="h-5 w-5" />,
    eyebrow: 'What is this?',
    title: 'A safe place to learn investing',
    body: (
      <>
        GiffMeMoney is a <b>practice</b> money app. You get pretend cash, put it to work, and watch what
        happens — with <b>zero real money at risk</b>. Think of it as a flight simulator for investing: make
        every mistake here, for free, before it ever costs you a rupee.
      </>
    ),
  },
  {
    icon: <Coins className="h-5 w-5" />,
    eyebrow: 'The big idea',
    title: 'How money actually grows',
    body: (
      <>
        When you invest, your money buys a tiny piece of something real. It can grow three ways:{' '}
        <b>🐐 the price rises</b> (buy low, sell higher), <b>🌳 it pays you</b> (some companies share
        profits), and <b>♻️ compounding</b> (you earn on last year&apos;s earnings too). The third one is the
        quiet superpower — but it needs time.
      </>
    ),
  },
  {
    icon: <Brain className="h-5 w-5" />,
    eyebrow: 'Who picks for you',
    title: 'Five "brains" vote on every choice',
    body: (
      <>
        Instead of guessing, the app asks <b>five proven methods</b> — is it a good business? which way is the
        crowd moving? has it stretched too far? how do we avoid big losses? what could happen next? — and goes
        with the <b>weighted vote</b>. Like a council of experienced advisors, not one loud opinion.
      </>
    ),
  },
  {
    icon: <SlidersHorizontal className="h-5 w-5" />,
    eyebrow: 'Built for you',
    title: 'Easy mode and Expert mode',
    body: (
      <>
        See the <b>Easy ⇄ Expert</b> switch at the top? <b>Easy</b> keeps things big, calm and in plain words.{' '}
        <b>Expert</b> shows every number and control. Same app, two lenses — flip it any time. New here? Stay
        on Easy.
      </>
    ),
  },
  {
    icon: <Shield className="h-5 w-5" />,
    eyebrow: 'The honest truth',
    title: 'Slow and steady — not get-rich-quick',
    body: (
      <>
        Good investing earns roughly <b>8–10% a year</b> on average, not 100× in a month. Anyone promising
        otherwise is selling a lie. This app is built to tell you the truth and help you grow money the real
        way: <b>small, regular, patient.</b>
      </>
    ),
  },
];

function Tour(): JSX.Element {
  const [i, setI] = useState(0);
  const step = STEPS[i] as Step;
  const last = STEPS.length - 1;

  return (
    <Card className="flex flex-col gap-4 overflow-hidden">
      <div className="flex items-center justify-between">
        <Badge tone="accent" variant="soft" size="sm" icon={<Compass className="h-[0.875rem] w-[0.875rem]" />}>
          2-minute tour
        </Badge>
        <span className="text-xs tnum text-muted">
          {i + 1} / {STEPS.length}
        </span>
      </div>

      <div key={i} className="flex animate-fade-in flex-col gap-3 sm:flex-row sm:items-start sm:gap-5">
        <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-2xl bg-primary/12 text-primary">
          {step.icon}
        </div>
        <div className="min-w-0">
          <div className="text-[0.6875rem] font-bold uppercase tracking-[0.15em] text-primary">
            {step.eyebrow}
          </div>
          <h3 className="mt-0.5 text-xl font-bold tracking-tight text-text">{step.title}</h3>
          <p className="mt-2 text-[0.9375rem] leading-relaxed text-muted">{step.body}</p>
        </div>
      </div>

      <div className="flex items-center justify-between gap-3 border-t border-border pt-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setI((v) => Math.max(0, v - 1))}
          disabled={i === 0}
          leftIcon={<ChevronLeft className="h-4 w-4" />}
        >
          Back
        </Button>

        {/* Progress dots */}
        <div className="flex items-center gap-1.5">
          {STEPS.map((_, k) => (
            <button
              key={k}
              type="button"
              aria-label={`Go to step ${k + 1}`}
              onClick={() => setI(k)}
              className={cn(
                'h-2 rounded-full transition-all',
                k === i ? 'w-5 bg-primary' : 'w-2 bg-border hover:bg-muted',
              )}
            />
          ))}
        </div>

        {i < last ? (
          <Button variant="primary" size="sm" onClick={() => setI((v) => Math.min(last, v + 1))} rightIcon={<ChevronRight className="h-4 w-4" />}>
            Next
          </Button>
        ) : (
          <Link to="/invest">
            <Button variant="primary" size="sm" rightIcon={<ArrowRight className="h-4 w-4" />}>
              Try it
            </Button>
          </Link>
        )}
      </div>
    </Card>
  );
}

/* ================================================================== */
/* 2. Compounding playground (the signature, hands-on moment)          */
/* ================================================================== */

const ANNUAL_RATE = 0.08; // a realistic long-run average — NOT a promise

function useCompounding(start: number, monthly: number, years: number) {
  return useMemo(() => {
    const r = ANNUAL_RATE / 12;
    const series: { year: number; balance: number; paidIn: number }[] = [];
    let bal = start;
    let paid = start;
    series.push({ year: 0, balance: bal, paidIn: paid });
    for (let m = 1; m <= years * 12; m += 1) {
      bal = bal * (1 + r) + monthly;
      paid += monthly;
      if (m % 12 === 0) series.push({ year: m / 12, balance: bal, paidIn: paid });
    }
    const contributed = start + monthly * 12 * years;
    return { fv: bal, contributed, growth: Math.max(0, bal - contributed), series };
  }, [start, monthly, years]);
}

function Range({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format: (v: number) => string;
}): JSX.Element {
  return (
    <label className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-muted">{label}</span>
        <span className="text-sm font-bold tnum text-text">{format(value)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-surface-2 accent-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT"
      />
    </label>
  );
}

function CompoundingPlayground(): JSX.Element {
  const tokens = useChartTokens();
  const [start, setStart] = useState(20);
  const [monthly, setMonthly] = useState(20);
  const [years, setYears] = useState(10);
  const { fv, contributed, growth, series } = useCompounding(start, monthly, years);

  return (
    <Card className="flex flex-col gap-4">
      <CardHeader>
        <div>
          <CardTitle icon={<TrendingUp className="h-4 w-4" />}>Try it: watch money grow</CardTitle>
          <CardDescription>Drag the sliders. This is the superpower of small, regular investing.</CardDescription>
        </div>
        <Badge tone="primary" variant="soft" size="sm">
          interactive
        </Badge>
      </CardHeader>

      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <div className="flex flex-col justify-between gap-4">
          <div className="flex flex-col gap-4">
            <Range label="Start with" value={start} min={0} max={500} step={5} onChange={setStart} format={(v) => formatCurrency(v, { maximumFractionDigits: 0, minimumFractionDigits: 0 })} />
            <Range label="Add each month" value={monthly} min={0} max={200} step={5} onChange={setMonthly} format={(v) => formatCurrency(v, { maximumFractionDigits: 0, minimumFractionDigits: 0 })} />
            <Range label="For this many years" value={years} min={1} max={30} step={1} onChange={setYears} format={(v) => `${v} ${v === 1 ? 'year' : 'years'}`} />
          </div>

          <div className="rounded-2xl border border-primary/30 bg-primary/[0.06] p-4">
            <div className="text-xs font-medium text-muted">You could have about</div>
            <div className="text-3xl font-bold tracking-tight tnum text-primary">
              {formatCurrency(fv, { maximumFractionDigits: 0, minimumFractionDigits: 0 })}
            </div>
            <div className="mt-1 text-[0.75rem] leading-relaxed text-muted">
              You put in {formatCurrency(contributed, { maximumFractionDigits: 0, minimumFractionDigits: 0 })};
              growth added{' '}
              <span className="font-semibold text-success">
                {formatCurrency(growth, { maximumFractionDigits: 0, minimumFractionDigits: 0 })}
              </span>
              .
            </div>
          </div>
        </div>

        <div className="min-h-[14rem]">
          <ResponsiveContainer width="100%" height="100%" minHeight={224}>
            <AreaChart data={series} margin={{ top: 8, right: 8, bottom: 0, left: -8 }}>
              <defs>
                <linearGradient id="grow" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={tokens.primary} stopOpacity={0.28} />
                  <stop offset="100%" stopColor={tokens.primary} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <XAxis dataKey="year" tick={{ fill: tokens.muted, fontSize: 11 }} tickLine={false} axisLine={{ stroke: tokens.grid }} tickFormatter={(v: number) => `${v}y`} minTickGap={20} />
              <YAxis tick={{ fill: tokens.muted, fontSize: 11 }} tickLine={false} axisLine={false} width={44} tickFormatter={(v: number) => `$${formatNumber(v, 0)}`} />
              <Tooltip
                cursor={{ stroke: tokens.muted, strokeDasharray: '3 3' }}
                formatter={(v: number) => [formatCurrency(v, { maximumFractionDigits: 0, minimumFractionDigits: 0 }), 'Balance']}
                labelFormatter={(l) => `Year ${l}`}
                contentStyle={{ background: tokens.surface, border: `1px solid ${tokens.grid}`, borderRadius: '0.75rem', fontSize: '0.75rem' }}
              />
              <Area type="monotone" dataKey="balance" stroke={tokens.primary} strokeWidth={2.5} fill="url(#grow)" isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <p className="text-[0.6875rem] leading-relaxed text-muted">
        Assumes a ~8%/year average (a typical long-run market figure) — not a guarantee. Real years go up and
        down; this shows the <i>shape</i> of patience, which is what actually builds wealth.
      </p>
    </Card>
  );
}

/* ================================================================== */
/* 3. Myth busters (flip to reveal)                                   */
/* ================================================================== */

interface Myth {
  myth: string;
  truth: string;
}

const MYTHS: Myth[] = [
  { myth: '“$20 can become $4,000 in a month.”', truth: 'That would need +19% every single day. Nobody does that — it’s a lottery ticket, and the app flags it as impossible.' },
  { myth: '“Trading super fast makes more money.”', truth: 'Every trade pays a fee. Faster usually means LESS after costs — our Speed Lab proves it with real numbers.' },
  { myth: '“A good app can predict the market.”', truth: 'No one predicts it reliably. The honest move is to show a range of outcomes and manage risk — which is what we do.' },
  { myth: '“Investing is only for rich people.”', truth: 'You can start with $20 and add a little each month. Time does the heavy lifting, not the size of your first deposit.' },
];

function MythCard({ m }: { m: Myth }): JSX.Element {
  const [revealed, setRevealed] = useState(false);
  return (
    <button
      type="button"
      onClick={() => setRevealed((v) => !v)}
      className={cn(
        'flex min-h-[6.5rem] flex-col items-start gap-1.5 rounded-2xl border p-4 text-left transition-colors',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
        revealed ? 'border-success/40 bg-success/[0.06]' : 'border-border bg-surface-2/40 hover:border-muted',
      )}
    >
      {revealed ? (
        <>
          <Badge tone="success" variant="soft" size="sm">the truth</Badge>
          <p className="text-[0.875rem] leading-relaxed text-text">{m.truth}</p>
        </>
      ) : (
        <>
          <Badge tone="danger" variant="soft" size="sm">myth</Badge>
          <p className="text-[0.9375rem] font-semibold leading-relaxed text-text">{m.myth}</p>
          <span className="mt-auto text-[0.6875rem] text-muted">tap to reveal the truth →</span>
        </>
      )}
    </button>
  );
}

/* ================================================================== */
/* 4. Jargon buster                                                   */
/* ================================================================== */

const JARGON: { term: string; plain: string }[] = [
  { term: 'Portfolio', plain: 'All the things you own, together — your whole basket.' },
  { term: 'Diversify', plain: 'Don’t put all eggs in one basket; spread across many.' },
  { term: 'Volatility', plain: 'How wildly a price jumps around. High = bumpy ride.' },
  { term: 'Dividend', plain: 'A small cash payment some companies give their owners.' },
  { term: 'Stop-loss', plain: 'An auto-sell that limits how much one bet can lose.' },
  { term: 'Composite score', plain: 'How strongly all our methods agree on something.' },
];

function JargonBuster(): JSX.Element {
  return (
    <Card className="flex flex-col gap-3">
      <CardTitle icon={<Brain className="h-4 w-4" />}>Jargon buster</CardTitle>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
        {JARGON.map((j) => (
          <div key={j.term} className="rounded-xl border border-border bg-surface-2/40 p-3">
            <div className="text-sm font-semibold text-text">{j.term}</div>
            <div className="mt-0.5 text-[0.8125rem] leading-relaxed text-muted">{j.plain}</div>
          </div>
        ))}
      </div>
    </Card>
  );
}

/* ================================================================== */
/* 5. Pick your path                                                  */
/* ================================================================== */

interface Path {
  to: string;
  icon: JSX.Element;
  title: string;
  body: string;
  tone: string;
}

const PATHS: Path[] = [
  { to: '/invest', icon: <Wallet className="h-5 w-5" />, title: 'Invest your practice money', body: 'Add pretend funds and build your first basket.', tone: 'bg-primary/12 text-primary' },
  { to: '/', icon: <Sparkles className="h-5 w-5" />, title: 'See today’s top idea', body: 'The dashboard picks one clear idea, in plain words.', tone: 'bg-accent/12 text-accent' },
  { to: '/real-time', icon: <Radio className="h-5 w-5" />, title: 'Watch it live', body: 'Press start and watch money spread and rotate.', tone: 'bg-success/12 text-success' },
  { to: '/speed-lab', icon: <Zap className="h-5 w-5" />, title: 'Why fast trading loses', body: 'A 1-click experiment that proves it with numbers.', tone: 'bg-warning/14 text-warning' },
];

function PickYourPath(): JSX.Element {
  return (
    <div className="space-y-3">
      <h2 className="flex items-center gap-2 text-sm font-semibold tracking-tight text-text">
        <Rocket className="h-4 w-4 text-primary" aria-hidden />
        Ready? Pick where to start
      </h2>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {PATHS.map((p) => (
          <Link
            key={p.to}
            to={p.to}
            className="group flex flex-col gap-2 rounded-2xl border border-border bg-surface p-4 shadow-soft transition-shadow hover:shadow-card focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT"
          >
            <span className={cn('flex h-10 w-10 items-center justify-center rounded-xl', p.tone)}>{p.icon}</span>
            <div className="text-sm font-semibold tracking-tight text-text">{p.title}</div>
            <div className="text-[0.75rem] leading-relaxed text-muted">{p.body}</div>
            <span className="mt-1 inline-flex items-center gap-1 text-[0.75rem] font-medium text-primary">
              Open <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
            </span>
          </Link>
        ))}
      </div>
    </div>
  );
}

/* ================================================================== */
/* Page                                                               */
/* ================================================================== */

export default function GuidePage(): JSX.Element {
  return (
    <div className="mx-auto max-w-4xl space-y-5">
      {/* Hero */}
      <div className="flex flex-col gap-2">
        <Badge tone="primary" variant="soft" size="sm" icon={<Leaf className="h-[0.875rem] w-[0.875rem]" />}>
          Start here
        </Badge>
        <h1 className="text-balance text-2xl font-bold tracking-tight text-text lg:text-3xl">
          New to investing? Let&apos;s make this make sense.
        </h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted">
          No finance background needed. Take the quick tour, play with the money grower, and start with pretend
          cash — so you learn how everything works before a single real rupee is ever involved.
        </p>
      </div>

      <Tour />
      <CompoundingPlayground />

      {/* Myth busters */}
      <div className="space-y-3">
        <h2 className="flex items-center gap-2 text-sm font-semibold tracking-tight text-text">
          <Shield className="h-4 w-4 text-primary" aria-hidden />
          Myth busters
          <span className="text-xs font-normal text-muted">(tap a card)</span>
        </h2>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {MYTHS.map((m) => (
            <MythCard key={m.myth} m={m} />
          ))}
        </div>
      </div>

      <JargonBuster />
      <PickYourPath />

      <p className="text-center text-[0.6875rem] leading-relaxed text-muted">
        Everything here is an educational simulation — practice money, $0 real. Not financial advice.
      </p>
    </div>
  );
}
