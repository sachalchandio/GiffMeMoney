/**
 * RiskMetricGrid — the annualized risk metrics for an asset laid out as a dense
 * responsive grid of labelled tiles. Volatility / VaR / CVaR / max-drawdown are
 * decimal fractions (rendered as percents); beta / Sharpe / Sortino / Calmar are
 * ratios. Risk-tone hints (good/neutral/bad) are derived per-metric. Colors come
 * from semantic tokens only.
 */

import type { ReactNode } from 'react';
import { Tooltip } from '@/components/ui/Tooltip';
import type { RiskMetrics } from '@/lib/types';
import { formatFractionPct, formatRatio } from '@/lib/format';
import { cn } from '@/lib/utils';

export interface RiskMetricGridProps {
  metrics: RiskMetrics;
  className?: string;
}

type Tone = 'good' | 'neutral' | 'bad';

interface MetricTile {
  label: string;
  value: string;
  hint: string;
  tone: Tone;
}

const TONE_VALUE: Record<Tone, string> = {
  good: 'text-success',
  bad: 'text-danger',
  neutral: 'text-text',
};

/** Higher is better (Sharpe/Sortino/Calmar). */
function highGoodTone(v: number, good: number, bad: number): Tone {
  if (v >= good) return 'good';
  if (v <= bad) return 'bad';
  return 'neutral';
}

/** Lower magnitude is better (vol/VaR/CVaR/drawdown). */
function lowGoodTone(v: number, good: number, bad: number): Tone {
  const m = Math.abs(v);
  if (m <= good) return 'good';
  if (m >= bad) return 'bad';
  return 'neutral';
}

function buildTiles(m: RiskMetrics): MetricTile[] {
  return [
    {
      label: 'Annual vol',
      value: formatFractionPct(m.annualVol, { digits: 1 }),
      hint: 'Annualized standard deviation of returns.',
      tone: lowGoodTone(m.annualVol, 0.2, 0.5),
    },
    {
      label: 'Beta',
      value: formatRatio(m.beta),
      hint: 'Sensitivity to the broad market (1.0 = market).',
      tone: 'neutral',
    },
    {
      label: 'Sharpe',
      value: formatRatio(m.sharpe),
      hint: 'Excess return per unit of total risk (annualized).',
      tone: highGoodTone(m.sharpe, 1, 0),
    },
    {
      label: 'Sortino',
      value: formatRatio(m.sortino),
      hint: 'Excess return per unit of downside risk (annualized).',
      tone: highGoodTone(m.sortino, 1.5, 0),
    },
    {
      label: 'Calmar',
      value: formatRatio(m.calmar),
      hint: 'CAGR divided by the magnitude of max drawdown.',
      tone: highGoodTone(m.calmar, 1, 0),
    },
    {
      label: 'Max drawdown',
      value: formatFractionPct(-Math.abs(m.maxDrawdown), { digits: 1 }),
      hint: 'Largest peak-to-trough decline.',
      tone: lowGoodTone(m.maxDrawdown, 0.15, 0.4),
    },
    {
      label: 'VaR 95%',
      value: formatFractionPct(-Math.abs(m.var95), { digits: 1 }),
      hint: 'Daily 95% Value-at-Risk (expected worst loss, normal days).',
      tone: lowGoodTone(m.var95, 0.02, 0.06),
    },
    {
      label: 'CVaR 95%',
      value: formatFractionPct(-Math.abs(m.cvar95), { digits: 1 }),
      hint: 'Expected loss in the worst 5% of days (tail risk).',
      tone: lowGoodTone(m.cvar95, 0.03, 0.08),
    },
  ];
}

function Tile({ tile }: { tile: MetricTile }): JSX.Element {
  return (
    <Tooltip content={tile.hint as ReactNode}>
      <div className="flex w-full flex-col gap-0.5 rounded-xl border border-border bg-surface-2 px-3 py-2.5 text-left">
        <span className="text-[0.6875rem] font-medium text-muted">{tile.label}</span>
        <span className={cn('text-base font-semibold tracking-tight tnum', TONE_VALUE[tile.tone])}>
          {tile.value}
        </span>
      </div>
    </Tooltip>
  );
}

export function RiskMetricGrid({ metrics, className }: RiskMetricGridProps): JSX.Element {
  const tiles = buildTiles(metrics);
  return (
    <div className={cn('grid grid-cols-2 gap-2 sm:grid-cols-4', className)}>
      {tiles.map((t) => (
        <Tile key={t.label} tile={t} />
      ))}
    </div>
  );
}
