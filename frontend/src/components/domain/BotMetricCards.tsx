/**
 * BotMetricCards — the headline realized metrics for one simulated bot run
 * ({@link BotMetrics}) as a responsive grid of tiles: total return, performance
 * vs buy-&-hold, Sharpe, max drawdown and win rate (plus the final paper value).
 * Tone hints color each value good/neutral/bad. Every figure is from a
 * SIMULATION on synthetic data — no real funds.
 *
 * Semantic tokens only — no hardcoded hex. Light/dark aware.
 */

import type { ReactNode } from 'react';
import { Tooltip } from '@/components/ui/Tooltip';
import type { BotMetrics } from '@/lib/types';
import { formatCurrency, formatPct, formatRatio } from '@/lib/format';
import { cn } from '@/lib/utils';

export interface BotMetricCardsProps {
  metrics: BotMetrics;
  /** Show the final paper value tile (default: true). */
  showFinalValue?: boolean;
  className?: string;
}

type Tone = 'good' | 'neutral' | 'bad';

interface Tile {
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

function signTone(v: number): Tone {
  if (v > 0) return 'good';
  if (v < 0) return 'bad';
  return 'neutral';
}

/** Higher is better with good/bad thresholds. */
function highGoodTone(v: number, good: number, bad: number): Tone {
  if (v >= good) return 'good';
  if (v <= bad) return 'bad';
  return 'neutral';
}

function buildTiles(m: BotMetrics, showFinalValue: boolean): Tile[] {
  const tiles: Tile[] = [
    {
      label: 'Total return',
      value: formatPct(m.totalReturnPct, { sign: true, digits: 1 }),
      hint: 'Total simulated return over the run.',
      tone: signTone(m.totalReturnPct),
    },
    {
      label: 'vs Buy & Hold',
      value: formatPct(m.vsBenchmarkPct, { sign: true, digits: 1 }),
      hint: 'Final-value outperformance vs an equal-weight buy-&-hold of the same candidates (percentage points).',
      tone: signTone(m.vsBenchmarkPct),
    },
    {
      label: 'Sharpe',
      value: formatRatio(m.sharpe),
      hint: 'Annualized Sharpe ratio of the bot’s daily simulated returns.',
      tone: highGoodTone(m.sharpe, 1, 0),
    },
    {
      label: 'Max drawdown',
      value: formatPct(-Math.abs(m.maxDrawdownPct), { digits: 1 }),
      hint: 'Worst peak-to-trough decline of the bot’s paper value.',
      tone: highGoodTone(-Math.abs(m.maxDrawdownPct), -15, -40),
    },
    {
      label: 'Win rate',
      value: formatPct(m.winRatePct, { digits: 0 }),
      hint: 'Share of rebalance periods that were profitable.',
      tone: highGoodTone(m.winRatePct, 55, 45),
    },
  ];
  if (showFinalValue) {
    tiles.push({
      label: 'Final value',
      value: formatCurrency(m.finalValue),
      hint: 'The bot’s final total paper value (cash + marked positions).',
      tone: 'neutral',
    });
  }
  return tiles;
}

function MetricTile({ tile }: { tile: Tile }): JSX.Element {
  return (
    <Tooltip content={tile.hint as ReactNode}>
      <div className="flex w-full flex-col gap-0.5 rounded-xl border border-border bg-surface-2 px-3 py-2.5 text-left">
        <span className="text-[0.6875rem] font-medium text-muted">{tile.label}</span>
        <span className={cn('text-lg font-semibold tracking-tight tnum', TONE_VALUE[tile.tone])}>
          {tile.value}
        </span>
      </div>
    </Tooltip>
  );
}

export function BotMetricCards({
  metrics,
  showFinalValue = true,
  className,
}: BotMetricCardsProps): JSX.Element {
  const tiles = buildTiles(metrics, showFinalValue);
  return (
    <div className={cn('grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6', className)}>
      {tiles.map((t) => (
        <MetricTile key={t.label} tile={t} />
      ))}
    </div>
  );
}
