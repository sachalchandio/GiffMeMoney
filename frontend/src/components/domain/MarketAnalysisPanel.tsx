/**
 * MarketAnalysisPanel — the market context the simulated bot acted on: the
 * regime at the end of the run (with a compact regime timeline strip showing how
 * it shifted across rebalances) plus the "top signals" — the sleeves the bot
 * leaned into most, surfaced from the recent buy trades. This explains *why* the
 * bot tilted where it did, in plain terms.
 *
 * Simulated paper-trading on synthetic data. Semantic tokens only — no hardcoded
 * hex. Light/dark aware.
 */

import { useMemo } from 'react';
import { Activity, TrendingDown, TrendingUp, Sparkles } from 'lucide-react';
import type { ReactNode } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge, type BadgeTone } from '@/components/ui/Badge';
import { Tooltip } from '@/components/ui/Tooltip';
import type { BotRegime, BotTrade } from '@/lib/types';
import { formatCurrency } from '@/lib/format';
import { cn } from '@/lib/utils';

export interface MarketAnalysisPanelProps {
  /** The regime label at each rebalance, in order (oldest → newest). */
  regimeTimeline: BotRegime[];
  /** Every simulated trade — recent buys reveal the signals the bot acted on. */
  trades: BotTrade[];
  /** How many top signals to surface (default: 5). */
  topN?: number;
  className?: string;
}

function regimeTone(regime: BotRegime): BadgeTone {
  switch (regime) {
    case 'bull':
      return 'success';
    case 'bear':
      return 'danger';
    case 'neutral':
    default:
      return 'warning';
  }
}

function regimeIcon(regime: BotRegime): ReactNode {
  switch (regime) {
    case 'bull':
      return <TrendingUp className="h-3 w-3" aria-hidden />;
    case 'bear':
      return <TrendingDown className="h-3 w-3" aria-hidden />;
    case 'neutral':
    default:
      return <Activity className="h-3 w-3" aria-hidden />;
  }
}

function regimeLabel(regime: BotRegime): string {
  switch (regime) {
    case 'bull':
      return 'Bull';
    case 'bear':
      return 'Bear';
    case 'neutral':
    default:
      return 'Neutral';
  }
}

/** Tailwind dot color for the timeline strip (semantic tokens). */
function regimeDotClass(regime: BotRegime): string {
  switch (regime) {
    case 'bull':
      return 'bg-success';
    case 'bear':
      return 'bg-danger';
    case 'neutral':
    default:
      return 'bg-warning';
  }
}

interface Signal {
  symbol: string;
  strategy: string;
  amount: number;
}

export function MarketAnalysisPanel({
  regimeTimeline,
  trades,
  topN = 5,
  className,
}: MarketAnalysisPanelProps): JSX.Element {
  const current: BotRegime =
    regimeTimeline.length > 0 ? (regimeTimeline[regimeTimeline.length - 1] as BotRegime) : 'neutral';

  // Top signals = the sleeves the bot most recently bought into, ranked by the
  // capital it committed (the momentum tilt made visible). Aggregate buys per
  // (symbol, strategy) and take the largest.
  const signals = useMemo(() => {
    const agg = new Map<string, Signal>();
    for (const tr of trades) {
      if (tr.side !== 'buy') continue;
      const id = `${tr.symbol}|${tr.strategy}`;
      const prev = agg.get(id);
      if (prev) prev.amount += tr.amount;
      else agg.set(id, { symbol: tr.symbol, strategy: tr.strategy, amount: tr.amount });
    }
    return Array.from(agg.values())
      .sort((a, b) => b.amount - a.amount)
      .slice(0, topN);
  }, [trades, topN]);

  return (
    <Card className={cn('flex flex-col gap-3', className)}>
      <CardHeader>
        <CardTitle icon={<Sparkles className="h-4 w-4" />}>Market analysis</CardTitle>
        <Tooltip content="Detected market regime at the most recent rebalance.">
          <Badge tone={regimeTone(current)} icon={regimeIcon(current)}>
            {regimeLabel(current)} regime
          </Badge>
        </Tooltip>
      </CardHeader>

      {/* Regime timeline strip across rebalances. */}
      {regimeTimeline.length > 0 && (
        <div>
          <div className="mb-1 flex items-center justify-between">
            <span className="text-[0.6875rem] font-medium uppercase tracking-wide text-muted">
              Regime over time
            </span>
            <span className="text-[0.625rem] text-muted">{regimeTimeline.length} rebalances</span>
          </div>
          <div className="flex flex-wrap gap-1">
            {regimeTimeline.map((r, i) => (
              <Tooltip key={i} content={`Rebalance ${i + 1}: ${regimeLabel(r)}`}>
                <span
                  className={cn('h-2.5 w-2.5 rounded-sm', regimeDotClass(r))}
                  aria-label={`Rebalance ${i + 1}: ${regimeLabel(r)} regime`}
                />
              </Tooltip>
            ))}
          </div>
        </div>
      )}

      {/* Top signals the bot acted on. */}
      <div>
        <span className="mb-1.5 block text-[0.6875rem] font-medium uppercase tracking-wide text-muted">
          Top signals acted on
        </span>
        {signals.length === 0 ? (
          <p className="text-xs text-muted">No buy signals recorded in this run.</p>
        ) : (
          <ul className="flex flex-col divide-y divide-border">
            {signals.map((s) => (
              <li key={`${s.symbol}-${s.strategy}`} className="flex items-center gap-2 py-2">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-primary/12 text-[0.625rem] font-semibold text-primary">
                  {s.symbol.slice(0, 3)}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm font-medium text-text">{s.symbol}</span>
                  </div>
                  <p className="line-clamp-1 text-[0.6875rem] text-muted">{s.strategy}</p>
                </div>
                <span className="shrink-0 text-xs font-medium tnum text-text">
                  {formatCurrency(s.amount)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Card>
  );
}
