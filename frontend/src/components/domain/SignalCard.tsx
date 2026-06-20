/**
 * SignalCard — one quant strategy's signal for an asset, rendered as a compact
 * card: name + category, the score (-100..100) with its derived stance, the
 * model confidence, the human-readable formula (monospace), the plain-English
 * rationale, and (when present) a lightweight realized-backtest summary. Colors
 * via semantic tokens only.
 */

import { useId } from 'react';
import { useChartTokens } from '@/theme/tokens';
import { Card } from '@/components/ui/Card';
import { StanceBadge } from '@/components/ui/StanceBadge';
import { Badge } from '@/components/ui/Badge';
import type { StrategySignal } from '@/lib/types';
import { formatPct, formatProbability, formatRatio } from '@/lib/format';
import { clamp, cn, stanceTextColor, stanceTone } from '@/lib/utils';

export interface SignalCardProps {
  signal: StrategySignal;
  className?: string;
}

/** A slim horizontal score meter centered at 0, spanning -100..100. */
function ScoreMeter({ score }: { score: number }): JSX.Element {
  const tokens = useChartTokens();
  const id = useId();
  const v = clamp(score, -100, 100);
  const tone = stanceTone(score >= 0 ? 'BUY' : 'SELL');
  const color = tone === 'positive' ? tokens.up : tokens.down;
  // Map -100..100 to a 0..100% fill measured from the centre line.
  const halfPct = (Math.abs(v) / 100) * 50;
  const left = v >= 0 ? 50 : 50 - halfPct;
  return (
    <div
      className="relative h-1.5 w-full overflow-hidden rounded-full bg-surface-2"
      role="meter"
      aria-valuemin={-100}
      aria-valuemax={100}
      aria-valuenow={Math.round(v)}
      aria-labelledby={id}
    >
      {/* centre tick */}
      <span className="absolute left-1/2 top-0 h-full w-px -translate-x-1/2 bg-border" aria-hidden />
      <span
        className="absolute top-0 h-full rounded-full transition-[width,left] duration-300"
        style={{ left: `${left}%`, width: `${halfPct}%`, backgroundColor: color }}
        aria-hidden
      />
    </div>
  );
}

function BacktestRow({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div className="flex flex-col">
      <span className="text-[10px] text-muted">{label}</span>
      <span className="text-xs font-semibold tnum text-text">{value}</span>
    </div>
  );
}

export function SignalCard({ signal, className }: SignalCardProps): JSX.Element {
  const scoreId = useId();
  const bt = signal.backtest;

  return (
    <Card muted className={cn('flex h-full flex-col gap-2.5 p-3', className)}>
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h4 className="truncate text-sm font-semibold tracking-tight text-text" title={signal.strategyName}>
            {signal.strategyName}
          </h4>
          <Badge tone="neutral" size="sm" className="mt-1">
            {signal.category}
          </Badge>
        </div>
        <StanceBadge stance={signal.stance} size="sm" />
      </div>

      <div className="flex items-center justify-between gap-2">
        <span id={scoreId} className="text-[11px] font-medium text-muted">
          Score
        </span>
        <div className="flex items-center gap-2">
          <span className={cn('text-sm font-semibold tnum', stanceTextColor(signal.stance))}>
            {signal.score > 0 ? '+' : ''}
            {Math.round(signal.score)}
          </span>
          <span className="text-[11px] tnum text-muted">{formatProbability(signal.confidence)} conf.</span>
        </div>
      </div>
      <ScoreMeter score={signal.score} />

      <code className="block overflow-x-auto rounded-lg bg-surface px-2 py-1.5 text-[11px] leading-snug text-muted no-scrollbar">
        {signal.formula}
      </code>

      <p className="text-xs leading-relaxed text-muted">{signal.rationale}</p>

      {bt && (
        <div className="mt-auto grid grid-cols-3 gap-2 border-t border-border pt-2.5">
          <BacktestRow label="Sharpe" value={formatRatio(bt.sharpe)} />
          <BacktestRow label="CAGR" value={formatPct(bt.cagr * 100, { sign: true, digits: 1 })} />
          <BacktestRow label="Max DD" value={formatPct(-Math.abs(bt.maxDrawdown) * 100, { digits: 1 })} />
        </div>
      )}
    </Card>
  );
}
