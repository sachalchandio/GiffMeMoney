/**
 * AttributionTable — per-sleeve realized contribution for one simulated bot run
 * ({@link SleeveAttribution}[]). Columns: sleeve, realized P&L, contribution
 * (signed % of the run's total P&L), win-rate and trade count. The bot's BEST
 * contributor is highlighted green and the WORST red — so it's obvious "which
 * trading did the best and which did the worst".
 *
 * Simulated paper-trading only — realized P&L is synthetic. Semantic tokens only,
 * no hardcoded hex. Light/dark aware, horizontally scrollable on narrow widths.
 */

import { ArrowDownRight, ArrowUpRight, Minus } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import type { SleeveAttribution, BotVerdict } from '@/lib/types';
import { formatCurrency, formatPct, formatNumber } from '@/lib/format';
import { cn, changeTextColor } from '@/lib/utils';

export interface AttributionTableProps {
  attribution: SleeveAttribution[];
  /** Header label for the first column (default: "Sleeve"). */
  keyLabel?: string;
  className?: string;
}

const HEAD_CELL = 'px-3 py-2 text-[11px] font-semibold uppercase tracking-wide text-muted';
const NUM_CELL = 'px-3 py-2 text-right tnum';

function NumHead({ children }: { children: React.ReactNode }): JSX.Element {
  return <th className={cn(HEAD_CELL, 'text-right')}>{children}</th>;
}

function verdictBadge(verdict: BotVerdict): JSX.Element | null {
  if (verdict === 'best') {
    return (
      <Badge tone="success" size="sm" icon={<ArrowUpRight className="h-3 w-3" aria-hidden />}>
        Best
      </Badge>
    );
  }
  if (verdict === 'worst') {
    return (
      <Badge tone="danger" size="sm" icon={<ArrowDownRight className="h-3 w-3" aria-hidden />}>
        Worst
      </Badge>
    );
  }
  return null;
}

/** Row tint for the best (green) / worst (red) sleeve. */
function rowTint(verdict: BotVerdict): string {
  if (verdict === 'best') return 'bg-success/8';
  if (verdict === 'worst') return 'bg-danger/8';
  return '';
}

export function AttributionTable({
  attribution,
  keyLabel = 'Sleeve',
  className,
}: AttributionTableProps): JSX.Element {
  // Defensive sort: best → worst by contribution (the backend already orders
  // this, but a stable display order shouldn't depend on that).
  const rows = [...attribution].sort((a, b) => b.contributionPct - a.contributionPct);

  return (
    <div className={cn('overflow-x-auto', className)}>
      <table className="w-full min-w-[560px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-border text-left">
            <th className={HEAD_CELL}>{keyLabel}</th>
            <NumHead>Realized P&amp;L</NumHead>
            <NumHead>Contribution</NumHead>
            <NumHead>Win rate</NumHead>
            <NumHead>Trades</NumHead>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={row.key}
              className={cn('border-b border-border/70 transition-colors', rowTint(row.verdict))}
            >
              <td className="px-3 py-2">
                <div className="flex items-center gap-2">
                  <span className="text-[13px] font-medium text-text">{row.key}</span>
                  {verdictBadge(row.verdict)}
                </div>
              </td>
              <td className={cn(NUM_CELL, 'font-medium', changeTextColor(row.realizedPnl))}>
                {row.realizedPnl >= 0 ? '+' : '−'}
                {formatCurrency(Math.abs(row.realizedPnl))}
              </td>
              <td className={cn(NUM_CELL, changeTextColor(row.contributionPct))}>
                {formatPct(row.contributionPct, { sign: true, digits: 1 })}
              </td>
              <td className={cn(NUM_CELL, 'text-muted')}>
                {formatPct(row.winRate * 100, { digits: 0 })}
              </td>
              <td className={cn(NUM_CELL, 'text-muted')}>{formatNumber(row.trades, 0)}</td>
            </tr>
          ))}

          {rows.length === 0 && (
            <tr>
              <td colSpan={5} className="px-3 py-8 text-center text-sm text-muted">
                <span className="inline-flex items-center gap-1.5">
                  <Minus className="h-4 w-4 opacity-60" aria-hidden />
                  No sleeve attribution for this run.
                </span>
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
