/**
 * LeaderboardTable — per-asset strategy leaderboard, ranking the catalog's
 * strategies by *realized* backtest performance for one symbol
 * ({@link StrategyLeaderboard}). Columns: rank, strategy (+ category), Sharpe,
 * CAGR, total return, max drawdown, Calmar, win-rate, trades. A pinned
 * buy-&-hold benchmark row gives context. Clicking a row selects that strategy.
 *
 * Returns/drawdowns are decimals on the wire → rendered with `formatFractionPct`;
 * Sharpe/Calmar are ratios. Colors via semantic tokens only. No `any`.
 */

import { Trophy } from 'lucide-react';
import { Badge } from '@/components/ui/Badge';
import { categoryTone } from './StrategyCard';
import { cn } from '@/lib/utils';
import { changeTextColor } from '@/lib/utils';
import { formatFractionPct, formatRatio, formatNumber } from '@/lib/format';
import type {
  BacktestMetricsDTO,
  StrategyLeaderboard,
  StrategyLeaderboardEntry,
} from '@/lib/types';

export interface LeaderboardTableProps {
  leaderboard: StrategyLeaderboard;
  /** Highlight the row matching this strategy id. */
  selectedId?: string | undefined;
  /** Row click → select that strategy in the parent. */
  onSelect?: (id: string) => void;
  className?: string;
}

const HEAD_CELL = 'px-3 py-2 text-[0.6875rem] font-semibold uppercase tracking-wide text-muted';
const NUM_CELL = 'px-3 py-2 text-right tnum';

/** Right-aligned numeric header. */
function NumHead({ children }: { children: React.ReactNode }): JSX.Element {
  return <th className={cn(HEAD_CELL, 'text-right')}>{children}</th>;
}

export function LeaderboardTable({
  leaderboard,
  selectedId,
  onSelect,
  className,
}: LeaderboardTableProps): JSX.Element {
  const { entries, benchmark } = leaderboard;

  return (
    <div className={cn('overflow-x-auto', className)}>
      <table className="w-full min-w-[40rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-border text-left">
            <th className={cn(HEAD_CELL, 'w-10')}>#</th>
            <th className={HEAD_CELL}>Strategy</th>
            <NumHead>Sharpe</NumHead>
            <NumHead>CAGR</NumHead>
            <NumHead>Total</NumHead>
            <NumHead>Max DD</NumHead>
            <NumHead>Calmar</NumHead>
            <NumHead>Win</NumHead>
            <NumHead>Trades</NumHead>
          </tr>
        </thead>
        <tbody>
          {/* Buy & hold benchmark, pinned for context */}
          <BenchmarkRow benchmark={benchmark} />

          {entries.map((entry) => (
            <LeaderRow
              key={entry.strategyId}
              entry={entry}
              selected={entry.strategyId === selectedId}
              onSelect={onSelect}
            />
          ))}

          {entries.length === 0 && (
            <tr>
              <td colSpan={9} className="px-3 py-8 text-center text-sm text-muted">
                No backtest results for this asset yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function BenchmarkRow({ benchmark }: { benchmark: BacktestMetricsDTO }): JSX.Element {
  return (
    <tr className="border-b border-border bg-surface-2/60">
      <td className="px-3 py-2 text-center text-muted">—</td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="text-[0.8125rem] font-medium text-text">Buy &amp; Hold</span>
          <Badge tone="neutral" size="sm" variant="outline">
            Benchmark
          </Badge>
        </div>
      </td>
      <td className={cn(NUM_CELL, 'text-text')}>{formatRatio(benchmark.sharpe)}</td>
      <td className={cn(NUM_CELL, changeTextColor(benchmark.cagr))}>
        {formatFractionPct(benchmark.cagr, { sign: true, digits: 1 })}
      </td>
      <td className={cn(NUM_CELL, changeTextColor(benchmark.totalReturn))}>
        {formatFractionPct(benchmark.totalReturn, { sign: true, digits: 1 })}
      </td>
      <td className={cn(NUM_CELL, 'text-danger')}>
        {formatFractionPct(benchmark.maxDrawdown, { digits: 1 })}
      </td>
      <td className={cn(NUM_CELL, 'text-text')}>{formatRatio(benchmark.calmar)}</td>
      <td className={cn(NUM_CELL, 'text-muted')}>
        {formatFractionPct(benchmark.winRate, { digits: 0 })}
      </td>
      <td className={cn(NUM_CELL, 'text-muted')}>—</td>
    </tr>
  );
}

function LeaderRow({
  entry,
  selected,
  onSelect,
}: {
  entry: StrategyLeaderboardEntry;
  selected: boolean;
  onSelect?: (id: string) => void;
}): JSX.Element {
  const interactive = Boolean(onSelect);
  const isPodium = entry.rank <= 3 && entry.supported;

  return (
    <tr
      onClick={interactive ? () => onSelect?.(entry.strategyId) : undefined}
      className={cn(
        'border-b border-border/70 transition-colors',
        interactive && 'cursor-pointer hover:bg-surface-2',
        selected && 'bg-primary/8',
      )}
    >
      <td className="px-3 py-2">
        <span
          className={cn(
            'inline-flex h-5 min-w-5 items-center justify-center rounded-md px-1 text-[0.6875rem] font-semibold tnum',
            isPodium ? 'bg-primary/12 text-primary' : 'text-muted',
          )}
        >
          {isPodium ? <Trophy className="h-3 w-3" aria-hidden /> : entry.rank}
        </span>
      </td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-2">
          <span className="text-[0.8125rem] font-medium text-text">{entry.strategyName}</span>
          <Badge tone={categoryTone(entry.category)} size="sm" variant="soft">
            {entry.category}
          </Badge>
          {!entry.supported && (
            <Badge tone="neutral" size="sm" variant="outline" title="Not bar-by-bar backtestable">
              snapshot
            </Badge>
          )}
        </div>
      </td>
      <td className={cn(NUM_CELL, 'font-medium text-text')}>{formatRatio(entry.sharpe)}</td>
      <td className={cn(NUM_CELL, changeTextColor(entry.cagr))}>
        {formatFractionPct(entry.cagr, { sign: true, digits: 1 })}
      </td>
      <td className={cn(NUM_CELL, changeTextColor(entry.totalReturn))}>
        {formatFractionPct(entry.totalReturn, { sign: true, digits: 1 })}
      </td>
      <td className={cn(NUM_CELL, 'text-danger')}>
        {formatFractionPct(entry.maxDrawdown, { digits: 1 })}
      </td>
      <td className={cn(NUM_CELL, 'text-text')}>{formatRatio(entry.calmar)}</td>
      <td className={cn(NUM_CELL, 'text-muted')}>
        {formatFractionPct(entry.winRate, { digits: 0 })}
      </td>
      <td className={cn(NUM_CELL, 'text-muted')}>
        {entry.supported ? formatNumber(entry.trades, 0) : '—'}
      </td>
    </tr>
  );
}
