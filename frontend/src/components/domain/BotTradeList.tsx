/**
 * BotTradeList — the simulated auto-trader's paper-trade blotter (newest first).
 * Each row shows a buy/sell icon, the symbol, the sleeve/strategy that motivated
 * the trade, the dollar magnitude, the (simulated) fill price and a relative
 * time. Buys tint primary, sells accent. Every trade here is simulated — no real
 * order was placed.
 *
 * Semantic tokens only — no hardcoded hex. Light/dark aware, scrolls within a
 * capped height for long runs.
 */

import { ArrowDownLeft, ArrowUpRight, ListOrdered } from 'lucide-react';
import type { ReactNode } from 'react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import type { BotSide, BotTrade } from '@/lib/types';
import { formatCurrency, formatPrice, formatRelativeTime } from '@/lib/format';
import { cn } from '@/lib/utils';

export interface BotTradeListProps {
  trades: BotTrade[];
  /** Cap the number of rows rendered (default: 30). Pass 0 for all. */
  limit?: number;
  /** Reference time for the relative timestamps (default: Date.now()). */
  now?: number;
  className?: string;
}

const META: Record<BotSide, { label: string; icon: ReactNode; tint: string }> = {
  buy: { label: 'Buy', icon: <ArrowUpRight className="h-4 w-4" />, tint: 'bg-primary/12 text-primary' },
  sell: { label: 'Sell', icon: <ArrowDownLeft className="h-4 w-4" />, tint: 'bg-accent/12 text-accent' },
};

export function BotTradeList({
  trades,
  limit = 30,
  now,
  className,
}: BotTradeListProps): JSX.Element {
  // Newest first; the backend records in chronological order.
  const ordered = [...trades].sort((a, b) => b.t - a.t);
  const rows = limit && limit > 0 ? ordered.slice(0, limit) : ordered;

  return (
    <Card className={cn('flex flex-col gap-3', className)} flush>
      <div className="p-4 pb-0">
        <CardHeader>
          <CardTitle icon={<ListOrdered className="h-4 w-4" />}>Simulated trades</CardTitle>
          {trades.length > 0 && (
            <span className="text-[11px] text-muted">{trades.length} total</span>
          )}
        </CardHeader>
      </div>

      {rows.length === 0 ? (
        <div className="px-4 pb-6 pt-2 text-center text-xs text-muted">
          No trades in this simulated run.
        </div>
      ) : (
        <ul className="flex max-h-[28rem] flex-col divide-y divide-border overflow-y-auto">
          {rows.map((tr, i) => {
            const meta = META[tr.side];
            return (
              <li key={`${tr.t}-${tr.symbol}-${i}`} className="flex items-center gap-3 px-4 py-3">
                <span
                  className={cn(
                    'flex h-9 w-9 shrink-0 items-center justify-center rounded-xl',
                    meta.tint,
                  )}
                >
                  {meta.icon}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-text">{meta.label}</span>
                    <Badge tone="neutral" size="sm">
                      {tr.symbol}
                    </Badge>
                  </div>
                  <p className="line-clamp-1 text-[11px] text-muted">
                    {tr.strategy} · @ {formatPrice(tr.price)} · {formatRelativeTime(tr.t, now)}
                  </p>
                </div>
                <span
                  className={cn(
                    'shrink-0 text-sm font-semibold tnum',
                    tr.side === 'buy' ? 'text-primary' : 'text-accent',
                  )}
                >
                  {formatCurrency(tr.amount)}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}
