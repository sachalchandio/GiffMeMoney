/**
 * TransactionList — the account ledger (newest first). Each entry shows a typed
 * icon (deposit / withdrawal / buy / sell), an amount signed by its cash-flow
 * direction, the symbol (for trades), a note, a status pill and a relative time.
 * Colors via semantic tokens only.
 */

import type { ReactNode } from 'react';
import { ArrowDownLeft, ArrowUpRight, Minus, Plus, Receipt } from 'lucide-react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Skeleton } from '@/components/ui/Skeleton';
import type { Transaction, TxnType } from '@/lib/types';
import { formatCurrency, formatRelativeTime } from '@/lib/format';
import { cn } from '@/lib/utils';

export interface TransactionListProps {
  transactions: Transaction[];
  loading?: boolean;
  /** Cap the number of rows rendered (default: all). */
  limit?: number;
  className?: string;
}

const META: Record<TxnType, { label: string; icon: ReactNode; tint: string; sign: 1 | -1 }> = {
  deposit: { label: 'Deposit', icon: <Plus className="h-4 w-4" />, tint: 'bg-success/12 text-success', sign: 1 },
  withdrawal: { label: 'Withdrawal', icon: <Minus className="h-4 w-4" />, tint: 'bg-danger/12 text-danger', sign: -1 },
  buy: { label: 'Buy', icon: <ArrowUpRight className="h-4 w-4" />, tint: 'bg-primary/12 text-primary', sign: -1 },
  sell: { label: 'Sell', icon: <ArrowDownLeft className="h-4 w-4" />, tint: 'bg-accent/12 text-accent', sign: 1 },
};

export function TransactionList({ transactions, loading = false, limit, className }: TransactionListProps): JSX.Element {
  const rows = typeof limit === 'number' ? transactions.slice(0, limit) : transactions;

  return (
    <Card className={cn('flex flex-col gap-3', className)} flush>
      <div className="p-4 pb-0">
        <CardHeader>
          <CardTitle icon={<Receipt className="h-4 w-4" />}>Activity</CardTitle>
          {!loading && transactions.length > 0 && (
            <span className="text-[11px] text-muted">{transactions.length} total</span>
          )}
        </CardHeader>
      </div>

      {loading ? (
        <ul className="flex flex-col">
          {Array.from({ length: 4 }).map((_, i) => (
            <li key={i} className="flex items-center gap-3 px-4 py-3">
              <Skeleton circle className="h-9 w-9" />
              <div className="flex-1">
                <Skeleton className="h-3.5 w-24" />
                <Skeleton className="mt-1.5 h-3 w-16" />
              </div>
              <Skeleton className="h-4 w-16" />
            </li>
          ))}
        </ul>
      ) : rows.length === 0 ? (
        <div className="px-4 pb-6 pt-2 text-center text-xs text-muted">No activity yet.</div>
      ) : (
        <ul className="flex flex-col divide-y divide-border">
          {rows.map((txn) => {
            const meta = META[txn.type];
            const failed = txn.status === 'failed';
            const signed = meta.sign * txn.amount;
            return (
              <li key={txn.id} className="flex items-center gap-3 px-4 py-3">
                <span className={cn('flex h-9 w-9 shrink-0 items-center justify-center rounded-xl', meta.tint)}>
                  {meta.icon}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-text">{meta.label}</span>
                    {txn.symbol && (
                      <Badge tone="neutral" size="sm">
                        {txn.symbol}
                      </Badge>
                    )}
                    {failed && (
                      <Badge tone="danger" size="sm">
                        Failed
                      </Badge>
                    )}
                  </div>
                  <p className="line-clamp-1 text-[11px] text-muted">
                    {txn.note || txn.ref} · {formatRelativeTime(txn.createdAt)}
                  </p>
                </div>
                <span
                  className={cn(
                    'shrink-0 text-sm font-semibold tnum',
                    failed ? 'text-muted line-through' : signed >= 0 ? 'text-success' : 'text-danger',
                  )}
                >
                  {signed >= 0 ? '+' : '−'}
                  {formatCurrency(Math.abs(txn.amount))}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}
