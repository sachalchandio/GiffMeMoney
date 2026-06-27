/**
 * WalletHeader — the Invest page hero. Shows the wallet's cash, invested value,
 * total value and total P&L (color-coded, derived live from the market store so
 * the figures move every tick), with prominent Add Funds + Withdraw actions and
 * a "Demo / sandbox — no real charge" tag. Colors via semantic tokens only.
 */

import { useMemo } from 'react';
import { ArrowDownToLine, Plus, ShieldCheck, Wallet as WalletIcon } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Skeleton } from '@/components/ui/Skeleton';
import { useMarketStore } from '@/store/marketStore';
import type { Position, Wallet } from '@/lib/types';
import { formatCurrency, formatPct } from '@/lib/format';
import { changeTextColor, cn } from '@/lib/utils';

export interface WalletHeaderProps {
  wallet: Wallet | undefined;
  positions: Position[];
  loading?: boolean;
  onAddFunds: () => void;
  onWithdraw: () => void;
  className?: string;
}

/** Live mark-to-market of a position list using the latest streamed prices. */
function useLiveTotals(
  wallet: Wallet | undefined,
  positions: Position[],
): { invested: number; total: number; cost: number; pnl: number; pnlPct: number } {
  const prices = useMarketStore((s) => s.prices);
  return useMemo(() => {
    let invested = 0;
    let cost = 0;
    for (const p of positions) {
      const live = prices[p.symbol.toUpperCase()]?.price;
      const price = typeof live === 'number' && Number.isFinite(live) ? live : p.currentPrice;
      invested += price * p.units;
      cost += p.costBasis;
    }
    const cash = wallet?.cashBalance ?? 0;
    const total = cash + invested;
    const pnl = invested - cost;
    const pnlPct = cost > 0 ? (pnl / cost) * 100 : 0;
    return { invested, total, cost, pnl, pnlPct };
  }, [positions, prices, wallet?.cashBalance]);
}

export function WalletHeader({
  wallet,
  positions,
  loading = false,
  onAddFunds,
  onWithdraw,
  className,
}: WalletHeaderProps): JSX.Element {
  const live = useLiveTotals(wallet, positions);
  const currency = wallet?.currency ?? 'USD';
  const cash = wallet?.cashBalance ?? 0;

  if (loading || !wallet) {
    return (
      <Card className={cn('flex flex-col gap-4', className)}>
        <Skeleton className="h-4 w-28" />
        <Skeleton className="h-9 w-44" />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-14 w-full" />
          ))}
        </div>
      </Card>
    );
  }

  return (
    <Card className={cn('relative overflow-hidden', className)}>
      {/* Subtle brand glow */}
      <div
        className="pointer-events-none absolute -right-16 -top-20 h-48 w-48 rounded-full bg-primary/10 blur-3xl"
        aria-hidden
      />

      <div className="relative flex flex-col gap-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-xs font-medium text-muted">
              <span className="flex h-7 w-7 items-center justify-center rounded-xl bg-primary/12 text-primary">
                <WalletIcon className="h-4 w-4" aria-hidden />
              </span>
              <span className="tracking-tight">Total portfolio value</span>
            </div>
            <div className="mt-1.5 flex flex-wrap items-end gap-3">
              <span className="text-3xl font-semibold tracking-tight tnum text-text">
                {formatCurrency(live.total, { currency })}
              </span>
              <span
                className={cn(
                  'mb-1 inline-flex items-center gap-1 rounded-lg bg-surface-2 px-2 py-0.5 text-sm font-medium tnum',
                  changeTextColor(live.pnl),
                )}
              >
                {formatCurrency(live.pnl, { currency })}
                <span className="text-xs">({formatPct(live.pnlPct, { sign: true, digits: 2 })})</span>
              </span>
            </div>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <Button variant="primary" size="md" leftIcon={<Plus className="h-4 w-4" />} onClick={onAddFunds}>
              Add Funds
            </Button>
            <Button
              variant="outline"
              size="md"
              leftIcon={<ArrowDownToLine className="h-4 w-4" />}
              onClick={onWithdraw}
              disabled={cash <= 0}
            >
              Withdraw
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <Metric label="Cash available" value={formatCurrency(cash, { currency })} />
          <Metric label="Invested" value={formatCurrency(live.invested, { currency })} />
          <Metric
            label="Total return"
            value={formatPct(live.pnlPct, { sign: true, digits: 2 })}
            tone={changeTextColor(live.pnl)}
            className="col-span-2 sm:col-span-1"
          />
        </div>

        <div className="flex items-center justify-between gap-2">
          <Badge tone="warning" variant="soft" size="sm" icon={<ShieldCheck className="h-3 w-3" />}>
            Demo / sandbox — no real charge
          </Badge>
          {wallet.savedCards.length > 0 && (
            <span className="text-[0.6875rem] text-muted">
              {wallet.savedCards.length} saved card{wallet.savedCards.length === 1 ? '' : 's'}
            </span>
          )}
        </div>
      </div>
    </Card>
  );
}

function Metric({
  label,
  value,
  tone,
  className,
}: {
  label: string;
  value: string;
  tone?: string;
  className?: string;
}): JSX.Element {
  return (
    <div className={cn('rounded-xl border border-border bg-surface-2/50 px-3 py-2.5', className)}>
      <p className="text-[0.6875rem] font-medium uppercase tracking-wide text-muted">{label}</p>
      <p className={cn('mt-0.5 text-lg font-semibold tracking-tight tnum text-text', tone)}>{value}</p>
    </div>
  );
}
