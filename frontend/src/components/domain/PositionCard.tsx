/**
 * PositionCard — one open holding. Shows units, average cost, live market value
 * and unrealized P&L ($ and %, recomputed from the live store price every tick),
 * allocation %, a 30-day price sparkline, and a Sell control (a partial dollar
 * amount or the whole position) wired to the sell mutation. Colors via tokens.
 */

import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Loader2, TrendingDown, TrendingUp } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { MiniSpark } from '@/components/charts/MiniSpark';
import { Skeleton } from '@/components/ui/Skeleton';
import { useCandles } from '@/hooks/useAssets';
import { useSell } from '@/hooks/usePortfolioState';
import { useLivePrice } from '@/store/marketStore';
import type { Position, SellRequest } from '@/lib/types';
import { formatCurrency, formatNumber, formatPct, formatPrice } from '@/lib/format';
import { assetClassLabel, changeTextColor, cn } from '@/lib/utils';

export interface PositionCardProps {
  position: Position;
  className?: string;
}

export function PositionCard({ position, className }: PositionCardProps): JSX.Element {
  const [selling, setSelling] = useState(false);
  const [sellAmount, setSellAmount] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const sell = useSell();
  const live = useLivePrice(position.symbol);
  const candles = useCandles(position.symbol, '1d', 30);
  const spark = useMemo(() => (candles.data ?? []).map((c) => c.c), [candles.data]);

  const price = live?.price ?? position.currentPrice;
  const marketValue = price * position.units;
  const pnl = marketValue - position.costBasis;
  const pnlPct = position.costBasis > 0 ? (pnl / position.costBasis) * 100 : 0;
  const Trend = pnl >= 0 ? TrendingUp : TrendingDown;

  const amountNum = Number(sellAmount);
  const partialValid = Number.isFinite(amountNum) && amountNum > 0 && amountNum <= marketValue + 0.01;

  const doSell = (all: boolean): void => {
    setError(null);
    if (!all && !partialValid) {
      setError('Enter an amount up to the position value.');
      return;
    }
    const req: SellRequest = all
      ? { symbol: position.symbol, amount: null, all: true }
      : { symbol: position.symbol, amount: Math.round(amountNum * 100) / 100, all: false };
    sell.mutate(req, {
      onSuccess: () => {
        setSelling(false);
        setSellAmount('');
      },
      onError: (err) => setError(err instanceof Error ? err.message : 'Sell failed.'),
    });
  };

  return (
    <Card className={cn('flex flex-col gap-3', className)} interactive>
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Link
              to={`/asset/${encodeURIComponent(position.symbol)}`}
              className="truncate text-sm font-semibold tracking-tight text-text hover:text-primary"
            >
              {position.symbol}
            </Link>
            <Badge tone="neutral" size="sm">
              {assetClassLabel(position.asset.assetClass)}
            </Badge>
          </div>
          <p className="line-clamp-1 text-xs text-muted">{position.asset.name}</p>
        </div>
        <div className="shrink-0 text-right">
          <p className="text-sm font-semibold tnum text-text">{formatCurrency(marketValue, { currency: position.asset.currency })}</p>
          <p className={cn('inline-flex items-center justify-end gap-0.5 text-xs font-medium tnum', changeTextColor(pnl))}>
            <Trend className="h-3 w-3" aria-hidden />
            {formatCurrency(pnl, { currency: position.asset.currency })} ({formatPct(pnlPct, { sign: true, digits: 1 })})
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs sm:grid-cols-4">
        <Stat label="Units" value={formatNumber(position.units, 4)} />
        <Stat label="Avg cost" value={formatPrice(position.avgPrice, position.asset.currency)} />
        <Stat label="Price" value={formatPrice(price, position.asset.currency)} />
        <Stat label="Allocation" value={formatPct(position.allocationPct, { digits: 1 })} />
      </div>

      <div className="flex items-center justify-between gap-3">
        {candles.isPending ? (
          <Skeleton className="h-7 w-28" />
        ) : (
          <MiniSpark points={spark} width={120} height={28} aria-label={`${position.symbol} 30-day price`} />
        )}
        {!selling ? (
          <Button variant="outline" size="sm" onClick={() => setSelling(true)}>
            Sell
          </Button>
        ) : (
          <span className="text-[0.6875rem] text-muted">
            {position.realizedPnl !== 0 && (
              <>Realized {formatCurrency(position.realizedPnl, { currency: position.asset.currency })}</>
            )}
          </span>
        )}
      </div>

      {selling && (
        <div className="flex flex-col gap-2 rounded-xl border border-border bg-surface-2/40 p-2.5">
          <div className="flex items-center gap-2">
            <div className="relative flex-1">
              <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-xs text-muted">$</span>
              <input
                type="number"
                inputMode="decimal"
                min={0}
                max={marketValue}
                value={sellAmount}
                onChange={(e) => setSellAmount(e.target.value)}
                placeholder={`Up to ${formatCurrency(marketValue, { currency: position.asset.currency })}`}
                className="h-9 w-full rounded-xl border border-border bg-surface pl-5 pr-2 text-sm tnum text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT"
                aria-label={`Amount to sell of ${position.symbol}`}
              />
            </div>
            <Button variant="danger" size="sm" onClick={() => doSell(false)} disabled={!partialValid || sell.isPending}>
              {sell.isPending ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : 'Sell'}
            </Button>
          </div>
          <div className="flex items-center justify-between">
            <button
              type="button"
              onClick={() => doSell(true)}
              disabled={sell.isPending}
              className="text-[0.6875rem] font-medium text-danger hover:underline disabled:opacity-50"
            >
              Sell entire position
            </button>
            <button
              type="button"
              onClick={() => {
                setSelling(false);
                setError(null);
                setSellAmount('');
              }}
              className="text-[0.6875rem] font-medium text-muted hover:text-text"
            >
              Cancel
            </button>
          </div>
          {error && <p className="text-[0.6875rem] font-medium text-danger" role="alert">{error}</p>}
        </div>
      )}
    </Card>
  );
}

function Stat({ label, value }: { label: string; value: string }): JSX.Element {
  return (
    <div className="flex flex-col">
      <span className="text-[0.625rem] uppercase tracking-wide text-muted">{label}</span>
      <span className="tnum font-medium text-text">{value}</span>
    </div>
  );
}
