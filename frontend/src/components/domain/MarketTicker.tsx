/**
 * MarketTicker — the live "tape" strip across the top of the Dashboard.
 *
 * Reads the live price map from `marketStore` (filled by `useMarketSocket`) and
 * scrolls a marquee of `symbol · price · change%` chips. The list of symbols to
 * show is seeded from the asset universe so the strip is populated immediately;
 * each chip then re-renders on every tick. Colors come from semantic tokens.
 */

import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useMarketStore } from '@/store/marketStore';
import type { Asset } from '@/lib/types';
import { cn, changeTextColor } from '@/lib/utils';
import { formatPrice, formatPct } from '@/lib/format';

export interface MarketTickerProps {
  /** Universe used to seed/order the tape (live prices override when present). */
  assets: Asset[];
  className?: string;
}

interface TickerItem {
  symbol: string;
  price: number;
  changePct: number;
  currency: string;
}

/**
 * One immutable chip. Pulled out so it subscribes to a single symbol's live
 * price and only re-renders when that symbol ticks.
 */
function TickerChip({ symbol, currency }: { symbol: string; currency: string }): JSX.Element {
  const live = useMarketStore((s) => s.prices[symbol.toUpperCase()]);
  const changePct = live?.changePct ?? 0;
  return (
    <Link
      to={`/asset/${encodeURIComponent(symbol)}`}
      className="inline-flex items-center gap-2 px-3 py-2 transition-colors hover:bg-surface-2"
    >
      <span className="text-xs font-semibold tracking-tight text-text">{symbol}</span>
      <span className="text-xs tnum text-muted">
        {live ? formatPrice(live.price, currency) : '—'}
      </span>
      <span className={cn('text-xs font-medium tnum', changeTextColor(changePct))}>
        {formatPct(changePct, { digits: 2, sign: true })}
      </span>
    </Link>
  );
}

export function MarketTicker({ assets, className }: MarketTickerProps): JSX.Element {
  const items = useMemo<TickerItem[]>(
    () =>
      assets.map((a) => ({
        symbol: a.symbol,
        price: a.price,
        changePct: a.change24hPct,
        currency: a.currency,
      })),
    [assets],
  );

  if (items.length === 0) {
    return (
      <div
        className={cn(
          'h-10 animate-pulse rounded-xl border border-border bg-surface-2',
          className,
        )}
        aria-hidden
      />
    );
  }

  // Duplicate the sequence so the marquee scroll is seamless.
  const loop = [...items, ...items];

  return (
    <div
      className={cn(
        'group relative overflow-hidden rounded-xl border border-border bg-surface shadow-soft',
        className,
      )}
      aria-label="Live market tape"
    >
      {/* Edge fades */}
      <div className="pointer-events-none absolute inset-y-0 left-0 z-10 w-10 bg-gradient-to-r from-surface to-transparent" />
      <div className="pointer-events-none absolute inset-y-0 right-0 z-10 w-10 bg-gradient-to-l from-surface to-transparent" />
      <div className="flex w-max animate-[ticker_42s_linear_infinite] divide-x divide-border group-hover:[animation-play-state:paused]">
        {loop.map((item, i) => (
          <TickerChip key={`${item.symbol}-${i}`} symbol={item.symbol} currency={item.currency} />
        ))}
      </div>
      {/* Marquee keyframes (scoped, token-free). */}
      <style>{`@keyframes ticker { from { transform: translateX(0); } to { transform: translateX(-50%); } }`}</style>
    </div>
  );
}
