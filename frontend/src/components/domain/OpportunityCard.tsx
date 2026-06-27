/**
 * OpportunityCard — a single ranked investment idea (a `Recommendation`) as a
 * clickable card for the Dashboard "where to invest now" grid.
 *
 * Shows the rank, asset identity, a stance pill, the composite score, the
 * projected 1-year return, the live price + change (from `marketStore`, falling
 * back to the snapshot), and the headline / top reasons. The whole card links to
 * the asset detail page. Colors via semantic tokens only.
 */

import { Link } from 'react-router-dom';
import { ArrowUpRight } from 'lucide-react';
import type { Recommendation } from '@/lib/types';
import { Card } from '@/components/ui/Card';
import { StanceBadge } from '@/components/ui/StanceBadge';
import { useMarketStore } from '@/store/marketStore';
import { cn, changeTextColor, assetClassLabel } from '@/lib/utils';
import { formatPrice, formatPct } from '@/lib/format';

export interface OpportunityCardProps {
  recommendation: Recommendation;
  /** Featured cards get a slightly richer treatment (more reasons shown). */
  featured?: boolean;
  className?: string;
}

export function OpportunityCard({
  recommendation,
  featured = false,
  className,
}: OpportunityCardProps): JSX.Element {
  const { rank, asset, compositeScore, recommendation: stance, expectedReturn1YPct, headline, reasons } =
    recommendation;
  const live = useMarketStore((s) => s.prices[asset.symbol.toUpperCase()]);
  const price = live?.price ?? asset.price;
  const changePct = live?.changePct ?? asset.change24hPct;
  const reasonCount = featured ? 3 : 2;

  return (
    <Card
      interactive
      className={cn('relative flex h-full flex-col gap-3 p-4', className)}
    >
      <Link
        to={`/asset/${encodeURIComponent(asset.symbol)}`}
        className="absolute inset-0 rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT"
        aria-label={`Open ${asset.symbol} detail`}
      />

      <div className="flex items-start justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-surface-2 text-xs font-bold tnum text-muted">
            {rank}
          </span>
          <div className="min-w-0">
            <div className="flex items-center gap-1.5">
              <span className="truncate text-sm font-semibold tracking-tight text-text">
                {asset.symbol}
              </span>
              <span className="rounded-full bg-surface-2 px-1.5 text-[0.625rem] font-medium text-muted">
                {assetClassLabel(asset.assetClass)}
              </span>
            </div>
            <span className="block truncate text-xs text-muted" title={asset.name}>
              {asset.name}
            </span>
          </div>
        </div>
        <StanceBadge stance={stance} size="sm" />
      </div>

      <div className="grid grid-cols-3 gap-2">
        <div>
          <span className="block text-[0.625rem] uppercase tracking-wide text-muted">Score</span>
          <span className="text-sm font-semibold tnum text-text">
            {compositeScore > 0 ? '+' : ''}
            {Math.round(compositeScore)}
          </span>
        </div>
        <div>
          <span className="block text-[0.625rem] uppercase tracking-wide text-muted">1Y Proj.</span>
          <span className={cn('text-sm font-semibold tnum', changeTextColor(expectedReturn1YPct))}>
            {formatPct(expectedReturn1YPct, { digits: 1, sign: true })}
          </span>
        </div>
        <div>
          <span className="block text-[0.625rem] uppercase tracking-wide text-muted">Price</span>
          <span className="text-sm font-semibold tnum text-text">{formatPrice(price, asset.currency)}</span>
          <span className={cn('block text-[0.625rem] font-medium tnum', changeTextColor(changePct))}>
            {formatPct(changePct, { digits: 2, sign: true })}
          </span>
        </div>
      </div>

      <p className="text-xs leading-relaxed text-text">{headline}</p>

      {reasons.length > 0 && (
        <ul className="mt-auto space-y-1">
          {reasons.slice(0, reasonCount).map((reason, i) => (
            <li key={i} className="flex items-start gap-1.5 text-[0.6875rem] leading-relaxed text-muted">
              <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-primary" aria-hidden />
              <span className="line-clamp-2">{reason}</span>
            </li>
          ))}
        </ul>
      )}

      <span className="pointer-events-none absolute right-3 top-3 text-muted opacity-0 transition-opacity group-hover:opacity-100">
        <ArrowUpRight className="h-4 w-4" aria-hidden />
      </span>
    </Card>
  );
}
