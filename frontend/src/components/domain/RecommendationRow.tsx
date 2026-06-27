/**
 * RecommendationRow — one ranked recommendation rendered as an expandable card.
 * Collapsed: rank, asset identity, live price, composite score, stance, the 1Y
 * expected return, and the headline. Expanded: the bullet reasons plus a price
 * sparkline (1M of daily candles, lazily fetched on first expand). The whole card
 * links through to the asset-detail page. Colors via semantic tokens only.
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { ChevronDown } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { StanceBadge } from '@/components/ui/StanceBadge';
import { Badge } from '@/components/ui/Badge';
import { MiniSpark } from '@/components/charts/MiniSpark';
import { Skeleton } from '@/components/ui/Skeleton';
import { useCandles } from '@/hooks/useAssets';
import { useLivePrice } from '@/store/marketStore';
import type { Recommendation } from '@/lib/types';
import { formatPct, formatPrice, stanceLabel } from '@/lib/format';
import { assetClassLabel, changeTextColor, cn, stanceTextColor } from '@/lib/utils';

export interface RecommendationRowProps {
  rec: Recommendation;
  /** Start expanded (used for the top pick / tests). */
  defaultExpanded?: boolean;
  className?: string;
}

export function RecommendationRow({
  rec,
  defaultExpanded = false,
  className,
}: RecommendationRowProps): JSX.Element {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const { asset } = rec;

  // Lazily pull a short candle window only once the row has been opened.
  const candles = useCandles(expanded ? asset.symbol : undefined, '1d', 30);
  const spark = (candles.data ?? []).map((c) => c.c);

  const live = useLivePrice(asset.symbol);
  const price = live?.price ?? asset.price;
  const changePct = live?.changePct ?? asset.change24hPct;

  const panelId = `rec-panel-${asset.symbol}`;

  return (
    <Card interactive flush className={cn('overflow-hidden', className)}>
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
        aria-controls={panelId}
        className="flex w-full items-center gap-3 p-3 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT sm:gap-4"
      >
        {/* Rank */}
        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-primary/12 text-sm font-semibold tnum text-primary">
          {rec.rank}
        </span>

        {/* Identity */}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-semibold tracking-tight text-text">{asset.symbol}</span>
            <Badge tone="neutral" size="sm" className="hidden sm:inline-flex">
              {assetClassLabel(asset.assetClass)}
            </Badge>
          </div>
          <span className="line-clamp-1 text-xs text-muted">{asset.name}</span>
        </div>

        {/* Live price + 24h change */}
        <div className="hidden flex-col items-end sm:flex">
          <span className="text-sm font-medium tnum text-text">{formatPrice(price, asset.currency)}</span>
          <span className={cn('text-xs tnum', changeTextColor(changePct))}>
            {formatPct(changePct, { sign: true })}
          </span>
        </div>

        {/* 1Y expected return */}
        <div className="flex w-16 flex-col items-end sm:w-20">
          <span className="text-[0.625rem] uppercase tracking-wide text-muted">1Y</span>
          <span className={cn('text-sm font-semibold tnum', changeTextColor(rec.expectedReturn1YPct))}>
            {formatPct(rec.expectedReturn1YPct, { sign: true, digits: 1 })}
          </span>
        </div>

        {/* Composite score + stance */}
        <div className="flex w-20 flex-col items-end gap-1 sm:w-24">
          <span className={cn('text-base font-semibold tnum', stanceTextColor(rec.recommendation))}>
            {rec.compositeScore > 0 ? '+' : ''}
            {Math.round(rec.compositeScore)}
          </span>
          <StanceBadge stance={rec.recommendation} size="sm" />
        </div>

        <ChevronDown
          className={cn('h-4 w-4 shrink-0 text-muted transition-transform', expanded && 'rotate-180')}
          aria-hidden
        />
      </button>

      {expanded && (
        <div id={panelId} className="border-t border-border bg-surface-2/40 px-3 py-3 sm:px-4">
          <p className="text-sm text-text">{rec.headline}</p>

          <div className="mt-3 grid gap-4 sm:grid-cols-[1fr_auto] sm:items-start">
            <ul className="space-y-1.5">
              {rec.reasons.map((reason, i) => (
                <li key={i} className="flex gap-2 text-xs text-muted">
                  <span className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-primary" aria-hidden />
                  <span className="leading-relaxed">{reason}</span>
                </li>
              ))}
              {rec.reasons.length === 0 && (
                <li className="text-xs text-muted">No additional notes for this pick.</li>
              )}
            </ul>

            <div className="flex flex-col items-stretch gap-2 sm:w-44">
              <span className="text-[0.625rem] uppercase tracking-wide text-muted">30-day price</span>
              {candles.isPending ? (
                <Skeleton className="h-9 w-full" />
              ) : (
                <MiniSpark points={spark} width={176} height={36} className="w-full" />
              )}
              <Link
                to={`/asset/${encodeURIComponent(asset.symbol)}`}
                className="inline-flex h-8 items-center justify-center rounded-xl bg-primary px-3 text-xs font-medium text-white transition-colors hover:bg-primary-press focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT"
              >
                View {asset.symbol} · {stanceLabel(rec.recommendation)}
              </Link>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
