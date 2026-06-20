/**
 * RecommendationsPage — the ranked opportunity list across the whole universe.
 *
 * A class filter (All / Stocks / Crypto / ETFs) drives `useRecommendations`; each
 * pick renders as an expandable {@link RecommendationRow} (reasons + 1Y expected
 * + a price sparkline). The header surfaces a quick breakdown of how many picks
 * lean buy / hold / sell. Skeletons cover loading; a friendly empty/error state
 * covers the rest. Dense, responsive, light/dark via tokens.
 */

import { useMemo, useState } from 'react';
import { AlertTriangle, Sparkles } from 'lucide-react';
import { RecommendationRow } from '@/components/domain/RecommendationRow';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Tabs, type TabItem } from '@/components/ui/Tabs';
import { Skeleton } from '@/components/ui/Skeleton';
import { Button } from '@/components/ui/Button';
import { useRecommendations } from '@/hooks/useRecommendations';
import type { AssetClass, Recommendation } from '@/lib/types';
import { stanceTone } from '@/lib/utils';

type ClassFilter = 'all' | AssetClass;

const FILTER_ITEMS: TabItem<ClassFilter>[] = [
  { value: 'all', label: 'All' },
  { value: 'equity', label: 'Stocks' },
  { value: 'crypto', label: 'Crypto' },
  { value: 'etf', label: 'ETFs' },
];

interface ToneCounts {
  positive: number;
  neutral: number;
  negative: number;
}

function countByTone(recs: Recommendation[]): ToneCounts {
  return recs.reduce<ToneCounts>(
    (acc, r) => {
      acc[stanceTone(r.recommendation)] += 1;
      return acc;
    },
    { positive: 0, neutral: 0, negative: 0 },
  );
}

export default function RecommendationsPage(): JSX.Element {
  const [filter, setFilter] = useState<ClassFilter>('all');
  const assetClass = filter === 'all' ? undefined : filter;
  const query = useRecommendations(50, assetClass);

  const recs = query.data ?? [];
  const tones = useMemo(() => countByTone(recs), [recs]);

  return (
    <div className="flex flex-col gap-4">
      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-text">
            <Sparkles className="h-5 w-5 text-primary" aria-hidden />
            Recommendations
          </h1>
          <p className="mt-1 text-sm text-muted">
            Ranked across the universe by the blended composite score. Expand any pick for the reasons,
            its one-year outlook, and a recent price trend.
          </p>
        </div>

        {!query.isPending && recs.length > 0 && (
          <div className="flex shrink-0 items-center gap-2">
            <Badge tone="success" variant="soft">
              {tones.positive} buy
            </Badge>
            <Badge tone="warning" variant="soft">
              {tones.neutral} hold
            </Badge>
            <Badge tone="danger" variant="soft">
              {tones.negative} sell
            </Badge>
          </div>
        )}
      </div>

      {/* Filter */}
      <Tabs
        items={FILTER_ITEMS}
        value={filter}
        onChange={setFilter}
        variant="pill"
        aria-label="Filter recommendations by asset class"
        className="w-fit"
      />

      {/* Body */}
      {query.isError ? (
        <Card className="flex flex-col items-center gap-3 py-12 text-center">
          <AlertTriangle className="h-6 w-6 text-danger" aria-hidden />
          <div>
            <p className="text-sm font-medium text-text">Couldn&apos;t load recommendations</p>
            <p className="mt-1 text-xs text-muted">
              {query.error instanceof Error ? query.error.message : 'Please try again.'}
            </p>
          </div>
          <Button variant="secondary" size="sm" onClick={() => void query.refetch()}>
            Retry
          </Button>
        </Card>
      ) : query.isPending ? (
        <div className="flex flex-col gap-2.5">
          {Array.from({ length: 8 }).map((_, i) => (
            <Skeleton key={i} className="h-[4.25rem] w-full rounded-2xl" />
          ))}
        </div>
      ) : recs.length === 0 ? (
        <Card className="flex flex-col items-center gap-2 py-12 text-center">
          <p className="text-sm font-medium text-text">No recommendations for this filter</p>
          <p className="text-xs text-muted">Try a different asset class.</p>
        </Card>
      ) : (
        <div className="flex flex-col gap-2.5">
          {recs.map((rec) => (
            <RecommendationRow key={rec.asset.symbol} rec={rec} defaultExpanded={rec.rank === 1} />
          ))}
        </div>
      )}
    </div>
  );
}
