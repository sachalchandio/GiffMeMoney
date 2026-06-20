/**
 * StrategyLabPage — the quant-model library.
 *
 * Left/top: a searchable, category-grouped gallery of all ~73 {@link StrategyMeta}
 * (summary + formula + inputs + sources) rendered as {@link StrategyCard}s.
 * Selecting a strategy opens a detail panel showing:
 *   • its cross-asset rankings (`useStrategyRankings`) as a horizontal score bar chart,
 *   • an {@link EquityCurveChart} (strategy vs buy-&-hold) from `useBacktest(symbol, id)`
 *     for a chosen asset, and
 *   • the per-asset {@link LeaderboardTable} (`useLeaderboard`) ranking every strategy
 *     by realized backtest performance for that asset.
 *
 * Educational, scannable, fully responsive, light + dark. Colors via tokens only.
 * Owns only this file + the two domain components it imports. No `any`.
 */

import { useEffect, useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  Cell,
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { FlaskConical, LineChart, ListOrdered, Search, X } from 'lucide-react';

import { useStrategies, useStrategyRankings, useLeaderboard } from '@/hooks/useStrategies';
import { useBacktest } from '@/hooks/useBacktest';
import { useAssets } from '@/hooks/useAssets';

import { Card, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Select, type SelectOption } from '@/components/ui/Select';
import { Skeleton, SkeletonCard } from '@/components/ui/Skeleton';
import { EquityCurveChart } from '@/components/charts/EquityCurveChart';
import { ChartEmpty } from '@/components/charts/ChartEmpty';
import { ChartTooltip, type TooltipRow } from '@/components/charts/ChartTooltip';

import { StrategyCard } from '@/components/domain/StrategyCard';
import { LeaderboardTable } from '@/components/domain/LeaderboardTable';

import { useChartTokens } from '@/theme/tokens';
import { cn, stanceColorVar, stanceFromScore } from '@/lib/utils';
import { formatRatio } from '@/lib/format';
import type {
  Asset,
  BacktestResultDTO,
  RankingEntry,
  StrategyCategory,
  StrategyMeta,
} from '@/lib/types';

/** Display order for the category sections. */
const CATEGORY_ORDER: StrategyCategory[] = [
  'Valuation',
  'Fundamental',
  'Factor',
  'Risk-Adjusted',
  'Portfolio',
  'Technical',
  'Statistical',
  'Derivatives',
];

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function StrategyLabPage(): JSX.Element {
  const strategiesQuery = useStrategies();
  const assetsQuery = useAssets();

  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState<string | undefined>(undefined);
  const [symbol, setSymbol] = useState<string | undefined>(undefined);

  const strategies = useMemo(() => strategiesQuery.data ?? [], [strategiesQuery.data]);
  const assets = useMemo(() => assetsQuery.data ?? [], [assetsQuery.data]);

  // Default the backtest/leaderboard asset to the first universe symbol.
  useEffect(() => {
    if (!symbol && assets.length > 0) setSymbol(assets[0]?.symbol);
  }, [assets, symbol]);

  const selected = useMemo(
    () => strategies.find((s) => s.id === selectedId),
    [strategies, selectedId],
  );

  // Filter + group by category for the gallery.
  const grouped = useMemo(() => groupByCategory(strategies, query), [strategies, query]);
  const totalMatches = useMemo(
    () => grouped.reduce((acc, g) => acc + g.items.length, 0),
    [grouped],
  );

  return (
    <div className="mx-auto w-full max-w-content px-3 py-4 sm:px-4 lg:px-6">
      {/* Header */}
      <header className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-primary/12 text-primary">
              <FlaskConical className="h-4 w-4" aria-hidden />
            </span>
            <h1 className="text-lg font-semibold tracking-tight text-text">Strategy Lab</h1>
            {strategies.length > 0 && (
              <Badge tone="neutral" size="sm" variant="soft">
                {strategies.length} models
              </Badge>
            )}
          </div>
          <p className="max-w-2xl text-sm text-muted">
            Every quant model in GiffMeMoney — its formula, the data it consumes, and the research
            behind it. Pick one to see how it ranks the universe and how it has performed.
          </p>
        </div>

        {/* Search */}
        <div className="relative w-full sm:w-72">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" aria-hidden />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search strategies…"
            aria-label="Search strategies"
            className={cn(
              'h-9 w-full rounded-xl border border-border bg-surface pl-9 pr-3 text-sm text-text',
              'placeholder:text-muted transition-colors hover:border-muted',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
            )}
          />
        </div>
      </header>

      {/* Detail panel for the selected strategy */}
      {selected && (
        <StrategyDetail
          meta={selected}
          assets={assets}
          symbol={symbol}
          onSymbolChange={setSymbol}
          onSelect={setSelectedId}
          onClose={() => setSelectedId(undefined)}
          className="mt-4"
        />
      )}

      {/* Gallery */}
      <section className="mt-5">
        {strategiesQuery.isLoading ? (
          <GallerySkeleton />
        ) : strategiesQuery.isError ? (
          <Card className="text-sm text-danger">
            Couldn&apos;t load the strategy catalog. {String(strategiesQuery.error?.message ?? '')}
          </Card>
        ) : totalMatches === 0 ? (
          <Card className="flex flex-col items-center gap-2 py-10 text-center text-muted">
            <Search className="h-5 w-5 opacity-60" aria-hidden />
            <p className="text-sm">
              No strategies match “<span className="text-text">{query}</span>”.
            </p>
            <Button variant="ghost" size="sm" onClick={() => setQuery('')}>
              Clear search
            </Button>
          </Card>
        ) : (
          <div className="space-y-6">
            {grouped.map((group) => (
              <CategorySection
                key={group.category}
                category={group.category}
                items={group.items}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Gallery — category sections                                         */
/* ------------------------------------------------------------------ */

interface CategoryGroup {
  category: StrategyCategory;
  items: StrategyMeta[];
}

function groupByCategory(strategies: StrategyMeta[], query: string): CategoryGroup[] {
  const q = query.trim().toLowerCase();
  const match = (s: StrategyMeta): boolean => {
    if (!q) return true;
    return (
      s.name.toLowerCase().includes(q) ||
      s.id.toLowerCase().includes(q) ||
      s.category.toLowerCase().includes(q) ||
      s.summary.toLowerCase().includes(q) ||
      s.references.some((r) => r.toLowerCase().includes(q))
    );
  };

  const byCat = new Map<StrategyCategory, StrategyMeta[]>();
  for (const s of strategies) {
    if (!match(s)) continue;
    const list = byCat.get(s.category) ?? [];
    list.push(s);
    byCat.set(s.category, list);
  }

  const ordered: CategoryGroup[] = [];
  const seen = new Set<StrategyCategory>();
  for (const cat of CATEGORY_ORDER) {
    const items = byCat.get(cat);
    if (items && items.length > 0) {
      ordered.push({ category: cat, items });
      seen.add(cat);
    }
  }
  // Any categories not in the canonical order (defensive) appended alphabetically.
  for (const [cat, items] of byCat) {
    if (!seen.has(cat) && items.length > 0) ordered.push({ category: cat, items });
  }
  return ordered;
}

function CategorySection({
  category,
  items,
  selectedId,
  onSelect,
}: {
  category: StrategyCategory;
  items: StrategyMeta[];
  selectedId: string | undefined;
  onSelect: (id: string) => void;
}): JSX.Element {
  return (
    <div>
      <div className="mb-2.5 flex items-center gap-2">
        <h2 className="text-sm font-semibold tracking-tight text-text">{category}</h2>
        <span className="rounded-full bg-surface-2 px-1.5 py-0.5 text-[11px] font-medium tnum text-muted">
          {items.length}
        </span>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {items.map((meta) => (
          <StrategyCard
            key={meta.id}
            meta={meta}
            selected={meta.id === selectedId}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  );
}

function GallerySkeleton(): JSX.Element {
  return (
    <div className="space-y-6">
      {Array.from({ length: 2 }).map((_, s) => (
        <div key={s}>
          <Skeleton className="mb-2.5 h-4 w-28" />
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <SkeletonCard key={i} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Detail panel                                                        */
/* ------------------------------------------------------------------ */

function StrategyDetail({
  meta,
  assets,
  symbol,
  onSymbolChange,
  onSelect,
  onClose,
  className,
}: {
  meta: StrategyMeta;
  assets: Asset[];
  symbol: string | undefined;
  onSymbolChange: (symbol: string) => void;
  onSelect: (id: string) => void;
  onClose: () => void;
  className?: string;
}): JSX.Element {
  const rankingsQuery = useStrategyRankings(meta.id, 20);
  const backtestQuery = useBacktest(symbol, meta.id);
  const leaderboardQuery = useLeaderboard(symbol, 20);

  const assetOptions: SelectOption<string>[] = assets.map((a) => ({
    value: a.symbol,
    label: `${a.symbol} · ${a.name}`,
  }));

  return (
    <Card flush className={cn('overflow-hidden border-primary/30 shadow-card', className)}>
      {/* Detail header */}
      <div className="flex flex-col gap-3 border-b border-border bg-surface-2/50 p-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0 space-y-1.5">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-base font-semibold tracking-tight text-text">{meta.name}</h2>
            <Badge tone="primary" size="sm" variant="soft">
              {meta.category}
            </Badge>
          </div>
          <p className="max-w-3xl text-sm leading-relaxed text-muted">{meta.summary}</p>
          {meta.formula && (
            <code className="mt-1 inline-block max-w-full break-words rounded-lg bg-surface px-2.5 py-1.5 font-mono text-xs text-text">
              {meta.formula}
            </code>
          )}
          {meta.references.length > 0 && (
            <p className="pt-1 text-[11px] text-muted">
              <span className="font-medium">Sources:</span> {meta.references.join(' · ')}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2 self-start">
          <Select
            options={assetOptions}
            value={symbol ?? ''}
            onChange={onSymbolChange}
            size="sm"
            aria-label="Backtest asset"
            placeholder={assetOptions.length === 0 ? 'Loading…' : undefined}
          />
          <Button
            variant="ghost"
            size="icon"
            onClick={onClose}
            aria-label="Close strategy detail"
          >
            <X className="h-4 w-4" aria-hidden />
          </Button>
        </div>
      </div>

      {/* Detail body */}
      <div className="grid grid-cols-1 gap-4 p-4 xl:grid-cols-2">
        {/* Cross-asset rankings */}
        <section>
          <CardTitle icon={<ListOrdered className="h-4 w-4" />} className="mb-2">
            Cross-asset scores
          </CardTitle>
          <p className="mb-2 text-xs text-muted">
            How this model scores every asset in the universe ({'−'}100 bearish &rarr; +100
            bullish).
          </p>
          {rankingsQuery.isLoading ? (
            <Skeleton className="h-[300px] w-full" />
          ) : rankingsQuery.isError ? (
            <ChartEmpty height={300} label="Couldn't load rankings" />
          ) : (
            <RankingsBarChart entries={rankingsQuery.data?.entries ?? []} />
          )}
        </section>

        {/* Backtest equity curve */}
        <section>
          <CardTitle icon={<LineChart className="h-4 w-4" />} className="mb-2">
            Realized performance{symbol ? ` · ${symbol}` : ''}
          </CardTitle>
          <p className="mb-2 text-xs text-muted">
            Growth of $1 — this strategy&apos;s rules vs simply buying &amp; holding the asset.
          </p>
          {!symbol || backtestQuery.isLoading ? (
            <Skeleton className="h-[300px] w-full" />
          ) : backtestQuery.isError ? (
            <ChartEmpty height={300} label="Couldn't load backtest" />
          ) : backtestQuery.data ? (
            <>
              <EquityCurveChart result={backtestQuery.data} height={300} />
              {backtestQuery.data.supported && (
                <BacktestSummary result={backtestQuery.data} />
              )}
            </>
          ) : (
            <ChartEmpty height={300} />
          )}
        </section>
      </div>

      {/* Per-asset leaderboard */}
      <section className="border-t border-border p-4">
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <CardTitle icon={<ListOrdered className="h-4 w-4" />}>
            Strategy leaderboard{symbol ? ` · ${symbol}` : ''}
          </CardTitle>
          <span className="text-xs text-muted">Ranked by realized Sharpe for the selected asset</span>
        </div>
        {!symbol || leaderboardQuery.isLoading ? (
          <Skeleton className="h-48 w-full" />
        ) : leaderboardQuery.isError ? (
          <ChartEmpty height={160} label="Couldn't load leaderboard" />
        ) : leaderboardQuery.data ? (
          <LeaderboardTable
            leaderboard={leaderboardQuery.data}
            selectedId={meta.id}
            onSelect={onSelect}
          />
        ) : (
          <ChartEmpty height={160} />
        )}
      </section>
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Rankings bar chart                                                  */
/* ------------------------------------------------------------------ */

interface RankRow {
  symbol: string;
  name: string;
  score: number;
  stance: RankingEntry['stance'];
}

function RankingsBarChart({ entries }: { entries: RankingEntry[] }): JSX.Element {
  const tokens = useChartTokens();

  const rows: RankRow[] = useMemo(
    () =>
      entries
        .map((e) => ({
          symbol: e.asset.symbol,
          name: e.asset.name,
          score: e.score,
          stance: e.stance,
        }))
        .sort((a, b) => b.score - a.score),
    [entries],
  );

  if (rows.length === 0) {
    return <ChartEmpty height={300} label="No rankings available" />;
  }

  // Height scales with row count so labels never crush together.
  const height = Math.max(300, rows.length * 22 + 24);

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={rows}
          layout="vertical"
          margin={{ top: 4, right: 28, bottom: 4, left: 4 }}
          barCategoryGap={3}
        >
          <CartesianGrid stroke={tokens.grid} strokeDasharray="3 3" horizontal={false} />
          <XAxis
            type="number"
            domain={[-100, 100]}
            ticks={[-100, -50, 0, 50, 100]}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: tokens.grid }}
          />
          <YAxis
            type="category"
            dataKey="symbol"
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={52}
          />
          <Tooltip
            cursor={{ fill: tokens.surface2, opacity: 0.5 }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0]?.payload as RankRow | undefined;
              if (!row) return null;
              const rowsOut: TooltipRow[] = [
                {
                  label: 'Score',
                  value: `${row.score > 0 ? '+' : ''}${formatRatio(row.score, 1)}`,
                  color: stanceColorVar(row.stance),
                },
                { label: 'Stance', value: row.stance.replace('_', ' ') },
              ];
              return <ChartTooltip title={`${row.symbol} · ${row.name}`} rows={rowsOut} />;
            }}
          />
          <Bar dataKey="score" radius={[3, 3, 3, 3]} isAnimationActive={false}>
            {rows.map((row) => (
              <Cell key={row.symbol} fill={stanceColorVar(stanceFromScore(row.score))} />
            ))}
            <LabelList
              dataKey="score"
              position="right"
              formatter={(value: number) => `${value > 0 ? '+' : ''}${Math.round(value)}`}
              className="tnum"
              style={{ fill: tokens.muted, fontSize: 10 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Compact backtest stat strip                                         */
/* ------------------------------------------------------------------ */

function BacktestSummary({ result }: { result: BacktestResultDTO }): JSX.Element {
  const m = result.metrics;
  const items: { label: string; value: string; tone?: 'pos' | 'neg' }[] = [
    { label: 'Sharpe', value: formatRatio(m.sharpe), tone: m.sharpe >= 1 ? 'pos' : undefined },
    {
      label: 'CAGR',
      value: `${m.cagr >= 0 ? '+' : ''}${formatRatio(m.cagr * 100, 1)}%`,
      tone: m.cagr >= 0 ? 'pos' : 'neg',
    },
    { label: 'Max DD', value: `${formatRatio(m.maxDrawdown * 100, 1)}%`, tone: 'neg' },
    { label: 'Calmar', value: formatRatio(m.calmar) },
    { label: 'Win rate', value: `${Math.round(m.winRate * 100)}%` },
    { label: 'Trades', value: String(result.trades) },
  ];
  return (
    <div className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-6">
      {items.map((it) => (
        <div key={it.label} className="rounded-xl bg-surface-2 px-2.5 py-2 text-center">
          <div className="text-[10px] font-medium uppercase tracking-wide text-muted">
            {it.label}
          </div>
          <div
            className={cn(
              'tnum mt-0.5 text-sm font-semibold',
              it.tone === 'pos' && 'text-success',
              it.tone === 'neg' && 'text-danger',
              !it.tone && 'text-text',
            )}
          >
            {it.value}
          </div>
        </div>
      ))}
    </div>
  );
}
