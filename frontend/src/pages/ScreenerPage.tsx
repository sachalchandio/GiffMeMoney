/**
 * ScreenerPage — a sortable / filterable table across the whole universe.
 *
 * Columns: symbol/name, asset class, live price, 24h change, composite score +
 * stance, Sharpe, annualized vol, and projected 1-year return. Price/change come
 * from `useAssets` (overlaid with live `marketStore` ticks); the analytical
 * columns (composite / Sharpe / vol / 1Y) are pulled **lazily** per asset via
 * `useQueries` so the table paints immediately and enriches as each analysis
 * resolves (and the cache is shared with the asset-detail page).
 *
 * Filter by asset class + free-text search; click any column header to sort;
 * click a row to open `/asset/:symbol`. Renders as a real table on desktop and a
 * stack of cards on mobile. Colors via semantic tokens only.
 */

import { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQueries } from '@tanstack/react-query';
import { ArrowDown, ArrowUp, ArrowUpDown, Search, Table2, X } from 'lucide-react';
import { api } from '@/lib/api';
import { queryKeys } from '@/hooks/queryKeys';
import { useAssets } from '@/hooks/useAssets';
import { useMarketStore } from '@/store/marketStore';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { StanceBadge } from '@/components/ui/StanceBadge';
import { Skeleton } from '@/components/ui/Skeleton';
import { Tabs } from '@/components/ui/Tabs';
import type { Asset, AssetAnalysis, AssetClass, Stance } from '@/lib/types';
import { cn, changeTextColor, assetClassLabel, stanceFromScore } from '@/lib/utils';
import { formatPrice, formatPct, formatRatio, formatCompactCurrency } from '@/lib/format';

/* ------------------------------------------------------------------ */
/* Row model                                                           */
/* ------------------------------------------------------------------ */

interface ScreenerRow {
  asset: Asset;
  price: number;
  changePct: number;
  /** Analytical fields (undefined until that asset's analysis resolves). */
  composite?: number;
  stance?: Stance;
  sharpe?: number;
  annualVol?: number;
  return1YPct?: number;
  loading: boolean;
}

type SortKey =
  | 'symbol'
  | 'price'
  | 'changePct'
  | 'composite'
  | 'sharpe'
  | 'annualVol'
  | 'return1YPct';
type SortDir = 'asc' | 'desc';

interface ColumnDef {
  key: SortKey;
  label: string;
  /** Numeric columns are right-aligned; the symbol column is left-aligned. */
  numeric: boolean;
}

const COLUMNS: ColumnDef[] = [
  { key: 'symbol', label: 'Asset', numeric: false },
  { key: 'price', label: 'Price', numeric: true },
  { key: 'changePct', label: '24h', numeric: true },
  { key: 'composite', label: 'Composite', numeric: true },
  { key: 'sharpe', label: 'Sharpe', numeric: true },
  { key: 'annualVol', label: 'Vol (ann.)', numeric: true },
  { key: 'return1YPct', label: '1Y proj.', numeric: true },
];

const CLASS_TABS: { value: AssetClass | 'all'; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'equity', label: 'Stocks' },
  { value: 'crypto', label: 'Crypto' },
  { value: 'etf', label: 'ETFs' },
];

/** Compare for a sort key (undefined analytics sink to the bottom). */
function compareRows(a: ScreenerRow, b: ScreenerRow, key: SortKey, dir: SortDir): number {
  if (key === 'symbol') {
    const cmp = a.asset.symbol.localeCompare(b.asset.symbol);
    return dir === 'asc' ? cmp : -cmp;
  }
  const av = a[key];
  const bv = b[key];
  const aMissing = typeof av !== 'number' || !Number.isFinite(av);
  const bMissing = typeof bv !== 'number' || !Number.isFinite(bv);
  if (aMissing && bMissing) return 0;
  if (aMissing) return 1; // missing always last regardless of direction
  if (bMissing) return -1;
  const cmp = (av as number) - (bv as number);
  return dir === 'asc' ? cmp : -cmp;
}

/* ------------------------------------------------------------------ */
/* Live price overlay (subscribes per symbol so rows tick)             */
/* ------------------------------------------------------------------ */

function useLiveRows(base: ScreenerRow[]): ScreenerRow[] {
  const prices = useMarketStore((s) => s.prices);
  return useMemo(
    () =>
      base.map((row) => {
        const live = prices[row.asset.symbol.toUpperCase()];
        if (!live) return row;
        return { ...row, price: live.price, changePct: live.changePct };
      }),
    [base, prices],
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function ScreenerPage(): JSX.Element {
  const navigate = useNavigate();
  const [assetClass, setAssetClass] = useState<AssetClass | 'all'>('all');
  const [search, setSearch] = useState('');
  const [sortKey, setSortKey] = useState<SortKey>('composite');
  const [sortDir, setSortDir] = useState<SortDir>('desc');

  const assetsQuery = useAssets(assetClass === 'all' ? undefined : assetClass);
  const assets = useMemo(() => assetsQuery.data ?? [], [assetsQuery.data]);

  // Lazily fetch each asset's analysis (composite / Sharpe / vol / 1Y). Shares
  // the cache with the asset-detail page; long stale time avoids re-fetching.
  const analyses = useQueries({
    queries: assets.map((a) => ({
      queryKey: queryKeys.analysis(a.symbol),
      queryFn: () => api.getAnalysis(a.symbol),
      staleTime: 60_000,
    })),
  });

  const baseRows = useMemo<ScreenerRow[]>(
    () =>
      assets.map((asset, i) => {
        const q = analyses[i];
        const analysis = q?.data as AssetAnalysis | undefined;
        const return1Y = analysis?.expectedReturns.find((r) => r.horizon === '1Y');
        return {
          asset,
          price: asset.price,
          changePct: asset.change24hPct,
          composite: analysis?.compositeScore,
          stance: analysis?.recommendation,
          sharpe: analysis?.riskMetrics.sharpe,
          annualVol: analysis?.riskMetrics.annualVol,
          return1YPct: return1Y?.expectedReturnPct,
          loading: q?.isLoading ?? true,
        };
      }),
    [assets, analyses],
  );

  const liveRows = useLiveRows(baseRows);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    const rows = q
      ? liveRows.filter(
          (r) =>
            r.asset.symbol.toLowerCase().includes(q) ||
            r.asset.name.toLowerCase().includes(q) ||
            (r.asset.sector ?? '').toLowerCase().includes(q),
        )
      : liveRows;
    return [...rows].sort((a, b) => compareRows(a, b, sortKey, sortDir));
  }, [liveRows, search, sortKey, sortDir]);

  const onSort = (key: SortKey): void => {
    if (key === sortKey) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      // Default to a sensible direction: symbol ascending, metrics descending.
      setSortDir(key === 'symbol' ? 'asc' : 'desc');
    }
  };

  const isLoading = assetsQuery.isLoading;
  const open = (symbol: string): void => navigate(`/asset/${encodeURIComponent(symbol)}`);

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h1 className="flex items-center gap-2 text-lg font-semibold tracking-tight text-text lg:text-xl">
          <Table2 className="h-5 w-5 text-primary" aria-hidden />
          Screener
        </h1>
        <p className="text-xs text-muted">
          Sort and filter the full universe by price, momentum, composite score, risk and projection.
        </p>
      </div>

      {/* Controls */}
      <Card className="flex flex-col gap-3 p-3 sm:flex-row sm:items-center sm:justify-between">
        <Tabs
          items={CLASS_TABS}
          value={assetClass}
          onChange={setAssetClass}
          variant="pill"
          size="sm"
          aria-label="Filter by asset class"
        />
        <div className="relative w-full sm:max-w-xs">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" aria-hidden />
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search symbol, name, sector…"
            aria-label="Search the universe"
            className={cn(
              'h-9 w-full rounded-xl border border-border bg-surface pl-9 pr-9 text-sm text-text placeholder:text-muted',
              'transition-colors hover:border-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
            )}
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch('')}
              aria-label="Clear search"
              className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded-md p-0.5 text-muted hover:text-text"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </Card>

      {/* Desktop table */}
      <Card flush className="hidden overflow-hidden md:block">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-border bg-surface-2/60">
                {COLUMNS.map((col) => {
                  const active = sortKey === col.key;
                  return (
                    <th
                      key={col.key}
                      scope="col"
                      className={cn(
                        'whitespace-nowrap px-3 py-2.5 text-xs font-semibold text-muted',
                        col.numeric ? 'text-right' : 'text-left',
                      )}
                    >
                      <button
                        type="button"
                        onClick={() => onSort(col.key)}
                        className={cn(
                          'inline-flex items-center gap-1 rounded-md px-1 py-0.5 transition-colors hover:text-text',
                          col.numeric ? 'flex-row-reverse' : '',
                          active && 'text-text',
                        )}
                        aria-label={`Sort by ${col.label}`}
                      >
                        {active ? (
                          sortDir === 'asc' ? (
                            <ArrowUp className="h-3 w-3" aria-hidden />
                          ) : (
                            <ArrowDown className="h-3 w-3" aria-hidden />
                          )
                        ) : (
                          <ArrowUpDown className="h-3 w-3 opacity-40" aria-hidden />
                        )}
                        {col.label}
                      </button>
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                Array.from({ length: 10 }).map((_, i) => (
                  <tr key={i} className="border-b border-border last:border-0">
                    {COLUMNS.map((col) => (
                      <td key={col.key} className="px-3 py-3">
                        <Skeleton className={cn('h-4', col.numeric ? 'ml-auto w-12' : 'w-28')} />
                      </td>
                    ))}
                  </tr>
                ))
              ) : filtered.length === 0 ? (
                <tr>
                  <td colSpan={COLUMNS.length} className="px-3 py-10 text-center text-sm text-muted">
                    No assets match your filters.
                  </td>
                </tr>
              ) : (
                filtered.map((row) => (
                  <tr
                    key={row.asset.symbol}
                    onClick={() => open(row.asset.symbol)}
                    tabIndex={0}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        open(row.asset.symbol);
                      }
                    }}
                    className="cursor-pointer border-b border-border transition-colors last:border-0 hover:bg-surface-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-DEFAULT"
                  >
                    <td className="px-3 py-2.5">
                      <div className="flex items-center gap-2">
                        <div className="min-w-0">
                          <div className="flex items-center gap-1.5">
                            <span className="text-sm font-semibold tracking-tight text-text">
                              {row.asset.symbol}
                            </span>
                            <Badge tone="neutral" size="sm" variant="soft">
                              {assetClassLabel(row.asset.assetClass)}
                            </Badge>
                          </div>
                          <span className="block max-w-[11.25rem] truncate text-xs text-muted" title={row.asset.name}>
                            {row.asset.name}
                          </span>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-2.5 text-right">
                      <span className="text-sm tnum text-text">{formatPrice(row.price, row.asset.currency)}</span>
                      {row.asset.marketCap != null && (
                        <span className="block text-[0.625rem] tnum text-muted">
                          {formatCompactCurrency(row.asset.marketCap, row.asset.currency)} cap
                        </span>
                      )}
                    </td>
                    <td className={cn('px-3 py-2.5 text-right text-sm font-medium tnum', changeTextColor(row.changePct))}>
                      {formatPct(row.changePct, { digits: 2, sign: true })}
                    </td>
                    <td className="px-3 py-2.5 text-right">
                      <CompositeCell row={row} />
                    </td>
                    <td className="px-3 py-2.5 text-right text-sm tnum text-text">
                      <MetricCell loading={row.loading} value={row.sharpe} render={(v) => formatRatio(v, 2)} />
                    </td>
                    <td className="px-3 py-2.5 text-right text-sm tnum text-text">
                      <MetricCell loading={row.loading} value={row.annualVol} render={(v) => formatPct(v, { digits: 1 })} />
                    </td>
                    <td className="px-3 py-2.5 text-right">
                      <MetricCell
                        loading={row.loading}
                        value={row.return1YPct}
                        render={(v) => (
                          <span className={cn('text-sm font-medium tnum', changeTextColor(v))}>
                            {formatPct(v, { digits: 1, sign: true })}
                          </span>
                        )}
                      />
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Mobile cards */}
      <div className="space-y-2.5 md:hidden">
        {isLoading ? (
          Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-24 w-full rounded-2xl" />)
        ) : filtered.length === 0 ? (
          <Card className="py-8 text-center text-sm text-muted">No assets match your filters.</Card>
        ) : (
          filtered.map((row) => (
            <Card
              key={row.asset.symbol}
              interactive
              onClick={() => open(row.asset.symbol)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  open(row.asset.symbol);
                }
              }}
              className="cursor-pointer"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="text-sm font-semibold tracking-tight text-text">{row.asset.symbol}</span>
                    <Badge tone="neutral" size="sm">
                      {assetClassLabel(row.asset.assetClass)}
                    </Badge>
                  </div>
                  <span className="block truncate text-xs text-muted">{row.asset.name}</span>
                </div>
                <div className="text-right">
                  <span className="block text-sm font-semibold tnum text-text">
                    {formatPrice(row.price, row.asset.currency)}
                  </span>
                  <span className={cn('block text-xs font-medium tnum', changeTextColor(row.changePct))}>
                    {formatPct(row.changePct, { digits: 2, sign: true })}
                  </span>
                </div>
              </div>
              <div className="mt-3 grid grid-cols-4 gap-2">
                <MobileStat label="Score">
                  <CompositeCell row={row} compact />
                </MobileStat>
                <MobileStat label="Sharpe">
                  <MetricCell loading={row.loading} value={row.sharpe} render={(v) => formatRatio(v, 2)} />
                </MobileStat>
                <MobileStat label="Vol">
                  <MetricCell loading={row.loading} value={row.annualVol} render={(v) => formatPct(v, { digits: 0 })} />
                </MobileStat>
                <MobileStat label="1Y">
                  <MetricCell
                    loading={row.loading}
                    value={row.return1YPct}
                    render={(v) => (
                      <span className={cn('font-medium', changeTextColor(v))}>
                        {formatPct(v, { digits: 0, sign: true })}
                      </span>
                    )}
                  />
                </MobileStat>
              </div>
            </Card>
          ))
        )}
      </div>

      <p className="text-[0.6875rem] text-muted">
        Composite, Sharpe, vol and 1-year projection load per asset from the quant engine and update as
        they resolve. Educational simulation — not financial advice.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Cell helpers                                                        */
/* ------------------------------------------------------------------ */

function CompositeCell({ row, compact = false }: { row: ScreenerRow; compact?: boolean }): JSX.Element {
  if (row.composite === undefined) {
    return <Skeleton className={cn('h-4', compact ? 'mx-auto w-8' : 'ml-auto w-10')} />;
  }
  const stance = row.stance ?? stanceFromScore(row.composite);
  const score = Math.round(row.composite);
  if (compact) {
    return (
      <span className={cn('text-sm font-semibold tnum', changeTextColor(row.composite))}>
        {score > 0 ? '+' : ''}
        {score}
      </span>
    );
  }
  return (
    <div className="flex items-center justify-end gap-2">
      <span className="text-sm font-semibold tnum text-text">
        {score > 0 ? '+' : ''}
        {score}
      </span>
      <StanceBadge stance={stance} size="sm" />
    </div>
  );
}

function MetricCell({
  loading,
  value,
  render,
}: {
  loading: boolean;
  value: number | undefined;
  render: (v: number) => React.ReactNode;
}): JSX.Element {
  if (value === undefined || !Number.isFinite(value)) {
    return loading ? <Skeleton className="ml-auto h-4 w-10" /> : <span className="text-muted">—</span>;
  }
  return <>{render(value)}</>;
}

function MobileStat({ label, children }: { label: string; children: React.ReactNode }): JSX.Element {
  return (
    <div className="rounded-lg bg-surface-2 p-1.5 text-center">
      <span className="block text-[0.5625rem] uppercase tracking-wide text-muted">{label}</span>
      <span className="block text-xs tnum text-text">{children}</span>
    </div>
  );
}
