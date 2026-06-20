/**
 * AssetDetailPage (`/asset/:symbol`) — the full per-asset workup.
 *
 * Pulls the composite analysis (`useAnalysis`), price history (`useCandles`) and
 * a Monte Carlo simulation (`useMonteCarlo`). Lays out: a live header (price +
 * 24h + regime), a PriceChart with a candle/area toggle, the ScoreGauge for the
 * composite + recommendation, the per-horizon projection table, the annualized
 * risk grid, the scenario fan + Monte Carlo distribution (horizon-selectable),
 * every strategy signal grouped by category, and the educational disclaimer.
 * Dense, responsive, light/dark via tokens.
 */

import { useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { AlertTriangle, ArrowLeft, Info } from 'lucide-react';
import { useAnalysis, useMonteCarlo } from '@/hooks/useAnalysis';
import { useCandles } from '@/hooks/useAssets';
import { useLivePrice } from '@/store/marketStore';
import { Card, CardDescription, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Tabs } from '@/components/ui/Tabs';
import { Select, type SelectOption } from '@/components/ui/Select';
import { Skeleton, SkeletonText } from '@/components/ui/Skeleton';
import { ScoreGauge } from '@/components/ui/ScoreGauge';
import { PriceChart } from '@/components/charts/PriceChart';
import { ScenarioFanChart } from '@/components/charts/ScenarioFanChart';
import { DistributionChart } from '@/components/charts/DistributionChart';
import { ChartEmpty } from '@/components/charts/ChartEmpty';
import { SignalCard } from '@/components/domain/SignalCard';
import { HorizonTable } from '@/components/domain/HorizonTable';
import { RiskMetricGrid } from '@/components/domain/RiskMetricGrid';
import { RegimeBadge } from '@/components/domain/RegimeBadge';
import type { Horizon, StrategyCategory, StrategySignal } from '@/lib/types';
import { HORIZONS } from '@/lib/types';
import { formatCompact, formatPct, formatPrice, horizonLabel } from '@/lib/format';
import { assetClassLabel, changeTextColor, cn } from '@/lib/utils';

type ChartType = 'candle' | 'area';

const HORIZON_OPTIONS: SelectOption<Horizon>[] = HORIZONS.map((h) => ({
  value: h,
  label: horizonLabel(h),
}));

/** Group strategy signals by category, preserving first-seen category order. */
function groupByCategory(signals: StrategySignal[]): [StrategyCategory, StrategySignal[]][] {
  const map = new Map<StrategyCategory, StrategySignal[]>();
  for (const s of signals) {
    const bucket = map.get(s.category);
    if (bucket) bucket.push(s);
    else map.set(s.category, [s]);
  }
  return Array.from(map.entries());
}

export default function AssetDetailPage(): JSX.Element {
  const { symbol = '' } = useParams<{ symbol: string }>();
  const [chartType, setChartType] = useState<ChartType>('area');
  const [mcHorizon, setMcHorizon] = useState<Horizon>('1Y');

  const analysis = useAnalysis(symbol);
  const candles = useCandles(symbol, '1d', 365);
  const monteCarlo = useMonteCarlo(symbol, mcHorizon, 2000);

  const live = useLivePrice(symbol);
  const data = analysis.data;
  const asset = data?.asset;
  const price = live?.price ?? asset?.price ?? 0;
  const changePct = live?.changePct ?? asset?.change24hPct ?? 0;

  const grouped = useMemo(() => groupByCategory(data?.signals ?? []), [data?.signals]);

  /* ---- Error state ---- */
  if (analysis.isError) {
    return (
      <Card className="flex flex-col items-center gap-3 py-16 text-center">
        <AlertTriangle className="h-6 w-6 text-danger" aria-hidden />
        <div>
          <p className="text-sm font-medium text-text">Couldn&apos;t load {symbol || 'this asset'}</p>
          <p className="mt-1 text-xs text-muted">
            {analysis.error instanceof Error ? analysis.error.message : 'Please try again.'}
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={() => void analysis.refetch()}>
            Retry
          </Button>
          <Link
            to="/recommendations"
            className="inline-flex h-8 items-center rounded-xl border border-border px-3 text-xs font-medium text-text hover:bg-surface-2"
          >
            Back to recommendations
          </Link>
        </div>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Breadcrumb / back */}
      <Link
        to="/recommendations"
        className="inline-flex w-fit items-center gap-1.5 text-xs font-medium text-muted transition-colors hover:text-text"
      >
        <ArrowLeft className="h-3.5 w-3.5" aria-hidden />
        Recommendations
      </Link>

      {/* Header */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          {analysis.isPending || !asset ? (
            <>
              <Skeleton className="h-7 w-40" />
              <Skeleton className="mt-2 h-4 w-56" />
            </>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="text-xl font-semibold tracking-tight text-text">{asset.symbol}</h1>
                <Badge tone="neutral">{assetClassLabel(asset.assetClass)}</Badge>
                {asset.sector && (
                  <Badge tone="neutral" variant="outline">
                    {asset.sector}
                  </Badge>
                )}
                {data?.regime && <RegimeBadge regime={data.regime} />}
              </div>
              <p className="mt-1 line-clamp-1 text-sm text-muted">{asset.name}</p>
            </>
          )}
        </div>

        {asset && (
          <div className="flex shrink-0 items-end gap-4">
            <div className="flex flex-col items-end">
              <span className="text-2xl font-semibold tracking-tight tnum text-text">
                {formatPrice(price, asset.currency)}
              </span>
              <span className={cn('text-sm font-medium tnum', changeTextColor(changePct))}>
                {formatPct(changePct, { sign: true })} <span className="text-muted">24h</span>
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Price chart + composite gauge */}
      <div className="grid gap-3 lg:grid-cols-[1fr_320px] lg:gap-4">
        <Card flush className="p-4">
          <CardHeader>
            <CardTitle>Price history</CardTitle>
            <Tabs
              items={[
                { value: 'area', label: 'Area' },
                { value: 'candle', label: 'Candles' },
              ]}
              value={chartType}
              onChange={setChartType}
              variant="pill"
              size="sm"
              aria-label="Price chart style"
            />
          </CardHeader>
          <div className="mt-3">
            {candles.isPending ? (
              <Skeleton className="h-[320px] w-full" />
            ) : (
              <PriceChart candles={candles.data ?? []} type={chartType} height={320} />
            )}
          </div>
        </Card>

        <Card className="flex flex-col items-center justify-center gap-3">
          <CardTitle className="self-start">Composite signal</CardTitle>
          {analysis.isPending || !data ? (
            <Skeleton circle className="h-40 w-40" />
          ) : (
            <>
              <ScoreGauge
                score={data.compositeScore}
                stance={data.recommendation}
                size={184}
                caption={`${Math.round(data.confidence * 100)}% confidence · ${data.strategyCount} strategies`}
              />
              <p className="text-center text-xs leading-relaxed text-muted">{data.rationaleSummary}</p>
            </>
          )}
        </Card>
      </div>

      {/* Top reasons */}
      {data && data.topReasons.length > 0 && (
        <Card>
          <CardTitle>Why this rating</CardTitle>
          <ul className="mt-3 grid gap-2 sm:grid-cols-2">
            {data.topReasons.map((reason, i) => (
              <li key={i} className="flex gap-2 text-sm text-muted">
                <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" aria-hidden />
                <span className="leading-relaxed">{reason}</span>
              </li>
            ))}
          </ul>
        </Card>
      )}

      {/* Projections: horizon table + scenario fan */}
      <div className="grid gap-3 lg:grid-cols-2 lg:gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Projected returns</CardTitle>
            <CardDescription>Bear / base / bull · probability of a gain · tail risk</CardDescription>
          </CardHeader>
          <div className="mt-3">
            {analysis.isPending || !data ? (
              <Skeleton className="h-44 w-full" />
            ) : (
              <HorizonTable expectedReturns={data.expectedReturns} />
            )}
          </div>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Scenario fan</CardTitle>
            <CardDescription>Total return across the five horizons</CardDescription>
          </CardHeader>
          <div className="mt-3">
            {analysis.isPending || !data ? (
              <Skeleton className="h-[280px] w-full" />
            ) : (
              <ScenarioFanChart expectedReturns={data.expectedReturns} height={280} />
            )}
          </div>
        </Card>
      </div>

      {/* Risk metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Risk profile</CardTitle>
          <CardDescription>Annualized, from the simulated return series</CardDescription>
        </CardHeader>
        <div className="mt-3">
          {analysis.isPending || !data ? (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-16 w-full" />
              ))}
            </div>
          ) : (
            <RiskMetricGrid metrics={data.riskMetrics} />
          )}
        </div>
      </Card>

      {/* Monte Carlo distribution */}
      <Card>
        <CardHeader>
          <div>
            <CardTitle>Monte Carlo outlook</CardTitle>
            <CardDescription>Simulated terminal-price distribution</CardDescription>
          </div>
          <Select
            options={HORIZON_OPTIONS}
            value={mcHorizon}
            onChange={setMcHorizon}
            size="sm"
            aria-label="Monte Carlo horizon"
          />
        </CardHeader>

        {monteCarlo.isError ? (
          <ChartEmpty className="mt-3" height={280} label="Simulation unavailable" />
        ) : monteCarlo.isPending || !monteCarlo.data ? (
          <Skeleton className="mt-3 h-[280px] w-full" />
        ) : (
          <>
            <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
              <McStat
                label="Expected"
                value={formatPct(monteCarlo.data.expectedReturnPct, { sign: true })}
                tone={changeTextColor(monteCarlo.data.expectedReturnPct)}
              />
              <McStat label="P(profit)" value={`${Math.round(monteCarlo.data.probPositive * 100)}%`} />
              <McStat
                label="VaR 95%"
                value={formatPct(-Math.abs(monteCarlo.data.var95Pct))}
                tone="text-danger"
              />
              <McStat
                label="CVaR 95%"
                value={formatPct(-Math.abs(monteCarlo.data.cvar95Pct))}
                tone="text-danger"
              />
            </div>
            <div className="mt-3">
              <DistributionChart result={monteCarlo.data} height={280} />
            </div>
            <p className="mt-2 text-[11px] text-muted">
              {formatCompact(monteCarlo.data.sims, 0)} simulated paths over {monteCarlo.data.steps}{' '}
              steps · {horizonLabel(monteCarlo.data.horizon)} horizon.
            </p>
          </>
        )}
      </Card>

      {/* Strategy signals grouped by category */}
      <div className="flex flex-col gap-3">
        <div className="flex items-baseline justify-between">
          <h2 className="text-base font-semibold tracking-tight text-text">Strategy signals</h2>
          {data && (
            <span className="text-xs text-muted">
              {data.signals.length} model{data.signals.length === 1 ? '' : 's'}
            </span>
          )}
        </div>

        {analysis.isPending || !data ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 6 }).map((_, i) => (
              <Card key={i} muted className="p-3">
                <Skeleton className="h-4 w-32" />
                <SkeletonText className="mt-3" lines={3} />
              </Card>
            ))}
          </div>
        ) : data.signals.length === 0 ? (
          <Card className="py-10 text-center text-sm text-muted">No strategy signals for this asset.</Card>
        ) : (
          grouped.map(([category, signals]) => (
            <section key={category} className="flex flex-col gap-2.5">
              <div className="flex items-center gap-2">
                <h3 className="text-sm font-semibold tracking-tight text-text">{category}</h3>
                <Badge tone="neutral" size="sm">
                  {signals.length}
                </Badge>
              </div>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {signals.map((signal) => (
                  <SignalCard key={signal.strategyId} signal={signal} />
                ))}
              </div>
            </section>
          ))
        )}
      </div>

      {/* Disclaimer */}
      <p className="flex items-start gap-2 rounded-xl border border-border bg-surface-2/60 px-3 py-2.5 text-xs leading-relaxed text-muted">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" aria-hidden />
        <span>
          {data?.disclaimer ??
            'For educational and informational use only. Not investment advice. Figures are model-derived from simulated data.'}
        </span>
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Local helpers                                                       */
/* ------------------------------------------------------------------ */

function McStat({
  label,
  value,
  tone = 'text-text',
}: {
  label: string;
  value: string;
  tone?: string;
}): JSX.Element {
  return (
    <div className="flex flex-col gap-0.5 rounded-xl border border-border bg-surface-2 px-3 py-2.5">
      <span className="text-[11px] font-medium text-muted">{label}</span>
      <span className={cn('text-base font-semibold tracking-tight tnum', tone)}>{value}</span>
    </div>
  );
}
