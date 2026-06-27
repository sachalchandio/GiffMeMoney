/**
 * DashboardPage — the "where to invest now" home.
 *
 * Composes:
 *  - a live MarketTicker strip (marketStore) seeded from the universe;
 *  - market-breadth + index stat row (useMarketSummary);
 *  - a featured #1 opportunity with its ScenarioFanChart + regime badge
 *    (useRecommendations → useAnalysis of the top pick);
 *  - a sector heatmap (useMarketSummary);
 *  - a grid of top opportunities (OpportunityCard);
 *  - top movers (gainers / losers) lists;
 *  - a portfolio snapshot CTA (usePortfolioState).
 *
 * Every async section renders a skeleton while loading. Fully responsive and
 * theme-correct (semantic tokens only).
 */

import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  ArrowDownRight,
  Sparkles,
  TrendingUp,
  Wallet as WalletIcon,
  Gauge,
} from 'lucide-react';
import { Card, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { StatCard } from '@/components/ui/StatCard';
import { StanceBadge } from '@/components/ui/StanceBadge';
import { Skeleton, SkeletonCard } from '@/components/ui/Skeleton';
import { EasyOnly, ExpertOnly, WhatThisMeans } from '@/components/ui/ModeView';
import { useUiMode } from '@/theme/UiModeProvider';
import { stanceLabel } from '@/lib/format';
import { ScenarioFanChart } from '@/components/charts/ScenarioFanChart';
import { MarketTicker } from '@/components/domain/MarketTicker';
import { SectorHeatmap } from '@/components/domain/SectorHeatmap';
import { OpportunityCard } from '@/components/domain/OpportunityCard';
import { useAssets } from '@/hooks/useAssets';
import { useMarketSummary, useRecommendations } from '@/hooks/useRecommendations';
import { useAnalysis } from '@/hooks/useAnalysis';
import { usePortfolioState } from '@/hooks/usePortfolioState';
import { useMarketStore } from '@/store/marketStore';
import type { Recommendation, RegimeInfo } from '@/lib/types';
import { cn, changeTextColor } from '@/lib/utils';
import { formatPrice, formatPct, formatCompactCurrency, formatNumber } from '@/lib/format';

/* ------------------------------------------------------------------ */
/* Regime badge                                                        */
/* ------------------------------------------------------------------ */

function RegimeBadge({ regime }: { regime: RegimeInfo }): JSX.Element {
  const tone =
    regime.regime === 'bull' ? 'success' : regime.regime === 'bear' ? 'danger' : 'neutral';
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Badge tone={tone} variant="soft" size="sm" icon={<Activity className="h-3 w-3" />}>
        {regime.regime.charAt(0).toUpperCase() + regime.regime.slice(1)} regime
      </Badge>
      <Badge
        tone={regime.volRegime === 'high' ? 'warning' : 'neutral'}
        variant="outline"
        size="sm"
      >
        {regime.volRegime.charAt(0).toUpperCase() + regime.volRegime.slice(1)} vol
      </Badge>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Featured #1 opportunity                                             */
/* ------------------------------------------------------------------ */

function FeaturedPick({ pick }: { pick: Recommendation }): JSX.Element {
  const analysis = useAnalysis(pick.asset.symbol);
  const live = useMarketStore((s) => s.prices[pick.asset.symbol.toUpperCase()]);
  const price = live?.price ?? pick.asset.price;
  const changePct = live?.changePct ?? pick.asset.change24hPct;
  const expectedReturns = analysis.data?.expectedReturns ?? [];
  const regime = analysis.data?.regime ?? null;

  return (
    <Card className="flex h-full flex-col gap-4 p-4">
      <CardHeader>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Badge tone="primary" variant="soft" size="sm" icon={<Sparkles className="h-3 w-3" />}>
              Top pick
            </Badge>
            {regime && <RegimeBadge regime={regime} />}
          </div>
          <Link
            to={`/asset/${encodeURIComponent(pick.asset.symbol)}`}
            className="mt-2 flex items-baseline gap-2 hover:underline"
          >
            <span className="text-xl font-semibold tracking-tight text-text">{pick.asset.symbol}</span>
            <span className="truncate text-sm text-muted">{pick.asset.name}</span>
          </Link>
        </div>
        <StanceBadge stance={pick.recommendation} />
      </CardHeader>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric label="Price" value={formatPrice(price, pick.asset.currency)} sub={formatPct(changePct, { digits: 2, sign: true })} subColor={changeTextColor(changePct)} />
        <Metric
          label="Composite"
          value={`${pick.compositeScore > 0 ? '+' : ''}${Math.round(pick.compositeScore)}`}
        />
        <Metric
          label="1Y projection"
          value={formatPct(pick.expectedReturn1YPct, { digits: 1, sign: true })}
          valueColor={changeTextColor(pick.expectedReturn1YPct)}
        />
        <Metric label="Confidence" value={`${Math.round(pick.confidence * 100)}%`} />
      </div>

      <div>
        <div className="mb-1 flex items-center gap-2 text-xs font-medium text-muted">
          <TrendingUp className="h-3.5 w-3.5" aria-hidden />
          Scenario projection across horizons
        </div>
        {analysis.isLoading ? (
          <Skeleton className="h-[13.75rem] w-full" />
        ) : (
          <ScenarioFanChart expectedReturns={expectedReturns} height={220} />
        )}
      </div>

      <WhatThisMeans>
        Our models rate <b>{pick.asset.symbol}</b> a <b>{stanceLabel(pick.recommendation)}</b>. Over the
        next year they project about{' '}
        <b>{formatPct(pick.expectedReturn1YPct, { sign: true, digits: 1 })}</b>, with{' '}
        <b>{Math.round(pick.confidence * 100)}%</b> confidence — an estimate, never a promise.
      </WhatThisMeans>

      <p className="text-xs leading-relaxed text-text">{pick.headline}</p>

      <div className="mt-auto flex items-center justify-between gap-2 border-t border-border pt-3">
        <span className="text-[0.6875rem] text-muted">Educational simulation — not financial advice.</span>
        <Link to={`/asset/${encodeURIComponent(pick.asset.symbol)}`}>
          <Button variant="outline" size="sm" rightIcon={<ArrowRight className="h-3.5 w-3.5" />}>
            Full analysis
          </Button>
        </Link>
      </div>
    </Card>
  );
}

function Metric({
  label,
  value,
  sub,
  valueColor,
  subColor,
}: {
  label: string;
  value: string;
  sub?: string;
  valueColor?: string;
  subColor?: string;
}): JSX.Element {
  return (
    <div className="rounded-xl bg-surface-2 p-2.5">
      <span className="block text-[0.625rem] uppercase tracking-wide text-muted">{label}</span>
      <span className={cn('text-base font-semibold tnum text-text', valueColor)}>{value}</span>
      {sub && <span className={cn('block text-[0.6875rem] font-medium tnum text-muted', subColor)}>{sub}</span>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Top movers                                                          */
/* ------------------------------------------------------------------ */

function MoverRow({ rec, up }: { rec: Recommendation; up: boolean }): JSX.Element {
  const live = useMarketStore((s) => s.prices[rec.asset.symbol.toUpperCase()]);
  const changePct = live?.changePct ?? rec.asset.change24hPct;
  const price = live?.price ?? rec.asset.price;
  return (
    <Link
      to={`/asset/${encodeURIComponent(rec.asset.symbol)}`}
      className="flex items-center justify-between gap-2 rounded-xl px-2 py-1.5 transition-colors hover:bg-surface-2"
    >
      <div className="flex min-w-0 items-center gap-2">
        <span
          className={cn(
            'flex h-6 w-6 shrink-0 items-center justify-center rounded-lg',
            up ? 'bg-success/12 text-success' : 'bg-danger/12 text-danger',
          )}
        >
          {up ? <ArrowUpRight className="h-3.5 w-3.5" /> : <ArrowDownRight className="h-3.5 w-3.5" />}
        </span>
        <div className="min-w-0">
          <span className="block text-xs font-semibold tracking-tight text-text">{rec.asset.symbol}</span>
          <span className="block truncate text-[0.6875rem] text-muted">{rec.asset.name}</span>
        </div>
      </div>
      <div className="text-right">
        <span className="block text-xs tnum text-text">{formatPrice(price, rec.asset.currency)}</span>
        <span className={cn('block text-[0.6875rem] font-medium tnum', changeTextColor(changePct))}>
          {formatPct(changePct, { digits: 2, sign: true })}
        </span>
      </div>
    </Link>
  );
}

/* ------------------------------------------------------------------ */
/* Portfolio snapshot CTA                                              */
/* ------------------------------------------------------------------ */

function PortfolioSnapshot(): JSX.Element {
  const state = usePortfolioState();

  if (state.isLoading) {
    return <SkeletonCard />;
  }

  const data = state.data;
  const hasPositions = (data?.positions.length ?? 0) > 0;

  return (
    <Card className="flex h-full flex-col gap-3 p-4">
      <CardHeader>
        <CardTitle icon={<WalletIcon className="h-4 w-4" />}>Your portfolio</CardTitle>
        <Link to="/invest">
          <Button variant="ghost" size="sm" rightIcon={<ArrowRight className="h-3.5 w-3.5" />}>
            Open
          </Button>
        </Link>
      </CardHeader>

      {data && hasPositions ? (
        <>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <span className="block text-[0.625rem] uppercase tracking-wide text-muted">Total value</span>
              <span className="text-xl font-semibold tnum text-text">
                {formatCompactCurrency(data.totalValue, data.wallet.currency)}
              </span>
            </div>
            <div>
              <span className="block text-[0.625rem] uppercase tracking-wide text-muted">Total P&amp;L</span>
              <span className={cn('text-xl font-semibold tnum', changeTextColor(data.totalPnl))}>
                {formatPct(data.totalPnlPct, { digits: 2, sign: true })}
              </span>
            </div>
          </div>
          <div className="mt-auto flex items-center justify-between gap-2 text-xs text-muted">
            <span>
              {data.positions.length} position{data.positions.length === 1 ? '' : 's'} ·{' '}
              {formatCompactCurrency(data.wallet.cashBalance, data.wallet.currency)} cash
            </span>
          </div>
        </>
      ) : (
        <div className="flex flex-1 flex-col items-start justify-center gap-2">
          <p className="text-sm text-text">You haven&apos;t invested yet.</p>
          <p className="text-xs text-muted">
            Fund a simulated wallet and put the recommendations to work.
          </p>
          <Link to="/invest" className="mt-1">
            <Button variant="primary" size="sm" leftIcon={<WalletIcon className="h-3.5 w-3.5" />}>
              Add funds &amp; invest
            </Button>
          </Link>
        </div>
      )}
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function DashboardPage(): JSX.Element {
  const { isEasy } = useUiMode();
  const assets = useAssets();
  const summary = useMarketSummary();
  const recs = useRecommendations(9);

  const recommendations = recs.data ?? [];
  const featured = recommendations[0];
  const rest = recommendations.slice(1);

  const breadth = summary.data?.breadth;
  const totalBreadth = breadth ? breadth.advancers + breadth.decliners + breadth.unchanged : 0;
  const advPct = totalBreadth > 0 ? ((breadth?.advancers ?? 0) / totalBreadth) * 100 : 0;

  const indices = summary.data?.indices ?? [];
  const topGainers = summary.data?.topGainers ?? [];
  const topLosers = summary.data?.topLosers ?? [];

  const tickerAssets = useMemo(() => assets.data ?? [], [assets.data]);

  return (
    <div className="space-y-4 lg:space-y-5">
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-lg font-semibold tracking-tight text-text lg:text-xl">
          Where to invest now
        </h1>
        <EasyOnly>
          <p className="text-sm text-muted">
            Your top idea right now, in plain language — picked by our models from{' '}
            {summary.data ? formatNumber(totalBreadth, 0) : '—'} markets we track.
          </p>
          <Link
            to="/guide"
            className="mt-1 inline-flex w-fit items-center gap-1.5 rounded-full border border-primary/30 bg-primary/[0.06] px-3 py-1 text-xs font-medium text-primary transition-colors hover:bg-primary/12"
          >
            <Sparkles className="h-3.5 w-3.5" aria-hidden />
            New here? Take the 2-minute tour
            <ArrowRight className="h-3.5 w-3.5" aria-hidden />
          </Link>
        </EasyOnly>
        <ExpertOnly>
          <p className="text-xs text-muted">
            Composite leaders from {summary.data ? formatNumber(totalBreadth, 0) : '—'} tracked assets,
            ranked by {recommendations.length} live quant models.
          </p>
        </ExpertOnly>
      </div>

      {/* Live ticker */}
      <MarketTicker assets={tickerAssets} />

      {/* Breadth + indices stat row (detail view) */}
      <ExpertOnly>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4 lg:gap-4">
        {summary.isLoading ? (
          Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} />)
        ) : (
          <>
            <StatCard
              label="Advancers"
              value={formatNumber(breadth?.advancers ?? 0, 0)}
              hint={`${advPct.toFixed(0)}% of market breadth`}
              icon={<TrendingUp className="h-4 w-4" />}
            />
            <StatCard
              label="Decliners"
              value={formatNumber(breadth?.decliners ?? 0, 0)}
              hint={`${formatNumber(breadth?.unchanged ?? 0, 0)} unchanged`}
              icon={<Gauge className="h-4 w-4" />}
            />
            {indices.slice(0, 2).map((idx) => (
              <StatCard
                key={idx.name}
                label={idx.name}
                value={formatNumber(idx.level, 2)}
                deltaPct={idx.changePct}
              />
            ))}
            {indices.length < 2 &&
              Array.from({ length: 2 - indices.length }).map((_, i) => (
                <StatCard key={`pad-${i}`} label="Index" value="—" />
              ))}
          </>
        )}
      </div>
      </ExpertOnly>

      {/* Featured pick (+ sector heatmap in Expert) */}
      <ExpertOnly>
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-3 lg:gap-4">
          <div className="lg:col-span-2">
            {recs.isLoading || !featured ? (
              <SkeletonCard className="h-[28.75rem]" />
            ) : (
              <FeaturedPick pick={featured} />
            )}
          </div>
          <Card className="flex flex-col gap-3 p-4">
            <CardHeader>
              <CardTitle icon={<Activity className="h-4 w-4" />}>Sector heatmap</CardTitle>
              <CardDescription>Today&apos;s sector moves</CardDescription>
            </CardHeader>
            {summary.isLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : (
              <SectorHeatmap sectors={summary.data?.sectors ?? []} />
            )}
          </Card>
        </div>
      </ExpertOnly>
      <EasyOnly>
        {recs.isLoading || !featured ? (
          <SkeletonCard className="h-[28.75rem]" />
        ) : (
          <FeaturedPick pick={featured} />
        )}
      </EasyOnly>

      {/* Top opportunities */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="flex items-center gap-2 text-sm font-semibold tracking-tight text-text">
            <Sparkles className="h-4 w-4 text-primary" aria-hidden />
            Top opportunities
          </h2>
          <Link to="/recommendations">
            <Button variant="ghost" size="sm" rightIcon={<ArrowRight className="h-3.5 w-3.5" />}>
              All recommendations
            </Button>
          </Link>
        </div>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4 lg:gap-4">
          {recs.isLoading
            ? Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={i} className="h-44" />)
            : rest.slice(0, 8).map((rec) => <OpportunityCard key={rec.asset.symbol} recommendation={rec} />)}
        </div>
      </div>

      {/* Movers (Expert) + portfolio snapshot (both modes) */}
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3 lg:gap-4">
        <ExpertOnly>
          <Card className="flex flex-col gap-2 p-4">
            <CardTitle icon={<ArrowUpRight className="h-4 w-4 text-success" />}>Top gainers</CardTitle>
            <div className="space-y-0.5">
              {summary.isLoading ? (
                Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-10 w-full" />)
              ) : topGainers.length > 0 ? (
                topGainers.slice(0, 5).map((rec) => <MoverRow key={rec.asset.symbol} rec={rec} up />)
              ) : (
                <p className="px-2 py-3 text-xs text-muted">No gainers right now.</p>
              )}
            </div>
          </Card>

          <Card className="flex flex-col gap-2 p-4">
            <CardTitle icon={<ArrowDownRight className="h-4 w-4 text-danger" />}>Top losers</CardTitle>
            <div className="space-y-0.5">
              {summary.isLoading ? (
                Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-10 w-full" />)
              ) : topLosers.length > 0 ? (
                topLosers.slice(0, 5).map((rec) => <MoverRow key={rec.asset.symbol} rec={rec} up={false} />)
              ) : (
                <p className="px-2 py-3 text-xs text-muted">No losers right now.</p>
              )}
            </div>
          </Card>
        </ExpertOnly>

        <div className={isEasy ? 'lg:col-span-3' : 'lg:col-span-1'}>
          <PortfolioSnapshot />
        </div>
      </div>
    </div>
  );
}
