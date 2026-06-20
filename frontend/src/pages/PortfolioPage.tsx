/**
 * PortfolioPage — the analytical Markowitz optimizer.
 *
 * Pick a basket of assets, an objective (max Sharpe / min volatility / target
 * return) and a risk-free rate, then run `usePortfolioOpt` (POST
 * /api/portfolio/optimize). Renders the efficient frontier + capital-market line +
 * tangency portfolio (EfficientFrontierChart), the optimal weights as a donut +
 * bars, and the expected return / volatility / Sharpe stats. A CTA links to the
 * Invest page for putting real (simulated) money to work. Dense, responsive,
 * light/dark via tokens.
 */

import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, GaugeCircle, Plus, Sliders, TrendingUp, X } from 'lucide-react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Select, type SelectOption } from '@/components/ui/Select';
import { Tabs, type TabItem } from '@/components/ui/Tabs';
import { Skeleton } from '@/components/ui/Skeleton';
import { EfficientFrontierChart } from '@/components/charts/EfficientFrontierChart';
import { AllocationDonut } from '@/components/charts/AllocationDonut';
import { useAssets } from '@/hooks/useAssets';
import { usePortfolioOpt } from '@/hooks/usePortfolioOpt';
import { paletteColor, useChartTokens } from '@/theme/tokens';
import type {
  Asset,
  AssetClass,
  PortfolioRequest,
  PortfolioResult,
  Position,
} from '@/lib/types';
import { formatFractionPct, formatPct, formatRatio } from '@/lib/format';
import { assetClassLabel, cn } from '@/lib/utils';

type Objective = PortfolioRequest['objective'];

const OBJECTIVE_ITEMS: TabItem<Objective>[] = [
  { value: 'max_sharpe', label: 'Max Sharpe' },
  { value: 'min_volatility', label: 'Min Volatility' },
  { value: 'target_return', label: 'Target Return' },
];

type ClassFilter = 'all' | AssetClass;

const CLASS_ITEMS: TabItem<ClassFilter>[] = [
  { value: 'all', label: 'All' },
  { value: 'equity', label: 'Stocks' },
  { value: 'crypto', label: 'Crypto' },
  { value: 'etf', label: 'ETFs' },
];

/**
 * Synthesize the minimal {@link Position} shape the AllocationDonut needs from
 * optimizer weights (it slices by `marketValue`, so weight*100 is sufficient and
 * the labels/symbols carry through). Other Position fields are zeroed.
 */
function weightsToDonutPositions(result: PortfolioResult, assetsBySymbol: Map<string, Asset>): Position[] {
  return result.weights
    .filter((w) => w.weight > 0.0005)
    .map((w) => {
      const asset = assetsBySymbol.get(w.symbol);
      const base: Asset = asset ?? {
        symbol: w.symbol,
        name: w.symbol,
        assetClass: 'equity',
        sector: null,
        currency: 'USD',
        price: 0,
        change24hPct: 0,
        marketCap: null,
        volume24h: null,
      };
      return {
        symbol: w.symbol,
        asset: base,
        units: 0,
        costBasis: 0,
        avgPrice: 0,
        currentPrice: 0,
        marketValue: w.weight * 100,
        unrealizedPnl: 0,
        unrealizedPnlPct: 0,
        allocationPct: w.weight * 100,
        realizedPnl: 0,
        openedAt: 0,
      };
    });
}

export default function PortfolioPage(): JSX.Element {
  const tokens = useChartTokens();
  const [filter, setFilter] = useState<ClassFilter>('all');
  const [selected, setSelected] = useState<string[]>([]);
  const [pick, setPick] = useState<string>('');
  const [objective, setObjective] = useState<Objective>('max_sharpe');
  const [riskFreePct, setRiskFreePct] = useState<string>('4');
  const [targetPct, setTargetPct] = useState<string>('15');
  const [error, setError] = useState<string | null>(null);

  const assetsQuery = useAssets(filter === 'all' ? undefined : filter);
  const optimize = usePortfolioOpt();
  const result = optimize.data;

  const assets = useMemo<Asset[]>(() => assetsQuery.data ?? [], [assetsQuery.data]);
  const allAssets = useAssets();
  const assetsBySymbol = useMemo(() => {
    const map = new Map<string, Asset>();
    for (const a of allAssets.data ?? []) map.set(a.symbol, a);
    return map;
  }, [allAssets.data]);

  const available = useMemo<SelectOption<string>[]>(() => {
    const chosen = new Set(selected);
    return assets
      .filter((a) => !chosen.has(a.symbol))
      .map((a) => ({ value: a.symbol, label: `${a.symbol} · ${a.name}` }));
  }, [assets, selected]);

  const addSymbol = (symbol: string): void => {
    if (!symbol || selected.includes(symbol)) return;
    setSelected((prev) => [...prev, symbol]);
    setPick('');
  };
  const removeSymbol = (symbol: string): void => setSelected((prev) => prev.filter((s) => s !== symbol));

  const run = (): void => {
    setError(null);
    if (selected.length < 2) {
      setError('Pick at least two assets to optimize a portfolio.');
      return;
    }
    const riskFreeRate = Number(riskFreePct) / 100;
    const targetReturn = objective === 'target_return' ? Number(targetPct) / 100 : null;
    if (objective === 'target_return' && (!Number.isFinite(targetReturn) || (targetReturn ?? 0) <= 0)) {
      setError('Enter a positive target return.');
      return;
    }
    const req: PortfolioRequest = {
      symbols: selected,
      riskFreeRate: Number.isFinite(riskFreeRate) ? riskFreeRate : 0.04,
      objective,
      targetReturn,
    };
    optimize.mutate(req, {
      onError: (err) => setError(err instanceof Error ? err.message : 'Optimization failed.'),
    });
  };

  const donutPositions = useMemo(
    () => (result ? weightsToDonutPositions(result, assetsBySymbol) : []),
    [result, assetsBySymbol],
  );
  const sortedWeights = useMemo(
    () => (result ? [...result.weights].filter((w) => w.weight > 0.0005).sort((a, b) => b.weight - a.weight) : []),
    [result],
  );

  return (
    <div className="flex flex-col gap-4">
      {/* Heading */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-text">
            <GaugeCircle className="h-5 w-5 text-primary" aria-hidden />
            Portfolio Optimizer
          </h1>
          <p className="mt-1 text-sm text-muted">
            Mean-variance optimization (Markowitz). Pick a basket and objective to find the efficient frontier and the
            optimal weights.
          </p>
        </div>
        <Link
          to="/invest"
          className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-xl border border-border bg-surface px-3 text-sm font-medium text-text transition-colors hover:border-primary hover:text-primary"
        >
          Put money to work
          <ArrowRight className="h-4 w-4" aria-hidden />
        </Link>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3 lg:gap-4">
        {/* Inputs */}
        <Card className="flex flex-col gap-4 lg:col-span-1">
          <CardHeader>
            <CardTitle icon={<Sliders className="h-4 w-4" />}>Inputs</CardTitle>
          </CardHeader>

          {/* Asset picker */}
          <div className="flex flex-col gap-2">
            <Tabs
              items={CLASS_ITEMS}
              value={filter}
              onChange={setFilter}
              variant="pill"
              size="sm"
              aria-label="Filter assets by class"
              className="w-fit"
            />
            <div className="flex items-end gap-2">
              <div className="flex-1">
                <Select
                  options={available}
                  value={pick}
                  onChange={setPick}
                  placeholder={assetsQuery.isPending ? 'Loading…' : 'Add an asset…'}
                  size="sm"
                  fullWidth
                  aria-label="Choose an asset"
                />
              </div>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => addSymbol(pick)}
                disabled={!pick}
                leftIcon={<Plus className="h-3.5 w-3.5" />}
              >
                Add
              </Button>
            </div>

            {selected.length === 0 ? (
              <p className="rounded-xl border border-dashed border-border py-4 text-center text-xs text-muted">
                No assets selected yet.
              </p>
            ) : (
              <ul className="flex flex-wrap gap-1.5">
                {selected.map((s) => {
                  const asset = assetsBySymbol.get(s);
                  return (
                    <li key={s}>
                      <span className="inline-flex items-center gap-1 rounded-lg border border-border bg-surface-2 py-1 pl-2 pr-1 text-xs">
                        <span className="font-medium text-text">{s}</span>
                        {asset && <span className="text-muted">{assetClassLabel(asset.assetClass)}</span>}
                        <button
                          type="button"
                          onClick={() => removeSymbol(s)}
                          aria-label={`Remove ${s}`}
                          className="flex h-4 w-4 items-center justify-center rounded text-muted hover:text-danger"
                        >
                          <X className="h-3 w-3" aria-hidden />
                        </button>
                      </span>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>

          {/* Objective */}
          <div className="flex flex-col gap-1.5">
            <span className="text-xs font-medium text-muted">Objective</span>
            <Tabs items={OBJECTIVE_ITEMS} value={objective} onChange={setObjective} variant="pill" size="sm" aria-label="Objective" />
          </div>

          {/* Risk-free + target */}
          <div className="grid grid-cols-2 gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-xs font-medium text-muted">Risk-free (%)</span>
              <div className="relative">
                <input
                  type="number"
                  inputMode="decimal"
                  step="0.1"
                  value={riskFreePct}
                  onChange={(e) => setRiskFreePct(e.target.value)}
                  className="h-9 w-full rounded-xl border border-border bg-surface px-3 text-sm tnum text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT"
                  aria-label="Risk-free rate percent"
                />
              </div>
            </label>
            <label className={cn('flex flex-col gap-1', objective !== 'target_return' && 'opacity-50')}>
              <span className="text-xs font-medium text-muted">Target return (%)</span>
              <input
                type="number"
                inputMode="decimal"
                step="0.5"
                value={targetPct}
                onChange={(e) => setTargetPct(e.target.value)}
                disabled={objective !== 'target_return'}
                className="h-9 w-full rounded-xl border border-border bg-surface px-3 text-sm tnum text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT disabled:cursor-not-allowed"
                aria-label="Target return percent"
              />
            </label>
          </div>

          {error && (
            <p className="rounded-lg bg-danger/10 px-3 py-2 text-xs font-medium text-danger" role="alert">
              {error}
            </p>
          )}

          <Button variant="primary" size="lg" fullWidth onClick={run} loading={optimize.isPending} disabled={selected.length < 2}>
            Optimize
          </Button>
        </Card>

        {/* Results */}
        <div className="flex flex-col gap-3 lg:col-span-2 lg:gap-4">
          {/* Stats */}
          <div className="grid grid-cols-3 gap-3 lg:gap-4">
            <StatTile
              label="Expected return"
              value={result ? formatFractionPct(result.expectedReturn, { digits: 1, sign: true }) : '—'}
              tone={result && result.expectedReturn >= 0 ? 'text-success' : result ? 'text-danger' : undefined}
              icon={<TrendingUp className="h-4 w-4" />}
            />
            <StatTile label="Volatility" value={result ? formatFractionPct(result.volatility, { digits: 1 }) : '—'} />
            <StatTile label="Sharpe" value={result ? formatRatio(result.sharpe) : '—'} />
          </div>

          {/* Frontier */}
          <Card className="flex flex-col gap-3">
            <CardHeader>
              <CardTitle>Efficient frontier</CardTitle>
              {result && (
                <Badge tone="warning" variant="soft" size="sm">
                  ★ tangency portfolio
                </Badge>
              )}
            </CardHeader>
            {optimize.isPending ? (
              <Skeleton className="h-[300px] w-full" />
            ) : result ? (
              <EfficientFrontierChart result={result} />
            ) : (
              <div className="flex flex-col items-center justify-center gap-2 py-16 text-center">
                <GaugeCircle className="h-7 w-7 text-muted" aria-hidden />
                <p className="text-sm font-medium text-text">No optimization yet</p>
                <p className="text-xs text-muted">Pick at least two assets and run the optimizer.</p>
              </div>
            )}
          </Card>

          {/* Weights: donut + bars */}
          {result && sortedWeights.length > 0 && (
            <Card className="flex flex-col gap-4">
              <CardHeader>
                <CardTitle>Optimal weights</CardTitle>
                <span className="text-[11px] text-muted">
                  risk-free {formatFractionPct(result.riskFreeRate, { digits: 1 })}
                </span>
              </CardHeader>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-[auto_1fr] sm:items-center">
                <AllocationDonut positions={donutPositions} showLegend={false} height={200} />
                <ul className="flex flex-col gap-2">
                  {sortedWeights.map((w, i) => (
                    <li key={w.symbol} className="flex items-center gap-3">
                      <span className="flex w-16 items-center gap-1.5 truncate text-xs font-medium text-text">
                        <span
                          className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm"
                          style={{ backgroundColor: paletteColor(tokens, i) }}
                          aria-hidden
                        />
                        {w.symbol}
                      </span>
                      <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${Math.max(2, w.weight * 100)}%`, backgroundColor: paletteColor(tokens, i) }}
                        />
                      </div>
                      <span className="w-12 shrink-0 text-right text-xs tnum text-muted">
                        {formatPct(w.weight * 100, { digits: 1 })}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function StatTile({
  label,
  value,
  tone,
  icon,
}: {
  label: string;
  value: string;
  tone?: string;
  icon?: JSX.Element;
}): JSX.Element {
  return (
    <Card className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 text-xs font-medium text-muted">
        {icon && (
          <span className="flex h-7 w-7 items-center justify-center rounded-xl bg-primary/12 text-primary">{icon}</span>
        )}
        <span className="tracking-tight">{label}</span>
      </div>
      <span className={cn('text-2xl font-semibold tracking-tight tnum text-text', tone)}>{value}</span>
    </Card>
  );
}
