/**
 * AllocationBuilder — split available cash across multiple assets.
 *
 * Pick assets by class tabs (Stocks / Crypto / ETFs / All), add rows, set a
 * dollar amount per asset (with a live % readout), and watch an
 * "allocated / remaining of cash" bar. "Suggest for me" runs the advisor for the
 * chosen risk tolerance and fills the rows; "Invest" posts the basket. All
 * allocation arithmetic lives in the exported {@link computeAllocation} helper so
 * it is unit-testable. Colors via semantic tokens only.
 */

import { useEffect, useMemo, useState } from 'react';
import { Loader2, Plus, Sparkles, Trash2, Wand2 } from 'lucide-react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Tabs, type TabItem } from '@/components/ui/Tabs';
import { Select, type SelectOption } from '@/components/ui/Select';
import { ProgressBar } from '@/components/ui/ProgressBar';
import { SyntheticDataBanner } from '@/components/domain/SyntheticDataBanner';
import { useAssets } from '@/hooks/useAssets';
import { useInvest } from '@/hooks/usePortfolioState';
import { useAdvisor } from '@/hooks/useAdvisor';
import type {
  AllocationAdvice,
  AllocationItem,
  Asset,
  AssetClass,
  InvestRequest,
  RiskTolerance,
} from '@/lib/types';
import { formatCurrency, formatPct } from '@/lib/format';
import { assetClassLabel, clamp, cn, round } from '@/lib/utils';

export interface AllocationBuilderProps {
  cash: number;
  /** Notify the page when the latest advisor result lands (drives the band). */
  onAdvice?: (advice: AllocationAdvice) => void;
  onInvested?: () => void;
  /**
   * Externally-supplied rows (e.g. picks pushed from the AdvicePanel). When the
   * reference changes the rows are replaced. The token bumps on every apply so
   * the same basket can be re-applied.
   */
  appliedRows?: { token: number; rows: AllocationRow[] } | null;
  className?: string;
}

/** One editable allocation row (a symbol + a dollar amount). */
export interface AllocationRow {
  symbol: string;
  amount: number;
}

/** Derived allocation totals + per-row weights, clamped to the cash budget. */
export interface AllocationSummary {
  allocated: number;
  remaining: number;
  /** allocated / cash, clamped to [0,1]. */
  fraction: number;
  /** Over-allocated beyond the cash budget. */
  overBudget: boolean;
  /** Per-symbol weight of the *allocated* total (sums to ~1 when allocated>0). */
  weights: Record<string, number>;
}

/**
 * Pure allocation arithmetic. Sums the row amounts (non-finite → 0), computes
 * the remaining budget and the fraction of cash used, flags over-budget, and
 * returns each row's share of the allocated total.
 */
export function computeAllocation(rows: AllocationRow[], cash: number): AllocationSummary {
  const allocated = rows.reduce((acc, r) => acc + (Number.isFinite(r.amount) && r.amount > 0 ? r.amount : 0), 0);
  const safeCash = Number.isFinite(cash) && cash > 0 ? cash : 0;
  const remaining = round(safeCash - allocated, 2);
  const fraction = safeCash > 0 ? clamp(allocated / safeCash, 0, 1) : 0;
  const weights: Record<string, number> = {};
  for (const r of rows) {
    const amt = Number.isFinite(r.amount) && r.amount > 0 ? r.amount : 0;
    weights[r.symbol] = allocated > 0 ? amt / allocated : 0;
  }
  return { allocated: round(allocated, 2), remaining, fraction, overBudget: allocated > safeCash + 0.005, weights };
}

type ClassFilter = 'all' | AssetClass;

const CLASS_ITEMS: TabItem<ClassFilter>[] = [
  { value: 'all', label: 'All' },
  { value: 'equity', label: 'Stocks' },
  { value: 'crypto', label: 'Crypto' },
  { value: 'etf', label: 'ETFs' },
];

const RISK_ITEMS: TabItem<RiskTolerance>[] = [
  { value: 'conservative', label: 'Conservative' },
  { value: 'balanced', label: 'Balanced' },
  { value: 'aggressive', label: 'Aggressive' },
];

export function AllocationBuilder({
  cash,
  onAdvice,
  onInvested,
  appliedRows,
  className,
}: AllocationBuilderProps): JSX.Element {
  const [filter, setFilter] = useState<ClassFilter>('all');
  const [risk, setRisk] = useState<RiskTolerance>('balanced');
  const [rows, setRows] = useState<AllocationRow[]>([]);
  const [pickSymbol, setPickSymbol] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  // Last advisor result, kept so we can surface its honesty banner + cash sleeve.
  const [lastAdvice, setLastAdvice] = useState<AllocationAdvice | null>(null);

  // Replace rows when the page applies an external basket (token bumps each apply).
  const appliedToken = appliedRows?.token ?? null;
  useEffect(() => {
    if (appliedRows && appliedRows.rows.length > 0) {
      setRows(appliedRows.rows.map((r) => ({ symbol: r.symbol, amount: round(r.amount, 2) })));
      setError(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [appliedToken]);

  const assetsQuery = useAssets(filter === 'all' ? undefined : filter);
  const advisor = useAdvisor();
  const invest = useInvest();

  const assets = useMemo<Asset[]>(() => assetsQuery.data ?? [], [assetsQuery.data]);
  const summary = useMemo(() => computeAllocation(rows, cash), [rows, cash]);

  const available = useMemo<SelectOption<string>[]>(() => {
    const chosen = new Set(rows.map((r) => r.symbol));
    return assets
      .filter((a) => !chosen.has(a.symbol))
      .map((a) => ({ value: a.symbol, label: `${a.symbol} · ${a.name}` }));
  }, [assets, rows]);

  const addRow = (symbol: string): void => {
    if (!symbol || rows.some((r) => r.symbol === symbol)) return;
    // Default each new row to an even split of the remaining budget.
    const suggested = round(Math.max(0, summary.remaining), 2);
    setRows((prev) => [...prev, { symbol, amount: suggested }]);
    setPickSymbol('');
  };

  const setRowAmount = (symbol: string, value: string): void => {
    const amount = Number(value);
    setRows((prev) => prev.map((r) => (r.symbol === symbol ? { ...r, amount: Number.isFinite(amount) ? amount : 0 } : r)));
  };

  const removeRow = (symbol: string): void => setRows((prev) => prev.filter((r) => r.symbol !== symbol));

  const suggest = (): void => {
    setError(null);
    const amount = cash > 0 ? round(cash, 2) : 1000;
    advisor.mutate(
      { amount, riskTolerance: risk, assetClasses: filter === 'all' ? null : [filter] },
      {
        onSuccess: (advice) => {
          setRows(advice.items.map((it) => ({ symbol: it.asset.symbol, amount: round(it.amount, 2) })));
          setLastAdvice(advice);
          onAdvice?.(advice);
        },
        onError: (err) => setError(err instanceof Error ? err.message : 'Could not fetch a suggestion.'),
      },
    );
  };

  const submitInvest = (): void => {
    setError(null);
    const allocations: AllocationItem[] = rows
      .filter((r) => Number.isFinite(r.amount) && r.amount > 0)
      .map((r) => ({ symbol: r.symbol, amount: round(r.amount, 2) }));
    if (allocations.length === 0) {
      setError('Add at least one funded allocation.');
      return;
    }
    if (summary.overBudget) {
      setError('Allocations exceed your available cash.');
      return;
    }
    const req: InvestRequest = { allocations };
    invest.mutate(req, {
      onSuccess: () => {
        setRows([]);
        onInvested?.();
      },
      onError: (err) => setError(err instanceof Error ? err.message : 'Invest failed.'),
    });
  };

  const canInvest = rows.some((r) => r.amount > 0) && !summary.overBudget && cash > 0 && !invest.isPending;

  return (
    <Card className={cn('flex flex-col gap-4', className)}>
      <CardHeader>
        <div>
          <CardTitle icon={<Sparkles className="h-4 w-4" />}>Build your allocation</CardTitle>
          <p className="mt-1 text-xs text-muted">Split your cash across assets, or let the advisor suggest a basket.</p>
        </div>
        <Badge tone="primary" variant="soft" size="sm">
          {formatCurrency(cash)} cash
        </Badge>
      </CardHeader>

      {/* Suggest-for-me */}
      <div className="flex flex-col gap-2 rounded-xl border border-border bg-surface-2/40 p-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <Wand2 className="h-4 w-4 text-accent" aria-hidden />
          <span className="text-xs font-medium text-text">Suggest for me</span>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <Tabs items={RISK_ITEMS} value={risk} onChange={setRisk} variant="pill" size="sm" aria-label="Risk tolerance" />
          <Button
            variant="accent"
            size="sm"
            onClick={suggest}
            loading={advisor.isPending}
            leftIcon={!advisor.isPending ? <Sparkles className="h-3.5 w-3.5" /> : undefined}
          >
            Suggest
          </Button>
        </div>
      </div>

      {/* Advisor honesty banner + cash-sleeve note (only after a suggestion) */}
      {lastAdvice && (
        <>
          <SyntheticDataBanner synthetic={lastAdvice.syntheticData} targetWarning={lastAdvice.targetWarning} />
          {(lastAdvice.cashWeight ?? 0) > 0.0005 && (
            <p className="text-[0.6875rem] text-muted">
              Advisor parked{' '}
              <span className="font-medium text-text">{formatPct((lastAdvice.cashWeight ?? 0) * 100, { digits: 0 })}</span>{' '}
              ({formatCurrency(lastAdvice.cashAmount ?? 0)}) as uninvested cash — the suggested rows below total the risky
              sleeve, not all of your cash.
            </p>
          )}
        </>
      )}

      {/* Class tabs + add row */}
      <div className="flex flex-col gap-2">
        <Tabs items={CLASS_ITEMS} value={filter} onChange={setFilter} variant="pill" size="sm" aria-label="Filter assets by class" className="w-fit" />
        <div className="flex items-end gap-2">
          <div className="flex-1">
            <Select
              options={available}
              value={pickSymbol}
              onChange={setPickSymbol}
              placeholder={assetsQuery.isPending ? 'Loading assets…' : 'Add an asset…'}
              size="sm"
              fullWidth
              aria-label="Choose an asset to add"
            />
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => addRow(pickSymbol)}
            disabled={!pickSymbol}
            leftIcon={<Plus className="h-3.5 w-3.5" />}
          >
            Add
          </Button>
        </div>
      </div>

      {/* Rows */}
      {rows.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border py-6 text-center text-xs text-muted">
          No allocations yet. Add an asset or tap Suggest.
        </div>
      ) : (
        <ul className="flex flex-col gap-2">
          {rows.map((row) => {
            const asset = assets.find((a) => a.symbol === row.symbol);
            const weight = summary.weights[row.symbol] ?? 0;
            return (
              <li key={row.symbol} className="flex items-center gap-3 rounded-xl border border-border bg-surface px-3 py-2">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold tracking-tight text-text">{row.symbol}</span>
                    {asset && (
                      <Badge tone="neutral" size="sm" className="hidden sm:inline-flex">
                        {assetClassLabel(asset.assetClass)}
                      </Badge>
                    )}
                  </div>
                  {asset && <span className="line-clamp-1 text-[0.6875rem] text-muted">{asset.name}</span>}
                </div>

                <span className="w-12 shrink-0 text-right text-xs tnum text-muted">{formatPct(weight * 100, { digits: 0 })}</span>

                <div className="relative w-28 shrink-0">
                  <span className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-xs text-muted">$</span>
                  <input
                    type="number"
                    inputMode="decimal"
                    min={0}
                    value={row.amount === 0 ? '' : row.amount}
                    onChange={(e) => setRowAmount(row.symbol, e.target.value)}
                    placeholder="0"
                    className="h-9 w-full rounded-xl border border-border bg-surface-2 pl-5 pr-2 text-right text-sm tnum text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT"
                    aria-label={`Amount for ${row.symbol}`}
                  />
                </div>

                <button
                  type="button"
                  onClick={() => removeRow(row.symbol)}
                  aria-label={`Remove ${row.symbol}`}
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl text-muted transition-colors hover:bg-danger/10 hover:text-danger"
                >
                  <Trash2 className="h-4 w-4" aria-hidden />
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {/* Allocated / remaining bar */}
      <div className="flex flex-col gap-1.5">
        <ProgressBar
          value={summary.fraction}
          tone={summary.overBudget ? 'danger' : 'primary'}
          label={`Allocated ${formatCurrency(summary.allocated)}`}
          valueLabel={formatPct(summary.fraction * 100, { digits: 0 })}
        />
        <p className={cn('text-xs tnum', summary.overBudget ? 'text-danger' : 'text-muted')}>
          {summary.overBudget
            ? `Over budget by ${formatCurrency(Math.abs(summary.remaining))}`
            : `${formatCurrency(summary.remaining)} remaining of ${formatCurrency(cash)}`}
        </p>
      </div>

      {error && (
        <p className="rounded-lg bg-danger/10 px-3 py-2 text-xs font-medium text-danger" role="alert">
          {error}
        </p>
      )}

      <Button
        variant="primary"
        size="lg"
        fullWidth
        onClick={submitInvest}
        disabled={!canInvest}
        loading={invest.isPending}
      >
        {invest.isPending ? (
          <span className="inline-flex items-center gap-1">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Investing
          </span>
        ) : (
          `Invest ${summary.allocated > 0 ? formatCurrency(summary.allocated) : ''}`.trim()
        )}
      </Button>
    </Card>
  );
}
