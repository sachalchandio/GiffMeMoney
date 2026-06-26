/**
 * AdvicePanel — the "where to invest now" recommendation. Runs the allocation
 * advisor for an amount + risk tolerance and renders the suggested basket (each
 * pick with its weight, dollar amount, composite score, 1Y expected return and a
 * one-line rationale), the blended expected return / vol / Sharpe, and a 5-horizon
 * expected-outcome fan for the whole basket. An "Apply" hook lets the page push
 * the picks into the allocation builder. Colors via semantic tokens only.
 */

import { useEffect, useState } from 'react';
import { Compass, PiggyBank, ShieldAlert, Sparkles, TrendingDown, TrendingUp } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Tabs, type TabItem } from '@/components/ui/Tabs';
import { ScenarioFanChart } from '@/components/charts/ScenarioFanChart';
import { SyntheticDataBanner } from '@/components/domain/SyntheticDataBanner';
import { useAdvisor } from '@/hooks/useAdvisor';
import type { AllocationAdvice, ExpectedReturn, RiskTolerance } from '@/lib/types';
import { formatCurrency, formatFractionPct, formatPct, formatRatio } from '@/lib/format';
import { changeTextColor, cn, round } from '@/lib/utils';

/**
 * Pick the horizon used for the headline downside read-out. Prefers the 1Y
 * horizon (the advisor's sizing horizon); falls back to the last horizon that
 * carries either a CVaR or a bear-case figure, else the last horizon.
 */
function downsideHorizon(horizons: ExpectedReturn[]): ExpectedReturn | null {
  if (horizons.length === 0) return null;
  const oneYear = horizons.find((h) => h.horizon === '1Y');
  if (oneYear) return oneYear;
  const withDownside = [...horizons]
    .reverse()
    .find((h) => h.cvarPct != null || h.bearPct != null);
  return withDownside ?? horizons[horizons.length - 1] ?? null;
}

export interface AdvicePanelProps {
  /** Amount to size the basket for (e.g. the wallet cash). */
  amount: number;
  /** Optional externally-supplied advice (e.g. shared from the builder). */
  advice?: AllocationAdvice | null;
  /** Apply the advised picks into the allocation builder. */
  onApply?: (advice: AllocationAdvice) => void;
  className?: string;
}

const RISK_ITEMS: TabItem<RiskTolerance>[] = [
  { value: 'conservative', label: 'Conservative' },
  { value: 'balanced', label: 'Balanced' },
  { value: 'aggressive', label: 'Aggressive' },
];

export function AdvicePanel({ amount, advice: external, onApply, className }: AdvicePanelProps): JSX.Element {
  const advisor = useAdvisor();
  const [risk, setRisk] = useState<RiskTolerance>('balanced');
  const [advice, setAdvice] = useState<AllocationAdvice | null>(external ?? null);

  // Adopt advice passed from the parent (e.g. when the builder runs the advisor).
  useEffect(() => {
    if (external) {
      setAdvice(external);
      setRisk(external.riskTolerance);
    }
  }, [external]);

  const run = (nextRisk: RiskTolerance): void => {
    setRisk(nextRisk);
    const amt = amount > 0 ? round(amount, 2) : 1000;
    advisor.mutate(
      { amount: amt, riskTolerance: nextRisk, assetClasses: null },
      { onSuccess: (res) => setAdvice(res) },
    );
  };

  return (
    <Card className={cn('flex flex-col gap-4', className)}>
      <CardHeader>
        <div>
          <CardTitle icon={<Compass className="h-4 w-4" />}>Where to invest now</CardTitle>
          <p className="mt-1 text-xs text-muted">
            A Markowitz-sized basket from the highest-conviction picks across the universe.
          </p>
        </div>
      </CardHeader>

      <div className="flex flex-wrap items-center gap-2">
        <Tabs items={RISK_ITEMS} value={risk} onChange={(r) => run(r)} variant="pill" size="sm" aria-label="Advisor risk tolerance" />
        <Button
          variant="accent"
          size="sm"
          onClick={() => run(risk)}
          loading={advisor.isPending}
          leftIcon={!advisor.isPending ? <Sparkles className="h-3.5 w-3.5" /> : undefined}
        >
          {advice ? 'Refresh' : 'Get picks'}
        </Button>
      </div>

      {advisor.isError && (
        <p className="rounded-lg bg-danger/10 px-3 py-2 text-xs font-medium text-danger" role="alert">
          {advisor.error instanceof Error ? advisor.error.message : 'Could not fetch advice.'}
        </p>
      )}

      {!advice ? (
        <div className="rounded-xl border border-dashed border-border py-8 text-center text-xs text-muted">
          Pick a risk level to see a suggested basket.
        </div>
      ) : (
        <>
          {/* Honesty: infeasible-target warning + synthetic-data note */}
          <SyntheticDataBanner synthetic={advice.syntheticData} targetWarning={advice.targetWarning} />

          {/* Blended headline metrics (of the risky sleeve) */}
          <div className="grid grid-cols-3 gap-2">
            <Headline label="Exp. return" value={formatFractionPct(advice.expectedReturn, { digits: 1, sign: true })} tone={changeTextColor(advice.expectedReturn)} />
            <Headline label="Volatility" value={formatFractionPct(advice.expectedVol, { digits: 1 })} />
            <Headline label="Sharpe" value={formatRatio(advice.sharpe)} />
          </div>

          {/* Cash sleeve + basket downside (bear case / CVaR) */}
          {(() => {
            const cashWeight = advice.cashWeight ?? 0;
            const cashAmount = advice.cashAmount ?? 0;
            const down = downsideHorizon(advice.horizons);
            const showCash = cashWeight > 0.0005 || cashAmount > 0.005;
            const bear = down?.bearPct ?? null;
            const cvar = down?.cvarPct ?? null;
            if (!showCash && bear == null && cvar == null) return null;
            return (
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {showCash && (
                  <div className="flex items-center gap-2.5 rounded-xl border border-border bg-surface-2/40 px-3 py-2">
                    <PiggyBank className="h-4 w-4 shrink-0 text-primary" aria-hidden />
                    <div className="min-w-0">
                      <p className="text-[10px] font-medium uppercase tracking-wide text-muted">Cash sleeve</p>
                      <p className="text-sm font-semibold tnum text-text">
                        {formatFractionPct(cashWeight, { digits: 0 })}
                        <span className="ml-1 text-[11px] font-normal text-muted">· {formatCurrency(cashAmount)} held</span>
                      </p>
                    </div>
                  </div>
                )}
                {(bear != null || cvar != null) && (
                  <div className="flex items-center gap-2.5 rounded-xl border border-danger/25 bg-danger/5 px-3 py-2">
                    <ShieldAlert className="h-4 w-4 shrink-0 text-danger" aria-hidden />
                    <div className="min-w-0">
                      <p className="text-[10px] font-medium uppercase tracking-wide text-muted">
                        Downside{down ? ` · ${down.horizon}` : ''}
                      </p>
                      <p className="flex flex-wrap items-baseline gap-x-2 text-sm font-semibold tnum text-text">
                        {bear != null && (
                          <span className="inline-flex items-center gap-0.5 text-danger">
                            <TrendingDown className="h-3 w-3" aria-hidden />
                            {formatPct(bear, { sign: true, digits: 1 })} bear
                          </span>
                        )}
                        {cvar != null && (
                          <span className="text-[11px] font-normal text-muted">
                            CVaR −{formatPct(Math.abs(cvar), { digits: 1 })}
                          </span>
                        )}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            );
          })()}

          {/* Picks */}
          <ul className="flex flex-col gap-2">
            {advice.items.map((item) => (
              <li key={item.asset.symbol} className="rounded-xl border border-border bg-surface px-3 py-2.5">
                <div className="flex items-center gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <Link
                        to={`/asset/${encodeURIComponent(item.asset.symbol)}`}
                        className="text-sm font-semibold tracking-tight text-text hover:text-primary"
                      >
                        {item.asset.symbol}
                      </Link>
                      <Badge tone="primary" size="sm" variant="soft">
                        {formatPct(item.weight * 100, { digits: 0 })}
                      </Badge>
                    </div>
                    <p className="line-clamp-1 text-[11px] text-muted">{item.rationale}</p>
                  </div>
                  <div className="shrink-0 text-right">
                    <p className="text-sm font-medium tnum text-text">{formatCurrency(item.amount)}</p>
                    <p className={cn('inline-flex items-center justify-end gap-0.5 text-[11px] tnum', changeTextColor(item.expectedReturn1YPct))}>
                      <TrendingUp className="h-3 w-3" aria-hidden />
                      {formatPct(item.expectedReturn1YPct, { sign: true, digits: 1 })} 1Y
                    </p>
                  </div>
                </div>
              </li>
            ))}
          </ul>

          {/* Whole-basket horizon fan */}
          <div>
            <p className="mb-1 text-[11px] font-medium uppercase tracking-wide text-muted">Basket outlook · 5 horizons</p>
            <ScenarioFanChart expectedReturns={advice.horizons} height={200} />
          </div>

          {onApply && (
            <Button variant="primary" size="md" fullWidth onClick={() => onApply(advice)}>
              Use this allocation
            </Button>
          )}
        </>
      )}
    </Card>
  );
}

function Headline({ label, value, tone }: { label: string; value: string; tone?: string }): JSX.Element {
  return (
    <div className="rounded-xl border border-border bg-surface-2/50 px-2.5 py-2 text-center">
      <p className="text-[10px] font-medium uppercase tracking-wide text-muted">{label}</p>
      <p className={cn('mt-0.5 text-sm font-semibold tnum text-text', tone)}>{value}</p>
    </div>
  );
}
