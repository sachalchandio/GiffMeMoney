/**
 * AdvicePanel — the "where to invest now" recommendation. Runs the allocation
 * advisor for an amount + risk tolerance and renders the suggested basket (each
 * pick with its weight, dollar amount, composite score, 1Y expected return and a
 * one-line rationale), the blended expected return / vol / Sharpe, and a 5-horizon
 * expected-outcome fan for the whole basket. An "Apply" hook lets the page push
 * the picks into the allocation builder. Colors via semantic tokens only.
 */

import { useEffect, useState } from 'react';
import { Compass, Sparkles, TrendingUp } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Tabs, type TabItem } from '@/components/ui/Tabs';
import { ScenarioFanChart } from '@/components/charts/ScenarioFanChart';
import { useAdvisor } from '@/hooks/useAdvisor';
import type { AllocationAdvice, RiskTolerance } from '@/lib/types';
import { formatCurrency, formatFractionPct, formatPct, formatRatio } from '@/lib/format';
import { changeTextColor, cn, round } from '@/lib/utils';

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
          {/* Blended headline metrics */}
          <div className="grid grid-cols-3 gap-2">
            <Headline label="Exp. return" value={formatFractionPct(advice.expectedReturn, { digits: 1, sign: true })} tone={changeTextColor(advice.expectedReturn)} />
            <Headline label="Volatility" value={formatFractionPct(advice.expectedVol, { digits: 1 })} />
            <Headline label="Sharpe" value={formatRatio(advice.sharpe)} />
          </div>

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
