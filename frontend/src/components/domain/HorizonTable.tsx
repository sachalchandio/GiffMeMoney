/**
 * HorizonTable — the per-horizon projection grid: expected (base) return plus the
 * bull / bear scenario fan, probability of a positive return, and the
 * expected-shortfall (CVaR) at each horizon. All return fields are already in
 * percent units; `probPositive` is a 0..1 probability and `cvarPct` a positive
 * loss fraction in percent (falls back gracefully when the V2 fields are absent).
 * Colors via semantic tokens only.
 */

import { ProgressBar } from '@/components/ui/ProgressBar';
import type { ExpectedReturn } from '@/lib/types';
import { formatPct, formatProbability, horizonLabel } from '@/lib/format';
import { changeTextColor, cn } from '@/lib/utils';

export interface HorizonTableProps {
  expectedReturns: ExpectedReturn[];
  className?: string;
}

interface Row {
  horizon: string;
  base: number;
  bull: number;
  bear: number;
  probPositive: number;
  cvar: number | null;
}

function toRow(r: ExpectedReturn): Row {
  return {
    horizon: horizonLabel(r.horizon),
    base: r.basePct ?? r.expectedReturnPct,
    bull: r.bullPct ?? r.high,
    bear: r.bearPct ?? r.low,
    probPositive: r.probPositive,
    cvar: r.cvarPct ?? null,
  };
}

/** Probability bar tone: leans bullish above 55%, bearish below 45%. */
function probTone(p: number): 'success' | 'danger' | 'warning' {
  if (p >= 0.55) return 'success';
  if (p <= 0.45) return 'danger';
  return 'warning';
}

export function HorizonTable({ expectedReturns, className }: HorizonTableProps): JSX.Element {
  const rows = expectedReturns.map(toRow);

  if (rows.length === 0) {
    return (
      <div
        className={cn(
          'flex h-32 items-center justify-center rounded-xl border border-dashed border-border text-sm text-muted',
          className,
        )}
      >
        No projections available
      </div>
    );
  }

  return (
    <div className={cn('overflow-x-auto', className)}>
      <table className="w-full min-w-[34rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-border text-left text-[0.6875rem] font-medium uppercase tracking-wide text-muted">
            <th className="py-2 pr-3 font-medium">Horizon</th>
            <th className="py-2 px-3 text-right font-medium">Bear</th>
            <th className="py-2 px-3 text-right font-medium">Base</th>
            <th className="py-2 px-3 text-right font-medium">Bull</th>
            <th className="py-2 px-3 text-right font-medium">CVaR</th>
            <th className="py-2 pl-3 font-medium">P(up)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.horizon} className="border-b border-border/70 last:border-0">
              <td className="whitespace-nowrap py-2.5 pr-3 font-medium text-text">{row.horizon}</td>
              <td className={cn('py-2.5 px-3 text-right tnum', changeTextColor(row.bear))}>
                {formatPct(row.bear, { sign: true })}
              </td>
              <td className={cn('py-2.5 px-3 text-right font-semibold tnum', changeTextColor(row.base))}>
                {formatPct(row.base, { sign: true })}
              </td>
              <td className={cn('py-2.5 px-3 text-right tnum', changeTextColor(row.bull))}>
                {formatPct(row.bull, { sign: true })}
              </td>
              <td className="py-2.5 px-3 text-right tnum text-danger">
                {row.cvar == null ? <span className="text-muted">—</span> : formatPct(-Math.abs(row.cvar))}
              </td>
              <td className="py-2.5 pl-3">
                <div className="flex min-w-[7rem] items-center gap-2">
                  <ProgressBar
                    value={row.probPositive}
                    tone={probTone(row.probPositive)}
                    size="sm"
                    className="flex-1"
                    aria-label={`Probability of a positive ${row.horizon} return`}
                  />
                  <span className="w-9 shrink-0 text-right text-xs tnum text-muted">
                    {formatProbability(row.probPositive)}
                  </span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
