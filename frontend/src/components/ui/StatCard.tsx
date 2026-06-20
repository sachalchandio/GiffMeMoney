/**
 * StatCard — a labelled headline metric with an optional signed delta, hint,
 * leading icon, and inline sparkline. Used across the dashboard / wallet header /
 * portfolio summary. Delta color is derived from its sign (positive → success).
 */

import type { ReactNode } from 'react';
import { ArrowDownRight, ArrowUpRight } from 'lucide-react';
import { Card } from './Card';
import { Sparkline } from './Sparkline';
import { cn } from '@/lib/utils';
import { changeTextColor } from '@/lib/utils';
import { formatPct } from '@/lib/format';

export interface StatCardProps {
  label: string;
  /** Pre-formatted primary value (string from lib/format). */
  value: ReactNode;
  /** Signed delta in percent units (e.g. `2.3` → +2.30%). */
  deltaPct?: number | null;
  /** Override the delta text (else formatted from `deltaPct`). */
  deltaLabel?: string;
  /** Small secondary line below the value. */
  hint?: ReactNode;
  /** Leading icon (rendered in a tinted square). */
  icon?: ReactNode;
  /** Optional inline sparkline series. */
  spark?: number[];
  className?: string;
}

export function StatCard({
  label,
  value,
  deltaPct,
  deltaLabel,
  hint,
  icon,
  spark,
  className,
}: StatCardProps): JSX.Element {
  const hasDelta = typeof deltaPct === 'number' && Number.isFinite(deltaPct);
  const up = hasDelta && (deltaPct as number) >= 0;

  return (
    <Card className={cn('flex flex-col gap-2', className)}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-2 text-xs font-medium text-muted">
          {icon && (
            <span className="flex h-7 w-7 items-center justify-center rounded-xl bg-primary/12 text-primary">
              {icon}
            </span>
          )}
          <span className="tracking-tight">{label}</span>
        </div>
        {spark && spark.length > 1 && <Sparkline points={spark} width={72} height={24} />}
      </div>

      <div className="flex items-end justify-between gap-2">
        <div className="text-2xl font-semibold tracking-tight tnum text-text">{value}</div>
        {(hasDelta || deltaLabel) && (
          <span
            className={cn(
              'inline-flex items-center gap-0.5 text-xs font-medium tnum',
              changeTextColor(hasDelta ? (deltaPct as number) : 0),
            )}
          >
            {hasDelta &&
              (up ? (
                <ArrowUpRight className="h-3.5 w-3.5" aria-hidden />
              ) : (
                <ArrowDownRight className="h-3.5 w-3.5" aria-hidden />
              ))}
            {deltaLabel ?? formatPct(deltaPct, { digits: 2, sign: true })}
          </span>
        )}
      </div>

      {hint && <div className="text-xs text-muted">{hint}</div>}
    </Card>
  );
}
