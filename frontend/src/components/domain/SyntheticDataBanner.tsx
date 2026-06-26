/**
 * SyntheticDataBanner — the honesty banner shown above advisor output.
 *
 * Surfaces two things from the backend's :class:`AllocationAdvice`:
 *   1. `syntheticData` — the advice is computed on SYNTHETIC (made-up) market
 *      data, so it is not a real forecast and carries no real edge. We always
 *      say this plainly so a user can never mistake the numbers for a guarantee.
 *   2. `targetWarning` — a plain-English warning when the requested goal is
 *      physically extreme / infeasible. When present it is shown prominently in
 *      a warning tone (it is the louder of the two messages).
 *
 * Colors via semantic tokens only; works in light + dark.
 */

import { AlertTriangle, FlaskConical } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface SyntheticDataBannerProps {
  /** Whether the figures come from synthetic data (renders the honesty note). */
  synthetic?: boolean;
  /** Optional infeasible-target warning; shown prominently when present. */
  targetWarning?: string | null;
  className?: string;
}

/**
 * Render the synthetic-data honesty note and, when present, the infeasible
 * target warning. Renders nothing when there is neither a warning nor synthetic
 * data to disclose.
 */
export function SyntheticDataBanner({
  synthetic,
  targetWarning,
  className,
}: SyntheticDataBannerProps): JSX.Element | null {
  const warning = targetWarning?.trim() ? targetWarning.trim() : null;
  if (!warning && !synthetic) return null;

  return (
    <div className={cn('flex flex-col gap-2', className)}>
      {warning && (
        <div
          role="alert"
          className="flex items-start gap-2 rounded-xl border border-warning/30 bg-warning/10 px-3 py-2"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" aria-hidden />
          <p className="text-xs font-medium leading-snug text-warning">{warning}</p>
        </div>
      )}
      {synthetic && (
        <div className="flex items-start gap-2 rounded-xl border border-border bg-surface-2/50 px-3 py-2">
          <FlaskConical className="mt-0.5 h-4 w-4 shrink-0 text-muted" aria-hidden />
          <p className="text-[11px] leading-snug text-muted">
            Computed on <span className="font-medium text-text">synthetic data</span> — an educational
            simulation, not a real forecast and no guarantee of profit.
          </p>
        </div>
      )}
    </div>
  );
}
