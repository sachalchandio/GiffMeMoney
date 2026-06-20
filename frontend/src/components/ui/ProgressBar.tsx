/**
 * ProgressBar — a horizontal meter (0..1 value or explicit percent). Tone maps
 * to a semantic token; optional label + trailing value. Accessible via
 * role="progressbar" with aria-value attributes.
 */

import type { ReactNode } from 'react';
import { cn } from '@/lib/utils';
import { clamp } from '@/lib/utils';

export type ProgressTone = 'primary' | 'accent' | 'success' | 'danger' | 'warning';

export interface ProgressBarProps {
  /** Fraction 0..1. */
  value: number;
  tone?: ProgressTone;
  size?: 'sm' | 'md';
  label?: ReactNode;
  /** Trailing value text (e.g. a formatted percent). */
  valueLabel?: ReactNode;
  className?: string;
  'aria-label'?: string;
}

const TONE_BG: Record<ProgressTone, string> = {
  primary: 'bg-primary',
  accent: 'bg-accent',
  success: 'bg-success',
  danger: 'bg-danger',
  warning: 'bg-warning',
};

export function ProgressBar({
  value,
  tone = 'primary',
  size = 'md',
  label,
  valueLabel,
  className,
  'aria-label': ariaLabel,
}: ProgressBarProps): JSX.Element {
  const pct = clamp(value, 0, 1) * 100;
  return (
    <div className={cn('flex flex-col gap-1', className)}>
      {(label || valueLabel) && (
        <div className="flex items-center justify-between text-xs">
          {label && <span className="font-medium text-muted">{label}</span>}
          {valueLabel && <span className="tnum font-medium text-text">{valueLabel}</span>}
        </div>
      )}
      <div
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(pct)}
        aria-label={ariaLabel}
        className={cn(
          'w-full overflow-hidden rounded-full bg-surface-2',
          size === 'sm' ? 'h-1.5' : 'h-2.5',
        )}
      >
        <div
          className={cn('h-full rounded-full transition-[width] duration-300', TONE_BG[tone])}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
