/**
 * Badge — small status / category pill. Tones map to semantic tokens; the
 * `soft` style (default) uses a translucent background, `solid` a filled chip.
 */

import type { HTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/utils';

export type BadgeTone =
  | 'neutral'
  | 'primary'
  | 'accent'
  | 'success'
  | 'danger'
  | 'warning';
export type BadgeVariant = 'soft' | 'solid' | 'outline';
export type BadgeSize = 'sm' | 'md';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
  variant?: BadgeVariant;
  size?: BadgeSize;
  icon?: ReactNode;
}

const SOFT: Record<BadgeTone, string> = {
  neutral: 'bg-surface-2 text-muted',
  primary: 'bg-primary/12 text-primary',
  accent: 'bg-accent/12 text-accent',
  success: 'bg-success/12 text-success',
  danger: 'bg-danger/12 text-danger',
  warning: 'bg-warning/14 text-warning',
};

const SOLID: Record<BadgeTone, string> = {
  neutral: 'bg-surface-2 text-text',
  primary: 'bg-primary text-white',
  accent: 'bg-accent text-white',
  success: 'bg-success text-white',
  danger: 'bg-danger text-white',
  warning: 'bg-warning text-white',
};

const OUTLINE: Record<BadgeTone, string> = {
  neutral: 'border border-border text-muted',
  primary: 'border border-primary/40 text-primary',
  accent: 'border border-accent/40 text-accent',
  success: 'border border-success/40 text-success',
  danger: 'border border-danger/40 text-danger',
  warning: 'border border-warning/40 text-warning',
};

const SIZES: Record<BadgeSize, string> = {
  sm: 'h-5 px-1.5 text-[0.625rem] gap-1',
  md: 'h-6 px-2 text-xs gap-1.5',
};

export function Badge({
  tone = 'neutral',
  variant = 'soft',
  size = 'md',
  icon,
  className,
  children,
  ...rest
}: BadgeProps): JSX.Element {
  const styles = variant === 'solid' ? SOLID[tone] : variant === 'outline' ? OUTLINE[tone] : SOFT[tone];
  return (
    <span
      className={cn(
        'inline-flex items-center whitespace-nowrap rounded-full font-medium tracking-tight tnum',
        SIZES[size],
        styles,
        className,
      )}
      {...rest}
    >
      {icon && <span className="inline-flex shrink-0">{icon}</span>}
      {children}
    </span>
  );
}
