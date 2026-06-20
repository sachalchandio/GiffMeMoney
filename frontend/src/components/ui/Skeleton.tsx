/**
 * Skeleton loading placeholders. `Skeleton` is a single shimmer block (the
 * `.skeleton` utility lives in index.css); `SkeletonText` renders N lines and
 * `SkeletonCard` a generic card-shaped placeholder.
 */

import type { HTMLAttributes } from 'react';
import { cn } from '@/lib/utils';

export interface SkeletonProps extends HTMLAttributes<HTMLDivElement> {
  /** Convenience for `rounded-full`. */
  circle?: boolean;
}

export function Skeleton({ circle = false, className, ...rest }: SkeletonProps): JSX.Element {
  return (
    <div
      aria-hidden
      className={cn('skeleton', circle ? 'rounded-full' : 'rounded-xl', className)}
      {...rest}
    />
  );
}

export function SkeletonText({
  lines = 3,
  className,
}: {
  lines?: number;
  className?: string;
}): JSX.Element {
  return (
    <div className={cn('flex flex-col gap-2', className)} aria-hidden>
      {Array.from({ length: lines }).map((_, i) => (
        <Skeleton
          key={i}
          className={cn('h-3.5', i === lines - 1 ? 'w-2/3' : 'w-full')}
        />
      ))}
    </div>
  );
}

export function SkeletonCard({ className }: { className?: string }): JSX.Element {
  return (
    <div
      className={cn('rounded-2xl border border-border bg-surface p-4 shadow-soft', className)}
      aria-hidden
    >
      <Skeleton className="h-4 w-24" />
      <Skeleton className="mt-3 h-7 w-32" />
      <Skeleton className="mt-4 h-3.5 w-full" />
      <Skeleton className="mt-2 h-3.5 w-3/4" />
    </div>
  );
}
