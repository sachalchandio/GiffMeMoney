/**
 * Card surfaces — the rounded-2xl, soft-shadowed container the whole UI is built
 * on, plus a few composable sub-parts (header / title / content / footer).
 * Colors come from semantic tokens only.
 */

import { forwardRef, type HTMLAttributes, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Drop the inner padding (useful for tables / charts that bleed to edges). */
  flush?: boolean;
  /** Use the lighter `surface-2` background instead of `surface`. */
  muted?: boolean;
  /** Add a hover lift (for clickable cards). */
  interactive?: boolean;
}

export const Card = forwardRef<HTMLDivElement, CardProps>(function Card(
  { flush = false, muted = false, interactive = false, className, children, ...rest },
  ref,
) {
  return (
    <div
      ref={ref}
      className={cn(
        'rounded-2xl border border-border shadow-soft',
        muted ? 'bg-surface-2' : 'bg-surface',
        !flush && 'p-4',
        interactive &&
          'transition-shadow duration-150 hover:shadow-card focus-within:shadow-card',
        className,
      )}
      {...rest}
    >
      {children}
    </div>
  );
});

export function CardHeader({
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLDivElement>): JSX.Element {
  return (
    <div className={cn('flex items-start justify-between gap-3', className)} {...rest}>
      {children}
    </div>
  );
}

export function CardTitle({
  className,
  children,
  icon,
  ...rest
}: HTMLAttributes<HTMLHeadingElement> & { icon?: ReactNode }): JSX.Element {
  return (
    <h3
      className={cn('flex items-center gap-2 text-sm font-semibold tracking-tight text-text', className)}
      {...rest}
    >
      {icon && <span className="text-muted">{icon}</span>}
      {children}
    </h3>
  );
}

export function CardDescription({
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLParagraphElement>): JSX.Element {
  return (
    <p className={cn('text-xs text-muted', className)} {...rest}>
      {children}
    </p>
  );
}

export function CardContent({
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLDivElement>): JSX.Element {
  return (
    <div className={cn('mt-3', className)} {...rest}>
      {children}
    </div>
  );
}

export function CardFooter({
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLDivElement>): JSX.Element {
  return (
    <div className={cn('mt-3 flex items-center gap-2 border-t border-border pt-3', className)} {...rest}>
      {children}
    </div>
  );
}
