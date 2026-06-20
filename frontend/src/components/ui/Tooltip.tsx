/**
 * Tooltip — a lightweight, dependency-free hover/focus tooltip. Wraps a single
 * trigger; the bubble is positioned with CSS (top/bottom/left/right) and shown on
 * hover + keyboard focus. Linked to the trigger via `aria-describedby`.
 */

import { useId, useState, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export type TooltipSide = 'top' | 'bottom' | 'left' | 'right';

export interface TooltipProps {
  content: ReactNode;
  children: ReactNode;
  side?: TooltipSide;
  className?: string;
  /** Disable the tooltip (renders the trigger only). */
  disabled?: boolean;
}

const SIDE_POS: Record<TooltipSide, string> = {
  top: 'bottom-full left-1/2 -translate-x-1/2 mb-1.5',
  bottom: 'top-full left-1/2 -translate-x-1/2 mt-1.5',
  left: 'right-full top-1/2 -translate-y-1/2 mr-1.5',
  right: 'left-full top-1/2 -translate-y-1/2 ml-1.5',
};

export function Tooltip({
  content,
  children,
  side = 'top',
  className,
  disabled = false,
}: TooltipProps): JSX.Element {
  const id = useId();
  const [open, setOpen] = useState(false);

  if (disabled) return <>{children}</>;

  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <span aria-describedby={open ? id : undefined} className="inline-flex">
        {children}
      </span>
      <span
        role="tooltip"
        id={id}
        className={cn(
          'pointer-events-none absolute z-50 w-max max-w-[16rem] rounded-lg border border-border bg-surface px-2.5 py-1.5',
          'text-xs leading-snug text-text shadow-pop transition-opacity duration-150',
          SIDE_POS[side],
          open ? 'opacity-100' : 'opacity-0',
          className,
        )}
      >
        {content}
      </span>
    </span>
  );
}
