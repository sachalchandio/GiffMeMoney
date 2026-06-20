/**
 * Tabs — a headless-ish, accessible segmented control. Controlled (`value` +
 * `onChange`) or uncontrolled (`defaultValue`). Generic over the value type so
 * callers keep their string-literal unions. Renders a tablist with arrow-key
 * roving focus; the `pill` variant is a filled segmented control.
 */

import { useCallback, useId, useRef, useState, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface TabItem<T extends string> {
  value: T;
  label: ReactNode;
  icon?: ReactNode;
  disabled?: boolean;
}

export interface TabsProps<T extends string> {
  items: TabItem<T>[];
  value?: T;
  defaultValue?: T;
  onChange?: (value: T) => void;
  variant?: 'underline' | 'pill';
  size?: 'sm' | 'md';
  className?: string;
  'aria-label'?: string;
}

export function Tabs<T extends string>({
  items,
  value,
  defaultValue,
  onChange,
  variant = 'underline',
  size = 'md',
  className,
  'aria-label': ariaLabel,
}: TabsProps<T>): JSX.Element {
  const baseId = useId();
  const isControlled = value !== undefined;
  const [internal, setInternal] = useState<T | undefined>(
    defaultValue ?? items[0]?.value,
  );
  const active = (isControlled ? value : internal) as T | undefined;
  const refs = useRef<Array<HTMLButtonElement | null>>([]);

  const select = useCallback(
    (next: T) => {
      if (!isControlled) setInternal(next);
      onChange?.(next);
    },
    [isControlled, onChange],
  );

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent, index: number) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      e.preventDefault();
      const dir = e.key === 'ArrowRight' ? 1 : -1;
      const n = items.length;
      for (let step = 1; step <= n; step++) {
        const idx = (index + dir * step + n * step) % n;
        const item = items[idx];
        if (item && !item.disabled) {
          refs.current[idx]?.focus();
          select(item.value);
          break;
        }
      }
    },
    [items, select],
  );

  const isPill = variant === 'pill';

  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn(
        'flex items-center',
        isPill
          ? 'gap-1 rounded-xl border border-border bg-surface-2 p-1'
          : 'gap-1 border-b border-border',
        className,
      )}
    >
      {items.map((item, index) => {
        const selected = item.value === active;
        return (
          <button
            key={item.value}
            ref={(el) => {
              refs.current[index] = el;
            }}
            role="tab"
            id={`${baseId}-tab-${item.value}`}
            aria-selected={selected}
            aria-disabled={item.disabled || undefined}
            tabIndex={selected ? 0 : -1}
            disabled={item.disabled}
            onClick={() => !item.disabled && select(item.value)}
            onKeyDown={(e) => onKeyDown(e, index)}
            className={cn(
              'inline-flex items-center gap-1.5 whitespace-nowrap font-medium tracking-tight transition-colors',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
              size === 'sm' ? 'text-xs' : 'text-sm',
              item.disabled && 'cursor-not-allowed opacity-50',
              isPill
                ? cn(
                    'rounded-lg',
                    size === 'sm' ? 'h-7 px-2.5' : 'h-8 px-3',
                    selected
                      ? 'bg-surface text-text shadow-soft'
                      : 'text-muted hover:text-text',
                  )
                : cn(
                    'border-b-2 -mb-px',
                    size === 'sm' ? 'px-2.5 pb-2' : 'px-3 pb-2.5',
                    selected
                      ? 'border-primary text-text'
                      : 'border-transparent text-muted hover:text-text',
                  ),
            )}
          >
            {item.icon}
            {item.label}
          </button>
        );
      })}
    </div>
  );
}
