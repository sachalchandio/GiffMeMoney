/**
 * Select — a themed native `<select>` wrapper (keeps native a11y + mobile UX)
 * with a chevron affordance and optional label. Generic over the option value so
 * callers keep their literal-union types. Colors via semantic tokens only.
 */

import { useId, type SelectHTMLAttributes } from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface SelectOption<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
}

export interface SelectProps<T extends string>
  extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'value' | 'onChange' | 'size'> {
  options: SelectOption<T>[];
  value: T;
  onChange: (value: T) => void;
  label?: string;
  placeholder?: string;
  size?: 'sm' | 'md';
  fullWidth?: boolean;
}

export function Select<T extends string>({
  options,
  value,
  onChange,
  label,
  placeholder,
  size = 'md',
  fullWidth = false,
  className,
  id,
  ...rest
}: SelectProps<T>): JSX.Element {
  const autoId = useId();
  const selectId = id ?? autoId;

  return (
    <div className={cn('flex flex-col gap-1', fullWidth && 'w-full')}>
      {label && (
        <label htmlFor={selectId} className="text-xs font-medium text-muted">
          {label}
        </label>
      )}
      <div className={cn('relative', fullWidth && 'w-full')}>
        <select
          id={selectId}
          value={value}
          onChange={(e) => onChange(e.target.value as T)}
          className={cn(
            'w-full cursor-pointer appearance-none rounded-xl border border-border bg-surface pr-8 text-text',
            'transition-colors hover:border-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
            'disabled:cursor-not-allowed disabled:opacity-55',
            size === 'sm' ? 'h-8 pl-2.5 text-xs' : 'h-9 pl-3 text-sm',
            className,
          )}
          {...rest}
        >
          {placeholder && (
            <option value="" disabled>
              {placeholder}
            </option>
          )}
          {options.map((opt) => (
            <option key={opt.value} value={opt.value} disabled={opt.disabled}>
              {opt.label}
            </option>
          ))}
        </select>
        <ChevronDown
          className="pointer-events-none absolute right-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-muted"
          aria-hidden
        />
      </div>
    </div>
  );
}
