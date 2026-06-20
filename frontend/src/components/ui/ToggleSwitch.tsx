/**
 * ToggleSwitch — an accessible on/off switch (role="switch"). Controlled via
 * `checked` + `onChange`; supports an optional label + description rendered to
 * the side. Themed with the primary token when on.
 */

import { useId, type ReactNode } from 'react';
import { cn } from '@/lib/utils';

export interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: ReactNode;
  description?: ReactNode;
  disabled?: boolean;
  size?: 'sm' | 'md';
  /** Required when no visible `label` is rendered. */
  'aria-label'?: string;
  className?: string;
}

export function ToggleSwitch({
  checked,
  onChange,
  label,
  description,
  disabled = false,
  size = 'md',
  'aria-label': ariaLabel,
  className,
}: ToggleSwitchProps): JSX.Element {
  const id = useId();
  const track = size === 'sm' ? 'h-5 w-9' : 'h-6 w-11';
  const knob = size === 'sm' ? 'h-4 w-4' : 'h-5 w-5';
  const travel = size === 'sm' ? 'translate-x-4' : 'translate-x-5';

  const button = (
    <button
      type="button"
      role="switch"
      id={label ? id : undefined}
      aria-checked={checked}
      aria-label={!label ? ariaLabel : undefined}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        'relative inline-flex shrink-0 items-center rounded-full border border-transparent transition-colors duration-150',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
        track,
        checked ? 'bg-primary' : 'bg-surface-2 border-border',
        disabled && 'cursor-not-allowed opacity-50',
      )}
    >
      <span
        className={cn(
          'pointer-events-none inline-block transform rounded-full bg-white shadow-soft transition-transform duration-150',
          knob,
          checked ? travel : 'translate-x-0.5',
        )}
      />
    </button>
  );

  if (!label && !description) return <span className={className}>{button}</span>;

  return (
    <label
      htmlFor={id}
      className={cn(
        'flex cursor-pointer items-center justify-between gap-3',
        disabled && 'cursor-not-allowed',
        className,
      )}
    >
      <span className="flex flex-col">
        {label && <span className="text-sm font-medium text-text">{label}</span>}
        {description && <span className="text-xs text-muted">{description}</span>}
      </span>
      {button}
    </label>
  );
}
