/**
 * ModeToggle — the Easy ⇄ Expert segmented switch, the app's signature control.
 *
 * A two-segment pill with a sliding active highlight. Labels show from `sm` up;
 * on the narrowest screens it collapses to icons only. Flipping it visibly
 * transforms the whole app (copy, density, and — via the rem root-scale — size).
 */

import { Leaf, SlidersHorizontal } from 'lucide-react';
import { useUiMode, type UiMode } from '@/theme/UiModeProvider';
import { cn } from '@/lib/utils';

export interface ModeToggleProps {
  className?: string;
}

const OPTIONS: { mode: UiMode; label: string; icon: typeof Leaf; hint: string }[] = [
  { mode: 'easy', label: 'Easy', icon: Leaf, hint: 'Plain-language view for getting started' },
  { mode: 'expert', label: 'Expert', icon: SlidersHorizontal, hint: 'Full detail: every metric and control' },
];

export function ModeToggle({ className }: ModeToggleProps): JSX.Element {
  const { mode, setMode } = useUiMode();

  return (
    <div
      role="radiogroup"
      aria-label="Detail level"
      className={cn(
        'relative inline-flex items-center rounded-full border border-border bg-surface-2/70 p-0.5',
        className,
      )}
    >
      {/* Sliding active pill */}
      <span
        aria-hidden
        className={cn(
          'absolute top-0.5 bottom-0.5 w-[calc(50%-0.125rem)] rounded-full bg-surface shadow-soft',
          'transition-transform duration-200 ease-out',
          mode === 'expert' ? 'translate-x-[calc(100%+0.125rem)]' : 'translate-x-0',
        )}
      />
      {OPTIONS.map(({ mode: m, label, icon: Icon, hint }) => {
        const active = mode === m;
        return (
          <button
            key={m}
            type="button"
            role="radio"
            aria-checked={active}
            title={hint}
            onClick={() => setMode(m)}
            className={cn(
              'relative z-10 inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[0.8125rem] font-semibold tracking-tight transition-colors',
              'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-DEFAULT',
              active ? 'text-primary' : 'text-muted hover:text-text',
            )}
          >
            <Icon className="h-[1rem] w-[1rem]" aria-hidden />
            <span className="hidden sm:inline">{label}</span>
          </button>
        );
      })}
    </div>
  );
}
