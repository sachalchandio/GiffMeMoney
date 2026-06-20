/**
 * StrategyCard — a scannable, educational card for a single {@link StrategyMeta}.
 *
 * Renders the strategy's name + category pill, its plain-English summary, the
 * compact formula (in a monospace chip), the data inputs it consumes, and its
 * academic / book references. Used in the Strategy Lab gallery grid; clicking the
 * card selects the strategy (the parent wires `onSelect` + `selected`).
 *
 * Colors come only from semantic tokens (category → Badge tone). No `any`.
 */

import { BookOpen, FunctionSquare } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge, type BadgeTone } from '@/components/ui/Badge';
import { cn } from '@/lib/utils';
import type { StrategyCategory, StrategyMeta } from '@/lib/types';

export interface StrategyCardProps {
  meta: StrategyMeta;
  /** Marks the card as the active selection (ring + raised surface). */
  selected?: boolean;
  /** Click handler — selecting drives the detail panel. */
  onSelect?: (id: string) => void;
  className?: string;
}

/** Map each strategy category to a stable Badge tone (semantic tokens only). */
export function categoryTone(category: StrategyCategory): BadgeTone {
  switch (category) {
    case 'Valuation':
      return 'primary';
    case 'Factor':
      return 'accent';
    case 'Risk-Adjusted':
      return 'success';
    case 'Technical':
      return 'warning';
    case 'Statistical':
      return 'accent';
    case 'Portfolio':
      return 'primary';
    case 'Fundamental':
      return 'success';
    case 'Derivatives':
      return 'danger';
    default:
      return 'neutral';
  }
}

export function StrategyCard({
  meta,
  selected = false,
  onSelect,
  className,
}: StrategyCardProps): JSX.Element {
  const interactive = Boolean(onSelect);

  return (
    <Card
      interactive={interactive}
      onClick={interactive ? () => onSelect?.(meta.id) : undefined}
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : undefined}
      aria-pressed={interactive ? selected : undefined}
      onKeyDown={
        interactive
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onSelect?.(meta.id);
              }
            }
          : undefined
      }
      className={cn(
        'flex h-full flex-col gap-2.5',
        interactive && 'cursor-pointer',
        selected
          ? 'border-primary/60 shadow-card ring-2 ring-DEFAULT'
          : interactive && 'hover:border-muted/50',
        className,
      )}
    >
      {/* Header: name + category */}
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-sm font-semibold leading-snug tracking-tight text-text">
          {meta.name}
        </h3>
        <Badge tone={categoryTone(meta.category)} size="sm" variant="soft" className="shrink-0">
          {meta.category}
        </Badge>
      </div>

      {/* Summary */}
      <p className="line-clamp-3 text-xs leading-relaxed text-muted">{meta.summary}</p>

      {/* Formula chip */}
      {meta.formula && (
        <div className="flex items-start gap-1.5 rounded-xl bg-surface-2 px-2.5 py-1.5">
          <FunctionSquare className="mt-px h-3.5 w-3.5 shrink-0 text-muted" aria-hidden />
          <code className="break-words font-mono text-[11px] leading-snug text-text">
            {meta.formula}
          </code>
        </div>
      )}

      {/* Inputs */}
      {meta.inputs.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {meta.inputs.slice(0, 4).map((input) => (
            <span
              key={input}
              className="rounded-md bg-surface-2 px-1.5 py-0.5 text-[10px] font-medium text-muted"
            >
              {input}
            </span>
          ))}
          {meta.inputs.length > 4 && (
            <span className="rounded-md px-1 py-0.5 text-[10px] font-medium text-muted">
              +{meta.inputs.length - 4}
            </span>
          )}
        </div>
      )}

      {/* References — pin to the bottom for a tidy grid */}
      {meta.references.length > 0 && (
        <div className="mt-auto flex items-start gap-1.5 pt-1 text-[10px] leading-snug text-muted">
          <BookOpen className="mt-px h-3 w-3 shrink-0 opacity-70" aria-hidden />
          <span className="truncate" title={meta.references.join(' · ')}>
            {meta.references.join(' · ')}
          </span>
        </div>
      )}
    </Card>
  );
}
