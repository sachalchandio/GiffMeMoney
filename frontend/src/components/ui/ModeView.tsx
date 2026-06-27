/**
 * Mode-aware building blocks for the Easy / Expert lenses.
 *
 *  - {@link EasyOnly} / {@link ExpertOnly} — render children only in one mode.
 *  - {@link Explain} — a jargon term that, in Easy mode, shows the plain-English
 *    phrase (with the technical term on hover); in Expert mode shows the term
 *    (with the plain phrase as the hover definition). One word, two audiences.
 *  - {@link WhatThisMeans} — a soft callout used in Easy mode to translate a
 *    number or chart into a single human sentence.
 *
 * Keeping this logic in tiny primitives means every page can speak to both
 * audiences without branching its whole layout.
 */

import type { ReactNode } from 'react';
import { Info, Lightbulb } from 'lucide-react';
import { useUiMode } from '@/theme/UiModeProvider';
import { cn } from '@/lib/utils';

/** Render children only when the Easy (beginner) lens is active. */
export function EasyOnly({ children }: { children: ReactNode }): JSX.Element | null {
  const { isEasy } = useUiMode();
  return isEasy ? <>{children}</> : null;
}

/** Render children only when the Expert (full-detail) lens is active. */
export function ExpertOnly({ children }: { children: ReactNode }): JSX.Element | null {
  const { isExpert } = useUiMode();
  return isExpert ? <>{children}</> : null;
}

export interface ExplainProps {
  /** The technical term (shown in Expert mode, on hover in Easy mode). */
  term: string;
  /** The plain-English phrase (shown in Easy mode, on hover in Expert mode). */
  plain: string;
  className?: string;
}

/**
 * A term that adapts to the reader: Easy mode leads with the plain phrase and
 * keeps the jargon on hover; Expert mode leads with the term and keeps the
 * explanation on hover. Underlined with dots so it reads as "hover me".
 */
export function Explain({ term, plain, className }: ExplainProps): JSX.Element {
  const { isEasy } = useUiMode();
  const shown = isEasy ? plain : term;
  const tip = isEasy ? term : plain;
  return (
    <span
      title={tip}
      className={cn(
        'underline decoration-dotted decoration-muted/60 underline-offset-2 cursor-help',
        className,
      )}
    >
      {shown}
    </span>
  );
}

export interface WhatThisMeansProps {
  children: ReactNode;
  /** Tone of the callout. */
  tone?: 'neutral' | 'good' | 'warn';
  className?: string;
}

/**
 * A friendly "what this means" callout — shown only in Easy mode — that turns a
 * metric or chart into one plain sentence. In Expert mode it renders nothing.
 */
export function WhatThisMeans({
  children,
  tone = 'neutral',
  className,
}: WhatThisMeansProps): JSX.Element | null {
  const { isEasy } = useUiMode();
  if (!isEasy) return null;
  const palette =
    tone === 'good'
      ? 'border-success/30 bg-success/8 text-success'
      : tone === 'warn'
        ? 'border-warning/40 bg-warning/8 text-warning'
        : 'border-accent/30 bg-accent/8 text-accent';
  return (
    <div
      className={cn(
        'flex items-start gap-2 rounded-xl border px-3 py-2 text-[0.8125rem] leading-relaxed',
        palette,
        className,
      )}
    >
      <Lightbulb className="mt-0.5 h-[1rem] w-[1rem] shrink-0" aria-hidden />
      <span className="[&_b]:font-semibold">{children}</span>
    </div>
  );
}

/**
 * A tiny inline info chip with a hover title — used to attach a one-line
 * definition to a label in either mode without taking layout space.
 */
export function InfoHint({ text, className }: { text: string; className?: string }): JSX.Element {
  return (
    <span title={text} className={cn('inline-flex cursor-help text-muted/70', className)}>
      <Info className="h-[0.875rem] w-[0.875rem]" aria-hidden />
      <span className="sr-only">{text}</span>
    </span>
  );
}
