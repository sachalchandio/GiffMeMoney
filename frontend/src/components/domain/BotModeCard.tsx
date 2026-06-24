/**
 * BotModeCard — a selectable preset auto-trader mode ({@link BotMode}). Shows the
 * mode name, a risk badge (low / moderate / high), the plain-English summary, and
 * compact chips for the rotation style, portfolio objective and max held names.
 * Clicking selects the mode (parent wires `onSelect` + `selected`).
 *
 * The rotation is always momentum / bandit style — allocate MORE to recent
 * winners, LESS to losers; the bot never martingales (never doubles down on a
 * loser to "recover"). The card states this so the behaviour is never implied to
 * be a guaranteed-profit system. Everything is a SIMULATION on synthetic data.
 *
 * Semantic tokens only — no hardcoded hex. Light/dark aware.
 */

import { Bot, Layers, Repeat, ShieldCheck } from 'lucide-react';
import { Card } from '@/components/ui/Card';
import { Badge, type BadgeTone } from '@/components/ui/Badge';
import { Tooltip } from '@/components/ui/Tooltip';
import type { BotMode, BotRiskLevel, BotRotation } from '@/lib/types';
import { cn } from '@/lib/utils';

export interface BotModeCardProps {
  mode: BotMode;
  /** Marks the card as the active selection (ring + raised surface). */
  selected?: boolean;
  /** Click handler — selecting drives the run config. */
  onSelect?: (id: BotMode['id']) => void;
  className?: string;
}

function riskTone(level: BotRiskLevel): BadgeTone {
  switch (level) {
    case 'low':
      return 'success';
    case 'moderate':
      return 'warning';
    case 'high':
      return 'danger';
    default:
      return 'neutral';
  }
}

function riskLabel(level: BotRiskLevel): string {
  return `${level.charAt(0).toUpperCase()}${level.slice(1)} risk`;
}

/** Human label for a rotation style (momentum / bandit framing). */
function rotationLabel(rotation: BotRotation): string {
  switch (rotation) {
    case 'none':
      return 'Rebalance only';
    case 'slow':
      return 'Slow momentum';
    case 'moderate':
      return 'Momentum';
    case 'fast':
      return 'Fast momentum';
    case 'bandit':
      return 'Adaptive bandit';
    default:
      return rotation;
  }
}

function Chip({ icon, children, hint }: { icon: React.ReactNode; children: React.ReactNode; hint: string }): JSX.Element {
  return (
    <Tooltip content={hint}>
      <span className="inline-flex items-center gap-1 rounded-md bg-surface-2 px-1.5 py-0.5 text-[10px] font-medium text-muted">
        <span className="text-muted">{icon}</span>
        {children}
      </span>
    </Tooltip>
  );
}

export function BotModeCard({
  mode,
  selected = false,
  onSelect,
  className,
}: BotModeCardProps): JSX.Element {
  const interactive = Boolean(onSelect);

  return (
    <Card
      interactive={interactive}
      onClick={interactive ? () => onSelect?.(mode.id) : undefined}
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : undefined}
      aria-pressed={interactive ? selected : undefined}
      onKeyDown={
        interactive
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                onSelect?.(mode.id);
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
      {/* Header: name + risk */}
      <div className="flex items-start justify-between gap-2">
        <h3 className="flex items-center gap-1.5 text-sm font-semibold leading-snug tracking-tight text-text">
          <Bot className="h-4 w-4 text-primary" aria-hidden />
          {mode.name}
        </h3>
        <Badge tone={riskTone(mode.riskLevel)} size="sm" variant="soft" className="shrink-0">
          {riskLabel(mode.riskLevel)}
        </Badge>
      </div>

      {/* Summary */}
      <p className="line-clamp-3 text-xs leading-relaxed text-muted">{mode.summary}</p>

      {/* Behaviour chips */}
      <div className="flex flex-wrap gap-1">
        <Chip
          icon={<Repeat className="h-3 w-3" aria-hidden />}
          hint="Rotation tilts toward recent winners (momentum / bandit) — never toward losers."
        >
          {rotationLabel(mode.rotation)}
        </Chip>
        <Chip icon={<Layers className="h-3 w-3" aria-hidden />} hint="Portfolio objective driving each rebalance.">
          {mode.objective}
        </Chip>
        <Chip
          icon={<ShieldCheck className="h-3 w-3" aria-hidden />}
          hint="Maximum number of sleeves (assets) held at once."
        >
          ≤ {mode.maxNames} names
        </Chip>
      </div>

      {/* Honesty note — momentum, not martingale; pinned to the bottom. */}
      <p className="mt-auto pt-1 text-[10px] leading-snug text-muted">
        Adds to recent winners, trims losers — never doubles down to chase losses.
      </p>
    </Card>
  );
}
