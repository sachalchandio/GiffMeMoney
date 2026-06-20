/**
 * StanceBadge — a stance pill (STRONG_BUY … STRONG_SELL) colored by its tone
 * (buy → success, hold → warning, sell → danger), per CONTRACT §8. Wraps the
 * generic Badge and uses the shared stance → tone mapping from lib/utils.
 */

import { Badge, type BadgeSize, type BadgeTone, type BadgeVariant } from './Badge';
import type { Stance } from '@/lib/types';
import { stanceLabel } from '@/lib/format';
import { stanceTone } from '@/lib/utils';

export interface StanceBadgeProps {
  stance: Stance;
  size?: BadgeSize;
  variant?: BadgeVariant;
  className?: string;
}

function toneForStance(stance: Stance): BadgeTone {
  switch (stanceTone(stance)) {
    case 'positive':
      return 'success';
    case 'negative':
      return 'danger';
    case 'neutral':
    default:
      return 'warning';
  }
}

export function StanceBadge({
  stance,
  size = 'md',
  variant = 'soft',
  className,
}: StanceBadgeProps): JSX.Element {
  return (
    <Badge tone={toneForStance(stance)} size={size} variant={variant} className={className}>
      {stanceLabel(stance)}
    </Badge>
  );
}
