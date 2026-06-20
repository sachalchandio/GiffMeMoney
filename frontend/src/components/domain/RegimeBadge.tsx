/**
 * RegimeBadge — a compact market-regime pill (bull / bear / neutral) with the
 * volatility-regime bucket. Maps the regime to a semantic tone (bull → success,
 * bear → danger, neutral → warning) and surfaces the trend conviction on hover.
 * Colors via semantic tokens only.
 */

import { TrendingDown, TrendingUp, Activity } from 'lucide-react';
import type { ReactNode } from 'react';
import { Badge, type BadgeSize, type BadgeTone } from '@/components/ui/Badge';
import { Tooltip } from '@/components/ui/Tooltip';
import type { RegimeInfo } from '@/lib/types';

export interface RegimeBadgeProps {
  regime: RegimeInfo;
  size?: BadgeSize;
  /** Append the volatility-regime bucket (e.g. "· high vol"). */
  showVol?: boolean;
  className?: string;
}

function toneFor(regime: RegimeInfo['regime']): BadgeTone {
  switch (regime) {
    case 'bull':
      return 'success';
    case 'bear':
      return 'danger';
    case 'neutral':
    default:
      return 'warning';
  }
}

function iconFor(regime: RegimeInfo['regime']): ReactNode {
  switch (regime) {
    case 'bull':
      return <TrendingUp className="h-3 w-3" aria-hidden />;
    case 'bear':
      return <TrendingDown className="h-3 w-3" aria-hidden />;
    case 'neutral':
    default:
      return <Activity className="h-3 w-3" aria-hidden />;
  }
}

function regimeLabel(regime: RegimeInfo['regime']): string {
  switch (regime) {
    case 'bull':
      return 'Bull regime';
    case 'bear':
      return 'Bear regime';
    case 'neutral':
    default:
      return 'Neutral regime';
  }
}

export function RegimeBadge({
  regime,
  size = 'md',
  showVol = true,
  className,
}: RegimeBadgeProps): JSX.Element {
  const label = regimeLabel(regime.regime);
  const volText = `${regime.volRegime} vol`;

  return (
    <Tooltip
      content={
        <span className="tnum">
          {label} · trend {regime.trend >= 0 ? '+' : ''}
          {regime.trend.toFixed(2)} · {volText} · score {regime.score.toFixed(2)}
        </span>
      }
    >
      <Badge tone={toneFor(regime.regime)} size={size} icon={iconFor(regime.regime)} className={className}>
        {label}
        {showVol && <span className="font-normal opacity-80"> · {volText}</span>}
      </Badge>
    </Tooltip>
  );
}
