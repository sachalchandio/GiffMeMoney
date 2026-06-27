/**
 * ScoreGauge — a semicircular gauge for a composite/signal score in -100..100.
 * The arc fills proportionally and is colored by the implied stance (buy →
 * success, hold → warning, sell → danger). The derived stance label is shown
 * below the numeric value. Colors read from CSS-var tokens (no hardcoded hex).
 */

import { useId, useMemo } from 'react';
import { useChartTokens } from '@/theme/tokens';
import { clamp, stanceFromScore, stanceTone } from '@/lib/utils';
import { stanceLabel } from '@/lib/format';
import { cn } from '@/lib/utils';
import type { Stance } from '@/lib/types';

export interface ScoreGaugeProps {
  /** Composite/signal score in -100..100. */
  score: number;
  size?: number;
  /** Optional caption rendered under the stance label. */
  caption?: string;
  /** Override the derived stance (e.g. the backend recommendation). */
  stance?: Stance;
  className?: string;
  /** Hide the numeric/text center labels (pure arc). */
  hideLabel?: boolean;
}

function toneColor(stance: Stance, tokens: ReturnType<typeof useChartTokens>): string {
  switch (stanceTone(stance)) {
    case 'positive':
      return tokens.up;
    case 'negative':
      return tokens.down;
    case 'neutral':
    default:
      return tokens.warning;
  }
}

/** Point on a circle at `deg` (0° = left, 180° = right of a top semicircle). */
function polar(cx: number, cy: number, r: number, deg: number): [number, number] {
  const rad = (deg * Math.PI) / 180;
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
}

function arcPath(cx: number, cy: number, r: number, startDeg: number, endDeg: number): string {
  const [x1, y1] = polar(cx, cy, r, startDeg);
  const [x2, y2] = polar(cx, cy, r, endDeg);
  const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
  const sweep = endDeg > startDeg ? 1 : 0;
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${r} ${r} 0 ${largeArc} ${sweep} ${x2.toFixed(2)} ${y2.toFixed(2)}`;
}

export function ScoreGauge({
  score,
  size = 160,
  caption,
  stance,
  className,
  hideLabel = false,
}: ScoreGaugeProps): JSX.Element {
  const tokens = useChartTokens();
  const gradId = useId();
  const value = clamp(score, -100, 100);
  const derived = stance ?? stanceFromScore(value);
  const color = toneColor(derived, tokens);

  const geom = useMemo(() => {
    const stroke = Math.max(8, Math.round(size * 0.075));
    const cx = size / 2;
    const r = size / 2 - stroke / 2 - 2;
    const cy = size / 2 + r * 0.18; // nudge down so the semicircle is centered
    // Semicircle from 180° (left) sweeping through 360°/0° (top) to 0° (right).
    const start = 180;
    const end = 360;
    const frac = (value + 100) / 200; // 0..1
    const valueDeg = start + frac * (end - start);
    return { stroke, cx, cy, r, start, end, valueDeg, height: cy + stroke / 2 + 2 };
  }, [size, value]);

  return (
    <div
      className={cn('inline-flex flex-col items-center', className)}
      role="meter"
      aria-valuemin={-100}
      aria-valuemax={100}
      aria-valuenow={Math.round(value)}
      aria-label={`Score ${Math.round(value)} of 100, ${stanceLabel(derived)}`}
    >
      <svg width={size} height={geom.height} viewBox={`0 0 ${size} ${geom.height}`}>
        <defs>
          <linearGradient id={gradId} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={color} stopOpacity={0.65} />
            <stop offset="100%" stopColor={color} stopOpacity={1} />
          </linearGradient>
        </defs>
        {/* Track */}
        <path
          d={arcPath(geom.cx, geom.cy, geom.r, geom.start, geom.end)}
          fill="none"
          stroke={tokens.surface2}
          strokeWidth={geom.stroke}
          strokeLinecap="round"
        />
        {/* Value arc */}
        <path
          d={arcPath(geom.cx, geom.cy, geom.r, geom.start, geom.valueDeg)}
          fill="none"
          stroke={`url(#${gradId})`}
          strokeWidth={geom.stroke}
          strokeLinecap="round"
        />
      </svg>
      {!hideLabel && (
        <div className="-mt-[1.6em] flex flex-col items-center">
          <span className="text-2xl font-semibold tracking-tight tnum text-text">
            {value > 0 ? '+' : ''}
            {Math.round(value)}
          </span>
          <span className="text-xs font-medium" style={{ color }}>
            {stanceLabel(derived)}
          </span>
          {caption && <span className="mt-0.5 text-[0.625rem] text-muted">{caption}</span>}
        </div>
      )}
    </div>
  );
}
