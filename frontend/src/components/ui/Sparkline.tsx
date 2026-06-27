/**
 * Sparkline — a tiny inline trend line (pure SVG, no axes). Auto-colors by the
 * net direction of the series (up → success, down → danger) unless an explicit
 * `color` token is provided. Reads colors from CSS-var tokens via tokens.ts.
 */

import { useId, useMemo } from 'react';
import { useChartTokens } from '@/theme/tokens';
import { cn } from '@/lib/utils';

export interface SparklineProps {
  /** Series values (oldest → newest). */
  points: number[];
  width?: number;
  height?: number;
  /** Explicit stroke color (else auto by direction). */
  color?: string;
  /** Fill a soft area under the line. */
  fill?: boolean;
  strokeWidth?: number;
  className?: string;
  'aria-label'?: string;
}

export function Sparkline({
  points,
  width = 96,
  height = 28,
  color,
  fill = true,
  strokeWidth = 1.5,
  className,
  'aria-label': ariaLabel,
}: SparklineProps): JSX.Element {
  const tokens = useChartTokens();
  const gradientId = useId();

  const { line, area, stroke } = useMemo(() => {
    const n = points.length;
    if (n === 0) {
      return { line: '', area: '', stroke: color ?? tokens.muted };
    }
    const min = Math.min(...points);
    const max = Math.max(...points);
    const span = max - min || 1;
    const pad = strokeWidth;
    const innerW = width - pad * 2;
    const innerH = height - pad * 2;
    const stepX = n > 1 ? innerW / (n - 1) : 0;

    const coords = points.map((v, i) => {
      const x = pad + i * stepX;
      const y = pad + innerH - ((v - min) / span) * innerH;
      return [x, y] as const;
    });

    const linePath = coords
      .map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`)
      .join(' ');

    const first = coords[0];
    const last = coords[n - 1];
    const areaPath =
      first && last
        ? `${linePath} L${last[0].toFixed(2)},${(height - pad).toFixed(2)} L${first[0].toFixed(2)},${(
            height - pad
          ).toFixed(2)} Z`
        : '';

    const direction =
      (points[n - 1] ?? 0) - (points[0] ?? 0);
    const auto = direction >= 0 ? tokens.up : tokens.down;
    return { line: linePath, area: areaPath, stroke: color ?? auto };
  }, [points, width, height, strokeWidth, color, tokens]);

  if (points.length === 0) {
    return (
      <div
        className={cn('flex items-center justify-center text-[0.625rem] text-muted', className)}
        style={{ width, height }}
        aria-label={ariaLabel ?? 'No data'}
      >
        —
      </div>
    );
  }

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className={cn('overflow-visible', className)}
      role="img"
      aria-label={ariaLabel ?? 'Trend sparkline'}
      preserveAspectRatio="none"
    >
      {fill && (
        <>
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={stroke} stopOpacity={0.22} />
              <stop offset="100%" stopColor={stroke} stopOpacity={0} />
            </linearGradient>
          </defs>
          <path d={area} fill={`url(#${gradientId})`} stroke="none" />
        </>
      )}
      <path
        d={line}
        fill="none"
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
