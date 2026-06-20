/**
 * FactorRadar — a radar/spider chart of normalized factor scores (0..100) across
 * named axes (e.g. Valuation, Momentum, Quality, Risk, Technical). Optionally
 * overlays a second comparison series. Recharts, themed via tokens.
 */

import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { useChartTokens } from '@/theme/tokens';
import { formatNumber } from '@/lib/format';
import { ChartTooltip, type TooltipRow } from './ChartTooltip';
import { ChartEmpty } from './ChartEmpty';

/** One radar axis: a label and a 0..100 normalized value. */
export interface FactorAxis {
  label: string;
  value: number;
  /** Optional comparison value (e.g. universe average). */
  compare?: number;
}

export interface FactorRadarProps {
  axes: FactorAxis[];
  /** Name of the primary series (legend/tooltip). */
  seriesName?: string;
  /** Name of the comparison series, when `compare` values are present. */
  compareName?: string;
  height?: number;
  /** Radial domain max (default 100). */
  max?: number;
  className?: string;
}

interface RadarRow {
  axis: string;
  value: number;
  compare: number | undefined;
}

export function FactorRadar({
  axes,
  seriesName = 'Score',
  compareName = 'Benchmark',
  height = 280,
  max = 100,
  className,
}: FactorRadarProps): JSX.Element {
  const tokens = useChartTokens();

  if (axes.length === 0) {
    return <ChartEmpty height={height} label="No factor data" className={className} />;
  }

  const hasCompare = axes.some((a) => typeof a.compare === 'number');
  const rows: RadarRow[] = axes.map((a) => ({
    axis: a.label,
    value: a.value,
    compare: a.compare,
  }));

  return (
    <div className={className} style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <RadarChart data={rows} margin={{ top: 8, right: 16, bottom: 8, left: 16 }}>
          <PolarGrid stroke={tokens.grid} />
          <PolarAngleAxis dataKey="axis" tick={{ fill: tokens.muted, fontSize: 11 }} />
          <PolarRadiusAxis
            domain={[0, max]}
            tick={{ fill: tokens.muted, fontSize: 9 }}
            axisLine={false}
            tickCount={4}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0]?.payload as RadarRow | undefined;
              if (!row) return null;
              const out: TooltipRow[] = [
                { label: seriesName, value: formatNumber(row.value, 0), color: tokens.primary },
              ];
              if (hasCompare && typeof row.compare === 'number') {
                out.push({ label: compareName, value: formatNumber(row.compare, 0), color: tokens.accent });
              }
              return <ChartTooltip title={String(label)} rows={out} />;
            }}
          />
          {hasCompare && (
            <Radar
              name={compareName}
              dataKey="compare"
              stroke={tokens.accent}
              strokeWidth={1.5}
              fill={tokens.accent}
              fillOpacity={0.1}
              isAnimationActive={false}
            />
          )}
          <Radar
            name={seriesName}
            dataKey="value"
            stroke={tokens.primary}
            strokeWidth={2}
            fill={tokens.primary}
            fillOpacity={0.22}
            isAnimationActive={false}
          />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
