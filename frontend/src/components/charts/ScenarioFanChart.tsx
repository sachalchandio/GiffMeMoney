/**
 * ScenarioFanChart — projected total return across the 5 horizons with a
 * bull / base / bear "fan". Uses the optional V2 scenario fields
 * (`bullPct` / `basePct` / `bearPct`) when present and falls back to the
 * ~95th / mean / ~5th percentile band otherwise. Recharts, themed via tokens.
 */

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useChartTokens } from '@/theme/tokens';
import type { ExpectedReturn } from '@/lib/types';
import { formatPct, horizonLabel } from '@/lib/format';
import { ChartTooltip, type TooltipRow } from './ChartTooltip';
import { ChartEmpty } from './ChartEmpty';

export interface ScenarioFanChartProps {
  expectedReturns: ExpectedReturn[];
  height?: number;
  className?: string;
}

interface FanRow {
  horizon: string;
  bull: number;
  base: number;
  bear: number;
  /** Stacked fan: floor (bear) + lower band + upper band. */
  floor: number;
  lowerBand: number;
  upperBand: number;
}

function toRows(returns: ExpectedReturn[]): FanRow[] {
  return returns.map((r) => {
    const base = r.basePct ?? r.expectedReturnPct;
    const bull = r.bullPct ?? r.high;
    const bear = r.bearPct ?? r.low;
    const lo = Math.min(bull, base, bear);
    const hi = Math.max(bull, base, bear);
    return {
      horizon: horizonLabel(r.horizon),
      bull,
      base,
      bear,
      floor: lo,
      lowerBand: base - lo,
      upperBand: hi - base,
    };
  });
}

export function ScenarioFanChart({
  expectedReturns,
  height = 280,
  className,
}: ScenarioFanChartProps): JSX.Element {
  const tokens = useChartTokens();
  const rows = toRows(expectedReturns);

  if (rows.length === 0) {
    return <ChartEmpty height={height} label="No projection available" className={className} />;
  }

  return (
    <div className={className} style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -8 }}>
          <defs>
            <linearGradient id="fan-up" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={tokens.up} stopOpacity={0.24} />
              <stop offset="100%" stopColor={tokens.up} stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="fan-down" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={tokens.down} stopOpacity={0.05} />
              <stop offset="100%" stopColor={tokens.down} stopOpacity={0.2} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={tokens.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="horizon"
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: tokens.grid }}
          />
          <YAxis
            tickFormatter={(v: number) => `${v.toFixed(0)}%`}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={48}
          />
          <ReferenceLine y={0} stroke={tokens.grid} />
          <Tooltip
            cursor={{ stroke: tokens.muted, strokeDasharray: '3 3' }}
            content={({ active, payload, label }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0]?.payload as FanRow | undefined;
              if (!row) return null;
              const rowsOut: TooltipRow[] = [
                { label: 'Bull', value: formatPct(row.bull, { sign: true }), color: tokens.up },
                { label: 'Base', value: formatPct(row.base, { sign: true }), color: tokens.primary },
                { label: 'Bear', value: formatPct(row.bear, { sign: true }), color: tokens.down },
              ];
              return <ChartTooltip title={String(label)} rows={rowsOut} />;
            }}
          />
          {/* Shaded fan via a stacked floor + two bands. */}
          <Area
            type="monotone"
            dataKey="floor"
            stackId="fan"
            stroke="none"
            fill="transparent"
            isAnimationActive={false}
            activeDot={false}
          />
          <Area
            type="monotone"
            dataKey="lowerBand"
            stackId="fan"
            stroke="none"
            fill="url(#fan-down)"
            isAnimationActive={false}
            activeDot={false}
          />
          <Area
            type="monotone"
            dataKey="upperBand"
            stackId="fan"
            stroke="none"
            fill="url(#fan-up)"
            isAnimationActive={false}
            activeDot={false}
          />
          {/* Base trajectory on top. */}
          <Line
            type="monotone"
            dataKey="base"
            stroke={tokens.primary}
            strokeWidth={2}
            dot={{ r: 3, fill: tokens.primary, strokeWidth: 0 }}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
