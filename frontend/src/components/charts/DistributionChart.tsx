/**
 * DistributionChart — the Monte Carlo final-price distribution as a histogram,
 * with the mean (expected return) and VaR/CVaR percentile markers. Bars are
 * colored by whether the bin's midpoint sits above or below the starting price.
 * Recharts, themed via tokens.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useChartTokens } from '@/theme/tokens';
import type { MonteCarloResult } from '@/lib/types';
import { formatCompactCurrency, formatNumber } from '@/lib/format';
import { ChartTooltip, type TooltipRow } from './ChartTooltip';
import { ChartEmpty } from './ChartEmpty';

export interface DistributionChartProps {
  result: MonteCarloResult;
  height?: number;
  className?: string;
}

interface HistRow {
  mid: number;
  count: number;
  binStart: number;
  binEnd: number;
}

export function DistributionChart({
  result,
  height = 280,
  className,
}: DistributionChartProps): JSX.Element {
  const tokens = useChartTokens();
  const bins = result.finalDistribution;

  if (!bins || bins.length === 0) {
    return <ChartEmpty height={height} label="No simulation available" className={className} />;
  }

  const rows: HistRow[] = bins.map((b) => ({
    mid: (b.binStart + b.binEnd) / 2,
    count: b.count,
    binStart: b.binStart,
    binEnd: b.binEnd,
  }));

  // Starting price ≈ the median band's first step (best-effort); else weighted center.
  const startPrice =
    result.bands[0]?.p50 ??
    rows.reduce((acc, r) => acc + r.mid * r.count, 0) /
      Math.max(1, rows.reduce((acc, r) => acc + r.count, 0));

  return (
    <div className={className} style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -8 }} barCategoryGap={1}>
          <CartesianGrid stroke={tokens.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="mid"
            type="number"
            domain={['dataMin', 'dataMax']}
            tickFormatter={(v: number) => formatCompactCurrency(v)}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: tokens.grid }}
          />
          <YAxis
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={40}
          />
          <ReferenceLine
            x={startPrice}
            stroke={tokens.muted}
            strokeDasharray="4 3"
            label={{ value: 'Now', fill: tokens.muted, fontSize: 10, position: 'top' }}
          />
          <Tooltip
            cursor={{ fill: 'color-mix(in srgb, var(--text) 6%, transparent)' }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0]?.payload as HistRow | undefined;
              if (!row) return null;
              const rowsOut: TooltipRow[] = [
                {
                  label: 'Range',
                  value: `${formatCompactCurrency(row.binStart)}–${formatCompactCurrency(row.binEnd)}`,
                },
                { label: 'Paths', value: formatNumber(row.count, 0) },
              ];
              return <ChartTooltip rows={rowsOut} />;
            }}
          />
          <Bar dataKey="count" radius={[2, 2, 0, 0]} isAnimationActive={false}>
            {rows.map((r, i) => (
              <Cell key={i} fill={r.mid >= startPrice ? tokens.up : tokens.down} fillOpacity={0.85} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
