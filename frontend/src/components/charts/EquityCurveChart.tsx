/**
 * EquityCurveChart — strategy equity vs a buy-&-hold benchmark over time, from a
 * BacktestResultDTO. Both series are growth-of-$1 curves (the backend's
 * downsampled `equityCurve`). Recharts area+line, themed via tokens.
 */

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useChartTokens } from '@/theme/tokens';
import type { BacktestResultDTO } from '@/lib/types';
import { formatDate, formatRatio } from '@/lib/format';
import { ChartTooltip, type TooltipRow } from './ChartTooltip';
import { ChartEmpty } from './ChartEmpty';

export interface EquityCurveChartProps {
  result: BacktestResultDTO;
  height?: number;
  className?: string;
}

interface Row {
  t: number;
  strategy: number;
  benchmark: number;
}

export function EquityCurveChart({
  result,
  height = 300,
  className,
}: EquityCurveChartProps): JSX.Element {
  const tokens = useChartTokens();
  const rows: Row[] = result.equityCurve.map((p) => ({
    t: p.t,
    strategy: p.strategy,
    benchmark: p.benchmark,
  }));

  if (rows.length === 0) {
    return (
      <ChartEmpty
        height={height}
        label={result.supported ? 'No equity curve available' : 'Strategy not backtestable for this asset'}
        className={className}
      />
    );
  }

  return (
    <div className={className} style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -6 }}>
          <defs>
            <linearGradient id="equity-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={tokens.primary} stopOpacity={0.22} />
              <stop offset="100%" stopColor={tokens.primary} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={tokens.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="t"
            tickFormatter={(v: number) => formatDate(v * 1000)}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: tokens.grid }}
            minTickGap={48}
          />
          <YAxis
            tickFormatter={(v: number) => `${formatRatio(v)}×`}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={48}
            domain={['auto', 'auto']}
          />
          <Tooltip
            cursor={{ stroke: tokens.muted, strokeDasharray: '3 3' }}
            content={({ active, payload, label }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0]?.payload as Row | undefined;
              if (!row) return null;
              const rowsOut: TooltipRow[] = [
                { label: 'Strategy', value: `${formatRatio(row.strategy)}×`, color: tokens.primary },
                { label: 'Buy & Hold', value: `${formatRatio(row.benchmark)}×`, color: tokens.muted },
              ];
              return <ChartTooltip title={formatDate(Number(label) * 1000)} rows={rowsOut} />;
            }}
          />
          <Legend
            verticalAlign="top"
            height={24}
            iconType="plainline"
            wrapperStyle={{ fontSize: 11, color: tokens.muted }}
          />
          <Area
            type="monotone"
            name="Strategy"
            dataKey="strategy"
            stroke={tokens.primary}
            strokeWidth={2}
            fill="url(#equity-fill)"
            isAnimationActive={false}
            dot={false}
          />
          <Line
            type="monotone"
            name="Buy & Hold"
            dataKey="benchmark"
            stroke={tokens.muted}
            strokeWidth={1.5}
            strokeDasharray="5 4"
            isAnimationActive={false}
            dot={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
