/**
 * EfficientFrontierChart — Markowitz efficient frontier (risk vs return) with the
 * capital market line and the tangency (optimal) portfolio highlighted. Axes are
 * decimals from the backend rendered as percentages. Recharts, themed via tokens.
 */

import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import { useChartTokens } from '@/theme/tokens';
import type { PortfolioResult } from '@/lib/types';
import { formatFractionPct, formatRatio } from '@/lib/format';
import { ChartTooltip, type TooltipRow } from './ChartTooltip';
import { ChartEmpty } from './ChartEmpty';

export interface EfficientFrontierChartProps {
  result: PortfolioResult;
  height?: number;
  className?: string;
}

interface Pt {
  x: number;
  y: number;
  sharpe: number;
}

export function EfficientFrontierChart({
  result,
  height = 300,
  className,
}: EfficientFrontierChartProps): JSX.Element {
  const tokens = useChartTokens();

  const frontier: Pt[] = result.efficientFrontier.map((p) => ({
    x: p.volatility,
    y: p.expectedReturn,
    sharpe: p.sharpe,
  }));
  const cml: Pt[] = result.capitalMarketLine.map((p) => ({
    x: p.volatility,
    y: p.expectedReturn,
    sharpe: p.sharpe,
  }));
  const tangency: Pt[] = [
    { x: result.volatility, y: result.expectedReturn, sharpe: result.sharpe },
  ];

  if (frontier.length === 0 && cml.length === 0) {
    return <ChartEmpty height={height} label="Run the optimizer to see the frontier" className={className} />;
  }

  return (
    <div className={className} style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 8, right: 16, bottom: 16, left: 0 }}>
          <CartesianGrid stroke={tokens.grid} strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="x"
            name="Volatility"
            domain={['auto', 'auto']}
            tickFormatter={(v: number) => formatFractionPct(v, { digits: 0 })}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: tokens.grid }}
            label={{ value: 'Volatility', fill: tokens.muted, fontSize: 11, position: 'insideBottom', offset: -6 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="Return"
            domain={['auto', 'auto']}
            tickFormatter={(v: number) => formatFractionPct(v, { digits: 0 })}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={48}
          />
          <ZAxis range={[40, 40]} />
          <Tooltip
            cursor={{ stroke: tokens.muted, strokeDasharray: '3 3' }}
            content={({ active, payload }) => {
              if (!active || !payload || payload.length === 0) return null;
              const p = payload[0]?.payload as Pt | undefined;
              if (!p) return null;
              const rowsOut: TooltipRow[] = [
                { label: 'Return', value: formatFractionPct(p.y), color: tokens.primary },
                { label: 'Volatility', value: formatFractionPct(p.x), color: tokens.muted },
                { label: 'Sharpe', value: formatRatio(p.sharpe) },
              ];
              return <ChartTooltip rows={rowsOut} />;
            }}
          />
          {/* Capital market line (drawn as a connected scatter). */}
          {cml.length > 0 && (
            <Scatter
              name="Capital Market Line"
              data={cml}
              line={{ stroke: tokens.accent, strokeWidth: 1.5, strokeDasharray: '5 4' }}
              fill="transparent"
              shape={() => <g />}
              isAnimationActive={false}
            />
          )}
          {/* Efficient frontier curve. */}
          <Scatter
            name="Efficient Frontier"
            data={frontier}
            line={{ stroke: tokens.primary, strokeWidth: 2 }}
            fill={tokens.primary}
            fillOpacity={0.5}
            isAnimationActive={false}
          />
          {/* Tangency / optimal portfolio. */}
          <Scatter
            name="Optimal"
            data={tangency}
            fill={tokens.warning}
            shape="star"
            isAnimationActive={false}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
