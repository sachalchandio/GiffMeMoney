/**
 * BotEquityChart — the simulated auto-trader's paper value vs an equal-weight
 * buy-&-hold benchmark over the backtest, with the bot's drawdown shaded below
 * a baseline. Both value series share a left axis (dollars); the drawdown is a
 * faint danger-tinted area on a hidden right axis so the eye reads "how far
 * below the peak" without rescaling the value curves.
 *
 * Everything shown is a SIMULATION on synthetic data — paper-traded, no real
 * funds moved. Recharts + theme tokens (no hardcoded hex). Light/dark aware.
 */

import { useMemo } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useChartTokens } from '@/theme/tokens';
import type { BotEquityPoint } from '@/lib/types';
import { formatCompactCurrency, formatCurrency, formatDate, formatPct } from '@/lib/format';
import { ChartTooltip, type TooltipRow } from '@/components/charts/ChartTooltip';
import { ChartEmpty } from '@/components/charts/ChartEmpty';

export interface BotEquityChartProps {
  equityCurve: BotEquityPoint[];
  height?: number;
  /** Overlay the bot's drawdown as a shaded danger area (default: true). */
  showDrawdown?: boolean;
  className?: string;
}

interface Row {
  t: number;
  bot: number;
  benchmark: number;
  /** Drawdown in percent, clamped to <= 0 for the shaded area. */
  drawdown: number;
}

export function BotEquityChart({
  equityCurve,
  height = 300,
  showDrawdown = true,
  className,
}: BotEquityChartProps): JSX.Element {
  const tokens = useChartTokens();

  const { rows, minDrawdown } = useMemo(() => {
    const out: Row[] = equityCurve.map((p) => ({
      t: p.t,
      bot: p.botValue,
      benchmark: p.benchmarkValue,
      drawdown: Math.min(0, p.drawdownPct),
    }));
    const worst = out.reduce((acc, r) => Math.min(acc, r.drawdown), 0);
    return { rows: out, minDrawdown: worst };
  }, [equityCurve]);

  if (rows.length === 0) {
    return <ChartEmpty height={height} label="No simulated equity curve yet" className={className} />;
  }

  // Give the drawdown axis a little headroom below the worst value.
  const ddDomainMin = Math.min(-1, Math.floor(minDrawdown * 1.15));

  return (
    <div className={className} style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -6 }}>
          <defs>
            <linearGradient id="bot-equity-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={tokens.primary} stopOpacity={0.24} />
              <stop offset="100%" stopColor={tokens.primary} stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="bot-drawdown-fill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={tokens.down} stopOpacity={0.04} />
              <stop offset="100%" stopColor={tokens.down} stopOpacity={0.2} />
            </linearGradient>
          </defs>

          <CartesianGrid stroke={tokens.grid} strokeDasharray="3 3" vertical={false} />

          <XAxis
            dataKey="t"
            tickFormatter={(v: number) => formatDate(v)}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: tokens.grid }}
            minTickGap={48}
          />

          {/* Left axis: dollar value of the two equity series. */}
          <YAxis
            yAxisId="value"
            tickFormatter={(v: number) => formatCompactCurrency(v)}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={56}
            domain={['auto', 'auto']}
          />

          {/* Hidden right axis: drawdown (negative %), drawn below the baseline. */}
          {showDrawdown && (
            <YAxis
              yAxisId="drawdown"
              orientation="right"
              hide
              domain={[ddDomainMin, 0]}
            />
          )}

          <Tooltip
            cursor={{ stroke: tokens.muted, strokeDasharray: '3 3' }}
            content={({ active, payload, label }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0]?.payload as Row | undefined;
              if (!row) return null;
              const delta = row.bot - row.benchmark;
              const rowsOut: TooltipRow[] = [
                { label: 'Bot (paper)', value: formatCurrency(row.bot), color: tokens.primary },
                { label: 'Buy & Hold', value: formatCurrency(row.benchmark), color: tokens.muted },
                {
                  label: 'vs benchmark',
                  value: `${delta >= 0 ? '+' : '−'}${formatCurrency(Math.abs(delta))}`,
                  color: delta >= 0 ? tokens.up : tokens.down,
                },
              ];
              if (showDrawdown) {
                rowsOut.push({
                  label: 'Drawdown',
                  value: formatPct(row.drawdown, { digits: 1 }),
                  color: tokens.down,
                });
              }
              return <ChartTooltip title={formatDate(Number(label))} rows={rowsOut} />;
            }}
          />

          <Legend
            verticalAlign="top"
            height={24}
            iconType="plainline"
            wrapperStyle={{ fontSize: 11, color: tokens.muted }}
          />

          {/* Drawdown shading sits behind the value curves. */}
          {showDrawdown && (
            <Area
              yAxisId="drawdown"
              type="monotone"
              name="Drawdown"
              dataKey="drawdown"
              stroke="none"
              fill="url(#bot-drawdown-fill)"
              isAnimationActive={false}
              dot={false}
              legendType="none"
            />
          )}

          {showDrawdown && (
            <ReferenceLine yAxisId="drawdown" y={0} stroke={tokens.grid} strokeDasharray="2 2" />
          )}

          <Area
            yAxisId="value"
            type="monotone"
            name="Bot (paper)"
            dataKey="bot"
            stroke={tokens.primary}
            strokeWidth={2}
            fill="url(#bot-equity-fill)"
            isAnimationActive={false}
            dot={false}
          />

          <Line
            yAxisId="value"
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
