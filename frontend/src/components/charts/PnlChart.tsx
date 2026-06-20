/**
 * PnlChart — portfolio value over time. Plots the total value as a filled area
 * and (optionally) each position's value as faint overlay lines, merged on a
 * shared time axis. `live` highlights the most recent point with a pulsing marker
 * (the page feeds a history whose last point reflects live store prices).
 * Recharts, themed via tokens.
 */

import { useMemo } from 'react';
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useChartTokens, paletteColor } from '@/theme/tokens';
import type { PortfolioHistory } from '@/lib/types';
import { formatCompactCurrency, formatCurrency, formatDateTime } from '@/lib/format';
import { ChartTooltip, type TooltipRow } from './ChartTooltip';
import { ChartEmpty } from './ChartEmpty';

export interface PnlChartProps {
  history: PortfolioHistory;
  /** Emphasize the latest (live) point with a pulsing marker. */
  live?: boolean;
  /** Overlay per-position value lines (default: true). */
  showPositions?: boolean;
  height?: number;
  className?: string;
}

interface MergedRow {
  t: number;
  total: number;
  [symbol: string]: number;
}

export function PnlChart({
  history,
  live = false,
  showPositions = true,
  height = 300,
  className,
}: PnlChartProps): JSX.Element {
  const tokens = useChartTokens();

  const { rows, symbols } = useMemo(() => {
    const byTime = new Map<number, MergedRow>();
    for (const p of history.total) {
      byTime.set(p.t, { t: p.t, total: p.totalValue });
    }
    const syms: string[] = [];
    if (showPositions) {
      for (const pos of history.positions) {
        syms.push(pos.symbol);
        for (const pt of pos.points) {
          const row = byTime.get(pt.t);
          if (row) row[pos.symbol] = pt.value;
        }
      }
    }
    const merged = Array.from(byTime.values()).sort((a, b) => a.t - b.t);
    return { rows: merged, symbols: syms };
  }, [history, showPositions]);

  if (rows.length === 0) {
    return <ChartEmpty height={height} label="No portfolio history yet" className={className} />;
  }

  const lastT = rows[rows.length - 1]?.t;

  return (
    <div className={className} style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: -6 }}>
          <defs>
            <linearGradient id="pnl-total" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={tokens.primary} stopOpacity={0.26} />
              <stop offset="100%" stopColor={tokens.primary} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={tokens.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="t"
            tickFormatter={(v: number) => formatDateTime(v)}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: tokens.grid }}
            minTickGap={56}
          />
          <YAxis
            tickFormatter={(v: number) => formatCompactCurrency(v)}
            tick={{ fill: tokens.muted, fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={56}
            domain={['auto', 'auto']}
          />
          <Tooltip
            cursor={{ stroke: tokens.muted, strokeDasharray: '3 3' }}
            content={({ active, payload, label }) => {
              if (!active || !payload || payload.length === 0) return null;
              const row = payload[0]?.payload as MergedRow | undefined;
              if (!row) return null;
              const out: TooltipRow[] = [
                { label: 'Total', value: formatCurrency(row.total), color: tokens.primary },
              ];
              symbols.forEach((s, i) => {
                const v = row[s];
                if (typeof v === 'number') {
                  out.push({ label: s, value: formatCurrency(v), color: paletteColor(tokens, i) });
                }
              });
              return <ChartTooltip title={formatDateTime(Number(label))} rows={out} />;
            }}
          />
          <Area
            type="monotone"
            name="Total"
            dataKey="total"
            stroke={tokens.primary}
            strokeWidth={2}
            fill="url(#pnl-total)"
            isAnimationActive={false}
            dot={false}
            activeDot={
              live
                ? { r: 4, fill: tokens.primary, stroke: tokens.surface, strokeWidth: 2 }
                : { r: 3 }
            }
          />
          {showPositions &&
            symbols.map((s, i) => (
              <Line
                key={s}
                type="monotone"
                name={s}
                dataKey={s}
                stroke={paletteColor(tokens, i)}
                strokeWidth={1.25}
                strokeOpacity={0.55}
                dot={false}
                isAnimationActive={false}
                connectNulls
              />
            ))}
          {/* Live marker reference: a tiny dot at the latest point. */}
          {live && typeof lastT === 'number' && (
            <Line
              type="monotone"
              dataKey="total"
              data={[rows[rows.length - 1] as MergedRow]}
              stroke="none"
              dot={{ r: 4, fill: tokens.primary, stroke: tokens.surface, strokeWidth: 2 }}
              isAnimationActive={false}
              legendType="none"
              tooltipType="none"
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
