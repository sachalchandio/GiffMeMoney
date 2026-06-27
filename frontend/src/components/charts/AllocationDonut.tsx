/**
 * AllocationDonut — portfolio allocation by market value as a donut. Slices are
 * colored from the categorical palette; the center shows the total value. A
 * compact legend lists each holding with its weight. Recharts, themed via tokens.
 */

import { useMemo } from 'react';
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';
import { useChartTokens, paletteColor } from '@/theme/tokens';
import type { Position } from '@/lib/types';
import { formatCurrency, formatPct } from '@/lib/format';
import { ChartTooltip } from './ChartTooltip';
import { ChartEmpty } from './ChartEmpty';
import { cn } from '@/lib/utils';

export interface AllocationDonutProps {
  positions: Position[];
  height?: number;
  /** Show the legend column beside the donut (default: true). */
  showLegend?: boolean;
  className?: string;
}

interface Slice {
  symbol: string;
  value: number;
  weight: number;
  color: string;
}

export function AllocationDonut({
  positions,
  height = 260,
  showLegend = true,
  className,
}: AllocationDonutProps): JSX.Element {
  const tokens = useChartTokens();

  const { slices, total } = useMemo(() => {
    const totalValue = positions.reduce((acc, p) => acc + p.marketValue, 0);
    const out: Slice[] = positions
      .filter((p) => p.marketValue > 0)
      .sort((a, b) => b.marketValue - a.marketValue)
      .map((p, i) => ({
        symbol: p.symbol,
        value: p.marketValue,
        weight: totalValue > 0 ? p.marketValue / totalValue : 0,
        color: paletteColor(tokens, i),
      }));
    return { slices: out, total: totalValue };
  }, [positions, tokens]);

  if (slices.length === 0) {
    return <ChartEmpty height={height} label="No holdings yet" className={className} />;
  }

  return (
    <div className={cn('flex flex-col items-center gap-3 sm:flex-row', className)}>
      <div className="relative" style={{ width: height, height, minWidth: height }}>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={slices}
              dataKey="value"
              nameKey="symbol"
              innerRadius="62%"
              outerRadius="92%"
              paddingAngle={1.5}
              stroke="none"
              isAnimationActive={false}
            >
              {slices.map((s) => (
                <Cell key={s.symbol} fill={s.color} />
              ))}
            </Pie>
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || payload.length === 0) return null;
                const s = payload[0]?.payload as Slice | undefined;
                if (!s) return null;
                return (
                  <ChartTooltip
                    title={s.symbol}
                    rows={[
                      { label: 'Value', value: formatCurrency(s.value), color: s.color },
                      { label: 'Weight', value: formatPct(s.weight * 100, { digits: 1 }) },
                    ]}
                  />
                );
              }}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-[0.625rem] font-medium uppercase tracking-wide text-muted">Invested</span>
          <span className="text-lg font-semibold tnum text-text">{formatCurrency(total)}</span>
        </div>
      </div>

      {showLegend && (
        <ul className="grid w-full grid-cols-1 gap-1.5 sm:max-w-[12rem]">
          {slices.map((s) => (
            <li key={s.symbol} className="flex items-center justify-between gap-2 text-xs">
              <span className="flex items-center gap-1.5 truncate text-text">
                <span
                  className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm"
                  style={{ backgroundColor: s.color }}
                  aria-hidden
                />
                <span className="truncate font-medium">{s.symbol}</span>
              </span>
              <span className="tnum text-muted">{formatPct(s.weight * 100, { digits: 1 })}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
