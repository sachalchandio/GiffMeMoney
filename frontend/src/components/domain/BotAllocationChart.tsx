/**
 * BotAllocationChart — how the simulated bot spread its paper capital across
 * sleeves over the run, derived from the recorded trades: for each sleeve
 * (symbol) we net buys minus sells to get the capital it deployed, then show the
 * share as a donut with a compact weighted legend. This makes the momentum /
 * bandit tilt visible — winners that the bot kept adding to occupy bigger slices.
 *
 * Simulated paper-trading only. Recharts + theme tokens (no hardcoded hex).
 */

import { useMemo } from 'react';
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from 'recharts';
import { useChartTokens, paletteColor } from '@/theme/tokens';
import type { BotTrade } from '@/lib/types';
import { formatCurrency, formatPct } from '@/lib/format';
import { ChartTooltip } from '@/components/charts/ChartTooltip';
import { ChartEmpty } from '@/components/charts/ChartEmpty';
import { cn } from '@/lib/utils';

export interface BotAllocationChartProps {
  trades: BotTrade[];
  height?: number;
  /** Show the legend column beside the donut (default: true). */
  showLegend?: boolean;
  /** Collapse sleeves beyond this many into an "Other" slice (default: 8). */
  maxSlices?: number;
  className?: string;
}

interface Slice {
  symbol: string;
  value: number;
  weight: number;
  color: string;
}

export function BotAllocationChart({
  trades,
  height = 260,
  showLegend = true,
  maxSlices = 8,
  className,
}: BotAllocationChartProps): JSX.Element {
  const tokens = useChartTokens();

  const { slices, total } = useMemo(() => {
    // Net capital deployed per symbol = buys − sells, floored at 0 (a fully
    // exited sleeve contributes no standing allocation).
    const net = new Map<string, number>();
    for (const tr of trades) {
      const signed = tr.side === 'buy' ? tr.amount : -tr.amount;
      net.set(tr.symbol, (net.get(tr.symbol) ?? 0) + signed);
    }
    const deployed = Array.from(net.entries())
      .map(([symbol, v]) => ({ symbol, value: Math.max(0, v) }))
      .filter((d) => d.value > 0)
      .sort((a, b) => b.value - a.value);

    const totalValue = deployed.reduce((acc, d) => acc + d.value, 0);

    let head = deployed;
    let tailValue = 0;
    if (deployed.length > maxSlices) {
      head = deployed.slice(0, maxSlices - 1);
      tailValue = deployed.slice(maxSlices - 1).reduce((acc, d) => acc + d.value, 0);
    }

    const out: Slice[] = head.map((d, i) => ({
      symbol: d.symbol,
      value: d.value,
      weight: totalValue > 0 ? d.value / totalValue : 0,
      color: paletteColor(tokens, i),
    }));
    if (tailValue > 0) {
      out.push({
        symbol: 'Other',
        value: tailValue,
        weight: totalValue > 0 ? tailValue / totalValue : 0,
        color: tokens.muted,
      });
    }
    return { slices: out, total: totalValue };
  }, [trades, tokens, maxSlices]);

  if (slices.length === 0) {
    return <ChartEmpty height={height} label="No sleeves allocated yet" className={className} />;
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
                      { label: 'Deployed', value: formatCurrency(s.value), color: s.color },
                      { label: 'Weight', value: formatPct(s.weight * 100, { digits: 1 }) },
                    ]}
                  />
                );
              }}
            />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-[0.625rem] font-medium uppercase tracking-wide text-muted">Deployed</span>
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
