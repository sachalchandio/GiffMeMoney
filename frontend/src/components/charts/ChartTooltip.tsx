/**
 * ChartTooltip — a small themed tooltip surface shared by the recharts charts so
 * every chart's hover card matches the design tokens. Pure presentational; charts
 * pass a title + rows.
 */

import type { ReactNode } from 'react';

export interface TooltipRow {
  label: string;
  value: ReactNode;
  /** Swatch color (token string). Omit for no dot. */
  color?: string;
}

export interface ChartTooltipProps {
  title?: string;
  rows: TooltipRow[];
}

export function ChartTooltip({ title, rows }: ChartTooltipProps): JSX.Element {
  return (
    <div className="rounded-xl border border-border bg-surface px-3 py-2 shadow-pop">
      {title && <div className="mb-1 text-xs font-semibold text-text">{title}</div>}
      <div className="flex flex-col gap-1">
        {rows.map((r, i) => (
          <div key={i} className="flex items-center justify-between gap-4 text-xs">
            <span className="flex items-center gap-1.5 text-muted">
              {r.color && (
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ backgroundColor: r.color }}
                  aria-hidden
                />
              )}
              {r.label}
            </span>
            <span className="tnum font-medium text-text">{r.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
