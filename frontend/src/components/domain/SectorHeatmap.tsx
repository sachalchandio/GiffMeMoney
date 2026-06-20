/**
 * SectorHeatmap — a compact, responsive grid of sector tiles from the market
 * summary. Each tile is tinted by its `changePct` (green → red, intensity scaled
 * to the day's spread) and labelled with the sector, its change, and the number
 * of constituents. Color is applied via inline `color-mix` on the `--success` /
 * `--danger` tokens, so it stays theme-correct without hardcoded hex.
 */

import { useMemo } from 'react';
import type { SectorPerf } from '@/lib/types';
import { cn } from '@/lib/utils';
import { formatPct } from '@/lib/format';

export interface SectorHeatmapProps {
  sectors: SectorPerf[];
  className?: string;
}

/** Background tint for a tile: stronger color the further from flat. */
function tileStyle(changePct: number, maxAbs: number): React.CSSProperties {
  const intensity = maxAbs > 0 ? Math.min(1, Math.abs(changePct) / maxAbs) : 0;
  // 12%..58% mix so even small moves are legible but never overwhelming.
  const pct = (12 + intensity * 46).toFixed(0);
  const base = changePct >= 0 ? 'var(--success)' : 'var(--danger)';
  return { backgroundColor: `color-mix(in srgb, ${base} ${pct}%, var(--surface))` };
}

export function SectorHeatmap({ sectors, className }: SectorHeatmapProps): JSX.Element {
  const ordered = useMemo(
    () => [...sectors].sort((a, b) => b.changePct - a.changePct),
    [sectors],
  );
  const maxAbs = useMemo(
    () => ordered.reduce((m, s) => Math.max(m, Math.abs(s.changePct)), 0),
    [ordered],
  );

  if (ordered.length === 0) {
    return (
      <div
        className={cn(
          'flex h-40 items-center justify-center rounded-xl border border-dashed border-border text-sm text-muted',
          className,
        )}
      >
        No sector data
      </div>
    );
  }

  return (
    <div
      className={cn(
        'grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4',
        className,
      )}
    >
      {ordered.map((s) => {
        const up = s.changePct >= 0;
        return (
          <div
            key={s.sector}
            style={tileStyle(s.changePct, maxAbs)}
            className="flex min-h-[72px] flex-col justify-between rounded-xl border border-border/60 p-2.5"
          >
            <div className="flex items-start justify-between gap-1">
              <span className="truncate text-xs font-semibold tracking-tight text-text" title={s.sector}>
                {s.sector}
              </span>
              <span className="shrink-0 rounded-full bg-surface/70 px-1.5 text-[10px] font-medium tnum text-muted">
                {s.count}
              </span>
            </div>
            <span
              className={cn(
                'text-base font-semibold tnum',
                up ? 'text-success' : 'text-danger',
              )}
            >
              {formatPct(s.changePct, { digits: 2, sign: true })}
            </span>
          </div>
        );
      })}
    </div>
  );
}
