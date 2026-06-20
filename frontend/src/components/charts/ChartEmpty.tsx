/**
 * ChartEmpty — a consistent graceful empty state used by every chart when there
 * is no data to render. Dashed, muted, fills the chart's height.
 */

import { BarChart3 } from 'lucide-react';
import { cn } from '@/lib/utils';

export interface ChartEmptyProps {
  height?: number;
  label?: string;
  className?: string;
}

export function ChartEmpty({
  height = 280,
  label = 'No data to display',
  className,
}: ChartEmptyProps): JSX.Element {
  return (
    <div
      className={cn(
        'flex w-full flex-col items-center justify-center gap-2 rounded-xl border border-dashed border-border text-muted',
        className,
      )}
      style={{ height }}
    >
      <BarChart3 className="h-5 w-5 opacity-60" aria-hidden />
      <span className="text-sm">{label}</span>
    </div>
  );
}
