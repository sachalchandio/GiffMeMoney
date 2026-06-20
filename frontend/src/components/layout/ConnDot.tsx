/**
 * ConnDot — a live connection indicator driven by the market WebSocket status in
 * the zustand store. Green pulse when connected, amber while (re)connecting, red
 * when disconnected. Optional text label. Colors via semantic tokens.
 */

import { useConnStatus, type ConnStatus } from '@/store/marketStore';
import { cn } from '@/lib/utils';

const META: Record<ConnStatus, { dot: string; ring: string; label: string; pulse: boolean }> = {
  connected: { dot: 'bg-success', ring: 'bg-success/30', label: 'Live', pulse: true },
  connecting: { dot: 'bg-warning', ring: 'bg-warning/30', label: 'Connecting', pulse: true },
  reconnecting: { dot: 'bg-warning', ring: 'bg-warning/30', label: 'Reconnecting', pulse: true },
  disconnected: { dot: 'bg-danger', ring: 'bg-danger/30', label: 'Offline', pulse: false },
};

export interface ConnDotProps {
  /** Show the status text next to the dot. */
  showLabel?: boolean;
  className?: string;
}

export function ConnDot({ showLabel = true, className }: ConnDotProps): JSX.Element {
  const status = useConnStatus();
  const meta = META[status];

  return (
    <span
      className={cn('inline-flex items-center gap-1.5', className)}
      title={meta.label}
      aria-live="polite"
      aria-label={`Market feed ${meta.label}`}
    >
      <span className="relative inline-flex h-2 w-2">
        {meta.pulse && (
          <span
            className={cn('absolute inline-flex h-full w-full animate-ping rounded-full opacity-75', meta.ring)}
          />
        )}
        <span className={cn('relative inline-flex h-2 w-2 rounded-full', meta.dot)} />
      </span>
      {showLabel && <span className="text-xs font-medium text-muted">{meta.label}</span>}
    </span>
  );
}
