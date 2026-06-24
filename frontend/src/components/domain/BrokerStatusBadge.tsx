/**
 * BrokerStatusBadge — a compact, DISPLAY-ONLY indicator of (1) whether the
 * market data feeding the app is **Simulated** or **Live**, and (2) whether the
 * broker is in **Paper** (sandbox) or **LIVE** (real-money) mode. Reads the mode
 * from `/api/broker/status` ({@link useBrokerStatus}).
 *
 * SAFETY / HONESTY (this is a finance tool):
 *  - Paper / simulated reads as a calm, neutral/green chip.
 *  - LIVE reads as a loud RED WARNING chip (with an alert icon) — a real-money
 *    mode must never look like the safe default.
 *  - This component is DISPLAY-ONLY. There is deliberately NO "go live" / "enable
 *    real trading" control anywhere; switching to live is a documented, deliberate
 *    env/config action (see docs/DEPLOY.md). The badge only reflects state.
 *
 * The "Data" axis is derived from the broker mode: the simulated broker fills at
 * synthetic prices (Data: Simulated); the Alpaca adapter (paper/live) is wired to
 * the real market data feed (Data: Live). Semantic tokens only; light/dark aware.
 */

import { Activity, AlertTriangle, ShieldCheck } from 'lucide-react';
import { Badge, type BadgeSize } from '@/components/ui/Badge';
import { Tooltip } from '@/components/ui/Tooltip';
import { useBrokerStatus } from '@/hooks/useBroker';
import type { BrokerStatus } from '@/lib/types';
import { cn } from '@/lib/utils';

export interface BrokerStatusBadgeProps {
  size?: BadgeSize;
  /** Hide the "Data:" / "Broker:" prefixes (tighter on very narrow widths). */
  compact?: boolean;
  className?: string;
}

/** Whether the market data feeding the app is live (vs synthetic/simulated). */
function isLiveData(status: BrokerStatus): boolean {
  // The simulated broker fills at synthetic prices; any other mode is wired to
  // the real market-data feed.
  return status.mode !== 'simulated';
}

/** Whether the broker is placing REAL-MONEY (live) orders. */
function isLiveBroker(status: BrokerStatus): boolean {
  return status.liveEnabled && !status.paper;
}

export function BrokerStatusBadge({
  size = 'sm',
  compact = false,
  className,
}: BrokerStatusBadgeProps): JSX.Element | null {
  const { data, isPending, isError } = useBrokerStatus();

  // Fail safe + quiet: while loading or if the broker status can't be read, show
  // nothing rather than implying a (mis)state. The page-level panel surfaces any
  // error in full; the TopBar badge stays unobtrusive.
  if (isPending || isError || !data) return null;

  const liveData = isLiveData(data);
  const liveBroker = isLiveBroker(data);

  const dataLabel = `${compact ? '' : 'Data: '}${liveData ? 'Live' : 'Simulated'}`;
  const brokerLabel = `${compact ? '' : 'Broker: '}${liveBroker ? 'LIVE' : 'Paper'}`;

  const dataTip = liveData
    ? 'Live market data feed (read-only).'
    : 'Simulated market data — synthetic prices, no real feed.';
  const brokerTip = liveBroker
    ? 'LIVE broker — real orders can be placed against a real-money account.'
    : data.broker === 'alpaca'
      ? 'Paper broker (Alpaca PAPER sandbox) — no real money moves.'
      : 'Simulated paper broker — fills at the market price; no real money moves.';

  return (
    <span
      className={cn('inline-flex items-center gap-1.5', className)}
      aria-label={`${dataLabel}. ${brokerLabel}.`}
    >
      {/* Data: Simulated | Live */}
      <Tooltip content={<span>{dataTip}</span>}>
        <Badge
          tone={liveData ? 'primary' : 'neutral'}
          variant="soft"
          size={size}
          icon={<Activity className="h-3 w-3" aria-hidden />}
        >
          {dataLabel}
        </Badge>
      </Tooltip>

      {/* Broker: Paper (safe) | LIVE (red warning) */}
      <Tooltip content={<span>{brokerTip}</span>}>
        <Badge
          tone={liveBroker ? 'danger' : 'success'}
          variant={liveBroker ? 'solid' : 'soft'}
          size={size}
          icon={
            liveBroker ? (
              <AlertTriangle className="h-3 w-3" aria-hidden />
            ) : (
              <ShieldCheck className="h-3 w-3" aria-hidden />
            )
          }
          className={liveBroker ? 'animate-pulse' : undefined}
        >
          {brokerLabel}
        </Badge>
      </Tooltip>
    </span>
  );
}
