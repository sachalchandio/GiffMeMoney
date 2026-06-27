/**
 * BrokerPanel — a DISPLAY-ONLY view of the go-live broker layer: the broker mode
 * (simulated / paper / LIVE), the account summary (cash / equity / buying power),
 * open positions, and recorded orders. Reads `/api/broker/{status,account,
 * positions,orders}` via the broker hooks.
 *
 * SAFETY / HONESTY (this is a finance tool):
 *  - A PROMINENT disclaimer makes the paper / simulated nature explicit; if the
 *    broker is ever in LIVE mode the disclaimer turns into a loud red warning.
 *  - This panel is READ-ONLY. There is deliberately NO order form and NO "go
 *    live" / "enable real trading" control — switching to live is a documented,
 *    deliberate env/config action (see docs/DEPLOY.md), never a UI button.
 *
 * Semantic tokens only; responsive; light/dark aware; strict TS (no `any`).
 */

import { useMemo } from 'react';
import {
  AlertTriangle,
  Banknote,
  Building2,
  ListOrdered,
  ShieldCheck,
  Wallet as WalletIcon,
} from 'lucide-react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge, type BadgeTone } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Skeleton } from '@/components/ui/Skeleton';
import { StatCard } from '@/components/ui/StatCard';
import { BrokerStatusBadge } from './BrokerStatusBadge';
import { useBrokerStatus, useBrokerAccount, useBrokerPositions, useBrokerOrders } from '@/hooks/useBroker';
import type {
  BrokerMode,
  BrokerOrder,
  BrokerOrderStatus,
  BrokerPosition,
  BrokerStatus,
} from '@/lib/types';
import { BROKER_DISCLAIMER } from '@/lib/types';
import { formatCurrency, formatDateTime, formatNumber, formatPct, formatPrice } from '@/lib/format';
import { cn, changeTextColor } from '@/lib/utils';

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/** Whether the broker is placing REAL-MONEY (live) orders. */
function isLive(status: BrokerStatus | undefined): boolean {
  return Boolean(status && status.liveEnabled && !status.paper);
}

/** Human label for a broker execution mode. */
function modeLabel(mode: BrokerMode): string {
  switch (mode) {
    case 'live':
      return 'LIVE';
    case 'paper':
      return 'Paper';
    case 'simulated':
    default:
      return 'Simulated';
  }
}

/** Badge tone for an order's lifecycle status. */
function orderStatusTone(status: BrokerOrderStatus): BadgeTone {
  switch (status) {
    case 'filled':
      return 'success';
    case 'partially_filled':
    case 'accepted':
    case 'pending':
      return 'warning';
    case 'rejected':
    case 'canceled':
      return 'danger';
    default:
      return 'neutral';
  }
}

/** Human label for an order status (e.g. `partially_filled` → `Partially filled`). */
function orderStatusLabel(status: BrokerOrderStatus): string {
  const s = status.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/* ------------------------------------------------------------------ */
/* Sub-components                                                      */
/* ------------------------------------------------------------------ */

/** One open position row, marked to the latest price. */
function PositionRow({ pos }: { pos: BrokerPosition }): JSX.Element {
  return (
    <li className="flex items-center gap-3 rounded-xl border border-border bg-surface px-3 py-2">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-semibold tracking-tight text-text">{pos.symbol}</span>
          {pos.paper && (
            <Badge tone="neutral" variant="outline" size="sm">
              paper
            </Badge>
          )}
        </div>
        <span className="text-[0.6875rem] text-muted">
          {formatNumber(pos.qty, 4)} @ {formatPrice(pos.avgEntryPrice)} · now {formatPrice(pos.currentPrice)}
        </span>
      </div>
      <div className="shrink-0 text-right">
        <div className="text-sm font-medium tnum text-text">{formatCurrency(pos.marketValue)}</div>
        <div className={cn('text-[0.6875rem] tnum', changeTextColor(pos.unrealizedPnl))}>
          {pos.unrealizedPnl >= 0 ? '+' : '−'}
          {formatCurrency(Math.abs(pos.unrealizedPnl))} ({formatPct(pos.unrealizedPnlPct, { sign: true, digits: 1 })})
        </div>
      </div>
    </li>
  );
}

/** One recorded order row. */
function OrderRow({ order }: { order: BrokerOrder }): JSX.Element {
  const sizing =
    order.notional != null
      ? formatCurrency(order.notional)
      : order.qty != null
        ? `${formatNumber(order.qty, 4)} units`
        : '—';
  return (
    <li className="flex items-center gap-3 rounded-xl border border-border bg-surface px-3 py-2">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span
            className={cn(
              'text-xs font-semibold uppercase tracking-wide',
              order.side === 'buy' ? 'text-success' : 'text-danger',
            )}
          >
            {order.side}
          </span>
          <span className="truncate text-sm font-semibold tracking-tight text-text">{order.symbol}</span>
          <Badge tone={orderStatusTone(order.status)} variant="soft" size="sm">
            {orderStatusLabel(order.status)}
          </Badge>
        </div>
        <span className="text-[0.6875rem] text-muted">
          {sizing}
          {order.filledQty > 0 && ` · filled ${formatNumber(order.filledQty, 4)} @ ${formatPrice(order.filledAvgPrice)}`}
        </span>
      </div>
      <span className="shrink-0 text-[0.6875rem] text-muted">{formatDateTime(order.createdAt)}</span>
    </li>
  );
}

/** A small bordered section with a title, used for positions / orders. */
function Section({
  title,
  icon,
  count,
  children,
}: {
  title: string;
  icon: JSX.Element;
  count?: number;
  children: React.ReactNode;
}): JSX.Element {
  return (
    <Card className="flex flex-col gap-3">
      <CardHeader>
        <CardTitle icon={icon}>{title}</CardTitle>
        {typeof count === 'number' && count > 0 && (
          <span className="text-[0.6875rem] text-muted">{count}</span>
        )}
      </CardHeader>
      {children}
    </Card>
  );
}

/* ------------------------------------------------------------------ */
/* Panel                                                               */
/* ------------------------------------------------------------------ */

export interface BrokerPanelProps {
  className?: string;
}

export function BrokerPanel({ className }: BrokerPanelProps): JSX.Element {
  const statusQuery = useBrokerStatus();
  const status = statusQuery.data;
  // Only fetch account/positions/orders once the broker is reachable.
  const ready = Boolean(status?.connected);
  const accountQuery = useBrokerAccount(ready);
  const positionsQuery = useBrokerPositions(ready);
  const ordersQuery = useBrokerOrders(ready);

  const live = isLive(status);
  const positions = useMemo<BrokerPosition[]>(() => positionsQuery.data ?? [], [positionsQuery.data]);
  const orders = useMemo<BrokerOrder[]>(() => ordersQuery.data ?? [], [ordersQuery.data]);

  return (
    <div className={cn('flex flex-col gap-4', className)}>
      {/* Heading + at-a-glance badge */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="flex items-center gap-2 text-sm font-semibold tracking-tight text-text">
          <Building2 className="h-4 w-4 text-primary" aria-hidden />
          Broker
        </h2>
        <BrokerStatusBadge />
      </div>

      {/* PROMINENT paper / sim (or LIVE warning) disclaimer */}
      <Card
        role="note"
        className={cn(
          'flex items-start gap-3',
          live ? 'border-danger/50 bg-danger/8' : 'border-warning/40 bg-warning/8',
        )}
      >
        {live ? (
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-danger" aria-hidden />
        ) : (
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-warning" aria-hidden />
        )}
        <div className="flex flex-col gap-0.5">
          <p className={cn('text-sm font-semibold', live ? 'text-danger' : 'text-text')}>
            {live
              ? 'LIVE trading is enabled — real orders place real money at risk'
              : 'Paper / simulated broker — no real money moves'}
          </p>
          <p className="text-xs leading-relaxed text-muted">{BROKER_DISCLAIMER}</p>
          <p className="text-[0.6875rem] leading-relaxed text-muted">
            This view is display-only. Real trading is never enabled from the UI — it stays a deliberate,
            documented env/config action (see docs/DEPLOY.md).
          </p>
        </div>
      </Card>

      {/* Status / connectivity strip */}
      {statusQuery.isPending ? (
        <Skeleton className="h-16 w-full" />
      ) : statusQuery.isError ? (
        <Card className="flex items-center justify-between gap-3 border-danger/30 bg-danger/8">
          <span className="text-xs text-danger">Couldn&apos;t reach the broker status endpoint.</span>
          <Button variant="secondary" size="sm" onClick={() => void statusQuery.refetch()}>
            Retry
          </Button>
        </Card>
      ) : status ? (
        <Card className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="neutral" variant="outline" size="sm">
              Backend: {status.broker}
            </Badge>
            <Badge tone={live ? 'danger' : 'success'} variant="soft" size="sm">
              Mode: {modeLabel(status.mode)}
            </Badge>
            <Badge tone={status.connected ? 'success' : 'danger'} variant="soft" size="sm">
              {status.connected ? 'Connected' : 'Not connected'}
            </Badge>
            <Badge tone={status.liveEnabled ? 'danger' : 'neutral'} variant="soft" size="sm">
              Live gate: {status.liveEnabled ? 'OPEN' : 'closed'}
            </Badge>
          </div>
          {status.message && <p className="text-[0.6875rem] leading-snug text-muted">{status.message}</p>}
        </Card>
      ) : null}

      {/* Account summary */}
      {ready && (
        <>
          {accountQuery.isPending ? (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <Skeleton key={i} className="h-24 w-full" />
              ))}
            </div>
          ) : accountQuery.isError ? (
            <Card className="flex items-center justify-between gap-3 border-danger/30 bg-danger/8">
              <span className="text-xs text-danger">Couldn&apos;t load the broker account.</span>
              <Button variant="secondary" size="sm" onClick={() => void accountQuery.refetch()}>
                Retry
              </Button>
            </Card>
          ) : accountQuery.data ? (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <StatCard
                label="Cash"
                value={formatCurrency(accountQuery.data.cash)}
                icon={<Banknote className="h-4 w-4" />}
                hint={`${accountQuery.data.currency} · ${modeLabel(accountQuery.data.mode)}`}
              />
              <StatCard
                label="Equity"
                value={formatCurrency(accountQuery.data.equity)}
                icon={<WalletIcon className="h-4 w-4" />}
                hint={accountQuery.data.paper ? 'Paper account' : 'Live account'}
              />
              <StatCard
                label="Buying power"
                value={formatCurrency(accountQuery.data.buyingPower)}
                icon={<Building2 className="h-4 w-4" />}
                hint={`#${accountQuery.data.accountId}`}
              />
            </div>
          ) : null}

          {/* Positions + orders */}
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2 lg:gap-4">
            <Section title="Positions" icon={<WalletIcon className="h-4 w-4" />} count={positions.length}>
              {positionsQuery.isPending ? (
                <div className="flex flex-col gap-2">
                  {Array.from({ length: 2 }).map((_, i) => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : positionsQuery.isError ? (
                <div className="flex items-center justify-between gap-3 rounded-xl border border-danger/30 bg-danger/8 px-3 py-2">
                  <span className="text-xs text-danger">Couldn&apos;t load positions.</span>
                  <Button variant="secondary" size="sm" onClick={() => void positionsQuery.refetch()}>
                    Retry
                  </Button>
                </div>
              ) : positions.length === 0 ? (
                <div className="rounded-xl border border-dashed border-border py-8 text-center text-xs text-muted">
                  No open positions.
                </div>
              ) : (
                <ul className="flex max-h-[20rem] flex-col gap-2 overflow-y-auto pr-0.5">
                  {positions.map((pos) => (
                    <PositionRow key={pos.symbol} pos={pos} />
                  ))}
                </ul>
              )}
            </Section>

            <Section title="Orders" icon={<ListOrdered className="h-4 w-4" />} count={orders.length}>
              {ordersQuery.isPending ? (
                <div className="flex flex-col gap-2">
                  {Array.from({ length: 2 }).map((_, i) => (
                    <Skeleton key={i} className="h-12 w-full" />
                  ))}
                </div>
              ) : ordersQuery.isError ? (
                <div className="flex items-center justify-between gap-3 rounded-xl border border-danger/30 bg-danger/8 px-3 py-2">
                  <span className="text-xs text-danger">Couldn&apos;t load orders.</span>
                  <Button variant="secondary" size="sm" onClick={() => void ordersQuery.refetch()}>
                    Retry
                  </Button>
                </div>
              ) : orders.length === 0 ? (
                <div className="rounded-xl border border-dashed border-border py-8 text-center text-xs text-muted">
                  No orders recorded yet.
                </div>
              ) : (
                <ul className="flex max-h-[20rem] flex-col gap-2 overflow-y-auto pr-0.5">
                  {orders.map((order) => (
                    <OrderRow key={order.id} order={order} />
                  ))}
                </ul>
              )}
            </Section>
          </div>
        </>
      )}
    </div>
  );
}
