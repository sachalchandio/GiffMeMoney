/**
 * InvestPage — the flagship simulated-brokerage experience.
 *
 * Wires the wallet header (live total / P&L from the market store), the Add Funds
 * and Withdraw modals, the allocation builder ("Suggest for me" → advisor), the
 * live P&L + allocation charts, the open positions (each with a Sell control), the
 * "where to invest now" advisor panel, and the transaction ledger. Data comes from
 * useWallet / usePortfolioState / usePortfolioHistory / useAdvisor; live P&L is
 * derived from the streaming store so figures move every tick. Dense, responsive,
 * light/dark via tokens.
 */

import { useMemo, useState } from 'react';
import { AlertTriangle, LineChart, PieChart, Wallet as WalletIcon } from 'lucide-react';
import { WalletHeader } from '@/components/domain/WalletHeader';
import { AddFundsModal } from '@/components/domain/AddFundsModal';
import { WithdrawModal } from '@/components/domain/WithdrawModal';
import { AllocationBuilder, type AllocationRow } from '@/components/domain/AllocationBuilder';
import { PositionCard } from '@/components/domain/PositionCard';
import { TransactionList } from '@/components/domain/TransactionList';
import { AdvicePanel } from '@/components/domain/AdvicePanel';
import { RiskControls } from '@/components/domain/RiskControls';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Skeleton } from '@/components/ui/Skeleton';
import { PnlChart } from '@/components/charts/PnlChart';
import { AllocationDonut } from '@/components/charts/AllocationDonut';
import { useWallet, useTransactions } from '@/hooks/useWallet';
import { usePortfolioState } from '@/hooks/usePortfolioState';
import { usePortfolioHistory } from '@/hooks/usePortfolioHistory';
import { useMarketStore } from '@/store/marketStore';
import type { AllocationAdvice, PortfolioHistory, Position } from '@/lib/types';
import { round } from '@/lib/utils';

/**
 * Splice a live "now" point onto the history so the P&L area chart ends on the
 * current mark-to-market total (cash + Σ live position values).
 */
function useLiveHistory(
  history: PortfolioHistory | undefined,
  positions: Position[],
  cash: number,
): PortfolioHistory | undefined {
  const prices = useMarketStore((s) => s.prices);
  return useMemo(() => {
    if (!history) return undefined;
    let invested = 0;
    for (const p of positions) {
      const live = prices[p.symbol.toUpperCase()]?.price;
      invested += (typeof live === 'number' && Number.isFinite(live) ? live : p.currentPrice) * p.units;
    }
    const now = Date.now();
    const total = [
      ...history.total,
      { t: now, totalValue: round(cash + invested, 2), invested: round(invested, 2), cash: round(cash, 2) },
    ];
    const livePositions = history.positions.map((ps) => {
      const pos = positions.find((p) => p.symbol === ps.symbol);
      if (!pos) return ps;
      const live = prices[pos.symbol.toUpperCase()]?.price ?? pos.currentPrice;
      const value = round(live * pos.units, 2);
      const pnl = round(value - pos.costBasis, 2);
      const pnlPct = pos.costBasis > 0 ? round((pnl / pos.costBasis) * 100, 2) : 0;
      return { ...ps, points: [...ps.points, { t: now, value, pnl, pnlPct }] };
    });
    return { total, positions: livePositions };
  }, [history, positions, cash, prices]);
}

export default function InvestPage(): JSX.Element {
  const wallet = useWallet();
  const portfolio = usePortfolioState();
  const history = usePortfolioHistory(120);
  const transactions = useTransactions();

  const [addOpen, setAddOpen] = useState(false);
  const [withdrawOpen, setWithdrawOpen] = useState(false);
  const [advice, setAdvice] = useState<AllocationAdvice | null>(null);
  const [applied, setApplied] = useState<{ token: number; rows: AllocationRow[] } | null>(null);

  const positions = useMemo<Position[]>(() => portfolio.data?.positions ?? [], [portfolio.data]);
  const cash = wallet.data?.cashBalance ?? 0;
  const liveHistory = useLiveHistory(history.data, positions, cash);

  const applyAdvice = (a: AllocationAdvice): void => {
    setAdvice(a);
    setApplied({
      token: Date.now(),
      rows: a.items.map((it) => ({ symbol: it.asset.symbol, amount: round(it.amount, 2) })),
    });
    // Scroll the builder into view on small screens.
    document.getElementById('allocation-builder')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  };

  const loadFailed = wallet.isError || portfolio.isError;

  return (
    <div className="flex flex-col gap-4">
      {/* Page heading */}
      <div className="flex flex-col gap-1">
        <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-text">
          <WalletIcon className="h-5 w-5 text-primary" aria-hidden />
          Invest
        </h1>
        <p className="text-sm text-muted">
          Fund your sandbox wallet, split it across assets, and watch real-time P&amp;L. No real money moves.
        </p>
      </div>

      {loadFailed && (
        <Card className="flex items-center gap-3 border-danger/30">
          <AlertTriangle className="h-5 w-5 shrink-0 text-danger" aria-hidden />
          <div className="flex-1">
            <p className="text-sm font-medium text-text">Couldn&apos;t load your wallet</p>
            <p className="text-xs text-muted">
              {(wallet.error ?? portfolio.error) instanceof Error
                ? (wallet.error ?? portfolio.error)?.message
                : 'Please try again.'}
            </p>
          </div>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              void wallet.refetch();
              void portfolio.refetch();
            }}
          >
            Retry
          </Button>
        </Card>
      )}

      {/* Wallet hero */}
      <WalletHeader
        wallet={wallet.data}
        positions={positions}
        loading={wallet.isPending}
        onAddFunds={() => setAddOpen(true)}
        onWithdraw={() => setWithdrawOpen(true)}
      />

      {/* Main grid: charts + builder/advisor */}
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3 lg:gap-4">
        {/* Left: charts + positions (span 2) */}
        <div className="flex flex-col gap-3 lg:col-span-2 lg:gap-4">
          {/* P&L chart */}
          <Card className="flex flex-col gap-3">
            <CardHeader>
              <CardTitle icon={<LineChart className="h-4 w-4" />}>Portfolio value</CardTitle>
              <Badge tone="success" variant="soft" size="sm">
                Live
              </Badge>
            </CardHeader>
            {history.isPending ? (
              <Skeleton className="h-[18.75rem] w-full" />
            ) : liveHistory ? (
              <PnlChart history={liveHistory} live showPositions={positions.length > 0 && positions.length <= 8} />
            ) : (
              <div className="py-12 text-center text-xs text-muted">No history yet.</div>
            )}
          </Card>

          {/* Positions + allocation donut */}
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2 xl:gap-4">
            <Card className="flex flex-col gap-3">
              <CardHeader>
                <CardTitle icon={<PieChart className="h-4 w-4" />}>Allocation</CardTitle>
              </CardHeader>
              {portfolio.isPending ? (
                <Skeleton className="h-[16.25rem] w-full" />
              ) : (
                <AllocationDonut positions={positions} />
              )}
            </Card>

            <Card className="flex flex-col gap-3">
              <CardHeader>
                <CardTitle>Holdings</CardTitle>
                {positions.length > 0 && (
                  <span className="text-[0.6875rem] text-muted">{positions.length}</span>
                )}
              </CardHeader>
              {portfolio.isPending ? (
                <div className="flex flex-col gap-2">
                  {Array.from({ length: 2 }).map((_, i) => (
                    <Skeleton key={i} className="h-24 w-full" />
                  ))}
                </div>
              ) : positions.length === 0 ? (
                <div className="rounded-xl border border-dashed border-border py-8 text-center text-xs text-muted">
                  No holdings yet. Build an allocation to get started.
                </div>
              ) : (
                <div className="flex max-h-[26.25rem] flex-col gap-2 overflow-y-auto pr-0.5">
                  {positions.map((p) => (
                    <PositionCard key={p.symbol} position={p} />
                  ))}
                </div>
              )}
            </Card>
          </div>

          <TransactionList transactions={transactions.data ?? []} loading={transactions.isPending} limit={12} />
        </div>

        {/* Right: builder + advisor */}
        <div className="flex flex-col gap-3 lg:gap-4">
          <div id="allocation-builder">
            <AllocationBuilder
              cash={cash}
              onAdvice={setAdvice}
              appliedRows={applied}
              onInvested={() => {
                void portfolio.refetch();
                void wallet.refetch();
              }}
            />
          </div>

          <AdvicePanel amount={cash > 0 ? cash : 1000} advice={advice} onApply={applyAdvice} />

          <RiskControls positionsCount={positions.length} />
        </div>
      </div>

      {/* Modals */}
      <AddFundsModal open={addOpen} onClose={() => setAddOpen(false)} savedCards={wallet.data?.savedCards ?? []} />
      <WithdrawModal open={withdrawOpen} onClose={() => setWithdrawOpen(false)} wallet={wallet.data} />
    </div>
  );
}
