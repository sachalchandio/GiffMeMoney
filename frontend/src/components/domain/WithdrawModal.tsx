/**
 * WithdrawModal — the simulated payout flow. Amount (capped at the cash balance)
 * with a "Max" shortcut and an optional destination note; confirms via the wallet
 * withdraw mutation. Labelled as a sandbox payout. Colors via semantic tokens only.
 */

import { useEffect, useState } from 'react';
import { ArrowDownToLine, ShieldCheck } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { useWithdraw } from '@/hooks/useWallet';
import type { Wallet, WithdrawRequest } from '@/lib/types';
import { formatCurrency } from '@/lib/format';
import { cn } from '@/lib/utils';
import { ModalShell, Field } from './AddFundsModal';

export interface WithdrawModalProps {
  open: boolean;
  onClose: () => void;
  wallet: Wallet | undefined;
  onSuccess?: () => void;
}

export function WithdrawModal({ open, onClose, wallet, onSuccess }: WithdrawModalProps): JSX.Element | null {
  const withdraw = useWithdraw();
  const cash = wallet?.cashBalance ?? 0;
  const currency = wallet?.currency ?? 'USD';

  const [amount, setAmount] = useState<string>('');
  const [destination, setDestination] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setAmount('');
      setDestination('');
      setError(null);
      withdraw.reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  const amountNum = Number(amount);
  const valid = Number.isFinite(amountNum) && amountNum > 0 && amountNum <= cash;
  const canSubmit = valid && !withdraw.isPending;

  const submit = (): void => {
    setError(null);
    if (!valid) {
      setError(`Enter an amount between $1 and ${formatCurrency(cash, { currency })}.`);
      return;
    }
    const req: WithdrawRequest = {
      amount: amountNum,
      destination: destination.trim() === '' ? null : destination.trim(),
    };
    withdraw.mutate(req, {
      onSuccess: () => {
        onSuccess?.();
        onClose();
      },
      onError: (err) => setError(err instanceof Error ? err.message : 'Withdrawal failed.'),
    });
  };

  return (
    <ModalShell title="Withdraw cash" onClose={onClose}>
      <div className="flex flex-col gap-4">
        <div className="rounded-xl border border-border bg-surface-2/50 px-3 py-2.5">
          <p className="text-[11px] font-medium uppercase tracking-wide text-muted">Available cash</p>
          <p className="mt-0.5 text-xl font-semibold tnum text-text">{formatCurrency(cash, { currency })}</p>
        </div>

        <Field label="Amount">
          <div className="relative">
            <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-sm text-muted">$</span>
            <input
              type="number"
              inputMode="decimal"
              min={1}
              max={cash}
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className={cn(
                'h-10 w-full rounded-xl border bg-surface pl-7 pr-16 text-lg font-semibold tnum text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT',
                amount !== '' && !valid ? 'border-danger' : 'border-border',
              )}
              aria-label="Withdraw amount"
            />
            <button
              type="button"
              onClick={() => setAmount(String(cash))}
              disabled={cash <= 0}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded-lg bg-surface-2 px-2 py-1 text-xs font-medium text-primary disabled:opacity-50"
            >
              Max
            </button>
          </div>
        </Field>

        <Field label="Destination (optional)">
          <input
            placeholder="Bank •••• 1234"
            value={destination}
            onChange={(e) => setDestination(e.target.value)}
            className="h-10 w-full rounded-xl border border-border bg-surface px-3 text-sm text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT"
            aria-label="Destination"
          />
        </Field>

        {error && (
          <p className="rounded-lg bg-danger/10 px-3 py-2 text-xs font-medium text-danger" role="alert">
            {error}
          </p>
        )}

        <div className="flex items-center justify-between gap-3">
          <Badge tone="warning" variant="soft" size="sm" icon={<ShieldCheck className="h-3 w-3" />}>
            Sandbox payout
          </Badge>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="md" onClick={onClose}>
              Cancel
            </Button>
            <Button
              variant="primary"
              size="md"
              onClick={submit}
              disabled={!canSubmit}
              loading={withdraw.isPending}
              leftIcon={!withdraw.isPending ? <ArrowDownToLine className="h-4 w-4" /> : undefined}
            >
              {`Withdraw ${valid ? formatCurrency(amountNum, { currency }) : ''}`.trim()}
            </Button>
          </div>
        </div>
      </div>
    </ModalShell>
  );
}
