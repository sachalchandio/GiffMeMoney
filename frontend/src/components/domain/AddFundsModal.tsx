/**
 * AddFundsModal — the simulated (sandbox) deposit flow.
 *
 * Amount with quick chips (incl. $20), a debit-card form with a live brand icon
 * and Luhn check (number / exp / cvc / holder), a "Remember this card" toggle,
 * and a saved-card picker. Submits a {@link DepositRequest} to the wallet deposit
 * mutation. Clearly labelled "Demo / sandbox — no real charge"; raw PAN/CVC are
 * never persisted by the client. Colors via semantic tokens only.
 */

import { useEffect, useMemo, useState } from 'react';
import { CreditCard, Loader2, ShieldCheck, X } from 'lucide-react';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { ToggleSwitch } from '@/components/ui/ToggleSwitch';
import { useDeposit } from '@/hooks/useWallet';
import type { DepositRequest, SavedCard } from '@/lib/types';
import { formatCurrency } from '@/lib/format';
import { cn } from '@/lib/utils';
import {
  brandFor,
  brandLabel,
  cvcLength,
  cvcValid,
  DEMO_TEST_CARD,
  digitsOnly,
  expectedLength,
  expiryValid,
  formatCardNumber,
  luhn,
  maskCardNumber,
  type CardBrand,
} from '@/lib/payment';

export interface AddFundsModalProps {
  open: boolean;
  onClose: () => void;
  savedCards: SavedCard[];
  /** Called after a successful deposit (page can toast / refocus). */
  onSuccess?: () => void;
}

const QUICK_CHIPS = [20, 50, 100, 500, 1000] as const;
const MAX_AMOUNT = 10_000;

function BrandGlyph({ brand }: { brand: CardBrand }): JSX.Element {
  const tone: Record<CardBrand, string> = {
    visa: 'text-[#1a1f71] dark:text-[#7b87ff]',
    mastercard: 'text-warning',
    amex: 'text-accent',
    discover: 'text-warning',
    unknown: 'text-muted',
  };
  return (
    <span className={cn('inline-flex items-center gap-1 text-xs font-semibold', tone[brand])}>
      <CreditCard className="h-4 w-4" aria-hidden />
      {brandLabel(brand)}
    </span>
  );
}

export function AddFundsModal({ open, onClose, savedCards, onSuccess }: AddFundsModalProps): JSX.Element | null {
  const deposit = useDeposit();

  const [amount, setAmount] = useState<string>('100');
  const [savedCardId, setSavedCardId] = useState<string>('');
  const [number, setNumber] = useState<string>('');
  const [holder, setHolder] = useState<string>('');
  const [exp, setExp] = useState<string>('');
  const [cvc, setCvc] = useState<string>('');
  const [remember, setRemember] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Reset transient state whenever the modal opens.
  useEffect(() => {
    if (open) {
      setAmount('100');
      setSavedCardId(savedCards[0]?.id ?? '');
      setNumber('');
      setHolder('');
      setExp('');
      setCvc('');
      setRemember(false);
      setError(null);
      deposit.reset();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const usingSaved = savedCardId !== '';
  const brand = useMemo(() => brandFor(number), [number]);

  const amountNum = Number(amount);
  const amountValid = Number.isFinite(amountNum) && amountNum > 0 && amountNum <= MAX_AMOUNT;

  const expMonth = Number(exp.split('/')[0]?.trim() ?? '');
  const rawYear = Number(exp.split('/')[1]?.trim() ?? '');
  const expYear = rawYear > 0 && rawYear < 100 ? 2000 + rawYear : rawYear;

  const cardComplete = usingSaved
    ? true
    : luhn(number) &&
      expiryValid(expMonth, expYear) &&
      cvcValid(cvc, brand) &&
      holder.trim().length > 1;

  const canSubmit = amountValid && cardComplete && !deposit.isPending;

  if (!open) return null;

  const submit = (): void => {
    setError(null);
    if (!amountValid) {
      setError(`Enter an amount between $1 and ${formatCurrency(MAX_AMOUNT)}.`);
      return;
    }

    // For a saved card the backend re-uses the token; we still send a card payload
    // shape (masked on the server). For a saved card we reuse its expiry + a sentinel.
    const chosen = savedCards.find((c) => c.id === savedCardId);
    const req: DepositRequest = usingSaved && chosen
      ? {
          amount: amountNum,
          saveCard: false,
          savedCardId: chosen.id,
          card: {
            number: `0000000000000000`.slice(0, expectedLength(brandFor(chosen.brand))),
            expMonth: chosen.expMonth,
            expYear: chosen.expYear,
            cvc: '000',
            holder: chosen.holder,
          },
        }
      : {
          amount: amountNum,
          saveCard: remember,
          savedCardId: null,
          card: {
            number: digitsOnly(number),
            expMonth,
            expYear,
            cvc: digitsOnly(cvc),
            holder: holder.trim(),
          },
        };

    deposit.mutate(req, {
      onSuccess: () => {
        onSuccess?.();
        onClose();
      },
      onError: (err) => setError(err instanceof Error ? err.message : 'Deposit failed.'),
    });
  };

  return (
    <ModalShell title="Add funds" onClose={onClose}>
      <div className="flex flex-col gap-4">
        {/* Amount */}
        <Field label="Amount">
          <div className="relative">
            <span className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-sm text-muted">$</span>
            <input
              type="number"
              inputMode="decimal"
              min={1}
              max={MAX_AMOUNT}
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              className="h-10 w-full rounded-xl border border-border bg-surface pl-7 pr-3 text-lg font-semibold tnum text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT"
              aria-label="Deposit amount"
            />
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {QUICK_CHIPS.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => setAmount(String(c))}
                className={cn(
                  'rounded-lg border px-2.5 py-1 text-xs font-medium tnum transition-colors',
                  Number(amount) === c
                    ? 'border-primary bg-primary/12 text-primary'
                    : 'border-border bg-surface-2 text-muted hover:text-text',
                )}
              >
                ${c}
              </button>
            ))}
          </div>
        </Field>

        {/* Saved-card picker */}
        {savedCards.length > 0 && (
          <Field label="Pay with">
            <div className="flex flex-col gap-1.5">
              {savedCards.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  onClick={() => setSavedCardId(c.id)}
                  className={cn(
                    'flex items-center justify-between gap-2 rounded-xl border px-3 py-2 text-left text-sm transition-colors',
                    savedCardId === c.id ? 'border-primary bg-primary/8' : 'border-border bg-surface-2 hover:border-muted',
                  )}
                >
                  <span className="flex items-center gap-2 text-text">
                    <CreditCard className="h-4 w-4 text-muted" aria-hidden />
                    <span className="font-medium">{brandLabel(brandFor(c.brand))}</span>
                    <span className="tnum text-muted">•••• {c.last4}</span>
                  </span>
                  <span className="text-xs tnum text-muted">
                    {String(c.expMonth).padStart(2, '0')}/{String(c.expYear).slice(-2)}
                  </span>
                </button>
              ))}
              <button
                type="button"
                onClick={() => setSavedCardId('')}
                className={cn(
                  'flex items-center gap-2 rounded-xl border px-3 py-2 text-left text-sm transition-colors',
                  !usingSaved ? 'border-primary bg-primary/8 text-text' : 'border-border bg-surface-2 text-muted hover:text-text',
                )}
              >
                <CreditCard className="h-4 w-4" aria-hidden />
                Use a new card
              </button>
            </div>
          </Field>
        )}

        {/* New-card form */}
        {!usingSaved && (
          <div className="flex flex-col gap-3 rounded-xl border border-border bg-surface-2/40 p-3">
            <Field label="Card number">
              <div className="relative">
                <input
                  inputMode="numeric"
                  autoComplete="cc-number"
                  placeholder="1234 5678 9012 3456"
                  value={formatCardNumber(number)}
                  onChange={(e) => setNumber(digitsOnly(e.target.value).slice(0, expectedLength(brandFor(e.target.value))))}
                  className={cn(
                    'h-10 w-full rounded-xl border bg-surface pl-3 pr-24 text-sm tnum text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT',
                    number.length > 0 && !luhn(number) && digitsOnly(number).length >= expectedLength(brand)
                      ? 'border-danger'
                      : 'border-border',
                  )}
                  aria-label="Card number"
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2">
                  <BrandGlyph brand={brand} />
                </span>
              </div>
              {number.length > 0 && !luhn(number) && digitsOnly(number).length >= expectedLength(brand) && (
                <p className="mt-1 text-[11px] text-danger">That card number fails the Luhn check.</p>
              )}
            </Field>

            <Field label="Cardholder name">
              <input
                autoComplete="cc-name"
                placeholder="Jordan Rivera"
                value={holder}
                onChange={(e) => setHolder(e.target.value)}
                className="h-10 w-full rounded-xl border border-border bg-surface px-3 text-sm text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT"
                aria-label="Cardholder name"
              />
            </Field>

            <div className="grid grid-cols-2 gap-3">
              <Field label="Expiry (MM/YY)">
                <input
                  inputMode="numeric"
                  autoComplete="cc-exp"
                  placeholder="08/29"
                  value={exp}
                  onChange={(e) => setExp(e.target.value)}
                  className="h-10 w-full rounded-xl border border-border bg-surface px-3 text-sm tnum text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT"
                  aria-label="Card expiry"
                />
              </Field>
              <Field label={`CVC (${cvcLength(brand)} digits)`}>
                <input
                  inputMode="numeric"
                  autoComplete="cc-csc"
                  placeholder={brand === 'amex' ? '1234' : '123'}
                  value={cvc}
                  onChange={(e) => setCvc(digitsOnly(e.target.value).slice(0, cvcLength(brand)))}
                  className="h-10 w-full rounded-xl border border-border bg-surface px-3 text-sm tnum text-text outline-none transition-colors focus-visible:ring-2 focus-visible:ring-DEFAULT"
                  aria-label="Card CVC"
                />
              </Field>
            </div>

            <ToggleSwitch
              checked={remember}
              onChange={setRemember}
              label="Remember this card"
              description="Stored masked only (brand + last 4)."
              size="sm"
            />

            <button
              type="button"
              onClick={() => {
                setNumber(digitsOnly(DEMO_TEST_CARD));
                setHolder('Demo Investor');
                setExp('08/29');
                setCvc('123');
              }}
              className="self-start text-[11px] font-medium text-primary hover:underline"
            >
              Use the demo test card
            </button>
          </div>
        )}

        {error && (
          <p className="rounded-lg bg-danger/10 px-3 py-2 text-xs font-medium text-danger" role="alert">
            {error}
          </p>
        )}

        <div className="flex items-center justify-between gap-3">
          <Badge tone="warning" variant="soft" size="sm" icon={<ShieldCheck className="h-3 w-3" />}>
            Demo / sandbox — no real charge
          </Badge>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="md" onClick={onClose}>
              Cancel
            </Button>
            <Button variant="primary" size="md" onClick={submit} disabled={!canSubmit} loading={deposit.isPending}>
              {deposit.isPending ? (
                <span className="inline-flex items-center gap-1">
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> Processing
                </span>
              ) : (
                `Deposit ${amountValid ? formatCurrency(amountNum) : ''}`.trim()
              )}
            </Button>
          </div>
        </div>

        {usingSaved && savedCardId && (
          <p className="text-[11px] text-muted">
            Charging {maskCardNumber('0000' + (savedCards.find((c) => c.id === savedCardId)?.last4 ?? ''))} (saved card).
          </p>
        )}
      </div>
    </ModalShell>
  );
}

/* ------------------------------------------------------------------ */
/* Local modal primitives (shared by the invest modals)               */
/* ------------------------------------------------------------------ */

export function ModalShell({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}): JSX.Element {
  // Close on Escape.
  useEffect(() => {
    const onKey = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-black/40 p-0 backdrop-blur-sm sm:items-center sm:p-4"
      role="dialog"
      aria-modal="true"
      aria-label={title}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="max-h-[92vh] w-full max-w-md animate-slide-up overflow-y-auto rounded-t-2xl border border-border bg-surface p-4 shadow-pop sm:rounded-2xl">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-base font-semibold tracking-tight text-text">{title}</h2>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="flex h-8 w-8 items-center justify-center rounded-xl text-muted transition-colors hover:bg-surface-2 hover:text-text"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

export function Field({ label, children }: { label: string; children: React.ReactNode }): JSX.Element {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs font-medium text-muted">{label}</span>
      {children}
    </label>
  );
}
