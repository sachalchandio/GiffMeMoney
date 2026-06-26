/**
 * RiskControls — the post-buy loss-control panel for the Invest page.
 *
 * Lets the user configure the account's protective EXIT rules — stop-loss,
 * trailing-stop, take-profit and a portfolio max-drawdown circuit-breaker —
 * each an optional positive percent (toggle off = rule disabled). "Save policy"
 * persists them (`PUT /api/portfolio/risk`); "Apply protections" evaluates the
 * policy now and auto-sells / de-risks any breaching positions
 * (`POST /api/portfolio/risk/apply`), then surfaces the triggered actions.
 *
 * HONESTY: these are mechanical, after-the-fact exits on a SIMULATION over
 * synthetic data — they do NOT guarantee a profit or prevent loss. The panel
 * states this plainly and never implies otherwise. Colors via semantic tokens.
 */

import { useEffect, useMemo, useState } from 'react';
import { Check, Save, ShieldCheck, Siren } from 'lucide-react';
import { Card, CardHeader, CardTitle } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { ToggleSwitch } from '@/components/ui/ToggleSwitch';
import { useApplyRisk, useRiskPolicy, useSetRiskPolicy } from '@/hooks/useRiskPolicy';
import type { RiskAction, RiskPolicy } from '@/lib/types';
import { formatCurrency } from '@/lib/format';
import { cn } from '@/lib/utils';

export interface RiskControlsProps {
  /** Number of open positions; the Apply action is disabled when there are none. */
  positionsCount: number;
  className?: string;
}

/** One configurable rule's identity within the policy. */
type RuleKey = keyof Pick<
  RiskPolicy,
  'stopLossPct' | 'trailingStopPct' | 'takeProfitPct' | 'maxDrawdownPct'
>;

interface RuleSpec {
  key: RuleKey;
  label: string;
  help: string;
  /** Sensible default percent applied when the rule is toggled ON from empty. */
  fallback: number;
}

const RULES: readonly RuleSpec[] = [
  { key: 'stopLossPct', label: 'Stop-loss', help: 'Exit a position once down this % from entry.', fallback: 10 },
  { key: 'trailingStopPct', label: 'Trailing stop', help: 'Exit once down this % from its high-water mark.', fallback: 15 },
  { key: 'takeProfitPct', label: 'Take-profit', help: 'Exit once up this % from entry.', fallback: 40 },
  { key: 'maxDrawdownPct', label: 'Max drawdown', help: 'Raise cash once total value is down this % from its peak.', fallback: 25 },
] as const;

/** Editable draft: per-rule enabled flag + a string input (so the box can be empty). */
type Draft = Record<RuleKey, { on: boolean; value: string }>;

function policyToDraft(policy: RiskPolicy | undefined): Draft {
  const out = {} as Draft;
  for (const { key } of RULES) {
    const v = policy?.[key];
    const on = typeof v === 'number' && Number.isFinite(v) && v > 0;
    out[key] = { on, value: on ? String(v) : '' };
  }
  return out;
}

/** Translate the draft into the wire policy (disabled / blank → `null`). */
function draftToPolicy(draft: Draft): RiskPolicy {
  const out: RiskPolicy = {};
  for (const { key } of RULES) {
    const cell = draft[key];
    const n = Number(cell.value);
    out[key] = cell.on && Number.isFinite(n) && n > 0 ? n : null;
  }
  return out;
}

const ACTION_LABEL: Record<RiskAction['action'], string> = {
  stop_loss: 'Stop-loss',
  trailing_stop: 'Trailing stop',
  take_profit: 'Take-profit',
  drawdown: 'Drawdown',
};

export function RiskControls({ positionsCount, className }: RiskControlsProps): JSX.Element {
  const policyQuery = useRiskPolicy();
  const setPolicy = useSetRiskPolicy();
  const applyRisk = useApplyRisk();

  const [draft, setDraft] = useState<Draft>(() => policyToDraft(undefined));
  const [saved, setSaved] = useState(false);

  // Adopt the loaded policy once (and whenever it changes server-side).
  useEffect(() => {
    if (policyQuery.data) setDraft(policyToDraft(policyQuery.data));
  }, [policyQuery.data]);

  const anyEnabled = useMemo(() => RULES.some((r) => draft[r.key].on), [draft]);

  const setRuleOn = (key: RuleKey, on: boolean): void => {
    setSaved(false);
    setDraft((prev) => {
      const cell = prev[key];
      const spec = RULES.find((r) => r.key === key);
      const value = on && cell.value.trim() === '' ? String(spec?.fallback ?? 10) : cell.value;
      return { ...prev, [key]: { on, value } };
    });
  };

  const setRuleValue = (key: RuleKey, value: string): void => {
    setSaved(false);
    setDraft((prev) => ({ ...prev, [key]: { ...prev[key], value } }));
  };

  const save = (): void => {
    setSaved(false);
    setPolicy.mutate(draftToPolicy(draft), { onSuccess: () => setSaved(true) });
  };

  const apply = (): void => {
    setSaved(false);
    applyRisk.mutate();
  };

  const result = applyRisk.data;
  const actions = result?.actions ?? [];
  const canApply = positionsCount > 0 && anyEnabled && !applyRisk.isPending;

  return (
    <Card className={cn('flex flex-col gap-4', className)}>
      <CardHeader>
        <div>
          <CardTitle icon={<ShieldCheck className="h-4 w-4" />}>Risk protections</CardTitle>
          <p className="mt-1 text-xs text-muted">
            Mechanical exit rules for your open positions — off by default.
          </p>
        </div>
        {anyEnabled ? (
          <Badge tone="primary" variant="soft" size="sm">
            Armed
          </Badge>
        ) : (
          <Badge tone="neutral" variant="soft" size="sm">
            Off
          </Badge>
        )}
      </CardHeader>

      {/* Rules */}
      <ul className="flex flex-col gap-2">
        {RULES.map((rule) => {
          const cell = draft[rule.key];
          return (
            <li
              key={rule.key}
              className="flex flex-col gap-2 rounded-xl border border-border bg-surface-2/40 px-3 py-2.5 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-text">{rule.label}</span>
                </div>
                <p className="line-clamp-1 text-[11px] text-muted">{rule.help}</p>
              </div>
              <div className="flex items-center gap-2.5">
                <div className="relative w-24">
                  <input
                    type="number"
                    inputMode="decimal"
                    min={0}
                    step="any"
                    value={cell.value}
                    disabled={!cell.on}
                    onChange={(e) => setRuleValue(rule.key, e.target.value)}
                    placeholder="0"
                    className={cn(
                      'h-9 w-full rounded-xl border border-border bg-surface pr-6 pl-2.5 text-right text-sm tnum text-text outline-none transition-colors',
                      'focus-visible:ring-2 focus-visible:ring-DEFAULT',
                      !cell.on && 'cursor-not-allowed opacity-50',
                    )}
                    aria-label={`${rule.label} threshold percent`}
                  />
                  <span className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-xs text-muted">%</span>
                </div>
                <ToggleSwitch
                  checked={cell.on}
                  onChange={(on) => setRuleOn(rule.key, on)}
                  size="sm"
                  aria-label={`Enable ${rule.label}`}
                />
              </div>
            </li>
          );
        })}
      </ul>

      {/* Errors */}
      {(setPolicy.isError || applyRisk.isError) && (
        <p className="rounded-lg bg-danger/10 px-3 py-2 text-xs font-medium text-danger" role="alert">
          {(setPolicy.error ?? applyRisk.error) instanceof Error
            ? (setPolicy.error ?? applyRisk.error)?.message
            : 'Could not update protections.'}
        </p>
      )}

      {/* Apply result */}
      {result &&
        (actions.length > 0 ? (
          <div className="flex flex-col gap-2 rounded-xl border border-warning/30 bg-warning/10 px-3 py-2.5">
            <p className="flex items-center gap-1.5 text-xs font-semibold text-warning">
              <Siren className="h-3.5 w-3.5" aria-hidden />
              {actions.length} protection{actions.length === 1 ? '' : 's'} triggered
            </p>
            <ul className="flex flex-col gap-1">
              {actions.map((a, i) => (
                <li key={`${a.symbol}-${a.action}-${i}`} className="flex items-center justify-between gap-2 text-[11px]">
                  <span className="min-w-0 truncate text-text">
                    <span className="font-semibold">{a.symbol}</span>{' '}
                    <span className="text-muted">{ACTION_LABEL[a.action]}</span>
                  </span>
                  <span className="shrink-0 tnum text-muted">
                    {formatCurrency(a.amount)}{' '}
                    <span className={a.realizedPnl >= 0 ? 'text-success' : 'text-danger'}>
                      ({a.realizedPnl >= 0 ? '+' : '−'}
                      {formatCurrency(Math.abs(a.realizedPnl))})
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <p className="flex items-center gap-1.5 rounded-xl border border-success/25 bg-success/5 px-3 py-2 text-xs font-medium text-success">
            <Check className="h-3.5 w-3.5" aria-hidden />
            All positions within your limits — nothing triggered.
          </p>
        ))}

      {/* Actions */}
      <div className="flex flex-col gap-2 sm:flex-row">
        <Button
          variant="secondary"
          size="md"
          fullWidth
          onClick={save}
          loading={setPolicy.isPending}
          leftIcon={!setPolicy.isPending ? <Save className="h-3.5 w-3.5" /> : undefined}
        >
          {saved ? 'Saved' : 'Save policy'}
        </Button>
        <Button
          variant="primary"
          size="md"
          fullWidth
          onClick={apply}
          disabled={!canApply}
          loading={applyRisk.isPending}
          leftIcon={!applyRisk.isPending ? <ShieldCheck className="h-3.5 w-3.5" /> : undefined}
        >
          Apply protections
        </Button>
      </div>

      {positionsCount === 0 && (
        <p className="text-[11px] text-muted">Add a holding to apply protections.</p>
      )}

      <p className="text-[11px] leading-snug text-muted">
        {result?.disclaimer ??
          'Educational simulation on synthetic data. Risk controls are mechanical, after-the-fact exits — they do not guarantee a profit or prevent loss.'}
      </p>
    </Card>
  );
}
