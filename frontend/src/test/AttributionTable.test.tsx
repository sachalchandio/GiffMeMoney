/**
 * AttributionTable best/worst highlighting (BOT.md test matrix).
 *
 * The simulated auto-trader's per-sleeve attribution must make it obvious "which
 * trading did the best and which did the worst": the single best contributor is
 * tinted green with a Best badge, the single worst is tinted red with a Worst
 * badge, and neutral sleeves carry no verdict badge. The table also defends its
 * own best→worst display order regardless of the input order. Everything shown is
 * simulated paper-trading on synthetic data.
 */

import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AttributionTable } from '@/components/domain/AttributionTable';
import type { SleeveAttribution } from '@/lib/types';

function row(over: Partial<SleeveAttribution> & { key: string }): SleeveAttribution {
  return {
    key: over.key,
    realizedPnl: over.realizedPnl ?? 0,
    contributionPct: over.contributionPct ?? 0,
    winRate: over.winRate ?? 0.5,
    trades: over.trades ?? 2,
    verdict: over.verdict ?? 'neutral',
  };
}

// Deliberately UNSORTED input (worst first) so we also test the defensive sort.
const ATTRIBUTION: SleeveAttribution[] = [
  row({ key: 'ADA', realizedPnl: -80, contributionPct: -40, verdict: 'worst', trades: 3 }),
  row({ key: 'MSFT', realizedPnl: 5, contributionPct: 2, verdict: 'neutral', trades: 2 }),
  row({ key: 'NVDA', realizedPnl: 120, contributionPct: 58, verdict: 'best', trades: 4 }),
];

/** The closest table row (<tr>) ancestor of an element. */
function rowOf(el: HTMLElement): HTMLElement {
  const tr = el.closest('tr');
  if (!tr) throw new Error('no <tr> ancestor');
  return tr as HTMLElement;
}

describe('AttributionTable', () => {
  it('flags the best sleeve green and the worst sleeve red', () => {
    render(<AttributionTable attribution={ATTRIBUTION} />);

    // Verdict badges are present exactly once each.
    expect(screen.getByText('Best')).toBeInTheDocument();
    expect(screen.getByText('Worst')).toBeInTheDocument();

    const bestRow = rowOf(screen.getByText('NVDA'));
    const worstRow = rowOf(screen.getByText('ADA'));
    // Best row carries the success tint; worst row the danger tint.
    expect(bestRow.className).toContain('bg-success/8');
    expect(worstRow.className).toContain('bg-danger/8');
    expect(bestRow.className).not.toContain('bg-danger/8');
    expect(worstRow.className).not.toContain('bg-success/8');
  });

  it('does not badge a neutral sleeve', () => {
    render(<AttributionTable attribution={ATTRIBUTION} />);
    const neutralRow = rowOf(screen.getByText('MSFT'));
    // The neutral row has no Best/Worst badge and no verdict tint.
    expect(neutralRow.textContent).not.toMatch(/Best|Worst/);
    expect(neutralRow.className).not.toContain('bg-success/8');
    expect(neutralRow.className).not.toContain('bg-danger/8');
  });

  it('renders rows in best → worst order regardless of input order', () => {
    render(<AttributionTable attribution={ATTRIBUTION} />);
    const dataRows = screen
      .getAllByRole('row')
      .filter((r) => r.querySelector('td')); // drop the header row
    const keys = dataRows.map((r) => r.querySelector('td')?.textContent?.trim());
    // NVDA (best, +120) → MSFT (neutral, +5) → ADA (worst, −80) by contribution.
    expect(keys[0]).toContain('NVDA');
    expect(keys[1]).toContain('MSFT');
    expect(keys[2]).toContain('ADA');
  });

  it('shows an empty-state message when there is no attribution', () => {
    render(<AttributionTable attribution={[]} />);
    expect(screen.getByText(/No sleeve attribution for this run/i)).toBeInTheDocument();
    expect(screen.queryByText('Best')).not.toBeInTheDocument();
    expect(screen.queryByText('Worst')).not.toBeInTheDocument();
  });
});
