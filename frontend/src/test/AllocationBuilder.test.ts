/**
 * AllocationBuilder math — exercises the pure {@link computeAllocation} helper:
 * amounts sum, remaining clamps, over-budget flagging, and per-row weight
 * readouts. (The interactive component is covered indirectly by the InvestPage
 * render test; the arithmetic is asserted directly here.)
 */

import { describe, expect, it } from 'vitest';
import { computeAllocation, type AllocationRow } from '@/components/domain/AllocationBuilder';

const rows = (...pairs: [string, number][]): AllocationRow[] =>
  pairs.map(([symbol, amount]) => ({ symbol, amount }));

describe('computeAllocation', () => {
  it('sums funded rows and computes the remaining budget', () => {
    const r = computeAllocation(rows(['AAPL', 300], ['MSFT', 200]), 1000);
    expect(r.allocated).toBe(500);
    expect(r.remaining).toBe(500);
    expect(r.fraction).toBeCloseTo(0.5, 6);
    expect(r.overBudget).toBe(false);
  });

  it('derives per-symbol weights of the allocated total (sums to ~1)', () => {
    const r = computeAllocation(rows(['AAPL', 300], ['MSFT', 100]), 1000);
    expect(r.weights.AAPL).toBeCloseTo(0.75, 6);
    expect(r.weights.MSFT).toBeCloseTo(0.25, 6);
    const sum = Object.values(r.weights).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 6);
  });

  it('clamps the fill fraction to 1 and flags over-budget allocations', () => {
    const r = computeAllocation(rows(['AAPL', 800], ['MSFT', 700]), 1000);
    expect(r.allocated).toBe(1500);
    expect(r.fraction).toBe(1); // clamped
    expect(r.remaining).toBe(-500);
    expect(r.overBudget).toBe(true);
  });

  it('treats non-finite / negative amounts as zero', () => {
    const r = computeAllocation(rows(['AAPL', Number.NaN], ['MSFT', -50], ['NVDA', 100]), 1000);
    expect(r.allocated).toBe(100);
    expect(r.weights.NVDA).toBeCloseTo(1, 6);
    expect(r.weights.AAPL).toBe(0);
    expect(r.weights.MSFT).toBe(0);
  });

  it('handles a zero / invalid cash budget without dividing by zero', () => {
    const r = computeAllocation(rows(['AAPL', 100]), 0);
    expect(r.fraction).toBe(0);
    expect(r.overBudget).toBe(true); // any spend exceeds a zero budget
    const empty = computeAllocation([], 0);
    expect(empty.allocated).toBe(0);
    expect(empty.fraction).toBe(0);
    expect(empty.overBudget).toBe(false);
  });

  it('full allocation leaves no remaining and is not over budget', () => {
    const r = computeAllocation(rows(['AAPL', 600], ['MSFT', 400]), 1000);
    expect(r.remaining).toBe(0);
    expect(r.fraction).toBe(1);
    expect(r.overBudget).toBe(false);
  });
});
