/**
 * UI-utility tests (FRONTEND.md test matrix) — covers the pure helpers in
 * `lib/utils.ts`: the Tailwind-aware classname merge, the stance → tone/color
 * token mapping, score → stance thresholds, and the small numeric helpers.
 * Color helpers must return semantic tokens (never hardcoded hex).
 */

import { describe, expect, it } from 'vitest';
import {
  assetClassLabel,
  changeTextColor,
  clamp,
  cn,
  colorIndex,
  initials,
  round,
  stanceBadgeClasses,
  stanceColorVar,
  stanceFromScore,
  stanceTextColor,
  stanceTone,
  sum,
} from '@/lib/utils';
import type { Stance } from '@/lib/types';

describe('cn', () => {
  it('merges conditionals and de-dupes conflicting Tailwind classes', () => {
    expect(cn('p-2', false && 'hidden', 'p-4')).toBe('p-4');
    expect(cn('text-text', undefined, 'font-medium')).toBe('text-text font-medium');
  });
});

describe('stanceTone', () => {
  it('groups stances into positive / neutral / negative', () => {
    expect(stanceTone('STRONG_BUY')).toBe('positive');
    expect(stanceTone('BUY')).toBe('positive');
    expect(stanceTone('HOLD')).toBe('neutral');
    expect(stanceTone('SELL')).toBe('negative');
    expect(stanceTone('STRONG_SELL')).toBe('negative');
  });
});

describe('stance color helpers return semantic tokens', () => {
  it('maps tone → text token', () => {
    expect(stanceTextColor('BUY')).toBe('text-success');
    expect(stanceTextColor('HOLD')).toBe('text-warning');
    expect(stanceTextColor('SELL')).toBe('text-danger');
  });

  it('maps tone → soft badge classes', () => {
    expect(stanceBadgeClasses('STRONG_BUY')).toContain('text-success');
    expect(stanceBadgeClasses('HOLD')).toContain('text-warning');
    expect(stanceBadgeClasses('STRONG_SELL')).toContain('text-danger');
  });

  it('maps tone → CSS variable (no hardcoded hex)', () => {
    expect(stanceColorVar('BUY')).toBe('var(--success)');
    expect(stanceColorVar('HOLD')).toBe('var(--warning)');
    expect(stanceColorVar('SELL')).toBe('var(--danger)');
  });
});

describe('changeTextColor', () => {
  it('keys off the sign of the value', () => {
    expect(changeTextColor(1.2)).toBe('text-success');
    expect(changeTextColor(-1.2)).toBe('text-danger');
    expect(changeTextColor(0)).toBe('text-muted');
  });
});

describe('stanceFromScore', () => {
  it('maps the -100..100 score onto the backend thresholds', () => {
    const cases: [number, Stance][] = [
      [80, 'STRONG_BUY'],
      [60, 'STRONG_BUY'],
      [40, 'BUY'],
      [20, 'BUY'],
      [0, 'HOLD'],
      [-19, 'HOLD'],
      [-20, 'SELL'],
      [-40, 'SELL'],
      [-60, 'STRONG_SELL'],
      [-100, 'STRONG_SELL'],
    ];
    for (const [score, stance] of cases) {
      expect(stanceFromScore(score)).toBe(stance);
    }
  });
});

describe('assetClassLabel', () => {
  it('humanizes the asset classes', () => {
    expect(assetClassLabel('equity')).toBe('Stocks');
    expect(assetClassLabel('crypto')).toBe('Crypto');
    expect(assetClassLabel('etf')).toBe('ETFs');
  });
});

describe('numeric helpers', () => {
  it('clamps into range', () => {
    expect(clamp(5, 0, 10)).toBe(5);
    expect(clamp(-1, 0, 10)).toBe(0);
    expect(clamp(11, 0, 10)).toBe(10);
  });

  it('rounds to fixed decimals', () => {
    expect(round(1.23456)).toBe(1.23);
    expect(round(1.23456, 3)).toBe(1.235);
  });

  it('sums while ignoring non-finite values', () => {
    expect(sum([1, 2, 3])).toBe(6);
    expect(sum([1, Number.NaN, Number.POSITIVE_INFINITY, 4])).toBe(5);
    expect(sum([])).toBe(0);
  });
});

describe('initials', () => {
  it('derives 1-2 letter initials and handles edge cases', () => {
    expect(initials('Ada Lovelace')).toBe('AL');
    expect(initials('Cher')).toBe('CH');
    expect(initials('  jane  q  public ')).toBe('JP');
    expect(initials('')).toBe('?');
  });
});

describe('colorIndex', () => {
  it('is deterministic and stays within the palette range', () => {
    const a = colorIndex('AAPL');
    expect(a).toBe(colorIndex('AAPL'));
    expect(a).toBeGreaterThanOrEqual(0);
    expect(a).toBeLessThan(8);
    expect(colorIndex('NVDA', 4)).toBeLessThan(4);
  });
});
