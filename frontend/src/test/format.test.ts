/**
 * Formatter tests (FRONTEND.md test matrix) — exercises the display formatters
 * in `lib/format.ts`. These are pure string producers, so we assert exact
 * output (en-US locale) including the NaN/Infinity → 0 safety coercion, the
 * percent-vs-fraction distinction, signed deltas, and the relative-time ladder.
 */

import { describe, expect, it } from 'vitest';
import {
  formatCompact,
  formatCompactCurrency,
  formatCurrency,
  formatDate,
  formatFractionPct,
  formatNumber,
  formatPct,
  formatPrice,
  formatProbability,
  formatRatio,
  formatRelativeTime,
  formatSigned,
  horizonLabel,
  stanceLabel,
} from '@/lib/format';

describe('formatCurrency', () => {
  it('formats a plain USD amount with two decimals', () => {
    expect(formatCurrency(1234.5)).toBe('$1,234.50');
  });

  it('coerces NaN / Infinity / null / undefined to $0.00', () => {
    expect(formatCurrency(Number.NaN)).toBe('$0.00');
    expect(formatCurrency(Number.POSITIVE_INFINITY)).toBe('$0.00');
    expect(formatCurrency(null)).toBe('$0.00');
    expect(formatCurrency(undefined)).toBe('$0.00');
  });

  it('honours digit overrides', () => {
    expect(formatCurrency(1000, { minimumFractionDigits: 0, maximumFractionDigits: 0 })).toBe(
      '$1,000',
    );
  });
});

describe('formatCompactCurrency', () => {
  it('falls back to plain currency under 1,000', () => {
    expect(formatCompactCurrency(950)).toBe('$950.00');
  });

  it('compacts millions and billions (up to 2 fraction digits, keeps trailing zeros)', () => {
    expect(formatCompactCurrency(1_200_000)).toBe('$1.20M');
    expect(formatCompactCurrency(1_230_000)).toBe('$1.23M');
    expect(formatCompactCurrency(3_400_000_000)).toBe('$3.40B');
  });
});

describe('formatNumber / formatCompact', () => {
  it('adds thousands separators', () => {
    expect(formatNumber(1234567.891)).toBe('1,234,567.89');
  });

  it('compacts large magnitudes', () => {
    expect(formatCompact(1_200_000)).toBe('1.2M');
    expect(formatCompact(850_000)).toBe('850K');
  });
});

describe('formatPct / formatFractionPct / formatProbability', () => {
  it('treats the input as already-percent units', () => {
    expect(formatPct(12.3)).toBe('12.30%');
    expect(formatPct(12.345, { digits: 1 })).toBe('12.3%');
  });

  it('adds a leading + only for positive values when sign requested', () => {
    expect(formatPct(5, { sign: true })).toBe('+5.00%');
    expect(formatPct(-5, { sign: true })).toBe('-5.00%');
    expect(formatPct(0, { sign: true })).toBe('0.00%');
  });

  it('scales a 0..1 fraction into percent units', () => {
    expect(formatFractionPct(0.123)).toBe('12.30%');
    expect(formatFractionPct(0.5, { digits: 0, sign: true })).toBe('+50%');
  });

  it('rounds a probability to a whole-number percent', () => {
    expect(formatProbability(0.731)).toBe('73%');
    expect(formatProbability(0)).toBe('0%');
    expect(formatProbability(1)).toBe('100%');
  });
});

describe('formatSigned', () => {
  it('prefixes + / unicode minus and uses fixed digits', () => {
    expect(formatSigned(1.234)).toBe('+1.23');
    expect(formatSigned(-1.234)).toBe('−1.23');
    expect(formatSigned(0)).toBe('0.00');
  });
});

describe('formatPrice', () => {
  it('uses more decimals for sub-$1 assets', () => {
    expect(formatPrice(123.456)).toBe('$123.46');
    expect(formatPrice(0.1234)).toBe('$0.1234');
    expect(formatPrice(0.001234)).toBe('$0.001234');
  });
});

describe('formatRatio', () => {
  it('renders a fixed-precision number', () => {
    expect(formatRatio(1.236)).toBe('1.24');
    expect(formatRatio(2, 1)).toBe('2.0');
  });
});

describe('formatDate / formatRelativeTime', () => {
  it('renders a short absolute date', () => {
    // 2026-06-17T12:00:00Z — assert the parts to stay timezone-tolerant.
    const out = formatDate(Date.UTC(2026, 5, 17, 12));
    expect(out).toContain('2026');
    expect(out).toContain('Jun');
  });

  it('walks the relative-time ladder', () => {
    const now = 10_000_000_000;
    expect(formatRelativeTime(now, now)).toBe('just now');
    expect(formatRelativeTime(now - 5_000, now)).toBe('just now'); // < 45s
    expect(formatRelativeTime(now - 5 * 60_000, now)).toBe('5m ago');
    expect(formatRelativeTime(now - 2 * 3_600_000, now)).toBe('2h ago');
    expect(formatRelativeTime(now - 3 * 86_400_000, now)).toBe('3d ago');
    expect(formatRelativeTime(now + 1_000, now)).toBe('just now'); // future
  });
});

describe('horizonLabel / stanceLabel', () => {
  it('humanizes horizon codes', () => {
    expect(horizonLabel('1D')).toBe('1 Day');
    expect(horizonLabel('5Y')).toBe('5 Years');
    expect(horizonLabel('XX')).toBe('XX');
  });

  it('title-cases an underscore-delimited stance code', () => {
    expect(stanceLabel('STRONG_BUY')).toBe('Strong Buy');
    expect(stanceLabel('HOLD')).toBe('Hold');
  });
});
