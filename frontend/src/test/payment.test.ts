/**
 * Payment-helper tests (FRONTEND.md test matrix) — covers the sandbox Add-Funds
 * card helpers in `lib/payment.ts`: Luhn validation, brand/IIN detection,
 * brand-aware grouping + masking, and expiry/CVC validation.
 */

import { describe, expect, it } from 'vitest';
import {
  DEMO_TEST_CARD,
  brandFor,
  brandLabel,
  cvcLength,
  cvcValid,
  digitsOnly,
  expectedLength,
  expiryValid,
  formatCardNumber,
  luhn,
  maskCardNumber,
} from '@/lib/payment';

describe('digitsOnly', () => {
  it('strips spaces, dashes, and other non-digits', () => {
    expect(digitsOnly('4242 4242-4242_4242')).toBe('4242424242424242');
    expect(digitsOnly('abc')).toBe('');
  });
});

describe('luhn', () => {
  it('accepts known-good PANs (incl. the demo test card)', () => {
    expect(luhn(DEMO_TEST_CARD)).toBe(true);
    expect(luhn('4242424242424242')).toBe(true); // Visa test
    expect(luhn('5555555555554444')).toBe(true); // Mastercard test
    expect(luhn('378282246310005')).toBe(true); // Amex test (15 digits)
    expect(luhn('6011111111111117')).toBe(true); // Discover test
  });

  it('rejects a number that fails the checksum', () => {
    expect(luhn('4242424242424241')).toBe(false);
  });

  it('rejects implausible lengths (< 12 or > 19 digits)', () => {
    expect(luhn('4242')).toBe(false);
    expect(luhn('42424242424242424242')).toBe(false); // 20 digits
  });
});

describe('brandFor', () => {
  it('detects the major networks from their IIN ranges', () => {
    expect(brandFor('4111111111111111')).toBe('visa');
    expect(brandFor('5500000000000004')).toBe('mastercard');
    expect(brandFor('2221000000000009')).toBe('mastercard'); // new 2-series range
    expect(brandFor('340000000000009')).toBe('amex');
    expect(brandFor('371449635398431')).toBe('amex');
    expect(brandFor('6011000000000004')).toBe('discover');
    expect(brandFor('6500000000000002')).toBe('discover');
  });

  it('returns "unknown" for an empty or unrecognized prefix', () => {
    expect(brandFor('')).toBe('unknown');
    expect(brandFor('9999000000000000')).toBe('unknown');
  });
});

describe('expectedLength / cvcLength', () => {
  it('uses 15/4 for Amex and 16/3 otherwise', () => {
    expect(expectedLength('amex')).toBe(15);
    expect(expectedLength('visa')).toBe(16);
    expect(cvcLength('amex')).toBe(4);
    expect(cvcLength('mastercard')).toBe(3);
  });
});

describe('formatCardNumber', () => {
  it('groups non-Amex cards into blocks of four and caps at 16 digits', () => {
    expect(formatCardNumber('4242424242424242')).toBe('4242 4242 4242 4242');
    expect(formatCardNumber('42424242424242429999')).toBe('4242 4242 4242 4242');
  });

  it('groups Amex as 4-6-5', () => {
    expect(formatCardNumber('378282246310005')).toBe('3782 822463 10005');
  });
});

describe('maskCardNumber', () => {
  it('masks all but the last four digits', () => {
    expect(maskCardNumber('4242 4242 4242 4242')).toBe('•••• 4242');
  });
});

describe('brandLabel', () => {
  it('humanizes each brand', () => {
    expect(brandLabel('visa')).toBe('Visa');
    expect(brandLabel('mastercard')).toBe('Mastercard');
    expect(brandLabel('amex')).toBe('Amex');
    expect(brandLabel('discover')).toBe('Discover');
    expect(brandLabel('unknown')).toBe('Card');
  });
});

describe('expiryValid', () => {
  const now = new Date(2026, 5, 17); // 2026-06-17

  it('accepts a future expiry and the current month', () => {
    expect(expiryValid(12, 2026, now)).toBe(true);
    expect(expiryValid(6, 2026, now)).toBe(true); // current month, end-of-month valid
  });

  it('rejects a past expiry and out-of-range months/years', () => {
    expect(expiryValid(5, 2026, now)).toBe(false); // last month
    expect(expiryValid(1, 2020, now)).toBe(false);
    expect(expiryValid(0, 2027, now)).toBe(false);
    expect(expiryValid(13, 2027, now)).toBe(false);
    expect(expiryValid(6, 1999, now)).toBe(false);
  });
});

describe('cvcValid', () => {
  it('checks the digit length for the brand', () => {
    expect(cvcValid('123', 'visa')).toBe(true);
    expect(cvcValid('12', 'visa')).toBe(false);
    expect(cvcValid('1234', 'amex')).toBe(true);
    expect(cvcValid('123', 'amex')).toBe(false);
  });
});
