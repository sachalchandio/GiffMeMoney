/**
 * BrokerStatusBadge tests (go-live UI safety matrix).
 *
 * The badge must, from `/api/broker/status` alone, make two things unambiguous:
 *  1. whether the market DATA is Simulated or Live, and
 *  2. whether the BROKER is in Paper (safe) or LIVE (real-money) mode.
 *
 * SAFETY: paper / simulated must read as a calm, safe chip (success/neutral),
 * while LIVE must read as a loud RED WARNING (danger) — a real-money mode can
 * never visually resemble the safe default. The badge is display-only; there is
 * deliberately no "go live" control. These tests pin that contract.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { UseQueryResult } from '@tanstack/react-query';
import type { BrokerStatus } from '@/lib/types';

/* ---- Mock the broker hook so the badge renders against a fixed status ---- */
const useBrokerStatus = vi.fn<() => UseQueryResult<BrokerStatus>>();

vi.mock('@/hooks/useBroker', () => ({
  useBrokerStatus: () => useBrokerStatus(),
}));

import { BrokerStatusBadge } from '@/components/domain/BrokerStatusBadge';

function makeStatus(over: Partial<BrokerStatus> = {}): BrokerStatus {
  return {
    broker: 'simulated',
    mode: 'simulated',
    paper: true,
    connected: true,
    liveEnabled: false,
    baseUrl: null,
    message: null,
    disclaimer: 'Simulated / paper trading — no real money moves.',
    ...over,
  };
}

/** Build a minimal "success" query result carrying `data`. */
function ok(status: BrokerStatus): UseQueryResult<BrokerStatus> {
  return {
    data: status,
    isPending: false,
    isError: false,
    isSuccess: true,
  } as unknown as UseQueryResult<BrokerStatus>;
}

/** The badge chip element whose text contains the given fragment. */
function chip(fragment: RegExp): HTMLElement {
  return screen.getByText(fragment).closest('span') as HTMLElement;
}

beforeEach(() => {
  useBrokerStatus.mockReset();
});

describe('BrokerStatusBadge', () => {
  it('shows Data: Simulated + Broker: Paper for the default simulated broker', () => {
    useBrokerStatus.mockReturnValue(ok(makeStatus()));
    render(<BrokerStatusBadge />);

    expect(screen.getByText(/Data: Simulated/)).toBeInTheDocument();
    expect(screen.getByText(/Broker: Paper/)).toBeInTheDocument();
    // Safe broker chip is success-toned (NOT a danger warning).
    const broker = chip(/Broker: Paper/);
    expect(broker.className).toContain('text-success');
    expect(broker.className).not.toContain('bg-danger');
  });

  it('reads Data: Live for a paper Alpaca broker but keeps Broker: Paper (safe)', () => {
    // Alpaca paper sandbox: real market data, but still no real money.
    useBrokerStatus.mockReturnValue(
      ok(makeStatus({ broker: 'alpaca', mode: 'paper', paper: true, liveEnabled: false })),
    );
    render(<BrokerStatusBadge />);

    expect(screen.getByText(/Data: Live/)).toBeInTheDocument();
    expect(screen.getByText(/Broker: Paper/)).toBeInTheDocument();
    expect(chip(/Broker: Paper/).className).toContain('text-success');
  });

  it('renders Broker: LIVE as a loud RED WARNING when live is fully enabled', () => {
    useBrokerStatus.mockReturnValue(
      ok(makeStatus({ broker: 'alpaca', mode: 'live', paper: false, liveEnabled: true })),
    );
    render(<BrokerStatusBadge />);

    expect(screen.getByText(/Data: Live/)).toBeInTheDocument();
    const broker = chip(/Broker: LIVE/);
    // The solid danger badge uses a red background — clearly distinct from paper.
    expect(broker.className).toContain('bg-danger');
    expect(broker.className).not.toContain('text-success');
  });

  it('renders nothing while loading or on error (fail-quiet)', () => {
    useBrokerStatus.mockReturnValue({
      data: undefined,
      isPending: true,
      isError: false,
      isSuccess: false,
    } as unknown as UseQueryResult<BrokerStatus>);
    const { container, rerender } = render(<BrokerStatusBadge />);
    expect(container.firstChild).toBeNull();

    useBrokerStatus.mockReturnValue({
      data: undefined,
      isPending: false,
      isError: true,
      isSuccess: false,
    } as unknown as UseQueryResult<BrokerStatus>);
    rerender(<BrokerStatusBadge />);
    expect(container.firstChild).toBeNull();
  });
});
