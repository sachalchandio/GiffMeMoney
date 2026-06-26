/**
 * RiskControls + SyntheticDataBanner behaviour.
 *
 * These cover the new post-buy loss-control UI and the advisor honesty banner.
 * RiskControls must: load the stored policy, let a rule be toggled on/off (which
 * gates its percent input), persist via `PUT /api/portfolio/risk` on Save, and
 * surface the triggered protective actions returned by
 * `POST /api/portfolio/risk/apply`. The SyntheticDataBanner must always disclose
 * synthetic data and prominently show an infeasible-target warning when present.
 *
 * Everything here is a SIMULATION on synthetic data — the panel never implies a
 * guaranteed profit; we assert the honesty disclaimer renders.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from '@/theme/ThemeProvider';
import type { RiskApplyResult, RiskPolicy } from '@/lib/types';

/* ---- Mock the typed api client (the three risk methods the hooks call) ---- */
const getRiskPolicy = vi.fn<() => Promise<RiskPolicy>>();
const setRiskPolicy = vi.fn<(p: RiskPolicy) => Promise<RiskPolicy>>();
const applyRisk = vi.fn<() => Promise<RiskApplyResult>>();

vi.mock('@/lib/api', () => ({
  api: {
    getRiskPolicy: () => getRiskPolicy(),
    setRiskPolicy: (p: RiskPolicy) => setRiskPolicy(p),
    applyRisk: () => applyRisk(),
  },
}));

import { RiskControls } from '@/components/domain/RiskControls';
import { SyntheticDataBanner } from '@/components/domain/SyntheticDataBanner';

const OFF_POLICY: RiskPolicy = {
  stopLossPct: null,
  trailingStopPct: null,
  takeProfitPct: null,
  maxDrawdownPct: null,
};

function renderControls(positionsCount = 1): void {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(
    <QueryClientProvider client={client}>
      <ThemeProvider>
        <RiskControls positionsCount={positionsCount} />
      </ThemeProvider>
    </QueryClientProvider>,
  );
}

describe('RiskControls', () => {
  beforeEach(() => {
    getRiskPolicy.mockResolvedValue(OFF_POLICY);
    setRiskPolicy.mockImplementation((p) => Promise.resolve(p));
    applyRisk.mockResolvedValue({
      actions: [],
      policy: OFF_POLICY,
      state: {
        wallet: {
          accountId: 'demo',
          cashBalance: 0,
          investedValue: 0,
          totalValue: 0,
          currency: 'USD',
          savedCards: [],
        },
        positions: [],
        totalCost: 0,
        totalValue: 0,
        totalPnl: 0,
        totalPnlPct: 0,
      },
      triggered: false,
      disclaimer:
        'Educational simulation on synthetic market data — not financial advice. Risk controls are mechanical, after-the-fact exits; they do not guarantee a profit or prevent loss.',
    });
  });

  it('shows all four rules disabled by default and the honesty disclaimer', async () => {
    renderControls();
    expect(await screen.findByText('Risk protections')).toBeInTheDocument();
    // All four rule inputs start disabled (policy is OFF).
    const stopInput = screen.getByLabelText(/stop-loss threshold percent/i) as HTMLInputElement;
    expect(stopInput).toBeDisabled();
    expect(screen.getByText('Off')).toBeInTheDocument();
    expect(screen.getByText(/do not guarantee a profit or prevent loss/i)).toBeInTheDocument();
  });

  it('toggling a rule enables its input and Save persists the policy', async () => {
    const user = userEvent.setup();
    renderControls();
    await screen.findByText('Risk protections');

    // Toggle stop-loss on → input enabled + a default value applied.
    await user.click(screen.getByRole('switch', { name: /enable stop-loss/i }));
    const stopInput = screen.getByLabelText(/stop-loss threshold percent/i) as HTMLInputElement;
    expect(stopInput).not.toBeDisabled();
    expect(stopInput.value).toBe('10');
    // Status flips to "Armed" once a rule is on.
    expect(screen.getByText('Armed')).toBeInTheDocument();

    // Edit the threshold and Save → PUT receives the numeric policy.
    await user.clear(stopInput);
    await user.type(stopInput, '12');
    await user.click(screen.getByRole('button', { name: /save policy/i }));

    await waitFor(() => expect(setRiskPolicy).toHaveBeenCalledTimes(1));
    expect(setRiskPolicy).toHaveBeenCalledWith(
      expect.objectContaining({ stopLossPct: 12, trailingStopPct: null }),
    );
  });

  it('Apply protections surfaces the triggered actions returned by the API', async () => {
    const user = userEvent.setup();
    applyRisk.mockResolvedValueOnce({
      actions: [
        {
          symbol: 'AAPL',
          action: 'stop_loss',
          reason: 'down 12% from entry; stop-loss is 10%',
          amount: 88,
          unitsSold: 0.8,
          price: 110,
          realizedPnl: -12,
        },
      ],
      policy: { ...OFF_POLICY, stopLossPct: 10 },
      state: {
        wallet: {
          accountId: 'demo',
          cashBalance: 88,
          investedValue: 0,
          totalValue: 88,
          currency: 'USD',
          savedCards: [],
        },
        positions: [],
        totalCost: 0,
        totalValue: 0,
        totalPnl: 0,
        totalPnlPct: 0,
      },
      triggered: true,
      disclaimer: 'Educational simulation — do not guarantee a profit or prevent loss.',
    });

    renderControls(1);
    await screen.findByText('Risk protections');
    // Arm a rule so Apply is enabled.
    await user.click(screen.getByRole('switch', { name: /enable stop-loss/i }));
    await user.click(screen.getByRole('button', { name: /apply protections/i }));

    expect(await screen.findByText(/1 protection triggered/i)).toBeInTheDocument();
    expect(screen.getByText('AAPL')).toBeInTheDocument();
    expect(applyRisk).toHaveBeenCalledTimes(1);
  });

  it('disables Apply when there are no positions', async () => {
    renderControls(0);
    await screen.findByText('Risk protections');
    expect(screen.getByRole('button', { name: /apply protections/i })).toBeDisabled();
    expect(screen.getByText(/add a holding to apply protections/i)).toBeInTheDocument();
  });
});

describe('SyntheticDataBanner', () => {
  it('renders the synthetic-data honesty note when synthetic is true', () => {
    render(<SyntheticDataBanner synthetic />);
    expect(screen.getByText(/synthetic data/i)).toBeInTheDocument();
    expect(screen.getByText(/no guarantee of profit/i)).toBeInTheDocument();
  });

  it('renders an infeasible-target warning prominently', () => {
    render(
      <SyntheticDataBanner synthetic targetWarning="Reaching $1M from $20 in 7 days is not realistic." />,
    );
    expect(screen.getByRole('alert')).toHaveTextContent(/not realistic/i);
  });

  it('renders nothing when there is neither synthetic data nor a warning', () => {
    const { container } = render(<SyntheticDataBanner synthetic={false} targetWarning={null} />);
    expect(container).toBeEmptyDOMElement();
  });
});
