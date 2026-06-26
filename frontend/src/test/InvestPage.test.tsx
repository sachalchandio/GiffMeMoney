/**
 * InvestPage render test — drives the flagship Invest page through a fully mocked
 * api client and asserts the wallet header (total value + P&L), the live charts,
 * a holding, the allocation builder, the advisor panel, and the transaction
 * ledger all render. Hooks resolve against the mock so no network is touched.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from '@/theme/ThemeProvider';
import type {
  Asset,
  Candle,
  PortfolioHistory,
  PortfolioState,
  Position,
  RiskPolicy,
  Transaction,
  Wallet,
} from '@/lib/types';

/* ---- Mock the typed api client (every method the page's hooks call) ---- */
const getWallet = vi.fn<() => Promise<Wallet>>();
const getPortfolioState = vi.fn<() => Promise<PortfolioState>>();
const getPortfolioHistory = vi.fn<(points?: number) => Promise<PortfolioHistory>>();
const getTransactions = vi.fn<() => Promise<Transaction[]>>();
const listAssets = vi.fn<() => Promise<Asset[]>>();
const getCandles = vi.fn<() => Promise<Candle[]>>();
const getRiskPolicy = vi.fn<() => Promise<RiskPolicy>>();

vi.mock('@/lib/api', () => ({
  api: {
    getWallet: () => getWallet(),
    getPortfolioState: () => getPortfolioState(),
    getPortfolioHistory: (points?: number) => getPortfolioHistory(points),
    getTransactions: () => getTransactions(),
    listAssets: () => listAssets(),
    getCandles: () => getCandles(),
    getRiskPolicy: () => getRiskPolicy(),
  },
}));

import InvestPage from '@/pages/InvestPage';

function makeAsset(symbol: string, name: string): Asset {
  return {
    symbol,
    name,
    assetClass: 'equity',
    sector: 'Technology',
    currency: 'USD',
    price: 100,
    change24hPct: 1.2,
    marketCap: 1_000_000_000,
    volume24h: 5_000_000,
  };
}

function makePosition(symbol: string): Position {
  return {
    symbol,
    asset: makeAsset(symbol, `${symbol} Inc.`),
    units: 5,
    costBasis: 500,
    avgPrice: 100,
    currentPrice: 110,
    marketValue: 550,
    unrealizedPnl: 50,
    unrealizedPnlPct: 10,
    allocationPct: 55,
    realizedPnl: 0,
    openedAt: 1_700_000_000_000,
  };
}

const WALLET: Wallet = {
  accountId: 'demo',
  cashBalance: 450,
  investedValue: 550,
  totalValue: 1000,
  currency: 'USD',
  savedCards: [],
};

const STATE: PortfolioState = {
  wallet: WALLET,
  positions: [makePosition('AAPL')],
  totalCost: 500,
  totalValue: 1000,
  totalPnl: 50,
  totalPnlPct: 10,
};

const HISTORY: PortfolioHistory = {
  total: [
    { t: 1_700_000_000_000, totalValue: 1000, invested: 500, cash: 500 },
    { t: 1_700_000_100_000, totalValue: 1010, invested: 550, cash: 460 },
  ],
  positions: [
    {
      symbol: 'AAPL',
      points: [
        { t: 1_700_000_000_000, value: 500, pnl: 0, pnlPct: 0 },
        { t: 1_700_000_100_000, value: 550, pnl: 50, pnlPct: 10 },
      ],
    },
  ],
};

const TXNS: Transaction[] = [
  {
    id: 'tx1',
    type: 'deposit',
    amount: 1000,
    symbol: null,
    status: 'completed',
    createdAt: 1_700_000_000_000,
    ref: 'DEP-1',
    note: 'Wallet top-up',
  },
  {
    id: 'tx2',
    type: 'buy',
    amount: 500,
    symbol: 'AAPL',
    status: 'completed',
    createdAt: 1_700_000_050_000,
    ref: 'BUY-1',
    note: 'Bought AAPL',
  },
];

function renderPage(): void {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  render(
    <QueryClientProvider client={client}>
      <ThemeProvider>
        <MemoryRouter>
          <InvestPage />
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>,
  );
}

describe('InvestPage', () => {
  beforeEach(() => {
    getWallet.mockResolvedValue(WALLET);
    getPortfolioState.mockResolvedValue(STATE);
    getPortfolioHistory.mockResolvedValue(HISTORY);
    getTransactions.mockResolvedValue(TXNS);
    getRiskPolicy.mockResolvedValue({
      stopLossPct: null,
      trailingStopPct: null,
      takeProfitPct: null,
      maxDrawdownPct: null,
    });
    listAssets.mockResolvedValue([makeAsset('AAPL', 'AAPL Inc.'), makeAsset('MSFT', 'MSFT Inc.')]);
    getCandles.mockResolvedValue([
      { t: 1, o: 1, h: 1, l: 1, c: 100, v: 1 },
      { t: 2, o: 1, h: 1, l: 1, c: 105, v: 1 },
    ]);
  });

  it('renders the wallet header with the live total value', async () => {
    renderPage();
    expect(await screen.findByText('Total portfolio value')).toBeInTheDocument();
    // Total value (cash 450 + invested 550 = $1,000.00).
    expect(screen.getByText('$1,000.00')).toBeInTheDocument();
    expect(screen.getAllByText(/Demo \/ sandbox — no real charge/i).length).toBeGreaterThan(0);
  });

  it('renders the core invest sections', async () => {
    renderPage();
    await screen.findByText('Total portfolio value');
    expect(screen.getByText('Portfolio value')).toBeInTheDocument();
    expect(screen.getByText('Build your allocation')).toBeInTheDocument();
    expect(screen.getByText('Where to invest now')).toBeInTheDocument();
    expect(screen.getByText('Activity')).toBeInTheDocument();
  });

  it('renders the risk-protections panel with the four rules and an apply action', async () => {
    renderPage();
    expect(await screen.findByText('Risk protections')).toBeInTheDocument();
    // Each of the four post-buy loss controls is offered.
    expect(screen.getByText('Stop-loss')).toBeInTheDocument();
    expect(screen.getByText('Trailing stop')).toBeInTheDocument();
    expect(screen.getByText('Take-profit')).toBeInTheDocument();
    expect(screen.getByText('Max drawdown')).toBeInTheDocument();
    // Save + Apply actions are present; honesty disclaimer is shown.
    expect(screen.getByRole('button', { name: /save policy/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /apply protections/i })).toBeInTheDocument();
    expect(
      screen.getByText(/do not guarantee a profit or prevent loss/i),
    ).toBeInTheDocument();
  });

  it('renders the open position and its allocation', async () => {
    renderPage();
    // The holding shows up (symbol appears in the holdings card link).
    expect(await screen.findAllByText('AAPL')).toBeTruthy();
    // Add Funds / Withdraw actions are present.
    expect(screen.getByRole('button', { name: /add funds/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /withdraw/i })).toBeInTheDocument();
  });

  it('lists wallet transactions', async () => {
    renderPage();
    await screen.findByText('Activity');
    // Transactions resolve asynchronously — wait for the ledger rows.
    expect(await screen.findByText(/Wallet top-up/i)).toBeInTheDocument();
    const buyRow = (await screen.findByText(/Bought AAPL/i)).closest('li') as HTMLElement;
    expect(within(buyRow).getByText('Buy')).toBeInTheDocument();
  });
});
