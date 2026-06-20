/**
 * RecommendationsPage render test — drives the page through a mocked api client
 * and asserts the ranked picks render, the rank-1 pick is expanded with its
 * reasons, and the class filter re-queries the api.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider } from '@/theme/ThemeProvider';
import type { Asset, Candle, Recommendation, Stance } from '@/lib/types';

/* ---- Mock the typed api client ---- */
const getRecommendations =
  vi.fn<(limit?: number, assetClass?: string) => Promise<Recommendation[]>>();
const getCandles = vi.fn<(symbol: string, interval: string, limit: number) => Promise<Candle[]>>();

vi.mock('@/lib/api', () => ({
  api: {
    getRecommendations: (limit?: number, assetClass?: string) =>
      getRecommendations(limit, assetClass),
    getCandles: (symbol: string, interval: string, limit: number) =>
      getCandles(symbol, interval, limit),
  },
}));

/* ---- Import after the mock is registered ---- */
import RecommendationsPage from '@/pages/RecommendationsPage';

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

function makeRec(rank: number, symbol: string, recommendation: Stance): Recommendation {
  return {
    rank,
    asset: makeAsset(symbol, `${symbol} Inc.`),
    compositeScore: 70 - rank * 10,
    recommendation,
    confidence: 0.8,
    expectedReturn1YPct: 18 - rank,
    headline: `${symbol} headline for rank ${rank}`,
    reasons: [`${symbol} reason one`, `${symbol} reason two`],
  };
}

function renderPage(): void {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  render(
    <QueryClientProvider client={client}>
      <ThemeProvider>
        <MemoryRouter>
          <RecommendationsPage />
        </MemoryRouter>
      </ThemeProvider>
    </QueryClientProvider>,
  );
}

describe('RecommendationsPage', () => {
  beforeEach(() => {
    getRecommendations.mockResolvedValue([
      makeRec(1, 'AAA', 'STRONG_BUY'),
      makeRec(2, 'BBB', 'BUY'),
      makeRec(3, 'CCC', 'HOLD'),
    ]);
    getCandles.mockResolvedValue([
      { t: 1, o: 1, h: 1, l: 1, c: 100, v: 1 },
      { t: 2, o: 1, h: 1, l: 1, c: 105, v: 1 },
    ]);
  });

  it('renders the ranked recommendations', async () => {
    renderPage();
    expect(await screen.findByText('AAA')).toBeInTheDocument();
    expect(screen.getByText('BBB')).toBeInTheDocument();
    expect(screen.getByText('CCC')).toBeInTheDocument();
    // Tone summary chips.
    expect(screen.getByText('2 buy')).toBeInTheDocument();
  });

  it('expands the rank-1 pick by default with its reasons', async () => {
    renderPage();
    await screen.findByText('AAA');
    expect(screen.getByText('AAA headline for rank 1')).toBeInTheDocument();
    expect(screen.getByText('AAA reason one')).toBeInTheDocument();
  });

  it('re-queries the api when the asset-class filter changes', async () => {
    renderPage();
    await screen.findByText('AAA');
    const user = userEvent.setup();
    const tablist = screen.getByRole('tablist', { name: /filter recommendations/i });
    await user.click(within(tablist).getByRole('tab', { name: 'Crypto' }));
    await waitFor(() => {
      expect(getRecommendations).toHaveBeenCalledWith(50, 'crypto');
    });
  });
});
