/**
 * React-query hook for the backfilled portfolio value / P&L history (seed for
 * the real-time charts).
 */

import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { queryKeys } from './queryKeys';
import type { PortfolioHistory } from '@/lib/types';

/** Backfilled total + per-position value/P&L series. */
export function usePortfolioHistory(points?: number): UseQueryResult<PortfolioHistory> {
  return useQuery({
    queryKey: queryKeys.portfolioHistory(points),
    queryFn: () => api.getPortfolioHistory(points),
    staleTime: 30_000,
  });
}
