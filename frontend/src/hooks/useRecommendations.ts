/**
 * React-query hooks for ranked recommendations + the market summary.
 */

import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { queryKeys } from './queryKeys';
import type { AssetClass, MarketSummary, Recommendation } from '@/lib/types';

/** Ranked recommendations across the universe (optionally filtered by class). */
export function useRecommendations(
  limit?: number,
  assetClass?: AssetClass,
): UseQueryResult<Recommendation[]> {
  return useQuery({
    queryKey: queryKeys.recommendations(limit, assetClass),
    queryFn: () => api.getRecommendations(limit, assetClass),
    staleTime: 30_000,
  });
}

/** Dashboard market summary: breadth, movers, sectors, indices. */
export function useMarketSummary(): UseQueryResult<MarketSummary> {
  return useQuery({
    queryKey: queryKeys.marketSummary,
    queryFn: () => api.getMarketSummary(),
    staleTime: 30_000,
    refetchInterval: 60_000,
  });
}
