/**
 * React-query hooks for per-asset composite analysis + Monte Carlo simulation.
 */

import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { queryKeys } from './queryKeys';
import type { AssetAnalysis, Horizon, MonteCarloResult } from '@/lib/types';

/** Full composite analysis (signals, horizons, regime, risk) for an asset. */
export function useAnalysis(symbol: string | undefined): UseQueryResult<AssetAnalysis> {
  return useQuery({
    queryKey: queryKeys.analysis(symbol ?? ''),
    queryFn: () => api.getAnalysis(symbol as string),
    enabled: Boolean(symbol),
    staleTime: 30_000,
  });
}

/** Monte Carlo simulation (price bands + final distribution) for an asset. */
export function useMonteCarlo(
  symbol: string | undefined,
  horizon: Horizon = '1Y',
  sims = 2000,
): UseQueryResult<MonteCarloResult> {
  return useQuery({
    queryKey: queryKeys.monteCarlo(symbol ?? '', horizon, sims),
    queryFn: () => api.getMonteCarlo(symbol as string, horizon, sims),
    enabled: Boolean(symbol),
    staleTime: 60_000,
  });
}
