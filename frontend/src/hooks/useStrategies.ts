/**
 * React-query hooks for the strategy catalog, per-strategy rankings, and the
 * per-asset strategy leaderboard.
 */

import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { queryKeys } from './queryKeys';
import type { StrategyLeaderboard, StrategyMeta, StrategyRanking } from '@/lib/types';

/** The full quant model catalog (~73 metas). */
export function useStrategies(): UseQueryResult<StrategyMeta[]> {
  return useQuery({
    queryKey: queryKeys.strategies,
    queryFn: () => api.listStrategies(),
    staleTime: Infinity,
  });
}

/** Cross-asset rankings for a single strategy. */
export function useStrategyRankings(
  id: string | undefined,
  limit?: number,
): UseQueryResult<StrategyRanking> {
  return useQuery({
    queryKey: queryKeys.strategyRankings(id ?? '', limit),
    queryFn: () => api.getStrategyRankings(id as string, limit),
    enabled: Boolean(id),
    staleTime: 30_000,
  });
}

/** Per-asset leaderboard: strategies ranked by realized backtest performance. */
export function useLeaderboard(
  symbol: string | undefined,
  limit?: number,
): UseQueryResult<StrategyLeaderboard> {
  return useQuery({
    queryKey: queryKeys.leaderboard(symbol ?? '', limit),
    queryFn: () => api.getLeaderboard(symbol as string, limit),
    enabled: Boolean(symbol),
    staleTime: 60_000,
  });
}
