/**
 * React-query hooks for backtests (strategy×asset), exposed from both the
 * asset and the strategy entry points (the two backend routes return the same
 * `BacktestResultDTO`).
 */

import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { queryKeys } from './queryKeys';
import type { BacktestResultDTO } from '@/lib/types';

/** Backtest a strategy on an asset via the asset route (`/assets/{s}/backtest`). */
export function useBacktest(
  symbol: string | undefined,
  strategyId: string | undefined,
): UseQueryResult<BacktestResultDTO> {
  return useQuery({
    queryKey: queryKeys.assetBacktest(symbol ?? '', strategyId ?? ''),
    queryFn: () => api.getBacktest(symbol as string, strategyId as string),
    enabled: Boolean(symbol) && Boolean(strategyId),
    staleTime: 60_000,
  });
}

/** Backtest via the strategy route (`/strategies/{id}/backtest?symbol=`). */
export function useStrategyBacktest(
  strategyId: string | undefined,
  symbol: string | undefined,
): UseQueryResult<BacktestResultDTO> {
  return useQuery({
    queryKey: queryKeys.strategyBacktest(strategyId ?? '', symbol ?? ''),
    queryFn: () => api.getStrategyBacktest(strategyId as string, symbol as string),
    enabled: Boolean(strategyId) && Boolean(symbol),
    staleTime: 60_000,
  });
}
