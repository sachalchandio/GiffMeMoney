/**
 * React-query hooks for the asset universe + per-asset snapshot/candles.
 */

import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { queryKeys } from './queryKeys';
import type { Asset, AssetClass, Candle, HealthStatus } from '@/lib/types';

/** List the universe (optionally filtered by asset class). */
export function useAssets(assetClass?: AssetClass): UseQueryResult<Asset[]> {
  return useQuery({
    queryKey: queryKeys.assets(assetClass),
    queryFn: () => api.listAssets(assetClass),
    staleTime: 30_000,
  });
}

/** A single asset snapshot. */
export function useAsset(symbol: string | undefined): UseQueryResult<Asset> {
  return useQuery({
    queryKey: queryKeys.asset(symbol ?? ''),
    queryFn: () => api.getAsset(symbol as string),
    enabled: Boolean(symbol),
    staleTime: 15_000,
  });
}

/** OHLCV candles for an asset. */
export function useCandles(
  symbol: string | undefined,
  interval = '1d',
  limit = 365,
): UseQueryResult<Candle[]> {
  return useQuery({
    queryKey: queryKeys.candles(symbol ?? '', interval, limit),
    queryFn: () => api.getCandles(symbol as string, interval, limit),
    enabled: Boolean(symbol),
    staleTime: 60_000,
  });
}

/** Backend health probe (also reports the universe size). */
export function useHealth(): UseQueryResult<HealthStatus> {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => api.getHealth(),
    staleTime: 60_000,
    refetchInterval: 60_000,
  });
}
