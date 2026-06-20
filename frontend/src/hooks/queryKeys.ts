/**
 * Centralized react-query key factory so hooks + mutations invalidate
 * consistently. Keys are readonly tuples — stable + serializable.
 */

import type { AssetClass, Horizon } from '@/lib/types';

export const queryKeys = {
  health: ['health'] as const,

  assets: (assetClass?: AssetClass) => ['assets', assetClass ?? 'all'] as const,
  asset: (symbol: string) => ['asset', symbol.toUpperCase()] as const,
  candles: (symbol: string, interval: string, limit: number) =>
    ['candles', symbol.toUpperCase(), interval, limit] as const,
  analysis: (symbol: string) => ['analysis', symbol.toUpperCase()] as const,
  monteCarlo: (symbol: string, horizon: Horizon, sims: number) =>
    ['montecarlo', symbol.toUpperCase(), horizon, sims] as const,

  recommendations: (limit?: number, assetClass?: AssetClass) =>
    ['recommendations', limit ?? 'default', assetClass ?? 'all'] as const,

  marketSummary: ['marketSummary'] as const,

  strategies: ['strategies'] as const,
  strategyRankings: (id: string, limit?: number) =>
    ['strategyRankings', id, limit ?? 'default'] as const,
  strategyBacktest: (id: string, symbol: string) =>
    ['strategyBacktest', id, symbol.toUpperCase()] as const,
  assetBacktest: (symbol: string, strategyId: string) =>
    ['assetBacktest', symbol.toUpperCase(), strategyId] as const,
  leaderboard: (symbol: string, limit?: number) =>
    ['leaderboard', symbol.toUpperCase(), limit ?? 'default'] as const,

  // invest
  wallet: ['wallet'] as const,
  cards: ['cards'] as const,
  transactions: ['transactions'] as const,
  portfolioState: ['portfolioState'] as const,
  portfolioHistory: (points?: number) => ['portfolioHistory', points ?? 'default'] as const,
} as const;
