/**
 * React-query hooks for the simulated auto-trader (paper-trading bot).
 *
 * HONESTY / SAFETY: every result is a SIMULATION on synthetic data — paper-traded,
 * no real money moves, no live broker. Rotation is momentum / bandit style (more
 * to recent winners, less to losers) and never martingale. See {@link useBotRun}.
 *
 * - {@link useBotModes}    — the five preset modes (cached, GET).
 * - {@link useBotRun}      — backtest one mode (mutation).
 * - {@link useBotCompare}  — run several modes side-by-side (mutation).
 */

import {
  useMutation,
  useQuery,
  type UseMutationResult,
  type UseQueryResult,
} from '@tanstack/react-query';
import { api } from '@/lib/api';
import { queryKeys } from './queryKeys';
import type {
  BotCompareRequest,
  BotMode,
  BotRunRequest,
  BotRunResult,
} from '@/lib/types';

/** Fetch the preset auto-trader modes (rarely change → long stale time). */
export function useBotModes(): UseQueryResult<BotMode[]> {
  return useQuery({
    queryKey: queryKeys.botModes,
    queryFn: () => api.getBotModes(),
    staleTime: 5 * 60_000,
  });
}

/** Run one simulated backtest for a given config. */
export function useBotRun(): UseMutationResult<BotRunResult, Error, BotRunRequest> {
  return useMutation<BotRunResult, Error, BotRunRequest>({
    mutationFn: (req: BotRunRequest) => api.runBotBacktest(req),
  });
}

/** Run several modes against one shared base config for side-by-side compare. */
export function useBotCompare(): UseMutationResult<BotRunResult[], Error, BotCompareRequest> {
  return useMutation<BotRunResult[], Error, BotCompareRequest>({
    mutationFn: (body: BotCompareRequest) => api.compareBotModes(body),
  });
}
