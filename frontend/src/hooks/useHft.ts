/**
 * React-query hooks for the High-Frequency Simulation Lab (paper-only).
 *
 * HONESTY / SAFETY: every result is a SIMULATION on synthetic data. The lab
 * shows — truthfully — that trading faster / in smaller portions usually makes
 * LESS money once costs are charged, and it is explicit that a web app cannot
 * trade in microseconds. No real money moves.
 *
 * - {@link useHftCostPresets} — the transaction-cost presets (cached, GET).
 * - {@link useHftSim}         — run one short-horizon simulation (mutation).
 * - {@link useHftSweep}       — run the turnover sweep (mutation).
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
  HftCostModel,
  HftSimRequest,
  HftSimResult,
  HftSweepRequest,
  HftSweepResult,
} from '@/lib/types';

/** Fetch the transaction-cost presets (rarely change → long stale time). */
export function useHftCostPresets(): UseQueryResult<HftCostModel[]> {
  return useQuery({
    queryKey: queryKeys.hftCostPresets,
    queryFn: () => api.getHftCostPresets(),
    staleTime: 10 * 60_000,
  });
}

/** Run one short-horizon simulation (gross vs net vs buy-&-hold). */
export function useHftSim(): UseMutationResult<HftSimResult, Error, HftSimRequest> {
  return useMutation<HftSimResult, Error, HftSimRequest>({
    mutationFn: (req: HftSimRequest) => api.runHftSim(req),
  });
}

/** Run the turnover sweep and get the curve + net-of-cost optimum + verdict. */
export function useHftSweep(): UseMutationResult<HftSweepResult, Error, HftSweepRequest> {
  return useMutation<HftSweepResult, Error, HftSweepRequest>({
    mutationFn: (body: HftSweepRequest) => api.runHftSweep(body),
  });
}
