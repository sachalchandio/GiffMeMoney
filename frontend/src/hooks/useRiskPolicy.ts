/**
 * React-query hooks for the account's post-buy loss-control policy.
 *
 * - `useRiskPolicy()` reads the stored policy (`GET /api/portfolio/risk`).
 * - `useSetRiskPolicy()` replaces it (`PUT /api/portfolio/risk`).
 * - `useApplyRisk()` evaluates it and takes any protective sells
 *   (`POST /api/portfolio/risk/apply`), then invalidates the portfolio /
 *   wallet / transactions / history queries so the live views reconcile.
 *
 * HONESTY: these are mechanical, after-the-fact exits on a SIMULATION over
 * synthetic data — applying a policy does not guarantee a profit or prevent
 * loss. The UI must never imply otherwise.
 */

import {
  useMutation,
  useQuery,
  useQueryClient,
  type UseMutationResult,
  type UseQueryResult,
} from '@tanstack/react-query';
import { api } from '@/lib/api';
import { queryKeys } from './queryKeys';
import type { RiskApplyResult, RiskPolicy } from '@/lib/types';

/** The account's stored post-buy loss-control policy (all-OFF by default). */
export function useRiskPolicy(): UseQueryResult<RiskPolicy> {
  return useQuery({
    queryKey: queryKeys.riskPolicy,
    queryFn: () => api.getRiskPolicy(),
    staleTime: 30_000,
  });
}

/** Replace the account's risk policy; refreshes the cached policy on success. */
export function useSetRiskPolicy(): UseMutationResult<RiskPolicy, Error, RiskPolicy> {
  const qc = useQueryClient();
  return useMutation<RiskPolicy, Error, RiskPolicy>({
    mutationFn: (policy: RiskPolicy) => api.setRiskPolicy(policy),
    onSuccess: (saved) => {
      qc.setQueryData(queryKeys.riskPolicy, saved);
    },
  });
}

/**
 * Evaluate the policy and take any protective actions. Because protective sells
 * mutate the portfolio + wallet + ledger, invalidate those queries on success.
 */
export function useApplyRisk(): UseMutationResult<RiskApplyResult, Error, void> {
  const qc = useQueryClient();
  return useMutation<RiskApplyResult, Error, void>({
    mutationFn: () => api.applyRisk(),
    onSuccess: async (result) => {
      qc.setQueryData(queryKeys.riskPolicy, result.policy);
      await Promise.all([
        qc.invalidateQueries({ queryKey: queryKeys.portfolioState }),
        qc.invalidateQueries({ queryKey: queryKeys.wallet }),
        qc.invalidateQueries({ queryKey: queryKeys.transactions }),
        qc.invalidateQueries({ queryKey: ['portfolioHistory'] }),
      ]);
    },
  });
}
