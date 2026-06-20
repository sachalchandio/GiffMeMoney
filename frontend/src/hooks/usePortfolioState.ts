/**
 * React-query hooks for the invest portfolio state plus the invest / sell
 * mutations. On success they invalidate the portfolio, wallet, transactions,
 * and history so live views reconcile.
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
import type { InvestRequest, PortfolioState, SellRequest } from '@/lib/types';

/** The mark-to-market portfolio state (positions + totals). */
export function usePortfolioState(): UseQueryResult<PortfolioState> {
  return useQuery({
    queryKey: queryKeys.portfolioState,
    queryFn: () => api.getPortfolioState(),
    staleTime: 10_000,
  });
}

/** Invalidate portfolio + wallet + transactions + (any) history queries. */
function useInvalidatePortfolio(): () => Promise<void> {
  const qc = useQueryClient();
  return async () => {
    await Promise.all([
      qc.invalidateQueries({ queryKey: queryKeys.portfolioState }),
      qc.invalidateQueries({ queryKey: queryKeys.wallet }),
      qc.invalidateQueries({ queryKey: queryKeys.transactions }),
      qc.invalidateQueries({ queryKey: ['portfolioHistory'] }),
    ]);
  };
}

/** Spend cash across one or more symbols. */
export function useInvest(): UseMutationResult<PortfolioState, Error, InvestRequest> {
  const invalidate = useInvalidatePortfolio();
  return useMutation<PortfolioState, Error, InvestRequest>({
    mutationFn: (req: InvestRequest) => api.invest(req),
    onSuccess: () => invalidate(),
  });
}

/** Reduce or liquidate a position. */
export function useSell(): UseMutationResult<PortfolioState, Error, SellRequest> {
  const invalidate = useInvalidatePortfolio();
  return useMutation<PortfolioState, Error, SellRequest>({
    mutationFn: (req: SellRequest) => api.sell(req),
    onSuccess: () => invalidate(),
  });
}
