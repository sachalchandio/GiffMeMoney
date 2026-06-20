/**
 * React-query mutation hook for the analytical Markowitz optimizer
 * (efficient frontier). This is the *analytical* endpoint
 * (`POST /api/portfolio/optimize`) — distinct from the invest portfolio.
 */

import { useMutation, type UseMutationResult } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { PortfolioRequest, PortfolioResult } from '@/lib/types';

/** Run a mean-variance optimization and return the frontier + tangency result. */
export function usePortfolioOpt(): UseMutationResult<PortfolioResult, Error, PortfolioRequest> {
  return useMutation<PortfolioResult, Error, PortfolioRequest>({
    mutationFn: (req: PortfolioRequest) => api.optimizePortfolio(req),
  });
}
