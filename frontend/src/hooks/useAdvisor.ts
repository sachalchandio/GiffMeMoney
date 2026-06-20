/**
 * React-query mutation hook for the allocation advisor
 * (`POST /api/advisor/allocate`) — the "where to invest now" basket.
 */

import { useMutation, type UseMutationResult } from '@tanstack/react-query';
import { api } from '@/lib/api';
import type { AdviceRequest, AllocationAdvice } from '@/lib/types';

/** Request a Markowitz-sized basket for an amount + risk tolerance. */
export function useAdvisor(): UseMutationResult<AllocationAdvice, Error, AdviceRequest> {
  return useMutation<AllocationAdvice, Error, AdviceRequest>({
    mutationFn: (req: AdviceRequest) => api.advise(req),
  });
}
