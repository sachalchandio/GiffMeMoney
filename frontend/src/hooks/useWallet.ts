/**
 * React-query hooks for the simulated wallet: balance, saved cards,
 * transactions, and the deposit / withdraw / delete-card mutations. Mutations
 * invalidate the wallet, cards, transactions, and portfolio state so every view
 * reconciles after a money movement.
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
import type {
  DepositRequest,
  SavedCard,
  Transaction,
  Wallet,
  WalletTxnResponse,
  WithdrawRequest,
} from '@/lib/types';

/** The current wallet (cash + invested + total + saved cards). */
export function useWallet(): UseQueryResult<Wallet> {
  return useQuery({
    queryKey: queryKeys.wallet,
    queryFn: () => api.getWallet(),
    staleTime: 10_000,
  });
}

/** The account's masked saved cards. */
export function useCards(): UseQueryResult<SavedCard[]> {
  return useQuery({
    queryKey: queryKeys.cards,
    queryFn: () => api.getCards(),
    staleTime: 30_000,
  });
}

/** The account ledger (newest first). */
export function useTransactions(): UseQueryResult<Transaction[]> {
  return useQuery({
    queryKey: queryKeys.transactions,
    queryFn: () => api.getTransactions(),
    staleTime: 10_000,
  });
}

/** Invalidate every money-affected query (wallet, cards, txns, portfolio). */
function useInvalidateWallet(): () => Promise<void> {
  const qc = useQueryClient();
  return async () => {
    await Promise.all([
      qc.invalidateQueries({ queryKey: queryKeys.wallet }),
      qc.invalidateQueries({ queryKey: queryKeys.cards }),
      qc.invalidateQueries({ queryKey: queryKeys.transactions }),
      qc.invalidateQueries({ queryKey: queryKeys.portfolioState }),
    ]);
  };
}

/** Deposit funds (simulated card charge). */
export function useDeposit(): UseMutationResult<WalletTxnResponse, Error, DepositRequest> {
  const invalidate = useInvalidateWallet();
  return useMutation<WalletTxnResponse, Error, DepositRequest>({
    mutationFn: (req: DepositRequest) => api.deposit(req),
    onSuccess: () => invalidate(),
  });
}

/** Withdraw cash (simulated payout). */
export function useWithdraw(): UseMutationResult<WalletTxnResponse, Error, WithdrawRequest> {
  const invalidate = useInvalidateWallet();
  return useMutation<WalletTxnResponse, Error, WithdrawRequest>({
    mutationFn: (req: WithdrawRequest) => api.withdraw(req),
    onSuccess: () => invalidate(),
  });
}

/** Delete a saved card. */
export function useDeleteCard(): UseMutationResult<{ ok: boolean }, Error, string> {
  const invalidate = useInvalidateWallet();
  return useMutation<{ ok: boolean }, Error, string>({
    mutationFn: (id: string) => api.deleteCard(id),
    onSuccess: () => invalidate(),
  });
}
