/**
 * React-query hooks for the go-live broker layer (read-only / DISPLAY-ONLY).
 *
 * SAFETY / HONESTY: the broker ships in **simulated** mode (paper fills against
 * the market provider price; no real money). The real Alpaca adapter defaults to
 * Alpaca's PAPER endpoint; LIVE trading is hard-gated and OFF by default. These
 * hooks only READ the broker's status / account / positions / orders — there is
 * deliberately NO order-placement or "enable live trading" hook here. Enabling
 * real trading stays a documented, deliberate env/config action (docs/DEPLOY.md).
 *
 * - {@link useBrokerStatus}    — mode / paper / connectivity snapshot (badge).
 * - {@link useBrokerAccount}   — cash / equity / buying power.
 * - {@link useBrokerPositions} — open positions marked to the latest price.
 * - {@link useBrokerOrders}    — recorded orders (newest first).
 */

import { useQuery, type UseQueryResult } from '@tanstack/react-query';
import { api } from '@/lib/api';
import { queryKeys } from './queryKeys';
import type {
  BrokerAccount,
  BrokerOrder,
  BrokerPosition,
  BrokerStatus,
} from '@/lib/types';

/** The broker mode / paper flag / connectivity snapshot (drives the badge). */
export function useBrokerStatus(): UseQueryResult<BrokerStatus> {
  return useQuery({
    queryKey: queryKeys.brokerStatus,
    queryFn: () => api.getBrokerStatus(),
    staleTime: 30_000,
  });
}

/** The broker account summary (cash, equity, buying power). */
export function useBrokerAccount(
  enabled = true,
): UseQueryResult<BrokerAccount> {
  return useQuery({
    queryKey: queryKeys.brokerAccount,
    queryFn: () => api.getBrokerAccount(),
    staleTime: 15_000,
    enabled,
  });
}

/** Open broker positions marked to the latest price. */
export function useBrokerPositions(
  enabled = true,
): UseQueryResult<BrokerPosition[]> {
  return useQuery({
    queryKey: queryKeys.brokerPositions,
    queryFn: () => api.getBrokerPositions(),
    staleTime: 15_000,
    enabled,
  });
}

/** Recorded broker orders, newest first. */
export function useBrokerOrders(
  enabled = true,
): UseQueryResult<BrokerOrder[]> {
  return useQuery({
    queryKey: queryKeys.brokerOrders,
    queryFn: () => api.getBrokerOrders(),
    staleTime: 15_000,
    enabled,
  });
}
