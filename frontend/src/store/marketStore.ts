/**
 * Live market store (zustand): the latest streamed price per symbol plus the
 * WebSocket connection status. `useMarketSocket` writes ticks here; pages derive
 * live P&L / tickers by reading from it so updates land every tick without extra
 * fetches.
 */

import { create } from 'zustand';
import type { PricePoint } from '@/lib/types';

/** Connection lifecycle of the market WebSocket. */
export type ConnStatus = 'connecting' | 'connected' | 'reconnecting' | 'disconnected';

/** A single symbol's live price snapshot (upper-cased key in the map). */
export interface LivePrice {
  symbol: string;
  price: number;
  changePct: number;
  /** Unix ms of the sample. */
  t: number;
}

interface MarketState {
  /** symbol(UPPER) -> latest live price. */
  prices: Record<string, LivePrice>;
  /** Current WebSocket connection status. */
  connStatus: ConnStatus;
  /** Unix ms of the most recent heartbeat (0 if none yet). */
  lastHeartbeat: number;

  /** Replace the whole price map (initial `snapshot`). */
  setSnapshot: (points: PricePoint[]) => void;
  /** Merge a batch of ticks into the price map. */
  applyTicks: (points: PricePoint[]) => void;
  /** Update the connection status. */
  setConnStatus: (status: ConnStatus) => void;
  /** Record a heartbeat timestamp (unix ms). */
  recordHeartbeat: (t: number) => void;
  /** Clear all live prices (e.g. on logout / hard reset). */
  reset: () => void;
}

function toLive(p: PricePoint): LivePrice {
  return { symbol: p.symbol.toUpperCase(), price: p.price, changePct: p.changePct, t: p.t };
}

function indexBySymbol(points: PricePoint[]): Record<string, LivePrice> {
  const out: Record<string, LivePrice> = {};
  for (const p of points) {
    const live = toLive(p);
    out[live.symbol] = live;
  }
  return out;
}

export const useMarketStore = create<MarketState>((set) => ({
  prices: {},
  connStatus: 'disconnected',
  lastHeartbeat: 0,

  setSnapshot: (points) => set({ prices: indexBySymbol(points) }),

  applyTicks: (points) =>
    set((state) => {
      if (points.length === 0) return state;
      const next = { ...state.prices };
      for (const p of points) {
        const live = toLive(p);
        next[live.symbol] = live;
      }
      return { prices: next };
    }),

  setConnStatus: (connStatus) => set({ connStatus }),

  recordHeartbeat: (t) => set({ lastHeartbeat: t }),

  reset: () => set({ prices: {}, lastHeartbeat: 0 }),
}));

/* ------------------------------------------------------------------ */
/* Selectors (stable references; safe to use as hook selectors)        */
/* ------------------------------------------------------------------ */

/** Subscribe to a single symbol's live price (or `undefined` if unseen). */
export function useLivePrice(symbol: string | undefined): LivePrice | undefined {
  return useMarketStore((s) => (symbol ? s.prices[symbol.toUpperCase()] : undefined));
}

/** Subscribe to the connection status. */
export function useConnStatus(): ConnStatus {
  return useMarketStore((s) => s.connStatus);
}

/**
 * Read a symbol's latest live price imperatively (no subscription). Useful in
 * event handlers / derivations where a re-render isn't desired.
 */
export function getLivePrice(symbol: string): LivePrice | undefined {
  return useMarketStore.getState().prices[symbol.toUpperCase()];
}
