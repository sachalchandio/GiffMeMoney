/**
 * Live market WebSocket hook.
 *
 * Connects to `${WS}` (derived from the API base, see `getWsUrl`), handles the
 * three server message kinds — `snapshot`, `tick` (PricePoint[]), `heartbeat` —
 * and writes them into {@link useMarketStore}. Reconnects automatically with
 * capped exponential backoff. Mount this once (in the app shell) so the store is
 * fed for the whole session.
 */

import { useEffect, useRef } from 'react';
import { getWsUrl } from '@/lib/api';
import { useMarketStore } from '@/store/marketStore';
import type { PricePoint } from '@/lib/types';

/** Server → client message shapes on `/ws`. */
type ServerMessage =
  | { type: 'snapshot'; data: PricePoint[] }
  | { type: 'tick'; data: PricePoint[] }
  | { type: 'heartbeat'; t: number };

const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 15000;

function isPricePointArray(value: unknown): value is PricePoint[] {
  return Array.isArray(value);
}

export interface UseMarketSocketOptions {
  /** Set false to disable the connection entirely (e.g. before login). */
  enabled?: boolean;
}

/**
 * Open and maintain the market WebSocket for the component's lifetime.
 *
 * @param options.enabled - When false, no socket is opened (default true).
 */
export function useMarketSocket(options: UseMarketSocketOptions = {}): void {
  const { enabled = true } = options;

  const setSnapshot = useMarketStore((s) => s.setSnapshot);
  const applyTicks = useMarketStore((s) => s.applyTicks);
  const setConnStatus = useMarketStore((s) => s.setConnStatus);
  const recordHeartbeat = useMarketStore((s) => s.recordHeartbeat);

  // Mutable refs survive reconnect cycles without re-running the effect.
  const socketRef = useRef<WebSocket | null>(null);
  const backoffRef = useRef<number>(INITIAL_BACKOFF_MS);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const closedByUserRef = useRef<boolean>(false);

  useEffect(() => {
    if (!enabled) {
      setConnStatus('disconnected');
      return undefined;
    }

    closedByUserRef.current = false;

    const clearReconnect = (): void => {
      if (reconnectTimerRef.current !== null) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
    };

    const scheduleReconnect = (): void => {
      if (closedByUserRef.current) return;
      setConnStatus('reconnecting');
      clearReconnect();
      const delay = backoffRef.current;
      reconnectTimerRef.current = setTimeout(() => {
        backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
        connect();
      }, delay);
    };

    const handleMessage = (event: MessageEvent<string>): void => {
      let msg: ServerMessage;
      try {
        msg = JSON.parse(event.data) as ServerMessage;
      } catch {
        return;
      }
      switch (msg.type) {
        case 'snapshot':
          if (isPricePointArray(msg.data)) setSnapshot(msg.data);
          break;
        case 'tick':
          if (isPricePointArray(msg.data)) applyTicks(msg.data);
          break;
        case 'heartbeat':
          recordHeartbeat(typeof msg.t === 'number' ? msg.t : Date.now());
          break;
        default:
          break;
      }
    };

    const connect = (): void => {
      if (closedByUserRef.current) return;
      setConnStatus(socketRef.current ? 'reconnecting' : 'connecting');

      let ws: WebSocket;
      try {
        ws = new WebSocket(getWsUrl());
      } catch {
        scheduleReconnect();
        return;
      }
      socketRef.current = ws;

      ws.onopen = (): void => {
        backoffRef.current = INITIAL_BACKOFF_MS;
        setConnStatus('connected');
      };
      ws.onmessage = handleMessage;
      ws.onerror = (): void => {
        // The browser fires `close` after `error`; reconnect is handled there.
      };
      ws.onclose = (): void => {
        socketRef.current = null;
        if (closedByUserRef.current) {
          setConnStatus('disconnected');
          return;
        }
        scheduleReconnect();
      };
    };

    connect();

    return () => {
      closedByUserRef.current = true;
      clearReconnect();
      const ws = socketRef.current;
      socketRef.current = null;
      if (ws) {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        try {
          ws.close();
        } catch {
          /* ignore */
        }
      }
      setConnStatus('disconnected');
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled]);
}
