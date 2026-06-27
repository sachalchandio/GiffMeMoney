/**
 * Real-Time mode controller hook (paper simulation).
 *
 * HONESTY / SAFETY: this drives an accelerated SIMULATION — no real money, no
 * broker, $0 real. {@link useLiveSim} owns the session lifecycle and an interval
 * that ticks the backend for a real-time feel; the caller renders the snapshot.
 *
 * Returns the current state, running flag, and start / pause / resume / reset
 * controls. Pausing stops the interval; the session stays alive on the server.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/lib/api';
import type { LiveSimStartRequest, LiveSimState } from '@/lib/types';

export interface UseLiveSim {
  state: LiveSimState | null;
  running: boolean;
  error: string | null;
  starting: boolean;
  /** Start a fresh session (resets any current one) and begin ticking. */
  start: (config: LiveSimStartRequest) => Promise<void>;
  /** Pause ticking (session stays alive on the server). */
  pause: () => void;
  /** Resume ticking the current session. */
  resume: () => void;
  /** Stop ticking and clear the local session. */
  reset: () => void;
}

/** How often (ms) to advance the sim for the real-time feel. */
const TICK_MS = 1500;

export function useLiveSim(): UseLiveSim {
  const [state, setState] = useState<LiveSimState | null>(null);
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sessionRef = useRef<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const inflightRef = useRef(false);

  const clearTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const doTick = useCallback(async () => {
    const sid = sessionRef.current;
    if (!sid || inflightRef.current) return;
    inflightRef.current = true;
    try {
      const next = await api.tickLiveSim({ sessionId: sid, steps: null });
      setState(next);
      if (next.finished) {
        clearTimer();
        setRunning(false);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Tick failed');
      clearTimer();
      setRunning(false);
    } finally {
      inflightRef.current = false;
    }
  }, [clearTimer]);

  const startTimer = useCallback(() => {
    clearTimer();
    timerRef.current = setInterval(() => {
      void doTick();
    }, TICK_MS);
    setRunning(true);
  }, [clearTimer, doTick]);

  const start = useCallback(
    async (config: LiveSimStartRequest) => {
      setStarting(true);
      setError(null);
      clearTimer();
      try {
        const initial = await api.startLiveSim(config);
        sessionRef.current = initial.sessionId;
        setState(initial);
        startTimer();
      } catch (e) {
        setError(e instanceof Error ? e.message : 'Could not start the simulation');
      } finally {
        setStarting(false);
      }
    },
    [clearTimer, startTimer],
  );

  const pause = useCallback(() => {
    clearTimer();
    setRunning(false);
  }, [clearTimer]);

  const resume = useCallback(() => {
    if (sessionRef.current && !state?.finished) startTimer();
  }, [startTimer, state?.finished]);

  const reset = useCallback(() => {
    clearTimer();
    setRunning(false);
    const sid = sessionRef.current;
    sessionRef.current = null;
    setState(null);
    if (sid) void api.stopLiveSim(sid).catch(() => undefined);
  }, [clearTimer]);

  // Clean up the interval on unmount.
  useEffect(() => () => clearTimer(), [clearTimer]);

  return { state, running, error, starting, start, pause, resume, reset };
}
