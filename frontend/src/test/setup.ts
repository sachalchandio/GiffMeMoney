/**
 * Vitest + Testing Library global setup (jsdom).
 *
 * - Extends `expect` with jest-dom matchers.
 * - Auto-cleans the DOM after each test.
 * - Polyfills `matchMedia`, `ResizeObserver`, and `WebSocket` (jsdom omits these),
 *   so the ThemeProvider, recharts' ResponsiveContainer, and the market socket
 *   hook don't crash under test.
 */

import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

afterEach(() => {
  cleanup();
});

// matchMedia (ThemeProvider / system theme detection)
if (!window.matchMedia) {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string): MediaQueryList => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }),
  });
}

// ResizeObserver (recharts ResponsiveContainer)
class ResizeObserverStub {
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}
if (!('ResizeObserver' in globalThis)) {
  (globalThis as unknown as { ResizeObserver: typeof ResizeObserverStub }).ResizeObserver =
    ResizeObserverStub;
}

// A no-op WebSocket so useMarketSocket can mount without a real server.
class WebSocketStub {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  readyState = WebSocketStub.CONNECTING;
  onopen: ((ev: unknown) => void) | null = null;
  onclose: ((ev: unknown) => void) | null = null;
  onerror: ((ev: unknown) => void) | null = null;
  onmessage: ((ev: unknown) => void) | null = null;
  constructor(public url: string) {}
  send(): void {}
  close(): void {
    this.readyState = WebSocketStub.CLOSED;
    this.onclose?.({});
  }
}
if (!('WebSocket' in globalThis)) {
  (globalThis as unknown as { WebSocket: typeof WebSocketStub }).WebSocket = WebSocketStub;
}
