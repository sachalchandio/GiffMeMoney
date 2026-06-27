/**
 * UI density/literacy mode — "Easy" vs "Expert".
 *
 * This is the app's signature: the same data shown through two lenses. **Easy**
 * is for someone new to investing — plainer words, one clear answer, bigger type
 * and more breathing room. **Expert** is dense and tabular, every ratio and
 * control on screen.
 *
 * Because the whole UI is sized in `rem`, the mode also nudges the root font size
 * (set via `data-ui-mode` on `<html>` + a rule in `index.css`), so Easy mode
 * literally renders larger and calmer without touching a single component.
 *
 * Persisted to `localStorage('giff_ui_mode')`; mirrors the ThemeProvider pattern.
 * Defaults to **easy** (beginner-first).
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';

export type UiMode = 'easy' | 'expert';

const STORAGE_KEY = 'giff_ui_mode';

interface UiModeContextValue {
  /** The active mode. */
  mode: UiMode;
  /** True when the beginner-friendly lens is active. */
  isEasy: boolean;
  /** True when the dense, full-detail lens is active. */
  isExpert: boolean;
  /** Set the mode explicitly. */
  setMode: (mode: UiMode) => void;
  /** Flip between easy and expert. */
  toggleMode: () => void;
}

const UiModeContext = createContext<UiModeContextValue | null>(null);

function readStored(): UiMode {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === 'expert' || v === 'easy' ? v : 'easy';
  } catch {
    return 'easy';
  }
}

function applyAttr(mode: UiMode): void {
  if (typeof document === 'undefined') return;
  document.documentElement.setAttribute('data-ui-mode', mode);
}

export function UiModeProvider({ children }: { children: ReactNode }): JSX.Element {
  const [mode, setModeState] = useState<UiMode>(() => readStored());

  // Keep the <html> attribute in sync so CSS (root font scale) reacts.
  useEffect(() => {
    applyAttr(mode);
  }, [mode]);

  const setMode = useCallback((next: UiMode) => {
    setModeState(next);
    try {
      localStorage.setItem(STORAGE_KEY, next);
    } catch {
      /* ignore persistence failures (private mode, etc.) */
    }
    applyAttr(next);
  }, []);

  const toggleMode = useCallback(() => {
    setModeState((m) => {
      const next: UiMode = m === 'easy' ? 'expert' : 'easy';
      try {
        localStorage.setItem(STORAGE_KEY, next);
      } catch {
        /* ignore */
      }
      applyAttr(next);
      return next;
    });
  }, []);

  const value = useMemo<UiModeContextValue>(
    () => ({ mode, isEasy: mode === 'easy', isExpert: mode === 'expert', setMode, toggleMode }),
    [mode, setMode, toggleMode],
  );

  return <UiModeContext.Provider value={value}>{children}</UiModeContext.Provider>;
}

/** Access the UI mode. Throws if used outside a {@link UiModeProvider}. */
export function useUiMode(): UiModeContextValue {
  const ctx = useContext(UiModeContext);
  if (!ctx) throw new Error('useUiMode must be used within a UiModeProvider');
  return ctx;
}
