/**
 * Light/dark theme context: persisted (`localStorage('giff_theme')`),
 * system-aware (follows `prefers-color-scheme` until the user picks), and
 * applied by toggling the `dark` class on `<html>` (Tailwind `darkMode: 'class'`).
 *
 * The pre-paint script in `index.html` applies the initial class to avoid a
 * flash; this provider keeps React state + the DOM in sync afterwards.
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

export type ThemeMode = 'light' | 'dark';

const STORAGE_KEY = 'giff_theme';

interface ThemeContextValue {
  /** The currently applied theme. */
  theme: ThemeMode;
  /** Whether the user has made an explicit choice (vs following the system). */
  isExplicit: boolean;
  /** Set the theme explicitly. */
  setTheme: (mode: ThemeMode) => void;
  /** Toggle between light and dark (becomes explicit). */
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextValue | null>(null);

function systemPrefersDark(): boolean {
  if (typeof window === 'undefined' || !window.matchMedia) return true;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

function readStored(): ThemeMode | null {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return v === 'light' || v === 'dark' ? v : null;
  } catch {
    return null;
  }
}

function applyClass(theme: ThemeMode): void {
  if (typeof document === 'undefined') return;
  document.documentElement.classList.toggle('dark', theme === 'dark');
}

export function ThemeProvider({ children }: { children: ReactNode }): JSX.Element {
  const [explicit, setExplicit] = useState<ThemeMode | null>(() => readStored());
  const [systemDark, setSystemDark] = useState<boolean>(() => systemPrefersDark());

  const theme: ThemeMode = explicit ?? (systemDark ? 'dark' : 'light');

  // Keep the DOM class in sync with the resolved theme.
  useEffect(() => {
    applyClass(theme);
  }, [theme]);

  // Follow the system preference while no explicit choice has been made.
  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return undefined;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = (e: MediaQueryListEvent): void => setSystemDark(e.matches);
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, []);

  const setTheme = useCallback((mode: ThemeMode) => {
    setExplicit(mode);
    try {
      localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      /* ignore persistence failures (private mode, etc.) */
    }
    applyClass(mode);
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  }, [theme, setTheme]);

  const value = useMemo<ThemeContextValue>(
    () => ({ theme, isExplicit: explicit !== null, setTheme, toggleTheme }),
    [theme, explicit, setTheme, toggleTheme],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

/** Access the theme context. Throws if used outside a {@link ThemeProvider}. */
export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within a ThemeProvider');
  return ctx;
}
