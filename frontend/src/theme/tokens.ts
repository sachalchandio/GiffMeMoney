/**
 * Chart color tokens read live from the CSS variables (CONTRACT §8), so every
 * chart matches the active light/dark theme without hardcoded hex.
 *
 * Recharts/lightweight-charts need concrete color strings, so we resolve the
 * computed values of the CSS vars at call time. Components should call
 * {@link useChartTokens} (re-reads on theme change) or {@link readChartTokens}
 * directly inside a render that already depends on the theme.
 */

import { useMemo } from 'react';
import { useTheme } from './ThemeProvider';

export interface ChartTokens {
  /** Bullish / up series color (= --success). */
  up: string;
  /** Bearish / down series color (= --danger). */
  down: string;
  /** Brand primary (emerald). */
  primary: string;
  /** Accent (indigo-violet). */
  accent: string;
  warning: string;
  /** Foreground text color. */
  text: string;
  /** Muted text / axis label color. */
  muted: string;
  /** Grid / border line color. */
  grid: string;
  /** Surface (card) background. */
  surface: string;
  surface2: string;
  bg: string;
  /** 8-color categorical palette for multi-series charts. */
  palette: [string, string, string, string, string, string, string, string];
}

/** Read a CSS custom property off `:root` (falls back to a sane default). */
function readVar(name: string, fallback: string): string {
  if (typeof window === 'undefined' || typeof document === 'undefined') return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/** Snapshot the current chart tokens from the live CSS variables. */
export function readChartTokens(): ChartTokens {
  return {
    up: readVar('--success', '#16a34a'),
    down: readVar('--danger', '#e5484d'),
    primary: readVar('--primary', '#0f9e6e'),
    accent: readVar('--accent', '#6d5efc'),
    warning: readVar('--warning', '#f59e0b'),
    text: readVar('--text', '#0b1220'),
    muted: readVar('--text-muted', '#5b6472'),
    grid: readVar('--border', '#e6e8ee'),
    surface: readVar('--surface', '#ffffff'),
    surface2: readVar('--surface-2', '#f1f3f7'),
    bg: readVar('--bg', '#f7f8fa'),
    palette: [
      readVar('--c-1', '#0f9e6e'),
      readVar('--c-2', '#6d5efc'),
      readVar('--c-3', '#f59e0b'),
      readVar('--c-4', '#2563eb'),
      readVar('--c-5', '#db2777'),
      readVar('--c-6', '#0891b2'),
      readVar('--c-7', '#65a30d'),
      readVar('--c-8', '#d97706'),
    ],
  };
}

/**
 * Hook variant — recomputes whenever the theme flips so charts re-render with
 * the right colors. Depends on {@link useTheme} for the theme key.
 */
export function useChartTokens(): ChartTokens {
  const { theme } = useTheme();
  // `theme` in the dep array ensures we re-read CSS vars after a class swap.
  return useMemo(() => readChartTokens(), [theme]);
}

/** Pick a categorical palette color by index (wraps). */
export function paletteColor(tokens: ChartTokens, index: number): string {
  const arr = tokens.palette;
  return arr[((index % arr.length) + arr.length) % arr.length] as string;
}
