import type { Config } from 'tailwindcss';

/**
 * Tailwind maps semantic color names to the CSS variables defined in
 * `src/index.css` (both `:root` light + `.dark` token blocks per CONTRACT §8).
 * Components only ever reference these semantic classes (e.g. `bg-surface`,
 * `text-muted`, `border-border`) — never hardcoded hex. Charts read the same
 * tokens via `theme/tokens.ts`.
 */
const config: Config = {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: 'var(--bg)',
        surface: 'var(--surface)',
        'surface-2': 'var(--surface-2)',
        border: 'var(--border)',
        text: 'var(--text)',
        muted: 'var(--text-muted)',
        primary: {
          DEFAULT: 'var(--primary)',
          press: 'var(--primary-press)',
        },
        accent: 'var(--accent)',
        success: 'var(--success)',
        danger: 'var(--danger)',
        warning: 'var(--warning)',
      },
      borderColor: {
        DEFAULT: 'var(--border)',
      },
      ringColor: {
        DEFAULT: 'var(--ring)',
      },
      boxShadow: {
        soft: '0 1px 2px 0 var(--shadow), 0 8px 24px -12px var(--shadow)',
        card: '0 1px 3px 0 var(--shadow), 0 12px 32px -16px var(--shadow)',
        pop: '0 16px 48px -16px var(--shadow)',
      },
      borderRadius: {
        xl: '0.75rem',
        '2xl': '1rem',
      },
      fontFamily: {
        sans: [
          'Inter',
          'ui-sans-serif',
          'system-ui',
          '-apple-system',
          'Segoe UI',
          'Roboto',
          'Helvetica Neue',
          'Arial',
          'sans-serif',
        ],
      },
      maxWidth: {
        content: '1440px',
      },
      keyframes: {
        'fade-in': {
          from: { opacity: '0' },
          to: { opacity: '1' },
        },
        'slide-up': {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in-left': {
          from: { transform: 'translateX(-100%)' },
          to: { transform: 'translateX(0)' },
        },
        shimmer: {
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        'fade-in': 'fade-in 0.2s ease-out',
        'slide-up': 'slide-up 0.24s ease-out',
        'slide-in-left': 'slide-in-left 0.2s ease-out',
        shimmer: 'shimmer 1.6s infinite',
      },
    },
  },
  plugins: [],
};

export default config;
