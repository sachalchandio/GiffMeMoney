/**
 * ScoreGauge + StanceBadge render tests (FRONTEND.md test matrix).
 *
 * ScoreGauge derives a stance from the -100..100 score, exposes accessible
 * meter semantics, clamps out-of-range input, and respects an explicit stance
 * override. It reads chart colors via the theme, so it renders inside a
 * ThemeProvider. StanceBadge renders the humanized stance label.
 */

import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ThemeProvider } from '@/theme/ThemeProvider';
import { ScoreGauge } from '@/components/ui/ScoreGauge';
import { StanceBadge } from '@/components/ui/StanceBadge';
import type { ReactElement } from 'react';

function renderThemed(ui: ReactElement): void {
  render(<ThemeProvider>{ui}</ThemeProvider>);
}

describe('ScoreGauge', () => {
  it('exposes accessible meter semantics for the score', () => {
    renderThemed(<ScoreGauge score={42} />);
    const meter = screen.getByRole('meter');
    expect(meter).toHaveAttribute('aria-valuemin', '-100');
    expect(meter).toHaveAttribute('aria-valuemax', '100');
    expect(meter).toHaveAttribute('aria-valuenow', '42');
    expect(meter).toHaveAttribute('aria-label', expect.stringContaining('Buy'));
  });

  it('renders the numeric value with a sign and the derived stance label', () => {
    renderThemed(<ScoreGauge score={70} />);
    expect(screen.getByText('+70')).toBeInTheDocument();
    expect(screen.getByText('Strong Buy')).toBeInTheDocument();
  });

  it('clamps an out-of-range score into -100..100', () => {
    renderThemed(<ScoreGauge score={500} />);
    expect(screen.getByRole('meter')).toHaveAttribute('aria-valuenow', '100');
    expect(screen.getByText('+100')).toBeInTheDocument();
  });

  it('honours an explicit stance override over the derived stance', () => {
    // Score implies HOLD, but the override forces SELL.
    renderThemed(<ScoreGauge score={0} stance="SELL" />);
    expect(screen.getByText('Sell')).toBeInTheDocument();
    expect(screen.queryByText('Hold')).not.toBeInTheDocument();
  });

  it('renders a caption and can hide the center labels', () => {
    const { rerender } = render(
      <ThemeProvider>
        <ScoreGauge score={10} caption="composite" />
      </ThemeProvider>,
    );
    expect(screen.getByText('composite')).toBeInTheDocument();

    rerender(
      <ThemeProvider>
        <ScoreGauge score={10} hideLabel />
      </ThemeProvider>,
    );
    expect(screen.queryByText('composite')).not.toBeInTheDocument();
    // The meter (svg arc) is still present even with labels hidden.
    expect(screen.getByRole('meter')).toBeInTheDocument();
  });
});

describe('StanceBadge', () => {
  it('renders the humanized stance label', () => {
    render(<StanceBadge stance="STRONG_BUY" />);
    expect(screen.getByText('Strong Buy')).toBeInTheDocument();
  });

  it('applies a danger-toned style for sell stances', () => {
    render(<StanceBadge stance="STRONG_SELL" />);
    const badge = screen.getByText('Strong Sell');
    expect(badge.className).toContain('text-danger');
  });
});
