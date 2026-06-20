/**
 * PriceChart — OHLCV price chart powered by lightweight-charts (v4). Renders a
 * candlestick or area series from `Candle[]`, themed from the live CSS-var tokens
 * (recreated on theme flip). Responsive to its container via a ResizeObserver.
 */

import { useEffect, useRef } from 'react';
import {
  ColorType,
  CrosshairMode,
  LineStyle,
  createChart,
  type AreaData,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import { useChartTokens } from '@/theme/tokens';
import type { Candle } from '@/lib/types';
import { cn } from '@/lib/utils';

export interface PriceChartProps {
  candles: Candle[];
  /** Series style. */
  type?: 'candle' | 'area';
  height?: number;
  className?: string;
}

export function PriceChart({
  candles,
  type = 'candle',
  height = 320,
  className,
}: PriceChartProps): JSX.Element {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const tokens = useChartTokens();

  // (Re)build the chart whenever the theme or series type changes.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return undefined;

    const chart = createChart(container, {
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: tokens.muted,
        fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: tokens.grid, style: LineStyle.Dotted },
        horzLines: { color: tokens.grid, style: LineStyle.Dotted },
      },
      rightPriceScale: { borderColor: tokens.grid },
      timeScale: { borderColor: tokens.grid, timeVisible: false, secondsVisible: false },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: tokens.muted, labelBackgroundColor: tokens.primary },
        horzLine: { color: tokens.muted, labelBackgroundColor: tokens.primary },
      },
      handleScale: { axisPressedMouseMove: false },
    });
    chartRef.current = chart;

    let series: ISeriesApi<'Candlestick'> | ISeriesApi<'Area'>;
    if (type === 'area') {
      const areaSeries = chart.addAreaSeries({
        lineColor: tokens.primary,
        topColor: applyAlpha(tokens.primary, 0.28),
        bottomColor: applyAlpha(tokens.primary, 0.02),
        lineWidth: 2,
        priceLineVisible: false,
      });
      areaSeries.setData(toAreaData(candles));
      series = areaSeries;
    } else {
      const candleSeries = chart.addCandlestickSeries({
        upColor: tokens.up,
        downColor: tokens.down,
        borderUpColor: tokens.up,
        borderDownColor: tokens.down,
        wickUpColor: tokens.up,
        wickDownColor: tokens.down,
        priceLineVisible: false,
      });
      candleSeries.setData(toCandleData(candles));
      series = candleSeries;
    }

    chart.timeScale().fitContent();

    // Belt-and-braces resize (autoSize handles most cases).
    const ro = new ResizeObserver(() => {
      if (container) chart.applyOptions({ width: container.clientWidth });
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      // Drop the ref so we never touch a disposed chart.
      if (chartRef.current === chart) chartRef.current = null;
      // `series` is owned by the chart and removed with it.
      void series;
      chart.remove();
    };
  }, [type, tokens]);

  // Push fresh data without rebuilding the chart on `candles` changes.
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    chart.timeScale().fitContent();
  }, [candles]);

  return (
    <div className={cn('w-full', className)}>
      {candles.length === 0 ? (
        <div
          className="flex items-center justify-center rounded-xl border border-dashed border-border text-sm text-muted"
          style={{ height }}
        >
          No price history
        </div>
      ) : (
        <div ref={containerRef} style={{ height }} />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function toCandleData(candles: Candle[]): CandlestickData<Time>[] {
  return candles.map((c) => ({
    time: c.t as UTCTimestamp,
    open: c.o,
    high: c.h,
    low: c.l,
    close: c.c,
  }));
}

function toAreaData(candles: Candle[]): AreaData<Time>[] {
  return candles.map((c) => ({ time: c.t as UTCTimestamp, value: c.c }));
}

/** Blend a hex/rgb color toward transparent via CSS color-mix (token-safe). */
function applyAlpha(color: string, alpha: number): string {
  return `color-mix(in srgb, ${color} ${Math.round(alpha * 100)}%, transparent)`;
}
