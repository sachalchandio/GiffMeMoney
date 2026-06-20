/**
 * MiniSpark — a chart-namespace alias of the UI Sparkline (FRONTEND.md lists it
 * under charts). Same direction-aware coloring + token theming; thin defaults for
 * inline use in tables and cards.
 */

import { Sparkline, type SparklineProps } from '@/components/ui/Sparkline';

export type MiniSparkProps = SparklineProps;

export function MiniSpark(props: MiniSparkProps): JSX.Element {
  return <Sparkline fill {...props} />;
}
